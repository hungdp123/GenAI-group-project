"""Generate Basic RAG answers for the frozen 92-claim development experiment.

Uses the existing single-pass pipeline. No generation occurs on import or --help.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

from finsight.agents.narrative.rag.basic_rag import run_basic_narrative_rag
from finsight.config import PRICE_TABLE, get_settings


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(records: list[dict], attempts: list[dict], expected: int) -> dict:
    statuses = Counter(r["result"]["status"] for r in records)
    usage = [r["result"]["llm"] for r in attempts if r["result"].get("llm")]
    unknown_usage = len(attempts) - len(usage)
    unknown_cost = unknown_usage + sum(u.get("cost_usd") is None for u in usage)
    complete = len(records) == expected
    return {
        "status": ("complete" if statuses.get("OK", 0) == expected else "complete_with_errors")
        if complete else "partial",
        "expected_claims": expected, "completed_claims": len(records),
        "technically_valid_claims": statuses.get("OK", 0), "statuses": dict(statuses),
        "pipeline_attempts": len(attempts),
        "responses_with_usage": len(usage),
        "input_tokens": sum(u.get("input_tokens") or 0 for u in usage),
        "output_tokens": sum(u.get("output_tokens") or 0 for u in usage),
        "estimated_cost_usd": sum(u.get("cost_usd") or 0 for u in usage),
        "attempts_without_returned_usage": unknown_usage,
        "attempts_without_cost_estimate": unknown_cost,
        "cost_estimate_complete": unknown_cost == 0,
        "generation_ms": sum(u.get("generation_ms") or 0 for u in usage),
        "attempt_elapsed_ms": sum(r["elapsed_ms"] for r in attempts),
        "verdicts": dict(Counter((r["result"].get("answer") or {}).get("verdict", "NO_VALID_ANSWER")
                                 for r in records)),
        "verdict_accuracy": None,
        "verdict_accuracy_status": "NOT_SCORED: development verdict labels require independent review",
        "limitations": [
            "Technical schema/citation checks and limited numeric consistency checks are reused from Basic RAG.",
            "Exact quotes and valid citation IDs do not establish semantic citation support or faithfulness.",
            "Draft proposed verdicts are not treated as ground truth; accuracy/Macro-F1 are not reported.",
            "Agent uses two generated search strings; this is separate from Step 4 claim-query rankings.",
            "Reported cost is a local estimate, not account spending; unknown usage/cost is not zero cost.",
            "Force termination may prevent saving an in-flight response and its usage.",
        ],
    }


def run_batch(args) -> dict:
    experiment, queries_path, output = args.experiment_dir.resolve(), args.queries.resolve(), args.output_dir.resolve()
    experiment_config = read(experiment / "configuration.json")
    query_hash = digest(queries_path)
    if query_hash != experiment_config["questions_sha256"]:
        raise ValueError("Questions do not match the experiment's recorded dataset hash")
    questions = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(questions) != 92 or any(q.get("split") != "dev" for q in questions):
        raise ValueError("Expected exactly 92 development claims; heldout is excluded")
    if len({q["query_id"] for q in questions}) != 92:
        raise ValueError("Duplicate question IDs")
    if {q["ticker"] for q in questions} != set(experiment_config["tickers"]):
        raise ValueError("Experiment and question company scopes differ")
    if any(not q.get("evaluation_eligible", False) for q in questions) and not args.allow_draft:
        raise ValueError("Unreviewed development claims require --allow-draft")
    if not 1 <= args.top_k <= 25 or args.candidate_k < args.top_k or args.max_tokens < 1 or args.ollama_context < 1:
        raise ValueError("Invalid top-k, candidate-k, max-tokens or ollama-context")
    # Output must not be able to overwrite experiment inputs.
    companies = experiment / "companies"
    if output == experiment or output in experiment.parents or companies in [output, *output.parents]:
        raise ValueError("Use a separate batch output directory, outside company index folders")
    settings = get_settings()
    if not args.ollama_model and settings.llm_provider == "mock":
        raise ValueError("Configure a real shared LLM provider, or supply --ollama-model")
    configuration = {
        "experiment_dir": str(experiment), "queries_sha256": query_hash,
        "backend": args.backend, "top_k": args.top_k, "candidate_k": args.candidate_k,
        "max_tokens": args.max_tokens, "ollama_context": args.ollama_context if args.ollama_model else None,
        "ollama_host": args.ollama_host if args.ollama_model else None,
        "provider": "ollama" if args.ollama_model else settings.llm_provider,
        "model": args.ollama_model or settings.llm_model,
        "index_sha256": {t: digest(companies / t / "index.json") for t in experiment_config["tickers"]},
        "code_sha256": {name: digest(Path(__file__).parent.parent / name) for name in (
            "evaluation/run_basic_rag_batch.py", "rag/basic_rag.py", "rag/collect_evidence.py",
            "retrieval/retriever.py")},
    }
    configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise ValueError("Output exists; use --resume or a new --output-dir")
        if read(output / "configuration.json") != configuration:
            raise ValueError("Configuration, source indexes or code changed; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases").mkdir(exist_ok=True)
    (output / "attempts").mkdir(exist_ok=True)
    write(output / "configuration.json", configuration)
    records, attempts = [], []
    for path in sorted((output / "attempts").glob("*.json")):
        saved = read(path)
        if saved["configuration_hash"] != configuration_hash:
            raise ValueError("Attempt configuration mismatch")
        attempts.append(saved)

    def save_summary():
        report = summarize(records, attempts, len(questions))
        report.update(updated_at=datetime.now(timezone.utc).isoformat(), configuration=configuration,
                      results=[{"query_id": r["query_id"], "ticker": r["ticker"],
                                "status": r["result"]["status"], "output": r["output"]} for r in records])
        write(output / "report.json", report)
        return report

    for ordinal, q in enumerate(questions, 1):
        case_id = hashlib.sha256(q["query_id"].encode()).hexdigest()
        case_path = output / "cases" / f"{q['ticker']}_{case_id}.json"
        # A saved failed attempt is reused unless retries are explicitly requested.
        saved = read(case_path) if case_path.exists() else None
        if saved is not None and (
            saved["configuration_hash"] != configuration_hash or saved["query_id"] != q["query_id"]
        ):
            raise ValueError(f"Saved case identity mismatch: {q['query_id']}")
        if saved is not None and (saved["result"]["status"] == "OK" or not args.retry_failed):
            records.append(saved)
            print(f"[{ordinal}/92] Reusing {q['ticker']} {q['query_id']}", flush=True)
            continue
        print(f"[{ordinal}/92] Generating {q['ticker']} {q['query_id']}", flush=True)
        started = time.perf_counter()
        try:
            result = run_basic_narrative_rag(
                q["ticker"], q["claim"], date.fromisoformat(q["cutoff_date"]),
                backend=args.backend, top_k=args.top_k, candidate_k=args.candidate_k,
                max_tokens=args.max_tokens, ollama_model=args.ollama_model,
                ollama_host=args.ollama_host, ollama_context=args.ollama_context,
                experiment_dir=experiment)
        except Exception as exc:
            result = {"status": "PIPELINE_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        llm = result.get("llm")
        if llm and llm.get("provider") != "ollama" and llm.get("model") not in PRICE_TABLE:
            # The shared client returns zero for unknown prices; expose that as unknown.
            llm["cost_usd"] = None
        record = {"query_id": q["query_id"], "ticker": q["ticker"], "claim": q["claim"],
                  "configuration_hash": configuration_hash, "result": result,
                  "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                  "output": str(case_path.relative_to(output))}
        write(output / "attempts" / f"{case_id}_{uuid.uuid4().hex}.json", record)
        write(case_path, record)
        attempts.append(record)
        records.append(record)
        save_summary()
    return save_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-context", type=int, default=20000)
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    try:
        report = run_batch(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Batch failed: {exc}\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"configuration", "results"}}, indent=2))
    if report["technically_valid_claims"] != report["expected_claims"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
