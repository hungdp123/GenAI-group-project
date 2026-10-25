"""Shared RAG capture engine with a one/two-claim pilot CLI.

Run from the repository root. The matrix runner reuses this engine for each
configuration. This module's own CLI stays limited to pilot cases. No RAGAS calls.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

from scripts.narrative.evaluation.evaluate_rag_pilot import (
    case_identity, load_cases, load_reference, read_json, write_json,
)
from scripts.narrative.evaluation.resume import archive_state, capture_compatible


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    # Imports do not initialize embedding models or make model calls.
    from finsight.agents.narrative.rag import basic_rag, evidence_summary, verdict_baseline
    from finsight.config import get_settings

    workspace = Path.cwd()
    output, corpus = args.output_dir.resolve(), args.corpus_root.resolve()
    if not 1 <= args.top_k <= 25 or args.candidate_k < args.top_k or args.max_tokens < 1:
        raise ValueError("Require 1 <= top-k <= 25, candidate-k >= top-k and max-tokens >= 1")
    if args.ollama_context < 1:
        raise ValueError("Ollama context must be positive")
    resume = getattr(args, "resume", False)
    prepare_only = getattr(args, "prepare_only", False)
    retrieval_only = getattr(args, "retrieval_only", False)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Output directory is not empty; use a new run name")
    cases = load_cases(args.cases)
    required = {"id", "claims", "type", "company", "source_doc", "gold_answer", "gold_snippet", "cut_off"}
    if not cases or any(required - row.keys() for row in cases):
        raise ValueError("Missing required CSV case fields")
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("CSV case IDs must be unique")
    if args.limit < 1:
        raise ValueError("Case limit must be positive")
    selected = cases[:args.limit]
    if len(selected) != args.limit:
        raise ValueError("CSV has fewer cases than the requested pilot limit")
    preflights, reference_cache = {}, {}
    for case in selected:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", case["id"])
                or not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", case["company"])):
            raise ValueError("Invalid case ID or company ticker")
        if case["type"] not in {"supported", "contradicting", "unsupported"}:
            raise ValueError("Unknown case type")
        date.fromisoformat(case["cut_off"])
        folder = corpus / case["company"]
        parsed = workspace / "data/narrative/processed" / case["company"] / "documents.jsonl"
        build = read_json(folder / "build_report.json")
        if (build["status"] != "COMPLETE" or build["inputs"]["parsed_sha256"] != sha(parsed)
                or build["index_sha256"] != sha(folder / "indexes/index.json")):
            raise ValueError(f"Incomplete or stale build for {case['company']}")
        reference, _, chunks = load_reference(case, workspace, corpus, reference_cache)
        if args.backend != "bm25":
            vectors = read_json(folder / "indexes/qdrant/qdrant_metadata.json")
            if (vectors["corpus_fingerprint"] != build["corpus_fingerprint"]
                    or vectors["chunk_ids"] != [c["chunk_id"] for c in chunks]
                    or vectors["vector_count"] != len(chunks)
                    or not (folder / "indexes/qdrant/storage").is_dir()):
                raise ValueError(f"Missing or inconsistent vector build for {case['company']}")
        preflights[case["id"]] = {
            "status": "PASS", "parsed_sha256": sha(parsed),
            "index_sha256": build["index_sha256"], "corpus_fingerprint": build["corpus_fingerprint"],
            "source_map_sha256": sha(folder / "source_map.jsonl"),
            "gold_source_block_1based": reference.get("line"),
            "semantic_labels": "NOT_INDEPENDENTLY_ADJUDICATED",
        }
    settings = get_settings()
    if not prepare_only and not retrieval_only and not args.ollama_model and settings.llm_provider == "mock":
        raise ValueError("Select --ollama-model or configure a real shared LLM provider")
    output.mkdir(parents=True, exist_ok=True)
    try:
        corpus_setting = str(corpus.relative_to(workspace))
    except ValueError:
        corpus_setting = str(corpus)
    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"{len(selected)}-case single-configuration RAG capture; no judge calls",
        "dataset": str(args.cases), "dataset_sha256": sha(args.cases),
        "case_ids": [c["id"] for c in selected], "corpus_root": corpus_setting,
        "selected_cases_sha256": case_identity(selected),
        "backend": args.backend, "source_priority": "P0", "top_k_per_direction": args.top_k,
        "retrieval_only": retrieval_only,
        "candidate_k": args.candidate_k, "max_output_tokens": args.max_tokens,
        "provider": "none" if retrieval_only else ("ollama" if args.ollama_model else settings.llm_provider),
        "model": None if retrieval_only else args.ollama_model or settings.llm_model,
        "context_window_tokens": args.ollama_context if args.ollama_model else None,
        "ollama_host": args.ollama_host if args.ollama_model else None,
        "reasoning_effort": os.getenv("FINSIGHT_REASONING_EFFORT"),
        "reranker_model": os.getenv("FINSIGHT_NARRATIVE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "rerank_overflow": args.rerank_overflow, "gold_fields_supplied_to_llm": False,
        "code_hashes": {str(p): sha(p) for p in [
            Path(__file__), Path("finsight/agents/narrative/rag/basic_rag.py"),
            Path("finsight/agents/narrative/rag/collect_evidence.py"),
            Path("finsight/agents/narrative/rag/evidence_summary.py"),
            Path("finsight/agents/narrative/retrieval/retriever.py"),
            Path("finsight/agents/narrative/rag/verdict_baseline.py"),
            Path("finsight/llm/client.py"),
        ]},
    }
    if (output / "configuration.json").exists():
        previous = read_json(output / "configuration.json")

        if not capture_compatible(previous, configuration, load_cases(output / "cases.csv"), selected):
            raise ValueError("Resume configuration/dataset/code differs; use a new run version")
        configuration["created_utc"] = previous["created_utc"]
    else:
        previous = None
    # Check existing build/source identities before updating the selected-case snapshot.
    for case in selected:
        preflight = output / "cases" / case["id"] / "preflight.json"
        if preflight.exists() and read_json(preflight) != preflights[case["id"]]:
            raise ValueError(f"Saved build/source identity changed for {case['id']}")
    if previous != configuration:
        if previous is not None:
            archive_state(output, previous)
        write_json(output / "configuration.json", configuration)
        with (output / "cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    for case in selected:
        preflight = output / "cases" / case["id"] / "preflight.json"
        write_json(preflight, preflights[case["id"]])
    if prepare_only:
        return {"status": "PREFLIGHT_PASSED", "case_count": len(selected), "records": []}
    original_collector = basic_rag.NarrativeEvidenceAgent
    original_complete = evidence_summary.complete
    original_ollama = verdict_baseline._complete_ollama
    captured = {}
    current = output
    previous_records = ({r["id"]: r for r in read_json(output / "report.json").get("records", [])}
                        if (output / "report.json").exists() else {})
    models = getattr(args, "model_cache", {})

    class CapturedCollector(original_collector):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            captured["collector"] = self
            if args.backend != "bm25":
                from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
                from finsight.agents.narrative.retrieval.retrieval_experiments import (
                    SentenceTransformersBackend, SentenceTransformersReranker,
                )

                def factory(ticker):
                    path = corpus / ticker / "indexes/index.json"
                    metadata = read_json(path.parent / "qdrant/qdrant_metadata.json")
                    name = metadata["embedding_model"]
                    if ("embedding", name) not in models:
                        models["embedding", name] = SentenceTransformersBackend(name)
                    reranker = None
                    if args.backend == "r4":
                        name = configuration["reranker_model"]
                        if ("reranker", name) not in models:
                            models["reranker", name] = SentenceTransformersReranker(name)
                        reranker = models["reranker", name]
                    return CorpusRetriever(
                        company_ticker=ticker, index_path=path, qdrant_dir=path.parent / "qdrant",
                        backend=args.backend, candidate_k=args.candidate_k,
                        embedding_backend=models["embedding", metadata["embedding_model"]],
                        reranker_backend=reranker, rerank_overflow=args.rerank_overflow,
                    )

                self.retriever_factory = factory

    def save_response(response):
        write_json(current / "llm_response.json", asdict(response))
        return response

    def capture_complete(*positional, **keywords):
        write_json(current / "llm_input.json", {
            "provider": settings.llm_provider, "model": keywords.get("model") or settings.llm_model,
            "system": keywords.get("system"),
            "prompt": keywords.get("prompt", positional[0] if positional else None),
            "max_tokens": keywords.get("max_tokens"),
        })
        return save_response(original_complete(*positional, **keywords))

    def capture_ollama(**keywords):
        write_json(current / "llm_input.json", {"provider": "ollama", **keywords, "think": False})
        return save_response(original_ollama(**keywords))

    records = []
    overrides = {"FINSIGHT_NARRATIVE_SEARCH_LOG": str(output / "searches.jsonl"),
                 "FINSIGHT_USAGE_LEDGER": str(output / "llm_usage.jsonl")}
    with (patch.dict(os.environ, overrides),
          patch.object(basic_rag, "NarrativeEvidenceAgent", CapturedCollector),
          patch.object(evidence_summary, "complete", capture_complete),
          patch.object(verdict_baseline, "_complete_ollama", capture_ollama)):
        # Scope overrides to this run. Templates otherwise take precedence over corpus_root.
        for key in ("FINSIGHT_NARRATIVE_INDEX_TEMPLATE", "FINSIGHT_NARRATIVE_QDRANT_TEMPLATE"):
            os.environ.pop(key, None)
        for case in selected:
            current = output / "cases" / case["id"]
            captured.clear()
            started = time.perf_counter()
            reused = (current / "result.json").exists()
            print(f"{'Reusing saved' if reused else 'Running'} {case['id']} ({args.backend})", flush=True)
            if reused:
                if not resume or not (current / "retrieval.json").exists():
                    raise ValueError("Existing or incomplete case; do not overwrite saved model work")
                result = read_json(current / "result.json")
            else:
                if (current / "llm_input.json").exists() or (current / "llm_response.json").exists():
                    raise ValueError("Interrupted model attempt needs review; automatic duplicate call refused")
                result = basic_rag.run_basic_narrative_rag(
                    case["company"], case["claims"], date.fromisoformat(case["cut_off"]),
                    backend=args.backend, corpus_root=corpus, top_k=args.top_k,
                    candidate_k=args.candidate_k, max_tokens=args.max_tokens,
                    ollama_model=args.ollama_model, ollama_host=args.ollama_host,
                    ollama_context=args.ollama_context, rerank_overflow=args.rerank_overflow,
                    retrieval_only=retrieval_only,
                )
                trace = captured["collector"].last_search
                write_json(current / "retrieval.json", trace)
                write_json(current / "result.json", result)
            records.append({"id": case["id"], "status": result["status"],
                            "reused": reused,
                            "elapsed_ms": previous_records.get(case["id"], {}).get("elapsed_ms") if reused
                            else round((time.perf_counter() - started) * 1000, 3),
                            "llm": result["llm"], "result": str(current / "result.json")})
            report = {"status": "IN_PROGRESS", "case_count": len(records), "records": records,
                      "input_tokens": sum(r["llm"].get("input_tokens") or 0 for r in records),
                      "output_tokens": sum(r["llm"].get("output_tokens") or 0 for r in records),
                      "recorded_cost_usd": sum(r["llm"].get("cost_usd") or 0 for r in records),
                      "unknown_usage_cases": sum(r["llm"].get("input_tokens") is None for r in records),
                      "retrieval_only": retrieval_only,
                      "ragas": "NOT_RUN"}
            write_json(output / "report.json", report)
            print(f"{case['id']}: {result['status']}", flush=True)
            fatal = ("insufficient_quota", "credit_balance_exhausted", "MissingAPIKey",
                     "AuthenticationError", "Could not reach Ollama", "APIConnectionError")
            if not reused and result["status"] == "GENERATION_ERROR" and any(
                    marker in result.get("generation_error", "") for marker in fatal):
                report["status"] = "STOPPED_PROVIDER_ERROR"
                write_json(output / "report.json", report)
                raise ValueError(
                    "Model provider is unavailable or rejected access/quota; saved the attempt and stopped"
                )
    report["status"] = "COMPLETE" if all(r["status"] == "OK" for r in records) else "COMPLETE_WITH_ERRORS"
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path,
                        default=Path("data/narrative/evaluation/test_claims_v1/cases.csv"))
    parser.add_argument("--limit", type=int, choices=(1, 2), default=2)
    parser.add_argument("--corpus-root", type=Path,
                        default=Path("data/narrative/processed/chunking_strategies/C4/full_v1"))
    parser.add_argument("--backend", choices=("bm25", "dense", "hybrid", "r4"), default="bm25")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--ollama-model")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-context", type=int, default=20000)
    parser.add_argument("--rerank-overflow", choices=("error", "truncate"), default="error")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip LLM preparation and save raw retrieval evidence/rankings only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Pilot failed: {type(exc).__name__}: {exc}\n")
    print(f"Saved {report['case_count']} cases: {args.output_dir / 'report.json'}")
    print("Next: python -m scripts.narrative.evaluation.evaluate_rag_pilot --run-dir "
          f'"{args.output_dir}"')


if __name__ == "__main__":
    main()
