"""Score a saved RAG pilot without retrieval, generation or judge-model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from scripts.narrative.datasets.narrative_reference_mapping import covers


METRICS = ("source_hit_at_5", "source_hit_at_10", "known_reference_recall_at_5",
           "known_reference_recall_at_10", "mrr_at_5", "mrr_at_10",
           "bundle_completion_rr_at_10")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def case_identity(cases: list[dict]) -> str:
    return hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_reference(case: dict, workspace: Path, corpus_root: Path,
                   cache: dict | None = None) -> tuple[dict, dict, list[dict]]:
    ticker = case["company"]
    parsed = workspace / "data/narrative/processed" / ticker / "documents.jsonl"
    folder = corpus_root / ticker
    key = (str(corpus_root), ticker)
    if cache is None or key not in cache:
        blocks = [json.loads(line) for line in parsed.read_text(encoding="utf-8").splitlines() if line.strip()]
        maps = {row["chunk_id"]: row for row in (
            json.loads(line) for line in (folder / "source_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )}
        index = read_json(folder / "indexes/index.json")
        if cache is not None:
            cache.clear()  # Keep only one company's source/index data in memory.
            cache[key] = blocks, maps, index
    else:
        blocks, maps, index = cache[key]
    if case["type"] == "unsupported":
        if case["gold_snippet"].strip():
            raise ValueError("Unsupported cases must not supply positive gold evidence")
        return {}, maps, index["chunks"]
    match = re.search(r"block (\d+) \| chars (\d+):(\d+)", case["source_doc"])
    if not match:
        raise ValueError(f"Missing source anchor for {case['id']}")
    line, start, end = map(int, match.groups())
    block = blocks[line - 1]
    quote = block["text"][start:end]
    if (block["document_id"] != case["source_doc"].split(" | ")[0]
            or not case["gold_snippet"].strip()
            or " ".join(quote.split()) != " ".join(case["gold_snippet"].split())):
        raise ValueError(f"Source anchor does not match gold text for {case['id']}")
    if block["publication_date"] > case["cut_off"]:
        raise ValueError(f"Gold source is after cutoff for {case['id']}")
    return {"block": block, "line": line, "start": start, "end": end}, maps, index["chunks"]


def score_direction(hits: list[dict], reference: dict, maps: dict) -> dict:
    if not reference:
        return dict.fromkeys(METRICS)
    block, line, start, end = (reference[k] for k in ("block", "line", "start", "end"))

    def spans(hit):
        row = maps[hit["chunk_id"]]
        if row["document_id"] != block["document_id"] or row["block_line_1based"] != line:
            return []
        return [row["body_char_span"]] + row.get("context_char_spans", [])

    def complete(prefix):
        return covers(block["text"], start, end, [s for h in prefix for s in spans(h)])

    def source_hit(k):
        return int(any(maps[h["chunk_id"]]["document_id"] == block["document_id"]
                       for h in hits[:k]))

    first = next((n for n, hit in enumerate(hits[:10], 1) if complete([hit])), None)
    bundle = next((n for n in range(1, min(10, len(hits)) + 1) if complete(hits[:n])), None)
    return {"source_hit_at_5": source_hit(5), "source_hit_at_10": source_hit(10),
            "known_reference_recall_at_5": int(complete(hits[:5])),
            "known_reference_recall_at_10": int(complete(hits[:10])),
            "mrr_at_5": 1 / first if first and first <= 5 else 0,
            "mrr_at_10": 1 / first if first else 0,
            "bundle_completion_rr_at_10": 1 / bundle if bundle else 0,
            "first_complete_reference_rank": first}


def evaluate(run_dir: Path, workspace: Path) -> dict:
    config = read_json(run_dir / "configuration.json")
    corpus_root = Path(config["corpus_root"])
    if not corpus_root.is_absolute():
        corpus_root = workspace / corpus_root
    cases = load_cases(run_dir / "cases.csv")
    if config.get("selected_cases_sha256") and case_identity(cases) != config["selected_cases_sha256"]:
        raise ValueError("Saved case snapshot differs from its recorded identity")
    records, reference_cache = [], {}
    for case in cases:
        folder = run_dir / "cases" / case["id"]
        result, trace = read_json(folder / "result.json"), read_json(folder / "retrieval.json")
        preflight = read_json(folder / "preflight.json")
        parsed = workspace / "data/narrative/processed" / case["company"] / "documents.jsonl"
        index_path = corpus_root / case["company"] / "indexes/index.json"
        for path, field in ((parsed, "parsed_sha256"), (index_path, "index_sha256")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != preflight[field]:
                raise ValueError(f"Saved input hash differs: {path}")
        if preflight.get("source_map_sha256"):
            mapping_path = corpus_root / case["company"] / "source_map.jsonl"
            if hashlib.sha256(mapping_path.read_bytes()).hexdigest() != preflight["source_map_sha256"]:
                raise ValueError(f"Saved source map hash differs: {mapping_path}")
        reference, maps, chunks = load_reference(case, workspace, corpus_root, reference_cache)
        indexed = {c["chunk_id"]: c for c in chunks}
        errors, directions = [], {}
        if trace.get("claim") != case["claims"] or trace.get("cutoff") != case["cut_off"]:
            raise ValueError(f"Saved trace request differs for {case['id']}")
        if trace.get("index", {}).get("corpus_fingerprint") != read_json(
                corpus_root / case["company"] / "indexes/index.json")["corpus_fingerprint"]:
            raise ValueError(f"Saved trace/index fingerprint differs for {case['id']}")
        for direction in ("support", "counter"):
            hits = trace[direction]
            if [h["rank"] for h in hits] != list(range(1, len(hits) + 1)):
                raise ValueError("Ranked results must retain their original consecutive order")
            if len({h["chunk_id"] for h in hits}) != len(hits):
                raise ValueError("Duplicate evidence IDs in a directional ranking")
            for hit in hits:
                original = indexed.get(hit["chunk_id"])
                if not original or original["text"] != hit["text"]:
                    errors.append("Source ID/text mismatch")
                elif (original["company_ticker"] != case["company"]
                      or original["publication_date"] > case["cut_off"]):
                    errors.append("Company or cutoff mismatch")
            if errors:
                raise ValueError(f"Invalid saved evidence for {case['id']}: {errors}")
            directions[direction] = score_direction(hits, reference, maps)
        try:
            raw = json.loads(result.get("raw_model_output", "{}"))
        except (TypeError, ValueError):
            raw = {}
        raw_facts = raw.get("facts", []) if isinstance(raw, dict) else []
        raw_facts = [f for f in raw_facts if isinstance(f, dict)] if isinstance(raw_facts, list) else []
        facts = [f for f in raw_facts if isinstance(f.get("quote"), str)]
        candidates = {c["evidence_id"]: c["content"] for c in result.get("candidate_evidence", [])}
        exact = sum(bool(f["quote"].strip()) and f["quote"] in candidates.get(
            f.get("evidence_id"), "") for f in facts)
        normalized = sum(" ".join(f["quote"].split()) in " ".join(
            candidates.get(f.get("evidence_id"), "").split()) for f in facts if f["quote"].strip())
        records.append({"id": case["id"], "type": case["type"], "status": result["status"],
                        "retrieval_only": result.get("pipeline") == "narrative_evidence_retrieval",
                        "company": case["company"],
                        "directional_retrieval": directions, "source_validation_passed": True,
                        "citation_validation": result["citation_validation"],
                        "raw_facts": len(raw_facts),
                        "raw_span_references": sum("source_span_id" in f for f in raw_facts),
                        "exact_raw_quotes": exact,
                        "whitespace_normalized_raw_quotes": normalized,
                        "accepted_facts": len(result.get("facts", [])),
                        "accepted_summary_sentences": len(result.get("summary", [])),
                        "llm": result["llm"]})
    aggregates = {}
    for direction in ("support", "counter"):
        aggregates[direction] = {}
        for key in METRICS:
            values = [r["directional_retrieval"][direction][key] for r in records
                      if r["directional_retrieval"][direction][key] is not None]
            aggregates[direction][key] = statistics.mean(values) if values else None
    return {"scope": "Saved single-configuration runs only; no judge-model evaluation",
            "cases": len(cases), "case_types": dict(Counter(c["type"] for c in cases)),
            "directional_retrieval": aggregates,
            "technical_passes": sum(r["status"] == "OK" and not r["retrieval_only"] for r in records),
            "retrieval_only_cases": sum(r["retrieval_only"] for r in records),
            "input_tokens": sum(r["llm"].get("input_tokens") or 0 for r in records),
            "output_tokens": sum(r["llm"].get("output_tokens") or 0 for r in records),
            "recorded_cost_usd": sum(r["llm"].get("cost_usd") or 0 for r in records),
            "unknown_usage_cases": sum(r["llm"].get("input_tokens") is None for r in records),
            "generation_ms": sum(r["llm"].get("generation_ms") or 0 for r in records),
            "new_retrieval_requests": 0, "new_llm_requests": 0, "ragas": "NOT_RUN",
            "limitations": ["Known annotated spans only; alternative evidence is unjudged.",
                            "Source Hit@K counts any passage from the annotated document, not evidence relevance.",
                            "MRR@5/10 uses the first single passage covering the gold span, not a document hit.",
                            "Unsupported positive-reference metrics are N/A.",
                            "Counter is a search direction, not a counter-evidence label.",
                            "Whitespace-normalized diagnostics do not override strict citation validation.",
                            "Technical success is not semantic accuracy. Precision/nDCG are not scored."],
            "records": records}


def render_report(report: dict) -> str:
    lines = ["# RAG configuration evaluation", "", report["scope"], "",
             "| Case | Status | Source checks | Citation checks | Accepted facts / summary sentences |",
             "|---|---|---|---|---|"]
    errors = []
    for row in report["records"]:
        citation = "Pass" if row["citation_validation"].get("passed") else "Fail / not run"
        lines.append(f"| {row['id']} | {row['status']} | Pass | {citation} | "
                     f"{row['accepted_facts']} / {row['accepted_summary_sentences']} |")
        for error in row["citation_validation"].get("errors", []):
            errors.append(f"- {row['id']}: {error}")
    if errors:
        lines.extend(["", "Citation errors:", "", *errors])
    lines += ["", "| Search direction | Source Hit@5 | Source Hit@10 | "
              "Known-reference Recall@5 | Recall@10 | MRR@5 | MRR@10 |",
              "|---|---|---|---|---|---|---|"]
    for direction, scores in report["directional_retrieval"].items():
        def fmt(key):
            value = scores[key]
            return "N/A" if value is None else f"{value:.4f}"
        lines.append(f"| {direction} | {fmt('source_hit_at_5')} | {fmt('source_hit_at_10')} | "
                     f"{fmt('known_reference_recall_at_5')} | {fmt('known_reference_recall_at_10')} | "
                     f"{fmt('mrr_at_5')} | {fmt('mrr_at_10')} |")
    generation_cases = report["cases"] - report["retrieval_only_cases"]
    accepted = f"{report['technical_passes']}/{generation_cases}" if generation_cases else "N/A"
    lines += ["", f"Technically accepted generated outputs: {accepted}. "
              f"Retrieval-only cases (generation not assessed): {report['retrieval_only_cases']}.", "",
              f"Recorded tokens: {report['input_tokens']:,} input / {report['output_tokens']:,} output. "
              f"Recorded API cost: ${report['recorded_cost_usd']:.6f}. "
              f"Cases with unknown usage: {report['unknown_usage_cases']}.", "",
              f"Generation time: {report['generation_ms']/1000:.2f} seconds; excludes retrieval. "
              "Local hardware/electricity cost is not measured.", "",
              "## Limits", ""] + [f"- {note}" for note in report["limitations"]]
    lines += ["- RAGAS and verdict accuracy are not evaluated.", "",
              "Each case folder contains result.json, retrieval.json and captured LLM inputs/response "
              "when generation was attempted. Raw output is retained when strict validation fails."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.run_dir.resolve(), Path.cwd())
    write_json(args.run_dir / "evaluation.json", report)
    (args.run_dir / "evaluation.md").write_text(render_report(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"Readable report: {args.run_dir / 'evaluation.md'}")


if __name__ == "__main__":
    main()
