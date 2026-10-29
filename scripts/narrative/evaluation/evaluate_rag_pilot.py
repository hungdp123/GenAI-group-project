"""Score a saved RAG pilot without retrieval, generation or judge-model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from scripts.narrative.datasets.narrative_reference_mapping import covers


METRICS = ("source_hit_at_5", "source_hit_at_10", "known_reference_recall_at_5",
           "known_reference_recall_at_10", "mrr_at_5", "mrr_at_10",
           "bundle_completion_rr_at_10")
STANCE_LABELS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTING")


def stance_record(case: dict, result: dict) -> dict:
    """Score only accepted assessments; missing historical labels and errors are not refusals."""
    from finsight.agents.narrative.rag.evidence_summary import _validate_stance

    # Partial evidence cannot establish the whole claim under the three-stance policy.
    # Preserve its positive gold snippets for retrieval evaluation and its original CSV type.
    expected = "UNSUPPORTED" if case["type"] == "partially supported" else case["type"].upper()
    if expected not in STANCE_LABELS:
        raise ValueError(f"Unknown case type: {case['type']}")
    retrieval_only = result.get("pipeline") == "narrative_evidence_retrieval"
    state = result.get("assessment_status", "NOT_ASSESSED")
    errors = []
    if retrieval_only:
        state = "NOT_ASSESSED"
    elif state == "ASSESSED":
        errors = _validate_stance(result, len(result.get("facts", [])))
        if result.get("status") not in {"OK", "NO_EVIDENCE"}:
            errors.append("Assessment has an unsuccessful generation status")
        elif result["status"] == "OK" and not result.get("citation_validation", {}).get("passed"):
            errors.append("Assessment has not passed citation validation")
        elif result["status"] == "NO_EVIDENCE" and (
                result.get("candidate_evidence") or result.get("facts") or result.get("stance") != "UNSUPPORTED"):
            errors.append("Empty-retrieval assessment is inconsistent")
        if errors:
            state = "ERROR"
    elif state == "ERROR" or result.get("status") in {"GENERATION_ERROR", "INVALID_GENERATION"}:
        state = "ERROR"
    else:
        state = "NOT_ASSESSED"
    predicted = result.get("stance") if state == "ASSESSED" else None
    return {"expected_stance": expected, "predicted_stance": predicted,
            "stance_assessment_status": state, "stance_validation_errors": errors,
            "stance_correct": predicted == expected if predicted is not None else None,
            "stance_reason": result.get("stance_reason") if predicted is not None else None,
            "stance_reason_code": result.get("stance_reason_code") if predicted is not None else None}


def summarize_stances(records: list[dict]) -> dict:
    """Each case contributes once. Classification uses valid outputs; coverage retains failures."""
    eligible = [r for r in records if not r.get("retrieval_only", False)]
    valid = [r for r in eligible if r.get("stance_assessment_status") == "ASSESSED"]
    matrix = {gold: {pred: 0 for pred in STANCE_LABELS} for gold in STANCE_LABELS}
    for row in valid:
        matrix[row["expected_stance"]][row["predicted_stance"]] += 1
    per_label = []
    for label in STANCE_LABELS:
        tp = matrix[label][label]
        gold_count = sum(matrix[label].values())
        predicted_count = sum(matrix[g][label] for g in STANCE_LABELS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        per_label.append({"stance": label, "support": gold_count, "predicted": predicted_count,
                          "precision": precision if valid else None, "recall": recall if valid else None,
                          "f1": 2 * tp / (gold_count + predicted_count) if gold_count + predicted_count else
                          (0.0 if valid else None)})
    correct = sum(r["stance_correct"] for r in valid)
    negatives = [r for r in valid if r["expected_stance"] != "SUPPORTED"]
    false_supports = sum(r["predicted_stance"] == "SUPPORTED" for r in negatives)
    return {"eligible_cases": len(eligible), "valid_assessments": len(valid), "correct": correct,
            "assessment_errors": sum(r.get("stance_assessment_status") == "ERROR" for r in eligible),
            "not_assessed": sum(r.get("stance_assessment_status") == "NOT_ASSESSED" for r in eligible),
            "accuracy": correct / len(valid) if valid else None,
            "macro_f1": statistics.mean(r["f1"] for r in per_label) if valid else None,
            "assessment_completion_rate": len(valid) / len(eligible) if eligible else None,
            "all_case_accuracy": correct / len(eligible) if eligible else None,
            "false_supports": false_supports, "valid_negative_cases": len(negatives),
            "false_support_rate": false_supports / len(negatives) if negatives else None,
            "per_label": per_label, "confusion_matrix": matrix,
            "denominators": "accuracy/F1/confusion: valid assessments only; completion/all-case accuracy: "
                            "all saved full-RAG cases; false support: valid gold UNSUPPORTED/CONTRADICTING. "
                            "Macro F1 averages all three labels; undefined class scores are zero when any "
                            "valid assessment exists. Null predictions never become UNSUPPORTED."}


def stance_markdown(scores: dict) -> list[str]:
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    lines = ["", "## Narrative stance evaluation", "",
             f"Valid assessments: {scores['valid_assessments']} / {scores['eligible_cases']}; "
             f"errors: {scores['assessment_errors']}; not assessed: {scores['not_assessed']}.", "",
             "| Metric | Value |", "|---|---|"]
    lines += [f"| {key} | {fmt(scores[key])} |" for key in
              ("accuracy", "macro_f1", "false_support_rate", "assessment_completion_rate", "all_case_accuracy")]
    lines += ["", "| Stance | Valid gold cases | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    lines += [f"| {r['stance']} | {r['support']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} |"
              for r in scores["per_label"]]
    lines += ["", "Confusion matrix: rows are expected labels; columns are predictions.", "",
              "| Expected / predicted | " + " | ".join(STANCE_LABELS) + " |", "|---|---|---|---|"]
    lines += [f"| {g} | " + " | ".join(str(scores["confusion_matrix"][g][p]) for p in STANCE_LABELS) + " |"
              for g in STANCE_LABELS]
    return lines + ["", scores["denominators"]]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    """Publish a complete checkpoint atomically, tolerating brief Windows sharing locks.

    Unique same-directory staging files avoid collisions. Persistent failures retain
    both the last good destination and the new staging payload for inspection.
    """
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            # Access denied / sharing violation / lock violation can be transient
            # when Windows readers or background scanners hold the destination.
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 7:
                raise
            time.sleep(min(0.1 * 2 ** attempt, 0.8))


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
    matches = list(re.finditer(r"block (\d+) \| chars (\d+):(\d+)", case["source_doc"]))
    if not matches:
        raise ValueError(f"Missing source anchor for {case['id']}")
    references, quotes, previous_end = [], [], 0
    for match in matches:
        line, start, end = map(int, match.groups())
        if not 1 <= line <= len(blocks):
            raise ValueError(f"Source block is out of range for {case['id']}")
        block = blocks[line - 1]
        # Each anchor must carry its own document identity, including flattened CSV cells.
        prefix = case["source_doc"][previous_end:match.start()]
        headers = list(re.finditer(r"([^\s|]+)[ \t]+\|[ \t]+([^\s|][^\n|]*)", prefix))
        document_id = headers[-1].group(1) if headers else None
        if block["document_id"] != document_id or not 0 <= start < end <= len(block["text"]):
            raise ValueError(f"Source anchor identity/bounds mismatch for {case['id']} at block {line}")
        if block["publication_date"] > case["cut_off"]:
            raise ValueError(f"Gold source is after cutoff for {case['id']} at block {line}")
        references.append({"block": block, "line": line, "start": start, "end": end})
        quotes.append(block["text"][start:end])
        previous_end = match.end()
    if (not case["gold_snippet"].strip()
            or " ".join(" ".join(quotes).split()) != " ".join(case["gold_snippet"].split())):
        raise ValueError(f"Source anchor does not match gold text for {case['id']}")
    reference = references[0]
    if len(references) > 1:
        reference = {**reference, "spans": references}
    return reference, maps, index["chunks"]


def score_direction(hits: list[dict], reference: dict, maps: dict) -> dict:
    if not reference:
        return dict.fromkeys(METRICS)
    references = reference.get("spans", [reference])

    def spans(hit, ref):
        row = maps[hit["chunk_id"]]
        if row["document_id"] != ref["block"]["document_id"] or row["block_line_1based"] != ref["line"]:
            return []
        return [row["body_char_span"]] + row.get("context_char_spans", [])

    def complete(prefix):
        return all(covers(ref["block"]["text"], ref["start"], ref["end"],
                          [s for h in prefix for s in spans(h, ref)]) for ref in references)

    def source_hit(k):
        documents = {ref["block"]["document_id"] for ref in references}
        return int(any(maps[h["chunk_id"]]["document_id"] in documents
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
                        **stance_record(case, result),
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
            "stance_metrics": summarize_stances(records),
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
    lines += stance_markdown(report["stance_metrics"])
    lines += ["- RAGAS and the Auditor's final-verdict accuracy are not evaluated here. "
              "Stance accuracy compares Narrative's accepted prediction with the reviewed CSV type.", "",
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
