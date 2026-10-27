"""Evaluate saved RAG matrix outputs offline; no retrieval or judge calls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path
import statistics

from scripts.narrative.evaluation.evaluate_rag_pilot import METRICS, evaluate, read_json, render_report, write_json


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(records: list[dict], direction: str) -> dict:
    positive = [r for r in records if r["type"] != "unsupported"]
    row = {"cases": len(records), "positive_reference_cases": len(positive),
           "unsupported_cases": sum(r["type"] == "unsupported" for r in records),
           "ok_outputs": sum(r["status"] == "OK" and not r["retrieval_only"] for r in records),
           "retrieval_only_outputs": sum(r["retrieval_only"] for r in records),
           "no_evidence_outputs": sum(r["status"] == "NO_EVIDENCE" for r in records),
           "generation_errors": sum(r["status"] == "GENERATION_ERROR" for r in records),
           "invalid_generations": sum(r["status"] == "INVALID_GENERATION" for r in records)}
    for key in METRICS:
        row[key] = mean(r["directional_retrieval"][direction][key] for r in positive)
        row["equal_company_" + key] = mean(
            mean(r["directional_retrieval"][direction][key] for r in positive if r["company"] == ticker)
            for ticker in sorted({r["company"] for r in positive}))
    return row


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(run_dir: Path, workspace: Path) -> dict:
    manifest = read_json(run_dir / "matrix_configuration.json")
    configuration_rows, rows, company_rows, case_rows = [], [], [], []
    expected_ids = manifest["selected_case_ids"]
    for entry in manifest["configurations"]:
        folder = run_dir / entry["folder"]
        report = evaluate(folder, workspace)
        ids = [r["id"] for r in report["records"]]
        if ids != expected_ids:
            raise ValueError(f"Missing/different cases in {entry['folder']}; incomplete runs are not scored as zeros")
        config = read_json(folder / "configuration.json")
        if config.get("retrieval_only", False) != manifest.get("retrieval_only", False):
            raise ValueError("Retrieval-only/full-RAG modes do not match the matrix")
        if config["backend"] != entry["backend"] or config["dataset_sha256"] != manifest["dataset_sha256"]:
            raise ValueError("Configuration/backend/dataset identities do not match the matrix")
        write_json(folder / "evaluation.json", report)
        (folder / "evaluation.md").write_text(render_report(report), encoding="utf-8")
        records = report["records"]
        configuration_rows.append({
            "strategy": entry["strategy"], "backend": entry["backend"],
            "cases": report["cases"], "status_counts": dict(Counter(r["status"] for r in records)),
            "input_tokens": report["input_tokens"], "output_tokens": report["output_tokens"],
            "recorded_cost_usd": report["recorded_cost_usd"],
            "unknown_usage_cases": report["unknown_usage_cases"], "generation_ms": report["generation_ms"],
            "retrieval_only_cases": report["retrieval_only_cases"],
        })
        for direction in ("support", "counter"):
            for stratum in ("all", "supported", "contradicting", "unsupported"):
                subset = records if stratum == "all" else [r for r in records if r["type"] == stratum]
                rows.append({"strategy": entry["strategy"], "backend": entry["backend"],
                             "direction": direction, "stratum": stratum, **summarize(subset, direction)})
            for ticker in sorted({r["company"] for r in records}):
                company_rows.append({"strategy": entry["strategy"], "backend": entry["backend"],
                                     "direction": direction, "company": ticker,
                                     **summarize([r for r in records if r["company"] == ticker], direction)})
            for r in records:
                case_rows.append({"strategy": entry["strategy"], "backend": entry["backend"],
                                  "direction": direction, "id": r["id"], "company": r["company"],
                                  "type": r["type"], "status": r["status"],
                                  **{key: r["directional_retrieval"][direction][key] for key in METRICS}})
    save_csv(run_dir / "comparison.csv", rows)
    save_csv(run_dir / "per_company.csv", company_rows)
    save_csv(run_dir / "per_case.csv", case_rows)
    result = {
        "status": "SCORED_SAVED_OUTPUTS", "configurations": len(configuration_rows),
        "cases_per_configuration": len(expected_ids),
        "case_configuration_runs": sum(r["cases"] for r in configuration_rows),
        "input_tokens": sum(r["input_tokens"] for r in configuration_rows),
        "output_tokens": sum(r["output_tokens"] for r in configuration_rows),
        "recorded_cost_usd": sum(r["recorded_cost_usd"] for r in configuration_rows),
        "unknown_usage_cases": sum(r["unknown_usage_cases"] for r in configuration_rows),
        "source_priority": "P0", "ragas": "NOT_RUN", "new_llm_requests": 0,
        "retrieval_only": manifest.get("retrieval_only", False),
        "new_retrieval_requests": 0, "configuration_results": configuration_rows,
        "aggregates": rows,
        "limitations": ["Known annotated source spans only; labels still need independent review.",
                        "Source Hit@K counts any passage from the annotated document, not evidence relevance.",
                        "MRR@5/10 uses the first single passage covering the complete gold span, not a document hit.",
                        "Each direction is scored separately; their union is not a top-10 ranking.",
                        "Unsupported positive-reference metrics are N/A, not zero.",
                        "Strict citation status is preserved; raw summaries are not judged semantically.",
                        "Fixed-arm vectors may represent truncated inputs; inspect build reports.",
                        "R4 pair truncation follows the explicitly selected policy.",
                        "Full prompt text is saved, but model-side context truncation is not independently verified.",
                        "Generation time excludes retrieval and is affected by model/cache state."],
    }
    write_json(run_dir / "comparison.json", result)
    lines = ["# RAG configuration comparison", "",
             f"{len(expected_ids)} claims x {len(configuration_rows)} configurations = "
             f"{result['case_configuration_runs']} saved pipeline runs. Priority P0. No judge calls.", "",
             "| Strategy | Retrieval | Direction | Source Hit@5 | Source Hit@10 | "
             "Recall@5 | Recall@10 | MRR@5 | MRR@10 | OK outputs / cases |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    if result["retrieval_only"]:
        lines[2] += " Retrieval-only: LLM preparation and citation/summary quality were not assessed."
    for row in (r for r in rows if r["stratum"] == "all"):
        def fmt(key, percent=False):
            v = row[key]
            return "N/A" if v is None else (f"{v:.2%}" if percent else f"{v:.4f}")
        output_status = "N/A (retrieval only)" if result["retrieval_only"] else f"{row['ok_outputs']} / {row['cases']}"
        lines.append(f"| {row['strategy']} | {row['backend']} | {row['direction']} | "
                     f"{fmt('source_hit_at_5', True)} | {fmt('source_hit_at_10', True)} | "
                     f"{fmt('known_reference_recall_at_5', True)} | "
                     f"{fmt('known_reference_recall_at_10', True)} | {fmt('mrr_at_5')} | {fmt('mrr_at_10')} | "
                     f"{output_status} |")
    lines += ["", "OK output counts are technical outcomes, not answer accuracy. "
              "See comparison.csv for type-specific and equal-company metrics, per_company.csv "
              "for company results and per_case.csv for individual cases.", "",
              f"Recorded tokens: {result['input_tokens']:,} input / {result['output_tokens']:,} output. "
              f"Recorded API cost: ${result['recorded_cost_usd']:.6f}; "
              f"cases with unknown usage: {result['unknown_usage_cases']}.", "", "## Limits", ""]
    lines += [f"- {s}" for s in result["limitations"]]
    (run_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(args.run_dir.resolve(), Path.cwd())
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Scoring stopped: {type(exc).__name__}: {exc}\n")
    print(f"Scored {result['case_configuration_runs']} saved runs; no model calls.")
    print(f"Report: {args.run_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
