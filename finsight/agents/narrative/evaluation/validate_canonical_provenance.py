"""Validate provenance in saved canonical development rankings without retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from datetime import date, datetime, timezone


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(experiment: Path, queries_path: Path) -> dict:
    experiment, queries_path = experiment.resolve(), queries_path.resolve()
    config = read_json(experiment / "configuration.json")
    query_hash = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    if query_hash != config["questions_sha256"]:
        raise ValueError("Question file hash differs from the experiment configuration")
    questions = read_jsonl(queries_path)
    if len(questions) != 92 or any(q.get("split") != "dev" for q in questions):
        raise ValueError("Expected the same 92 development questions used by this experiment")
    if {q["ticker"] for q in questions} != set(config["tickers"]):
        raise ValueError("Question company scope differs from the experiment")
    question_by_id = {q["query_id"]: q for q in questions}
    checks, errors = [], []
    total, missing_urls, content_types = 0, 0, Counter()
    for ticker in config["tickers"]:
        company = experiment / "companies" / ticker
        index = read_json(company / "index.json")
        source_map = {row["chunk_id"]: row for row in read_jsonl(company / "source_map.jsonl")}
        inputs = read_json(experiment.parents[1] / "parsed" / ticker / "input_manifest.json")
        source_hashes = {row["document_id"]: row["sha256"] for row in inputs["sources"]}
        chunks = {row["chunk_id"]: row for row in index["chunks"]}
        rankings = read_jsonl(company / "rankings.jsonl")
        expected = {q["query_id"] for q in questions if q["ticker"] == ticker}
        for entry in rankings:
            qid, method = entry.get("query_id"), entry.get("method")
            if qid not in expected or method not in {"R1", "R2", "R3", "R4"}:
                errors.append(f"{ticker}: unexpected query or method in ranking: {qid}/{method}")
                continue
            q = question_by_id[qid]
            if entry.get("cutoff") != q["cutoff_date"] or entry.get("query") != q["claim"]:
                errors.append(f"{ticker}/{qid}/{method}: query or cutoff mismatch")
            seen = set()
            for rank, item in enumerate(entry.get("ranking", []), 1):
                total += 1
                cid = item.get("chunk_id")
                if cid in seen:
                    errors.append(f"{ticker}/{qid}/{method}: duplicate passage {cid}")
                seen.add(cid)
                chunk, mapping = chunks.get(cid), source_map.get(cid)
                if chunk is None or mapping is None:
                    errors.append(f"{ticker}/{qid}/{method} rank {rank}: passage missing from index/source map: {cid}")
                    continue
                fields = {
                    "document_id": "document_id",
                    "company_ticker": "company_ticker",
                    "company_name": "company_name",
                    "document_type": "document_type",
                    "publication_date": "publication_date",
                    "source_tier": "source_tier",
                    "source_url": "source_url",
                    "local_path": "local_path",
                    "citation_span": "citation_span",
                    "text": "text",
                    "text_sha256": "text_sha256",
                }
                for source_field, chunk_field in fields.items():
                    if item.get(source_field) != chunk.get(chunk_field):
                        errors.append(f"{ticker}/{qid}/{method} rank {rank}: {source_field} differs from index")
                        break
                if chunk.get("company_ticker", "").upper() != ticker:
                    errors.append(f"{ticker}/{qid}/{method} rank {rank}: company mismatch")
                if date.fromisoformat(chunk["publication_date"]) > date.fromisoformat(q["cutoff_date"]):
                    errors.append(f"{ticker}/{qid}/{method} rank {rank}: passage after cutoff")
                if hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest() != chunk["text_sha256"]:
                    errors.append(f"{ticker}/{qid}/{method} rank {rank}: passage text hash mismatch")
                if mapping.get("document_id") != chunk.get("document_id") or mapping.get(
                    "source_sha256"
                ) != source_hashes.get(chunk.get("document_id")):
                    errors.append(f"{ticker}/{qid}/{method} rank {rank}: source mapping identity mismatch")
                spans = mapping.get("source_char_spans", [])
                if not spans or any(len(span) != 2 or span[0] < 0 or span[1] <= span[0] for span in spans):
                    errors.append(f"{ticker}/{qid}/{method} rank {rank}: invalid source character spans")
                if not item.get("citation_span"):
                    errors.append(f"{ticker}/{qid}/{method} rank {rank}: missing citation span")
                missing_urls += not bool(chunk.get("source_url"))
                content_types[chunk.get("content_type", "narrative")] += 1
        observed = {(x.get("query_id"), x.get("method")) for x in rankings}
        wanted = {(q["query_id"], m) for q in questions if q["ticker"] == ticker for m in ("R1", "R2", "R3", "R4")}
        if observed != wanted:
            errors.append(f"{ticker}: expected {len(wanted)} query/method rankings, found {len(observed)}")
        checks.append(
            {
                "ticker": ticker,
                "query_method_rankings": len(observed),
                "expected_query_method_rankings": len(wanted),
                "corpus_fingerprint": index["corpus_fingerprint"],
            }
        )
    return {
        "status": "complete" if not errors else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved canonical92 rankings; no retrieval/model calls",
        "claim_count": len(questions),
        "company_count": len(config["tickers"]),
        "query_method_rankings": sum(x["query_method_rankings"] for x in checks),
        "ranked_passages_checked": total,
        "missing_url_occurrences": missing_urls,
        "content_types": dict(content_types),
        "error_count": len(errors),
        "errors": errors,
        "query_set_sha256": query_hash,
        "overflow_policy": config["overflow_policy"],
        "companies": checks,
        "limitations": [
            "Checks saved ranked passage identity, metadata, date scope, citation metadata and source-map identity.",
            (
                "Does not rerun retrieval, judge relevance/entailment, inspect original PDF "
                "accuracy, or validate generated answers."
            ),
            (
                "Recorded character spans are checked for shape and source identity; "
                "source-text OCR/extraction accuracy is outside this audit."
            ),
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.experiment_dir, args.queries)
    output = args.output or args.experiment_dir / "provenance_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        k: report[k]
        for k in (
            "status",
            "claim_count",
            "company_count",
            "query_method_rankings",
            "ranked_passages_checked",
            "missing_url_occurrences",
            "error_count",
        )
    }
    summary["output"] = str(output)
    print(json.dumps(summary, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
