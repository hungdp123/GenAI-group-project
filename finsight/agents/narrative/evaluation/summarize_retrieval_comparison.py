"""Audit and score saved retrieval rankings without loading embedding models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from finsight.agents.narrative.evaluation.run_retrieval_comparison import _load_queries, _metrics


def summarize(workspace: Path, results_path: Path, queries_path: Path) -> dict[str, Any]:
    raw = results_path.read_bytes()
    report = json.loads(raw)
    queries, fingerprint = _load_queries(queries_path)
    if fingerprint != report["query_set_sha256"]:
        raise ValueError("Query set changed after the run")
    if report.get("selected_tickers") is not None:
        selected = set(report["selected_tickers"])
        queries = [q for q in queries if q["ticker"].upper() in selected]
        if queries != report["queries"]:
            raise ValueError("Selected queries do not match the recorded subset")
    query_map = {q["query_id"]: q for q in queries}
    if len(query_map) != len(queries):
        raise ValueError("Duplicate query IDs")
    indexes = {}
    for ticker, expected in report["corpus_fingerprints"].items():
        path = workspace / "data/narrative/processed" / ticker / "indexes/index.json"
        index = json.loads(path.read_text(encoding="utf-8"))
        if index["corpus_fingerprint"] != expected:
            raise ValueError(f"{ticker} corpus changed after the run")
        indexes[ticker] = {c["chunk_id"]: c for c in index["chunks"]}
    for q in queries:
        index = indexes[q["ticker"].upper()]
        for chunk_id in set(q["expected_support_chunk_ids"] + q["expected_counter_chunk_ids"]):
            c = index.get(chunk_id)
            if c is None or c["publication_date"] > q["cutoff_date"] or c["company_ticker"] != q["ticker"]:
                raise ValueError(f"Missing or ineligible gold chunk: {chunk_id}")

    expected_pairs = {(q["query_id"], m) for q in queries for m in report["methods"]}
    actual_pairs = [(r["query_id"], r["method"]) for r in report["results"]]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Missing or duplicate query/method results")

    ks = [k for k in (1, 3, 5, 10) if k <= report["top_k"]]
    rows, audits = [], []
    for result in report["results"]:
        q = query_map[result["query_id"]]
        index = indexes[q["ticker"].upper()]
        for stance, ranking_key, gold_key in (
            ("support", "support_rankings", "expected_support_chunk_ids"),
            ("counter", "counter_rankings", "expected_counter_chunk_ids"),
        ):
            ranking = result[ranking_key]
            for k in ks:
                rows.append({"query_id": q["query_id"], "ticker": q["ticker"], "method": result["method"],
                             "stance": stance, "k": k, **_metrics(ranking, set(q[gold_key]), k)})
            audit = {"query_id": q["query_id"], "method": result["method"], "stance": stance,
                     "returned": len(ranking), "cutoff_leaks": 0, "wrong_company_results": 0,
                     "duplicate_results": len(ranking) - len({c["chunk_id"] for c in ranking}),
                     "missing_citation_metadata": 0, "index_mismatches": 0, "missing_source_files": 0,
                     "citation_issues": []}
            for c in ranking:
                canonical = index.get(c["chunk_id"])
                audit["cutoff_leaks"] += c["publication_date"] > q["cutoff_date"]
                audit["wrong_company_results"] += c["company_ticker"].upper() != q["ticker"].upper()
                missing = [f for f in ("document_id", "text", "source_url", "citation_span", "source_tier")
                           if not c.get(f)]
                audit["missing_citation_metadata"] += bool(missing)
                if missing:
                    audit["citation_issues"].append({"chunk_id": c["chunk_id"], "missing_fields": missing})
                fields = ("text", "document_id", "source_url", "citation_span", "source_tier",
                          "company_ticker", "publication_date")
                audit["index_mismatches"] += canonical is None or any(
                    canonical.get(f) != c.get(f) for f in fields
                )
                source = c.get("local_path")
                audit["missing_source_files"] += not source or not Path(source).is_file()
            audits.append(audit)

    aggregates = []
    for method in report["methods"]:
        for stance in ("support", "counter"):
            for k in ks:
                group = [r for r in rows if r["method"] == method and r["stance"] == stance and r["k"] == k]
                aggregates.append({"method": method, "stance": stance, "k": k, "query_count": len(group),
                                   **{key: sum(r[key] for r in group) / len(group)
                                      for key in (f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}")}})
    checks = {m: {key: sum(a[key] for a in audits if a["method"] == m)
                  for key in ("returned", "cutoff_leaks", "wrong_company_results", "duplicate_results",
                              "missing_citation_metadata", "index_mismatches", "missing_source_files")}
              for m in report["methods"]}
    return {"created_at": report["created_at"], "raw_results_sha256": hashlib.sha256(raw).hexdigest(),
            "selected_tickers": report.get("selected_tickers"),
            "query_set_sha256": fingerprint, "corpus_fingerprints": report["corpus_fingerprints"],
            "chunk_counts": {t: len(index) for t, index in indexes.items()}, "queries": queries,
            "embedding_model": report["embedding_model"], "reranker_model": report["reranker_model"],
            "top_k": report["top_k"], "candidate_k": report["candidate_k"],
            "aggregates": aggregates, "checks": checks, "per_query": rows, "citation_audits": audits,
            "limitations": [
                f"{len(queries)} development claims ({len(queries) * 2} directional queries) with sparse curated labels; no heldout evaluation or method selection.",
                "Precision counts labeled positives only; unlabelled passages were not judged irrelevant.",
                "Citation checks establish index consistency and metadata presence, not source-text entailment.",
                "Recorded retrieval latency covers both queries and all methods together, not each method separately.",
                "Support/counter ranking overlap is allowed; duplicates are counted within each ranking.",
            ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    args = parser.parse_args()
    evaluation = summarize(args.workspace.resolve(), args.results, args.queries)
    output = args.results.parent
    (output / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = evaluation["per_query"]
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"checks": evaluation["checks"], "aggregates": evaluation["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
