"""Run a frozen multi-company comparison of BM25, dense, hybrid, and reranked search.

Input JSONL rows use ``query_id``, ``ticker``, ``cutoff_date``, ``support_query``,
``counter_query``, ``expected_support_chunk_ids``, and
``expected_counter_chunk_ids``. Outputs are isolated under each ticker's
``experiments/retrieval_comparison`` directory for dense caches. The combined
report is saved in ``data/narrative/experiments/retrieval_comparison``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from finsight.agents.narrative.processing.index import load_index
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DenseVectorCache,
    FiveWayRetriever,
    SentenceTransformersBackend,
    SentenceTransformersReranker,
)


def _load_queries(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    queries = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not queries:
        raise ValueError(f"No queries found in {path}")
    required = {"query_id", "ticker", "cutoff_date", "support_query", "counter_query",
                "expected_support_chunk_ids", "expected_counter_chunk_ids"}
    for row in queries:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Query {row.get('query_id', '<unknown>')} missing fields: {sorted(missing)}")
        date.fromisoformat(row["cutoff_date"])
    return queries, hashlib.sha256(raw).hexdigest()


def _metrics(results: list[dict[str, Any]], expected: set[str], k: int) -> dict[str, float | int]:
    ranked = results[:k]
    ids = [row["chunk_id"] for row in ranked]
    hits = [index + 1 for index, chunk_id in enumerate(ids) if chunk_id in expected]
    recall = len(set(ids) & expected) / len(expected) if expected else 0.0
    precision = len(set(ids) & expected) / len(ids) if ids else 0.0
    reciprocal_rank = 1.0 / hits[0] if hits else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in hits)
    ideal_hits = min(len(expected), k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {f"recall@{k}": recall, f"precision@{k}": precision, "mrr": reciprocal_rank,
            f"ndcg@{k}": dcg / ideal if ideal else 0.0,
            "relevant_retrieved": len(set(ids) & expected), "expected_relevant": len(expected),
            "returned": len(ids)}


def run_comparison(workspace: Path, queries_path: Path, *, top_k: int = 10,
                   embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                   reranker_model: str = DEFAULT_RERANKER_MODEL,
                   candidate_k: int = 50, tickers: list[str] | None = None,
                   output_dir: Path | None = None) -> dict[str, Any]:
    workspace = workspace.resolve()
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("Require candidate_k >= top_k >= 1")
    queries, query_fingerprint = _load_queries(queries_path)
    selected_tickers = sorted({ticker.upper() for ticker in tickers}) if tickers else None
    if selected_tickers is not None:
        unknown = set(selected_tickers) - {q["ticker"].upper() for q in queries}
        if unknown:
            raise ValueError(f"Companies absent from approved query file: {sorted(unknown)}")
        queries = [q for q in queries if q["ticker"].upper() in selected_tickers]
    embedding = SentenceTransformersBackend(embedding_model)
    reranker = SentenceTransformersReranker(reranker_model)
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        by_ticker.setdefault(query["ticker"].upper(), []).append(query)

    report: dict[str, Any] = {
        "status": "complete", "scope": "current canonical corpus, including promoted PDF/chart content where present",
        "created_at": datetime.now(timezone.utc).isoformat(), "query_set": str(queries_path.resolve()),
        "query_set_sha256": query_fingerprint, "query_count": len(queries), "top_k": top_k,
        "selected_tickers": selected_tickers,
        "candidate_k": candidate_k,
        "queries": queries,
        "embedding_model": embedding.model_name, "reranker_model": reranker.model_name,
        "methods": ["bm25", "dense", "hybrid_rrf", "hybrid_cross_encoder"], "results": [], "summary": {},
        "corpus_fingerprints": {},
    }
    metric_rows: dict[str, list[dict[str, float | int]]] = {name: [] for name in report["methods"]}
    for ticker, ticker_queries in sorted(by_ticker.items()):
        print(f"[{ticker}] Loading current corpus and synchronizing experiment dense cache", flush=True)
        company_root = workspace / "data/narrative/processed" / ticker
        index_path = company_root / "indexes/index.json"
        index = load_index(index_path)
        report["corpus_fingerprints"][ticker] = index["corpus_fingerprint"]
        cache = DenseVectorCache.load_or_build(
            index, company_root / "experiments/retrieval_comparison/dense_cache.json.gz", embedding
        )
        retriever = FiveWayRetriever(index, cache, embedding, reranker)
        for query in ticker_queries:
            print(f"[{ticker}] Retrieving support/counter evidence for {query['query_id']}", flush=True)
            cutoff = date.fromisoformat(query["cutoff_date"])
            started = time.perf_counter()
            support = retriever.search(query["support_query"], top_k, cutoff, ticker, candidate_k=candidate_k)
            counter = retriever.search(query["counter_query"], top_k, cutoff, ticker, candidate_k=candidate_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            expected_support = set(query["expected_support_chunk_ids"])
            expected_counter = set(query["expected_counter_chunk_ids"])
            for method in report["methods"]:
                support_rows, counter_rows = support[method], counter[method]
                metrics = {
                    "support": _metrics(support_rows, expected_support, top_k),
                    "counter_evidence": _metrics(counter_rows, expected_counter, top_k),
                    "cutoff_leaks": sum(date.fromisoformat(row["publication_date"]) > cutoff
                                         for row in support_rows + counter_rows),
                    "wrong_company_results": sum(row["company_ticker"].upper() != ticker
                                                  for row in support_rows + counter_rows),
                    "invalid_citations": sum(not (row.get("chunk_id") and row.get("text")
                                                   and row.get("source_url") and row.get("citation_span"))
                                              for row in support_rows + counter_rows),
                    "duplicate_results": sum(len(rows) - len({row["chunk_id"] for row in rows})
                                             for rows in (support_rows, counter_rows)),
                    "support_counter_overlap": len({row["chunk_id"] for row in support_rows}
                                                   & {row["chunk_id"] for row in counter_rows}),
                    "both_directions_in_top_k": (
                        bool({row["chunk_id"] for row in support_rows} & expected_support)
                        and bool({row["chunk_id"] for row in counter_rows} & expected_counter)
                    ),
                    "retrieval_latency_ms_both_queries": round(elapsed_ms, 3),
                }
                metric_rows[method].append({
                    "support_recall": metrics["support"][f"recall@{top_k}"],
                    "counter_recall": metrics["counter_evidence"][f"recall@{top_k}"],
                    "support_precision": metrics["support"][f"precision@{top_k}"],
                    "counter_precision": metrics["counter_evidence"][f"precision@{top_k}"],
                    "support_mrr": metrics["support"]["mrr"], "counter_mrr": metrics["counter_evidence"]["mrr"],
                    **{key: metrics[key] for key in ("cutoff_leaks", "wrong_company_results", "invalid_citations",
                                                     "duplicate_results", "support_counter_overlap")},
                    "both_directions_in_top_k": int(metrics["both_directions_in_top_k"]),
                    "retrieval_latency_ms_both_queries": metrics["retrieval_latency_ms_both_queries"],
                })
                report["results"].append({
                    "query_id": query["query_id"], "ticker": ticker, "cutoff_date": query["cutoff_date"],
                    "method": method, "metrics": metrics,
                    "support_rankings": support_rows, "counter_rankings": counter_rows,
                    "expected_support_chunk_ids": sorted(expected_support),
                    "expected_counter_chunk_ids": sorted(expected_counter),
                })
    for method, rows in metric_rows.items():
        count = max(1, len(rows))
        report["summary"][method] = {
            key: sum(float(row[key]) for row in rows) / count
            for key in ("support_recall", "counter_recall", "support_precision", "counter_precision",
                        "support_mrr", "counter_mrr", "both_directions_in_top_k",
                        "retrieval_latency_ms_both_queries")
        }
        report["summary"][method].update({
            key: sum(int(row[key]) for row in rows)
            for key in ("cutoff_leaks", "wrong_company_results", "invalid_citations", "duplicate_results",
                        "support_counter_overlap")
        })
    output = (workspace / output_dir if output_dir is not None
              else workspace / "data/narrative/experiments/retrieval_comparison")
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--ticker", action="append", help="Run only this company; repeat to select several.")
    parser.add_argument("--output-dir", type=Path, help="Separate result folder, relative to workspace or absolute.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    args = parser.parse_args()
    report = run_comparison(args.workspace, args.queries, top_k=args.top_k,
                            embedding_model=args.embedding_model, reranker_model=args.reranker_model,
                            candidate_k=args.candidate_k, tickers=args.ticker, output_dir=args.output_dir)
    print(json.dumps({"status": report["status"], "query_count": report["query_count"],
                      "query_set_sha256": report["query_set_sha256"], "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
