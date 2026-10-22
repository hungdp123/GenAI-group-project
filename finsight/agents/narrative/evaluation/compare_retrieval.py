"""Run the five narrative-retrieval approaches on the same query and corpus.

Example:
    python -m finsight.agents.narrative.evaluation.compare_retrieval \
      --query "NVDA data center demand risks and customer concentration" \
      --cutoff 2025-07-31
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from finsight.agents.narrative.corpus.processing.index import load_index
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DenseVectorCache,
    FiveWayRetriever,
    SentenceTransformersBackend,
    SentenceTransformersReranker,
)


def compare_query(
    *,
    index_path: Path,
    query: str,
    cutoff_date: date,
    company_ticker: str,
    top_k: int,
    candidate_k: int,
    embedding_model: str,
    reranker_model: str,
    dense_cache_path: Path | None = None,
) -> dict[str, Any]:
    index = load_index(index_path)
    cache_path = dense_cache_path or index_path.with_name("dense_vectors.json.gz")
    embeddings = SentenceTransformersBackend(embedding_model)
    vectors = DenseVectorCache.load_or_build(index, cache_path, embeddings)
    reranker = SentenceTransformersReranker(reranker_model)
    retriever = FiveWayRetriever(index, vectors, embeddings, reranker)
    rankings = retriever.search(
        query,
        top_k,
        cutoff_date,
        company_ticker,
        candidate_k=candidate_k,
    )
    return {
        "corpus_fingerprint": index["corpus_fingerprint"],
        "company_ticker": company_ticker.upper(),
        "query": query,
        "cutoff_date": cutoff_date.isoformat(),
        "top_k": top_k,
        "candidate_k": max(top_k, candidate_k),
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "rankings": rankings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/narrative/processed/NVDA/indexes/index.json"))
    parser.add_argument("--dense-cache", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True)
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_query(
        index_path=args.index,
        query=args.query,
        cutoff_date=args.cutoff,
        company_ticker=args.ticker,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        dense_cache_path=args.dense_cache,
    )
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(f"Saved comparison output to {args.output}")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
