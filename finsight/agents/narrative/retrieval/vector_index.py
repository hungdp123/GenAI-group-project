"""Persistent local FAISS index and BM25/FAISS hybrid retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from finsight.agents.narrative.corpus.processing.index import search_index
from finsight.agents.narrative.retrieval.retrieval_experiments import EmbeddingBackend


FAISS_SCHEMA_VERSION = "1.0"
RRF_K = 60


def _faiss_modules():
    try:
        import faiss
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "FAISS retrieval requires the optional dependency. Install with: "
            "pip install -e '.[retrieval]'"
        ) from exc
    return faiss, np


@dataclass
class FaissVectorIndex:
    index: Any
    corpus_fingerprint: str
    model_name: str
    chunk_ids: list[str]
    dimension: int


def build_faiss_index(
    index_payload: dict[str, Any],
    output_dir: Path,
    backend: EmbeddingBackend,
) -> dict[str, Any]:
    """Embed all BM25 chunks and persist a normalized inner-product FAISS index."""

    faiss, np = _faiss_modules()
    chunks = index_payload["chunks"]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    vectors = np.asarray(backend.encode_documents([chunk["text"] for chunk in chunks]), dtype="float32")
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] < 1:
        raise ValueError(f"Invalid embedding matrix shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain non-finite values")
    faiss.normalize_L2(vectors)
    faiss_index = faiss.IndexFlatIP(int(vectors.shape[1]))
    faiss_index.add(vectors)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(faiss_index, str(output_dir / "faiss.index"))
    metadata = {
        "schema_version": FAISS_SCHEMA_VERSION,
        "corpus_fingerprint": index_payload["corpus_fingerprint"],
        "embedding_model": backend.model_name,
        "dimension": int(vectors.shape[1]),
        "vector_count": int(vectors.shape[0]),
        "chunk_ids": chunk_ids,
        "distance": "cosine_via_normalized_inner_product",
    }
    (output_dir / "faiss_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def load_faiss_index(output_dir: Path, index_payload: dict[str, Any]) -> FaissVectorIndex:
    faiss, _ = _faiss_modules()
    metadata = json.loads((output_dir / "faiss_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != FAISS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported FAISS metadata schema: {metadata.get('schema_version')!r}")
    if metadata.get("corpus_fingerprint") != index_payload.get("corpus_fingerprint"):
        raise ValueError("FAISS and BM25 corpus fingerprints differ")
    expected_ids = [chunk["chunk_id"] for chunk in index_payload["chunks"]]
    if metadata.get("chunk_ids") != expected_ids:
        raise ValueError("FAISS chunk order differs from the BM25 index")
    index = faiss.read_index(str(output_dir / "faiss.index"))
    if index.ntotal != len(expected_ids):
        raise ValueError("FAISS vector count differs from metadata")
    return FaissVectorIndex(
        index=index,
        corpus_fingerprint=metadata["corpus_fingerprint"],
        model_name=metadata["embedding_model"],
        chunk_ids=metadata["chunk_ids"],
        dimension=int(metadata["dimension"]),
    )


def search_faiss(
    vector_index: FaissVectorIndex,
    index_payload: dict[str, Any],
    query_vector: Sequence[float],
    *,
    top_k: int,
    cutoff_date: date,
    company_ticker: str,
) -> list[dict[str, Any]]:
    """Search FAISS and apply the same ticker/date filters as BM25."""

    if top_k < 1 or not vector_index.chunk_ids:
        return []
    faiss, np = _faiss_modules()
    query = np.asarray([list(query_vector)], dtype="float32")
    if query.shape != (1, vector_index.dimension):
        raise ValueError(
            f"Query dimension {query.shape[1] if query.ndim == 2 else None} "
            f"does not match FAISS dimension {vector_index.dimension}"
        )
    faiss.normalize_L2(query)
    scores, positions = vector_index.index.search(query, vector_index.index.ntotal)
    ticker = company_ticker.upper()
    cutoff = cutoff_date.isoformat()
    chunks = index_payload["chunks"]
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0], strict=True):
        if position < 0:
            continue
        chunk = chunks[int(position)]
        if chunk["company_ticker"].upper() != ticker or chunk["publication_date"] > cutoff:
            continue
        results.append({**chunk, "retrieval_score": round(float(score), 6)})
        if len(results) == top_k:
            break
    return results


def reciprocal_rank_fusion(
    bm25_results: Sequence[dict[str, Any]],
    vector_results: Sequence[dict[str, Any]],
    *,
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    methods: dict[str, list[str]] = {}
    for method, ranking in (("bm25", bm25_results), ("faiss", vector_results)):
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item["chunk_id"]
            by_id[chunk_id] = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            methods.setdefault(chunk_id, []).append(method)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        {
            **by_id[chunk_id],
            "retrieval_score": round(scores[chunk_id], 8),
            "retrieval_method": "hybrid_rrf",
            "matched_by": methods[chunk_id],
        }
        for chunk_id in ranked[:top_k]
    ]


def hybrid_search(
    index_payload: dict[str, Any],
    vector_index: FaissVectorIndex,
    backend: EmbeddingBackend,
    query: str,
    *,
    top_k: int,
    cutoff_date: date,
    company_ticker: str,
    candidate_k: int = 50,
) -> dict[str, list[dict[str, Any]]]:
    candidate_k = max(top_k, candidate_k)
    bm25 = search_index(index_payload, query, candidate_k, cutoff_date, company_ticker)
    dense = search_faiss(
        vector_index,
        index_payload,
        backend.encode_query(query),
        top_k=candidate_k,
        cutoff_date=cutoff_date,
        company_ticker=company_ticker,
    )
    return {
        "bm25": bm25[:top_k],
        "faiss": dense[:top_k],
        "hybrid": reciprocal_rank_fusion(bm25, dense, top_k=top_k),
    }
