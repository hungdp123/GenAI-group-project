"""Persistent local Qdrant index and BM25/vector hybrid retrieval."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

from finsight.agents.narrative.corpus.processing.index import search_index
from finsight.agents.narrative.retrieval.retrieval_experiments import EmbeddingBackend


QDRANT_SCHEMA_VERSION = "1.0"
QDRANT_COLLECTION = "narrative_chunks"
RRF_K = 60


def _qdrant_modules():
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Local Qdrant retrieval requires the optional dependency. Install with: "
            "pip install -e '.[retrieval]'"
        ) from exc
    return QdrantClient, models


@dataclass
class QdrantVectorIndex:
    client: Any
    collection_name: str
    corpus_fingerprint: str
    model_name: str
    chunk_ids: list[str]
    dimension: int


def build_qdrant_index(
    index_payload: dict[str, Any],
    output_dir: Path,
    backend: EmbeddingBackend,
) -> dict[str, Any]:
    """Embed every BM25 chunk and persist it in embedded, local Qdrant."""

    QdrantClient, models = _qdrant_modules()
    chunks = index_payload["chunks"]
    if not chunks:
        raise ValueError("Cannot build a vector index without chunks")
    vectors = backend.encode_documents([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count differs from chunk count")
    vectors = [list(map(float, vector)) for vector in vectors]
    dimension = len(vectors[0])
    if dimension < 1 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("Embeddings have inconsistent or empty dimensions")
    if any(not math.isfinite(value) for vector in vectors for value in vector):
        raise ValueError("Embeddings contain non-finite values")

    output_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(output_dir / "storage"))
    try:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            points = [
                models.PointStruct(
                    id=position,
                    vector=vectors[position],
                    payload={
                        "position": position,
                        "chunk_id": chunks[position]["chunk_id"],
                        "company_ticker": chunks[position]["company_ticker"],
                        "publication_date": chunks[position]["publication_date"],
                        "publication_date_is_estimated": chunks[position].get(
                            "publication_date_is_estimated", False
                        ),
                        "publication_date_precision": chunks[position].get(
                            "publication_date_precision", "day"
                        ),
                        "publication_date_basis": chunks[position].get("publication_date_basis"),
                        "document_id": chunks[position]["document_id"],
                        "document_type": chunks[position]["document_type"],
                        "source_tier": chunks[position]["source_tier"],
                        "source_url": chunks[position]["source_url"],
                        "citation_span": chunks[position]["citation_span"],
                        "page": chunks[position].get("page"),
                        "chart_id": chunks[position].get("chart_id"),
                        "source_ref": chunks[position].get("source_ref"),
                        "content_type": chunks[position].get("content_type", "narrative"),
                        "extraction_method": chunks[position].get("extraction_method"),
                        "validation_status": chunks[position].get("validation_status"),
                    },
                )
                for position in range(start, min(start + batch_size, len(chunks)))
            ]
            client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
    finally:
        client.close()

    metadata = {
        "schema_version": QDRANT_SCHEMA_VERSION,
        "corpus_fingerprint": index_payload["corpus_fingerprint"],
        "source_corpus_fingerprint": index_payload.get("source_corpus_fingerprint"),
        "pdf_processing": index_payload.get("pdf_processing", {}),
        "embedding_model": backend.model_name,
        "dimension": dimension,
        "vector_count": len(vectors),
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "distance": "cosine",
        "collection_name": QDRANT_COLLECTION,
    }
    (output_dir / "qdrant_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def load_qdrant_index(
    output_dir: Path,
    index_payload: dict[str, Any],
    *,
    expected_model_name: str | None = None,
    expected_dimension: int | None = None,
) -> QdrantVectorIndex:
    QdrantClient, _ = _qdrant_modules()
    metadata = json.loads((output_dir / "qdrant_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != QDRANT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Qdrant metadata schema: {metadata.get('schema_version')!r}")
    if metadata.get("corpus_fingerprint") != index_payload.get("corpus_fingerprint"):
        raise ValueError("Qdrant and BM25 corpus fingerprints differ")
    if expected_model_name is not None and metadata.get("embedding_model") != expected_model_name:
        raise ValueError(
            f"Qdrant embedding model {metadata.get('embedding_model')!r} "
            f"does not match configured model {expected_model_name!r}"
        )
    if expected_dimension is not None and int(metadata.get("dimension", 0)) != expected_dimension:
        raise ValueError(
            f"Qdrant dimension {metadata.get('dimension')!r} "
            f"does not match configured dimension {expected_dimension}"
        )
    expected_ids = [chunk["chunk_id"] for chunk in index_payload["chunks"]]
    if metadata.get("chunk_ids") != expected_ids:
        raise ValueError("Qdrant chunk order differs from the BM25 index")
    client = QdrantClient(path=str(output_dir / "storage"))
    collection_name = metadata["collection_name"]
    collection = client.get_collection(collection_name)
    if collection.points_count != len(expected_ids):
        client.close()
        raise ValueError("Qdrant vector count differs from metadata")
    vector_config = collection.config.params.vectors
    collection_dimension = getattr(vector_config, "size", None)
    if collection_dimension is not None and int(collection_dimension) != int(metadata["dimension"]):
        client.close()
        raise ValueError("Qdrant collection dimension differs from metadata")
    return QdrantVectorIndex(
        client=client,
        collection_name=collection_name,
        corpus_fingerprint=metadata["corpus_fingerprint"],
        model_name=metadata["embedding_model"],
        chunk_ids=metadata["chunk_ids"],
        dimension=int(metadata["dimension"]),
    )


def search_qdrant(
    vector_index: QdrantVectorIndex,
    index_payload: dict[str, Any],
    query_vector: Sequence[float],
    *,
    top_k: int,
    cutoff_date: date,
    company_ticker: str,
) -> list[dict[str, Any]]:
    """Search local Qdrant with the same ticker and publication cutoff as BM25."""

    if top_k < 1 or not vector_index.chunk_ids:
        return []
    query = list(map(float, query_vector))
    if len(query) != vector_index.dimension:
        raise ValueError(
            f"Query dimension {len(query)} does not match Qdrant dimension {vector_index.dimension}"
        )
    response = vector_index.client.query_points(
        collection_name=vector_index.collection_name,
        query=query,
        limit=len(vector_index.chunk_ids),
        with_payload=True,
    )
    ticker = company_ticker.upper()
    cutoff = cutoff_date.isoformat()
    chunks = index_payload["chunks"]
    results: list[dict[str, Any]] = []
    for point in response.points:
        position = int(point.payload["position"])
        chunk = chunks[position]
        if chunk["company_ticker"].upper() != ticker or chunk["publication_date"] > cutoff:
            continue
        results.append({**chunk, "retrieval_score": round(float(point.score), 6)})
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
    for method, ranking in (("bm25", bm25_results), ("qdrant", vector_results)):
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item["chunk_id"]
            by_id[chunk_id] = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            methods.setdefault(chunk_id, []).append(method)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    selected = ranked[:top_k]
    # Exact financial labels and values are often stronger lexical signals than
    # general-purpose semantic embeddings. Keep the top two BM25 candidates in
    # the final set so fusion cannot reduce exact-match recall; Qdrant still
    # reorders those candidates and supplies the remaining results.
    anchors = [item["chunk_id"] for item in bm25_results[: min(2, top_k)]]
    for anchor in anchors:
        if anchor in selected:
            continue
        replace_at = next(
            (index for index in range(len(selected) - 1, -1, -1) if selected[index] not in anchors),
            None,
        )
        if replace_at is None:
            break
        selected[replace_at] = anchor
    selected.sort(key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        {
            **by_id[chunk_id],
            "retrieval_score": round(scores[chunk_id], 8),
            "retrieval_method": "hybrid_rrf",
            "matched_by": methods[chunk_id],
        }
        for chunk_id in selected
    ]


def hybrid_search(
    index_payload: dict[str, Any],
    vector_index: QdrantVectorIndex,
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
    dense = search_qdrant(
        vector_index,
        index_payload,
        backend.encode_query(query),
        top_k=candidate_k,
        cutoff_date=cutoff_date,
        company_ticker=company_ticker,
    )
    return {
        "bm25": bm25[:top_k],
        "qdrant": dense[:top_k],
        "hybrid": reciprocal_rank_fusion(bm25, dense, top_k=top_k),
    }
