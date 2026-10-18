"""Comparable BM25, dense, hybrid, and cross-encoder retrieval experiments.

The corpus remains the existing JSON BM25 index. Dense vectors are kept in a
separate compressed cache so the baseline index and its provenance stay intact.
Install the optional ``retrieval`` extra to use Sentence Transformers models.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, Sequence

from finsight.agents.narrative.processing.index import search_index


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
DENSE_SCHEMA_VERSION = "1.0"


class EmbeddingBackend(Protocol):
    model_name: str

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def encode_query(self, text: str) -> Sequence[float]: ...


class RerankerBackend(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


class SentenceTransformersBackend:
    """Lazy adapter: importing this module does not require optional ML packages."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Dense retrieval requires the optional dependency. Install with: "
                "pip install -e .[retrieval]"
            ) from exc
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except OSError:
            self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        encode = getattr(self.model, "encode_document", self.model.encode)
        vectors = encode(list(texts), normalize_embeddings=True, show_progress_bar=True)
        return vectors.tolist()

    def encode_query(self, text: str) -> Sequence[float]:
        encode = getattr(self.model, "encode_query", self.model.encode)
        vector = encode(text, normalize_embeddings=True, show_progress_bar=False)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        # encode_query returns a 1-D vector for one string in supported versions.
        if vector and isinstance(vector[0], list):
            return vector[0]
        return vector


class SentenceTransformersReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires the optional dependency. Install with: "
                "pip install -e .[retrieval]"
            ) from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        scores = self.model.predict([(query, text) for text in texts], show_progress_bar=False)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return [float(score) for score in scores]


@dataclass
class DenseVectorCache:
    corpus_fingerprint: str
    model_name: str
    chunk_ids: list[str]
    vectors: list[list[float]]

    @classmethod
    def load_or_build(
        cls,
        index_payload: dict[str, Any],
        cache_path: Path,
        backend: EmbeddingBackend,
    ) -> "DenseVectorCache":
        fingerprint = index_payload["corpus_fingerprint"]
        chunk_ids = [chunk["chunk_id"] for chunk in index_payload["chunks"]]
        if cache_path.exists():
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                cached = json.load(handle)
            if (
                cached.get("schema_version") == DENSE_SCHEMA_VERSION
                and cached.get("corpus_fingerprint") == fingerprint
                and cached.get("model_name") == backend.model_name
                and cached.get("chunk_ids") == chunk_ids
            ):
                return cls(fingerprint, backend.model_name, chunk_ids, cached["vectors"])

        vectors = [list(map(float, row)) for row in backend.encode_documents(
            [chunk["text"] for chunk in index_payload["chunks"]]
        )]
        _validate_vectors(vectors, len(chunk_ids))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": DENSE_SCHEMA_VERSION,
            "corpus_fingerprint": fingerprint,
            "model_name": backend.model_name,
            "chunk_ids": chunk_ids,
            "vectors": vectors,
        }
        with gzip.open(cache_path, "wt", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        return cls(fingerprint, backend.model_name, chunk_ids, vectors)


def _validate_vectors(vectors: Sequence[Sequence[float]], expected_rows: int) -> None:
    if len(vectors) != expected_rows:
        raise ValueError(f"Embedding count {len(vectors)} does not match chunk count {expected_rows}")
    if not vectors:
        return
    dimensions = len(vectors[0])
    if not dimensions or any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Embeddings must be non-empty vectors with consistent dimensions")


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Query and document embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right, strict=True))


def _diverse(results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    locations: set[tuple[str, str]] = set()
    for result in results:
        location = (result["document_id"], result["citation_span"])
        if location in locations:
            deferred.append(result)
        else:
            locations.add(location)
            chosen.append(result)
        if len(chosen) == top_k:
            return chosen
    return (chosen + deferred)[:top_k]


class FiveWayRetriever:
    """Run the requested five approaches against one filtered candidate corpus."""

    def __init__(
        self,
        index_payload: dict[str, Any],
        dense_cache: DenseVectorCache,
        embedding_backend: EmbeddingBackend,
        reranker: RerankerBackend,
        *,
        rrf_k: int = RRF_K,
    ) -> None:
        if dense_cache.corpus_fingerprint != index_payload["corpus_fingerprint"]:
            raise ValueError("Dense cache fingerprint does not match the BM25 corpus")
        if dense_cache.chunk_ids != [chunk["chunk_id"] for chunk in index_payload["chunks"]]:
            raise ValueError("Dense cache chunk order does not match the BM25 corpus")
        self.index = index_payload
        self.cache = dense_cache
        self.embedding_backend = embedding_backend
        self.reranker = reranker
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int,
        cutoff_date: date,
        company_ticker: str,
        *,
        candidate_k: int = 50,
    ) -> dict[str, list[dict[str, Any]]]:
        if top_k < 1:
            return {name: [] for name in self.method_names}
        candidate_k = max(top_k, candidate_k)
        ticker = company_ticker.upper()
        chunks = self.index["chunks"]
        eligible = [
            chunk for chunk in chunks
            if chunk["company_ticker"].upper() == ticker
            and chunk["publication_date"] <= cutoff_date.isoformat()
        ]
        eligible_ids = {chunk["chunk_id"] for chunk in eligible}

        bm25 = search_index(self.index, query, candidate_k, cutoff_date, ticker)
        query_vector = list(map(float, self.embedding_backend.encode_query(query)))
        dense_scored = [
            (_dot(query_vector, self.cache.vectors[position]), chunk)
            for position, chunk in enumerate(chunks)
            if chunk["chunk_id"] in eligible_ids
        ]
        dense_scored.sort(key=lambda pair: (-pair[0], pair[1]["chunk_id"]))
        dense = [
            {**chunk, "retrieval_score": round(score, 6)}
            for score, chunk in dense_scored[:candidate_k]
        ]
        dense = _diverse(dense, candidate_k)
        hybrid = self._rrf(bm25, dense, candidate_k)

        bm25_top = bm25[:candidate_k]
        hybrid_top = hybrid[:candidate_k]
        methods = {
            "bm25": bm25,
            "dense": dense,
            "hybrid_rrf": hybrid,
            "bm25_cross_encoder": self._rerank(query, bm25_top),
            "hybrid_cross_encoder": self._rerank(query, hybrid_top),
        }
        return {
            name: [dict(item, retrieval_method=name) for item in values[:top_k]]
            for name, values in methods.items()
        }

    @property
    def method_names(self) -> tuple[str, ...]:
        return ("bm25", "dense", "hybrid_rrf", "bm25_cross_encoder", "hybrid_cross_encoder")

    def _rrf(
        self,
        first: Sequence[dict[str, Any]],
        second: Sequence[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        scores: dict[str, float] = {}
        for ranking in (first, second):
            for rank, item in enumerate(ranking, start=1):
                chunk_id = item["chunk_id"]
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        fused = [
            {**self._chunk_by_id(chunk_id), "retrieval_score": round(scores[chunk_id], 8)}
            for chunk_id in ranked[:limit]
        ]
        return _diverse(fused, limit)

    def _chunk_by_id(self, chunk_id: str) -> dict[str, Any]:
        for chunk in self.index["chunks"]:
            if chunk["chunk_id"] == chunk_id:
                return chunk
        raise KeyError(chunk_id)

    def _rerank(self, query: str, candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        scores = self.reranker.score(query, [item["text"] for item in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned a different number of scores than candidates")
        ranked = [
            {**item, "retrieval_score": float(score)}
            for item, score in zip(candidates, scores, strict=True)
        ]
        ranked.sort(key=lambda item: (-item["retrieval_score"], item["chunk_id"]))
        return _diverse(ranked, len(ranked))
