"""Small deterministic BM25 index used by the narrative vertical slice."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from finsight.agents.narrative.corpus.processing.chunk import NarrativeChunk


INDEX_SCHEMA_VERSION = "1.0"
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def build_index(
    chunks: Iterable[NarrativeChunk],
    output_path: Path,
    *,
    corpus_fingerprint: str,
    target_words: int,
    overlap_words: int,
) -> dict[str, Any]:
    """Persist unique chunks plus enough metadata for deterministic BM25 search."""
    unique: list[NarrativeChunk] = []
    seen_text: set[str] = set()
    duplicates = 0
    for chunk in chunks:
        if chunk.text_sha256 in seen_text:
            duplicates += 1
            continue
        seen_text.add(chunk.text_sha256)
        unique.append(chunk)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "corpus_fingerprint": corpus_fingerprint,
        "chunking": {"target_words": target_words, "overlap_words": overlap_words},
        "chunk_count": len(unique),
        "duplicates_excluded": duplicates,
        "chunks": [chunk.to_json() for chunk in unique],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_index(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported narrative index schema: {payload.get('schema_version')!r}")
    return payload


def search_index(
    payload: dict[str, Any],
    query: str,
    top_k: int,
    cutoff_date: date,
    company_ticker: str,
) -> list[dict[str, Any]]:
    if top_k < 1:
        return []
    company_ticker = company_ticker.upper()
    candidates = [
        chunk for chunk in payload["chunks"]
        if chunk["company_ticker"].upper() == company_ticker
        and date.fromisoformat(chunk["publication_date"]) <= cutoff_date
    ]
    query_terms = list(dict.fromkeys(tokenize(query)))
    if not candidates or not query_terms:
        return []

    tokenized = [tokenize(chunk["text"]) for chunk in candidates]
    average_length = sum(len(tokens) for tokens in tokenized) / len(tokenized)
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized) for term in query_terms
    }
    scored: list[tuple[float, str, dict[str, Any]]] = []
    k1, b = 1.5, 0.75
    for chunk, tokens in zip(candidates, tokenized, strict=True):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1 + (len(candidates) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(tokens) / average_length)
            score += inverse_frequency * (frequency * (k1 + 1) / denominator)
        if score > 0:
            scored.append((score, chunk["chunk_id"], chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))

    results: list[dict[str, Any]] = []
    used_locations: set[tuple[str, str]] = set()
    deferred: list[tuple[float, dict[str, Any]]] = []
    for score, _, chunk in scored:
        location = (chunk["document_id"], chunk["citation_span"])
        if location in used_locations:
            deferred.append((score, chunk))
            continue
        used_locations.add(location)
        results.append({**chunk, "retrieval_score": round(score, 6)})
        if len(results) == top_k:
            return results
    for score, chunk in deferred:
        results.append({**chunk, "retrieval_score": round(score, 6)})
        if len(results) == top_k:
            break
    return results
