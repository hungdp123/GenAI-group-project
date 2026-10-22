"""Cited-passage Basic RAG retrieval with selectable BM25, dense, or hybrid search.

Run ``python -m finsight.agents.narrative.retrieval.retriever --help`` for the passage CLI.
The instance is scoped to its configured ticker; callers must supply the company
identified in the claim. Retrieval returns neutral evidence, without generation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from finsight.agents.narrative.corpus.processing.index import load_index, search_index
from finsight.contracts.schemas import EvidenceItem, SourceTier, Stance
from finsight.llm.usage import record_tool_call

if TYPE_CHECKING:
    from finsight.agents.narrative.retrieval.retrieval_experiments import EmbeddingBackend


BACKENDS = ("bm25", "dense", "hybrid", "auto")
_VECTOR_LOCK_GUARD = threading.Lock()
_VECTOR_LOCKS: dict[str, threading.RLock] = {}


class CorpusRetriever:
    name = "narrative_corpus_retriever"

    def __init__(
        self,
        index_path: Path | str | None = None,
        company_ticker: str | None = None,
        qdrant_dir: Path | str | None = None,
        *,
        backend: str | None = None,
        corpus_root: Path | str | None = None,
        candidate_k: int | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> None:
        ticker = company_ticker if company_ticker is not None else os.getenv("FINSIGHT_NARRATIVE_TICKER", "NVDA")
        self.company_ticker = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", self.company_ticker):
            raise ValueError("Company ticker must contain only letters, numbers, dots, or hyphens")
        selected = backend if backend is not None else os.getenv("FINSIGHT_NARRATIVE_BACKEND", "bm25")
        self.backend = selected.strip().lower()
        if self.backend not in BACKENDS:
            raise ValueError(f"Unknown retrieval backend {selected!r}; choose from {', '.join(BACKENDS)}")
        self.candidate_k = candidate_k if candidate_k is not None else int(
            os.getenv("FINSIGHT_NARRATIVE_CANDIDATE_K", "50")
        )
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        root = Path(corpus_root or os.getenv("FINSIGHT_NARRATIVE_CORPUS_ROOT", "data/narrative/processed"))
        configured_path = index_path or os.getenv("FINSIGHT_NARRATIVE_INDEX")
        self._explicit_index = configured_path is not None
        self.index_path = (
            Path(configured_path) if configured_path else root / self.company_ticker / "indexes/index.json"
        )
        configured_qdrant = qdrant_dir or os.getenv("FINSIGHT_NARRATIVE_QDRANT")
        inferred_qdrant = self.index_path.parent / "qdrant"
        self.qdrant_dir = Path(configured_qdrant) if configured_qdrant else inferred_qdrant
        self._payload: dict | None = None
        self._embedding_backend = embedding_backend
        self.last_backend: str | None = None

    def _index(self) -> dict:
        if self._payload is None:
            self._payload = load_index(self.index_path)
        return self._payload

    def coverage_limitations(self, cutoff_date: date) -> list[dict]:
        """Expose PDF/chart extraction gaps for this company's temporally eligible sources."""
        cutoff = cutoff_date.isoformat()
        return [item for item in self._index().get("coverage_limitations", [])
                if item.get("company_ticker") == self.company_ticker
                and item.get("publication_date", "9999-12-31") <= cutoff]

    def retrieve(self, query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]:
        """Return original passages scoped to this instance's company and cutoff.

        No eligible corpus, a blank query, or nonpositive K returns an empty list.
        Explicit missing paths, missing requested vector indexes, and stale vector
        metadata raise errors; dense/hybrid never silently change to BM25.
        """
        record_tool_call()
        self.last_backend = None
        if top_k < 1 or not query.strip():
            return []
        if not self._explicit_index and not self.index_path.is_file():
            return []
        payload = self._index()
        if not any(item["company_ticker"].upper() == self.company_ticker
                   and item["publication_date"] <= cutoff_date.isoformat() for item in payload["chunks"]):
            return []
        self.last_backend = self.backend
        if self.backend == "auto":
            self.last_backend = "hybrid" if (self.qdrant_dir / "qdrant_metadata.json").is_file() else "bm25"
        if self.last_backend == "bm25":
            matches = search_index(payload, query, top_k, cutoff_date, self.company_ticker)
        else:
            matches = self._vector_matches(payload, query, top_k, cutoff_date)
        return [
            EvidenceItem(
                evidence_id=item["chunk_id"],
                sub_claim_id="claim",
                stance=Stance.NEUTRAL,
                content=item["text"],
                source_tier=SourceTier(item["source_tier"]),
                doc_type=item["document_type"],
                company=item["company_name"],
                period=item.get("period"),
                publication_date=date.fromisoformat(item["publication_date"]),
                publication_date_is_estimated=bool(item.get("publication_date_is_estimated", False)),
                publication_date_precision=item.get("publication_date_precision", "day"),
                publication_date_basis=item.get("publication_date_basis"),
                url=item.get("source_url"),
                citation_span=item.get("citation_span"),
                produced_by=self.name,
            )
            for item in matches
        ]

    def _vector_matches(self, payload: dict, query: str, top_k: int, cutoff_date: date) -> list[dict]:
        # The orchestrator can collect several sub-claims for the same company
        # concurrently. Embedded Qdrant permits one open client per storage path.
        key = str(self.qdrant_dir.resolve()).casefold()
        with _VECTOR_LOCK_GUARD:
            lock = _VECTOR_LOCKS.setdefault(key, threading.RLock())
        with lock:
            return self._vector_matches_locked(payload, query, top_k, cutoff_date)

    def _vector_matches_locked(self, payload: dict, query: str, top_k: int, cutoff_date: date) -> list[dict]:
        from finsight.agents.narrative.retrieval.qdrant_index import hybrid_search, load_qdrant_index, search_qdrant
        from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend

        if not (self.qdrant_dir / "qdrant_metadata.json").is_file():
            raise FileNotFoundError(
                f"Backend {self.last_backend!r} requires a Qdrant index at {self.qdrant_dir}. "
                f"Build it with: python -m finsight.agents.narrative.retrieval.build_vector_databases "
                f"--ticker {self.company_ticker}"
            )
        vectors = load_qdrant_index(
            self.qdrant_dir, payload,
            expected_model_name=self._embedding_backend.model_name if self._embedding_backend is not None else None,
        )
        try:
            if self._embedding_backend is None:
                self._embedding_backend = SentenceTransformersBackend(vectors.model_name)
            if self.last_backend == "dense":
                return search_qdrant(
                    vectors, payload, self._embedding_backend.encode_query(query), top_k=top_k,
                    cutoff_date=cutoff_date, company_ticker=self.company_ticker,
                )
            return hybrid_search(
                payload, vectors, self._embedding_backend, query, top_k=top_k,
                cutoff_date=cutoff_date, company_ticker=self.company_ticker, candidate_k=self.candidate_k,
            )["hybrid"]
        finally:
            # Release embedded Qdrant's storage lock even if encoding/search fails.
            # Cache the model, but let other calls/instances reopen the database.
            vectors.client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Company to retrieve for (never inferred from query text)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--cutoff", required=True, type=date.fromisoformat)
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=None)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--qdrant-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optionally save the same JSON printed to stdout")
    args = parser.parse_args()
    try:
        retriever = CorpusRetriever(
            index_path=args.index, company_ticker=args.ticker, qdrant_dir=args.qdrant_dir,
            backend=args.backend, corpus_root=args.corpus_root, candidate_k=args.candidate_k,
        )
        items = retriever.retrieve(args.query, args.top_k, args.cutoff)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Retrieval failed: {exc}\n")
    result = {
        "status": "OK" if items else "NO_EVIDENCE",
        "company_ticker": retriever.company_ticker, "cutoff_date": args.cutoff.isoformat(),
        "query": args.query, "requested_backend": retriever.backend, "backend_used": retriever.last_backend,
        "top_k": args.top_k, "candidate_k": max(args.top_k, retriever.candidate_k),
        "corpus_fingerprint": retriever._payload.get("corpus_fingerprint") if retriever._payload is not None else None,
        "evidence": [item.model_dump(mode="json") for item in items],
        "citation_gaps": [{"evidence_id": item.evidence_id,
                           "missing_fields": [name for name in ("url", "citation_span") if not getattr(item, name)]}
                          for item in items if not item.url or not item.citation_span],
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
