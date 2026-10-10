"""Corpus-backed implementation of the shared Retriever protocol."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from finsight.agents.narrative.index import load_index, search_index
from finsight.contracts.schemas import EvidenceItem, SourceTier, Stance
from finsight.llm.usage import record_tool_call


class CorpusRetriever:
    name = "narrative_corpus_retriever"

    def __init__(self, index_path: Path | str | None = None, company_ticker: str | None = None) -> None:
        configured_path = os.getenv("FINSIGHT_NARRATIVE_INDEX")
        self.index_path = Path(
            index_path or configured_path or "data/narrative/processed/indexes/NVDA/index.json"
        )
        self.company_ticker = (company_ticker or os.getenv("FINSIGHT_NARRATIVE_TICKER") or "NVDA").upper()
        self._payload: dict | None = None

    def _index(self) -> dict:
        if self._payload is None:
            self._payload = load_index(self.index_path)
        return self._payload

    def retrieve(self, query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]:
        record_tool_call()
        matches = search_index(self._index(), query, top_k, cutoff_date, self.company_ticker)
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
                url=item["source_url"],
                citation_span=item["citation_span"],
                produced_by=self.name,
            )
            for item in matches
        ]
