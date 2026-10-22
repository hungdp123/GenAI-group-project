"""Orchestrator retrieval entry point and the retained offline demo retriever.

Register ``NarrativeRetriever`` for real corpus retrieval. ``StubRetriever`` keeps
the deterministic demo behavior used by the project's tests.
"""

from __future__ import annotations

import re
import os
from datetime import date
from pathlib import Path

from finsight.agents.demo_corpus import NARRATIVE_CHUNKS
from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
from finsight.agents.claim_extraction.thesaurus import corpus_aliases
from finsight.contracts.schemas import EvidenceItem, Stance
from finsight.llm.usage import record_tool_call


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class NarrativeRetriever:
    """Real Retriever contract with explicit or unambiguous company scope.

    With no configured ticker, infer the company from the query. An unscoped
    query returns no evidence; an ambiguous query requires an explicit ticker.
    This avoids the lower-level retriever's default NVDA scope in a registry.
    Construction and import do not parse PDFs, rebuild indexes or call models.
    """

    name = "narrative_corpus_retriever"

    def __init__(self, *, company_ticker: str | None = None,
                 corpus_root: Path | str | None = None, backend: str | None = None,
                 candidate_k: int | None = None, **retriever_options) -> None:
        configured = company_ticker or os.getenv("FINSIGHT_NARRATIVE_TICKER")
        self.company_ticker = configured.strip().upper() if configured else None
        self.corpus_root = corpus_root
        self.backend = backend
        self.candidate_k = candidate_k
        self.retriever_options = retriever_options

    def retrieve(self, query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]:
        if type(cutoff_date) is not date:
            raise ValueError("cutoff_date must be a datetime.date")
        if top_k < 1 or not query.strip():
            return []
        ticker = self.company_ticker
        if ticker is None:
            candidates = {
                company for company, aliases in corpus_aliases().items()
                if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", query, re.I)
                       for alias in [company, *aliases])
            }
            if len(candidates) > 1:
                raise ValueError("Query names multiple companies; configure company_ticker explicitly")
            if not candidates:
                return []
            ticker = candidates.pop()
        retriever = CorpusRetriever(company_ticker=ticker, corpus_root=self.corpus_root,
                                    backend=self.backend, candidate_k=self.candidate_k,
                                    **self.retriever_options)
        return retriever.retrieve(query, top_k, cutoff_date)


class StubRetriever:
    name = "retriever_stub"

    def retrieve(self, query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]:
        record_tool_call()
        q = _tokens(query)
        candidates = [c for c in NARRATIVE_CHUNKS if c["publication_date"] <= cutoff_date]
        ranked = sorted(candidates, key=lambda c: (-len(q & _tokens(c["content"])), c["evidence_id"]))
        return [
            EvidenceItem(**{k: v for k, v in c.items() if k != "topics"}, sub_claim_id="claim",
                         stance=Stance.NEUTRAL, company="Company X", produced_by=self.name)
            for c in ranked[:top_k]
        ]
