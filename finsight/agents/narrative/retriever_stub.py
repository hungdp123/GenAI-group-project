# PLACEHOLDER: owned by Person 2 (Stella). Replace with real implementation. Do not change the signature.
"""Basic RAG retriever stub: bag-of-words overlap ranking over the demo corpus."""

from __future__ import annotations

import re
from datetime import date

from finsight.agents.demo_corpus import NARRATIVE_CHUNKS
from finsight.contracts.schemas import EvidenceItem, Stance
from finsight.llm.usage import record_tool_call


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class StubRetriever:
    name = "retriever_stub"

    def retrieve(self, query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]:
        record_tool_call()
        q = _tokens(query)
        candidates = [c for c in NARRATIVE_CHUNKS if c["publication_date"] <= cutoff_date]
        ranked = sorted(candidates, key=lambda c: (-len(q & _tokens(c["content"])), c["evidence_id"]))
        return [
            EvidenceItem(**{k: v for k, v in c.items() if k != "topics"}, claim_id="thesis",
                         stance=Stance.NEUTRAL, company="Company X", produced_by=self.name)
            for c in ranked[:top_k]
        ]
