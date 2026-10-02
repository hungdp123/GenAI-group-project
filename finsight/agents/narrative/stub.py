# PLACEHOLDER: owned by Person 2 (Stella). Replace with real implementation. Do not change the signature.
"""Narrative agent stub: returns supporting AND challenging chunks from the demo corpus by claim topic."""

from __future__ import annotations

from datetime import date

from finsight.agents.demo_corpus import NARRATIVE_CHUNKS, company_name
from finsight.contracts.schemas import Claim, ClaimType, EvidenceItem
from finsight.llm.usage import record_tool_call

_TOPIC_BY_TYPE = {
    ClaimType.DEMAND_ATTRIBUTION: "demand",
    ClaimType.CONTINUATION: "continuation",
    ClaimType.BUSINESS_RISK: "risk",
    ClaimType.COMPARATIVE: "comparative",
    ClaimType.HISTORICAL_GROWTH: "growth",
}


class StubNarrativeAgent:
    name = "narrative_stub"

    def gather(self, claim: Claim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]:
        if claim.company_ticker != "CMPX":
            return []
        record_tool_call()  # vector search
        topic = _TOPIC_BY_TYPE.get(claim.claim_type)
        items = []
        for chunk in NARRATIVE_CHUNKS:
            if chunk["publication_date"] > cutoff_date:
                continue
            if topic is not None and topic not in chunk["topics"]:
                continue
            stance = chunk["topics"][topic] if topic else next(iter(chunk["topics"].values()))
            fields = {k: v for k, v in chunk.items() if k != "topics"}
            items.append(EvidenceItem(**fields, claim_id=claim.claim_id, stance=stance,
                                      company=company_name(claim.company_ticker), produced_by=self.name))
        return items
