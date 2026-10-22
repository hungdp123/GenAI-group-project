# PLACEHOLDER: owned by Person 2 (Stella). Replace with real implementation. Do not change the signature.
"""Narrative agent stub: returns supporting AND challenging chunks from the demo corpus by sub-claim topic."""

from __future__ import annotations

from datetime import date

from finsight.agents.demo_corpus import NARRATIVE_CHUNKS, company_name
from finsight.contracts.schemas import SubClaim, SubClaimType, EvidenceItem
from finsight.llm.usage import record_tool_call

_TOPIC_BY_TYPE = {
    SubClaimType.DEMAND_ATTRIBUTION: "demand",
    SubClaimType.CONTINUATION: "continuation",
    SubClaimType.BUSINESS_RISK: "risk",
    SubClaimType.COMPARATIVE: "comparative",
    SubClaimType.HISTORICAL_GROWTH: "growth",
}


class StubNarrativeAgent:
    name = "narrative_stub"

    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]:
        if sub_claim.company_ticker != "CMPX":
            return []
        record_tool_call()  # vector search
        topic = _TOPIC_BY_TYPE.get(sub_claim.sub_claim_type)
        items = []
        for chunk in NARRATIVE_CHUNKS:
            if chunk["publication_date"] > cutoff_date:
                continue
            if topic is not None and topic not in chunk["topics"]:
                continue
            stance = chunk["topics"][topic] if topic else next(iter(chunk["topics"].values()))
            fields = {k: v for k, v in chunk.items() if k != "topics"}
            items.append(EvidenceItem(**fields, sub_claim_id=sub_claim.sub_claim_id, stance=stance,
                                      company=company_name(sub_claim.company_ticker), produced_by=self.name))
        return items
