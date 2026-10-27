"""Routing: which evidence agent checks each sub-claim. Owner: Person 5 (Hung).

The orchestrator receives the sub-claims from claim extraction and decides, one sub-claim at a
time, whether the Financial Analysis Agent, the Narrative Evidence Agent, or both should gather
evidence. Each agent is then called as `gather(sub_claim, cutoff_date, feedback)`.

    decisions = route_sub_claims(sub_claims, kb)
    for d in decisions:
        print(d.sub_claim_id, d.collectors, d.reason)

# DECISION: routing is deterministic and never calls an LLM. Each sub-claim goes to the collectors the
# knowledge layer lists for its type ("Who looks it up" in docs/knowledge_layer.md section 5), so the
# same sub-claim type always reaches the same agent. Verdict caps and tier requirements also come
# from the knowledge layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from finsight.contracts.schemas import KnowledgeRule, SubClaim
from finsight.knowledge.loader import KnowledgeLayer


@dataclass(frozen=True)
class RouteDecision:
    """The orchestrator's routing for one sub-claim, plus the arguments each agent's gather() receives."""

    sub_claim: SubClaim
    collectors: list[str]
    reason: str
    cutoff_date: date | None = None
    feedback: str | None = None

    @property
    def sub_claim_id(self) -> str:
        return self.sub_claim.sub_claim_id

    @property
    def target(self) -> str:
        """One of: financial, narrative, both, none."""
        if len(self.collectors) == 2:
            return "both"
        return self.collectors[0] if self.collectors else "none"

    def agent_input(self) -> dict:
        """What the Financial and Narrative agents need (sub_claim fields + gather() arguments)."""
        return {
            "sub_claim_id": self.sub_claim.sub_claim_id,
            "company_ticker": self.sub_claim.company_ticker,
            "text": self.sub_claim.text,
            "cutoff_date": self.cutoff_date,
            "feedback": self.feedback,
        }


def route_sub_claim(sub_claim: SubClaim, rule: KnowledgeRule, *, cutoff_date: date | None = None,
                    feedback: str | None = None) -> RouteDecision:
    """Decide financial / narrative / both / none for one sub-claim from its knowledge-layer rule."""
    why = "refused type" if rule.refuse else "knowledge-layer rule"
    reason = f"{why} for {rule.sub_claim_type.value}"
    collectors = [] if rule.refuse else list(rule.collectors)
    return RouteDecision(sub_claim, collectors, reason, cutoff_date, feedback)


def route_sub_claims(sub_claims: list[SubClaim], kb: KnowledgeLayer, *, cutoff_date: date | None = None,
                     feedback: dict[str, str] | None = None) -> list[RouteDecision]:
    """Route every sub-claim from claim extraction, in order."""
    feedback = feedback or {}
    return [
        route_sub_claim(c, kb.rule_for(c.sub_claim_type), cutoff_date=cutoff_date,
                        feedback=feedback.get(c.sub_claim_id))
        for c in sub_claims
    ]
