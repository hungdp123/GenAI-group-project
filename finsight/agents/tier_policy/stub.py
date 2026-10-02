# PLACEHOLDER: owned by Person 4 (Irine). Replace with real implementation. Do not change the signature.
"""Tier policy stub: naive knowledge-layer caps."""

from __future__ import annotations

from finsight.contracts.schemas import SubClaim, EvidenceItem, KnowledgeRule, Stance, Verdict


class StubTierPolicy:
    name = "tier_policy_stub"

    def apply(self, sub_claim: SubClaim, proposed: Verdict, evidence: list[EvidenceItem],
              rule: KnowledgeRule) -> tuple[Verdict, str | None]:
        cap = rule.highest_verdict
        if cap is not None and proposed.support_rank > cap.support_rank:
            return cap, f"Knowledge layer caps {rule.sub_claim_type.value} at {cap.value}"
        eligible = [e for e in evidence
                    if e.source_tier in rule.tier_requirement and e.stance is Stance.SUPPORTING]
        if proposed.is_positive and not eligible:
            return Verdict.INSUFFICIENT_EVIDENCE, "No supporting evidence from a required tier"
        return proposed, None
