"""Which agent a gold fact belongs to, decided from the gold label alone (never from a system's run).

The same rule-based router FinSight uses: the fact's type (from its words, else the stored catalog type) picks
the knowledge-layer rule, and the rule names the collectors. Written into the gold file by
scripts/build_dev_gold.py as `route`, so a labeler can read and correct it.
"""

from __future__ import annotations

from finsight.contracts.schemas import SubClaim, SubClaimType


def fact_route(fact_text: str, stored_type: str | None = None, ticker: str | None = None) -> str:
    """"financial", "narrative", "financial+narrative" or "none"."""
    from finsight.agents.claim_extraction.extract import _quote_claim
    from finsight.knowledge.loader import default_knowledge_layer

    claim, _reason = _quote_claim(fact_text, ticker, 1)
    sub_type = claim.sub_claim_type
    if sub_type is SubClaimType.OTHER and stored_type:
        try:
            sub_type = SubClaimType(stored_type)
        except ValueError:
            pass
    rule = default_knowledge_layer().rule_for(sub_type)
    collectors = [] if rule.refuse else sorted(rule.collectors)
    return "+".join(collectors) or "none"


__all__ = ["fact_route", "SubClaim"]
