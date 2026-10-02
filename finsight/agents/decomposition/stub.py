# PLACEHOLDER: owned by Person 3 (Amelia). Replace with real implementation. Do not change the signature.
"""Keyword-based decomposer stub. Deterministic; produces the 4 demo sub-claims for the demo claim."""

from __future__ import annotations

import re
from datetime import date

from finsight.agents.demo_corpus import company_name, detect_ticker
from finsight.contracts.schemas import SubClaim, SubClaimType

# (sub-claim type, trigger regex, sub-claim template, is_quantitative), in output order.
_RULES: list[tuple[SubClaimType, str, str, bool]] = [
    (SubClaimType.PERSONALIZED_ADVICE, r"\bshould i\b|\bmy (portfolio|savings|retirement)\b|how much should i",
     "The user asks what they personally should do with {name}.", False),
    (SubClaimType.HISTORICAL_GROWTH, r"\brevenue\b|\bgrowth\b|\bgrew\b",
     "{name}'s recent revenue growth is strong.", True),
    (SubClaimType.PROFITABILITY_CASH_LEVERAGE, r"\bmargins?\b|\bcash flow\b|\bdebt\b|\bprofit",
     "{name}'s margins are expanding.", True),
    (SubClaimType.DEMAND_ATTRIBUTION, r"\bAI\b|\bdemand\b",
     "{name}'s revenue growth is driven by AI infrastructure demand.", False),
    (SubClaimType.COMPARATIVE, r"biggest|\bbest\b|winner|better than|\bvs\.?\b",
     "{name} is the biggest winner from AI infrastructure demand.", False),
    (SubClaimType.CONTINUATION, r"getting started|will continue|keep growing|long runway",
     "{name}'s growth is just getting started.", False),
    (SubClaimType.BUSINESS_RISK, r"\brisks?\b|\bthreat|competition",
     "{name} faces no material business risk to this growth.", False),
]


class StubDecomposer:
    name = "decomposition_stub"

    def decompose(self, claim: str, cutoff_date: date) -> list[SubClaim]:
        ticker = detect_ticker(claim)
        name = company_name(ticker)
        sub_claims: list[SubClaim] = []
        for sub_claim_type, pattern, template, quantitative in _RULES:
            if re.search(pattern, claim, flags=re.IGNORECASE):
                sub_claims.append(SubClaim(
                    sub_claim_id=f"sc{len(sub_claims) + 1}", text=template.format(name=name),
                    sub_claim_type=sub_claim_type, company_ticker=ticker, is_quantitative=quantitative,
                ))
        if not sub_claims:
            sub_claims.append(SubClaim(sub_claim_id="sc1", text=claim.strip()[:300], sub_claim_type=SubClaimType.OTHER,
                                company_ticker=ticker))
        return sub_claims
