# PLACEHOLDER: owned by Person 3 (Amelia). Replace with real implementation. Do not change the signature.
"""Keyword-based decomposer stub. Deterministic; produces the 4 demo claims for the demo thesis."""

from __future__ import annotations

import re
from datetime import date

from finsight.agents.demo_corpus import company_name, detect_ticker
from finsight.contracts.schemas import Claim, ClaimType

# (claim type, trigger regex, claim template, is_quantitative), in output order.
_RULES: list[tuple[ClaimType, str, str, bool]] = [
    (ClaimType.PERSONALIZED_ADVICE, r"\bshould i\b|\bmy (portfolio|savings|retirement)\b|how much should i",
     "The user asks what they personally should do with {name}.", False),
    (ClaimType.HISTORICAL_GROWTH, r"\brevenue\b|\bgrowth\b|\bgrew\b",
     "{name}'s recent revenue growth is strong.", True),
    (ClaimType.PROFITABILITY_CASH_LEVERAGE, r"\bmargins?\b|\bcash flow\b|\bdebt\b|\bprofit",
     "{name}'s margins are expanding.", True),
    (ClaimType.DEMAND_ATTRIBUTION, r"\bAI\b|\bdemand\b",
     "{name}'s revenue growth is driven by AI infrastructure demand.", False),
    (ClaimType.COMPARATIVE, r"biggest|\bbest\b|winner|better than|\bvs\.?\b",
     "{name} is the biggest winner from AI infrastructure demand.", False),
    (ClaimType.CONTINUATION, r"getting started|will continue|keep growing|long runway",
     "{name}'s growth is just getting started.", False),
    (ClaimType.BUSINESS_RISK, r"\brisks?\b|\bthreat|competition",
     "{name} faces no material business risk to this growth.", False),
]


class StubDecomposer:
    name = "decomposition_stub"

    def decompose(self, thesis: str, cutoff_date: date) -> list[Claim]:
        ticker = detect_ticker(thesis)
        name = company_name(ticker)
        claims: list[Claim] = []
        for claim_type, pattern, template, quantitative in _RULES:
            if re.search(pattern, thesis, flags=re.IGNORECASE):
                claims.append(Claim(
                    claim_id=f"c{len(claims) + 1}", text=template.format(name=name),
                    claim_type=claim_type, company_ticker=ticker, is_quantitative=quantitative,
                ))
        if not claims:
            claims.append(Claim(claim_id="c1", text=thesis.strip()[:300], claim_type=ClaimType.OTHER,
                                company_ticker=ticker))
        return claims
