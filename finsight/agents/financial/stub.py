# PLACEHOLDER: owned by Person 1 (Venes). Replace with real implementation. Do not change the signature.
"""Financial agent stub: computes growth / margin deterministically from the XBRL fixture."""

from __future__ import annotations

from datetime import date

from finsight.agents.demo_corpus import company_name, load_xbrl_facts
from finsight.contracts.schemas import SubClaim, SubClaimType, EvidenceItem, SourceTier, Stance
from finsight.llm.usage import record_tool_call


def _latest_pair(facts: list[dict], metric: str) -> tuple[dict, dict] | None:
    """(current, prior-year) facts for `metric`, by period_end."""
    rows = sorted((f for f in facts if f["metric"] == metric), key=lambda f: f["period_end"])
    return (rows[-1], rows[-2]) if len(rows) >= 2 else None


class StubFinancialAgent:
    name = "financial_stub"

    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]:
        if not sub_claim.company_ticker:
            return []
        record_tool_call()  # xbrl lookup
        facts = load_xbrl_facts(sub_claim.company_ticker, cutoff_date)
        if sub_claim.sub_claim_type is SubClaimType.HISTORICAL_GROWTH:
            return self._revenue_growth(sub_claim, facts)
        if sub_claim.sub_claim_type in (SubClaimType.PROFITABILITY, SubClaimType.PROFITABILITY_CASH_LEVERAGE):
            return self._gross_margin(sub_claim, facts)
        return []

    def _evidence(self, sub_claim: SubClaim, fact: dict, **fields) -> EvidenceItem:
        ticker = sub_claim.company_ticker or ""
        return EvidenceItem(
            sub_claim_id=sub_claim.sub_claim_id, source_tier=SourceTier.TIER_1, doc_type=fact["form"],
            company=company_name(ticker), period=fact["period"],
            publication_date=date.fromisoformat(fact["filed"]),
            url=f"https://example.com/sec/{ticker.lower()}/{fact['accession']}",
            produced_by=self.name, **fields,
        )

    def _revenue_growth(self, sub_claim: SubClaim, facts: list[dict]) -> list[EvidenceItem]:
        pair = _latest_pair(facts, "Revenues")
        if pair is None:
            return []
        cur, prev = pair
        growth = round((cur["value"] - prev["value"]) / prev["value"], 4)
        record_tool_call()  # growth calculator
        return [self._evidence(
            sub_claim, cur, evidence_id=f"XBRL-{cur['accession']}-REV-YOY",
            stance=Stance.SUPPORTING if growth >= 0.10 else Stance.CHALLENGING,
            content=(f"Revenues were ${cur['value']:,.0f}m in {cur['period']} vs ${prev['value']:,.0f}m "
                     f"in {prev['period']} ({growth:+.0%} YoY)."),
            citation_span="XBRL us-gaap:Revenues", computed_value=growth, metric_name="revenue_growth_yoy",
        )]

    def _gross_margin(self, sub_claim: SubClaim, facts: list[dict]) -> list[EvidenceItem]:
        rev, gp = _latest_pair(facts, "Revenues"), _latest_pair(facts, "GrossProfit")
        if rev is None or gp is None:
            return []
        cur_m = round(gp[0]["value"] / rev[0]["value"], 4)
        prev_m = round(gp[1]["value"] / rev[1]["value"], 4)
        record_tool_call()  # margin calculator
        sub_claims_expansion = any(w in sub_claim.text.lower() for w in ("expand", "improv", "rising", "higher"))
        stance = Stance.SUPPORTING if (cur_m > prev_m) == sub_claims_expansion else Stance.CHALLENGING
        return [self._evidence(
            sub_claim, rev[0], evidence_id=f"XBRL-{rev[0]['accession']}-GM", stance=stance,
            content=f"Gross margin was {cur_m:.1%} in {rev[0]['period']} vs {prev_m:.1%} in {rev[1]['period']}.",
            citation_span="XBRL us-gaap:GrossProfit / us-gaap:Revenues",
            computed_value=cur_m, metric_name="gross_margin",
        )]
