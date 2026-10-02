"""Auditor evaluation (Person 3, Amelia): error-injection test.

Start from CORRECT evidence for a few sub-claims, then generate broken copies, one mistake at a time,
of the kinds the specialist agents can make. A good auditor:
- catches broken evidence: does NOT endorse it (endorse = passed and SUPPORTED / PARTIALLY_SUPPORTED);
- does not raise false alarms: DOES endorse the correct originals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from finsight.config import AgentRegistry
from finsight.contracts.schemas import (
    AuditResult,
    EvidenceItem,
    SourceTier,
    Stance,
    SubClaim,
    SubClaimType,
)
from finsight.eval.agent_eval.common import AgentReport, CaseResult, implementation_name, ratio


@dataclass
class Base:
    base_id: str
    kind: str  # "financial" | "narrative"
    sub_claim: SubClaim
    evidence: list[EvidenceItem]


def _item(**kw) -> EvidenceItem:
    base = dict(sub_claim_id="sc1", stance=Stance.SUPPORTING, company="Company X", period="Q2 FY2025",
                publication_date=date(2025, 7, 25), source_tier=SourceTier.TIER_1, doc_type="10-Q",
                url="https://example.com/sec/cmpx/10-Q/2025-Q2")
    return EvidenceItem(**(base | kw))


BASES = [
    Base("growth", "financial",
         SubClaim(sub_claim_id="sc1", text="Company X's revenue grew strongly year over year.",
                  sub_claim_type=SubClaimType.HISTORICAL_GROWTH, company_ticker="CMPX", is_quantitative=True),
         [_item(evidence_id="XBRL-REV-YOY", produced_by="financial", citation_span="XBRL us-gaap:Revenues",
                content="Revenues were $4,560m in Q2 FY2025 vs $4,000m in Q2 FY2024 (+14% YoY).",
                computed_value=0.14, metric_name="revenue_growth_yoy")]),
    Base("margin", "financial",
         SubClaim(sub_claim_id="sc1", text="Company X's gross margin is shrinking.",
                  sub_claim_type=SubClaimType.PROFITABILITY_CASH_LEVERAGE, company_ticker="CMPX",
                  is_quantitative=True),
         [_item(evidence_id="XBRL-GM", produced_by="financial",
                citation_span="XBRL us-gaap:GrossProfit / us-gaap:Revenues",
                content="Gross margin was 52.1% in Q2 FY2025 vs 54.3% in Q2 FY2024.",
                computed_value=0.521, metric_name="gross_margin")]),
    Base("demand", "narrative",
         SubClaim(sub_claim_id="sc1", text="Company X's revenue growth is driven by AI infrastructure demand.",
                  sub_claim_type=SubClaimType.DEMAND_ATTRIBUTION, company_ticker="CMPX"),
         [_item(evidence_id="CMPX-10Q-MDA-DC", produced_by="narrative",
                citation_span="Item 2. MD&A - Segment Results - Data Center",
                content="Data Center segment revenue increased 31% year over year, primarily driven by "
                        "customer demand for AI infrastructure.")]),
]


def _edit(base: Base, **changes) -> tuple[SubClaim, list[EvidenceItem]]:
    return base.sub_claim, [base.evidence[0].model_copy(update=changes)]


def _scaled(base: Base, factor: float) -> tuple[SubClaim, list[EvidenceItem]]:
    return _edit(base, computed_value=round(base.evidence[0].computed_value * factor, 4))


# name -> (applies to kind, how to break it)
MUTATIONS: dict[str, tuple[str, Callable[[Base], tuple[SubClaim, list[EvidenceItem]]]]] = {
    "units / scale (x100)": ("financial", lambda b: _scaled(b, 100)),
    "calculation error": ("financial", lambda b: _scaled(b, 0.85)),
    "stated number contradicts": ("financial", lambda b: (
        b.sub_claim.model_copy(update={"text": b.sub_claim.text.rstrip(".") + " by 30%."}), b.evidence)),
    "wrong metric": ("financial", lambda b: _edit(b, metric_name="data_center_segment_revenue_growth",
                                                   computed_value=0.31)),
    "wrong comparison period": ("financial", lambda b: _edit(
        b, metric_name=(b.evidence[0].metric_name or "") + "_qoq",
        content=b.evidence[0].content.replace("Q2 FY2024", "Q1 FY2025"))),
    "date = period end, not filing date": ("any", lambda b: _edit(b, publication_date=date(2025, 6, 30))),
    "wrong company": ("any", lambda b: _edit(b, company="Company Y")),
    "missing provenance": ("any", lambda b: _edit(b, url=None, citation_span=None)),
    "conflicting evidence": ("financial", lambda b: (b.sub_claim, b.evidence + [_item(
        evidence_id="ER-CONFLICT", produced_by="narrative", doc_type="earnings_release",
        source_tier=SourceTier.TIER_2, citation_span="Highlights",
        content="Total revenue fell 3% and gross margin improved year over year.")])),
    "citation doesn't support": ("narrative", lambda b: _edit(
        b, content="Company X presented its AI roadmap at a developer conference.")),
    "stance mislabelled": ("narrative", lambda b: _edit(
        b, content="Client segment revenue grew 6% and Embedded grew 4%, reflecting PC refresh and "
                   "industrial recovery rather than AI infrastructure.")),
    "hedged language as proof": ("narrative", lambda b: _edit(
        b, content="Management said AI demand may or may not explain recent growth, depending on budgets.")),
    "wrong source tier": ("narrative", lambda b: _edit(
        b, doc_type="news", source_tier=SourceTier.TIER_2, url="https://example.com/news/2025-07-28/company-x")),
}


def endorses(result: AuditResult) -> bool:
    return result.passed and result.verdict.is_positive


def evaluate(registry: AgentRegistry) -> AgentReport:
    agent = registry.auditor
    report = AgentReport("auditor", "Person 3 (Amelia)", implementation_name(agent))
    errors = 0

    def run(case_id: str, sub_claim: SubClaim, evidence: list[EvidenceItem]) -> AuditResult | None:
        nonlocal errors
        try:
            fresh = type(agent)()  # fresh instance so no state (e.g. retry counters) leaks between cases
        except TypeError:  # constructor needs arguments: reuse the registered instance
            fresh = agent
        try:
            return fresh.audit(sub_claim, evidence)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            report.cases.append(CaseResult(case_id, False, f"raised {type(exc).__name__}: {exc}"))
            return None

    clean_ok = clean_total = 0
    for base in BASES:
        result = run(f"clean/{base.base_id}", base.sub_claim, base.evidence)
        if result is None:
            continue
        clean_total += 1
        clean_ok += endorses(result)
        report.cases.append(CaseResult(f"clean/{base.base_id}", endorses(result),
                                       "correct evidence endorsed" if endorses(result)
                                       else f"FALSE ALARM: {result.verdict.value} {result.failure_reasons}"))

    per_type: dict[str, tuple[int, int]] = {}
    for name, (kind, mutate) in MUTATIONS.items():
        for base in BASES:
            if kind not in ("any", base.kind):
                continue
            sub_claim, evidence = mutate(base)
            result = run(f"{name}/{base.base_id}", sub_claim, evidence)
            if result is None:
                continue
            caught = not endorses(result)
            got, total = per_type.get(name, (0, 0))
            per_type[name] = (got + caught, total + 1)
            reason = "; ".join(result.failure_reasons) or result.verdict.value
            report.cases.append(CaseResult(f"{name}/{base.base_id}", caught,
                                           f"caught ({reason})" if caught else "MISSED: endorsed broken evidence"))

    caught_all = sum(g for g, _ in per_type.values())
    total_all = sum(t for _, t in per_type.values())
    report.metrics = {
        "broken cases": total_all,
        "catch rate (broken evidence rejected)": ratio(caught_all, total_all),
        "false-alarm rate (correct evidence rejected)": ratio(clean_total - clean_ok, clean_total),
        **{f"  catches: {name}": ratio(g, t) for name, (g, t) in per_type.items()},
        "errors": errors,
    }
    report.notes.append("Endorse = passed with SUPPORTED or PARTIALLY_SUPPORTED. Both rates matter: misses let "
                        "wrong evidence through; false alarms waste the retry and hide correct sub-claims.")
    return report
