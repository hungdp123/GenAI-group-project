"""Group an expanded phrase into one report item. The section 6 rollup decides the parent label.

A company-wide phrase such as "financially strong" becomes four child claims. The report and the
overall label use the parent, not the four children as four claims.
"""

from __future__ import annotations

from dataclasses import dataclass

from finsight.agents.orchestrator.aggregation import overall_verdict
from finsight.contracts.schemas import SubClaimResult, Verdict

# rule name -> True when "favorable" in section 6 is read the other way (weak, poor, in trouble).
_COMPANY_WIDE: dict[str, bool] = {
    "financially_strong": False,
    "strong_fundamentals": False,
    "healthy_company": False,
    "good_company": False,
    "financially_weak": True,
    "weak_fundamentals": True,
    "poor_company": True,
    "in_trouble": True,
}

_MEASURES = ("profitability", "cash_generation", "liquidity", "leverage")

# Growth words. Section 11: the size of the change is not tested.
SIZE_NOT_TESTED = frozenset({"strong_growth", "explosive_growth"})


@dataclass(frozen=True)
class ReportUnit:
    """One thing the report counts. A parent unit holds its children."""

    verdict: Verdict
    results: tuple[SubClaimResult, ...]
    parent_span: str | None = None
    expansion_rule: str | None = None
    quote_time: str | None = None

    @property
    def is_parent(self) -> bool:
        if not self.parent_span:
            return False
        if len(self.results) > 1:
            return True
        return self.expansion_rule in _COMPANY_WIDE


def report_units(results: list[SubClaimResult]) -> list[ReportUnit]:
    """Collapse children that share a parent phrase. Other claims stay one each, in order."""
    groups: dict[tuple[str, str], list[SubClaimResult]] = {}
    order: list[tuple[str, str] | int] = []
    singles: list[SubClaimResult] = []
    for result in results:
        claim = result.sub_claim
        if claim.parent_span and claim.expansion_rule:
            key = (claim.parent_span, claim.expansion_rule)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(result)
        else:
            order.append(len(singles))
            singles.append(result)
    units: list[ReportUnit] = []
    for item in order:
        if isinstance(item, int):
            result = singles[item]
            units.append(ReportUnit(verdict=result.verdict, results=(result,)))
            continue
        children = tuple(groups[item])
        rule = item[1]
        units.append(ReportUnit(
            verdict=_parent_verdict(children, rule),
            results=children,
            parent_span=item[0],
            expansion_rule=rule,
            quote_time=next((c.sub_claim.quote_time for c in children if c.sub_claim.quote_time), None),
        ))
    return units


def _parent_verdict(children: tuple[SubClaimResult, ...], rule: str) -> Verdict:
    if rule not in _COMPANY_WIDE:
        return overall_verdict([c.verdict for c in children])
    return _rollup(_labels(children, _COMPANY_WIDE[rule]))


def _labels(children: tuple[SubClaimResult, ...], weak: bool) -> list[str]:
    """Four slots. A missing measure is a missing test, not a failure."""
    by_measure = {c.sub_claim.measure: c for c in children if c.sub_claim.measure in _MEASURES}
    if by_measure:
        chosen = [by_measure.get(measure) for measure in _MEASURES]
    else:
        chosen = list(children[:4]) + [None] * (4 - len(children))
    return [_slot(child.verdict if child else None, weak) for child in chosen]


def _slot(verdict: Verdict | None, weak: bool) -> str:
    if verdict is None or verdict is Verdict.INSUFFICIENT_EVIDENCE:
        return "missing"
    if verdict is Verdict.PARTIALLY_SUPPORTED:
        return "partial"
    met = verdict is Verdict.CONTRADICTED if weak else verdict is Verdict.SUPPORTED
    return "met" if met else "unmet"


def rollup_counts(children: tuple[SubClaimResult, ...], rule: str) -> tuple[int, int]:
    """How many of the four tests met the phrase, and how many were missing."""
    labels = _labels(children, _COMPANY_WIDE.get(rule, False))
    return labels.count("met"), labels.count("missing")


def _rollup(labels: list[str]) -> Verdict:
    """Section 6 parent rollup. `labels` has one entry per test."""
    met = labels.count("met")
    missing = labels.count("missing")
    unmet = labels.count("unmet")
    partial = labels.count("partial")
    if missing >= 2:
        return Verdict.INSUFFICIENT_EVIDENCE
    if missing == 0 and partial == 0 and met == 4:
        return Verdict.SUPPORTED
    if missing == 0 and unmet == 0 and partial and met < 4:
        return Verdict.PARTIALLY_SUPPORTED if met else Verdict.INSUFFICIENT_EVIDENCE
    if missing == 0 and met in (2, 3):
        return Verdict.PARTIALLY_SUPPORTED
    if missing == 0 and met in (0, 1):
        return Verdict.CONTRADICTED
    if missing == 1:
        others = [label for label in labels if label != "missing"]
        if others.count("unmet") == len(others):
            return Verdict.CONTRADICTED
        if others.count("met") >= 2:
            return Verdict.PARTIALLY_SUPPORTED
        return Verdict.INSUFFICIENT_EVIDENCE
    return Verdict.INSUFFICIENT_EVIDENCE
