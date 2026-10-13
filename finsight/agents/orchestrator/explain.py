"""Plain-language wording for end users. Owner: Person 5 (Hung).

Everything here is built BY RULE from the verdicts and evidence the pipeline already produced:
no LLM, no new numbers (numbers only appear inside quoted evidence text), no advice language.
The UI (Person 4) can show `FinalAssessment.summary` and `SubClaimResult.explanation` directly,
and reuse `verdict_label` / `source_label` for its own layout.
"""

from __future__ import annotations

from datetime import date

from finsight.agents.orchestrator.expansion import SIZE_NOT_TESTED, ReportUnit, report_units, rollup_counts
from finsight.contracts.schemas import (
    AssessmentStatus,
    EvidenceItem,
    SourceTier,
    SubClaimResult,
    SubClaimType,
    Verdict,
)

VERDICT_LABELS = {
    Verdict.SUPPORTED: "Supported",
    Verdict.PARTIALLY_SUPPORTED: "Partly supported",
    Verdict.CONTRADICTED: "Contradicted",
    Verdict.INSUFFICIENT_EVIDENCE: "Can't be confirmed",
}

_DOC_NAMES = {
    "10-q": "Quarterly report (10-Q) filed with the SEC",
    "10-q/a": "Amended quarterly report (10-Q/A) filed with the SEC",
    "10-k": "Annual report (10-K) filed with the SEC",
    "10-k/a": "Amended annual report (10-K/A) filed with the SEC",
    "8-k": "Current report (8-K) filed with the SEC",
    "earnings_release": "Company earnings release",
    "ir_remarks": "Management remarks on the earnings call",
    "news": "News article",
}

_TIER_NOTES = {
    SourceTier.TIER_1: "Tier 1: official filing, most reliable",
    SourceTier.TIER_2: "Tier 2: company's own announcement, reliable",
    SourceTier.TIER_3: "Tier 3: news, can't confirm a claim on its own",
}

# The four reasons a Partly supported label can have. The report names one of them.
_PARTIAL_REASON = {
    "forward-looking": "Partly supported because this is about the future.",
    "partial match": "Partly supported because the filing agrees only in part.",
    "more than one cause": "Partly supported because more than one cause is named.",
    "size not tested": "Partly supported because the size of the change was not tested.",
}


def partial_reason(result: SubClaimResult) -> str | None:
    """Why a Partly supported label was used. One of the four short reasons, or None."""
    if result.verdict is not Verdict.PARTIALLY_SUPPORTED:
        return None
    if result.sub_claim.expansion_rule in SIZE_NOT_TESTED:
        return "size not tested"
    if result.sub_claim.sub_claim_type is SubClaimType.CONTINUATION:
        return "forward-looking"
    if result.sub_claim.sub_claim_type is SubClaimType.DEMAND_ATTRIBUTION:
        return "more than one cause"
    return "partial match"


def unit_partial_reason(unit: ReportUnit) -> str | None:
    """The reason shown next to a Partly supported line. A mix of reasons is a partial match."""
    if unit.verdict is not Verdict.PARTIALLY_SUPPORTED:
        return None
    reasons = {reason for result in unit.results if (reason := partial_reason(result))}
    if len(reasons) == 1:
        return next(iter(reasons))
    return "partial match"


def verdict_label(verdict: Verdict | None) -> str:
    return VERDICT_LABELS.get(verdict, "Not assessed") if verdict else "Not assessed"


def friendly_date(d: date) -> str:
    return f"{d.day} {d:%b %Y}"


def source_label(e: EvidenceItem) -> str:
    """e.g. 'Quarterly report (10-Q) filed with the SEC, 25 Jul 2025 (Tier 1: official filing, most reliable)'."""
    doc = _DOC_NAMES.get(e.doc_type.lower(), e.doc_type)
    return f"{doc}, {friendly_date(e.publication_date)} ({_TIER_NOTES[e.source_tier]})"


def explain(result: SubClaimResult, as_of: date | None, not_assessed_reason: str | None) -> str:
    """One or two short plain sentences: why this sub-claim got its verdict. The evidence itself is shown
    separately (FinalAssessment / rendered details), so this does not repeat it."""
    when = f" before {friendly_date(as_of)}" if as_of else ""
    sub_type = result.sub_claim.sub_claim_type
    reliable_for = [e for e in result.supporting if e.source_tier is not SourceTier.TIER_3]
    reliable_against = [e for e in result.challenging if e.source_tier is not SourceTier.TIER_3]
    news_only = any(e.source_tier is SourceTier.TIER_3 for e in result.supporting)
    news_note = (" News articles repeat this point, but news alone can't confirm it."
                 if news_only and result.verdict is not Verdict.SUPPORTED else "")

    def finish(sentence: str) -> str:
        note = _size_note(result)
        if result.sub_claim.measure == "segment":
            note = "This is about that segment, not the whole company. " + (note or "")
        return note + sentence if note else sentence

    if result.verdict is Verdict.INSUFFICIENT_EVIDENCE:
        if not_assessed_reason and (
            not_assessed_reason.startswith("The number or ticker")
            or not_assessed_reason.startswith("A segment figure")
        ):
            return finish(not_assessed_reason)
        if sub_type is SubClaimType.COMPARATIVE:
            return finish("This compares the company with its competitors, but FinSight only has this company's own "
                          "documents, so the comparison can't be checked." + news_note)
        if not_assessed_reason and "failed" in not_assessed_reason and "audit" not in not_assessed_reason:
            return finish("Part of the system failed while checking this point, so it couldn't be confirmed.")
        return finish(f"No official filing or company report{when} confirms this." + news_note)
    if result.verdict is Verdict.SUPPORTED:
        tier1 = any(e.source_tier is SourceTier.TIER_1 for e in reliable_for)
        source = "official filings" if tier1 else "the company's own reports"
        return finish(f"Backed up by {source} (see the evidence below).")
    if result.verdict is Verdict.CONTRADICTED:
        return finish("The evidence below points the other way.")
    # PARTIALLY_SUPPORTED. The short reason is the one the label names.
    reason = partial_reason(result)
    lead = "" if reason == "size not tested" else _PARTIAL_REASON[reason or "partial match"] + " "
    if reliable_for and reliable_against:
        sentence = "Some evidence supports this and some points the other way (see below)."
    elif reliable_for:
        sentence = "The evidence below points this way."
    else:
        sentence = "Only weak evidence points this way."
    return finish(lead + sentence + news_note)


def summary(results: list[SubClaimResult], overall: Verdict | None) -> str:
    """Overall sentence + how the points break down. An expanded phrase counts as one point."""
    units = report_units(results)
    if not units:
        return "FinSight couldn't find any specific point in this claim to check."
    counts: dict[str, int] = {}
    for unit in units:
        label = verdict_label(unit.verdict).lower()
        counts[label] = counts.get(label, 0) + 1
    order = [verdict_label(v).lower() for v in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED,
                                                 Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE)]
    breakdown = ", ".join(f"{counts[k]} {k}" for k in order if k in counts)
    n = len(units)
    headline = _OVERALL_SENTENCES.get(overall, "Overall, FinSight couldn't reach a verdict on this claim.")
    if n == 1:
        return f"{headline} FinSight checked it against company filings and reports."
    return (f"{headline} FinSight split it into {n} points and checked each against company filings "
            f"and reports: {breakdown}.")


_OVERALL_SENTENCES = {
    Verdict.SUPPORTED: "Overall, this claim is supported.",
    Verdict.PARTIALLY_SUPPORTED: "Overall, this claim is partly supported.",
    Verdict.CONTRADICTED: "Overall, the evidence contradicts this claim.",
    Verdict.INSUFFICIENT_EVIDENCE: "Overall, this claim can't be confirmed from the available evidence.",
}


_PLAIN_TEST = {
    "profitability": "Operating margin was above zero and did not fall.",
    "cash_generation": "Free cash flow was above zero and did not fall.",
    "liquidity": (
        "The current ratio was at least 1 and did not fall. "
        "This is a definition, not a ranking against other companies."
    ),
    "leverage": "Debt-to-equity did not rise. Only the direction is tested.",
}

_MEASURE_NAME = {
    "profitability": "Profitability",
    "cash_generation": "Cash generation",
    "liquidity": "Liquidity",
    "leverage": "Leverage",
}

_COMPANY_WIDE = frozenset({
    "financially_strong", "strong_fundamentals", "healthy_company", "good_company",
    "financially_weak", "weak_fundamentals", "poor_company", "in_trouble",
})

_LIMITS = ("The size of these figures was not compared with other companies. "
           "Leverage was tested on direction only.")


def _size_note(result: SubClaimResult) -> str:
    rule = result.sub_claim.expansion_rule
    if rule not in SIZE_NOT_TESTED:
        return ""
    return "The size of the change was not tested. "


def unit_heading(unit: ReportUnit) -> str:
    """The one line the summary uses. Children of an expanded phrase are not listed again."""
    if unit.parent_span and unit.expansion_rule in _COMPANY_WIDE:
        return f'The post said "{unit.parent_span}". FinSight tested this as four measures.'
    if unit.parent_span and len(unit.results) > 1:
        return f'The post said "{unit.parent_span}". FinSight tested the measures named in the post.'
    return unit.results[0].sub_claim.text


def expansion_block(unit: ReportUnit) -> str:
    """The sentences the report shows for one expanded phrase. Figures are quoted, not recalculated."""
    when = f" ({unit.quote_time})" if unit.quote_time else ""
    opening = f'The post said "{unit.parent_span}"{when}.'
    if unit.expansion_rule in _COMPANY_WIDE:
        lines = [opening + " FinSight tested this as four measures."]
        for result in unit.results:
            measure = result.sub_claim.measure or ""
            name = _MEASURE_NAME.get(measure, result.sub_claim.text)
            test = _PLAIN_TEST.get(measure, "")
            lines.append(f"{name}. {test} {_figure(result)} {verdict_label(result.verdict)}.")
        labels = _count_line(unit)
        lines.append(labels)
        lines.append(_LIMITS)
        return " ".join(lines)
    lines = [opening + " FinSight tested the measures named in the post."]
    for result in unit.results:
        name = result.sub_claim.measure or result.sub_claim.text
        lines.append(f"{name}. {_figure(result)} {verdict_label(result.verdict)}.")
    lines.append(f"Taken together, the phrase is {verdict_label(unit.verdict)}.")
    lines.append("Only the measures named in the post were tested.")
    return " ".join(lines)


def _figure(result: SubClaimResult) -> str:
    found = result.supporting or result.challenging
    if not found:
        return "No filing figure was found for this test."
    return f'The filing says: "{found[0].content.strip()}"'


def _count_line(unit: ReportUnit) -> str:
    met, missing = rollup_counts(unit.results, unit.expansion_rule or "")
    label = verdict_label(unit.verdict)
    if missing:
        return f"{met} of 4 met the test and {missing} could not be checked, so the phrase is {label}."
    return f"{met} of 4 met the test, so the phrase is {label}."


def stop_summary(status: AssessmentStatus, reason: str | None, question: str | None) -> str:
    if status is AssessmentStatus.NEEDS_CLARIFICATION:
        return f"FinSight needs a bit more information before it can check this claim. {question or ''}".strip()
    if status is AssessmentStatus.REFUSED:
        return f"FinSight didn't analyse this request. {reason or ''}".strip()
    return f"Something went wrong, so FinSight couldn't finish checking this claim. {reason or ''}".strip()
