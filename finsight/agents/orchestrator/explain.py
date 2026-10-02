"""Plain-language wording for end users. Owner: Person 5 (Hung).

Everything here is built BY RULE from the verdicts and evidence the pipeline already produced:
no LLM, no new numbers (numbers only appear inside quoted evidence text), no advice language.
The UI (Person 4) can show `FinalAssessment.summary` and `SubClaimResult.explanation` directly,
and reuse `verdict_label` / `source_label` for its own layout.
"""

from __future__ import annotations

from datetime import date

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

# Why some kinds of statement can only ever be partly confirmed from company documents.
_PARTIAL_BY_NATURE = {
    SubClaimType.DEMAND_ATTRIBUTION: "company documents can show what management says drives growth, but can't "
                                     "fully prove the cause",
    SubClaimType.CONTINUATION: "this is about the future, and past results or management expectations can't "
                               "prove what will happen",
    SubClaimType.OTHER: "this statement didn't fit a standard type, so FinSight is cautious about it",
}


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

    if result.verdict is Verdict.INSUFFICIENT_EVIDENCE:
        if sub_type is SubClaimType.COMPARATIVE:
            return ("This compares the company with its competitors, but FinSight only has this company's own "
                    "documents, so the comparison can't be checked." + news_note)
        if not_assessed_reason and "failed" in not_assessed_reason and "audit" not in not_assessed_reason:
            return "Part of the system failed while checking this point, so it couldn't be confirmed."
        return f"No official filing or company report{when} confirms this." + news_note
    if result.verdict is Verdict.SUPPORTED:
        tier1 = any(e.source_tier is SourceTier.TIER_1 for e in reliable_for)
        source = "official filings" if tier1 else "the company's own reports"
        return f"Backed up by {source} (see the evidence below)."
    if result.verdict is Verdict.CONTRADICTED:
        return "The evidence below points the other way."
    # PARTIALLY_SUPPORTED
    if reliable_for and reliable_against:
        sentence = "Some evidence supports this and some points the other way (see below)."
    elif reliable_for:
        sentence = "The evidence below points this way."
    else:
        sentence = "Only weak evidence points this way."
    if sub_type in _PARTIAL_BY_NATURE:
        sentence += f" It can only be partly confirmed because {_PARTIAL_BY_NATURE[sub_type]}."
    return sentence + news_note


def summary(results: list[SubClaimResult], overall: Verdict | None) -> str:
    """Overall sentence + how the points break down."""
    if not results:
        return "FinSight couldn't find any specific point in this claim to check."
    counts: dict[str, int] = {}
    for r in results:
        label = verdict_label(r.verdict).lower()
        counts[label] = counts.get(label, 0) + 1
    order = [verdict_label(v).lower() for v in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED,
                                                 Verdict.CONTRADICTED, Verdict.INSUFFICIENT_EVIDENCE)]
    breakdown = ", ".join(f"{counts[k]} {k}" for k in order if k in counts)
    n = len(results)
    return (f"Overall, this claim is {verdict_label(overall).lower()}. FinSight split it into {n} "
            f"point{'s' if n != 1 else ''} and checked each against company filings and reports: {breakdown}.")


def stop_summary(status: AssessmentStatus, reason: str | None, question: str | None) -> str:
    if status is AssessmentStatus.NEEDS_CLARIFICATION:
        return f"FinSight needs a bit more information before it can check this claim. {question or ''}".strip()
    if status is AssessmentStatus.REFUSED:
        return f"FinSight didn't analyse this request. {reason or ''}".strip()
    return f"Something went wrong, so FinSight couldn't finish checking this claim. {reason or ''}".strip()
