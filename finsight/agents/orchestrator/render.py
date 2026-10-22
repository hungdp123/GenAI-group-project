"""Plain-text rendering of a FinalAssessment (CLI output and the text the output advice gate checks).

Reader view (default): summary first, then each point in plain words, then the evidence with
readable source names. Technical view (`technical=True`): the developer table with sub-claim types,
tiers, retries, audit rationale and caps.
"""

from __future__ import annotations

import textwrap

from finsight.agents.orchestrator.expansion import report_units
from finsight.agents.orchestrator.explain import (
    expansion_block,
    friendly_date,
    source_label,
    unit_heading,
    unit_partial_reason,
    verdict_label,
)
from finsight.contracts.schemas import AssessmentStatus, EvidenceItem, FinalAssessment, SubClaimResult

WIDTH = 96


def _wrap(text: str, indent: str = "", first: str | None = None) -> list[str]:
    first = indent if first is None else first
    return textwrap.wrap(text, WIDTH, initial_indent=first, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False) or [first.rstrip()]


def _as_of(fa: FinalAssessment) -> str:
    if not fa.as_of_date:
        return "-"
    note = " (no date was given, so the latest available data was used)" if fa.cutoff_was_defaulted else ""
    return f"{friendly_date(fa.as_of_date)}{note}"


def _evidence(e: EvidenceItem) -> list[str]:
    lines = _wrap(f'"{e.content.strip()}"', indent="        ", first="      - ")
    lines += _wrap(f"Source: {source_label(e)}", indent="          ", first="        ")
    where = "; ".join(x for x in (e.citation_span, e.url) if x)
    if where:
        lines += _wrap(f"Where: {where}", indent="          ", first="        ")
    return lines


def render_assessment(fa: FinalAssessment, *, show_evidence: bool = True, technical: bool = False) -> str:
    if technical:
        return render_technical(fa, show_evidence=show_evidence)
    lines = ["FINSIGHT CHECK", ""]
    if fa.claim:
        lines += _wrap(f'Claim checked: "{fa.claim.strip()}"', indent="               ", first="")
    lines += [f"Information up to: {_as_of(fa)}", ""]
    if fa.checked_post:
        lines += _wrap(fa.checked_post, indent="  ", first="")
        lines.append("")

    lines += ["SUMMARY"]
    if fa.status is AssessmentStatus.COMPLETED:
        lines += _wrap(fa.summary or f"Overall: {verdict_label(fa.overall_verdict)}.", indent="  ", first="  ")
        units = report_units(fa.sub_claim_results)
        if units:
            lines.append("")
            width = max(len(_tagged(unit)) for unit in units)
            for i, unit in enumerate(units, start=1):
                tag = f"[{_tagged(unit)}]".ljust(width + 1)
                lines += _wrap(unit_heading(unit), indent=" " * (6 + width + 1), first=f"  {i}. {tag} ")
    else:
        lines += _wrap(fa.summary or (fa.clarification_question or fa.refusal_reason or ""), indent="  ", first="  ")

    if fa.sub_claim_results:
        lines += ["", "DETAILS"]
        if not show_evidence:
            lines.append("(evidence hidden; run without --no-evidence to see the sources)")
        for i, unit in enumerate(report_units(fa.sub_claim_results), start=1):
            if unit.is_parent:
                lines += ["", * _wrap(f"{i}. {expansion_block(unit)}", indent="   ")]
                if show_evidence:
                    lines += _unit_evidence(unit.results, show_evidence=True)
                continue
            r = unit.results[0]
            lines += [""] + _wrap(f"{i}. {r.sub_claim.text}  ->  {_tagged(unit)}", indent="   ")
            if r.explanation:
                lines += _wrap(f"Why: {r.explanation}", indent="        ", first="   ")
            if show_evidence:
                lines += _unit_evidence((r,), show_evidence=True)

    lines += [""] + _wrap(f"Note: {fa.boundary_disclaimer}")
    if fa.status is AssessmentStatus.COMPLETED:
        lines += _wrap(REPORT_LIMITS, indent="  ", first="  ")
    lines += [f"Reference: {fa.request_id}"]
    return "\n".join(lines)


def _tagged(unit) -> str:
    label = verdict_label(unit.verdict)
    reason = unit_partial_reason(unit)
    return f"{label}: {reason}" if reason else label


REPORT_LIMITS = (
    "This label is a document check of the sentence against the filing. It can be wrong. "
    "The project window is 1 Apr 2026 through 30 Sep 2026. "
    "This check is in English."
)


def _unit_evidence(results: tuple[SubClaimResult, ...], *, show_evidence: bool) -> list[str]:
    lines: list[str] = []
    if not show_evidence:
        return lines
    any_evidence = False
    for result in results:
        if result.supporting:
            any_evidence = True
            lines.append("   Evidence for:")
            for item in result.supporting:
                lines += _evidence(item)
        if result.challenging:
            any_evidence = True
            lines.append("   Evidence against:")
            for item in result.challenging:
                lines += _evidence(item)
    if not any_evidence:
        lines.append("   Evidence: none found")
    return lines


def _cite(e: EvidenceItem) -> str:
    tier = e.source_tier.value.replace("TIER_", "T")
    span = f" [{e.citation_span}]" if e.citation_span else ""
    return f"[{tier} {e.doc_type} {e.publication_date}] {e.content}{span} <{e.url or e.evidence_id}>"


def render_technical(fa: FinalAssessment, *, show_evidence: bool = True) -> str:
    """Developer view: sub-claim types, tiers, retries, audit rationale, caps."""
    as_of = f"{fa.as_of_date}" + (" (defaulted to the corpus as-of date: no cut-off was given)"
                                  if fa.cutoff_was_defaulted else "")
    lines = [f"FinSight assessment {fa.request_id}", f"Status: {fa.status.value}    As of: {as_of}"]
    if fa.refusal_reason:
        lines.append(f"Reason: {fa.refusal_reason}")
    if fa.clarification_question:
        lines.append(f"Question: {fa.clarification_question}")
    if fa.overall_verdict:
        lines.append(f"Overall verdict: {fa.overall_verdict.value}")
    if fa.sub_claim_results:
        lines += ["", f"{'ID':<4} {'TYPE':<28} {'VERDICT':<22} {'TIER':<7} {'RETRY':<5} SUB-CLAIM"]
        for r in fa.sub_claim_results:
            tier = r.highest_tier_used.value if r.highest_tier_used else "-"
            lines.append(f"{r.sub_claim.sub_claim_id:<4} {r.sub_claim.sub_claim_type.value:<28} {r.verdict.value:<22} "
                         f"{tier:<7} {r.retries_used:<5} {r.sub_claim.text}")
        for r in fa.sub_claim_results:
            lines += ["", f"{r.sub_claim.sub_claim_id}: {r.rationale}"]
            if r.capped_by:
                lines.append(f"  capped by: {r.capped_by}")
            if show_evidence:
                lines += [f"  + {_cite(e)}" for e in r.supporting]
                lines += [f"  - {_cite(e)}" for e in r.challenging]
    if fa.evidence_gaps:
        lines += ["", "Evidence gaps:", *[f"  * {g}" for g in fa.evidence_gaps]]
    lines += ["", fa.boundary_disclaimer]
    return "\n".join(lines)
