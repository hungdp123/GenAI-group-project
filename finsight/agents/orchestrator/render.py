"""Plain-text rendering of a FinalAssessment (CLI output and the text the output advice gate checks).

Reader view (default): summary first, then each point in plain words, then the evidence with
readable source names. Technical view (`technical=True`): the developer table with sub-claim types,
tiers, retries, audit rationale and caps.
"""

from __future__ import annotations

import textwrap

from finsight.agents.orchestrator.explain import friendly_date, source_label, verdict_label
from finsight.contracts.schemas import AssessmentStatus, EvidenceItem, FinalAssessment

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

    lines += ["SUMMARY"]
    if fa.status is AssessmentStatus.COMPLETED:
        lines += _wrap(fa.summary or f"Overall: {verdict_label(fa.overall_verdict)}.", indent="  ", first="  ")
        if fa.sub_claim_results:
            lines.append("")
            width = max(len(verdict_label(r.verdict)) for r in fa.sub_claim_results) + 2
            for i, r in enumerate(fa.sub_claim_results, start=1):
                tag = f"[{verdict_label(r.verdict)}]".ljust(width + 1)
                lines += _wrap(f"{r.sub_claim.text}", indent=" " * (6 + width + 1), first=f"  {i}. {tag} ")
    else:
        lines += _wrap(fa.summary or (fa.clarification_question or fa.refusal_reason or ""), indent="  ", first="  ")

    if fa.sub_claim_results:
        lines += ["", "DETAILS"]
        if not show_evidence:
            lines.append("(evidence hidden; run without --no-evidence to see the sources)")
        for i, r in enumerate(fa.sub_claim_results, start=1):
            lines += [""] + _wrap(f"{i}. {r.sub_claim.text}  ->  {verdict_label(r.verdict)}", indent="   ")
            if r.explanation:
                lines += _wrap(f"Why: {r.explanation}", indent="        ", first="   ")
            if show_evidence:
                if r.supporting:
                    lines.append("   Evidence for:")
                    for e in r.supporting:
                        lines += _evidence(e)
                if r.challenging:
                    lines.append("   Evidence against:")
                    for e in r.challenging:
                        lines += _evidence(e)
                if not r.supporting and not r.challenging:
                    lines.append("   Evidence: none found")

    lines += [""] + _wrap(f"Note: {fa.boundary_disclaimer}")
    lines += [f"Reference: {fa.request_id}"]
    return "\n".join(lines)


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
