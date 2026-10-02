"""Plain-text rendering of a FinalAssessment (CLI output and the text the output advice gate checks)."""

from __future__ import annotations

from finsight.contracts.schemas import EvidenceItem, FinalAssessment


def _cite(e: EvidenceItem) -> str:
    tier = e.source_tier.value.replace("TIER_", "T")
    span = f" [{e.citation_span}]" if e.citation_span else ""
    return f"[{tier} {e.doc_type} {e.publication_date}] {e.content}{span} <{e.url or e.evidence_id}>"


def render_assessment(fa: FinalAssessment, *, show_evidence: bool = True) -> str:
    as_of = f"{fa.as_of_date}" + (" (defaulted to the corpus as-of date: no cut-off was given)"
                                  if fa.cutoff_was_defaulted else "")
    lines = [
        f"FinSight assessment {fa.request_id}",
        f"Status: {fa.status.value}    As of: {as_of}",
    ]
    if fa.refusal_reason:
        lines.append(f"Reason: {fa.refusal_reason}")
    if fa.overall_verdict:
        lines.append(f"Overall verdict: {fa.overall_verdict.value}")
    if fa.claim_results:
        lines += ["", f"{'ID':<4} {'TYPE':<28} {'VERDICT':<22} {'TIER':<7} {'RETRY':<5} CLAIM"]
        for r in fa.claim_results:
            tier = r.highest_tier_used.value if r.highest_tier_used else "-"
            lines.append(f"{r.claim.claim_id:<4} {r.claim.claim_type.value:<28} {r.verdict.value:<22} "
                         f"{tier:<7} {r.retries_used:<5} {r.claim.text}")
        for r in fa.claim_results:
            lines += ["", f"{r.claim.claim_id}: {r.rationale}"]
            if r.capped_by:
                lines.append(f"  capped by: {r.capped_by}")
            if show_evidence:
                lines += [f"  + {_cite(e)}" for e in r.supporting]
                lines += [f"  - {_cite(e)}" for e in r.challenging]
    if fa.evidence_gaps:
        lines += ["", "Evidence gaps:", *[f"  * {g}" for g in fa.evidence_gaps]]
    lines += ["", fa.boundary_disclaimer]
    return "\n".join(lines)
