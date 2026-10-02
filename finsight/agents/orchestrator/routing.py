"""Deterministic routing, scope and verdict-guard rules used by the orchestrator. Owner: Person 5 (Hung).

Everything here is pure (no I/O, no LLM) so it is easy to test and to explain in the report.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from finsight.contracts.schemas import (
    Claim,
    EvidenceItem,
    KnowledgeRule,
    RoutePlanEntry,
    SourceTier,
    Stance,
    Verdict,
)

# ---------------------------------------------------------------------------
# Cut-off
# ---------------------------------------------------------------------------

_CUTOFF_RE = re.compile(
    r"cut[\s-]?off(?:\s+date)?\s*[:=-]?\s*"
    r"(\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
    flags=re.IGNORECASE,
)
_DATE_FORMATS = ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y")


def parse_date(text: str) -> date | None:
    cleaned = re.sub(r"[.,]", "", text.strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def resolve_cutoff(explicit: date | None, thesis: str, corpus_as_of: date) -> tuple[date, bool]:
    """Return (cutoff, was_defaulted).

    # DECISION: an explicit argument wins; otherwise a "Cut-off date: ..." phrase in the thesis is
    # honoured; otherwise default to the corpus as-of date.
    """
    if explicit is not None:
        return explicit, False
    match = _CUTOFF_RE.search(thesis)
    if match and (parsed := parse_date(match.group(1))):
        return parsed, False
    return corpus_as_of, True


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

_PRICE_PREDICTION_RE = re.compile(
    r"price target|target price|(stock|share) price will|will (hit|reach|trade at)\s*\$|to the moon",
    flags=re.IGNORECASE,
)
_MACRO_RE = re.compile(
    r"\b(s&p\s?500|nasdaq|dow jones|interest rates?|inflation|recession|the fed|gdp|the market)\b",
    flags=re.IGNORECASE,
)


def detect_company(thesis: str, companies: dict[str, list[str]], hint: str | None = None) -> str | None:
    """Ticker of the corpus company named in the thesis (or the explicit hint, if it is in the corpus)."""
    if hint:
        return hint.upper() if hint.upper() in companies else None
    for ticker, aliases in companies.items():
        if any(re.search(rf"\b{re.escape(a)}\b", thesis, flags=re.IGNORECASE) for a in aliases):
            return ticker
    return None


def scope_refusal(thesis: str, ticker: str | None) -> str | None:
    """Reason the request is out of scope, or None if in scope.

    # DECISION: every out-of-scope case is REFUSED (not COMPLETED/INSUFFICIENT) so the
    # behaviour is easy to score against `expect_refusal`.
    """
    if _PRICE_PREDICTION_RE.search(thesis):
        return "Price predictions are out of scope: FinSight evaluates company fundamentals, not price targets."
    if ticker is None and _MACRO_RE.search(thesis):
        return "Macro / market-level theses are out of scope: FinSight evaluates company-level theses only."
    if ticker is None:
        return "No company from the frozen corpus was identified in the thesis."
    return None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def plan_entry(claim: Claim, rule: KnowledgeRule) -> RoutePlanEntry:
    return RoutePlanEntry(
        claim_id=claim.claim_id, claim_type=claim.claim_type, collectors=list(rule.collectors),
        verdict_cap=rule.highest_verdict, tier_requirement=list(rule.tier_requirement), note=rule.note,
    )


# ---------------------------------------------------------------------------
# Evidence screening
# ---------------------------------------------------------------------------


def split_temporal(items: list[EvidenceItem], cutoff: date) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """(kept, dropped): hard filter on publication_date <= cutoff."""
    kept = [i for i in items if i.publication_date <= cutoff]
    return kept, [i for i in items if i.publication_date > cutoff]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_thesis_echo(item: EvidenceItem, thesis: str) -> bool:
    """True when an evidence item is just the user's thesis (the thesis is NEVER evidence)."""
    if item.doc_type.lower() in {"thesis", "user_input"}:
        return True
    content = _norm(item.content)
    return len(content) >= 20 and content in _norm(thesis)


# ---------------------------------------------------------------------------
# Verdict guards (applied after the TierPolicy agent; orchestrator-owned invariants)
# ---------------------------------------------------------------------------


def cap_verdict(verdict: Verdict, rule: KnowledgeRule) -> tuple[Verdict, str | None]:
    cap = rule.highest_verdict
    if cap is not None and verdict.support_rank > cap.support_rank:
        return cap, f"knowledge layer caps {rule.claim_type.value} at {cap.value}"
    return verdict, None


def eligible_support(evidence: list[EvidenceItem], rule: KnowledgeRule) -> list[EvidenceItem]:
    """SUPPORTING evidence from a required tier. Tier 3 never qualifies, whatever the YAML says."""
    return [
        e for e in evidence
        if e.stance is Stance.SUPPORTING and e.source_tier in rule.tier_requirement
        and e.source_tier is not SourceTier.TIER_3
    ]


def enforce_tier_requirement(
    verdict: Verdict, evidence: list[EvidenceItem], rule: KnowledgeRule
) -> tuple[Verdict, str | None]:
    """A positive verdict needs at least one eligible supporting item; otherwise INSUFFICIENT_EVIDENCE."""
    if verdict.is_positive and not eligible_support(evidence, rule):
        tiers = ", ".join(t.value for t in rule.tier_requirement) or "none"
        return Verdict.INSUFFICIENT_EVIDENCE, (
            f"tier rule: no supporting evidence from required tiers ({tiers}); Tier 3 cannot raise a verdict"
        )
    return verdict, None


def enforce_counter_evidence(
    verdict: Verdict, evidence: list[EvidenceItem], rule: KnowledgeRule
) -> tuple[Verdict, str | None]:
    """For counter-evidence-only types, eligible CHALLENGING evidence blocks SUPPORTED."""
    if not rule.counter_evidence_only or verdict is not Verdict.SUPPORTED:
        return verdict, None
    blocking = [e for e in evidence if e.stance is Stance.CHALLENGING and e.source_tier in rule.tier_requirement]
    if blocking:
        return Verdict.PARTIALLY_SUPPORTED, f"counter-evidence {blocking[0].evidence_id} blocks SUPPORTED"
    return verdict, None


def is_raise(before: Verdict, after: Verdict) -> bool:
    """True if `after` is a positive verdict that endorses the claim more than `before`."""
    return after.is_positive and after.support_rank > before.support_rank
