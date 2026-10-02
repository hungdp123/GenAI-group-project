"""Deterministic routing, scope and verdict-guard rules used by the orchestrator. Owner: Person 5 (Hung).

Everything here is pure (no I/O, no LLM) so it is easy to test and to explain in the report.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from finsight.contracts.schemas import (
    SubClaim,
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


def resolve_cutoff(explicit: date | None, claim: str, corpus_as_of: date) -> tuple[date, bool]:
    """Return (cutoff, was_defaulted).

    # DECISION: an explicit argument wins; otherwise a "Cut-off date: ..." phrase in the claim is
    # honoured; otherwise default to the corpus as-of date.
    """
    if explicit is not None:
        return explicit, False
    match = _CUTOFF_RE.search(claim)
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


def detect_company(claim: str, companies: dict[str, list[str]], hint: str | None = None) -> str | None:
    """Ticker of the corpus company named in the claim (or the explicit hint, if it is in the corpus)."""
    if hint:
        return hint.upper() if hint.upper() in companies else None
    for ticker, aliases in companies.items():
        if any(re.search(rf"\b{re.escape(a)}\b", claim, flags=re.IGNORECASE) for a in aliases):
            return ticker
    return None


def _covered(companies: dict[str, list[str]]) -> str:
    return ", ".join(f"{aliases[0]} ({ticker})" for ticker, aliases in companies.items())


def scope_screen(claim: str, ticker: str | None, hint: str | None,
                 companies: dict[str, list[str]]) -> tuple[str | None, str | None]:
    """Initial scope screening: (refusal_reason, clarification_question); both None if in scope.

    # DECISION: price predictions, macro claims and named companies outside the corpus are REFUSED.
    # A claim that names no company is unclear, so we ask which company (diagram: Clarification).
    """
    if _PRICE_PREDICTION_RE.search(claim):
        return "Price predictions are out of scope: FinSight evaluates company fundamentals, not price targets.", None
    if ticker is None and hint:
        return f"{hint} is not in the frozen corpus. FinSight covers: {_covered(companies)}.", None
    if ticker is None and _MACRO_RE.search(claim):
        return "Macro / market-level claims are out of scope: FinSight evaluates company-level claims only.", None
    if ticker is None:
        return None, (f"Which company is this claim about? FinSight covers: {_covered(companies)}. "
                      "Resubmit naming the company.")
    return None, None


CLARIFY_EMPTY = ("Please paste the investment claim you want checked, e.g. \"Company X's revenue growth is "
                 "driven by AI demand.\"")
CLARIFY_NO_SUB_CLAIM = ("Your claim has nothing specific FinSight can check. What should it verify? For example "
                        "revenue growth, margins, what is driving demand, or whether growth will continue.")


# ---------------------------------------------------------------------------
# Plan / re-plan
# ---------------------------------------------------------------------------


def plan_entry(sub_claim: SubClaim, rule: KnowledgeRule) -> RoutePlanEntry:
    return RoutePlanEntry(
        sub_claim_id=sub_claim.sub_claim_id, sub_claim_type=sub_claim.sub_claim_type, collectors=list(rule.collectors),
        verdict_cap=rule.highest_verdict, tier_requirement=list(rule.tier_requirement), note=rule.note,
    )


def replan_entry(entry: RoutePlanEntry, rule: KnowledgeRule) -> RoutePlanEntry:
    """Plan for a retry: the original collectors plus the rule's retry_collectors (no duplicates)."""
    extra = [c for c in rule.retry_collectors if c not in entry.collectors]
    return entry.model_copy(update={"collectors": [*entry.collectors, *extra]})


def tier_shortfall(verdict: Verdict, evidence: list[EvidenceItem], rule: KnowledgeRule) -> str | None:
    """The auditor's "source tier: enough?" question. Returns retry feedback when NOT enough, else None.

    A positive verdict needs SUPPORTING evidence from a required tier (Tier 3 never counts).
    Any other verdict needs at least one item from a required tier.
    """
    if not rule.tier_requirement:
        return None
    tiers = "/".join(t.value for t in rule.tier_requirement)
    if verdict.is_positive:
        if eligible_support(evidence, rule):
            return None
        return f"Need SUPPORTING evidence from {tiers}; Tier 3 news cannot support a verdict."
    if any(e.source_tier in rule.tier_requirement and e.source_tier is not SourceTier.TIER_3 for e in evidence):
        return None
    return f"No evidence from {tiers} was found; search {tiers} sources for this sub-claim."


# ---------------------------------------------------------------------------
# Evidence screening
# ---------------------------------------------------------------------------


def split_temporal(items: list[EvidenceItem], cutoff: date) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    """(kept, dropped): hard filter on publication_date <= cutoff."""
    kept = [i for i in items if i.publication_date <= cutoff]
    return kept, [i for i in items if i.publication_date > cutoff]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def is_claim_echo(item: EvidenceItem, claim: str) -> bool:
    """True when an evidence item is just the user's claim (the claim is NEVER evidence)."""
    if item.doc_type.lower() in {"claim", "user_input"}:
        return True
    content = _norm(item.content)
    return len(content) >= 20 and content in _norm(claim)


# ---------------------------------------------------------------------------
# Verdict guards (applied after the TierPolicy agent; orchestrator-owned invariants)
# ---------------------------------------------------------------------------


def cap_verdict(verdict: Verdict, rule: KnowledgeRule) -> tuple[Verdict, str | None]:
    cap = rule.highest_verdict
    if cap is not None and verdict.support_rank > cap.support_rank:
        return cap, f"knowledge layer caps {rule.sub_claim_type.value} at {cap.value}"
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
    """True if `after` is a positive verdict that endorses the sub-claim more than `before`."""
    return after.is_positive and after.support_rank > before.support_rank
