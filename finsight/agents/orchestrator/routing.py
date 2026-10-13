"""Deterministic routing, scope and verdict-guard rules used by the orchestrator. Owner: Person 5 (Hung).

Everything here is pure (no I/O, no LLM) so it is easy to test and to explain in the report.
"""

from __future__ import annotations

import re
from datetime import date, datetime

from finsight.agents.claim_extraction.thesaurus import find_mentions, has_alias
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
        if any(has_alias(claim, alias) for alias in aliases):
            return ticker
    mentioned = list(dict.fromkeys(ticker for ticker, _surface, _start in find_mentions(claim)))
    if len(mentioned) == 1 and mentioned[0] in companies:
        return mentioned[0]
    return None


def _covered(companies: dict[str, list[str]]) -> str:
    return ", ".join(f"{aliases[0]} ({ticker})" for ticker, aliases in companies.items())


def price_or_macro_refusal(claim: str, ticker: str | None) -> str | None:
    """Refuse a price prediction or a market-level claim before asking which influencer."""
    if _PRICE_PREDICTION_RE.search(claim):
        return "Price predictions are out of scope: FinSight evaluates company fundamentals, not price targets."
    if ticker is None and _MACRO_RE.search(claim):
        return "Macro / market-level claims are out of scope: FinSight evaluates company-level claims only."
    return None


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


# Canonical key -> names matched in the user prompt, longest first when scanning.
INFLUENCERS: dict[str, tuple[str, ...]] = {
    "joseph_carlson": ("Joseph Carlson", "Carlson"),
    "mark_roussin": ("Mark Roussin", "Roussin"),
    "alex_reffett": ("Alex Reffett", "Reffett"),
}
INFLUENCER_CHANNELS: dict[str, frozenset[str]] = {
    "joseph_carlson": frozenset({"joseph_carlson", "joseph_carlson_after_hours"}),
    "mark_roussin": frozenset({"mark_roussin"}),
    "alex_reffett": frozenset({"alex_reffett"}),
}


def influencer_names() -> str:
    return ", ".join(aliases[0] for aliases in INFLUENCERS.values())


def detect_influencer(text: str, hint: str | None = None) -> str | None:
    """Canonical influencer key named in the hint or the text. None if nobody is named."""
    if hint and hint.strip():
        named = _match_influencer(hint)
        if named:
            return named
    return _match_influencer(text)


def _match_influencer(text: str) -> str | None:
    pairs = sorted(
        ((key, alias) for key, aliases in INFLUENCERS.items() for alias in aliases),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    for key, alias in pairs:
        if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
            return key
    return None


def influencer_screen(text: str, hint: str | None) -> tuple[str | None, str | None, str | None]:
    """(influencer key, refusal, clarification). A blank hint is a reply that named nobody."""
    if hint is not None and not hint.strip():
        return None, REFUSE_NO_INFLUENCER, None
    if hint and hint.strip() and detect_influencer("", hint) is None:
        return None, (
            f"{hint.strip()} is not an influencer FinSight can check. "
            f"FinSight checks posts from: {influencer_names()}."
        ), None
    found = detect_influencer(text, hint)
    if found is None:
        return None, None, CLARIFY_INFLUENCER
    return found, None, None


CLARIFY_EMPTY = ("Please paste the investment claim you want checked, e.g. \"Company X's revenue growth is "
                 "driven by AI demand.\"")
CLARIFY_INFLUENCER = (
    "Which influencer should this claim be checked against? "
    f"FinSight checks posts from: {influencer_names()}. Resubmit naming the influencer."
)
REFUSE_NO_INFLUENCER = (
    "FinSight checks an influencer's claim about one company. "
    "No influencer was named, so this request is refused."
)
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


def cap_tier2_only_support(
    verdict: Verdict, evidence: list[EvidenceItem], rule: KnowledgeRule
) -> tuple[Verdict, str | None]:
    """Tier 2 may support a positive verdict, but SUPPORTED still needs a Tier 1 supporting item."""
    if verdict is not Verdict.SUPPORTED:
        return verdict, None
    if SourceTier.TIER_1 not in rule.tier_requirement or SourceTier.TIER_2 not in rule.tier_requirement:
        return verdict, None
    support = eligible_support(evidence, rule)
    if any(e.source_tier is SourceTier.TIER_1 for e in support):
        return verdict, None
    if any(e.source_tier is SourceTier.TIER_2 for e in support):
        return Verdict.PARTIALLY_SUPPORTED, "Tier 2 evidence can at most Partially Support"
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
