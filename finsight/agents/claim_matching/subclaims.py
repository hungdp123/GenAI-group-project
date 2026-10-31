"""Turn a match into the sub-claims that get verified, and into the ClaimMatch the report shows.

Rules:
  * The influencer's words are what get verified. The user's text is only used to find them.
  * A matched line with one fact stays one sub-claim (the stored catalog row).
  * A line with several facts is split per fact, so each fact gets its own verdict and the headline only
    covers what the user asked: "Google grew 24%, its twelfth straight quarter" becomes two sub-claims.
  * A user_only fact never becomes a sub-claim. It is listed as "they did not say this".
  * A split fact keeps the full line in parent_span. Missing context the line itself supplied
    (company, measure) is added in [brackets], the usual way to complete a quote.
  * A segment or product figure ("data center revenue grew 57%", "Azure growth", "Google Cloud's backlog") and
    other figures the SEC data does not hold (run rate, backlog, buyback authorization, customer concentration)
    go to the narrative agent, never the financial one: the XBRL store has only company-wide totals, so
    comparing "data center up 57%" with total revenue (+38%) would wrongly contradict it. A fact that takes its
    measure from earlier in the line ("..., which grew 57%") inherits the segment too: "[AMD data center revenue]".
"""

from __future__ import annotations

import re

from finsight.agents.claim_extraction.thesaurus import display_name, subject_ticker
from finsight.agents.claim_matching.facts import BOTH, CONFLICT, QUOTE_ONLY, USER_ONLY, Aligned, align, split_facts
from finsight.agents.claim_matching.matcher import MatchOutcome, clean_user_claim, price_facts
from finsight.contracts.schemas import ClaimMatch, FactAlignment, SubClaim, SubClaimType

ASKED = (BOTH, CONFLICT)
_METRIC_WORD = {"revenue": "revenue", "profit": "profit", "margin": "margin", "cash": "cash flow",
                "debt": "debt", "leverage": "leverage"}
PRICE_NOTE = "A share-price view is not checked against filings."

# Business lines inside a company. Matched only next to a measure ("data center revenue", "Azure growth",
# "revenue from Google Cloud") or as the sentence's subject ("Google Cloud is now making ..."), so a passing
# mention ("margins of 66% on the cloud") does not move a company-wide figure away from the financial agent.
_SEG = (r"google cloud|microsoft cloud|intelligent cloud|cloud|azure|aws|amazon web services|data ?cent(?:er|re)s?|"
        r"youtube|google search|search|advertising|ad|iphones?|ipads?|macs?|wearables|services|gaming|automotive|"
        r"energy|subscriptions?|reality labs|family of apps|waymo|networking|compute|segment|division")
_SEG_FIGURE = re.compile(
    rf"\b(?:{_SEG})\b(?:'s)?\s+(?:segment\s+|business\s+)?(?:net\s+)?(?:revenues?|sales|growth|grew|operating income|"
    rf"profits?|gross margins?|reported)\b"
    rf"|\b(?:revenues?|sales)\s+(?:from|of|in)\s+(?:the\s+)?(?:{_SEG})\b"
    rf"|\$?\d[\d.,]*\s*(?:billion|million)?\s+(?:dollars\s+)?(?:from|of|in)\s+(?:the\s+)?(?:{_SEG})\b"
    rf"|^\W*(?:\[[^\]]*\]\s*)?(?:{_SEG})\b(?:'s)?\s+(?:is|was|has|had|now|came|reported|grew|generated|made)",
    re.IGNORECASE)
# Figures the SEC XBRL store does not hold at all.
_NOT_XBRL = re.compile(
    r"\b(?:backlog|run[- ]rate|remaining performance obligations?|buy ?backs?|repurchases?|authori[sz](?:ed|ation)|"
    r"(?:existing|share|repurchase|buyback) program|"
    r"(?:came|come|comes) from (?:just |only )?(?:\w+ )?(?:direct )?customers)\b", re.IGNORECASE)


def segment_figure(span: str, line: str = "", inherits: bool = False) -> str | None:
    """The segment / non-XBRL phrase that makes this fact a narrative check, or None.

    `inherits`: the fact borrows its measure from earlier in the line, so the line's segment applies to it.
    """
    for text in ([span, line] if inherits else [span]):
        match = _SEG_FIGURE.search(text or "") or _NOT_XBRL.search(text or "")
        if match:
            return match.group(0).strip()
    return None


_WHOLE_COMPANY = re.compile(r"\b(?:total|overall|consolidated|company[- ]wide|the company|group)\b", re.IGNORECASE)


def _earlier_segment(span: str, line: str) -> str | None:
    """A fact with its own measure but no subject of its own, after a segment figure earlier in the line, is about
    that segment: "AWS reported $35.6 billion ..., up 24% ... with $12.5 billion of operating income" -> the $12.5
    billion is AWS operating income, not Amazon's. Not when the fact names a company or says total / overall."""
    start = line.find(span)
    match = _SEG_FIGURE.search(line)
    if start <= 0 or match is None or match.start() >= start:
        return None
    if subject_ticker(span) is not None or _WHOLE_COMPANY.search(span):
        return None
    return match.group(0).strip()


def _segment_phrase(found: str) -> str:
    """'data center revenues came' -> 'data center revenue' (the bracket text for a split fact)."""
    words = re.sub(r"[^A-Za-z ]", " ", found).split()
    keep = [w for w in words if w.lower() not in {"is", "was", "has", "had", "now", "came", "reported", "grew",
                                                  "generated", "made", "from", "of", "in", "the", "dollars",
                                                  "billion", "million", "s"}]
    return " ".join(keep).lower()


def _as_narrative(claim: SubClaim, found: str | None = None) -> SubClaim:
    """Send to the narrative agent. `measure` says "segment" only for a business line (the report then explains
    "about that segment, not the whole company"); a backlog, run rate or buyback keeps no measure."""
    is_segment = found is not None and _SEG_FIGURE.search(found) is not None
    return claim.model_copy(update={"sub_claim_type": SubClaimType.OTHER,
                                    "measure": "segment" if is_segment else None})


def alignment_for(user_claim: str, quotes: list[str]) -> list[Aligned]:
    """The same alignment the matcher computed, rebuilt from plain text (the state only stores text)."""
    facts, _price = price_facts(split_facts(clean_user_claim(user_claim)))
    return align(facts, [split_facts(quote) for quote in quotes])


def to_contract(outcome: MatchOutcome, posts: list[dict] | None = None) -> ClaimMatch:
    title, uploaded = None, None
    if outcome.lines:
        first = outcome.lines[0]
        title, uploaded = first.title or None, first.upload_date
        for post in posts or []:
            if str(post.get("video_id") or "") == first.video_id and post.get("title"):
                title = post["title"]
    facts = [_fact(item) for item in outcome.aligned]
    facts += [FactAlignment(bucket=USER_ONLY, user_span=fact.span, note=PRICE_NOTE) for fact in outcome.skipped]
    return ClaimMatch(
        status=outcome.status, method=outcome.method, user_claim=outcome.user_claim,
        video_id=outcome.video_id, title=title, upload_date=uploaded,
        quotes=[line.text for line in outcome.lines], quote_ids=[line.line_id for line in outcome.lines],
        facts=facts, candidates_considered=outcome.considered, question=outcome.question,
    )


def _fact(item: Aligned, sub_claim_id: str | None = None) -> FactAlignment:
    user_span = None
    if item.user is not None:
        user_span = item.user.label if item.user.value is not None else item.user.span
    return FactAlignment(
        bucket=item.bucket, user_span=user_span,
        quote_span=item.quote.span if item.quote is not None else None,
        sub_claim_id=sub_claim_id, note=item.note,
    )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _bucket(items: list[Aligned]) -> str:
    kinds = {item.bucket for item in items}
    if CONFLICT in kinds:
        return CONFLICT
    if BOTH in kinds:
        return BOTH
    return QUOTE_ONLY


def _self_contained(span: str, quote: str, ticker: str | None, inherited: frozenset[str],
                    segment: str | None = None) -> str:
    """The span, with [company, measure] in front when only the rest of the line supplied them. A measure
    inherited from a segment phrase keeps the segment: "[AMD data center revenue] which grew 57%"."""
    missing = []
    if ticker and subject_ticker(span) is None and subject_ticker(quote) in (ticker, None):
        missing.append(display_name(ticker))
    if segment is not None and inherited:
        if segment:                 # the line's segment phrase replaces the bare measure word
            missing.append(segment)
        inherited = frozenset()     # "" = the span names its own segment / figure; no measure word added
    elif segment:                   # own measure, segment named earlier in the line: "[Amazon aws] $12.5 billion ..."
        missing.append(segment)
    for metric in sorted(inherited):
        missing.append(_METRIC_WORD.get(metric, metric))
    span = span.strip()
    return f"[{' '.join(missing)}] {span}" if missing else span


def build_sub_claims(
    match: ClaimMatch,
    stored: list[SubClaim] | None,
    ticker: str | None,
    make_claim,
) -> tuple[list[SubClaim], ClaimMatch, dict[str, str]]:
    """Sub-claims for the matched lines, the match with each fact pointing at its sub-claim, and the
    not-assessed reasons of freshly built claims (keyed by final sub-claim id).

    `make_claim(text) -> (SubClaim, reason | None)` builds a claim (type, measure) for a split fact or an
    unstored line.
    """
    reasons: dict[str, str] = {}
    aligned = alignment_for(match.user_claim, match.quotes)
    by_text = {_norm(claim.text): claim for claim in stored or []}
    by_id = {claim.sub_claim_id: claim for claim in stored or []}
    sub_claims: list[SubClaim] = []
    facts: list[FactAlignment] = []
    for qi, quote in enumerate(match.quotes):
        items = [item for item in aligned if item.quote_index == qi]
        stored_id = match.quote_ids[qi] if qi < len(match.quote_ids) else None
        base = by_id.get(stored_id or "") or by_text.get(_norm(quote))
        groups: dict[str, list[Aligned]] = {}
        for item in items:
            groups.setdefault(item.quote.span if item.quote else quote, []).append(item)
        # One sub-claim per fact, so each fact gets its own verdict ("24%: Supported; twelfth straight quarter:
        # Insufficient") instead of one blended label for the whole line. A one-fact line stays the stored row.
        if len(groups) <= 1:
            reason = None
            if base is None:
                base_claim, reason = make_claim(quote)
                claim = base_claim.model_copy(update={"sub_claim_id": f"m{qi + 1}"})
            else:
                claim = base
                if segment_figure(quote):
                    claim = _as_narrative(claim, segment_figure(quote))
                elif claim.sub_claim_type is SubClaimType.OTHER:
                    # The catalog typed a line OTHER ("increased their revenue by 16% year over year"): read it
                    # again, so a calculable figure goes to the financial agent instead of the narrative one.
                    again, _reason = make_claim(quote)
                    if again.sub_claim_type is not SubClaimType.OTHER:
                        claim = claim.model_copy(update={"sub_claim_type": again.sub_claim_type,
                                                         "measure": claim.measure or again.measure})
            claim = claim.model_copy(update={"alignment": _bucket(items) if items else QUOTE_ONLY,
                                             "user_text": _user_text(items) or claim.user_text})
            if base is None and segment_figure(quote):
                claim = _as_narrative(claim, segment_figure(quote))
            if reason:
                reasons[claim.sub_claim_id] = reason
            sub_claims.append(claim)
            facts += [_fact(item, claim.sub_claim_id) for item in items]
            continue
        prefix = base.sub_claim_id if base else f"m{qi + 1}"
        for number, (span, group) in enumerate(groups.items(), start=1):
            fact = group[0].quote
            inherited = fact.inherited if fact else frozenset()
            segment = segment_figure(span, quote, inherits=bool(inherited)) or _earlier_segment(span, quote)
            in_line_only = segment is not None and segment_figure(span) is None
            text = _self_contained(span, quote, ticker, inherited,
                                   None if segment is None else (_segment_phrase(segment) if in_line_only else ""))
            fresh, reason = make_claim(text)
            if base is not None and fresh.sub_claim_type is SubClaimType.OTHER:
                # The fact's own words are too short to classify; the stored line's type still applies.
                fresh = fresh.model_copy(update={"sub_claim_type": base.sub_claim_type,
                                                 "measure": fresh.measure or base.measure})
            if segment:
                fresh = _as_narrative(fresh, segment)
            claim = fresh.model_copy(update={
                "sub_claim_id": f"{prefix}-f{number}",
                "company_ticker": (base.company_ticker if base else None) or ticker,
                "source_span": quote,
                "parent_span": quote,
                "context": quote,  # the whole line; clues.with_clues adds the sentences before it
                "expansion_rule": None,  # not a section 6 expansion: each fact is reported on its own
                "quote_time": base.quote_time if base else None,
                "alignment": _bucket(group),
                "user_text": _user_text(group),
            })
            if reason:
                reasons[claim.sub_claim_id] = reason
            sub_claims.append(claim)
            facts += [_fact(item, claim.sub_claim_id) for item in group]
    facts += [_fact(item) for item in aligned if item.bucket == USER_ONLY]
    facts += [fact for fact in match.facts if fact.bucket == USER_ONLY and fact.note == PRICE_NOTE]
    return sub_claims, match.model_copy(update={"facts": facts}), reasons


def _user_text(items: list[Aligned]) -> str | None:
    spans = list(dict.fromkeys(item.user.span for item in items if item.user is not None))
    return " / ".join(spans) or None


def headline_ids(sub_claims: list[SubClaim]) -> list[str]:
    """The sub-claims the overall verdict is about: the facts the user asked. All of them when none is marked."""
    asked = [claim.sub_claim_id for claim in sub_claims if claim.alignment in ASKED]
    if asked:
        return asked
    return [claim.sub_claim_id for claim in sub_claims]
