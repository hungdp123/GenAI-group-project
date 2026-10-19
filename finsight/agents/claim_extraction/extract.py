"""Turn one chosen post into typed claims. Writes no label.

Person 3's decomposer proposes the claims. This step keeps the company named in the prompt,
drops an advice line inside a stored post, and hands the claims to the orchestrator.
The post's words stay the claim. A label is decided later.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from finsight.agents.claim_extraction.prepare import prepare_post
from finsight.agents.claim_extraction.select import explain_miss, matching_claims
from finsight.agents.claim_extraction.thesaurus import find_mentions
from finsight.agents.orchestrator import routing
from finsight.contracts.schemas import AssessmentStatus, SubClaim, SubClaimType
from finsight.knowledge.loader import KnowledgeLayer


@dataclass
class Extraction:
    sub_claims: list[SubClaim] = field(default_factory=list)
    status: AssessmentStatus | None = None
    refusal_reason: str | None = None
    clarification_question: str | None = None
    notes: list[str] = field(default_factory=list)
    insufficient_reasons: dict[str, str] = field(default_factory=dict)
    dropped_by_cap: list[str] = field(default_factory=list)
    miss: str | None = None


def load_stored_claims(path: Path, video_id: str | None, company_ticker: str | None) -> list[SubClaim] | None:
    """Sentences stored for this post when the corpus was built.

    None means this post has no stored record yet, so extraction may run once.
    A list means the record exists and must be used as-is. This is not a vector database.
    """
    if not video_id or not path.exists():
        return None
    found = False
    claims: list[SubClaim] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("video_id") or "") != video_id:
            continue
        found = True
        ticker = row.get("company_ticker")
        if company_ticker and ticker and ticker != company_ticker:
            continue
        try:
            claim_type = SubClaimType(row["sub_claim_type"])
        except (KeyError, ValueError):
            continue
        claims.append(SubClaim(
            sub_claim_id=str(row.get("sub_claim_id") or f"c{len(claims) + 1}"),
            text=str(row.get("text") or row.get("source_span") or ""),
            sub_claim_type=claim_type,
            company_ticker=ticker or company_ticker,
            is_quantitative=bool(row.get("is_quantitative")),
            source_span=row.get("source_span"),
            parent_span=row.get("parent_span"),
            expansion_rule=row.get("expansion_rule"),
            measure=row.get("measure"),
            quote_time=row.get("quote_time"),
        ))
    return claims if found else None


def load_claim_rows(
    path: Path, video_ids: set[str], company_ticker: str | None,
) -> list[tuple[SubClaim, str]]:
    """Stored claims for these videos, each paired with its video id."""
    if not video_ids or not path.exists():
        return []
    rows: list[tuple[SubClaim, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        video_id = str(row.get("video_id") or "")
        if video_id not in video_ids:
            continue
        ticker = row.get("company_ticker")
        if company_ticker and ticker and ticker != company_ticker:
            continue
        try:
            claim_type = SubClaimType(row["sub_claim_type"])
        except (KeyError, ValueError):
            continue
        claim = SubClaim(
            sub_claim_id=str(row.get("sub_claim_id") or f"c{len(rows) + 1}"),
            text=str(row.get("text") or row.get("source_span") or ""),
            sub_claim_type=claim_type,
            company_ticker=ticker or company_ticker,
            is_quantitative=bool(row.get("is_quantitative")),
            source_span=row.get("source_span"),
            parent_span=row.get("parent_span"),
            expansion_rule=row.get("expansion_rule"),
            measure=row.get("measure"),
            quote_time=row.get("quote_time"),
        )
        rows.append((claim, video_id))
    return rows


def extraction_from_quotes(
    quotes: list[str], stored: list[SubClaim] | None, ticker: str | None,
) -> Extraction:
    """Check the sentences already chosen. A segment line is still checked."""
    out = Extraction()
    by_text = {_norm(claim.text): claim for claim in stored or []}
    sub_claims: list[SubClaim] = []
    for index, quote in enumerate(quotes, start=1):
        stored_claim = by_text.get(_norm(quote))
        if stored_claim is not None:
            sub_claims.append(stored_claim)
            continue
        claim, reason = _quote_claim(quote, ticker, index)
        sub_claims.append(claim)
        if reason:
            out.insufficient_reasons[claim.sub_claim_id] = reason
    sub_claims, out.dropped_by_cap = keep_within_cap(sub_claims)
    out.sub_claims = sub_claims
    out.notes.append(
        f"{len(sub_claims)} sub-claims: " + ", ".join(f"{c.sub_claim_id}={c.sub_claim_type.value}" for c in sub_claims)
    )
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _quote_claim(quote: str, ticker: str | None, index: int) -> tuple[SubClaim, str | None]:
    from finsight.agents.claim_extraction.build import _ADVICE_OR_PRICE, _classify

    kind = _classify(quote)
    reason = kind.reason if kind is not None else None
    claim_type = SubClaimType(kind.type) if kind is not None else SubClaimType.OTHER
    if kind is None and (_ADVICE_OR_PRICE.search(quote) or re.search(r"\bworth\b.{0,30}\$\s*\d", quote)):
        reason = "This line is about a price, so it is not checked against a filing."
    claim = SubClaim(
        sub_claim_id=f"q{index}",
        text=quote,
        sub_claim_type=claim_type,
        company_ticker=ticker,
        is_quantitative=bool(re.search(r"\d", quote)),
        source_span=quote,
        measure=kind.measure if kind is not None else None,
    )
    return claim, reason


def load_stored_reasons(path: Path, video_id: str | None, company_ticker: str | None) -> dict[str, str]:
    """Reasons stored with a sentence, such as a garbled number. Empty when the video has no record."""
    if not video_id or not path.exists():
        return {}
    reasons: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("video_id") or "") != video_id:
            continue
        ticker = row.get("company_ticker")
        if company_ticker and ticker and ticker != company_ticker:
            continue
        reason = row.get("insufficient_reason")
        claim_id = str(row.get("sub_claim_id") or "")
        if reason and claim_id:
            reasons[claim_id] = str(reason)
    return reasons


def extract_claims(
    *,
    text: str,
    cutoff: date,
    company_ticker: str | None,
    companies: dict[str, list[str]],
    from_stored_post: bool,
    decomposer: Any,
    knowledge: KnowledgeLayer,
    call: Callable[..., tuple[Any, Exception | None]],
    stored: list[SubClaim] | None = None,
    stored_reasons: dict[str, str] | None = None,
    focus: str | None = None,
) -> Extraction:
    """Use stored sentences when the corpus has them. Otherwise call the decomposer once.

    `focus` is the claim the user stated. Only stored rows that match it are checked.
    A transcript line that is not in that list is explained and is not verified.
    """
    out = Extraction()
    if stored is not None:
        sub_claims, error = stored, None
        if stored_reasons:
            out.insufficient_reasons.update(stored_reasons)
        out.notes.append(f"used {len(stored)} stored claim(s)")
    else:
        model_text = text
        held: list[SubClaim] = []
        if from_stored_post:
            prepared = prepare_post(text, company_ticker)
            model_text = prepared.text
            held = prepared.held
            out.notes.extend(prepared.notes)
            out.insufficient_reasons.update(prepared.insufficient_reasons)
            for reason, quote in prepared.dropped:
                out.notes.append(f"dropped ({reason}): {quote[:180]}")
        if model_text.strip():
            sub_claims, error = call("decomposer", decomposer.decompose, model_text, cutoff)
            if not error and isinstance(sub_claims, list):
                sub_claims = list(sub_claims) + held
        elif held:
            sub_claims, error = held, None
        else:
            sub_claims, error = call("decomposer", decomposer.decompose, text, cutoff)
    if error or not isinstance(sub_claims, list) or not all(isinstance(c, SubClaim) for c in sub_claims):
        out.status = AssessmentStatus.FAILED
        out.refusal_reason = f"Claim extraction failed: {error or 'invalid output'}"
        return out
    if len({c.sub_claim_id for c in sub_claims}) != len(sub_claims):
        sub_claims = [c.model_copy(update={"sub_claim_id": f"c{i + 1}"}) for i, c in enumerate(sub_claims)]
        out.notes.append("duplicate sub_claim_ids from decomposer; renumbered")
    before = len(sub_claims)
    sub_claims = _for_named_company(sub_claims, company_ticker, companies, from_stored_post)
    if len(sub_claims) < before:
        out.notes.append(f"dropped {before - len(sub_claims)} claim(s) about another company")
    kept_ids = {claim.sub_claim_id for claim in sub_claims}
    out.insufficient_reasons = {key: value for key, value in out.insufficient_reasons.items() if key in kept_ids}
    refusing = [
        c for c in sub_claims
        if c.sub_claim_type is SubClaimType.PERSONALIZED_ADVICE or knowledge.rule_for(c.sub_claim_type).refuse
    ]
    if refusing and from_stored_post:
        out.notes.append(f"dropped {len(refusing)} advice line(s) inside the stored post")
        sub_claims = [c for c in sub_claims if c not in refusing]
        refusing = []
    if focus:
        matched = matching_claims(focus, sub_claims)
        if not matched:
            out.miss = explain_miss(focus, text)
            out.notes.append(out.miss)
            return out
        if len(matched) < len(sub_claims):
            out.notes.append(f"kept {len(matched)} claim(s) matching the prompt")
        sub_claims = matched
    sub_claims, out.dropped_by_cap = keep_within_cap(sub_claims)
    if out.dropped_by_cap:
        out.notes.append(f"cap of 8 left out {len(out.dropped_by_cap)} claim(s)")
    out.sub_claims = sub_claims
    out.notes.append(
        f"{len(sub_claims)} sub-claims: " + ", ".join(f"{c.sub_claim_id}={c.sub_claim_type.value}" for c in sub_claims)
    )
    if refusing:
        note = knowledge.rule_for(refusing[0].sub_claim_type).note or "Personalized advice is out of scope."
        out.status = AssessmentStatus.REFUSED
        out.refusal_reason = f"Sub-claim {refusing[0].sub_claim_id}: {note}"
    elif from_stored_post and not sub_claims:
        out.notes.append("nothing in the post could be checked")
    elif not sub_claims or all(c.sub_claim_type is SubClaimType.OTHER for c in sub_claims):
        out.clarification_question = routing.CLARIFY_NO_SUB_CLAIM
    return out


# Keep this order when a post splits into more than 8 claims. The rest are listed, not checked.
_CAP = 8
_COMPANY_WIDE = frozenset({
    "financially_strong", "strong_fundamentals", "healthy_company", "good_company",
    "financially_weak", "weak_fundamentals", "poor_company", "in_trouble",
})
_PRIORITY = {
    SubClaimType.COMPARATIVE: 0,
    SubClaimType.CONTINUATION: 1,
    SubClaimType.DEMAND_ATTRIBUTION: 2,
    SubClaimType.BUSINESS_RISK: 6,
}
_NET_INCOME = re.compile(r"\b(earnings|profits?|net income)\b", flags=re.IGNORECASE)


def keep_within_cap(claims: list[SubClaim]) -> tuple[list[SubClaim], list[str]]:
    """At most 8 claims. Lower-priority quotes are returned so the report can list them."""
    if len(claims) <= _CAP:
        return claims, []
    ranked = sorted(enumerate(claims), key=lambda item: (_rank(item[1]), item[0]))
    keep_ids = {id(claims[index]) for index, _claim in ranked[:_CAP]}
    kept = [claim for claim in claims if id(claim) in keep_ids]
    dropped = [claim.text for claim in claims if id(claim) not in keep_ids]
    return kept, dropped


def _rank(claim: SubClaim) -> int:
    if claim.parent_span and claim.expansion_rule in _COMPANY_WIDE:
        return 5
    if claim.sub_claim_type is SubClaimType.HISTORICAL_GROWTH:
        return 4 if _NET_INCOME.search(claim.text) else 3
    return _PRIORITY.get(claim.sub_claim_type, 7)


def _for_named_company(
    sub_claims: list[SubClaim],
    company_ticker: str | None,
    companies: dict[str, list[str]],
    from_stored_post: bool,
) -> list[SubClaim]:
    """Keep a claim when it is about the company in the prompt. Do not rename another company onto it."""
    del companies
    if not (from_stored_post and company_ticker):
        return [
            c if c.company_ticker else c.model_copy(update={"company_ticker": company_ticker})
            for c in sub_claims
        ]
    kept = []
    for claim in sub_claims:
        mentioned = {ticker for ticker, _surface, _start in find_mentions(claim.text)}
        if mentioned and company_ticker not in mentioned:
            continue
        kept.append(claim.model_copy(update={"company_ticker": company_ticker}))
    return kept
