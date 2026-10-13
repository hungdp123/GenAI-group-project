"""Write one claim row per kept sentence, before anyone asks a question.

The decomposer is not called. Each row keeps the post's words and the company
that sentence is about. A video that talks about two companies gets a row for
each. A question reads data/influencers/claims/claims.jsonl and does not extract again.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from finsight.agents.claim_extraction.prepare import (
    GARBLED_REASON,
    SEGMENT_COMPANY_REASON,
    _AS_COMPANY,
    _CHECKABLE,
    _GARBLED,
    _SENTENCE,
    _drop_reason,
    _quote_time,
)
from finsight.agents.claim_extraction.thesaurus import COMPANY_NAMES, find_mentions, subject_ticker
from finsight.agents.orchestrator.posts import in_window, load_posts
from finsight.agents.orchestrator.routing import INFLUENCER_CHANNELS

_CHANNELS = frozenset(channel for group in INFLUENCER_CHANNELS.values() for channel in group)
_MEASURES = ("profitability", "cash_generation", "liquidity", "leverage")
_WIDE = (
    (re.compile(r"\bfinancially strong\b", flags=re.IGNORECASE), "financially_strong"),
    (re.compile(r"\bstrong fundamentals\b", flags=re.IGNORECASE), "strong_fundamentals"),
    (re.compile(r"\bhealthy company\b", flags=re.IGNORECASE), "healthy_company"),
    (re.compile(r"\bgood company\b", flags=re.IGNORECASE), "good_company"),
    (re.compile(r"\bfinancially weak\b", flags=re.IGNORECASE), "financially_weak"),
    (re.compile(r"\bweak fundamentals\b", flags=re.IGNORECASE), "weak_fundamentals"),
    (re.compile(r"\bpoor company\b", flags=re.IGNORECASE), "poor_company"),
    (re.compile(r"\bin trouble\b", flags=re.IGNORECASE), "in_trouble"),
)
_NAMED_MEASURE = re.compile(
    r"\b(revenue|sales|margin|profit|earnings|cash|debt|income|liquidity|leverage)\b",
    flags=re.IGNORECASE,
)
_ADVICE_OR_PRICE = re.compile(
    r"\b(you should|i(?:'m| am) (?:buying|investing|putting|willing)|"
    r"(?:buy|sell)(?:ing)? the (?:stock|shares)|i(?:'ll| will) (?:buy|sell)|"
    r"committing|option strategy|get assigned|\bstrike\b|"
    r"share price|stock price|price target|target price|forward p/?e|p/e\b|"
    r"market price|buy price|trading at|shares have|sold off)\b"
    r"|\b\d+(?:\.\d+)?\s*x\b"
    r"|\b(shares|the stock)\b.{0,40}\b(sold|fell|rose|up|down|plummet)",
    flags=re.IGNORECASE,
)
_IF_MODAL = re.compile(r"\bif\b.*\b(would|could|might)\b", flags=re.IGNORECASE | re.DOTALL)
_FORWARD = re.compile(
    r"\b(guid(?:ed|ance)|expects|expected|forecast|next quarter|next year|"
    r"just getting started|will continue|going forward|long-term target)\b",
    flags=re.IGNORECASE,
)
_COMPARE = re.compile(
    r"\b(biggest winner|better than|ahead of|cheaper than|dethrone|number two player)\b",
    flags=re.IGNORECASE,
)
_CAUSE = re.compile(r"\b(driven by|due to|because of|from ai|thanks to)\b", flags=re.IGNORECASE)
_RESULT = re.compile(
    r"\b(revenues?|sales|earnings|profits?|net income|margins?|growth|grew)\b",
    flags=re.IGNORECASE,
)
_RISK = re.compile(r"\b(risks?|threat|customer concentration)\b", flags=re.IGNORECASE)
_PROFIT = re.compile(
    r"\b(margins?|free cash flow|cash flow|liquidity|current ratio|leverage|debt|balance sheet)\b",
    flags=re.IGNORECASE,
)
_GROWTH_MEASURE = re.compile(
    r"\b(revenues?|sales|earnings|profits?|net income|top line)\b",
    flags=re.IGNORECASE,
)
_DIRECTION = re.compile(
    r"\b(grew|growth|rose|risen|doubled|fell|fallen|declined|expanded|expanding|"
    r"improved|improving|strong|weak|solid|healthy)\b"
    r"|up \d|down \d|year[- ]over[- ]year",
    flags=re.IGNORECASE,
)
_NUMBER = re.compile(r"\d|billion|million", flags=re.IGNORECASE)
_SEGMENT = re.compile(
    r"\b(segment|division|aws|azure|instagram|waymo|iphone|google cloud|youtube|data centers?)\b",
    flags=re.IGNORECASE,
)
_HARD_DROP = frozenset({
    "guest", "not the host", "question", "hypothetical", "joke", "list mention", "unclear they",
})


def claims_path(posts_dir: Path) -> Path:
    return posts_dir.parent / "claims" / "claims.jsonl"


def claims_from_transcript(
    text: str,
    *,
    video_id: str,
    influencer: str,
    title: str = "",
    upload_date: str = "",
) -> list[dict]:
    """Kept sentences from one transcript. The quote is the post's words."""
    emitted: list[dict] = []
    last_start: int | None = None
    for sentence in _SENTENCE.split(text or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        reason = _drop_reason(sentence, None)
        if reason == "retracted":
            if last_start is not None:
                del emitted[last_start:]
            last_start = None
            continue
        if reason in _HARD_DROP or _IF_MODAL.search(sentence):
            last_start = None
            continue
        start = len(emitted)
        emitted.extend(_rows_for(sentence, video_id, influencer, title, upload_date))
        last_start = start if len(emitted) > start else None
    kept: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in emitted:
        key = (
            row["company_ticker"],
            re.sub(r"\s+", " ", row["text"]).casefold(),
            row.get("measure"),
            row.get("expansion_rule"),
        )
        if key in seen:
            continue
        seen.add(key)
        row["sub_claim_id"] = f"{video_id}-{row['company_ticker']}-{len(kept) + 1}"
        kept.append(row)
    return kept


def build_claims(posts_dir: Path, out_path: Path | None = None) -> dict:
    """Walk stored videos in the project window and overwrite the claim table."""
    destination = out_path or claims_path(posts_dir)
    counts: Counter[str] = Counter()
    rows: list[dict] = []
    for post in load_posts(posts_dir):
        channel = str(post.get("influencer") or "")
        if channel not in _CHANNELS:
            counts["skipped_other_influencer"] += 1
            continue
        if not in_window(post):
            counts["skipped_outside_window"] += 1
            continue
        transcript = post.get("transcript") or ""
        if not transcript:
            counts["skipped_no_transcript"] += 1
            continue
        found = claims_from_transcript(
            transcript,
            video_id=str(post.get("video_id") or ""),
            influencer=channel,
            title=str(post.get("title") or ""),
            upload_date=str(post.get("upload_date") or ""),
        )
        counts["videos"] += 1
        counts["claims"] += len(found)
        for row in found:
            counts[f"type:{row['sub_claim_type']}"] += 1
        rows.extend(found)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    counts["path"] = 0
    return {"path": str(destination), "counts": counts, "rows": rows}


def _rows_for(sentence: str, video_id: str, influencer: str, title: str, upload_date: str) -> list[dict]:
    clauses = [part.strip(" ,") for part in re.split(r",\s+|;\s+", sentence) if part.strip()]
    typed = [(clause, kind) for clause in clauses if (kind := _classify(clause)) is not None]
    signatures = {(kind.type, kind.measure, kind.reason, kind.expansion) for _clause, kind in typed}
    pieces = _grouped(typed) if len(typed) >= 2 and len(signatures) > 1 else [(sentence, _classify(sentence))]
    rows = []
    for quote, kind in pieces:
        if kind is None or _ADVICE_OR_PRICE.search(quote):
            continue
        ticker = _subject(quote, _subject(sentence, None))
        if ticker is None:
            continue
        rows.extend(_materialize(quote, kind, ticker, video_id, influencer, title, upload_date))
    return rows


def _grouped(typed: list[tuple[str, "Kind"]]) -> list[tuple[str, "Kind"]]:
    groups: list[tuple[list[str], Kind]] = []
    for clause, kind in typed:
        if groups and groups[-1][1] == kind:
            groups[-1][0].append(clause)
        else:
            groups.append(([clause], kind))
    return [(", ".join(parts), kind) for parts, kind in groups]


def _subject(text: str, fallback: str | None) -> str | None:
    mentions = find_mentions(text)
    subject = subject_ticker(text, mentions)
    if subject in COMPANY_NAMES and subject != "CMPX":
        return subject
    if not mentions and fallback in COMPANY_NAMES and fallback != "CMPX":
        return fallback
    named = [ticker for ticker, _surface, _start in mentions if ticker != "CMPX"]
    if len(dict.fromkeys(named)) == 1:
        return named[0]
    return None


class Kind:
    def __init__(
        self,
        claim_type: str,
        measure: str | None = None,
        reason: str | None = None,
        expansion: str | None = None,
    ):
        self.type = claim_type
        self.measure = measure
        self.reason = reason
        self.expansion = expansion

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Kind) and (self.type, self.measure, self.reason, self.expansion) == (
            other.type, other.measure, other.reason, other.expansion,
        )


def _classify(text: str) -> Kind | None:
    if _ADVICE_OR_PRICE.search(text):
        return None
    if _GARBLED.search(text):
        return Kind("OTHER", reason=GARBLED_REASON)
    if _SEGMENT.search(text) and _AS_COMPANY.search(text):
        return Kind("OTHER", reason=SEGMENT_COMPANY_REASON)
    if _SEGMENT.search(text) and (_CHECKABLE.search(text) or _NUMBER.search(text)):
        return Kind("OTHER", measure="segment")
    for pattern, rule in _WIDE:
        if pattern.search(text) and not _NAMED_MEASURE.search(text):
            return Kind("PROFITABILITY_CASH_LEVERAGE", expansion=rule)
    if _FORWARD.search(text):
        return Kind("CONTINUATION", measure="guidance" if re.search(r"\bguid", text, flags=re.IGNORECASE) else None)
    if _COMPARE.search(text):
        return Kind("COMPARATIVE")
    if _RISK.search(text):
        return Kind("BUSINESS_RISK")
    if _CAUSE.search(text) and _RESULT.search(text) and not _NUMBER.search(text):
        return Kind("DEMAND_ATTRIBUTION")
    if _PROFIT.search(text):
        return Kind("PROFITABILITY_CASH_LEVERAGE")
    if _GROWTH_MEASURE.search(text) and (_DIRECTION.search(text) or _NUMBER.search(text)):
        return Kind("HISTORICAL_GROWTH")
    return None


def _materialize(
    quote: str,
    kind: Kind,
    ticker: str,
    video_id: str,
    influencer: str,
    title: str,
    upload_date: str,
) -> list[dict]:
    when = _quote_time(quote)
    base = {
        "video_id": video_id,
        "influencer": influencer,
        "title": title,
        "upload_date": upload_date,
        "company_ticker": ticker,
        "text": quote,
        "source_span": quote,
        "is_quantitative": bool(_NUMBER.search(quote)),
        "quote_time": when,
    }
    if kind.expansion:
        return [
            {
                **base,
                "sub_claim_type": kind.type,
                "measure": measure,
                "parent_span": quote,
                "expansion_rule": kind.expansion,
            }
            for measure in _MEASURES
        ]
    row = {**base, "sub_claim_type": kind.type, "measure": kind.measure}
    if kind.reason:
        row["insufficient_reason"] = kind.reason
    return [row]
