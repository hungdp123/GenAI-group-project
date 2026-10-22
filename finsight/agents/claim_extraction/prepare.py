"""Drop sentences that must not become a claim. Keep the quote of anything dropped.

Runs on one stored post, before the decomposer. A stored claim record is not
passed through here again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finsight.agents.claim_extraction.thesaurus import find_mentions, subject_ticker
from finsight.contracts.schemas import SubClaim, SubClaimType

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_QUESTION_START = re.compile(r"^(what|why|how|when|where|who)\b", flags=re.IGNORECASE)
_HYPOTHETICAL = re.compile(r"\bif\b.{0,80}\b(would|could|might)\b", flags=re.IGNORECASE)
_JOKE = re.compile(r"\b(just kidding|i'm joking|i am joking|this is a joke)\b", flags=re.IGNORECASE)
_NOT_HOST = re.compile(
    r"\banalysts?\s+(say|said|expect|see|think|have|had)\b"
    r"|\baccording to (the )?(analysts|street|wall street)\b"
    r"|\b(this (video|episode) is )?sponsored\b"
    r"|\b(quote tweet|quoted post|as posted by|reposted by)\b",
    flags=re.IGNORECASE,
)
_RETRACT = re.compile(
    r"\b(take that back|scratch that|i was wrong|let me correct|i misspoke)\b",
    flags=re.IGNORECASE,
)
_GUEST_LINE = re.compile(r"^\s*guest\s*:", flags=re.IGNORECASE)
_SPEAKER_MARK = re.compile(r"(?m)^\s*(host|guest)\s*:")
_THEY = re.compile(r"\b(they|their|them)\b", flags=re.IGNORECASE)
_CHECKABLE = re.compile(
    r"\b(revenue|sales|grew|growth|margin|profit|earnings|cash|debt|income|guidance)\b",
    flags=re.IGNORECASE,
)
_GARBLED = re.compile(r"\$\d*[IlO]\d|\b\d+\s*%\s*%")
_TIME = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")
_SEGMENT_WORD = re.compile(
    r"\b(segment|division|aws|azure|instagram|waymo|iphone|google cloud|youtube)\b",
    flags=re.IGNORECASE,
)
_AS_COMPANY = re.compile(
    r"\b(whole company|company-wide|as a whole|overall|the company('s)? (revenue|sales|growth|grew))\b",
    flags=re.IGNORECASE,
)

GARBLED_REASON = "The number or ticker in this sentence is unclear, so it is not treated as contradicted."
SEGMENT_COMPANY_REASON = (
    "A segment figure was presented as the whole company, so it is not checked as the company's result."
)

SPEAKER_NOTE = "The transcript does not mark who is speaking, so the speaker was not separated."
SCREEN_NOTE = (
    "A figure that appears only on a chart, a slide, or an image is not in the stored text, "
    "so it was not checked."
)
NO_TIME_NOTE = "This transcript has no timestamps, so the report cannot point to a moment in the video."


@dataclass
class Prepared:
    text: str
    dropped: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    held: list[SubClaim] = field(default_factory=list)
    insufficient_reasons: dict[str, str] = field(default_factory=dict)


def post_limits(transcript: str, video_id: str | None) -> str:
    """Limits that belong on the report for every stored video."""
    sentences = [SCREEN_NOTE]
    if not _SPEAKER_MARK.search(transcript or ""):
        sentences.append(SPEAKER_NOTE)
    if video_id:
        sentences.append(f"Video: https://www.youtube.com/watch?v={video_id}")
    if not _TIME.search(transcript or ""):
        sentences.append(NO_TIME_NOTE)
    return " ".join(sentences)


def prepare_post(text: str, ticker: str | None) -> Prepared:
    """Sentences the decomposer may see, plus quotes that were left out."""
    out = Prepared(text="")
    kept: list[str] = []
    held_quotes: list[tuple[str, str, str | None]] = []
    last: str | None = None
    for sentence in _SENTENCE.split(text or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        reason = _drop_reason(sentence, ticker)
        if reason == "retracted":
            if last == "kept" and kept:
                out.dropped.append(("retracted", kept.pop()))
            elif last == "held" and held_quotes:
                _kind, quote, _when = held_quotes.pop()
                out.dropped.append(("retracted", quote))
            out.dropped.append(("retracted", sentence))
            last = None
            continue
        if reason:
            out.dropped.append((reason, sentence))
            last = None
            continue
        hold = _hold_reason(sentence)
        if hold:
            held_quotes.append((hold, sentence, _quote_time(sentence)))
            last = "held"
            continue
        kept.append(sentence)
        last = "kept"
    out.text = " ".join(kept).strip()
    for index, (kind, quote, when) in enumerate(held_quotes, start=1):
        claim_id = f"held-{index}"
        measure = "segment" if kind == "segment" else None
        out.held.append(SubClaim(
            sub_claim_id=claim_id,
            text=quote,
            sub_claim_type=SubClaimType.OTHER,
            company_ticker=ticker,
            source_span=quote,
            measure=measure,
            quote_time=when,
        ))
        if kind == "garbled":
            out.insufficient_reasons[claim_id] = GARBLED_REASON
        elif kind == "segment_company":
            out.insufficient_reasons[claim_id] = SEGMENT_COMPANY_REASON
    if out.dropped:
        out.notes.append(f"left out {len(out.dropped)} sentence(s)")
    return out


def _drop_reason(sentence: str, ticker: str | None) -> str | None:
    if _GUEST_LINE.search(sentence):
        return "guest"
    if _NOT_HOST.search(sentence):
        return "not the host"
    if sentence.endswith("?") or _QUESTION_START.search(sentence):
        return "question"
    if _HYPOTHETICAL.search(sentence):
        return "hypothetical"
    if _JOKE.search(sentence):
        return "joke"
    if _RETRACT.search(sentence):
        return "retracted"
    mentions = find_mentions(sentence)
    tickers = list(dict.fromkeys(item[0] for item in mentions))
    if len(tickers) >= 3 and not _CHECKABLE.search(sentence):
        return "list mention"
    if tickers and ticker and ticker not in tickers:
        return "other company"
    if not tickers and _THEY.search(sentence):
        return "unclear they"
    if len(tickers) > 1 and ticker and subject_ticker(sentence, mentions) not in (None, ticker):
        return "other company"
    return None


def _hold_reason(sentence: str) -> str | None:
    """A sentence kept as its own claim, not handed to the company-wide calculator."""
    if _GARBLED.search(sentence):
        return "garbled"
    if _SEGMENT_WORD.search(sentence) and _AS_COMPANY.search(sentence):
        return "segment_company"
    if _SEGMENT_WORD.search(sentence) and _CHECKABLE.search(sentence):
        return "segment"
    return None


def _quote_time(sentence: str) -> str | None:
    match = _TIME.search(sentence)
    return match.group(1) if match else None
