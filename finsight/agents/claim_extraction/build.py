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
_TYPE_BY_MEASURE = {
    "profitability": "PROFITABILITY",
    "cash_generation": "CASH_GENERATION",
    "liquidity": "LIQUIDITY",
    "leverage": "LEVERAGE",
}
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
_MARGIN = re.compile(
    r"\b(gross margins?|operating margins?|net margins?|profit margins?|margins?)\b",
    flags=re.IGNORECASE,
)
_CASH = re.compile(r"\b(free cash flow|cash flow|fcf|cash generation)\b", flags=re.IGNORECASE)
_LIQUIDITY = re.compile(r"\b(liquidity|current ratio|cash ratio)\b", flags=re.IGNORECASE)
_CASH_RATIO = re.compile(r"\bcash ratio\b", flags=re.IGNORECASE)
_FCF_MARGIN = re.compile(r"\b(free cash flow|fcf) margins?\b", flags=re.IGNORECASE)
_FREE_CASH = re.compile(r"\bfree cash flow\b|\bfcf\b", flags=re.IGNORECASE)
_LEVERAGE = re.compile(r"\b(leverage|debt[- ]to[- ]equity|debt/equity)\b", flags=re.IGNORECASE)
_DEBT_AMOUNT = re.compile(r"\b(debt|indebted|balance sheet)\b", flags=re.IGNORECASE)
_REVENUE_WORD = re.compile(r"\b(revenues?|sales|top[- ]line)\b", flags=re.IGNORECASE)
_NET_INCOME_WORD = re.compile(r"\b(net income|earnings|profits?|bottom line)\b", flags=re.IGNORECASE)
_GROWTH_MEASURE = re.compile(
    r"\b(revenues?|sales|earnings|profits?|net income|top line)\b",
    flags=re.IGNORECASE,
)
_DIRECTION = re.compile(
    r"\b(grew|growth|rose|risen|doubled|fell|fallen|decline|declined|declining|"
    r"expanded|expanding|improved|improving|strong|weak|solid|healthy|higher|lower)\b"
    r"|\b(up|down)\b|shot up|year[- ]over[- ]year|all[- ]time highs?",
    flags=re.IGNORECASE,
)
_OPERATING_INCOME = re.compile(r"\boperating (income|profits?)\b", flags=re.IGNORECASE)
_EPS = re.compile(r"\bearnings per share\b|\beps\b", flags=re.IGNORECASE)
_GROSS_PROFIT = re.compile(r"\bgross profits?\b", flags=re.IGNORECASE)
_EXCLUDING_GAINS = re.compile(
    r"\b(exclud\w+|adjusted|non-gaap)\b.{0,60}\b(gains?|income|earnings|profits?)\b"
    r"|\b(income|earnings|profits?)\b.{0,60}\b(exclud\w+|adjusted|non-gaap)\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_RUN_RATE = re.compile(r"\brun[- ]rate\b|\bannuali[sz]", flags=re.IGNORECASE)
_SERVICES_MARGIN = re.compile(r"\bservices margin\b", flags=re.IGNORECASE)
_LEAVE_TYPE = frozenset({
    "DEMAND_ATTRIBUTION", "CONTINUATION", "BUSINESS_RISK", "COMPARATIVE", "PERSONALIZED_ADVICE",
})
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


def claims_catalog_path(posts_dir: Path) -> Path:
    """The field catalog when it has been built, otherwise the original claim list."""
    claims_dir = posts_dir.parent / "claims"
    versioned = claims_dir / "claim_update_v3.json"
    if versioned.exists():
        return versioned
    updated = claims_dir / "claims_updated.jsonl"
    if updated.exists():
        return updated
    return claims_path(posts_dir)


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


_PRICE_TALK = re.compile(
    r"\b(share price|stock price|price target|p\s*/\s*e\b|peg\b|"
    r"price\s*[- ]\s*to\s*[- ]\s*(?:earnings|sales))\b",
    flags=re.IGNORECASE,
)
_FORECAST_LINE = re.compile(r"\b(forecasts?|will report|forward estimates?)\b", flags=re.IGNORECASE)
_VERSUS = re.compile(r"\b(versus|vs\.?|compared with|compared to)\b", flags=re.IGNORECASE)
_CLAIM_SPLIT = re.compile(r",\s+|;\s+|\band\b|\bwith\b|\bgiving them\b", flags=re.IGNORECASE)
_LEAD_IN = re.compile(r"^(?:and|but)\s+", flags=re.IGNORECASE)
# Longer phrases first, so "operating margin" is not also counted as "margin" or "profit".
_PIECE_MEASURES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("fcf_margin", re.compile(r"\b(?:free cash flow|fcf) margins?\b", flags=re.IGNORECASE)),
    ("operating_margin", re.compile(r"\boperating margins?\b", flags=re.IGNORECASE)),
    ("gross_margin", re.compile(r"\bgross margins?\b", flags=re.IGNORECASE)),
    ("net_margin", re.compile(r"\b(?:net|profit) margins?\b", flags=re.IGNORECASE)),
    ("cash_ratio", re.compile(r"\bcash ratio\b", flags=re.IGNORECASE)),
    ("current_ratio", re.compile(r"\bcurrent ratio\b", flags=re.IGNORECASE)),
    ("operating_cash_flow", re.compile(r"\b(?:operating cash flow|cash from operations)\b", flags=re.IGNORECASE)),
    ("free_cash_flow", re.compile(r"\bfree cash flow\b|\bfcf\b", flags=re.IGNORECASE)),
    ("operating_income", re.compile(r"\boperating (?:income|profits?)\b|\bebit\b", flags=re.IGNORECASE)),
    ("eps", re.compile(r"\bearnings per share\b|\beps\b", flags=re.IGNORECASE)),
    ("gross_profit", re.compile(r"\bgross profits?\b", flags=re.IGNORECASE)),
    ("net_income", re.compile(r"\bnet income\b|\bnet profits?\b", flags=re.IGNORECASE)),
    ("revenue", re.compile(r"\b(?:revenues?|sales|top[- ]line)\b", flags=re.IGNORECASE)),
    ("profit", re.compile(r"\b(?:profits?|earnings|bottom line)\b", flags=re.IGNORECASE)),
    ("margin", re.compile(r"\bmargins?\b", flags=re.IGNORECASE)),
)
_MEASURE_KIND = {
    "revenue": ("HISTORICAL_GROWTH", "revenue"),
    "operating_income": ("HISTORICAL_GROWTH", "operating_income"),
    "eps": ("HISTORICAL_GROWTH", "diluted_eps"),
    "net_income": ("NET_INCOME_GROWTH", "net_income"),
    "profit": ("NET_INCOME_GROWTH", "net_income"),
    "operating_margin": ("PROFITABILITY", "operating_margin"),
    "gross_margin": ("PROFITABILITY", "gross_margin"),
    "net_margin": ("PROFITABILITY", "net_margin"),
    "margin": ("PROFITABILITY", None),
    "fcf_margin": ("CASH_GENERATION", "fcf_margin"),
    "free_cash_flow": ("CASH_GENERATION", "free_cash_flow"),
    "operating_cash_flow": ("CASH_GENERATION", "operating_cash_flow"),
    "gross_profit": ("OTHER", "gross_profit"),
    "cash_ratio": ("LIQUIDITY", "cash_ratio"),
    "current_ratio": ("LIQUIDITY", "current_ratio"),
}
_SPECIFIC_MARGIN = frozenset({"operating_margin", "gross_margin", "net_margin", "fcf_margin"})
_SPECIFIC_PROFIT = frozenset({"operating_income", "eps", "net_income"})


def _piece_measure(clause: str) -> str | None:
    for name, pattern in _PIECE_MEASURES:
        if pattern.search(clause):
            return name
    return None


def _keep_whole(text: str) -> bool:
    """A forecast, a comparison, a price remark, or a segment line stays one claim."""
    blocked = (
        _FORWARD.search(text) or _FORECAST_LINE.search(text) or _PRICE_TALK.search(text)
        or _SEGMENT.search(text) or _VERSUS.search(text)
    )
    if blocked:
        return True
    named = [ticker for ticker, _surface, _start in find_mentions(text) if ticker != "CMPX"]
    return len(dict.fromkeys(named)) >= 2


def _collapse_measure(measure: str, chosen: list[str]) -> str | None:
    if measure == "margin" and _SPECIFIC_MARGIN.intersection(chosen):
        return None
    if measure == "profit" and _SPECIFIC_PROFIT.intersection(chosen):
        return None
    return measure


def _kind_for_piece(measure: str, quote: str) -> Kind:
    if _no_financial_formula(quote) and measure in {"eps", "net_income", "profit", "gross_profit"}:
        return Kind("OTHER", measure=measure)
    if measure == "operating_cash_flow" and _ocf_growth(quote):
        return Kind("HISTORICAL_GROWTH", measure="operating_cash_flow")
    claim_type, stored_measure = _MEASURE_KIND[measure]
    return Kind(claim_type, measure=stored_measure)


def split_financial_measures(text: str) -> list[tuple[str, Kind]]:
    """One claim per stated result. A trailing figure with no new measure stays on the previous piece.

    Returns an empty list when the sentence should stay whole: one measure, a forecast,
    a comparison, a share-price remark, or a segment line.
    """
    if not text or _keep_whole(text):
        return []
    groups: list[tuple[str, list[str]]] = []
    for raw in _CLAIM_SPLIT.split(text):
        part = _LEAD_IN.sub("", raw).strip(" ,.;")
        if not part:
            continue
        measure = _piece_measure(part)
        if measure is None or not _moving(part):
            if groups:
                groups[-1][1].append(part)
            continue
        measure = _collapse_measure(measure, [name for name, _parts in groups])
        if measure is None:
            groups[-1][1].append(part)
            continue
        if groups and groups[-1][0] == measure:
            groups[-1][1].append(part)
            continue
        groups.append((measure, [part]))
    if len({name for name, _parts in groups}) < 2:
        return []
    return [(", ".join(parts), _kind_for_piece(name, ", ".join(parts))) for name, parts in groups]


def expand_stored_row(row: dict) -> list[dict]:
    """Replace one stored sentence with one row per measure. An unsplit row is returned unchanged."""
    if row.get("expansion_rule") or str(row.get("sub_claim_type") or "") in _LEAVE_TYPE:
        return [row]
    text = str(row.get("text") or row.get("source_span") or "")
    pieces = split_financial_measures(text)
    if len(pieces) < 2:
        return [row]
    from finsight.agents.claim_extraction.fields import company_source, parse_fields

    base_id = str(row.get("sub_claim_id") or "c")
    ticker = row.get("company_ticker")
    expanded = []
    for quote, kind in pieces:
        slug = kind.measure or kind.type.lower()
        piece = dict(row)
        piece.update({
            "text": quote,
            "source_span": quote,
            "parent_span": text,
            "sub_claim_type": kind.type,
            "measure": kind.measure,
            "sub_claim_id": f"{base_id}-{slug}",
            "is_quantitative": bool(_NUMBER.search(quote)),
            "quote_time": _quote_time(quote),
            "fields": parse_fields(quote, company=ticker).as_dict(),
            "company_source": company_source(quote, ticker),
        })
        expanded.append(piece)
    return expanded


def _rows_for(sentence: str, video_id: str, influencer: str, title: str, upload_date: str) -> list[dict]:
    measured = split_financial_measures(sentence)
    if len(measured) >= 2:
        rows = []
        for quote, kind in measured:
            if _ADVICE_OR_PRICE.search(quote):
                continue
            ticker = _subject(quote, _subject(sentence, None))
            if ticker is None:
                continue
            rows.extend(_materialize(quote, kind, ticker, video_id, influencer, title, upload_date))
            for extra in _extra_kinds(quote, kind):
                rows.extend(_materialize(quote, extra, ticker, video_id, influencer, title, upload_date))
        if rows:
            return rows
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
        for extra in _extra_kinds(quote, kind):
            rows.extend(_materialize(quote, extra, ticker, video_id, influencer, title, upload_date))
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
            return Kind("PROFITABILITY", expansion=rule)
    if _FORWARD.search(text):
        return Kind("CONTINUATION", measure="guidance" if re.search(r"\bguid", text, flags=re.IGNORECASE) else None)
    if _COMPARE.search(text):
        return Kind("COMPARATIVE")
    if _RISK.search(text):
        return Kind("BUSINESS_RISK")
    if _CAUSE.search(text) and _RESULT.search(text) and not _NUMBER.search(text):
        return Kind("DEMAND_ATTRIBUTION")
    if _LEVERAGE.search(text):
        return Kind("LEVERAGE")
    if _DEBT_AMOUNT.search(text):
        return Kind("OTHER")
    if _CASH_RATIO.search(text):
        return Kind("LIQUIDITY")
    if _LIQUIDITY.search(text):
        return Kind("LIQUIDITY")
    if _FCF_MARGIN.search(text):
        return Kind("CASH_GENERATION", measure="fcf_margin")
    if _ocf_growth(text):
        return Kind("HISTORICAL_GROWTH", measure="operating_cash_flow")
    if _CASH.search(text):
        return Kind("CASH_GENERATION")
    if _no_financial_formula(text):
        if _SERVICES_MARGIN.search(text):
            return Kind("OTHER", measure="segment")
        return Kind("OTHER")
    if _MARGIN.search(text):
        return Kind("PROFITABILITY")
    if _operating_income_growth(text):
        return Kind("HISTORICAL_GROWTH", measure="operating_income")
    if _eps_growth(text):
        return Kind("HISTORICAL_GROWTH", measure="diluted_eps")
    if _GROWTH_MEASURE.search(text) and _moving(text):
        if _NET_INCOME_WORD.search(text) and not _REVENUE_WORD.search(text):
            return Kind("NET_INCOME_GROWTH")
        return Kind("HISTORICAL_GROWTH")
    return None


def _moving(text: str) -> bool:
    return bool(_DIRECTION.search(text) or _NUMBER.search(text))


def _operating_income_growth(text: str) -> bool:
    return bool(_OPERATING_INCOME.search(text) and _moving(text))


def _eps_growth(text: str) -> bool:
    return bool(_EPS.search(text) and _moving(text))


def _ocf_growth(text: str) -> bool:
    """Operating cash flow changed. Free cash flow stays a cash-generation claim."""
    if _FREE_CASH.search(text) or not _moving(text):
        return False
    return bool(re.search(r"\b(operating cash flow|cash from operations|cash flow)\b", text, flags=re.IGNORECASE))


def _no_financial_formula(text: str) -> bool:
    """A figure the financial agent cannot calculate. Routing stays with the narrative agent."""
    if _RUN_RATE.search(text) or _EXCLUDING_GAINS.search(text) or _SERVICES_MARGIN.search(text):
        return True
    return bool(_GROSS_PROFIT.search(text) and not _MARGIN.search(text))


def _extra_kinds(text: str, kind: Kind) -> list[Kind]:
    """A margin sentence can also name operating-income or EPS growth. Keep both."""
    if kind.expansion or kind.type == "HISTORICAL_GROWTH" or _no_financial_formula(text):
        return []
    extras: list[Kind] = []
    if _operating_income_growth(text):
        extras.append(Kind("HISTORICAL_GROWTH", measure="operating_income"))
    if _eps_growth(text):
        extras.append(Kind("HISTORICAL_GROWTH", measure="diluted_eps"))
    if _ocf_growth(text):
        extras.append(Kind("HISTORICAL_GROWTH", measure="operating_cash_flow"))
    return extras


def specific_type(stored: str, text: str, measure: str | None) -> str:
    """Split an older merged profitability row into the measure it actually names.

    Operating-income growth and diluted EPS growth use the financial agent.
    A claim with no formula there stays with the narrative agent.
    """
    if measure in _TYPE_BY_MEASURE:
        return _TYPE_BY_MEASURE[measure]
    if stored not in _LEAVE_TYPE:
        if _no_financial_formula(text):
            return "OTHER"
        # A margin row stays a margin row. The growth line is added beside it.
        if stored != "PROFITABILITY" and (_operating_income_growth(text) or _eps_growth(text) or _ocf_growth(text)):
            return "HISTORICAL_GROWTH"
        if _CASH_RATIO.search(text):
            return "LIQUIDITY"
        if _FCF_MARGIN.search(text):
            return "CASH_GENERATION"
    if stored != "PROFITABILITY_CASH_LEVERAGE":
        return stored
    kind = _classify(text)
    if kind is None or kind.expansion:
        return stored
    return kind.type


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
                "sub_claim_type": _TYPE_BY_MEASURE[measure],
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
