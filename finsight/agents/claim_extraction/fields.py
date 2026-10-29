"""Turn a prompt or a spoken sentence into fields, then compare the fields.

Company, metric, direction, value, and unit come from alias tables and a number
parser. A line passes only when those fields agree. BM25 then orders the lines
that passed. No model is called.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from finsight.agents.claim_extraction.thesaurus import subject_ticker

_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", flags=re.IGNORECASE)
_MAGNITUDE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?:plus\s+)?(billion|million)\b",
    flags=re.IGNORECASE,
)
_DOLLARS = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)")
_PERIOD = re.compile(
    r"\b(?:q[1-4]\s*(?:fy\s*)?\d{2,4}|fiscal\s+q[1-4](?:\s+\d{4})?|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{4}|last quarter|this quarter|most recent quarter|trailing 12[- ]months?)\b",
    flags=re.IGNORECASE,
)
_TICKER_CUE = re.compile(r"\bstock ticker\s+([A-Za-z]{1,5})\b", flags=re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")

# A person named in a video is that company until another company is named.
_PEOPLE: tuple[tuple[str, str], ...] = (
    ("marc benioff", "CRM"),
    ("benioff", "CRM"),
    ("andy jassy", "AMZN"),
    ("satya nadella", "MSFT"),
    ("tim cook", "AAPL"),
    ("mark zuckerberg", "META"),
    ("jensen huang", "NVDA"),
)

_METRICS: dict[str, str] = {
    "revenue": "revenue", "revenues": "revenue", "sales": "revenue", "topline": "revenue",
    "profit": "profit", "profits": "profit", "earnings": "profit", "income": "profit",
    "margin": "margin", "margins": "margin", "profitability": "margin",
    "cash": "cash", "fcf": "cash",
    "debt": "debt", "leverage": "leverage",
}
_PHRASES: tuple[tuple[str, str], ...] = (
    ("debt-to-equity", "leverage"),
    ("debt to equity", "leverage"),
    ("free cash flow", "cash"),
    ("operating cash flow", "cash"),
    ("operating profit", "profit"),
    ("operating profits", "profit"),
    ("operating margin", "margin"),
    ("operating margins", "margin"),
    ("net income", "profit"),
    ("top line", "revenue"),
)
_UP = frozenset({
    "grew", "grow", "growth", "grown", "rising", "rise", "rose", "risen",
    "increase", "increased", "increasing", "higher", "up", "double", "doubled",
    "expand", "expanded", "expanding", "improved", "improving", "strong", "stronger",
})
# "leading to higher revenues" names a consequence. It is not the reported move.
_LEADING_TO = re.compile(r"\b(?:leading|leads|lead|led)\s+to\b", flags=re.IGNORECASE)
_RECORD_HIGH = re.compile(r"\b(?:all[- ]time|record)\s+highs?\b", flags=re.IGNORECASE)
_DOWN = frozenset({
    "fell", "fall", "fallen", "falling", "decline", "declined", "declining",
    "drop", "dropped", "down", "lower", "decrease", "decreased", "shrink", "shrank",
})
_FILLER = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "by", "from",
    "that", "this", "its", "it", "is", "are", "was", "were", "be", "been", "has", "have",
    "had", "at", "as", "we", "you", "our", "their", "his", "her",
})


@dataclass(frozen=True)
class ClaimFields:
    company: str | None
    metrics: frozenset[str]
    direction: str | None
    values: tuple[tuple[float, str], ...]
    period: str | None

    def as_dict(self) -> dict:
        return {
            "company": self.company,
            "metrics": sorted(self.metrics),
            "direction": self.direction,
            "values": [{"value": value, "unit": unit} for value, unit in self.values],
            "period": self.period,
        }


def parse_fields(text: str, company: str | None = None) -> ClaimFields:
    """Fields stated in this text. `company` fills in when the sentence does not name one."""
    raw = text or ""
    named = subject_ticker(raw)
    period_match = _PERIOD.search(raw)
    return ClaimFields(
        company=named or company,
        metrics=frozenset(_metrics(raw)),
        direction=_direction(raw),
        values=tuple(_values(raw)),
        period=period_match.group(0) if period_match else None,
    )


def fields_match(
    prompt: ClaimFields, candidate: ClaimFields, *, strict_value: bool = True, metric_required: bool = True,
) -> bool:
    """True when the candidate is the same claim. A stated 24% does not match 17%.

    `metric_required` is turned off only after no line names the measure. A line that names a
    different measure still fails. "22% is massive" can then match "revenue grew 22%".
    """
    if not (prompt.metrics or prompt.values or prompt.direction):
        return False
    if prompt.company and candidate.company and prompt.company != candidate.company:
        return False
    if prompt.direction and candidate.direction and candidate.direction != prompt.direction:
        return False
    shares_metric = bool(candidate.metrics and prompt.metrics <= candidate.metrics)
    if prompt.metrics and metric_required and not shares_metric:
        return False
    if prompt.metrics and not metric_required and candidate.metrics and not shares_metric:
        return False
    if strict_value and prompt.values and not _values_match(prompt.values, candidate.values):
        return False
    if not strict_value and prompt.metrics and not shares_metric:
        return False
    return True


def company_source(sentence: str, company: str | None) -> str:
    """How this sentence got its company: named, carried, person, or none."""
    if _unknown_ticker(sentence):
        return "none"
    if subject_ticker(sentence):
        return "named"
    if _person(sentence):
        return "person"
    if company:
        return "carried"
    return "none"


def company_for_sentences(sentences: list[str]) -> list[str | None]:
    """The company in force for each sentence. It holds until another company is named.

    "Stock ticker CELH" closes the previous company, because CELH is not in the table.
    Marc Benioff opens Salesforce.
    """
    current: str | None = None
    assigned: list[str | None] = []
    for sentence in sentences:
        if _unknown_ticker(sentence):
            current = None
            assigned.append(None)
            continue
        named = subject_ticker(sentence)
        person = _person(sentence)
        if named:
            current = named
        elif person:
            current = person
        assigned.append(current)
    return assigned


def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """BM25 over stemmed tokens. A short fragment does not outrank a full sentence."""
    if not documents:
        return []
    query_terms = list(dict.fromkeys(_stems(query)))
    tokenized = [_stems(document) for document in documents]
    if not query_terms:
        return [0.0] * len(documents)
    average = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    document_frequency = {
        term: sum(term in set(tokens) for tokens in tokenized) for term in query_terms
    }
    k1, b = 1.2, 0.75
    scores: list[float] = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(tokens) / average)
            score += inverse * (frequency * (k1 + 1) / denominator)
        scores.append(score)
    return scores


def _metrics(text: str) -> set[str]:
    found: set[str] = set()
    lowered = text.lower()
    for phrase, metric in _PHRASES:
        if phrase in lowered:
            found.add(metric)
            lowered = lowered.replace(phrase, " ")
    for word in _WORD.findall(lowered):
        metric = _METRICS.get(word)
        if metric:
            found.add(metric)
    return found


def _direction(text: str) -> str | None:
    """The move the sentence reports. Words after "leading to" are a consequence, not that move.

    "Revenues are at all-time highs" is up, even though the sentence never says "higher".
    """
    reported = _LEADING_TO.split(text or "", maxsplit=1)[0]
    words = set(_WORD.findall(reported.lower()))
    up = bool(words & _UP) or bool(_RECORD_HIGH.search(reported))
    down = bool(words & _DOWN)
    if up and not down:
        return "up"
    if down and not up:
        return "down"
    return None


def _values(text: str) -> list[tuple[float, str]]:
    values: list[tuple[float, str]] = []
    for match in _PERCENT.findall(text or ""):
        values.append((float(match), "percent"))
    for match in _MAGNITUDE.finditer(text or ""):
        number = float(match.group(1).replace(",", ""))
        if match.group(2).lower().startswith("mill"):
            number = number / 1000.0
        values.append((number, "billion"))
    for match in _DOLLARS.finditer(text or ""):
        number = float(match.group(1).replace(",", ""))
        if number >= 100:
            values.append((number, "dollar"))
    return values


def _values_match(
    prompt: tuple[tuple[float, str], ...],
    candidate: tuple[tuple[float, str], ...],
) -> bool:
    for number, unit in prompt:
        if any(_same_value(number, unit, other, other_unit) for other, other_unit in candidate):
            continue
        # A spoken "$110" in a revenue line is the same figure as "$110 billion".
        if unit == "billion" and any(
            other_unit == "dollar" and _close(number, other, False) for other, other_unit in candidate
        ):
            continue
        return False
    return True


def _same_value(left: float, left_unit: str, right: float, right_unit: str) -> bool:
    if left_unit != right_unit:
        return False
    return _close(left, right, left_unit == "percent")


def _close(left: float, right: float, percent: bool) -> bool:
    difference = abs(left - right)
    scale = max(left, right)
    if percent:
        allowance = 0.5 if scale < 50 else max(2.0, 0.10 * scale)
    else:
        allowance = max(0.6, 0.08 * scale)
    return difference <= allowance


def _person(sentence: str) -> str | None:
    lowered = sentence.lower()
    for name, ticker in _PEOPLE:
        if name in lowered:
            return ticker
    return None


def _unknown_ticker(sentence: str) -> bool:
    match = _TICKER_CUE.search(sentence or "")
    if not match:
        return False
    return subject_ticker(match.group(1)) is None


def _stems(text: str) -> list[str]:
    stems: list[str] = []
    for word in _WORD.findall((text or "").lower()):
        if word in _FILLER or word in _METRICS or word in _UP or word in _DOWN:
            canon = _METRICS.get(word) or ("up" if word in _UP else "down" if word in _DOWN else None)
            if canon:
                stems.append(canon)
            continue
        if word.endswith("s") and len(word) > 4:
            word = word[:-1]
        stems.append(word)
    return stems
