""" Claim reader: what a post line says in financial terms

Turns the post line (and the user's prompt) into a list of `Point`s: one per financial figure the agent should
report. No data is looked up here, and no model is called: the rules below are plain Python, so the same words
always give the same points. `agent.py` computes each point from the SEC figures.

How a line is read
  1. Measures: every mention of a measure the agent calculates (revenue, operating income, net income,
     operating cash flow, diluted EPS, the three margins, free cash flow, free cash flow margin, current ratio,
     cash ratio, debt to equity). Valuation and event words are ignored ("price-to-sales", "reports earnings
     on Tuesday", "taking profits"). "Google grew 24%" with no measure word means revenue.
  2. Company: a measure belongs to the company named last before it. A measure after another company's name
     ("while Tesla's revenue fell, Microsoft ...", "Nebius generated $582 million in revenue") is skipped.
  3. Left to the narrative agent (knowledge_layer.yaml fallback), each with a note: a segment or product
     (AWS revenue, data center growth, Services margin), a run rate, gross profit, EBITDA and other adjusted
     figures, net income excluding gains, a debt amount, and forecasts ("earnings are set to grow 75% next year").
  4. The prompt: when the user's prompt names measures, only those points are kept ("Only the earnings part is
     checked"). A prompt about a segment keeps nothing.
  5. Period, in this order: the prompt's period, the period next to the measure, the line's period,
     Claim extraction's period field, "is / are / now" on a margin or ratio (= latest quarter),
     else none (= latest fiscal year, rule book section 11).
  6. Phrases about the whole company ("very strong financial position") expand into the four rule-book
     checks: operating margin, free cash flow, current ratio, debt to equity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from finsight.agents.claim_extraction.thesaurus import COMPANY_NAMES, PRODUCTS, find_mentions
except ImportError:   # pragma: no cover - thesaurus not in this checkout
    COMPANY_NAMES, PRODUCTS = {}, {}

    def find_mentions(text, include_products=True):
        return []

GROWTH_MEASURES = ("Revenue", "OperatingIncome", "NetIncome", "OperatingCashFlow", "DilutedEPS")
LEVEL_MEASURES = ("gross_margin", "operating_margin", "net_margin", "free_cash_flow", "fcf_margin",
                  "current_ratio", "cash_ratio")
LEVERAGE = "leverage"
GENERIC_MARGINS = ["operating_margin", "gross_margin"]       # "margins" with no type
PROFITABILITY = ["operating_margin", "net_margin"]           # "profitability" (rule book section 11)
EXPANSION = ["operating_margin", "free_cash_flow", "current_ratio", LEVERAGE]   # rule book section 6
FAMILIES = {"profitability": {"gross_margin", "operating_margin", "net_margin"},
            "cash_generation": {"free_cash_flow", "OperatingCashFlow", "fcf_margin"},
            "liquidity": {"current_ratio", "cash_ratio"}, "leverage": {LEVERAGE}}

_NON_GAAP = "a non-GAAP or adjusted figure is not in the SEC data; the narrative agent checks it"
_GROSS_PROFIT = "gross profit is not a calculated measure; the narrative agent checks it"
_RUN_RATE = "a run rate is not a reported figure; the narrative agent checks it"
_DEBT = "a debt amount is not calculated (debt is not mapped in the SEC data)"
_FORECAST = "a forecast; past filings cannot check it (the narrative agent checks guidance)"
_EXCLUDING = "a figure that removes gains or one-time items is not the reported figure; the narrative agent checks it"
SEGMENT_NOTE = "a segment or product figure; the SEC data only has company-wide totals, the narrative agent checks it"
OTHER_COMPANY_NOTE = "about another company"

# (pattern, measures | "ignore" | "excluded:<note>"). Checked in order; a match hides its words from later rules,
# so "free cash flow" is not also read as "cash flow".
_LEXICON = [
    (r"price[- ]to[- ](?:sales|earnings|revenue)s?|p/?e\b|(?:\d+(?:\.\d+)?\s*x|times)\s+(?:sales|revenue|earnings)"
     r"|(?:sales|revenue|earnings) multiples?|earnings multiples?", "ignore"),
    (r"earnings (?:calls?|reports?|releases?|dates?|season|day|beats?|miss(?:es)?|previews?|announcements?)"
     r"|(?:report|reports|reported|reporting) (?:their |its |q[1-4] )?earnings|(?:after|before|ahead of) earnings"
     r"|(?:taking|take|took) profits|profit[- ]taking|operating leverage|a quarter of", "ignore"),
    (r"(?:free cash flow|fcf) margins?", ["fcf_margin"]),
    (r"free cash flows?|\bfcf\b", ["free_cash_flow"]),
    (r"operating cash flows?|cash flows? from operations|cash from operations", ["OperatingCashFlow"]),
    (r"cash generation|cash flows?", ["free_cash_flow"]),
    (r"ebitda(?: margins?)?|ebit\b|adjusted (?:earnings|net income|profits?|eps|operating income|margins?)"
     r"|non-gaap \w+", "excluded:" + _NON_GAAP),
    (r"gross margins?", ["gross_margin"]),
    (r"operating (?:profit )?margins?", ["operating_margin"]),
    (r"(?:net income|net profit|net|profit) margins?", ["net_margin"]),
    (r"margins?", GENERIC_MARGINS),
    (r"profitab\w*", PROFITABILITY),
    (r"current ratio|liquidity", ["current_ratio"]),
    (r"cash ratio", ["cash_ratio"]),
    (r"debt[- ]to[- ]equity|debt/equity|leverage", [LEVERAGE]),
    (r"debt|indebted|borrowings", "excluded:" + _DEBT),
    (r"earnings per share|\beps\b", ["DilutedEPS"]),
    (r"operating (?:income|profits?|earnings)", ["OperatingIncome"]),
    (r"gross profits?", "excluded:" + _GROSS_PROFIT),
    (r"run[- ]rate", "excluded:" + _RUN_RATE),
    (r"net income|net profits?|net earnings|earnings|profits?|bottom line", ["NetIncome"]),
    (r"revenues?|sales|top[- ]line", ["Revenue"]),
]
_EXPANSION_PHRASE = (r"\b(financially (?:strong|weak|healthy)|(?:strong|weak|solid|healthy) (?:financial position|"
                     r"fundamentals|balance sheet|financials)|financial position|(?:healthy|good|poor) company|"
                     r"in trouble)\b")
_SEGMENT_WORDS = (r"\b(aws|amazon web services|azure|google cloud|cloud|youtube|search|data ?cent(?:er|re)s?|gaming|"
                  r"iphones?|ipads?|macs?|wearables|services|subscriptions?|reality labs|family of apps|instagram|"
                  r"whatsapp|waymo|mercado ?pago|custom chips?|chip business|trainium|segments?|divisions?|"
                  r"business units?|ad(?:vertising)?|backlog|remaining performance obligations?|rpo|"
                  r"subscribers?|users|automotive|energy|hardware|commercial|government)\b")
# Forward-looking wording: the claim is about the future, so past figures are evidence about something else
_FORECAST_WORDS = (r"\b(will|would|going to|set to|expected to|expects?|expecting|forecast\w*|projected|projects?|"
                   r"guid(?:ance|ed|es)|outlook|next (?:year|quarter)|this coming|by the end of (?:this|next) year|"
                   r"in (?:five|ten|\d+) years)\b")
_EXCLUDING_WORDS = r"\b(exclud\w*|excluding|ex-|strip\w* out|one-?time|minus (?:the )?gains|without (?:the )?gains)"
_GROWTH_PCT = (r"\b(grew|grow|grows|growing|growth|up|rose|risen|increased?|declined?|fell|fallen|down|doubled|"
               r"jumped|surged|dropped|shot up)\b[^.;]{0,40}?\d+(?:\.\d+)?\s*%"
               r"|\d+(?:\.\d+)?\s*%[^.;]{0,25}?\b(growth|increase|decline|year[- ]over[- ]year|yoy)\b")
_HISTORY = (r"\b(first time since|since (?:it|google|the company|they) went public|(?:highest|lowest|best|worst)"
            r"(?: \w+)? since|all[- ]time highs?|record(?:[- ]level)?|best quarter ever|ever)\b")
# "operating margins are 34%", "is around 17%", "48.4% now": a level stated in the present = the latest quarter
_PRESENT = (r"\b(is|are|currently|sits? at|stands? at|sitting at)\s+(?:at\s+)?(?:about|around|nearly|over|roughly|"
            r"almost|just over|just under)?\s*\d|\d+(?:\.\d+)?\s*%\s*(?:now|currently|today)\b")
_PRICE = r"\b(stocks?|shares?|share price|market cap\w*|valuation|worth|price target)\b"
_OTHER_NAME = (r"\b([A-Z][A-Za-z]+(?:\s[A-Z][a-z]+)?)(?:'s)?\s+(?:generated|reported|posted|made|delivered|saw|grew|"
               r"has|had|announced)\b")
_NOT_A_NAME = {"It", "They", "The", "This", "That", "He", "She", "We", "I", "You", "Company", "Revenue", "Revenues",
               "Last", "Next", "So", "And", "But", "Then", "Now", "Well", "Earlier", "Management", "Wall Street",
               "Its", "Their", "Our", "Nobody", "Everyone", "Investors", "Analysts", "Q1", "Q2", "Q3", "Q4"}
_CUTOFF_TEXT = r"\bcut-?off(?: date)?\s*[:=]?\s*[^.?!]*"


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text or "", flags=re.IGNORECASE) is not None


# ------------------------------------------------------------------ periods named in words
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                      "nov", "dec"), 1)}
_QORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_NWORD = {"two": 2, "three": 3, "four": 4, "2": 2, "3": 3, "4": 4}
_YEAR = r"'?(\d{4}|\d{2})"
_FISCAL = r"(?:fy|fiscal(?: year)?)\s*"
_TTM = r"\b(ttm|ltm|trailing (?:12|twelve)[- ]months?|(?:last|past|prior) (?:12|twelve) months)\b"
_LAST_N_QUARTERS = r"\b(?:last|past|latest|recent|previous|prior) (two|three|four|2|3|4) quarters\b"
_LATEST_QUARTER = r"\bquarter(ly)?\b|\brecent(ly)?\b"           # rule book: "recent" or "quarter"
_LATEST_YEAR = r"\b(annual|yearly|full[- ]year|fiscal year|in a (?:single )?year|last year)\b"


def _year(s: str | None) -> int | None:
    if not s:
        return None
    y = int(s.lstrip("'"))
    return y + 2000 if y < 100 else y


def named_period(text: str, is_field: bool = False) -> tuple | None:
    """What period the words name, before looking at any data:
    ('ttm',), ('last_n', n), ('fiscal_q', fiscal_year|None, q), ('cal_q', year|None, q),
    ('month_q', year|None, month), ('fy', year), ('latest', 'quarter'|'annual').
    is_field: the text is Claim extraction's `period` field, where "March 2026" means the March 2026 quarter."""
    t = (text or "").lower()
    if re.search(_TTM, t):
        return ("ttm",)
    m = re.search(_LAST_N_QUARTERS, t)
    if m:
        return ("last_n", _NWORD[m.group(1)])
    m = (re.search(rf"\bq([1-4])\s*(?:of\s+)?{_FISCAL}{_YEAR}\b", t)
         or re.search(rf"\b{_FISCAL}{_YEAR}\s*q([1-4])\b", t))
    if m:
        a, b = m.groups()
        q, y = (a, b) if m.re.pattern.startswith(r"\bq") else (b, a)
        return ("fiscal_q", _year(y), int(q))
    m = re.search(rf"\bfiscal (?:q([1-4])|(first|second|third|fourth) quarter)(?:\s+(?:of\s+)?(?:{_FISCAL})?{_YEAR})?", t)
    if m:
        return ("fiscal_q", _year(m.group(3)), int(m.group(1)) if m.group(1) else _QORD[m.group(2)])
    m = re.search(rf"\b{_FISCAL}{_YEAR}\b", t) or re.search(rf"\bfull[- ]year\s+{_YEAR}\b", t)
    if m:
        return ("fy", _year(m.group(1)))
    m = re.search(rf"\bq([1-4])(?:\s+of)?(?:\s*{_YEAR})?\b", t)          # Q4 2025, Q4 of 2025, Q4'25
    if m:
        return ("cal_q", _year(m.group(2)), int(m.group(1)))
    m = re.search(rf"\b(first|second|third|fourth) quarter(?:\s+of)?(?:\s+{_YEAR})?\b", t)
    if m:
        return ("cal_q", _year(m.group(2)), _QORD[m.group(1)])
    m = (re.search(rf"\b{_MON}\s*(?:\d{{1,2}},?\s*)?(\d{{4}})?\s*quarter\b", t)
         or re.search(rf"\bquarter\s+end(?:ed|ing)\s+(?:in\s+|on\s+)?{_MON}\s*(?:\d{{1,2}},?\s*)?(\d{{4}})?", t)
         or (is_field and re.search(rf"^\s*{_MON}\s+(\d{{4}})\s*$", t)))
    if m:
        return ("month_q", _year(m.group(2)), _MONTHS[m.group(1)[:3]])
    if re.search(_LATEST_QUARTER, t.replace("a quarter of", "")):
        return ("latest", "quarter")
    if re.search(_LATEST_YEAR, t):
        return ("latest", "annual")
    return None


# ------------------------------------------------------------------ streaks and percentages
_PCT = r"(-?\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)"
_AT_LEAST = r"(over|more than|above|at least|north of|upwards of|in excess of)\s*$"
_COUNT_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_ORDINAL_WORDS = {w: i for i, w in enumerate(
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth "
    "fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth".split())}
_NUMWORD = r"(\d{1,2}|" + "|".join(list(_COUNT_WORDS) + list(_ORDINAL_WORDS)) + r")(?:st|nd|rd|th)?"


def minimum_percent(text: str) -> float | None:
    """A lower bound written as a percentage ("over 20%", "20%+"), as a decimal; used for streak thresholds."""
    for m in re.finditer(_PCT, text, flags=re.IGNORECASE):
        before, after = text[max(0, m.start() - 30):m.start()], text[m.end():m.end() + 6]
        if re.search(_AT_LEAST, before, flags=re.IGNORECASE) or re.match(r"\s*(\+|plus\b)", after, flags=re.IGNORECASE):
            return float(m.group(1)) / 100
    return None


def streak(text: str) -> tuple[int, str, float] | None:
    """(N, 'quarter'|'annual', growth threshold) for "twelfth straight quarter of double-digit growth"."""
    t = (text or "").lower()
    m = (re.search(rf"\b{_NUMWORD}\s+(?:straight|consecutive)\s+(quarter|year)s?\b", t)
         or re.search(rf"\b{_NUMWORD}\s+(quarter|year)s?\s+in a row\b", t))
    if not m:
        return None
    word = m.group(1)
    n = int(word) if word.isdigit() else _COUNT_WORDS.get(word) or _ORDINAL_WORDS.get(word)
    threshold = 1.0 if _has(r"triple[- ]digit", t) else 0.10 if _has(r"double[- ]digit", t) else 0.0
    floor = minimum_percent(t)
    if floor is not None:
        threshold = floor
    return n, ("quarter" if m.group(2) == "quarter" else "annual"), threshold


# ------------------------------------------------------------------ reading a line
@dataclass
class Point:
    """One figure to report. measure: a key from GROWTH_MEASURES, LEVEL_MEASURES or LEVERAGE."""
    measure: str
    period: tuple | None = None        # a named_period() spec; None = latest fiscal year
    words: str = ""                    # the words this point was read from
    growth_pct: bool = False           # the words state a growth percentage (adds a growth item for levels)
    streak: tuple | None = None
    history: str | None = None         # "first time since 2004": longer than the window, noted only


@dataclass
class Reading:
    points: list[Point] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)      # parts of the claim not checked, and why


@dataclass
class _Mention:
    start: int
    end: int
    measures: list[str] | None          # None for an excluded mention
    note: str = ""


def _mask(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _mentions(text: str) -> list[_Mention]:
    """Measure mentions in order of position (ignored words removed)."""
    work, found = text, []
    for pattern, what in _LEXICON:
        for m in list(re.finditer(rf"\b(?:{pattern})", work, flags=re.IGNORECASE)):
            if what != "ignore":
                if isinstance(what, str):
                    found.append(_Mention(m.start(), m.end(), None, what.split(":", 1)[1]))
                else:
                    found.append(_Mention(m.start(), m.end(), list(what)))
            work = _mask(work, m.start(), m.end())
    return sorted(found, key=lambda x: x.start)


def _names(ticker: str) -> set[str]:
    return {n.lower() for n in COMPANY_NAMES.get(ticker, [])} | {ticker.lower()}


def _company_marks(text: str, ticker: str) -> list[tuple[int, str]]:
    """(position, 'self' | 'other' | 'product') for every company or product named in the text."""
    marks = []
    products = {alias.lower() for names in PRODUCTS.values() for alias in names}
    for tick, surface, start in find_mentions(text):
        if surface.lower() in products or surface == "YouTube":
            marks.append((start, "product"))
        else:
            marks.append((start, "self" if tick == ticker else "other"))
    known = {n.lower() for names in COMPANY_NAMES.values() for n in names}
    for m in re.finditer(_OTHER_NAME, text):
        name = m.group(1)
        if name in _NOT_A_NAME or name.lower() in known or name.split()[0] in _NOT_A_NAME:
            continue
        marks.append((m.start(), "other"))
    return sorted(marks)


def _owner(marks: list[tuple[int, str]], pos: int) -> str:
    before = [kind for start, kind in marks if start < pos]
    return before[-1] if before else "self"


def _clause_start(text: str, pos: int) -> int:
    cut = max(text.rfind(c, 0, pos) for c in ".;:!?")
    return cut + 1


def _clean_prompt(prompt: str | None) -> str:
    return re.sub(_CUTOFF_TEXT, " ", prompt or "", flags=re.IGNORECASE).strip()


def _read(text: str, ticker: str, implied_ok: bool) -> Reading:
    """Points and notes from one text, before the prompt filter and period choice."""
    out = Reading()
    if not text:
        return out
    marks = _company_marks(text, ticker)
    mentions = _mentions(text)
    measured = [m for m in mentions if m.measures]
    for i, m in enumerate(mentions):
        start = max(_clause_start(text, m.start), mentions[i - 1].end if i else 0)
        nxt = next((x.start for x in mentions[i + 1:]), len(text))
        stop = min(nxt, len(text))
        sentence_end = re.search(r"[.;!?]", text[m.end:])
        if sentence_end:
            stop = min(stop, m.end + sentence_end.start())
        before, words = text[start:m.start], text[start:stop]
        owner = _owner(marks, m.start)
        label = text[m.start:m.end]
        if m.measures is None:
            if owner == "self":
                out.notes.append(f"'{label}' not checked: {m.note}")
            continue
        if owner == "other":
            continue
        if owner == "product" or _has(_SEGMENT_WORDS, before[-40:]) or _has(r"^\s*(from|of|at|in) (the )?"
                                                                             + _SEGMENT_WORDS, text[m.end:m.end + 30]):
            out.notes.append(f"'{label}' not checked: {SEGMENT_NOTE}")
            continue
        near_excluded = any(x.measures is None and x.note == _RUN_RATE and 0 <= x.start - m.end <= 3
                            for x in mentions)
        if near_excluded:
            continue                                     # "revenue run rate": noted with the run rate
        if _has(_FORECAST_WORDS, words):
            out.notes.append(f"'{label}' not checked: {_FORECAST}")
            continue
        if _has(_EXCLUDING_WORDS, words) or _has(_EXCLUDING_WORDS, text) and "NetIncome" in m.measures:
            out.notes.append(f"'{label}' not checked: {_EXCLUDING}")
            continue
        for measure in m.measures:
            out.points.append(Point(measure, named_period(words), words.strip(), _has(_GROWTH_PCT, words),
                                    streak(words) or streak(text), _history(words)))
    if _has(_EXPANSION_PHRASE, text) and _owner(marks, re.search(_EXPANSION_PHRASE, text, re.I).start()) == "self":
        have = {p.measure for p in out.points}
        out.points += [Point(x, None, re.search(_EXPANSION_PHRASE, text, re.I).group(0)) for x in EXPANSION
                       if x not in have]
    if not measured and not out.points and implied_ok and _has(_GROWTH_PCT, text):
        m = re.search(_GROWTH_PCT, text, flags=re.IGNORECASE)
        owner = _owner(marks, m.start())
        clause = text[_clause_start(text, m.start()):m.end()]
        if _has(_PRICE, clause) or _has(_FORECAST_WORDS, clause):
            pass                                          # "the stock is up 25%": a price, not a company figure
        elif owner == "self" and not _has(_SEGMENT_WORDS, clause):
            out.points.append(Point("Revenue", named_period(text), text.strip(), True, streak(text), _history(text)))
        elif owner == "product" or _has(_SEGMENT_WORDS, text):
            out.notes.append(f"'{m.group(0)}' not checked: {SEGMENT_NOTE}")
    return out


def _history(words: str) -> str | None:
    m = re.search(_HISTORY, words or "", flags=re.IGNORECASE)
    return m.group(0) if m else None


def _named_self(text: str, ticker: str) -> bool:
    return any(kind == "self" for _, kind in _company_marks(text, ticker))


def read_claim(line: str, ticker: str, prompt: str | None = None, period_field: str | None = None,
               family: str | None = None) -> Reading:
    """The points to report for one post line. prompt: what the user typed; family: Claim extraction's
    `measure` for an expanded row ("profitability", ...), which keeps only that family."""
    prompt = _clean_prompt(prompt)
    line_reading = _read(line, ticker, implied_ok=_named_self(line, ticker) or _has(r"\brevenue|\bgrew\b", prompt))
    out = Reading(notes=list(line_reading.notes))
    points = line_reading.points

    if prompt:
        asked = _read(prompt, ticker, implied_ok=True)
        wanted = {p.measure for p in asked.points}
        segment_prompt = not wanted and any(SEGMENT_NOTE in n for n in asked.notes)
        if segment_prompt:
            out.notes.append("The prompt asks about a segment or product; the narrative agent checks it.")
            return out
        if wanted:
            dropped = sorted({p.measure for p in points if p.measure not in wanted})
            points = [p for p in points if p.measure in wanted]
            if dropped:
                out.notes.append("Not in the user's claim, so not checked: " + ", ".join(dropped))
            if not points:
                out.notes.append("The post line does not state the measure the prompt asks about ("
                                 + ", ".join(sorted(wanted)) + ").")
        prompt_period = named_period(prompt)
    else:
        prompt_period = None
    if family in FAMILIES:
        points = [p for p in points if p.measure in FAMILIES[family]]

    line_period = named_period(line)
    field_period = named_period(period_field or "", is_field=True)
    seen = set()
    for p in points:
        period = prompt_period or p.period or line_period or field_period
        if period is None and p.measure in LEVEL_MEASURES and p.measure != "free_cash_flow" \
                and (_has(_PRESENT, p.words) or _has(_PRESENT, prompt)):
            period = ("latest", "quarter")              # "operating margins are 34%": the latest reported quarter
        p.period = period
        key = (p.measure, period)
        if key in seen:
            continue
        seen.add(key)
        out.points.append(p)
        if p.history and not any(p.history in n for n in out.notes):
            out.notes.append(f"'{p.history}' not checked: a history longer than the window "
                             "(latest 10-K plus four quarters)")
    out.notes = list(dict.fromkeys(out.notes))
    return out
