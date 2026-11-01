"""Split a sentence into atomic facts, then line up the user's facts with the influencer's.

A fact is one checkable statement: a measure, a direction, at most one figure, and a period.
"Revenue grew 22% to $110 billion" holds two facts (22% and $110 billion). "Revenue, operating
profit and operating margin all went up" is one fact about three measures.

Every fact keeps `span`, the exact words it came from, so a quote fact is always the
influencer's own words. Nothing here calls a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finsight.agents.claim_extraction.fields import _LEADING_TO, _close, parse_fields

# -- splitting ---------------------------------------------------------------------

_HARD_SPLIT = re.compile(r"\s*(?:[;:]|,(?!\d{3})|\s[-–—]\s)\s*")
_SOFT_SPLIT = re.compile(r"\s+\b(?:and|but|while|whereas|plus|with)\b\s+(?!\d+/\d)", flags=re.IGNORECASE)
# "doubled" / "more than doubling" is about +100%; "tripled" about +200%. Used only when no percent is stated.
# "Double-digit" / "triple-digit growth" names a range, not a doubling.
_FOLD_VALUES = ((re.compile(r"\b(?:doubl(?!e[- ]?digits?\b)\w*|twofold|2x)\b", re.IGNORECASE), 100.0),
                (re.compile(r"\b(?:tripl(?!e[- ]?digits?\b)\w*|threefold|3x)\b", re.IGNORECASE), 200.0))
_WORD = re.compile(r"[a-z0-9]+(?:\.\d+)?%?")

_STOP = frozenset("""
a an the of and or to in on for with by from at as is are was were be been being has have had it its it's
this that these those there their they them he she his her we our you your i me my so very really just
also than then which who whom whose what when where how about into over under up down out more most
said says say saying claim claims claimed think thinks believe believes according per cent percent
year years quarter quarters last this next now currently recent recently already still even only
went came got get gets seen see look looking going
""".split())
_GENERIC = frozenset({"revenue", "profit", "margin", "cash", "up", "down", "double", "debt", "leverage"})
_SYNONYMS = {
    "revenues": "revenue", "sales": "revenue", "topline": "revenue",
    "profits": "profit", "earnings": "profit", "income": "profit",
    "margins": "margin", "profitability": "margin",
    "consecutive": "straight", "row": "straight", "successive": "straight",
    "grew": "up", "grow": "up", "growth": "up", "growing": "up", "rose": "up", "rise": "up", "rising": "up",
    "increase": "up", "increased": "up", "higher": "up", "expanding": "up", "expanded": "up", "expand": "up",
    "fell": "down", "fall": "down", "falling": "down", "declined": "down", "decline": "down",
    "dropped": "down", "drop": "down", "lower": "down", "decreased": "down", "dipped": "down", "negative": "down",
    "doubled": "double", "doubling": "double", "twice": "double",
    "ai": "ai", "artificial": "ai", "intelligence": "ai",
}


# A business line inside a company. "Google Cloud revenue rose to $20 billion" is not company revenue.
_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("google cloud", "Google Cloud"), ("cloud", "cloud"), ("azure", "Azure"), ("aws", "AWS"),
    ("amazon web services", "AWS"), ("data center", "data center"), ("datacenter", "data center"),
    ("youtube", "YouTube"), ("search", "Search"), ("advertising", "advertising"), ("ad revenue", "advertising"),
    ("services", "Services"), ("iphone", "iPhone"), ("gaming", "gaming"), ("automotive", "automotive"),
    ("subscription", "subscriptions"), ("reality labs", "Reality Labs"), ("waymo", "Waymo"),
    ("commercial", "commercial"), ("government", "government"), ("segment", "segment"),
)


def segments_in(text: str) -> frozenset[str]:
    lowered = (text or "").lower()
    found = set()
    for needle, name in _SEGMENTS:
        if re.search(rf"\b{re.escape(needle)}\b", lowered):
            found.add(name)
    if found & {"Google Cloud", "Azure", "AWS"}:
        found.discard("cloud")
    return frozenset(found)


@dataclass(frozen=True)
class Fact:
    span: str
    metrics: frozenset[str]
    direction: str | None
    value: tuple[float, str] | None
    period: str | None
    words: frozenset[str]
    inherited: frozenset[str] = field(default_factory=frozenset)
    segments: frozenset[str] = field(default_factory=frozenset)

    @property
    def strong(self) -> bool:
        return self.value is not None or self.direction is not None

    @property
    def label(self) -> str:
        """Short name for the fact: the figure as the speaker wrote it, else the words."""
        if self.value is None:
            return self.span
        return _raw_figure(self.span, self.value) or _fmt(self.value)


def content_words(text: str) -> frozenset[str]:
    found = set()
    for word in _WORD.findall((text or "").lower()):
        if any(ch.isdigit() for ch in word) or word in _STOP or len(word) < 3:
            continue
        word = _SYNONYMS.get(word, word)
        if word.endswith("s") and len(word) > 4 and word not in _SYNONYMS.values():
            word = word[:-1]
        found.add(word)
    return frozenset(found)


_TRIM = " .,;:!?\"'"


def _ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges of clause-sized pieces, split at punctuation and at and / but / while."""
    cuts = sorted({m.start() for m in _HARD_SPLIT.finditer(text)} | {m.end() for m in _HARD_SPLIT.finditer(text)}
                  | {m.start() for m in _SOFT_SPLIT.finditer(text)} | {m.end() for m in _SOFT_SPLIT.finditer(text)})
    bounds = [0, *cuts, len(text)]
    out: list[tuple[int, int]] = []
    for start, end in zip(bounds, bounds[1:]):
        piece = text[start:end]
        if _HARD_SPLIT.fullmatch(piece) or _SOFT_SPLIT.fullmatch(piece) or not piece.strip(_TRIM):
            continue
        out.append(_trimmed(text, start, end))
    return out


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _TRIM:
        start += 1
    while end > start and text[end - 1] in _TRIM:
        end -= 1
    return start, end



# "Their best quarter ever by far", "a record quarter": a ranking claim of its own. Kept apart so it never
# drags a checkable figure in the same line down to "partly supported".
_SUPERLATIVE = re.compile(r"\b(?:best|worst|biggest|largest|highest|lowest|record|strongest)\b.{0,30}\bever\b"
                          r"|\brecord\s+(?:quarter|year|high|revenue|revenues)\b"
                          # a result against analyst estimates ("which beat expectations") is its own claim too
                          r"|\b(?:beat|beats|missed|misses|topped|ahead of)\s+(?:\w+\s+){0,2}"
                          r"(?:expectations?|estimates?|consensus)\b", re.IGNORECASE)


def _is_strong(text: str) -> bool:
    fields = parse_fields(text)
    return bool(fields.values or fields.direction or _SUPERLATIVE.search(text))


def _merge_weak(text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """A piece with no figure and no direction joins a neighbour, so a split never cuts a fact in half.

    A bare measure ("Revenue", "operating profit") joins the next piece, because the move comes after it.
    A short aside joins the previous piece. A merged piece spans the original text, separators included.
    """
    merged: list[tuple[int, int]] = []
    carry: int | None = None
    for index, (start, end) in enumerate(ranges):
        if carry is not None:
            start, carry = carry, None
        piece = text[start:end]
        if _is_strong(piece):
            merged.append((start, end))
            continue
        has_metric = bool(parse_fields(piece).metrics)
        is_last = index == len(ranges) - 1
        few_words = len(content_words(piece)) < 4
        if has_metric and not is_last:
            carry = start
        elif merged and few_words:
            merged[-1] = (merged[-1][0], end)
        elif not is_last and (not merged or few_words):
            carry = start
        elif merged and not _is_strong(text[merged[-1][0]:merged[-1][1]]):
            merged[-1] = (merged[-1][0], end)  # two plain clauses stay one statement
        else:
            merged.append((start, end))
    if carry is not None:
        if merged:
            merged[-1] = (merged[-1][0], ranges[-1][1])
        else:
            merged.append((carry, ranges[-1][1]))
    return merged


# "Reported earnings" / "earnings call" name the event, not the profit measure.
_EARNINGS_EVENT = re.compile(
    r"\b(?:reported|reports|report|posted|after|ahead of|before)\s+(?:their\s+|its\s+)?earnings\b"
    r"|\bearnings\s+(?:call|report|release|season|day)s?\b", flags=re.IGNORECASE)


def _fields(piece: str):
    return parse_fields(_EARNINGS_EVENT.sub(" results ", piece))


def split_facts(text: str) -> list[Fact]:
    """Facts in reading order. A piece with two figures becomes two facts that share its words."""
    text = text or ""
    whole = parse_fields(text)
    ranges = _merge_weak(text, _ranges(text))
    # "investing leading to higher revenues" names a hoped-for result, not a reported move.
    consequence = _LEADING_TO.search(text)
    facts: list[Fact] = []
    last_metrics: frozenset[str] = frozenset()
    last_segments: frozenset[str] = frozenset()
    last_direction: str | None = None
    for start, end in ranges:
        piece = text[start:end]
        fields = _fields(piece)
        metrics = frozenset(fields.metrics)
        inherited: frozenset[str] = frozenset()
        if not metrics:
            # A measure carries forward to the next clauses, never back to an earlier one.
            inherited = last_metrics
        else:
            last_metrics = metrics
        segments = segments_in(piece) or last_segments
        last_segments = segments
        direction = fields.direction
        if consequence is not None and start >= consequence.start():
            direction = None
        values = _dedupe_values(list(fields.values))
        if not any(unit == "percent" for _number, unit in values):
            values += [(implied, "percent") for pattern, implied in _FOLD_VALUES if pattern.search(piece)]
        values = values or [None]
        for value in values:
            # A percent with no direction word takes the move just named ("grew 20%, and 22% this quarter").
            own_direction = direction
            if own_direction is None and value is not None and value[1] == "percent":
                own_direction = last_direction
            facts.append(Fact(
                span=piece,
                metrics=metrics | inherited,
                direction=own_direction,
                value=value,
                period=fields.period or whole.period,
                words=content_words(piece),
                inherited=inherited,
                segments=segments,
            ))
        if direction:
            last_direction = direction
    return facts


_RAW = re.compile(
    r"(?:\$\s*)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(%|percent\b|(?:billion|million|trillion)\b(?:\s+[a-z]+)?)?",
    flags=re.IGNORECASE)


_COUNT_NOUNS = frozenset({"subscribers", "users", "customers", "people", "members", "employees", "vehicles",
                          "units", "cars", "devices", "accounts", "viewers", "downloads", "orders", "deliveries"})


def _raw_figure(span: str, value: tuple[float, str]) -> str | None:
    """The words that state `value` in `span`: "$110 billion", "20 million subscribers", "18.3%"."""
    number, unit = value
    for match in _RAW.finditer(span):
        raw = float(match.group(1).replace(",", ""))
        scale = (match.group(2) or "").lower()
        candidates = {raw}
        if scale.startswith("million"):
            candidates.add(raw / 1000.0)
        if unit == "percent" and not (scale.startswith("%") or scale.startswith("percent")):
            continue
        if any(abs(candidate - number) < 1e-6 for candidate in candidates):
            text = match.group(0).strip()
            words = text.split()
            # Keep the counted noun only for a count, not money: "20 million subscribers", "$110 billion".
            if len(words) >= 3 and (text.startswith("$") or words[-1].lower() not in _COUNT_NOUNS):
                text = " ".join(words[:-1])
            return text
    return None


def _dedupe_values(values: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """"$110 billion" is read as both $110 billion and $110. Keep one."""
    billions = {round(number, 3) for number, unit in values if unit == "billion"}
    out: list[tuple[float, str]] = []
    for number, unit in values:
        if unit == "dollar" and round(number, 3) in billions:
            continue
        if (number, unit) not in out:
            out.append((number, unit))
    return out


def per_measure(facts: list[Fact]) -> list[Fact]:
    """A user fact about several measures ("revenue, profit and margin all went up") is one fact per measure."""
    out: list[Fact] = []
    for fact in facts:
        if len(fact.metrics) > 1 and fact.value is None:
            for metric in sorted(fact.metrics):
                out.append(Fact(fact.span, frozenset({metric}), fact.direction, None, fact.period,
                                fact.words, fact.inherited, fact.segments))
        else:
            out.append(fact)
    return out


# -- alignment ---------------------------------------------------------------------

BOTH, CONFLICT, QUOTE_ONLY, USER_ONLY = "both", "conflict", "quote_only", "user_only"


@dataclass
class Link:
    kind: str  # both | conflict | none
    score: float
    note: str | None = None


def _same_figure(left: tuple[float, str], right: tuple[float, str]) -> bool | None:
    """True / False when the units are comparable, None when they are not."""
    (a, unit_a), (b, unit_b) = left, right
    money = {"billion", "dollar"}
    if unit_a == "percent" and unit_b == "percent":
        return _close(a, b, True)
    if unit_a in money and unit_b in money:
        # "Nearly $10 billion" is $10 billion; $2.4 billion is not $2 billion. A spoken "$110" in a revenue
        # line is $110 billion, so the two money units compare by number.
        return abs(a - b) <= max(0.05, 0.06 * max(a, b))
    return None


def _fmt(value: tuple[float, str]) -> str:
    number, unit = value
    text = f"{number:g}"
    return {"percent": f"{text}%", "billion": f"${text} billion", "dollar": f"${text}"}.get(unit, text)


def link(user: Fact, quote: Fact) -> Link:
    """How one user fact relates to one quote fact."""
    shared_metric = bool(user.metrics & quote.metrics)
    metric_clash = bool(user.metrics and quote.metrics and not shared_metric)
    if metric_clash:
        return Link("none", 0.0)
    if user.segments and quote.segments and not (user.segments & quote.segments):
        return Link("none", 0.0)  # AWS is not Azure
    overlap = len(user.words & quote.words) / max(1, len(user.words))
    opposite = bool(user.direction and quote.direction and user.direction != quote.direction)
    if user.value is not None:
        if quote.value is None:
            return Link("none", 0.0)
        same = _same_figure(user.value, quote.value)
        if same is None:
            return Link("none", 0.0)
        if same and not opposite and quote.segments and not user.segments:
            where = ", ".join(sorted(quote.segments))
            return Link(CONFLICT, 2.0 + overlap,
                        f"They said {_fmt(quote.value)} about {where}, not about the whole company.")
        if same and not opposite:
            return Link(BOTH, 3.0 + (0.5 if shared_metric else 0.0) + overlap)
        # Different figure on the same measure: the user misremembered (or the influencer misspoke).
        if shared_metric or (not user.metrics and not quote.metrics and overlap >= 0.5):
            what = "said it moved the other way" if same else f"said {_fmt(quote.value)}"
            note = f"You said {_fmt(user.value)}; they {what}."
            return Link(CONFLICT, 1.0 + (0.5 if shared_metric else 0.0) + overlap, note)
        return Link("none", 0.0)
    # The user stated no figure.
    distinctive = user.words - _GENERIC
    shared_distinctive = len(distinctive & quote.words) / max(1, len(distinctive))
    if opposite and (shared_metric or shared_distinctive >= 0.5):
        return Link(CONFLICT, 1.0 + overlap, f"You said it went {user.direction}; they said it went {quote.direction}.")
    if len(distinctive) >= 3 and not distinctive & quote.words:
        # "Its twelfth straight quarter of double-digit growth" is not "Revenue grew by 24%", even though both
        # are revenue going up: a fact with its own words must share some of them.
        return Link("none", 0.0)
    score = 2.0 * overlap + (1.0 if shared_metric else 0.0)
    if user.direction and user.direction == quote.direction:
        score += 1.0  # "revenue doubled" is the "up 106%" part, not the "$96.2 billion" level
    elif user.direction and quote.direction is None:
        score -= 0.5
    if quote.value is not None and shared_metric:
        score += 0.5  # "revenue went up" is the claim behind "revenue up 33%"
    if score >= 1.2:
        return Link(BOTH, score)
    return Link("none", score)


@dataclass
class Aligned:
    bucket: str
    user: Fact | None
    quote: Fact | None
    quote_index: int | None = None  # which quote line the fact came from
    note: str | None = None


def align(user_facts: list[Fact], quotes: list[list[Fact]]) -> list[Aligned]:
    """Line up the user's facts with the facts of the matched quote lines.

    Every quote fact appears once: both / conflict when some user fact points at it, else quote_only.
    Every user fact that points at nothing appears as user_only.
    """
    user_facts = per_measure(user_facts)
    flat = [(qi, fact) for qi, facts in enumerate(quotes) for fact in facts]
    best_for_quote: dict[int, tuple[Link, Fact]] = {}
    user_only: list[Fact] = []
    for user in user_facts:
        ranked = sorted(
            ((link(user, fact), index) for index, (_qi, fact) in enumerate(flat)),
            key=lambda item: ({BOTH: 2, CONFLICT: 1}.get(item[0].kind, 0), item[0].score),
            reverse=True,
        )
        if not ranked or ranked[0][0].kind == "none":
            user_only.append(user)
            continue
        found, index = ranked[0]
        previous = best_for_quote.get(index)
        if previous is None or (found.kind == BOTH and previous[0].kind != BOTH) or (
            found.kind == previous[0].kind and found.score > previous[0].score
        ):
            best_for_quote[index] = (found, user)
    out: list[Aligned] = []
    for index, (qi, fact) in enumerate(flat):
        if index in best_for_quote:
            found, user = best_for_quote[index]
            out.append(Aligned(found.kind, user, fact, qi, found.note))
        else:
            out.append(Aligned(QUOTE_ONLY, None, fact, qi))
    covered = {id(item.user) for item in out if item.user is not None}
    seen_only: set[tuple[str, tuple | None]] = set()
    for user in user_only:
        key = (user.span, user.value)
        if id(user) not in covered and key not in seen_only:
            seen_only.add(key)
            out.append(Aligned(USER_ONLY, user, None, None, f"They did not say {user.label} in the matched line."))
    # A user fact that lost its quote fact to a closer one is still covered, not user_only.
    return out


def coverage(user_facts: list[Fact], quote_facts: list[Fact]) -> tuple[float, int, int]:
    """(score, both, conflict) of one quote line against the user's facts."""
    score, both, conflict = 0.0, 0, 0
    for user in per_measure(user_facts):
        links = [link(user, fact) for fact in quote_facts]
        best = max(links, key=lambda item: ({BOTH: 2, CONFLICT: 1}.get(item.kind, 0), item.score), default=None)
        if best is None or best.kind == "none":
            continue
        score += best.score
        both += best.kind == BOTH
        conflict += best.kind == CONFLICT
    return score, both, conflict
