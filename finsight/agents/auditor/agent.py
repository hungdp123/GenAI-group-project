"""Auditor Agent (Person 3, Amelia): number checks by calculation, narrative checks by citation.

It checks the evidence the specialists hand over, then proposes a verdict from the evidence that survives.
It does NOT apply source-tier caps, the cut-off filter or the retry loop: the orchestrator does those
(`routing.tier_shortfall`, `routing.split_temporal`, `nodes.retry_or_accept`).

Result semantics (what the orchestrator does with it):
- `passed=False`: a specialist's evidence is defective (wrong units, wrong metric, no provenance, citation does
  not support the sub-claim ...). `failure_reasons` says what, `feedback_for_retry` tells the collector how to
  fix it. The orchestrator re-gathers once, then marks the sub-claim INSUFFICIENT_EVIDENCE.
- `passed=True`: the evidence is usable. `verdict` is the PROPOSED verdict; the orchestrator applies the
  knowledge-layer cap and tier rules afterwards.

Numbers in `EvidenceItem.computed_value` follow the contract: growth rates and margins are FRACTIONS (0.14),
not percents. A stated number that is more than 0.5 percentage points from the computed one is CONTRADICTED
(docs/claim_spec.md, team decision of 14 Oct).

STANCE: the Auditor judges it, collectors do not. Whatever `EvidenceItem.stance` a collector sends (the Financial
agent sends NEUTRAL, the Narrative agent mostly NEUTRAL) is ignored. Each item gets SUPPORTING, CHALLENGING or NEUTRAL
(not about this sub-claim / cannot tell) from the evidence itself:
- numbers: the direction the sub-claim states (grew / fell) against the sign of the computed change;
- text: ruled-out driver or opposite direction -> CHALLENGING; states the sub-claim -> SUPPORTING; else NEUTRAL.
One exception, and it only ever softens a verdict: a collector's CHALLENGING is kept when the Auditor's own
check says NEUTRAL (it cannot confirm it, but counter-evidence is never thrown away). A collector's SUPPORTING is
never trusted, and the Auditor's SUPPORTING / CHALLENGING judgments are never overridden.
The judged stances go back in `AuditResult.evidence_stances`; the orchestrator applies them. Only usable
(non-NEUTRAL) items are checked and counted. If items came back but none is usable, the audit FAILS so the
collector gets its one retry with feedback.

WHAT THE SUB-CLAIM STATES (read from its wording; the contract has no comparator field): a bound ("above 10%",
"less than 10% growth") is compared as a bound, not as a target; "turned negative" is decided by the sign of the
computed figure; "doubled" means 1 + growth in 1.9-2.1 (team rule); a stated dollar amount matches after rounding to
the precision it is stated in ("$110 billion" -> +-$0.5bn); a direction needs the current and prior figure in the
evidence: `comparison_value` (the prior-year figure of a level metric) when present, else the evidence text
(percents, dollar amounts, or a plain pair such as "2.60 vs 2.98").

Entailment ("does the cited passage support the sub-claim?") is rule-based in mock mode so the tests are
deterministic. With a real LLM provider, `complete()` also reads the financial skip reason and the narrative
stance, reason, and summary. It can downgrade a keyword match or add CHALLENGING. It cannot create support
the rules did not already find, and it cannot change a number check. OpenAI calls use gpt-5.4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from finsight.config import CORPUS_COMPANIES, get_settings
from finsight.knowledge.terms import mask_other_margins
from finsight.contracts.schemas import (
    AuditResult,
    CitationCheck,
    EvidenceItem,
    NumericCheck,
    SourceTier,
    Stance,
    SubClaim,
    Verdict,
)
from finsight.llm.client import DATA_SAFETY_PREAMBLE, complete, data_block

MATCH_BAND = 0.005          # 0.5 percentage points, as a fraction (knowledge layer section 10)
MIN_COVERAGE = 0.6          # share of the sub-claim's key words that the passage must contain

# Sub-claim types whose SUPPORTED verdict needs a Tier 1 computed record (knowledge layer: HISTORICAL_GROWTH,
# profitability / cash / leverage). Compared by value so new enum members need no change here.
COMPUTED_TYPES = {
    "HISTORICAL_GROWTH", "NET_INCOME_GROWTH", "PROFITABILITY_CASH_LEVERAGE", "PROFITABILITY",
    "CASH_GENERATION", "LIQUIDITY", "LEVERAGE",
}
# Types where hedged wording ("may", "could") is what a filing normally says, so it is not a defect.
HEDGE_OK_TYPES = {"BUSINESS_RISK"}

TIER1_DOCS = {"10-k", "10-q", "xbrl", "20-f"}
TIER2_DOCS = {"earnings_release", "ir_remarks", "earnings_call", "earnings_slides", "ir_presentation",
              "investor_day", "press_release"}
TIER3_DOCS = {"news"}

MARGINS = {"gross_margin", "operating_margin", "net_margin", "fcf_margin"}

# "first time since 2004", "all-time high", "in 20 years": the corpus holds the latest 10-K and four quarters, so
# a claim about a longer history cannot be verified (knowledge layer section 3). Not a defect of the evidence.
_LONG_HISTORY = re.compile(
    r"\bfirst time since\b|\bsince (?:19|20)\d{2}\b|\ball[- ]time\b|\brecord[- ](?:high|low)\b|"
    r"\bin (?:over |more than )?\d+ years\b|\bin decades\b|\bdecades?\b", re.I)
# "best quarter ever, $81.6 billion, up 85%": the superlative cannot be checked, but the figures can. "ever" only
# short-circuits to INSUFFICIENT when the line states no figure the record could confirm.
_EVER = re.compile(r"\bever\b", re.I)
_NOT_CHECKED = re.compile(r"\bnot checked\b", re.I)   # the financial agent could not check part of the claim


def _is_long_history(text: str) -> bool:
    """A long-history phrase with no figure to check. With a figure, the figure is still audited."""
    if _LONG_HISTORY.search(text) or _EVER.search(text):
        return not (_pcts(text) or _money(text))
    return False

# -- wording -------------------------------------------------------------------------------------

_UP = re.compile(r"\b(grew|grow\w*|increas\w*|ros[e]|rising|rise[sn]?|expand\w*|improv\w*|higher|accelerat\w*|"
                 r"surg\w*|jump\w*|climb\w*|widen\w*|doubl\w*|tripl\w*|up)\b", re.I)
_DOWN = re.compile(r"\b(shrink\w*|shrank|shrunk|declin\w*|fell|fall\w*|decreas\w*|lower|contract\w*|compress\w*|"
                   r"narrow\w*|drop\w*|slow\w*|slid\w*|down|deteriorat\w*|weaken\w*|dipped)\b", re.I)
_HEDGE = re.compile(r"\b((?-i:may)|might|could|possibly|potentially|uncertain|unclear|depending on|no assurance|"
                    r"cannot predict)\b", re.I)
_NEGATED_ATTRIBUTION = re.compile(
    r"(rather than|instead of|not (?:due to|driven by|because of|attributable to)|unrelated to|other than)", re.I)
_FORWARD = re.compile(
    r"\b(expect\w*|guid\w*|outlook|anticipat\w*|forecast\w*|continu\w*|will|going forward|"
    r"next (?:quarter|year|fiscal)|through (?:the )?next|plans? to|target\w*)\b",
    re.I,
)
_FORWARD_AGAINST = re.compile(
    r"\b(remain(?:s)? elevated|pressure\w*|headwind\w*|decelerat\w*|soften\w*|weaker|"
    r"cautio\w*|uncertain\w*|moderat\w*)\b",
    re.I,
)
_FORWARD_FOR = re.compile(r"\b(strong|robust|continu\w*|accelerat\w*|growth|grow\w*|increas\w*|higher|record)\b", re.I)
_QOQ = re.compile(r"\bqoq\b|quarter[- ](?:on|over)[- ]quarter|sequential", re.I)
_QOQ_CLAIM = _QOQ

_STOP = set("a an the of in on at to for by is are was were be been being it its this that these those and or but "
            "with as from into than then so such not no over under per also has have had will would can may "
            "company x year years quarter strongly very "
            "look looking specifically which just really actually here basically".split())
_SYNONYMS = {
    "grew": "grow", "growth": "grow", "growing": "grow", "increase": "grow", "increased": "grow",
    "increases": "grow", "rose": "grow", "rise": "grow", "rising": "grow", "up": "grow",
    "climbed": "grow", "climb": "grow", "climbing": "grow", "climbs": "grow",
    "reached": "", "reach": "", "reaches": "", "reaching": "",
    "driven": "driv", "drive": "driv", "drives": "driv", "driving": "driv", "due": "driv",
    "attributable": "driv", "result": "driv", "resulting": "driv", "reflecting": "driv", "fueled": "driv",
    "primarily": "", "mainly": "", "largely": "",
    "topped": "over", "exceeded": "over", "above": "over", "eclipsed": "over",
    "chips": "chip", "margins": "margin",
}
GENERIC_STEMS = {"revenu", "grow", "driv", "increas"}

_NUM = r"\d[\d,]*(?:\.\d+)?"
_PCT_RE = re.compile(rf"([+\-−])?\s?({_NUM})\s?(?:%|percent\b|per cent\b)", re.I)
_MONEY_RE = re.compile(rf"(-)?\$\s?({_NUM})\s?(bn|billion|b|million|m|thousand|k)?(?![a-z])", re.I)
_UNIT = {"bn": 1e9, "billion": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3, None: 1.0}
_PERIOD_NOISE = re.compile(r"\bQ[1-4]\b|\b(?:FY)?\s?(?:19|20)\d{2}\b", re.I)


@dataclass
class _Pct:
    value: float            # percent as written (14.0 for "+14%")
    decimals: int
    signed: bool
    negative: bool


def _pcts(text: str) -> list[_Pct]:
    out = []
    for m in _PCT_RE.finditer(text):
        raw = m.group(2).replace(",", "")
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append(_Pct(float(raw), decimals, m.group(1) is not None, m.group(1) in ("-", "−")))
    return out


def _money(text: str) -> list[float]:
    out = []
    for m in _MONEY_RE.finditer(text):
        unit = m.group(3).lower() if m.group(3) else None
        value = float(m.group(2).replace(",", "")) * _UNIT[unit]
        out.append(-value if m.group(1) else value)
    return out


def _money_half_unit(text: str) -> float:
    """Half a unit in the last stated digit of the first dollar amount.

    "$110 billion" is $0.5bn and "$96.2bn" is $0.05bn. A dollar amount matches after rounding
    to the stated precision.
    """
    m = _MONEY_RE.search(text)
    if not m:
        return 0.0
    raw = m.group(2).replace(",", "")
    decimals = len(raw.split(".")[1]) if "." in raw else 0
    unit = m.group(3).lower() if m.group(3) else None
    return 0.5 * 10 ** (-decimals) * _UNIT[unit]


def _bare_amount_scale(text: str, stated: float, record: float) -> float:
    """Scale for a dollar amount written without a unit ("$110 in Q1").

    Speakers drop "billion". When the amount has no unit and is far below the record, read it in the unit
    (million, billion, trillion) that lands it within a factor of 10 of the record. A genuinely wrong bare
    figure ("$20" against $109.9bn) still lands off and stays a mismatch. Amounts with a unit are never rescaled.
    """
    m = _MONEY_RE.search(text)
    if not m or m.group(3) or stated <= 0 or record <= 0 or stated >= record / 1e3:
        return 1.0
    for scale in (1e6, 1e9, 1e12):
        if record / 10 <= stated * scale <= record * 10:
            return scale
    return 1.0


# A stated dollar amount this many times above or below every dollar figure in the record is not a rounding
# slip or a misquote of that period: the speaker means another period (item 17: "the next quarter ... 13.5 billion"
# is a 2023 quarter, the latest quarter is $96.2bn). The record cannot speak to it.
OUT_OF_WINDOW_FACTOR = 3.0


_UNIT_AMOUNT_RE = re.compile(rf"(?<![\w$.])({_NUM})\s?(bn|billion|million|trillion)\b", re.I)


def _stated_amounts(text: str) -> list[float]:
    """Dollar amounts the sub-claim states: "$110 billion", or "13.5 billion" when the transcript drops the $."""
    found = _money(text)
    if found:
        return found
    return [float(m.group(1).replace(",", "")) * {"bn": 1e9, "billion": 1e9, "million": 1e6,
                                                   "trillion": 1e12}[m.group(2).lower()]
            for m in _UNIT_AMOUNT_RE.finditer(text)]


def _out_of_window(sub_claim: SubClaim, evidence: list[EvidenceItem]) -> bool:
    """True when the one dollar amount the sub-claim states is far from every dollar figure in the numeric evidence."""
    stated = _stated_amounts(sub_claim.text)
    if len(stated) != 1 or stated[0] <= 0 or _threshold(sub_claim.text) is not None:
        return False
    figures: list[float] = []
    for item in evidence:
        if item.computed_value is None:
            continue
        figures += [abs(v) for v in _money(item.content) if v]
    if not figures:
        return False
    amount = stated[0] * _bare_amount_scale(sub_claim.text, stated[0], max(figures))
    return amount > max(figures) * OUT_OF_WINDOW_FACTOR or amount < min(figures) / OUT_OF_WINDOW_FACTOR


_NEGATOR = re.compile(r"\b(?:not|never|no|without|n't|didn't|doesn't|hasn't|haven't|wasn't|isn't)\s+(?:\w+\s+)?$", re.I)


def _direction(text: str) -> str | None:
    """"up", "down" or None. A negated direction word flips: "did not fall" is up, "did not rise" is down."""
    up = down = False
    for pattern, is_up in ((_UP, True), (_DOWN, False)):
        for m in pattern.finditer(text):
            negated = bool(_NEGATOR.search(text[max(0, m.start() - 20): m.start()]))
            if is_up != negated:
                up = True
            else:
                down = True
    return "up" if up and not down else "down" if down and not up else None


def _stem(word: str) -> str:
    w = word.lower()
    if w in _SYNONYMS:
        return _SYNONYMS[w]
    for suffix in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suffix):
            return w[: -len(suffix)]
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def _stems(text: str, drop: set[str] | None = None) -> set[str]:
    out = set()
    for word in re.findall(r"[A-Za-z]{2,}|\d+", text):
        if word.lower() in _STOP:
            continue
        stem = _stem(word)
        if stem and stem not in (drop or set()):
            out.add(stem)
    return out


def _alias_words(ticker: str | None) -> set[str]:
    words = set()
    for alias in CORPUS_COMPANIES.get(ticker or "", []):
        words |= {w.lower() for w in re.findall(r"[A-Za-z]+", alias)}
    return words


def _polarity(claim: str, passage: str, key_stems: set[str]) -> tuple[int, int]:
    """(claim direction, passage direction) as +1 / -1 / 0, looking only at passage clauses about the claim."""
    claim_dir = {"up": 1, "down": -1, None: 0}[_direction(claim)]
    total = 0
    for clause in re.split(r"\band\b|;|,|\bbut\b|\bwhile\b|\.", passage):
        if not (_stems(clause) & key_stems):
            continue
        up, down = bool(_UP.search(clause)), bool(_DOWN.search(clause))
        total += (1 if up and not down else -1 if down and not up else 0)
    return claim_dir, (1 if total > 0 else -1 if total < 0 else 0)


def _polarity_stems(sub_claim: SubClaim) -> set[str]:
    """The words that say what the sub-claim is about: everything but the company name and the direction words."""
    direction_stems = {_stem(w) for w in _UP.findall(sub_claim.text) + _DOWN.findall(sub_claim.text)}
    return _stems(sub_claim.text, _alias_words(sub_claim.company_ticker)) - direction_stems


# -- comparators the sub-claim states (read from its wording; the contract has no comparator field) -------------

_ABOVE = r"(?:above|over|more than|greater than|higher than|exceed\w*|at least|north of)"
_BELOW = r"(?:below|under|less than|lower than|fewer than|at most|south of)"
_THRESH_PCT = re.compile(
    rf"\b({_ABOVE}|{_BELOW})\s+(?:about |around |roughly )?({_NUM})\s?(?:%|percent\b|per cent\b)",
    re.I,
)
_THRESH_USD = re.compile(
    rf"\b({_ABOVE}|{_BELOW})\s+(?:about |around |roughly )?"
    rf"(-)?\$\s?({_NUM})\s?(bn|billion|b|million|m|thousand|k)?(?![a-z])",
    re.I,
)
_MULTIPLE = {"doubl": (1.9, 2.1), "tripl": (2.85, 3.15), "halv": (0.45, 0.55)}  # doubled: 1.9 to 2.1
_NEGATIVE = re.compile(
    r"\b(?:turned|gone|went|going|is|are|was|were|been|become|became|remains?)?\s*"
    r"(?:negative|in the red|burn(?:ing|ed)? cash)\b",
    re.I,
)
_POSITIVE = re.compile(r"\b(?:turned|went|became|become|is|are|was|were)\s+positive\b", re.I)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_PLAIN_PAIR = re.compile(rf"(-?{_NUM})\s*x?[^\d]{{0,30}}?\b(?:vs\.?|versus|compared with|against)\s*(-?{_NUM})", re.I)


@dataclass
class _Threshold:
    op: str                 # "gt" or "lt"
    bound: float            # a fraction for percents, full USD for amounts
    kind: str               # "pct" or "usd"


def _threshold(text: str) -> _Threshold | None:
    """"above 10%", "less than 10% earnings growth", "over $10 billion": the stated number is a bound, not a target."""
    m = _THRESH_PCT.search(_PERIOD_NOISE.sub(" ", text))
    if m:
        op = "gt" if re.fullmatch(_ABOVE, m.group(1), re.I) else "lt"
        return _Threshold(op, float(m.group(2).replace(",", "")) / 100, "pct")
    m = _THRESH_USD.search(text)
    if m:
        op = "gt" if re.fullmatch(_ABOVE, m.group(1), re.I) else "lt"
        unit = m.group(4).lower() if m.group(4) else None
        value = float(m.group(3).replace(",", "")) * _UNIT[unit]
        return _Threshold(op, -value if m.group(2) else value, "usd")
    return None


def _multiple(text: str) -> tuple[float, float] | None:
    for stem, band in _MULTIPLE.items():
        # "double-digit growth" is a size of growth, not "revenue doubled"
        if re.search(rf"\b{stem}\w*\b(?![\s-]*digit)", text, re.I):

            return band
    return None


def _sign_claim(text: str) -> int:
    """-1 for "turned negative", +1 for "turned positive", 0 when the sub-claim says nothing about the sign."""
    if _POSITIVE.search(text):
        return 1
    if re.search(r"(?<!\$)\bnegative\b(?!\s+\$)|in the red", text, re.I) and not _money(text):
        return -1
    return 0


_FIGURE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent|billion|million|trillion|bn\b|x\b)", re.I)


def _states_checkable(text: str) -> bool:
    """A figure, bound, sign, multiple or direction a filing can confirm. "Very high margins" states none."""
    return bool(_FIGURE.search(text) or _money(text) or _threshold(text) or _sign_claim(text)
                or _multiple(text) or _direction(text))


def _check_threshold(th: _Threshold, metric: str, value: float, item: EvidenceItem) -> bool | None:
    """Does the computed figure sit on the right side of the bound? None when the figure is the wrong kind."""
    if th.kind == "pct":
        if not (_is_growth(metric) or _is_margin(metric) or metric.endswith("_ratio")):
            return None
        return value > th.bound if th.op == "gt" else value < th.bound
    amounts = _money(item.content)
    figure = value if abs(value) >= 1000 else (amounts[0] if amounts else None)
    if figure is None or _is_growth(metric) or _is_margin(metric):
        return None
    return figure > th.bound if th.op == "gt" else figure < th.bound


# -- concepts ------------------------------------------------------------------------------------

def _expected_concepts(sub_claim: SubClaim) -> set[str] | None:
    """Metric concepts the sub-claim is about, or None when the wording does not say.

    A line can name several measures ("lower margins, more of a drag on the free cash flow"), so every measure
    named is expected, not only the first one found."""
    text = mask_other_margins(sub_claim.text.lower())   # shared list: "margin buyers" is not a profit margin
    named: set[str] = set()
    margins: set[str] = set()
    if re.search(r"\bgross (profit )?margin", text):
        margins.add("gross_margin")
    if re.search(r"\boperating (profit |income )?margin", text):
        margins.add("operating_margin")
    if re.search(r"\b(net|profit|net income|net profit) margin", text):
        margins.add("net_margin")
    if not margins and (re.search(r"\bmargin", text) or re.search(r"\bprofitab", text)):
        margins = {"gross_margin", "operating_margin", "net_margin"}
    named |= margins
    if re.search(r"\bfree cash flow|\bfcf\b", text):
        named |= {"free_cash_flow", "fcf_margin"}
    elif re.search(r"\bcash flow|\bcash generation", text):
        named |= {"operating_cash_flow", "free_cash_flow", "fcf_margin"}
    if re.search(r"\bgross (profit|income)", text):
        named.add("gross_profit")
    if re.search(r"\bcurrent ratio", text):
        named.add("current_ratio")
    if re.search(r"\bcash ratio", text):
        named.add("cash_ratio")
    if sub_claim.sub_claim_type.value in ("HISTORICAL_GROWTH", "NET_INCOME_GROWTH"):
        if re.search(r"\beps\b|earnings per share", text):
            named.add("diluted_eps")
        if re.search(r"operating (income|profit)", text):
            named.add("operating_income")
        rest = re.sub(r"\b(gross|operating) (profit|income)s?\b", " ", text)   # "gross profit" is not net income
        if re.search(r"net income|\bearnings\b|\bprofits?\b|bottom line", rest):
            named.add("net_income")
        if re.search(r"\brevenues?\b|\bsales\b|\btop line", text) or not named:
            named.add("revenue")
    return named or None


def _concept(metric_name: str) -> str:
    name = metric_name.lower()
    changed = True
    while changed:                       # "revenue_growth_yoy_qoq" / "revenue_ttm_growth_yoy" -> "revenue"
        changed = False
        for suffix in ("_trend", "_qoq", "_yoy", "_growth", "_ttm", "_context"):
            if name.endswith(suffix):
                name, changed = name[: -len(suffix)], True
    return name


def _is_growth(metric_name: str) -> bool:
    return "_growth" in metric_name.lower()


def _is_margin(metric_name: str) -> bool:
    return _concept(metric_name) in MARGINS


@dataclass
class _Defect:
    evidence_id: str
    reason: str
    hint: str


@dataclass
class _Run:
    defects: list[_Defect] = field(default_factory=list)
    numeric_checks: list[NumericCheck] = field(default_factory=list)
    citation_checks: list[CitationCheck] = field(default_factory=list)
    numeric_ok: list[bool] = field(default_factory=list)   # stated-number comparisons that were possible
    model_words: list[str] = field(default_factory=list)

    def add(self, item: EvidenceItem, reason: str, hint: str) -> None:
        self.defects.append(_Defect(item.evidence_id, reason, hint))


class AuditorAgent:
    name = "auditor_agent"

    def __init__(self, use_llm: bool | None = None) -> None:
        # None: call the LLM only with a real provider (mock mode stays deterministic).
        self._use_llm = use_llm

    # -- entry point ----------------------------------------------------------------------------
    def audit(self, sub_claim: SubClaim, evidence: list[EvidenceItem],
          agent_reason: str | None = None, narrative_review: dict | None = None) -> AuditResult:
        note = _review_note(narrative_review, [])
        if not evidence:
            return AuditResult(sub_claim_id=sub_claim.sub_claim_id, passed=True,
                               verdict=Verdict.INSUFFICIENT_EVIDENCE, review_note=note)
        if _is_long_history(sub_claim.text):
            return AuditResult(sub_claim_id=sub_claim.sub_claim_id, passed=True,
                               verdict=Verdict.INSUFFICIENT_EVIDENCE, review_note=note)
        if _out_of_window(sub_claim, evidence):     # the stated amount belongs to a period the record does not cover
            return AuditResult(sub_claim_id=sub_claim.sub_claim_id, passed=True,
                               verdict=Verdict.INSUFFICIENT_EVIDENCE,
                               evidence_stances={i.evidence_id: Stance.NEUTRAL for i in evidence},
                               review_note=note)
        run = _Run()
        stances: list[Stance] = []
        expected = _expected_concepts(sub_claim)
        on_topic = expected is not None and any(
            i.computed_value is not None and _concept(i.metric_name or "") in expected for i in evidence)
        has_main = any(not (i.metric_name or "").endswith("_context") for i in evidence)
        unusable: list[tuple[EvidenceItem, str]] = []
        for item in evidence:                       # 1. judge the stance (the collector's own label is ignored)
            numeric = item.computed_value is not None
            # A context item is information only. It never moves the label.
            if (item.metric_name or "").endswith("_context") and has_main:
                stances.append(Stance.NEUTRAL)
                unusable.append((item, "context only (team ruling, 18 Oct): it does not change the label"))
                continue
            if (numeric and on_topic and item.metric_name and not (item.metric_name or "").endswith("_context")
                    and _concept(item.metric_name) not in expected):
                stances.append(Stance.NEUTRAL)          # an extra metric next to the right one: ignore it, no retry
                unusable.append((item, f"metric {item.metric_name} is not what the sub-claim is about"))
                continue
            stance, why = (
                self._numeric_stance(sub_claim, item) if numeric
                else self._text_stance(sub_claim, item, run, agent_reason, narrative_review)
            )
            if stance is Stance.NEUTRAL and item.stance is Stance.CHALLENGING:
                stance, why = Stance.CHALLENGING, ""    # collector flagged counter-evidence we cannot confirm: keep it
            stances.append(stance)
            if stance is Stance.NEUTRAL:
                unusable.append((item, why))
                continue
            self._hygiene(sub_claim, item, run)     # 2. only usable items are checked
            if numeric:
                checks_before = len(run.numeric_ok)
                self._numeric(sub_claim, item, run)
                # The stated number disagrees with this item's computed figure: the item is
                # counter-evidence, whatever direction check put it on the supporting side.
                if not all(run.numeric_ok[checks_before:]):
                    stances[-1] = Stance.CHALLENGING
            elif stance is Stance.SUPPORTING:
                run.citation_checks.append(CitationCheck(evidence_id=item.evidence_id, entails=True))
        judged = {item.evidence_id: st for item, st in zip(evidence, stances)}
        used = [(item, st) for item, st in zip(evidence, stances) if st is not Stance.NEUTRAL]
        if not any(c.entails for c in run.citation_checks):
            first_text = next((i for i, _ in unusable if i.computed_value is None), None)
            if first_text is not None and not any(i.computed_value is None for i, _ in used):
                run.citation_checks.append(CitationCheck(evidence_id=first_text.evidence_id, entails=False))
        numeric = next((c for c in run.numeric_checks if not c.matches), run.numeric_checks[0]
                       if run.numeric_checks else None)
        citation = next((c for c in run.citation_checks if not c.entails), run.citation_checks[0]
                        if run.citation_checks else None)
        if not used:                                # items came back, none is about the sub-claim: one retry
            reasons = [f"{i.evidence_id}: {why}" for i, why in unusable]
            hint = ("return passages that state the sub-claim or contradict it (also risk factors and "
                    "qualifying remarks), with an exact citation span; for numbers, state the current and "
                    "prior-year figures in the text.")
            return AuditResult(sub_claim_id=sub_claim.sub_claim_id, passed=False,
                               verdict=Verdict.INSUFFICIENT_EVIDENCE, failure_reasons=reasons,
                               numeric_check=numeric, citation_check=citation, feedback_for_retry=hint,
                               evidence_stances=judged, review_note=_review_note(narrative_review, run.model_words))
        if run.defects:
            reasons = [f"{d.evidence_id}: {d.reason}" for d in run.defects]
            feedback = " ".join(f"{d.evidence_id}: {d.hint}" for d in run.defects)
            return AuditResult(sub_claim_id=sub_claim.sub_claim_id, passed=False,
                               verdict=Verdict.INSUFFICIENT_EVIDENCE, failure_reasons=reasons,
                               numeric_check=numeric, citation_check=citation, feedback_for_retry=feedback,
                               evidence_stances=judged, review_note=_review_note(narrative_review, run.model_words))
        verdict = self._verdict(sub_claim, used, run)
        if verdict is Verdict.SUPPORTED and agent_reason and _NOT_CHECKED.search(agent_reason):
            verdict = Verdict.PARTIALLY_SUPPORTED
        if verdict is Verdict.SUPPORTED and _LONG_HISTORY.search(sub_claim.text):
            # "all-time" or "since 2004" cannot be verified from the corpus window.
            verdict = Verdict.PARTIALLY_SUPPORTED
        return AuditResult(sub_claim_id=sub_claim.sub_claim_id, passed=True,
                           verdict=verdict, numeric_check=numeric,
                           citation_check=citation, evidence_stances=judged,
                           review_note=_review_note(narrative_review, run.model_words))



    # -- verdict --------------------------------------------------------------------------------
    @staticmethod
    def _verdict(sub_claim: SubClaim, used: list[tuple[EvidenceItem, Stance]], run: _Run) -> Verdict:
        supporting = [e for e, st in used if st is Stance.SUPPORTING]
        challenging = [e for e, st in used if st is Stance.CHALLENGING]
        if run.numeric_ok and not any(run.numeric_ok):
            return Verdict.CONTRADICTED                 # the stated number is not the computed number
        if supporting and challenging:
            return Verdict.PARTIALLY_SUPPORTED
        if supporting:
            computed = any(e.computed_value is not None and e.source_tier is SourceTier.TIER_1 for e in supporting)
            if sub_claim.sub_claim_type.value in COMPUTED_TYPES and not computed:
                return Verdict.PARTIALLY_SUPPORTED      # quoted from text, no calculation behind it
            if sub_claim.sub_claim_type.value in COMPUTED_TYPES and not _states_checkable(sub_claim.text):
                return Verdict.INSUFFICIENT_EVIDENCE    # a level word with no bound: the record cannot confirm it
            return Verdict.SUPPORTED
        if challenging:
            return Verdict.CONTRADICTED
        return Verdict.INSUFFICIENT_EVIDENCE

    # -- checks that apply to every item --------------------------------------------------------
    def _hygiene(self, sub_claim: SubClaim, item: EvidenceItem, run: _Run) -> None:
        if not ((item.url or "").strip() or item.publisher_page) or not (item.citation_span or "").strip():
            run.add(item, "no url or citation span, so the quote cannot be checked against its source",
                    "give the source url and the exact citation span.")
        aliases = {a.lower() for a in CORPUS_COMPANIES.get(sub_claim.company_ticker or "", [])}
        if aliases:
            company = item.company.lower()
            if not any(a in company or company in a for a in aliases):
                run.add(item, f"evidence is about {item.company}, not the company in the sub-claim",
                        f"return evidence about {CORPUS_COMPANIES[sub_claim.company_ticker][0]} only.")
        doc = item.doc_type.lower().replace(" ", "_")
        expected = (SourceTier.TIER_1 if doc in TIER1_DOCS else SourceTier.TIER_2 if doc in TIER2_DOCS
                    else SourceTier.TIER_3 if doc in TIER3_DOCS else None)
        if expected is not None and item.source_tier is not expected:
            run.add(item, f"{item.doc_type} is labelled {item.source_tier.value} but is {expected.value}",
                    f"label {item.doc_type} evidence as {expected.value}.")
        period_end = re.search(r"\d{4}-\d{2}-\d{2}", item.period or "")
        on_quarter_end = (item.publication_date.month, item.publication_date.day) in {(3, 31), (6, 30), (9, 30),
                                                                                       (12, 31)}
        if doc in ("10-k", "10-q") and (on_quarter_end or (
                period_end and item.publication_date.isoformat() <= period_end.group(0))):
            run.add(item, f"publication date {item.publication_date} looks like the period end, not the filing date",
                    "use the date the filing was published (filed), not the period end.")

    # -- numbers --------------------------------------------------------------------------------
    def _numeric(self, sub_claim: SubClaim, item: EvidenceItem, run: _Run) -> None:
        metric, value = item.metric_name or "", float(item.computed_value)
        concept = _concept(metric)
        expected = _expected_concepts(sub_claim)
        if metric and expected is not None and concept not in expected:
            run.add(item, f"metric {metric} does not measure what the sub-claim is about",
                    f"return {' or '.join(sorted(expected))} for the whole company, not {concept}.")
        if metric and (_QOQ.search(metric.replace("_", " ")) or _QOQ.search(item.content)) \
                and not _QOQ_CLAIM.search(sub_claim.text):
            run.add(item, "quarter-on-quarter figure used where the sub-claim compares with a year earlier",
                    "compare with the same period one year earlier (year on year).")
        percents = _pcts(item.content)
        if (_is_growth(metric) or _is_margin(metric)) and not metric.endswith("_trend") and percents:
            self._internal_consistency(item, metric, value, percents, run)
        self._stated_number(sub_claim, item, metric, value, percents, run)

    @staticmethod
    def _internal_consistency(item: EvidenceItem, metric: str, value: float, percents: list[_Pct],
                              run: _Run) -> None:
        if _is_growth(metric):
            signed = [p for p in percents if p.signed]
            headline = signed[-1] if signed else percents[0]
        else:
            headline = percents[0]
        shown = (-headline.value if headline.negative else headline.value) / 100
        if not headline.signed:
            shown, check = abs(shown), abs(value)
        else:
            check = value
        tolerance = 0.5 * 10 ** (-headline.decimals) / 100 + 1e-4
        if abs(check - shown) > tolerance:
            scale = abs(value / shown) if shown else 0
            if 90 <= scale <= 110:
                reason = (f"computed_value {value} is a percent; the contract expects a fraction "
                          f"(the text says {headline.value:g}%)")
                hint = "report growth rates and margins as fractions (0.14 for 14%), not percents."
            else:
                reason = f"computed_value {value:.4f} does not match the {headline.value:g}% stated in the text"
                hint = "recompute the figure from the filing values and make the text match."
            run.add(item, reason, hint)
            run.numeric_checks.append(NumericCheck(metric_name=metric, stated=shown, computed=value, matches=False))
            return
        if _is_growth(metric) and re.search(r"\bvs\.?\b|versus|compared with", item.content, re.I):
            amounts = _money(item.content)
            if len(amounts) >= 2 and amounts[1]:
                growth = amounts[0] / amounts[1] - 1
                if abs(growth - value) > MATCH_BAND:
                    run.add(item, f"the two amounts in the text give {growth:+.1%}, not the computed {value:+.1%}",
                            "recompute the growth from the two reported amounts.")
                    run.numeric_checks.append(
                        NumericCheck(metric_name=metric, stated=growth, computed=value, matches=False))
                    return
        run.numeric_checks.append(NumericCheck(metric_name=metric, stated=None, computed=value, matches=True))

    @staticmethod
    def _stated_number(sub_claim: SubClaim, item: EvidenceItem, metric: str, value: float,
                       percents: list[_Pct], run: _Run) -> None:
        """Compare the number the sub-claim states with the computed record."""
        direction = _direction(sub_claim.text)
        threshold = _threshold(sub_claim.text)
        stated_pcts = [] if threshold is not None else _pcts(sub_claim.text)
        matches: list[bool] = []
        stated_frac = None
        if threshold is not None:
            met = _check_threshold(threshold, metric, value, item)
            if met is not None:
                matches.append(met)
        sign = _sign_claim(sub_claim.text)
        if sign and not _is_growth(metric):
            matches.append(value != 0 and (value < 0) == (sign < 0))
        band = _multiple(sub_claim.text)
        if band and _is_growth(metric):
            matches.append(band[0] <= 1 + value <= band[1])
        if stated_pcts and (_is_growth(metric) or _is_margin(metric)):
            first = stated_pcts[0]
            stated_frac = first.value / 100
            if _is_growth(metric):
                signed = -stated_frac if (first.negative or (direction == "down" and not first.signed)) \
                    else stated_frac
                matches.append(abs(signed - value) <= MATCH_BAND + 1e-9)
            else:
                candidates = [value]                       # "margin is 52%"
                if len(percents) >= 2:                     # "margin fell 2.2 points" / "fell 4%"
                    cur, prev = percents[0].value / 100, percents[1].value / 100
                    candidates += [abs(cur - prev)]
                    candidates += [abs((cur - prev) / prev)] if prev else []
                matches.append(any(abs(stated_frac - c) <= MATCH_BAND + 1e-9 for c in candidates))
        stated_money = [] if threshold is not None else _money(sub_claim.text)
        content_money = _money(item.content)
        if len(stated_money) == 1 and content_money:
            scale = _bare_amount_scale(sub_claim.text, stated_money[0], content_money[0])
            stated_amount = stated_money[0] * scale  # "$110 in Q1" is $110 billion when the record is $109,896m
            tol = max(                               # rounding to the precision the claim stated
                _money_half_unit(sub_claim.text) * scale, 1e-9 * abs(content_money[0]))
            matches.append(abs(stated_amount - content_money[0]) <= tol)
            if stated_frac is None:
                stated_frac = stated_money[0]       # the claimed dollar amount, shown next to the computed one
        if not matches:
            return
        ok = all(matches)
        run.numeric_ok.append(ok)
        run.numeric_checks.append(NumericCheck(metric_name=metric, stated=stated_frac, computed=value, matches=ok))

    # -- stance ---------------------------------------------------------------------------------
    @staticmethod
    def _movement(metric: str, value: float, item: EvidenceItem) -> float | None:
        """Change over the comparison period: the growth rate, or current minus prior from the item's own text."""
        if _is_growth(metric) and not metric.endswith("_trend"):
            return value
        prior = getattr(item, "comparison_value", None)      # level metric: the same figure a year earlier (schema 0.7)
        if prior is not None and not metric.endswith("_trend"):
            return value - float(prior)
        percents = _pcts(item.content)
        if _is_margin(metric) and len(percents) >= 2:
            return percents[0].value - percents[1].value
        amounts = _money(item.content)
        if len(amounts) >= 2 and not _is_margin(metric):
            return amounts[0] - amounts[1]
        # "as at 2026-06-30 vs 2.98x": the date is not a figure.
        pair = _PLAIN_PAIR.search(_DATE.sub(" ", item.content))
        if pair and not percents and not amounts:
            return float(pair.group(1).replace(",", "")) - float(pair.group(2).replace(",", ""))
        return None

    def _numeric_stance(self, sub_claim: SubClaim, item: EvidenceItem) -> tuple[Stance, str]:
        """Does the computed figure back the direction the sub-claim states? (The collector's label is ignored.)"""
        metric, value = item.metric_name or "", float(item.computed_value)
        threshold = _threshold(sub_claim.text)
        if threshold is not None:                   # "above 10%" / "less than 10% growth": a bound, not a direction
            met = _check_threshold(threshold, metric, value, item)
            if met is not None:
                return (Stance.SUPPORTING if met else Stance.CHALLENGING), ""
        sign = _sign_claim(sub_claim.text)
        if sign and not _is_growth(metric):         # "turned negative": the computed figure's own sign decides
            return (Stance.SUPPORTING if (value < 0) == (sign < 0) and value != 0 else Stance.CHALLENGING), ""
        band = _multiple(sub_claim.text)
        if band and _is_growth(metric):             # "doubled": 1 + growth must sit in the band
            return (Stance.SUPPORTING if band[0] <= 1 + value <= band[1] else Stance.CHALLENGING), ""
        direction = _direction(sub_claim.text)
        if direction is None and _is_growth(metric):
            direction = "up"                        # "revenue growth is strong": positive growth backs it
        if direction is None:
            return Stance.SUPPORTING, ""            # a level claim: a wrong stated number is caught elsewhere
        change = self._movement(metric, value, item)
        if change is None:
            return Stance.NEUTRAL, "cannot tell from the figure and its text whether it moved up or down"
        backs = change > 0 if direction == "up" else change < 0
        return (Stance.SUPPORTING if backs else Stance.CHALLENGING), ""

    def _text_stance(
        self, sub_claim: SubClaim, item: EvidenceItem, run: _Run,
        agent_reason: str | None, narrative_review: dict | None,
    ) -> tuple[Stance, str]:
        """SUPPORTING if the passage states the sub-claim, CHALLENGING if it rules it out or says the opposite,
        otherwise NEUTRAL with the reason. Rule-based; with a real LLM the model may only downgrade or challenge."""
        text, passage = sub_claim.text, item.content
        if sub_claim.sub_claim_type.value == "CONTINUATION":
            return self._continuation_stance(sub_claim, item, run, agent_reason, narrative_review)
        # Words that only say WHEN ("Q4 2025") or HOW MUCH the bound is ("over 100%") are not the topic: leave them
        # out of the key words, or a passage that states the claim in other words fails the coverage test.
        threshold = _threshold(text)
        topic_text = _THRESH_PCT.sub(" ", text) if threshold is not None and threshold.kind == "pct" else text
        key = _stems(_PERIOD_NOISE.sub(" ", topic_text), _alias_words(sub_claim.company_ticker))
        distinctive = key - GENERIC_STEMS
        coverage = len(key & _stems(passage)) / len(key) if key else 0.0
        for cue in _NEGATED_ATTRIBUTION.finditer(passage):
            if _stems(passage[cue.end(): cue.end() + 60]) & distinctive:
                return Stance.CHALLENGING, "it says the opposite: the claimed driver is ruled out"
        pol_stems = _polarity_stems(sub_claim)
        claim_dir, text_dir = _polarity(text, passage, pol_stems)
        pol_cover = len(pol_stems & _stems(passage)) / len(pol_stems) if pol_stems else 0.0
        if claim_dir and text_dir and claim_dir != text_dir and pol_cover >= MIN_COVERAGE:
            return Stance.CHALLENGING, "it says the opposite direction"
        rules_ok, why = True, ""
        if sub_claim.sub_claim_type.value not in HEDGE_OK_TYPES and _HEDGE.search(passage):
            rules_ok, why = False, "hedged wording is not proof"
        bound_ok = None
        if threshold is not None and threshold.kind == "pct":     # "over 100%": the passage must be on that side
            figures = [p.value / 100 * (-1 if p.negative else 1)
                       for clause in re.split(r"\.(?!\d)|;|\bbut\b|\bwhile\b", passage)
                       if _stems(clause) & distinctive for p in _pcts(clause)]
            if figures:
                bound_ok = any(f > threshold.bound if threshold.op == "gt" else f < threshold.bound for f in figures)
                if not bound_ok and coverage >= MIN_COVERAGE:
                    return Stance.CHALLENGING, "the figure in the passage is on the other side of the stated bound"
            elif rules_ok:
                rules_ok, why = False, "the passage has no figure for the bound in the sub-claim"
        stated_pct = {p.value for p in _pcts(_PERIOD_NOISE.sub(" ", topic_text))}
        stated_usd = {float(m.group(2).replace(",", "")) for m in _MONEY_RE.finditer(text)}
        shown_pct = {p.value for p in _pcts(passage)}
        shown_usd = {float(m.group(2).replace(",", "")) for m in _MONEY_RE.finditer(passage)}
        if rules_ok and ((stated_pct - shown_pct) or (stated_usd - shown_usd)):
            rules_ok, why = False, "the number in the sub-claim is not in the passage"
        if rules_ok and coverage < MIN_COVERAGE:
            rules_ok, why = False, f"only {coverage:.0%} of the sub-claim's key words appear"
        if self._llm_enabled():     # no word-overlap gate: a challenge often shares no words with the claim;
            # an off-topic passage comes back NEUTRAL and is ignored, and the model can never create support.
            verdict = self._ask_llm(text, passage, agent_reason, narrative_review)
            if verdict:
                run.model_words.append(f"{item.evidence_id} {verdict}")
            if verdict == "CONTRADICTS":
                return Stance.CHALLENGING, "the model judged the passage contradicts the sub-claim"
            if rules_ok and verdict not in (None, "ENTAILS"):
                return Stance.NEUTRAL, f"the model judged the passage {verdict.lower()}"
            # Filler can miss the word-overlap bar. ENTAILS may confirm a passage that already has the figure.
            if verdict == "ENTAILS" and not rules_ok and why.startswith("only "):
                return Stance.SUPPORTING, ""
        if rules_ok:
            return Stance.SUPPORTING, ""
        return Stance.NEUTRAL, f"the passage does not state or contradict the sub-claim ({why})"

    def _continuation_stance(
        self, sub_claim: SubClaim, item: EvidenceItem, run: _Run,
        agent_reason: str | None, narrative_review: dict | None,
    ) -> tuple[Stance, str]:
        """"Growth is just getting started": what counts is a forward-looking statement (guidance, outlook, "expects"),
        not word overlap with the claim's wording. Pressure / decline / "remain elevated" in it is counter-evidence.
        Rule-based approximation; the knowledge layer caps this type at PARTIALLY_SUPPORTED, Tier 2."""
        passage = item.content
        if not _FORWARD.search(passage):
            return Stance.NEUTRAL, "the passage has no forward-looking statement (guidance, outlook, expectation)"
        against = bool(_FORWARD_AGAINST.search(passage) or _DOWN.search(passage))
        stance = Stance.CHALLENGING if against else Stance.SUPPORTING if _FORWARD_FOR.search(passage) else None
        if stance is None:
            return Stance.NEUTRAL, "the forward-looking statement says nothing about growth continuing or slowing"
        if self._llm_enabled():
            verdict = self._ask_llm(sub_claim.text, passage, agent_reason, narrative_review)
            if verdict:
                run.model_words.append(f"{item.evidence_id} {verdict}")
            if verdict == "CONTRADICTS":
                return Stance.CHALLENGING, "the model judged the passage contradicts the sub-claim"
            if stance is Stance.SUPPORTING and verdict not in (None, "ENTAILS"):
                return Stance.NEUTRAL, f"the model judged the passage {verdict.lower()}"
        return stance, ""

    def _llm_enabled(self) -> bool:
        if self._use_llm is not None:
            return self._use_llm
        return get_settings().llm_provider != "mock"

    def _ask_llm(
        self, sub_claim_text: str, passage: str,
        agent_reason: str | None, narrative_review: dict | None,
    ) -> str | None:
        """Ask whether the passage establishes the claim. A shared percent is not enough.

        OpenAI uses gpt-5.4. Other providers keep the configured model. The answer is one word.
        """
        blocks = [data_block("sub_claim", sub_claim_text), data_block("passage", passage)]
        if agent_reason:
            blocks.append(data_block("financial_note", agent_reason))
        review = narrative_review or {}
        if review.get("stance"):
            blocks.append(data_block("narrative_stance", str(review["stance"])))
        if review.get("stance_reason"):
            blocks.append(data_block("narrative_reason", str(review["stance_reason"])))
        summary = [sentence for sentence in (review.get("summary") or []) if isinstance(sentence, str) and sentence]
        if summary:
            blocks.append(data_block("narrative_summary", " ".join(summary)))
        model = "gpt-5.4" if get_settings().llm_provider == "openai" else None
        resp = complete(
            system=DATA_SAFETY_PREAMBLE + (
                "You check whether a quoted passage establishes a sub-claim. "
                "A shared percentage or a few shared words is not enough. "
                "Weigh every material part of the sentence. "
                "Now at the start of a spoken sentence is a change of topic, unless the sentence uses it "
                "to claim the result is happening at the present moment rather than reporting a figure. "
                "A financial note and a narrative stance are other agents' work. Read them, then decide yourself. "
                "Answer with exactly one word: ENTAILS, NEUTRAL or CONTRADICTS."
            ),
            prompt="\n".join(blocks),
            model=model,
            max_tokens=1000,
        )
        match = re.search(r"\b(ENTAILS|NEUTRAL|CONTRADICTS)\b", resp.text.upper())
        return match.group(1) if match else None


def _review_note(narrative_review: dict | None, model_words: list[str]) -> str | None:
    """The narrative stance beside the auditor model's word. Neither one is the verdict by itself."""
    parts: list[str] = []
    review = narrative_review or {}
    stance = review.get("stance")
    if isinstance(stance, str) and stance:
        reason = review.get("stance_reason")
        parts.append(f"Narrative {stance}" + (f": {reason}" if isinstance(reason, str) and reason else ""))
    if model_words:
        parts.append("Auditor model: " + "; ".join(model_words))
    text = " ".join(parts).strip()
    return text or None
