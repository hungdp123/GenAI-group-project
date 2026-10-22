""" Financial Analysis Agent

Input (one claim per sub-claim; see README section 1):
  sub_claim.sub_claim_id, sub_claim.company_ticker, sub_claim.text   one claim: one company, one measure, with the
                                                                   period and stated figure if the influencer gave them
  cutoff_date                                                        resolved by the orchestrator; None = latest data
  feedback                                                           the Auditor's reason on the retry, else None
  (optional) sub_claim.user_prompt                                 only needed when `text` is not narrowed to one claim

How it works (an LLM understands the claim, Python computes, code enforces the rules):
  1. Understand: the model (team LLM client) reads the claim and decides what it needs. Its first action is
     always `understand`: from the words alone it lists each statement (measure, period, stated figure,
     direction, in scope or not); only then is the rule-based reading shown, as a cross-check. It then works one
     action at a time (planner.py), sees each result before its next step, and the Auditor's feedback on a
     retry. Its steps are in `agent.traces[sub_claim_id]`, its reading in `agent.readings[sub_claim_id]`.
     Scope: an explicit `measure` limits what it may report (see SCOPE_BY_MEASURE), because one sentence can
     arrive as several sub-claims with the same text ("good company" -> four). The type is only a hint.
  2. Compute: every action runs through deterministic code (tools.py): the formulas, the cut-off and the window
     cannot be bypassed, and only tool results become evidence. The model never writes a number.
  3. Fall back: when no model answers in the action format (the default mock provider, no API key, the spend
     limit), the rule-based reading (claim_reader.py) decides instead. It is also the model's hint.
     `agent.modes[sub_claim_id]` says which one was used. FINSIGHT_FINANCIAL_PLANNER=rules forces the rules.

Measures (knowledge_layer.yaml v0.6):
  growth   revenue, operating income, operating cash flow, diluted EPS, net income: year-on-year growth
  levels   gross / operating / net margin, free cash flow, free cash flow margin, current ratio, cash ratio:
           the figure and the same figure a year earlier (free cash flow also gets its growth rate when the
           claim states a percentage change)
  nothing  debt to equity (debt is not mapped), and what the YAML leaves to the narrative agent: a segment or
           product, a run rate, gross profit, adjusted / non-GAAP figures, net income excluding gains, forecasts

Which period (rule book section 11):
  - none named: the latest fiscal year vs the year before (the trailing twelve months go in the notes as context)
  - "trailing 12 months", "last twelve months", "TTM": the trailing twelve months
  - "recent" or "quarter": the latest quarter vs a year earlier; "last two quarters": each of the two
  - a named period ("Q1 2026", "Q4 of 2025", "the March quarter", "Q1 FY2026", "fiscal 2025"): that period
  - a margin or ratio stated in the present with a number ("is 34%"): the latest quarter
  - a streak within the window: one growth item per period
Window: the latest 10-K plus the latest four quarters filed by the cut-off. A period outside it gets nothing.

Output to the Auditor: list[EvidenceItem], stance NEUTRAL, metric names and content shapes as agreed:
  growth   "Revenue was $109,896m in the quarter ended 2026-03-31 vs $90,234m in the quarter ended 2025-03-31
            (+21.8% YoY)."   computed_value 0.218; comparison_value = the amount for the period
  level    "Operating margin was 40.6% in the quarter ended 2026-03-31 vs 41.5% in the quarter ended 2025-03-31."
            computed_value 0.406; comparison_value = the figure a year earlier
An empty list means nothing could be calculated; `agent.reasons[sub_claim_id]` says why (for no_evidence_reason).
It also lists the parts of a claim that were not checked, and context figures.

Data: the pulled SEC facts in data/sec/processed (see pull.py)
"""

from __future__ import annotations

import calendar
import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from importlib import resources
from pathlib import Path

from finsight.contracts.schemas import EvidenceItem, SourceTier, Stance, SubClaim
from finsight.llm.usage import record_tool_call

from . import planner
from .claim_reader import (GROWTH_MEASURES, LEVEL_MEASURES, LEVERAGE, SEGMENT_NOTE, Point, Reading,
                           minimum_percent, read_claim)
from .concepts import CONCEPTS, TAG_TO_CONCEPT, kind, unit
from .schema import FinancialEvidence
from .store import DEFAULT_DATA_DIR, FactStore, _iso, _read_csv
from .tools import (_ANCHORS, _near, _period_ends, compute_growth, compute_metric, compute_ttm,
                    compute_ttm_growth, get_financial_fact, prior_year_end)

try:   # the report shows the spoken company name ("Alphabet"), from Claim extraction's thesaurus
    from finsight.agents.claim_extraction.thesaurus import display_name as _thesaurus_name
except ImportError:   # pragma: no cover - thesaurus not in this checkout
    _thesaurus_name = None

# DECISION: the window is the latest 10-K plus the latest four quarters filed by the cut-off
# (knowledge_layer.md section 4). A claimed period outside it gets no evidence.
WINDOW = {"quarter": 4, "annual": 1}
# A streak can be checked only as far as the window reaches: four quarterly growth rates, or the two
# year-on-year growth rates the latest 10-K's three years of figures give.
STREAK_LIMIT = {"quarter": 4, "annual": 2}
# Results are usually announced (earnings release) up to this many days before the 10-Q / 10-K is filed.
ANNOUNCE_DAYS = 10

LEVERAGE_REASON = ("Debt is not mapped in the SEC data (companies tag it too differently to trust), "
                   "so debt to equity is not calculated.")
SEGMENT_REASON = "A segment or product figure: the SEC data only has company-wide totals."
NO_FIGURE_REASON = "The claim states no figure the SEC data can check."
# (to confirm with the Auditor owner) what goes in EvidenceItem.company: "ticker" (GOOGL) or "name" (Alphabet)
COMPANY_FIELD = "ticker"

# DECISION: which measures one sub-claim may report. Only an explicit `measure` limits it. One sentence can become
# several sub-claims with the same text: a section 6 expansion ("good company" -> four children told apart only by
# `measure`), or a margin sentence plus a growth child with measure="operating_income" (extract._with_growth).
# The sub_claim_type alone does NOT limit it: on claims_updated.jsonl (Oct 2026) the stored types are coarse (the
# merged PROFITABILITY_CASH_LEVERAGE split by keyword at load), and a type filter dropped a stated measure in 401
# of 1,733 financial sub-claims, e.g. "less than 10% earnings growth" typed PROFITABILITY, "operating income shot up
# 124% giving them a 66.2% operating margin" typed HISTORICAL_GROWTH. The type is shown to the model as a hint.
SCOPE_BY_MEASURE = {"operating_income": {"OperatingIncome"}, "diluted_eps": {"DilutedEPS"},
                    "operating_cash_flow": {"OperatingCashFlow"}, "fcf_margin": {"fcf_margin"},
                    "profitability": {"gross_margin", "operating_margin", "net_margin"},
                    "cash_generation": {"free_cash_flow", "fcf_margin", "OperatingCashFlow"},
                    "liquidity": {"current_ratio", "cash_ratio"}, "leverage": {LEVERAGE}}


def _type_name(sub_claim) -> str:
    kind = getattr(sub_claim, "sub_claim_type", None)
    return getattr(kind, "name", None) or getattr(kind, "value", None) or ""


def scope(sub_claim) -> set[str] | None:
    """The measures this sub-claim may report, or None for no limit (only an explicit `measure` limits it)."""
    measure = getattr(sub_claim, "measure", None)
    return set(SCOPE_BY_MEASURE[measure]) if measure in SCOPE_BY_MEASURE else None


def scope_text(sub_claim, allowed: set[str] | None) -> str:
    kind, measure = _type_name(sub_claim), getattr(sub_claim, "measure", None)
    if allowed is None:
        return (f"every measure the line states about this company. Claim extraction typed it {kind or 'unknown'}; "
                "that type comes from keywords and is often wrong, so read the words.") if kind else ""
    return (f"{kind} / {measure}: report only {', '.join(sorted(allowed))}. The same words were sent as other "
            "sub-claims for the other measures: note them in `understand` with in_scope false and do not compute "
            "them.")


_FLOW_METRICS = {"gross_margin", "operating_margin", "net_margin", "free_cash_flow", "fcf_margin"}  # 12-month figure
_NAMES = {"Revenue": "revenue", "NetIncome": "net_income", "OperatingIncome": "operating_income",
          "OperatingCashFlow": "operating_cash_flow", "DilutedEPS": "diluted_eps", "gross_margin": "gross_margin",
          "operating_margin": "operating_margin", "net_margin": "net_margin", "free_cash_flow": "free_cash_flow",
          "fcf_margin": "fcf_margin", "current_ratio": "current_ratio", "cash_ratio": "cash_ratio"}
_LABELS = {"Revenue": "Revenue", "NetIncome": "Net income", "OperatingIncome": "Operating income",
           "OperatingCashFlow": "Operating cash flow", "DilutedEPS": "Diluted EPS", "gross_margin": "Gross margin",
           "operating_margin": "Operating margin", "net_margin": "Net margin", "free_cash_flow": "Free cash flow",
           "fcf_margin": "Free cash flow margin", "current_ratio": "Current ratio", "cash_ratio": "Cash ratio",
           LEVERAGE: "Debt to equity"}
_MARGINS = {"gross_margin", "operating_margin", "net_margin", "fcf_margin"}
_BALANCE_METRICS = {"current_ratio", "cash_ratio"}          # built from balance-sheet figures: "as at" a date
_PCT = r"(-?\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)"


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text or "", flags=re.IGNORECASE) is not None


def user_prompt(sub_claim) -> str | None:
    """What the user typed, if the orchestrator put it on the sub-claim."""
    for name in ("user_prompt", "prompt", "user_claim"):
        value = getattr(sub_claim, name, None)
        if value:
            return str(value)
    return None


def period_field(sub_claim) -> str | None:
    """Claim extraction's period: `sub_claim.period`, or `fields.period` as stored in claims_updated.jsonl."""
    value = getattr(sub_claim, "period", None)
    if not value:
        fields = getattr(sub_claim, "fields", None)
        value = fields.get("period") if isinstance(fields, dict) else getattr(fields, "period", None)
    return value or None


@dataclass
class PeriodRef:
    """The period(s) to report. ends: concrete period ends, latest first (empty when not usable; reason says
    why). latest: the claim means the latest period rather than a dated one."""
    span: str
    ends: list[str] = field(default_factory=list)
    label: str = ""
    reason: str = ""
    named: bool = False       # a period is named (in the prompt, the line or the period field)
    ttm: bool = False         # trailing twelve months requested
    latest: bool = False

    @property
    def end(self) -> str | None:
        return self.ends[0] if self.ends else None


def _month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def _replace_year(d: date, y: int) -> date:
    return d.replace(year=y, day=min(d.day, calendar.monthrange(y, d.month)[1]))


def resolve_period(store: FactStore, ticker: str, spec: tuple | None, cutoff: date | None) -> PeriodRef:
    """Turn a period spec into period ends known at the cut-off. Does not apply the window (see the agent)."""
    if spec is None:                                       # rule book: no period named = latest fiscal year
        return PeriodRef("annual", label="latest fiscal year", latest=True)
    if spec[0] == "ttm":
        return PeriodRef("quarter", label="trailing twelve months", named=True, ttm=True, latest=True)
    if spec[0] == "last_n":
        return PeriodRef("quarter", label=f"last {spec[1]} quarters", named=True, latest=True)
    if spec[0] == "latest":
        return PeriodRef(spec[1], label=f"latest {spec[1]}", named=True, latest=True)
    q_ends = [date.fromisoformat(e) for e in _period_ends(store, ticker, cutoff, "quarter")]
    fy_ends = [date.fromisoformat(e) for e in _period_ends(store, ticker, cutoff, "annual")]
    ref = date.fromisoformat(_iso(cutoff)) if cutoff else (q_ends[-1] if q_ends else date.today())
    what = spec[0]
    span = "annual" if what == "fy" else "quarter"

    if what in ("cal_q", "month_q"):
        month = spec[2] * 3 if what == "cal_q" else spec[2]
        year = spec[1] or (ref.year if _month_end(ref.year, month) <= ref else ref.year - 1)
        target = _month_end(year, month)
        label = f"quarter ending around {target}"
    else:
        if not fy_ends:
            return PeriodRef(span, label="fiscal period", reason="the company's fiscal year-end is not in the data",
                             named=True)
        anchor = fy_ends[-1]                              # month and day of this company's fiscal year-end

        def fy_end(y):
            return _replace_year(anchor, y)

        if what == "fy":
            target = fy_end(spec[1])
            label = f"fiscal year ending around {target}"
        else:
            q = spec[2]
            fy = spec[1]
            if fy is None:                                # most recent fiscal quarter q that ended by `ref`
                fy = ref.year + 1
                while fy_end(fy) - timedelta(days=91 * (4 - q)) > ref:
                    fy -= 1
            target = fy_end(fy) - timedelta(days=91 * (4 - q))
            label = f"fiscal Q{q} {fy} (ending around {target})"

    known = fy_ends if span == "annual" else q_ends
    best = min(known, key=lambda e: abs((e - target).days), default=None)
    if best is not None and abs((best - target).days) <= 10:
        return PeriodRef(span, [best.isoformat()], label, named=True)
    if cutoff and target > ref:
        why = f"the {label} had not ended by the cut-off {ref}"
    elif not known or target > known[-1]:
        why = f"no filing for the {label} was filed by " + (f"the cut-off {ref}" if cutoff else "the latest data")
    else:
        why = f"the {label} is not in the data"
    return PeriodRef(span, [], label, why, named=True)


def _next_quarter_end(d: date) -> date:
    """The quarter end after `d`: three calendar months on for a month-end, else 13 weeks (52/53-week years)."""
    if d.day == calendar.monthrange(d.year, d.month)[1]:
        m = d.month + 3
        return _month_end(d.year + (m - 1) // 12, (m - 1) % 12 + 1)
    return d + timedelta(days=91)


def in_window(store: FactStore, ticker: str, period_end: str, span: str, cutoff: date | None) -> bool:
    """True if the period is the latest fiscal year or one of the latest four quarters filed by the cut-off."""
    return period_end in _period_ends(store, ticker, cutoff, span)[-WINDOW[span]:]


def _first_filed(store: FactStore, ticker: str, cutoff: date | None) -> dict[str, tuple[str, str]]:
    """{period end: (date first reported, form)} from the anchor concepts, as known at the cut-off."""
    out = {}
    for concept in _ANCHORS:
        for r in store.visible(ticker, concept, cutoff):
            e = r["period_end"]
            if e not in out or r["filed"] < out[e][0]:
                out[e] = (r["filed"], r["form"])
    return out


def unreported_newer_quarter(store: FactStore, ticker: str, cutoff: date | None) -> str | None:
    """Reason text when "the latest quarter" is unclear at the cut-off, else None.

    A newer quarter than the latest one filed has ended, and the cut-off is within ANNOUNCE_DAYS of when this
    company usually files (or later). The influencer may well be quoting that newer quarter from the earnings
    release, so the filed quarter is not safe to report as "the latest quarter".
    Example: Meta, cut-off 29 Apr 2026: results for the March quarter were out, the 10-Q came on 30 Apr."""
    if cutoff is None:
        return None
    first = _first_filed(store, ticker, cutoff)
    if not first:
        return None
    cut = date.fromisoformat(_iso(cutoff))
    latest = date.fromisoformat(max(first))
    nxt = _next_quarter_end(latest)
    fy_ends = [date.fromisoformat(e) for e in _period_ends(store, ticker, cutoff, "annual")]
    year_end = bool(fy_ends) and abs((_replace_year(fy_ends[-1], nxt.year) - nxt).days) <= 10
    lags = [(date.fromisoformat(filed) - date.fromisoformat(end)).days
            for end, (filed, form) in first.items() if form.startswith("10-K") == year_end]
    if not lags or cut < nxt + timedelta(days=max(0, min(lags) - ANNOUNCE_DAYS)):
        return None
    report = "10-K" if year_end else "10-Q"
    return (f"The quarter ending around {nxt} had ended by the cut-off {cut} and its results were probably "
            f"announced ({ticker} usually files its {report} about {min(lags)} days after the quarter ends), "
            f"but no filing for it is in the data by the cut-off. 'The latest quarter' could mean either quarter.")


# ------------------------------------------------------------------ data loading
def _demo_rows() -> list[dict]:
    """The placeholder fixture (finsight/data/xbrl_facts.json) converted to fact rows. Values are USD millions."""
    raw = json.loads(resources.files("finsight.data").joinpath("xbrl_facts.json").read_text("utf-8"))
    rows = []
    for ticker, body in raw.items():
        if not isinstance(body, dict):
            continue
        for f in body.get("facts", []):
            concept = TAG_TO_CONCEPT.get(f["metric"])
            if not concept:
                continue
            cls = "I" if kind(concept) == "instant" else "FY" if f["period"].upper().startswith("FY") else "Q"
            rows.append(dict(
                ticker=ticker, cik=0, concept=concept, xbrl_tag=f["metric"],
                value=f["value"] * (1e6 if f.get("unit") == "USD_m" else 1), unit=unit(concept), period_start="",
                period_end=f["period_end"], period_class=cls, form=f["form"], accession=f["accession"],
                filed=f["filed"], source_url=f"https://example.com/sec/{ticker.lower()}/{f['accession']}"))
    return rows


@lru_cache(maxsize=4)
def _load_store(processed_dir: str, mtime: float) -> FactStore:
    facts, filings = _demo_rows(), []
    path = Path(processed_dir)
    if mtime:
        real = _read_csv(path / "facts.csv")
        real_tickers = {r["ticker"] for r in real}
        facts = [r for r in facts if r["ticker"] not in real_tickers] + real
        if (path / "filings.csv").exists():
            filings = _read_csv(path / "filings.csv")
    return FactStore(facts, filings)


def default_store() -> FactStore:
    """Pulled SEC data if present, plus the demo fixture. Reloaded when facts.csv changes."""
    processed = Path(os.getenv("FINSIGHT_SEC_DATA", str(DEFAULT_DATA_DIR / "processed")))
    facts_csv = processed / "facts.csv"
    return _load_store(str(processed), facts_csv.stat().st_mtime if facts_csv.exists() else 0.0)


def company_display_name(ticker: str) -> str:
    """'Alphabet' for GOOGL, from the thesaurus; the ticker when the thesaurus does not know it."""
    if _thesaurus_name is not None:
        return _thesaurus_name(ticker) or ticker
    return ticker


# ------------------------------------------------------------------ formatting (the Auditor reads these shapes)
def _amount(item: str, value: float, unit_: str) -> str:
    """$ millions with commas ("$109,896m"), a percentage for margins ("40.6%"), "1.50x" for ratios, "$2.09" for EPS."""
    if item in _MARGINS:
        return f"{value:.1%}"
    if unit_ == "ratio":
        return f"{value:.2f}x"
    if unit_ == "USD/shares":
        return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"
    m = value / 1e6
    return f"-${abs(m):,.0f}m" if m < 0 else f"${m:,.0f}m"


def _period(ev: FinancialEvidence) -> str:
    if ev.span == "ttm":
        return f"twelve months ended {ev.period_end}"
    is_balance = (ev.metric in CONCEPTS and kind(ev.metric) == "instant") or ev.metric in _BALANCE_METRICS
    word = "as at" if is_balance else "year ended" if ev.span == "annual" else "quarter ended"
    return f"{word} {ev.period_end}"


def _in(period: str) -> str:
    """'in the quarter ended …' / 'as at …'"""
    return period if period.startswith("as at") else f"in the {period}"


def _doc_type(form: str) -> str:
    """The Auditor accepts 10-K and 10-Q: an amendment (10-K/A, 10-Q/A) is reported as its base form."""
    return "10-K" if form.upper().startswith("10-K") else "10-Q" if form.upper().startswith("10-Q") else form


def _company(ticker: str) -> str:
    return ticker if COMPANY_FIELD == "ticker" else company_display_name(ticker)


class XbrlFinancialAgent:
    name = "financial_xbrl"

    def __init__(self, store: FactStore | None = None) -> None:
        self.store = store if store is not None else default_store()
        # sub_claim_id -> what was not checked and why (the orchestrator copies it to `no_evidence_reason`)
        self.reasons: dict[str, str] = {}
        self.traces: dict[str, str] = {}       # sub_claim_id -> the planner's steps (thought, action, result)
        self.modes: dict[str, str] = {}        # sub_claim_id -> "llm" or "rules"
        self.readings: dict[str, list[dict]] = {}   # sub_claim_id -> the model's `understand` parts

    def _nothing(self, sub_claim: SubClaim, reason: str) -> list[EvidenceItem]:
        self.reasons[sub_claim.sub_claim_id] = reason
        return []

    # -------------------------------------------------------------- entry point
    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]:
        """Evidence for one claim. The model reads the claim and plans which figures to compute (planner.py);
        the rule-based reading (claim_reader.py) is its hint and the fallback when no model is available.
        `feedback` (the Auditor's note on a retry) is shown to the model."""
        sid = sub_claim.sub_claim_id
        for log in (self.reasons, self.traces, self.modes, self.readings):
            log.pop(sid, None)
        ticker, text = sub_claim.company_ticker, sub_claim.text
        if not ticker:
            return self._nothing(sub_claim, "No company on the sub-claim.")
        if ticker not in self.store.tickers:
            return self._nothing(sub_claim, f"{ticker} is not in the SEC data.")
        prompt = user_prompt(sub_claim)                         # optional: only when `text` is not one claim
        allowed = scope(sub_claim)
        reading = self._scoped(read_claim(text, ticker, prompt, period_field(sub_claim)), allowed)
        if os.getenv("FINSIGHT_FINANCIAL_PLANNER", "llm").lower() != "rules":
            try:
                items, notes, plan = self._plan(sub_claim, cutoff_date, feedback, prompt, reading, allowed)
                self.modes[sid], self.traces[sid], self.readings[sid] = "llm", plan.trace(), plan.reading
                return self._finish(sub_claim, items, notes)
            except planner.PlannerUnavailable as e:
                self.traces[sid] = f"Planner not used ({e}); the rule-based reading was used."
        self.modes[sid] = "rules"
        items, notes = [], list(reading.notes)
        for point in reading.points:
            new, why = self._answer(sub_claim, cutoff_date, point)
            items += new
            notes += why
        return self._finish(sub_claim, items, notes)

    @staticmethod
    def _scoped(reading: Reading, allowed: set[str] | None) -> Reading:
        """The rule-based reading limited to this sub-claim's scope; what falls outside is noted."""
        if allowed is None:
            return reading
        kept = [p for p in reading.points if p.measure in allowed]
        left = sorted({_LABELS.get(p.measure, p.measure) for p in reading.points if p.measure not in allowed})
        notes = list(reading.notes)
        if left:
            notes.append("Also in the line but outside this sub-claim's scope, so not reported here: "
                         + ", ".join(left))
        return Reading(points=kept, notes=notes)

    def _finish(self, sub_claim: SubClaim, items: list[EvidenceItem], notes: list[str]) -> list[EvidenceItem]:
        unique = list({item.evidence_id: item for item in items}.values())
        notes = list(dict.fromkeys(n for n in notes if n))
        if notes:
            self.reasons[sub_claim.sub_claim_id] = " ".join(n if n.endswith(".") else n + "." for n in notes)
        elif not unique:
            self.reasons[sub_claim.sub_claim_id] = NO_FIGURE_REASON
        return unique

    # -------------------------------------------------------------- the reasoning loop
    def _plan(self, sub_claim: SubClaim, cutoff: date, feedback: str | None, prompt: str | None,
              reading: Reading, allowed: set[str] | None = None) -> tuple[list[EvidenceItem], list[str], planner.Plan]:
        """Let the model read the claim and decide which figures it needs; each action runs through the same
        deterministic code as the rules, so the cut-off, the window, the formulas and the scope cannot be
        bypassed."""
        ticker = sub_claim.company_ticker
        hint = "; ".join([f"{p.measure} (period {p.period or 'none named'})" for p in reading.points]
                         + reading.notes)
        context = {"line": sub_claim.text, "prompt": prompt, "feedback": feedback, "ticker": ticker,
                   "company": company_display_name(ticker), "cutoff": cutoff or "latest data",
                   "scope": scope_text(sub_claim, allowed), "parent_span": getattr(sub_claim, "parent_span", None),
                   "period_field": period_field(sub_claim)}
        items, notes = [], []
        segment_only = any(SEGMENT_NOTE in n for n in reading.notes)
        rule_measures = {p.measure for p in reading.points}

        def execute(step: planner.Step) -> str:
            args = step.args
            if step.action == "understand":
                parts = [p for p in args.get("parts") or [] if isinstance(p, dict)]
                outside = [f"'{p.get('words', '')}' ({_LABELS.get(p.get('measure'), p.get('measure') or '?')})"
                           for p in parts if p.get("in_scope") is False]
                if outside:
                    notes.append("Also in the line but outside this sub-claim's scope, so not reported here: "
                                 + "; ".join(outside))
                in_scope = sum(1 for p in parts if p.get("in_scope") is not False)
                return (f"NOTED {len(parts)} part(s), {in_scope} in scope. For comparison, a rule-based reading of "
                        f"the same line (it can be wrong; trust the words): {hint or 'nothing found'}")
            if step.action == "skip":
                notes.append(f"'{args.get('part', '')}' not checked: {args.get('reason', 'no reason given')}")
                return "noted"
            if step.action == "list_periods":
                record_tool_call()
                q = _period_ends(self.store, ticker, cutoff, "quarter")[-WINDOW["quarter"]:]
                y = _period_ends(self.store, ticker, cutoff, "annual")[-WINDOW["annual"]:]
                unclear = unreported_newer_quarter(self.store, ticker, cutoff)
                return (f"Quarters filed by the cut-off (window): {', '.join(q) or 'none'}. Latest fiscal year: "
                        f"{', '.join(y) or 'none'}." + (f" Note: {unclear}" if unclear else ""))
            measure = args.get("measure")
            valid = GROWTH_MEASURES if step.action in ("report_growth", "report_streak") \
                else LEVEL_MEASURES + (LEVERAGE,)
            if measure not in valid:
                return f"ERROR: {measure!r} is not a measure for {step.action}; choose from {', '.join(valid)}"
            if allowed is not None and measure not in allowed:
                return (f"REFUSED: {measure} is outside this sub-claim's scope ({', '.join(sorted(allowed))}); "
                        "another sub-claim checks it. Do not compute it.")
            if segment_only and measure not in rule_measures:
                return ("REFUSED: in this claim the measure is a segment or product figure; the SEC data has only "
                        "company-wide totals. Skip it.")
            try:
                if step.action == "report_streak":
                    span = "annual" if args.get("span") == "annual" else "quarter"
                    run = (int(args.get("count", 0)), span, float(args.get("threshold") or 0.0))
                    point = Point(measure, ("latest", span), "", streak=run)
                else:
                    point = Point(measure, planner.period_spec(args.get("period")), "",
                                  growth_pct=bool(args.get("with_growth")))
            except (KeyError, TypeError, ValueError) as e:
                return f"ERROR: bad arguments ({e})"
            new, why = self._answer(sub_claim, cutoff, point)
            items.extend(new)
            notes.extend(why)
            if not new:
                return f"NO FIGURE: {'; '.join(why) or 'nothing available'}"
            return "OK: " + " | ".join(f"{ev.metric_name}: {ev.content}" for ev in new) \
                + (f" (also: {'; '.join(why)})" if why else "")

        plan = planner.run(context, execute)
        if not plan.finished:
            notes.append(f"The planner stopped after {planner.MAX_STEPS} steps without finishing")
        return items, notes, plan

    # -------------------------------------------------------------- one point of the claim
    def _answer(self, sub_claim: SubClaim, cutoff: date, point: Point) -> tuple[list[EvidenceItem], list[str]]:
        """Evidence for one point, and notes (what was not reported, and why)."""
        label = _LABELS.get(point.measure, point.measure)
        if point.measure == LEVERAGE:
            return [], [f"{label}: {LEVERAGE_REASON}"]
        period = self._resolve(sub_claim, cutoff, point.period)
        if period.reason:
            return [], [f"{label}: {period.reason}"]
        if point.measure in GROWTH_MEASURES:
            return self._growth(sub_claim, cutoff, point, period)
        return self._levels(sub_claim, cutoff, point, period)

    # -------------------------------------------------------------- the period to report
    def _resolve(self, sub_claim: SubClaim, cutoff: date, spec: tuple | None) -> PeriodRef:
        """The period(s) to report: the rule book's period rules, the window, and the
        "latest quarter announced but not filed yet" check."""
        ticker = sub_claim.company_ticker
        ref = resolve_period(self.store, ticker, spec, cutoff)
        if ref.reason:
            return ref
        if ref.latest:
            if ref.span == "quarter":
                why = unreported_newer_quarter(self.store, ticker, cutoff)
                if why:
                    ref.reason = why
                    return ref
            n = spec[1] if spec and spec[0] == "last_n" else 1
            ends = _period_ends(self.store, ticker, cutoff, ref.span)[-WINDOW[ref.span]:]
            if not ends:
                ref.reason = "No filing for this company on or before the cut-off."
            ref.ends = ends[::-1][:n]
            return ref
        if not in_window(self.store, ticker, ref.end, ref.span, cutoff):
            window = "latest fiscal year" if ref.span == "annual" else f"latest {WINDOW['quarter']} quarters"
            ref.ends, ref.reason = [], f"the {ref.label} is outside the window ({window} filed by the cut-off)"
        return ref

    def _ttm_context_end(self, ticker: str, period: PeriodRef, cutoff: date) -> str | None:
        """When no period is named: the latest quarter end, if a quarter was filed after the latest 10-K."""
        if period.named or period.span != "annual" or not period.end:
            return None
        q_ends = _period_ends(self.store, ticker, cutoff, "quarter")
        return q_ends[-1] if q_ends and q_ends[-1] > period.end else None

    def _context_note(self, item: EvidenceItem | None) -> list[str]:
        """DECISION: the trailing-twelve-month figure for a claim with no period is context only. It goes in the
        notes, not the evidence, so it cannot change the Auditor's verdict (team ruling, 18 Oct)."""
        return [f"Context only (not evidence): {item.content}"] if item else []

    # -------------------------------------------------------------- growth
    def _growth(self, sub_claim: SubClaim, cutoff: date, point: Point,
                period: PeriodRef) -> tuple[list[EvidenceItem], list[str]]:
        concept, label, ticker = point.measure, _LABELS[point.measure], sub_claim.company_ticker
        notes = []
        if point.streak:
            n, span, threshold = point.streak
            unit_word = "quarters" if span == "quarter" else "years"
            if n > STREAK_LIMIT[span]:
                notes.append(f"{label} streak not checked: the claim covers {n} {unit_word}; the window allows "
                             f"checking {STREAK_LIMIT[span]} {unit_word}")
                streak_items = []
            else:
                # DECISION: a streak within the window is sent as one growth item per period, so the Auditor can
                # check each against the threshold with the same rules as any growth figure.
                ends = _period_ends(self.store, ticker, cutoff, span)[-n:][::-1]
                streak_items = [i for i in (self._growth_item(sub_claim, cutoff, concept, e, span)[0] for e in ends)
                                if i]
                notes.append(f"{label} streak: {len(streak_items)} of {n} {unit_word} reported, one item each"
                             + (f" (threshold {threshold:.0%})" if threshold else ""))
            if not _has(_PCT, point.words) or minimum_percent(point.words) is not None:   # the streak is the claim
                return streak_items, notes
        else:
            streak_items = []

        main = []
        for end in period.ends:
            item, why = self._growth_item(sub_claim, cutoff, concept, end, period.span, ttm=period.ttm)
            main.append(item) if item else notes.append(f"{label}: {why}")
        context_end = self._ttm_context_end(ticker, period, cutoff)
        if context_end and main:
            notes += self._context_note(self._growth_item(sub_claim, cutoff, concept, context_end, "quarter",
                                                          ttm=True)[0])
        return main + [i for i in streak_items if i.evidence_id not in {m.evidence_id for m in main}], notes

    def _growth_item(self, sub_claim: SubClaim, cutoff: date, concept: str, end: str, span: str,
                     ttm: bool = False) -> tuple[EvidenceItem | None, str]:
        ticker = sub_claim.company_ticker
        record_tool_call()
        if ttm:
            ev = compute_ttm_growth(self.store, ticker, concept, end, cutoff)
        else:
            ev = compute_growth(self.store, ticker, concept, "yoy", span, end, cutoff)
        if ev.status != "ok":
            return None, ev.reason or "growth could not be computed"
        growth = round(ev.value, 6)
        # the two amounts behind the growth rate, so the text states them (the Auditor cross-checks the two)
        prev_end = _near(self.store, ticker, concept, str(ev.period_end), 365, cutoff)   # as the tools do
        cur, prev = self._one(ticker, concept, ev.period_end, cutoff, span, ttm), \
            self._one(ticker, concept, prev_end, cutoff, span, ttm)
        content = (f"{_LABELS[concept]} was {_amount(concept, cur.value, cur.unit)} {_in(_period(cur))} vs "
                   f"{_amount(concept, prev.value, prev.unit)} {_in(_period(prev))} ({growth:+.1%} YoY).")
        return self._item(sub_claim, [ev], ev.period_end, f"{_NAMES[concept]}_growth_yoy", growth, content,
                          _period(ev), comparison=(round(cur.value, 2), _period(cur))), ""

    # -------------------------------------------------------------- margins, cash, liquidity
    def _one(self, ticker: str, item: str, period_end, cutoff: date, span: str = "quarter",
             ttm: bool = False) -> FinancialEvidence:
        record_tool_call()
        if ttm:
            return compute_ttm(self.store, ticker, item, period_end, cutoff)
        if item in CONCEPTS:
            return get_financial_fact(self.store, ticker, item, span, period_end, cutoff)
        return compute_metric(self.store, ticker, item, span, period_end, cutoff)

    def _levels(self, sub_claim: SubClaim, cutoff: date, point: Point,
                period: PeriodRef) -> tuple[list[EvidenceItem], list[str]]:
        metric, items, notes = point.measure, [], []
        ttm = period.ttm and metric in _FLOW_METRICS      # a balance-sheet ratio has no 12-month figure
        with_growth = point.growth_pct and metric == "free_cash_flow"
        for end in period.ends:
            new, why = self._level_items(sub_claim, cutoff, metric, end, period.span, ttm, with_growth)
            items += new
            notes += why
        context_end = self._ttm_context_end(sub_claim.company_ticker, period, cutoff)
        if context_end and metric in _FLOW_METRICS and items:
            ctx, _ = self._level_items(sub_claim, cutoff, metric, context_end, "quarter", True)
            notes += self._context_note(ctx[0] if ctx else None)
        return items, notes

    def _level_items(self, sub_claim: SubClaim, cutoff: date, metric: str, end: str, span: str, ttm: bool,
                     with_growth: bool = False) -> tuple[list[EvidenceItem], list[str]]:
        ticker = sub_claim.company_ticker
        cur = self._one(ticker, metric, end, cutoff, span, ttm)
        if cur.status != "ok":
            return [], [f"{_LABELS[metric]}: {cur.reason}"]
        prev_end = prior_year_end(self.store, ticker, cur.period_end, cutoff)
        prev = self._one(ticker, metric, prev_end, cutoff, span, ttm) if prev_end else None
        when = _period(cur)
        content = f"{_LABELS[metric]} was {_amount(metric, cur.value, cur.unit)} {_in(when)}"
        evidence, comparison, notes = [cur], None, []
        if prev is not None and prev.status == "ok":
            evidence.append(prev)
            comparison = (round(prev.value, 6), _period(prev))
            content += f" vs {_amount(metric, prev.value, prev.unit)} {_in(_period(prev))}"
        else:
            notes.append(f"{_LABELS[metric]}: no figure for the same period a year earlier")
        items = [self._item(sub_claim, evidence, cur.period_end, _NAMES[metric], round(cur.value, 6),
                            content + ".", when, comparison=comparison)]
        if with_growth and comparison:
            if prev.value > 0:
                g = round(cur.value / prev.value - 1, 6)
                items.append(self._item(
                    sub_claim, evidence, cur.period_end, f"{_NAMES[metric]}_growth_yoy", g,
                    f"{_LABELS[metric]} was {_amount(metric, cur.value, cur.unit)} {_in(when)} vs "
                    f"{_amount(metric, prev.value, prev.unit)} {_in(_period(prev))} ({g:+.1%} YoY).", when,
                    comparison=(round(cur.value, 2), when)))
            else:
                notes.append(f"{_LABELS[metric]}: a growth rate is not meaningful because the figure a year earlier "
                             "was zero or negative (the direction is in the level item)")
        return items, notes

    # -------------------------------------------------------------- EvidenceItem
    def _item(self, sub_claim: SubClaim, evidence: list[FinancialEvidence], period_end, metric_name: str,
              value: float, content: str, period: str,
              comparison: tuple[float, str] | None = None) -> EvidenceItem:
        facts = [f for ev in evidence for f in ev.inputs]
        cited = max(facts, key=lambda f: (f.period_end, f.filed))     # the filing for the latest period
        tags = sorted({f"us-gaap:{f.xbrl_tag}" for f in facts})
        filings = sorted({f"{_doc_type(f.form)} {f.accession}" for f in facts})
        notes = sorted({w for ev in evidence for w in ev.warnings})
        if any(ev.derived for ev in evidence):
            notes.append("includes a figure derived from reported values (year-to-date subtraction or a fallback)")
        ticker = sub_claim.company_ticker or ""
        extra = {}
        if comparison is not None and {"comparison_value", "comparison_period"} <= set(EvidenceItem.model_fields):
            extra = {"comparison_value": comparison[0], "comparison_period": comparison[1]}
        return EvidenceItem(
            evidence_id=f"XBRL-{ticker}-{metric_name}-{period_end}",
            sub_claim_id=sub_claim.sub_claim_id, stance=Stance.NEUTRAL,   # reports figures; the Auditor judges
            content=content + (" Note: " + "; ".join(notes) + "." if notes else ""),
            source_tier=SourceTier.TIER_1, doc_type=_doc_type(cited.form),
            company=_company(ticker), period=period,
            publication_date=cited.filed, url=cited.source_url,
            citation_span="XBRL " + " / ".join(tags) + "; filings: " + ", ".join(filings),
            produced_by=self.name, computed_value=value, metric_name=metric_name, **extra)
