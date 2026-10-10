""" 
Deterministic financial tools 

When data is missing, they return status="insufficient_data" or "out_of_scope" (ticker not in corpus) 
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from typing import Optional

from .concepts import CONCEPTS, additive, kind, unit
from .schema import FinancialEvidence, FinancialFact, SeriesPoint, ValueCheck
from .store import SPAN_ORDER, FactStore, _iso

_CLS = {"quarter": "Q", "annual": "FY"}
# Concepts used to find a company's period ends. Revenue first; the others cover a company whose revenue tag
# is not mapped, so one tagging gap does not make every tool fail.
_ANCHORS = ("Revenue", "NetIncome", "OperatingCashFlow")
# The latest period is "stale" when it ended more than this many days before the cut-off: a normal filing
# lag is 1-3 months, so a longer gap means filings are missing from the data or the company stopped filing.
_STALE_DAYS = {"quarter": 200, "annual": 460}
_LENGTH_TOLERANCE_DAYS = 6      # a 53-week fiscal year adds 7 days to one quarter and to the year


def _eid(tool, **args):
    return hashlib.sha1(json.dumps([tool, args], sort_keys=True, default=str).encode()).hexdigest()[:12]


def _n(concept, v):
    return f"{v:,.2f}" if unit(concept) == "USD/shares" else f"{v:,.0f}"


def _ff(r):
    return FinancialFact(**r)


def _raw(store, t, concept, end, span, cutoff):
    """Reported value, or a quarter derived from two year-to-date facts.
    -> (value, [facts], formula, derived, warnings) or None."""
    if kind(concept) == "instant":
        f = store.fact(t, concept, "I", cutoff, end=end)
        return (f["value"], [f], f"{concept}={_n(concept, f['value'])}", False, []) if f else None
    f = store.fact(t, concept, _CLS[span], cutoff, end=end)
    if f:
        return f["value"], [f], f"{concept}={_n(concept, f['value'])}", False, []
    if span == "quarter" and additive(concept):
        # not reported as a standalone quarter: difference two year-to-date facts
        for i in (1, 2, 3):
            long = store.fact(t, concept, SPAN_ORDER[i], cutoff, end=end)
            if not long:
                continue
            # Take the shorter figure as it stood when the longer one was filed, so both are on the same
            # basis. Only if that does not exist, fall back to a later (possibly restated) version.
            warns = []
            short = store.fact(t, concept, SPAN_ORDER[i - 1], long["filed"], start=long["period_start"])
            if not short:
                short = store.fact(t, concept, SPAN_ORDER[i - 1], cutoff, start=long["period_start"])
                if short:
                    warns.append(f"{concept}: derived from figures filed at different times "
                                 f"({long['filed']} and {short['filed']}); a restatement may make them inconsistent")
            if short:
                v = long["value"] - short["value"]
                if concept == "Revenue" and v <= 0:
                    warns.append("Revenue: derived quarter is zero or negative; the inputs are probably inconsistent")
                formula = (f"{concept}={long['value']:,.0f} ({SPAN_ORDER[i]}) - "
                           f"{short['value']:,.0f} ({SPAN_ORDER[i - 1]}) = {v:,.0f}")
                return v, [long, short], formula, True, warns
    return None


def _val(store, t, concept, end, span, cutoff):
    """_raw plus documented fallbacks. -> (value, facts, formula, derived, warnings) or None."""
    r = _raw(store, t, concept, end, span, cutoff)
    if r:
        return r
    if concept == "GrossProfit":
        a = _raw(store, t, "Revenue", end, span, cutoff)
        b = _raw(store, t, "CostOfRevenue", end, span, cutoff)
        if a and b:
            v = a[0] - b[0]
            return (v, a[1] + b[1], f"GrossProfit = Revenue - CostOfRevenue = {v:,.0f}", True,
                    a[4] + b[4] + ["GrossProfit not tagged; computed as Revenue - CostOfRevenue"])
    if concept == "TotalLiabilities":
        a = _raw(store, t, "LiabilitiesAndEquity", end, span, cutoff)
        b = _raw(store, t, "ShareholdersEquity", end, span, cutoff)
        if a and b:
            v = a[0] - b[0]
            return (v, a[1] + b[1], f"TotalLiabilities = LiabilitiesAndEquity - Equity = {v:,.0f}", True,
                    ["Liabilities not tagged; computed as L&E - equity (may include non-controlling "
                     "or temporary equity) - verify manually"])
    return None


def _period_ends(store, t, cutoff, span):
    """Period end dates known for a company at the cut-off, from the anchor concepts and the balance sheet."""
    ends = set()
    for concept in _ANCHORS:
        ends.update(store.ends(t, concept, cutoff, ["FY"] if span == "annual" else SPAN_ORDER))
    if span == "quarter":
        ends.update(store.ends(t, "TotalAssets", cutoff, ["I"]))
    return sorted(ends)


def _days(part):
    """Length in days of the period a value covers, or None if it cannot be read off the facts."""
    facts, derived = part[1], part[3]
    if derived and len(facts) == 2:                     # derived quarter: from the short figure's end to the long one's
        return (date.fromisoformat(facts[0]["period_end"]) - date.fromisoformat(facts[1]["period_end"])).days
    if not derived and len(facts) == 1 and facts[0]["period_start"]:
        return (date.fromisoformat(facts[0]["period_end"]) - date.fromisoformat(facts[0]["period_start"])).days + 1
    return None


def _resolve_end(store, t, period_end, span, cutoff):
    if period_end not in (None, "latest"):
        return period_end.isoformat() if isinstance(period_end, date) else str(period_end)
    ends = _period_ends(store, t, cutoff, span)
    return ends[-1] if ends else None


def _near(store, t, concept, end, days, cutoff):
    """Comparable period end ~`days` earlier (+-10d, handles 52/53-week fiscal years). None if there is no
    period within 10 days of the target, e.g. after a change of fiscal year end: better no comparison than
    a wrong one."""
    target = date.fromisoformat(end) - timedelta(days=days)
    cands = store.ends(t, concept, cutoff) + _period_ends(store, t, cutoff, "quarter")
    best = min(sorted(set(cands)), key=lambda e: abs((date.fromisoformat(e) - target).days), default=None)
    return best if best and abs((date.fromisoformat(best) - target).days) <= 10 else None


def prior_year_end(store, ticker, period_end, cutoff_date=None):
    """The same period one year earlier, as an ISO date string, or None if not in the data."""
    return _near(store, ticker, _ANCHORS[0], str(period_end), 365, cutoff_date)


def _ev(tool, t, metric, span, end, cutoff, parts=None, value=None, unit="USD", formula="", reason=None,
        status=None, warnings=(), **idargs):
    facts, warns, derived = [], list(warnings), False
    for p in parts or []:
        facts += p[1]
        derived |= p[3]
        warns += p[4]
    uniq = list({(f["accession"], f["xbrl_tag"], f["period_start"], f["period_end"]): f for f in facts}.values())
    # Inputs for one and the same period normally come from one filing. If they do not, part of that period
    # was restated later and part was not.
    by_period = {}
    for f in uniq:
        by_period.setdefault((f["period_start"], f["period_end"]), set()).add(f["accession"])
    mixed = sorted(end_ for (_, end_), accns in by_period.items() if len(accns) > 1)
    if mixed:
        warns.append(f"inputs for the period ending {mixed[-1]} come from different filings "
                     "(some figures were restated later, others not)")
    return FinancialEvidence(
        evidence_id=_eid(tool, ticker=t, metric=metric, span=span, end=end, cutoff=cutoff, **idargs),
        tool=tool, ticker=t, metric=metric, span=span, period_end=end, cutoff_date=cutoff,
        status=status or ("ok" if value is not None else "insufficient_data"), value=value, unit=unit,
        as_of_date=max((f["filed"] for f in uniq), default=None), formula=formula,
        inputs=[_ff(f) for f in uniq], derived=derived, reason=reason, warnings=sorted(set(warns)))


def _guard(store, tool, t, metric, span, period_end, cutoff):
    """Shared entry checks. -> (end, None, notes) or (None, FinancialEvidence explaining why not, [])."""
    if span not in _CLS:
        return None, _ev(tool, t, metric, span, None, cutoff, reason="span must be 'quarter' or 'annual'"), []
    if t not in store.tickers:
        return None, _ev(tool, t, metric, span, None, cutoff, status="out_of_scope",
                         reason=f"{t} is not in the corpus"), []
    end = _resolve_end(store, t, period_end, span, cutoff)
    if end is None:
        return None, _ev(tool, t, metric, span, None, cutoff,
                         reason="no filings on or before the cut-off date"), []
    try:
        end_date = date.fromisoformat(end)
    except ValueError:
        return None, _ev(tool, t, metric, span, None, cutoff,
                         reason=f"period_end {end!r} is not a date in YYYY-MM-DD form"), []
    notes = []
    if cutoff is not None and period_end in (None, "latest"):
        gap = (date.fromisoformat(_iso(cutoff)) - end_date).days
        if gap > _STALE_DAYS[span]:
            notes.append(f"latest {span} in the data ended {end}, {gap} days before the cut-off: "
                         "newer filings may be missing")
    return end, None, notes


# ------------------------------------------------------------------ tools
def get_financial_fact(store: FactStore, ticker: str, concept: str, span: str = "quarter",
                       period_end="latest", cutoff_date: Optional[date] = None) -> FinancialEvidence:
    """A reported (or year-to-date-differenced) value for one concept and period."""
    tool = "get_financial_fact"
    end, bad, notes = _guard(store, tool, ticker, concept, span, period_end, cutoff_date)
    if bad:
        return bad
    if concept not in CONCEPTS:
        return _ev(tool, ticker, concept, span, end, cutoff_date,
                   reason=f"unknown concept; choose from {sorted(CONCEPTS)}")
    p = _val(store, ticker, concept, end, span, cutoff_date)
    if not p:
        why = f"{concept} not available for {span} ending {end}"
        if span == "quarter" and not additive(concept):
            why += " (not reported standalone; share and per-share figures cannot be derived from year-to-date values)"
        return _ev(tool, ticker, concept, span, end, cutoff_date, reason=why)
    return _ev(tool, ticker, concept, span, end, cutoff_date, [p], p[0], unit(concept), p[2], warnings=notes)


def compute_growth(store: FactStore, ticker: str, concept: str = "Revenue", basis: str = "yoy",
                   span: str = "quarter", period_end="latest", cutoff_date: Optional[date] = None) -> FinancialEvidence:
    """Growth of a concept vs the same period a year earlier (yoy) or the prior quarter (qoq)."""
    metric = f"{concept}_growth_{basis}"
    end, bad, notes = _guard(store, "compute_growth", ticker, metric, span, period_end, cutoff_date)
    if bad:
        return bad

    def mk(**k):
        return _ev("compute_growth", ticker, metric, span, end, cutoff_date, **k)

    if concept not in CONCEPTS:
        return mk(reason=f"unknown concept; choose from {sorted(CONCEPTS)}")
    if basis not in ("yoy", "qoq"):
        return mk(reason="basis must be 'yoy' or 'qoq'")
    if basis == "qoq" and span != "quarter":
        return mk(reason="qoq requires span='quarter'")
    cur = _val(store, ticker, concept, end, span, cutoff_date)
    prev_end = _near(store, ticker, concept, end, 365 if basis == "yoy" else 91, cutoff_date)
    prev = prev_end and _val(store, ticker, concept, prev_end, span, cutoff_date)
    if not cur or not prev:
        missing = "current" if not cur else "comparison"
        return mk(parts=[x for x in (cur, prev) if x], reason=f"missing {missing} period for {concept}")
    if prev[0] <= 0:
        return mk(parts=[cur, prev], reason="comparison value is zero or negative; growth rate not meaningful")
    g = cur[0] / prev[0] - 1
    # Things that make the two periods not strictly comparable. The number is still returned, with a warning.
    d_cur, d_prev = _days(cur), _days(prev)
    if d_cur and d_prev and abs(d_cur - d_prev) > _LENGTH_TOLERANCE_DAYS:
        notes.append(f"the two periods differ in length ({d_cur} vs {d_prev} days), e.g. a 53-week fiscal year")
    if cur[1][0]["xbrl_tag"] != prev[1][0]["xbrl_tag"]:
        notes.append(f"the two periods use different tags ({cur[1][0]['xbrl_tag']} vs {prev[1][0]['xbrl_tag']}); "
                     "check they mean the same thing")
    if unit(concept) == "shares" and not 0.55 < cur[0] / prev[0] < 1.8:
        notes.append("share count changed by more than a large issue or buyback would explain: "
                     "possibly a stock split not yet restated in the comparison period")
    if unit(concept) == "USD/shares":
        notes.append("per-share figures are not adjusted for stock splits; check for a split between the two periods")
    return mk(parts=[cur, prev], value=g, unit="ratio", warnings=notes,
              formula=f"{_n(concept, cur[0])} ({end}) / {_n(concept, prev[0])} ({prev_end}) - 1 = {g:.4f}")


_METRICS = {  # metric: (numerator concepts with sign, denominator concept or None)
    "gross_margin": ([("GrossProfit", 1)], "Revenue"),
    "operating_margin": ([("OperatingIncome", 1)], "Revenue"),
    "net_margin": ([("NetIncome", 1)], "Revenue"),
    "free_cash_flow": ([("OperatingCashFlow", 1), ("CapitalExpenditure", -1)], None),
    "fcf_margin": ([("OperatingCashFlow", 1), ("CapitalExpenditure", -1)], "Revenue"),
    "liabilities_to_equity": ([("TotalLiabilities", 1)], "ShareholdersEquity"),
    "current_ratio": ([("CurrentAssets", 1)], "CurrentLiabilities"),
    "cash_ratio": ([("CashAndEquivalents", 1)], "CurrentLiabilities"),
}
METRICS = tuple(_METRICS)


def compute_metric(store: FactStore, ticker: str, metric: str, span: str = "quarter",
                   period_end="latest", cutoff_date: Optional[date] = None) -> FinancialEvidence:
    """Margin, cash-flow, liquidity and leverage metrics. Balance-sheet inputs are taken at period_end."""
    end, bad, notes = _guard(store, "compute_metric", ticker, metric, span, period_end, cutoff_date)
    if bad:
        return bad

    def mk(**k):
        return _ev("compute_metric", ticker, metric, span, end, cutoff_date, warnings=notes, **k)

    if metric not in _METRICS:
        return mk(reason=f"unknown metric; choose from {sorted(_METRICS)}")
    num, den = _METRICS[metric]
    names = [c for c, _ in num] + ([den] if den else [])
    got = {c: _val(store, ticker, c, end, span, cutoff_date) for c in names}
    missing = [c for c in names if not got[c]]
    parts = [got[c] for c in names if got[c]]
    if missing:
        return mk(parts=parts, reason=f"missing inputs: {missing}")
    n = sum(got[c][0] * s for c, s in num)
    expr = " ".join(f"{'-' if s < 0 else '+'} {c} {got[c][0]:,.0f}" for c, s in num).lstrip("+ ")
    if den is None:
        return mk(parts=parts, value=n, unit="USD", formula=f"{expr} = {n:,.0f}")
    d = got[den][0]
    if d <= 0:
        return mk(parts=parts, reason=f"{den} is zero or negative; ratio not meaningful")
    return mk(parts=parts, value=n / d, unit="ratio", formula=f"({expr}) / {den} {d:,.0f} = {n / d:.4f}")


def compute_trend(store: FactStore, ticker: str, item: str, periods: int = 6, span: str = "quarter",
                  basis: str = "level", period_end="latest", cutoff_date: Optional[date] = None) -> FinancialEvidence:
    """A concept or metric over the last N consecutive periods ending at period_end, oldest first.
    basis='level' gives the value each period; basis='yoy' gives each period's year-on-year growth
    (concepts only; use it for seasonal businesses). `value` is last minus first; `direction` is
    'up'/'down' only if every step moves that way, otherwise 'flat' or 'mixed'."""
    metric = f"{item}_trend" + ("_yoy" if basis == "yoy" else "")
    end, bad, notes = _guard(store, "compute_trend", ticker, metric, span, period_end, cutoff_date)
    if bad:
        return bad

    def mk(**k):
        return _ev("compute_trend", ticker, metric, span, end, cutoff_date, periods=periods, **k)

    if item not in CONCEPTS and item not in _METRICS:
        return mk(reason=f"unknown item; choose a concept {sorted(CONCEPTS)} or metric {sorted(_METRICS)}")
    if basis not in ("level", "yoy") or (basis == "yoy" and item not in CONCEPTS):
        return mk(reason="basis must be 'level', or 'yoy' with a concept")
    periods = max(2, min(int(periods), 12))

    def one(e):
        if basis == "yoy":
            return compute_growth(store, ticker, item, "yoy", span, e, cutoff_date)
        if item in CONCEPTS:
            return get_financial_fact(store, ticker, item, span, e, cutoff_date)
        return compute_metric(store, ticker, item, span, e, cutoff_date)

    pts, e, stop = [], end, None
    while e and len(pts) < periods:
        ev = one(e)
        if ev.status != "ok":
            stop = f"{e}: {ev.reason}"
            break
        pts.append(ev)
        e = _near(store, ticker, _ANCHORS[0], e, 91 if span == "quarter" else 365, cutoff_date)
    pts.reverse()
    if len(pts) < 2:
        return mk(reason="fewer than 2 consecutive periods available" + (f" (stopped at {stop})" if stop else ""))
    vals = [p.value for p in pts]
    ups = sum(b > a for a, b in zip(vals, vals[1:]))
    downs = sum(b < a for a, b in zip(vals, vals[1:]))
    steps = len(vals) - 1
    direction = "flat" if ups == downs == 0 else "up" if downs == 0 else "down" if ups == 0 else "mixed"
    u = pts[-1].unit

    def show(v):
        return f"{v:.4f}" if u == "ratio" else f"{v:,.2f}" if u == "USD/shares" else f"{v:,.0f}"

    warns = {w for p in pts for w in p.warnings} | set(notes)
    if len(pts) < periods:
        warns.add(f"only {len(pts)} of {periods} requested periods available"
                  + (f" (stopped at {stop})" if stop else ""))
    if span == "quarter" and basis == "level":
        warns.add("consecutive quarters are not seasonally adjusted; use basis='yoy' for seasonal businesses")
    inputs = list({(f.accession, f.xbrl_tag, f.period_start, f.period_end): f for p in pts for f in p.inputs}.values())
    out = mk(value=vals[-1] - vals[0], unit=u,
             formula=" -> ".join(f"{show(p.value)} ({p.period_end})" for p in pts)
                     + f"; {ups} up, {downs} down of {steps} steps; last - first = {show(vals[-1] - vals[0])}")
    series = [SeriesPoint(period_end=p.period_end, value=p.value, derived=p.derived, evidence_id=p.evidence_id)
              for p in pts]
    return out.model_copy(update=dict(
        inputs=inputs, as_of_date=max(f.filed for f in inputs), derived=any(p.derived for p in pts),
        warnings=sorted(warns), direction=direction, series=series))


def check_stated_value(evidence: FinancialEvidence, stated: float, tolerance: float = 0.01) -> ValueCheck:
    """Execution-comparison helper. For unit='ratio' the tolerance is absolute (0.01 = 1 percentage
    point; pass stated 14% as 0.14). For other units it is relative (0.01 = 1%)."""
    c = evidence.value
    if c is None:
        return ValueCheck(stated=stated, computed=None, tolerance=tolerance, match=False,
                          evidence_id=evidence.evidence_id)
    ad = abs(stated - c)
    rd = ad / abs(c) if c else None
    ok = ad <= tolerance if evidence.unit == "ratio" else (rd is not None and rd <= tolerance)
    return ValueCheck(stated=stated, computed=c, abs_diff=ad, rel_diff=rd, tolerance=tolerance, match=ok,
                      evidence_id=evidence.evidence_id)


def list_periods(store: FactStore, ticker: str, cutoff_date: Optional[date] = None) -> dict:
    """What is in scope for a ticker at a cut-off: period ends and the filings they come from."""
    return {"ticker": ticker, "in_corpus": ticker in store.tickers, "cutoff_date": cutoff_date,
            "quarter_ends": _period_ends(store, ticker, cutoff_date, "quarter"),
            "annual_ends": _period_ends(store, ticker, cutoff_date, "annual"),
            "filings_in_scope": store.filings_in_scope(ticker, cutoff_date)}


# ------------------------------------------------------------------ LLM function-calling specs
def _spec(name, desc, props, required):
    common = {"ticker": {"type": "string", "description": "Company ticker in the corpus"},
              "span": {"type": "string", "enum": ["quarter", "annual"]},
              "period_end": {"type": "string", "description": "'latest' or fiscal period end date YYYY-MM-DD"},
              "cutoff_date": {"type": ["string", "null"],
                              "description": "YYYY-MM-DD; only filings filed on or before this date are used"}}
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": {**common, **props}, "required": ["ticker", *required]}}


_CONCEPT_ENUM = {"type": "string", "enum": sorted(CONCEPTS)}
TOOL_SPECS = [
    _spec("get_financial_fact", get_financial_fact.__doc__, {"concept": _CONCEPT_ENUM}, ["concept"]),
    _spec("compute_growth", compute_growth.__doc__,
          {"concept": _CONCEPT_ENUM, "basis": {"type": "string", "enum": ["yoy", "qoq"]}}, ["concept"]),
    _spec("compute_metric", compute_metric.__doc__, {"metric": {"type": "string", "enum": sorted(_METRICS)}},
          ["metric"]),
    _spec("compute_trend", compute_trend.__doc__,
          {"item": {"type": "string", "enum": sorted(CONCEPTS) + sorted(_METRICS)},
           "periods": {"type": "integer", "minimum": 2, "maximum": 12},
           "basis": {"type": "string", "enum": ["level", "yoy"]}}, ["item"]),
]
TOOLS = {"get_financial_fact": get_financial_fact, "compute_growth": compute_growth,
         "compute_metric": compute_metric, "compute_trend": compute_trend}


def call_tool(store: FactStore, name: str, args: dict) -> FinancialEvidence:
    """Single dispatch point for the agent."""
    a = dict(args)
    if a.get("cutoff_date"):
        a["cutoff_date"] = date.fromisoformat(str(a["cutoff_date"]))
    return TOOLS[name](store, **a)
