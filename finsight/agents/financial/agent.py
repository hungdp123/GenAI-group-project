""" Financial Analysis Agent 

Agent picks a metric from the sub-claim's type and wording by fixed rules,
computes it with the tools in `tools.py`, compares it with the same period a year earlier, and
labels the result SUPPORTING / CHALLENGING / NEUTRAL against the direction the sub-claim asserts.

Data: the pulled SEC facts in data/sec/processed (see pull.py)
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from functools import lru_cache
from importlib import resources
from pathlib import Path

from finsight.config import CORPUS_COMPANIES
from finsight.contracts.schemas import EvidenceItem, SourceTier, Stance, SubClaim, SubClaimType
from finsight.llm.usage import record_tool_call

from .concepts import CONCEPTS, TAG_TO_CONCEPT, kind, unit
from .schema import FinancialEvidence
from .store import DEFAULT_DATA_DIR, FactStore, _read_csv
from .tools import METRICS, _near, compute_growth, compute_metric, compute_trend, get_financial_fact, prior_year_end

# DECISION: "strong" growth means at least 10% year on year (same threshold as the placeholder stub).
STRONG_GROWTH_THRESHOLD = 0.10
MAX_ITEMS = 3   # at most this many metrics per sub-claim

# ------------------------------------------------------------------ wording rules
_UP = r"\b(expand|rising|rise[sn]?|rose|higher|grow|grew|increas|accelerat|widen|surg|soar|climb)"
_DOWN = r"\b(shrink|shrank|shrunk|declin|fall|fell|lower|decreas|contract|compress|narrow|drop|slow|slid)"
_GOOD = r"\b(improv|strong|health|better|robust|solid|recover|turnaround|working)"
_BAD = r"\b(weak|worse|worsen|deteriorat|struggl|troubl|unsustainab)"
_STRONG = r"\b(strong|rapid|surg|soar|boom|explosive|massive|huge|impressive)"
_TREND = r"\b(turnaround|accelerat|consistent|trend|recover|momentum|sustained|quarter after quarter)"
_ANNUAL = r"\b(annual|full[- ]year|fiscal year|yearly)"
_GROWTH_TOPIC = r"\b(grow|grew|revenue|sales|top line)"

# Growth sub-claims: which concept is growing. First match wins; default Revenue.
_GROWTH_CONCEPTS = [
    (r"\bearnings per share\b|\beps\b", "DilutedEPS"),
    (r"\boperating (income|profit)", "OperatingIncome"),
    (r"\bcash flow", "OperatingCashFlow"),
    (r"\bnet income\b|\bnet profit|\bearnings\b|\bprofits?\b|\bbottom line\b", "NetIncome"),
]

# Profitability / cash / leverage sub-claims: (pattern, item, higher_is_better). Checked in order; matched
# text is removed before later rules run, so "free cash flow" does not also match "cash".
# higher_is_better=None: neither direction is good or bad by itself (only explicit up/down wording counts).
_METRIC_RULES = [
    (r"\bgross margins?", "gross_margin", True),
    (r"\boperating margins?", "operating_margin", True),
    (r"\b(net|profit) margins?|\bprofitab", "net_margin", True),
    (r"\bfree cash flow|\bfcf\b", "free_cash_flow", True),
    (r"\bcash flow|\bcash generation", "OperatingCashFlow", True),
    (r"\bmargins?", "gross_margin", True),
    (r"\bliquidity|\bcurrent ratio", "current_ratio", True),
    # DECISION: debt is not mapped (tags are inconsistent), so leverage uses total liabilities / equity.
    (r"\bleverage|\bliabilit|\bindebted|\bbalance sheet|\bdebt", "liabilities_to_equity", False),
    (r"\bdilut|\bshare count|\bshares outstanding", "DilutedShares", False),
    (r"\binventor", "Inventory", None),
    (r"\bcash\b", "CashAndEquivalents", True),
    (r"\bnet (income|loss)|\bprofits?\b|\bearnings\b|\bloss(es)?\b", "NetIncome", True),
]
_DEFAULT_METRICS = ["gross_margin", "operating_margin", "net_margin"]   # first one with data is used

_NAMES = {
    "Revenue": "revenue", "CostOfRevenue": "cost_of_revenue", "GrossProfit": "gross_profit",
    "OperatingIncome": "operating_income", "NetIncome": "net_income", "OperatingCashFlow": "operating_cash_flow",
    "CapitalExpenditure": "capital_expenditure", "DilutedShares": "diluted_shares", "DilutedEPS": "diluted_eps",
    "TotalAssets": "total_assets", "TotalLiabilities": "total_liabilities",
    "LiabilitiesAndEquity": "liabilities_and_equity", "ShareholdersEquity": "shareholders_equity",
    "CashAndEquivalents": "cash_and_equivalents", "CurrentAssets": "current_assets",
    "CurrentLiabilities": "current_liabilities", "Inventory": "inventory",
}
_LABELS = {
    "gross_margin": "Gross margin", "operating_margin": "Operating margin", "net_margin": "Net margin",
    "free_cash_flow": "Free cash flow", "fcf_margin": "Free cash flow margin",
    "liabilities_to_equity": "Total liabilities to equity", "current_ratio": "Current ratio",
    "cash_ratio": "Cash ratio", "Revenue": "Revenue", "OperatingIncome": "Operating income",
    "NetIncome": "Net income", "OperatingCashFlow": "Operating cash flow", "DilutedShares": "Diluted shares",
    "DilutedEPS": "Diluted EPS", "Inventory": "Inventory", "CashAndEquivalents": "Cash and equivalents",
}
_MARGINS = {"gross_margin", "operating_margin", "net_margin", "fcf_margin"}


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def claim_direction(text: str) -> str | None:
    """'up' / 'down' if the wording asserts a movement, 'good' / 'bad' if it asserts quality, else None.
    Conflicting wording (both up and down, or both good and bad) gives None."""
    up, down = _has(_UP, text), _has(_DOWN, text)
    if up != down:
        return "up" if up else "down"
    if up and down:
        return None
    good, bad = _has(_GOOD, text), _has(_BAD, text)
    if good != bad:
        return "good" if good else "bad"
    return None


def expected_move(direction: str | None, higher_is_better: bool | None) -> int:
    """+1 if the sub-claim implies the metric went up, -1 if down, 0 if it implies neither."""
    if direction == "up":
        return 1
    if direction == "down":
        return -1
    if direction is None or higher_is_better is None:
        return 0
    return 1 if (direction == "good") == higher_is_better else -1


def _stance(moved: float, expected: int) -> Stance:
    if expected == 0 or moved == 0:
        return Stance.NEUTRAL
    return Stance.SUPPORTING if (moved > 0) == (expected > 0) else Stance.CHALLENGING


def pick_metrics(text: str) -> list[tuple[str, bool | None]]:
    """Metrics named by the sub-claim wording, in rule order, without duplicates."""
    remaining, picked = text, []
    for pattern, item, hib in _METRIC_RULES:
        if _has(pattern, remaining):
            remaining = re.sub(pattern, " ", remaining, flags=re.IGNORECASE)
            if item not in [p[0] for p in picked]:
                picked.append((item, hib))
    return picked[:MAX_ITEMS]


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


def _sec_dir() -> Path:
    """Where the pulled SEC files live. The project copy is data/financial."""
    env = os.getenv("FINSIGHT_SEC_DATA")
    if env:
        return Path(env)
    processed = DEFAULT_DATA_DIR / "processed"
    if (processed / "facts.csv").exists():
        return processed
    financial = Path("data") / "financial"
    if (financial / "facts.csv").exists():
        return financial
    return processed


def default_store() -> FactStore:
    """Pulled SEC data if present, plus the demo fixture. Reloaded when facts.csv changes."""
    processed = _sec_dir()
    facts_csv = processed / "facts.csv"
    return _load_store(str(processed), facts_csv.stat().st_mtime if facts_csv.exists() else 0.0)


# ------------------------------------------------------------------ formatting
def _fmt(item: str, value: float, unit_: str) -> str:
    if item in _MARGINS or item.endswith("_growth_yoy"):
        return f"{value:.1%}"
    if unit_ == "ratio":
        return f"{value:.2f}x"
    if unit_ == "shares":
        return f"{value / 1e6:,.1f}m shares"
    if unit_ == "USD/shares":
        return f"${value:,.2f}"
    sign, v = ("-" if value < 0 else ""), abs(value)
    if v >= 1e9:
        return f"{sign}${v / 1e9:,.2f}bn"
    return f"{sign}${v / 1e6:,.1f}m" if v >= 1e6 else f"{sign}${v:,.0f}"


def _period(ev: FinancialEvidence) -> str:
    is_balance = ev.metric in CONCEPTS and kind(ev.metric) == "instant"
    word = "as at" if is_balance else "year ended" if ev.span == "annual" else "quarter ended"
    return f"{word} {ev.period_end}"


class XbrlFinancialAgent:
    name = "financial_xbrl"

    def __init__(self, store: FactStore | None = None) -> None:
        self.store = store if store is not None else default_store()

    # -------------------------------------------------------------- entry point
    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]:
        ticker = sub_claim.company_ticker
        if not ticker or ticker not in self.store.tickers:
            return []
        kind_ = sub_claim.sub_claim_type
        if kind_ is SubClaimType.HISTORICAL_GROWTH:
            return self._growth(sub_claim, cutoff_date, retry=feedback is not None)
        if kind_ is SubClaimType.PROFITABILITY_CASH_LEVERAGE:
            return self._levels(sub_claim, cutoff_date, pick_metrics(sub_claim.text), use_default=True)
        if kind_ is SubClaimType.OTHER:
            # DECISION: unclassified sub-claims get numbers only when the wording names something measurable.
            picked = pick_metrics(sub_claim.text)
            if picked:
                return self._levels(sub_claim, cutoff_date, picked, use_default=False)
            if _has(_GROWTH_TOPIC, sub_claim.text):
                return self._growth(sub_claim, cutoff_date, retry=feedback is not None)
        # DECISION: other types (demand attribution, risk, continuation, comparative) get no financial
        # evidence, even on a retry: company-wide totals cannot show what drove growth or what will happen.
        return []

    # -------------------------------------------------------------- growth
    def _growth(self, sub_claim: SubClaim, cutoff: date, retry: bool) -> list[EvidenceItem]:
        text, ticker = sub_claim.text, sub_claim.company_ticker
        concept = next((c for pattern, c in _GROWTH_CONCEPTS if _has(pattern, text)), "Revenue")
        spans = ["annual"] if _has(_ANNUAL, text) else ["quarter"] + (["annual"] if retry else [])
        ev = None
        for span in spans:
            record_tool_call()
            ev = compute_growth(self.store, ticker, concept, "yoy", span, "latest", cutoff)
            if ev.status == "ok":
                break
        if ev is None or ev.status != "ok":
            return []
        growth = round(ev.value, 6)
        direction = claim_direction(text) or "up"   # a growth sub-claim asserts growth unless it says otherwise
        expected = expected_move(direction, True)
        if expected > 0 and _has(_STRONG, text):
            stance = Stance.SUPPORTING if growth >= STRONG_GROWTH_THRESHOLD else Stance.CHALLENGING
        else:
            stance = _stance(growth, expected)
        # the two levels behind the growth rate, so the text states them in readable units
        prev_end = _near(self.store, ticker, concept, str(ev.period_end), 365, cutoff)   # as compute_growth does
        cur_level = self._one(ticker, concept, ev.period_end, cutoff, ev.span)
        prev_level = self._one(ticker, concept, prev_end, cutoff, ev.span)
        metric_name = f"{_NAMES[concept]}_growth_yoy"
        content = (f"{_LABELS.get(concept, concept)} was {_fmt(concept, cur_level.value, cur_level.unit)} in the "
                   f"{_period(ev)} vs {_fmt(concept, prev_level.value, prev_level.unit)} a year earlier "
                   f"(period ended {prev_end}): {growth:+.1%} year on year.")
        items = [self._item(sub_claim, [ev], ev.period_end, metric_name, growth, stance, content, _period(ev))]
        if _has(_TREND, text):
            items += self._growth_trend(sub_claim, cutoff, concept, ev.span, expected)
        return items

    def _growth_trend(self, sub_claim: SubClaim, cutoff: date, concept: str, span: str,
                      expected: int) -> list[EvidenceItem]:
        record_tool_call()
        tr = compute_trend(self.store, sub_claim.company_ticker, concept, 4, span, "yoy", "latest", cutoff)
        if tr.status != "ok":
            return []
        moved = {"up": 1, "down": -1}.get(tr.direction, 0)
        content = (f"{_LABELS.get(concept, concept)} year-on-year growth by period, oldest first: "
                   + ", ".join(f"{p.value:+.1%} ({p.period_end})" for p in tr.series)
                   + f". Direction: {tr.direction}.")
        return [self._item(sub_claim, [tr], tr.series[-1].period_end, f"{_NAMES[concept]}_growth_yoy_trend",
                           round(tr.value, 6), _stance(moved, expected), content, _period(tr))]

    # -------------------------------------------------------------- margins, cash, leverage
    def _one(self, ticker: str, item: str, period_end, cutoff: date, span: str = "quarter") -> FinancialEvidence:
        record_tool_call()
        if item in METRICS:
            return compute_metric(self.store, ticker, item, span, period_end, cutoff)
        return get_financial_fact(self.store, ticker, item, span, period_end, cutoff)

    def _levels(self, sub_claim: SubClaim, cutoff: date, picked: list[tuple[str, bool | None]],
                use_default: bool) -> list[EvidenceItem]:
        ticker, direction, items = sub_claim.company_ticker, claim_direction(sub_claim.text), []
        span = "annual" if _has(_ANNUAL, sub_claim.text) else "quarter"
        candidates = picked or ([(m, True) for m in _DEFAULT_METRICS] if use_default else [])
        for item, higher_is_better in candidates:
            cur = self._one(ticker, item, "latest", cutoff, span)
            if cur.status != "ok":
                continue
            prev_end = prior_year_end(self.store, ticker, cur.period_end, cutoff)
            prev = self._one(ticker, item, prev_end, cutoff, span) if prev_end else None
            label, value = _LABELS.get(item, item), round(cur.value, 6)
            when = _period(cur)
            where = when if when.startswith("as at") else f"in the {when}"
            content = f"{label} was {_fmt(item, cur.value, cur.unit)} {where}"
            evidence, stance = [cur], Stance.NEUTRAL
            if prev is not None and prev.status == "ok":
                evidence.append(prev)
                stance = _stance(cur.value - prev.value, expected_move(direction, higher_is_better))
                content += f" vs {_fmt(item, prev.value, prev.unit)} a year earlier ({_period(prev)})"
            content += "."
            items.append(self._item(sub_claim, evidence, cur.period_end, _NAMES.get(item, item), value, stance,
                                    content, _period(cur)))
            if not picked:
                break   # default list: stop at the first metric that has data
        return items

    # -------------------------------------------------------------- EvidenceItem
    def _item(self, sub_claim: SubClaim, evidence: list[FinancialEvidence], period_end, metric_name: str,
              value: float, stance: Stance, content: str, period: str) -> EvidenceItem:
        facts = [f for ev in evidence for f in ev.inputs]
        cited = max(facts, key=lambda f: (f.period_end, f.filed))     # the filing for the latest period
        tags = sorted({f"us-gaap:{f.xbrl_tag}" for f in facts})
        filings = sorted({f"{f.form} {f.accession}" for f in facts})
        notes = sorted({w for ev in evidence for w in ev.warnings})
        if any(ev.derived for ev in evidence):
            notes.append("includes a figure derived from reported values (year-to-date subtraction or a fallback)")
        ticker = sub_claim.company_ticker or ""
        return EvidenceItem(
            evidence_id=f"XBRL-{ticker}-{metric_name}-{period_end}",
            sub_claim_id=sub_claim.sub_claim_id, stance=stance,
            content=content + (" Note: " + "; ".join(notes) + "." if notes else ""),
            source_tier=SourceTier.TIER_1, doc_type=cited.form,
            company=(CORPUS_COMPANIES.get(ticker) or [ticker])[0], period=period,
            publication_date=cited.filed, url=cited.source_url,
            citation_span="XBRL " + " / ".join(tags) + "; filings: " + ", ".join(filings),
            produced_by=self.name, computed_value=value, metric_name=metric_name)
