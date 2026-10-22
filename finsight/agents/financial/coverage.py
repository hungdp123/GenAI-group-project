"""
Reports which concepts and metrics are available per company + where the concept mapping needs additions.
Prints two grids (latest quarter, latest annual) and writes <data>/processed/coverage.csv.
Codes:  R reported   D derived from year-to-date figures   F fallback formula   - missing
"""

import argparse
import csv
import json
from datetime import date, timedelta
from pathlib import Path

from .concepts import CONCEPTS, TAG_TO_CONCEPT, unit
from .store import DEFAULT_DATA_DIR, FactStore
from .tools import METRICS, _resolve_end, _val, compute_metric

CODES = "Codes:  R reported   D derived from year-to-date figures   F fallback formula   - missing"

# words to look for in unmapped tags when a concept is missing
HINTS = {
    "Revenue": ["Revenue", "Sales"],
    "CostOfRevenue": ["CostOf", "CostsAndExpenses"],
    "GrossProfit": ["GrossProfit"],
    "OperatingIncome": ["OperatingIncome", "IncomeLossFromContinuingOperationsBefore"],
    "NetIncome": ["NetIncome", "ProfitLoss"],
    "OperatingCashFlow": ["NetCashProvidedByUsedInOperating"],
    "CapitalExpenditure": ["PaymentsToAcquire", "PaymentsForCapital"],
    "DilutedShares": ["WeightedAverageNumber", "SharesOutstanding"],
    "DilutedEPS": ["EarningsPerShare", "IncomeLossFromContinuingOperationsPerDilutedShare"],
    "TotalAssets": ["Assets"],
    "TotalLiabilities": ["Liabilities"],
    "LiabilitiesAndEquity": ["LiabilitiesAnd"],
    "ShareholdersEquity": ["StockholdersEquity"],
    "CashAndEquivalents": ["CashAndCashEquivalents", "CashCashEquivalents"],
    "CurrentAssets": ["AssetsCurrent"],
    "CurrentLiabilities": ["LiabilitiesCurrent"],
    "Inventory": ["Inventory"],
}

# only used for printing the grid in the result
ABBR = {
    "CostOfRevenue": "CostOfRev", "GrossProfit": "GrossProf", "OperatingIncome": "OpIncome",
    "OperatingCashFlow": "OpCashFlw", "CapitalExpenditure": "Capex", "DilutedShares": "DilShares",
    "DilutedEPS": "DilEPS", "TotalAssets": "Assets", "TotalLiabilities": "Liabs", "LiabilitiesAndEquity": "Liab+Eq",
    "ShareholdersEquity": "Equity", "CashAndEquivalents": "Cash", "CurrentAssets": "CurAssets",
    "CurrentLiabilities": "CurLiabs", "gross_margin": "GrossMgn", "operating_margin": "OpMgn",
    "net_margin": "NetMgn", "free_cash_flow": "FCF", "fcf_margin": "FCFMgn", "liabilities_to_equity": "Liab/Eq",
    "current_ratio": "CurRatio", "cash_ratio": "CashRatio",
}


# finds possible tags for a missing concept
def candidates(raw_dir, ticker, concept, latest_end):
    """Unmapped us-gaap tags in the raw file that look related and have recent data in the right unit"""
    p = Path(raw_dir) / "companyfacts" / f"{ticker}.json"
    if not p.exists():
        return []
    floor = (date.fromisoformat(latest_end) - timedelta(days=400)).isoformat()
    out = []
    for tag, body in json.loads(p.read_text(encoding="utf-8")).get("facts", {}).get("us-gaap", {}).items():
        if tag in TAG_TO_CONCEPT or not any(h in tag for h in HINTS[concept]):
            continue
        recent = [f for f in body.get("units", {}).get(unit(concept), [])
                  if f["end"] >= floor and f.get("form") in ("10-K", "10-Q")]
        if recent:
            last = max(recent, key=lambda f: (f["end"], f["filed"]))
            out.append((tag, last["end"], last["val"]))
    return sorted(out, key=lambda x: -abs(x[2]))[:6]


def _code(found, fallback, derived):
    return "-" if not found else "F" if fallback else "D" if derived else "R"


def build(store, cutoff=None):
    rows = []
    for t in store.tickers:
        for span in ("quarter", "annual"):
            end = _resolve_end(store, t, "latest", span, cutoff)
            for c in CONCEPTS:
                r = end and _val(store, t, c, end, span, cutoff)
                rows.append(dict(
                    ticker=t, span=span, period_end=end or "", kind="concept", item=c,
                    code=_code(r, r and r[4], r and r[3]), value=r[0] if r else "",
                    xbrl_tag=r[1][0]["xbrl_tag"] if r else "", note="; ".join(r[4]) if r else ""))
            for m in METRICS:
                e = compute_metric(store, t, m, span, "latest", cutoff)
                rows.append(dict(
                    ticker=t, span=span, period_end=end or "", kind="metric", item=m,
                    code=_code(e.status == "ok", e.warnings, e.derived), value="" if e.value is None else e.value,
                    xbrl_tag="", note=e.reason or "; ".join(e.warnings)))
    return rows


def grid(rows, span, kind, items):
    hdr = {i: ABBR.get(i, i)[:9] for i in items}
    lines = [f"{'':6} {'period end':10}  " + " ".join(f"{hdr[i]:>9}" for i in items)]
    for t in sorted({r["ticker"] for r in rows}):
        sub = {r["item"]: r for r in rows if r["ticker"] == t and r["span"] == span and r["kind"] == kind}
        period_end = next(iter(sub.values()))["period_end"]
        lines.append(f"{t:6} {period_end:10}  " + " ".join(f"{sub[i]['code']:>9}" for i in items))
    return "\n".join(lines)


def print_suggestions(rows, raw_dir):
    print("=== Missing concepts: unmapped tags in the raw file that may be the right one ===")
    seen = set()
    for r in rows:
        key = (r["ticker"], r["item"])
        if r["kind"] != "concept" or r["code"] != "-" or not r["period_end"] or key in seen:
            continue
        seen.add(key)
        found = candidates(raw_dir, r["ticker"], r["item"], r["period_end"])
        print(f"{r['ticker']} {r['item']} ({r['span']} {r['period_end']}):", "" if found else "no similar tag found")
        for tag, end, val in found:
            print(f"      {tag}  ({end}: {val:,.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--cutoff", default=None)
    args = ap.parse_args()
    data = Path(args.data)
    cutoff = date.fromisoformat(args.cutoff) if args.cutoff else None
    rows = build(FactStore.from_dir(data / "processed"), cutoff)
    print(CODES, "\n")
    for span in ("quarter", "annual"):
        print(f"=== Concepts, latest {span}" + (f" as of {cutoff}" if cutoff else "") + " ===")
        print(grid(rows, span, "concept", list(CONCEPTS)), "\n")
        print(f"=== Metrics, latest {span} ===")
        print(grid(rows, span, "metric", list(METRICS)), "\n")
    print_suggestions(rows, data / "raw")
    out = data / "processed" / "coverage.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    concept_rows = [r for r in rows if r["kind"] == "concept"]
    filled = sum(r["code"] != "-" for r in concept_rows)
    print(f"\n{filled}/{len(concept_rows)} concept cells filled. Full detail (values, tags used, reasons): {out}")


if __name__ == "__main__":
    main()
