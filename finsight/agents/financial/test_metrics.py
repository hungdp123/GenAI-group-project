"""
Accounting sanity checks on the pulled data.

This catches wrong tag mappings without reading a filing.

FAIL = an accounting identity is broken: the mapping is almost certainly wrong for that company.
WARN = suspicious: look at the filing before trusting the number.
INFO = worth knowing but not a problem by itself.
"""

import argparse
import csv
from datetime import date
from pathlib import Path

from .concepts import CONCEPTS, additive, kind
from .store import DEFAULT_DATA_DIR, FactStore
from .tools import _near, _raw, _resolve_end, _val

TOL = 0.01   # 1% relative tolerance for identities


# Tag pairs that differ by definition (with / without minority interests): reported as INFO, not WARN.
BY_DEFINITION = {"NetIncome", "ShareholdersEquity"}


def _rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


def _close(a, b, scale=None, tol=TOL):
    """Equal within `tol`, relative to the two values or to `scale` (so two figures near zero compare equal)."""
    return abs(a - b) <= tol * max(abs(a), abs(b), abs(scale or 0), 1e-9)


def _periods(store, ticker, cutoff, quarters, years):
    """[(span, period_end)] for the latest N quarters and N fiscal years."""
    out = []
    for span, n, step in (("quarter", quarters, 91), ("annual", years, 365)):
        end = _resolve_end(store, ticker, "latest", span, cutoff)
        while end and n > 0:
            out.append((span, end))
            end, n = _near(store, ticker, "Revenue", end, step, cutoff), n - 1
    return out


def _get(store, ticker, concept, end, span, cutoff):
    r = _raw(store, ticker, concept, end, span, cutoff)   # reported values only: no fallback formulas
    return r[0] if r else None


def check_period(store, ticker, span, end, cutoff):
    """Identity and plausibility checks for one period. Yields (check, level, detail)."""
    v = {c: _get(store, ticker, c, end, span, cutoff) for c in CONCEPTS}

    def have(*names):
        return all(v[n] is not None for n in names)

    if have("TotalAssets", "LiabilitiesAndEquity"):
        ok = _rel(v["TotalAssets"], v["LiabilitiesAndEquity"]) <= TOL
        yield ("assets = liabilities + equity", "PASS" if ok else "FAIL",
               f"Assets {v['TotalAssets']:,.0f} vs Liabilities and equity {v['LiabilitiesAndEquity']:,.0f}")
    if have("TotalLiabilities", "ShareholdersEquity", "LiabilitiesAndEquity"):
        total, parts = v["LiabilitiesAndEquity"], v["TotalLiabilities"] + v["ShareholdersEquity"]
        gap = (total - parts) / abs(total)
        level = "FAIL" if gap < -TOL else "WARN" if gap > 0.05 else "PASS"
        yield ("liabilities + equity add up", level,
               f"Liabilities {v['TotalLiabilities']:,.0f} + equity {v['ShareholdersEquity']:,.0f} = {parts:,.0f} "
               f"vs total {total:,.0f} (gap {gap:+.1%}; a positive gap is minority or temporary equity)")
    if have("GrossProfit", "Revenue", "CostOfRevenue"):
        ok = _rel(v["GrossProfit"], v["Revenue"] - v["CostOfRevenue"]) <= TOL
        yield ("gross profit = revenue - cost of revenue", "PASS" if ok else "FAIL",
               f"GrossProfit {v['GrossProfit']:,.0f} vs Revenue - CostOfRevenue "
               f"{v['Revenue'] - v['CostOfRevenue']:,.0f}")
    if have("Revenue"):
        yield ("revenue is positive", "PASS" if v["Revenue"] > 0 else "FAIL", f"Revenue {v['Revenue']:,.0f}")
    if have("Revenue", "CostOfRevenue"):
        share = v["CostOfRevenue"] / v["Revenue"] if v["Revenue"] else 0
        level = "FAIL" if share > 1.5 else "WARN" if share < 0.05 or share > 1 else "PASS"
        yield ("cost of revenue is a plausible share of revenue", level, f"CostOfRevenue / Revenue = {share:.1%}")
    for part, whole in (("CurrentAssets", "TotalAssets"), ("CashAndEquivalents", "TotalAssets"),
                        ("Inventory", "CurrentAssets"), ("CashAndEquivalents", "CurrentAssets"),
                        ("CurrentLiabilities", "TotalLiabilities")):
        if have(part, whole):
            ok = v[part] <= v[whole] * (1 + TOL)
            yield (f"{part} <= {whole}", "PASS" if ok else "FAIL", f"{v[part]:,.0f} vs {v[whole]:,.0f}")
    if have("DilutedEPS", "DilutedShares", "NetIncome") and v["NetIncome"]:
        implied = v["DilutedEPS"] * v["DilutedShares"]
        rounding = 0.005 * abs(v["DilutedShares"])          # EPS is reported to the cent
        ok = abs(implied - v["NetIncome"]) <= 0.05 * abs(v["NetIncome"]) + rounding
        yield ("EPS x diluted shares = net income", "PASS" if ok else "WARN",
               f"{v['DilutedEPS']:.2f} x {v['DilutedShares']:,.0f} = {implied:,.0f} vs NetIncome {v['NetIncome']:,.0f}")


def check_tags(store, ticker, periods, cutoff):
    """Tag ambiguity within a period and tag changes between periods. Yields (check, period, level, detail)."""
    for concept in CONCEPTS:
        used = []
        for span, end in periods:
            cls = "I" if kind(concept) == "instant" else {"quarter": "Q", "annual": "FY"}[span]
            rows = [r for r in store.visible(ticker, concept, cutoff)
                    if r["period_end"] == end and r["period_class"] == cls]
            latest = {}
            for r in sorted(rows, key=lambda r: r["filed"]):
                latest[r["xbrl_tag"]] = r["value"]
            if len(latest) > 1 and _rel(max(latest.values()), min(latest.values())) > TOL:
                detail = "; ".join(f"{t}={val:,.0f}" for t, val in latest.items())
                level = "INFO" if concept in BY_DEFINITION else "WARN"
                yield (f"{concept}: one meaning per period", f"{span} {end}", level,
                       f"two mapped tags disagree ({detail}); the first in the priority list is used")
            best = store.fact(ticker, concept, cls, cutoff, end=end)
            if best:
                used.append((span, end, best["xbrl_tag"]))
        for span in ("quarter", "annual"):
            tags = [(e, t) for s, e, t in used if s == span]
            changed = sorted({t for _, t in tags})
            if len(changed) > 1:
                yield (f"{concept}: same tag across periods", span, "INFO",
                       "tag differs by period: " + ", ".join(f"{e}: {t}" for e, t in tags))


def check_availability(store, ticker, cutoff):
    """A concept reported for quarters but not for the year (or the reverse) often means a partial tag."""
    q_end = _resolve_end(store, ticker, "latest", "quarter", cutoff)
    y_end = _resolve_end(store, ticker, "latest", "annual", cutoff)
    if not q_end or not y_end:
        return
    for concept in CONCEPTS:
        if kind(concept) == "instant" or not additive(concept):
            continue
        in_q = _val(store, ticker, concept, q_end, "quarter", cutoff) is not None
        in_y = _val(store, ticker, concept, y_end, "annual", cutoff) is not None
        if in_q != in_y:
            where = "the latest quarter but not the latest fiscal year" if in_q else \
                "the latest fiscal year but not the latest quarter"
            yield (f"{concept}: available for both quarter and year", f"{q_end} / {y_end}", "WARN",
                   f"reported for {where}: the tag may cover only part of this line")


def check_ytd(store, ticker, cutoff):
    """Standalone quarters should add up to the reported year-to-date figure."""
    for concept in ("Revenue", "NetIncome"):
        for long_cls, short_cls in (("H1", "Q"), ("9M", "H1")):
            longs = [r for r in store.visible(ticker, concept, cutoff) if r["period_class"] == long_cls]
            for end in sorted({r["period_end"] for r in longs})[-2:]:
                long = store.fact(ticker, concept, long_cls, cutoff, end=end)
                # compare like with like: the pieces as they stood when the year-to-date figure was filed
                first = store.fact(ticker, concept, short_cls, long["filed"], start=long["period_start"])
                last = store.fact(ticker, concept, "Q", long["filed"], end=end)
                if first and last and first["xbrl_tag"] == last["xbrl_tag"] == long["xbrl_tag"]:
                    revenue = store.fact(ticker, "Revenue", long_cls, cutoff, end=end)
                    ok = _close(first["value"] + last["value"], long["value"],
                                scale=revenue["value"] * 0.1 if revenue else None)
                    yield (f"{concept}: quarters add up to year-to-date", f"{long_cls} {end}",
                           "PASS" if ok else "FAIL",
                           f"{first['value']:,.0f} + {last['value']:,.0f} vs {long['value']:,.0f}")


def run(store, cutoff=None, quarters=4, years=2):
    rows = []
    for ticker in store.tickers:
        periods = _periods(store, ticker, cutoff, quarters, years)
        for span, end in periods:
            for check, level, detail in check_period(store, ticker, span, end, cutoff):
                rows.append(dict(ticker=ticker, check=check, period=f"{span} {end}", level=level, detail=detail))
        for source in (check_tags(store, ticker, periods, cutoff), check_availability(store, ticker, cutoff),
                       check_ytd(store, ticker, cutoff)):
            for check, period, level, detail in source:
                rows.append(dict(ticker=ticker, check=check, period=period, level=level, detail=detail))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--cutoff", default=None)
    ap.add_argument("--quarters", type=int, default=4)
    ap.add_argument("--years", type=int, default=2)
    args = ap.parse_args()
    data = Path(args.data)
    cutoff = date.fromisoformat(args.cutoff) if args.cutoff else None
    rows = run(FactStore.from_dir(data / "processed"), cutoff, args.quarters, args.years)
    counts = {level: sum(r["level"] == level for r in rows) for level in ("PASS", "INFO", "WARN", "FAIL")}
    for level in ("FAIL", "WARN", "INFO"):
        for r in (r for r in rows if r["level"] == level):
            print(f"[{level}] {r['ticker']:6} {r['check']} ({r['period']}): {r['detail']}")
    out = data / "processed" / "checks.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "check", "period", "level", "detail"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n{counts['PASS']} passed, {counts['INFO']} info, {counts['WARN']} warnings, {counts['FAIL']} failures. "
          f"Every result: {out}")


if __name__ == "__main__":
    main()
