"""
Compares a few figures from the pulled SEC data with the numbers in the eval set.

Run from the repo root:  python -m finsight.agents.financial.spot_check
"""

from finsight.agents.financial.store import FactStore
from finsight.agents.financial.tools import compute_growth, compute_metric, prior_year_end

store = FactStore.from_dir("data/sec/processed")

# (what, ticker, period end or "latest", expected %)
GROWTH = [
    ("Alphabet revenue growth, Q1 2026", "GOOGL", "2026-03-31", 21.8),
    ("Alphabet revenue growth, Q2 2026", "GOOGL", "2026-06-30", 24.2),
    ("Meta revenue growth, Q1 2026", "META", "2026-03-31", 33.1),
    ("Microsoft revenue growth, March 2026 quarter", "MSFT", "2026-03-31", 18.3),
]


def show(what, ev, expected):
    if ev.status != "ok":
        print(f"{what:48} NO VALUE  ({ev.reason})")
        return
    got = round(ev.value * 100, 1)
    ok = "OK" if abs(got - expected) <= 0.1 else "DIFFERENT"
    print(f"{what:48} got {got:5.1f}%  expected {expected:5.1f}%  {ok}   (period {ev.period_end})")


for what, ticker, end, expected in GROWTH:
    show(what, compute_growth(store, ticker, "Revenue", "yoy", "quarter", end), expected)

# Nvidia operating margin: latest quarter and the same quarter a year earlier
latest = compute_metric(store, "NVDA", "operating_margin", "quarter", "latest")
earlier_end = prior_year_end(store, "NVDA", latest.period_end) if latest.status == "ok" else None
earlier = compute_metric(store, "NVDA", "operating_margin", "quarter", earlier_end) if earlier_end else None
if earlier is not None:
    show("Nvidia operating margin, year earlier", earlier, 62.4)
show("Nvidia operating margin, latest quarter", latest, 60.4)
