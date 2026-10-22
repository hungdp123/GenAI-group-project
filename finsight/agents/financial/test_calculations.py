"""
Known-answer tests for the financial tools.

Every test builds a tiny fact store by hand so the right answer can be worked out on paper and
written next to the assert. If a test fails, either the formula in tools.py / test_metrics.py changed or
the test's arithmetic is wrong - the comment on each assert shows the hand calculation.
"""

from datetime import date
from itertools import count

from finsight.agents.financial import test_metrics as checks
from finsight.agents.financial.normalize import period_class
from finsight.agents.financial.store import FactStore
from finsight.agents.financial.tools import (
    check_stated_value, compute_growth, compute_metric, compute_trend, compute_ttm, compute_ttm_growth,
    get_financial_fact, list_periods,
)

DEFAULT_TAG = {
    "Revenue": "Revenues", "CostOfRevenue": "CostOfRevenue", "GrossProfit": "GrossProfit",
    "OperatingIncome": "OperatingIncomeLoss", "NetIncome": "NetIncomeLoss",
    "OperatingCashFlow": "NetCashProvidedByUsedInOperatingActivities",
    "CapitalExpenditure": "PaymentsToAcquirePropertyPlantAndEquipment",
    "DilutedShares": "WeightedAverageNumberOfDilutedSharesOutstanding", "DilutedEPS": "EarningsPerShareDiluted",
    "TotalAssets": "Assets", "TotalLiabilities": "Liabilities", "LiabilitiesAndEquity": "LiabilitiesAndStockholdersEquity",
    "ShareholdersEquity": "StockholdersEquity", "CashAndEquivalents": "CashAndCashEquivalentsAtCarryingValue",
    "CurrentAssets": "AssetsCurrent", "CurrentLiabilities": "LiabilitiesCurrent", "Inventory": "InventoryNet",
}
UNITS = {"DilutedShares": "shares", "DilutedEPS": "USD/shares"}
_accn = count(1)


def F(concept, value, end, start=None, filed=None, tag=None, ticker="TST", accn=None):
    """One fact row as it would come out of facts.csv. Instant facts have no start."""
    return dict(ticker=ticker, cik=1, concept=concept, xbrl_tag=tag or DEFAULT_TAG[concept], value=value,
                unit=UNITS.get(concept, "USD"), period_start=start or "", period_end=end,
                period_class=period_class(start, end), form="10-K" if filed and filed[5:7] == "02" else "10-Q",
                accession=accn or f"0000000000-00-{next(_accn):06d}", filed=filed, source_url="https://example.com")


def approx(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


# ------------------------------------------------------------------ fixture: calendar-year company "TST"
# Revenue (standalone quarters reported for Q1-Q3, Q4 only via the 10-K):
#   2024: Q1 100, Q2 120, Q3 140, (Q4 140)  FY 500   9M 360
#   2025: Q1 110, Q2 132, Q3 154, (Q4 164)  FY 560   9M 396
FILED = {"2024-03-31": "2024-05-01", "2024-06-30": "2024-08-01", "2024-09-30": "2024-11-01",
         "2024-12-31": "2025-02-01", "2025-03-31": "2025-05-01", "2025-06-30": "2025-08-01",
         "2025-09-30": "2025-11-01", "2025-12-31": "2026-02-01"}


def duration_rows(concept, year, q1, q2, q3, fy, ticker="TST", tag=None):
    y = str(year)
    s = f"{y}-01-01"
    return [
        F(concept, q1, f"{y}-03-31", s, FILED[f"{y}-03-31"], tag, ticker),
        F(concept, q2, f"{y}-06-30", f"{y}-04-01", FILED[f"{y}-06-30"], tag, ticker),
        F(concept, q1 + q2, f"{y}-06-30", s, FILED[f"{y}-06-30"], tag, ticker),
        F(concept, q3, f"{y}-09-30", f"{y}-07-01", FILED[f"{y}-09-30"], tag, ticker),
        F(concept, q1 + q2 + q3, f"{y}-09-30", s, FILED[f"{y}-09-30"], tag, ticker),
        F(concept, fy, f"{y}-12-31", s, FILED[f"{y}-12-31"], tag, ticker),
    ]


def instant(concept, value, end, ticker="TST"):
    return F(concept, value, end, None, FILED[end], ticker=ticker)


def base_facts():
    rows = []
    rows += duration_rows("Revenue", 2024, 100, 120, 140, 500)
    rows += duration_rows("Revenue", 2025, 110, 132, 154, 560)
    rows += duration_rows("CostOfRevenue", 2024, 60, 72, 84, 300)
    rows += duration_rows("CostOfRevenue", 2025, 66, 79, 92, 336)
    rows += duration_rows("GrossProfit", 2025, 44, 53, 62, 224)          # 2024 GrossProfit NOT tagged -> fallback
    rows += duration_rows("OperatingIncome", 2025, 22, 26, 31, 112)
    rows += duration_rows("NetIncome", 2025, 11, 13, 15, 56)
    rows += duration_rows("OperatingCashFlow", 2025, 30, 35, 40, 150)
    rows += duration_rows("CapitalExpenditure", 2025, 10, 10, 10, 40)
    # per-share and share counts: not additive, so no Q4 may be derived
    rows += [F("DilutedEPS", 0.56, "2025-12-31", "2025-01-01", "2026-02-01"),
             F("DilutedShares", 100, "2025-12-31", "2025-01-01", "2026-02-01"),
             F("DilutedEPS", 0.15, "2025-09-30", "2025-07-01", "2025-11-01"),
             F("DilutedEPS", 0.39, "2025-09-30", "2025-01-01", "2025-11-01")]   # 9M EPS: FY - 9M would "work"
    for end in ("2024-12-31", "2025-12-31"):
        rows += [instant("TotalAssets", 500, end), instant("LiabilitiesAndEquity", 500, end),
                 instant("ShareholdersEquity", 200, end), instant("CurrentAssets", 250, end),
                 instant("CurrentLiabilities", 125, end), instant("CashAndEquivalents", 50, end),
                 instant("Inventory", 40, end)]
    rows.append(instant("TotalLiabilities", 300, "2025-12-31"))         # 2024 Liabilities NOT tagged -> fallback
    for end in ("2025-03-31", "2025-06-30", "2025-09-30"):
        rows.append(instant("TotalAssets", 500, end))
    return rows


def store(rows=None):
    return FactStore(rows if rows is not None else base_facts())


# ------------------------------------------------------------------ single facts and derived quarters
def test_reported_quarter():
    ev = get_financial_fact(store(), "TST", "Revenue", "quarter", "2025-09-30")
    assert ev.status == "ok" and ev.value == 154 and not ev.derived
    assert len(ev.inputs) == 1 and ev.inputs[0].xbrl_tag == "Revenues"


def test_q4_derived_from_fy_minus_9m():
    ev = get_financial_fact(store(), "TST", "Revenue", "quarter", "latest")
    assert str(ev.period_end) == "2025-12-31"
    assert ev.value == 560 - 396 and ev.derived                          # FY 560 - 9M 396 = 164
    assert {i.period_class for i in ev.inputs} == {"FY", "9M"}


def test_annual_fact():
    assert get_financial_fact(store(), "TST", "Revenue", "annual").value == 560


def test_balance_sheet_fact():
    assert get_financial_fact(store(), "TST", "TotalAssets", "quarter", "2025-06-30").value == 500


def test_ytd_only_company_derives_every_quarter():
    # Company "YTD" files only cumulative figures: Q1 100, H1 250, 9M 420, FY 600
    rows = [F("Revenue", 100, "2025-03-31", "2025-01-01", "2025-05-01", ticker="YTD"),
            F("Revenue", 250, "2025-06-30", "2025-01-01", "2025-08-01", ticker="YTD"),
            F("Revenue", 420, "2025-09-30", "2025-01-01", "2025-11-01", ticker="YTD"),
            F("Revenue", 600, "2025-12-31", "2025-01-01", "2026-02-01", ticker="YTD")]
    s = store(rows)
    assert get_financial_fact(s, "YTD", "Revenue", "quarter", "2025-06-30").value == 150   # 250 - 100
    assert get_financial_fact(s, "YTD", "Revenue", "quarter", "2025-09-30").value == 170   # 420 - 250
    assert get_financial_fact(s, "YTD", "Revenue", "quarter", "2025-12-31").value == 180   # 600 - 420


def test_derived_quarter_uses_short_figure_as_filed_with_long_figure():
    # H1 was 250 when the 9M (420) was filed; H1 was restated to 260 later. Q3 must be 420 - 250 = 170,
    # not 420 - 260, because the 9M figure was built on the original H1.
    rows = [F("Revenue", 250, "2025-06-30", "2025-01-01", "2025-08-01", ticker="YTD"),
            F("Revenue", 420, "2025-09-30", "2025-01-01", "2025-11-01", ticker="YTD"),
            F("Revenue", 260, "2025-06-30", "2025-01-01", "2026-02-01", ticker="YTD")]
    ev = get_financial_fact(store(rows), "YTD", "Revenue", "quarter", "2025-09-30")
    assert ev.value == 170


def test_per_share_quarter_is_never_derived():
    # Q4 EPS cannot be FY EPS - 9M EPS (different share counts), so it must be missing, not computed
    ev = get_financial_fact(store(), "TST", "DilutedEPS", "quarter", "2025-12-31")
    assert ev.status == "insufficient_data" and ev.value is None


def test_tag_priority():
    # Two revenue tags for the same period: "Revenues" is first in the priority list and must win
    rows = [F("Revenue", 999, "2025-12-31", "2025-01-01", "2026-02-01",
              tag="RevenueFromContractWithCustomerExcludingAssessedTax"),
            F("Revenue", 560, "2025-12-31", "2025-01-01", "2026-02-01")]
    assert get_financial_fact(store(rows), "TST", "Revenue", "annual").value == 560


# ------------------------------------------------------------------ growth
def test_quarter_yoy_growth_with_derived_quarters():
    ev = compute_growth(store(), "TST", "Revenue", "yoy", "quarter", "2025-12-31")
    assert approx(ev.value, 164 / 140 - 1)                               # Q4 2025 164 vs Q4 2024 140 = +17.14%
    assert ev.unit == "ratio" and ev.derived


def test_quarter_yoy_growth_reported():
    ev = compute_growth(store(), "TST", "Revenue", "yoy", "quarter", "2025-09-30")
    assert approx(ev.value, 154 / 140 - 1)                               # +10.00%


def test_qoq_growth():
    ev = compute_growth(store(), "TST", "Revenue", "qoq", "quarter", "2025-12-31")
    assert approx(ev.value, 164 / 154 - 1)                               # +6.49%


def test_annual_yoy_growth():
    ev = compute_growth(store(), "TST", "Revenue", "yoy", "annual")
    assert approx(ev.value, 560 / 500 - 1)                               # +12.00%
    assert {str(i.period_end) for i in ev.inputs} == {"2025-12-31", "2024-12-31"}


def test_growth_refuses_negative_base():
    rows = [F("NetIncome", -50, "2024-12-31", "2024-01-01", "2025-02-01"),
            F("NetIncome", 30, "2025-12-31", "2025-01-01", "2026-02-01")]
    ev = compute_growth(store(rows), "TST", "NetIncome", "yoy", "annual")
    assert ev.status == "insufficient_data" and ev.value is None


def test_qoq_rejects_annual_span():
    ev = compute_growth(store(), "TST", "Revenue", "qoq", "annual")
    assert ev.status == "insufficient_data"


def test_53_week_year_is_flagged():
    # FY2024: 364 days; FY2025: 371 days (53 weeks). Growth is still returned, with a warning.
    rows = [F("Revenue", 520, "2024-12-28", "2023-12-31", "2025-02-01"),
            F("Revenue", 560, "2026-01-03", "2024-12-29", "2026-02-20")]
    ev = compute_growth(store(rows), "TST", "Revenue", "yoy", "annual")
    assert approx(ev.value, 560 / 520 - 1)
    assert any("differ in length" in w for w in ev.warnings)


# ------------------------------------------------------------------ metrics
def test_gross_margin_reported():
    ev = compute_metric(store(), "TST", "gross_margin", "annual", "2025-12-31")
    assert approx(ev.value, 224 / 560)                                   # 40.00%


def test_gross_margin_fallback_revenue_minus_cost():
    ev = compute_metric(store(), "TST", "gross_margin", "annual", "2024-12-31")
    assert approx(ev.value, (500 - 300) / 500)                           # GP not tagged: (500-300)/500 = 40.00%
    assert any("GrossProfit not tagged" in w for w in ev.warnings)


def test_operating_and_net_margin():
    assert approx(compute_metric(store(), "TST", "operating_margin", "annual").value, 112 / 560)   # 20%
    assert approx(compute_metric(store(), "TST", "net_margin", "annual").value, 56 / 560)         # 10%


def test_quarter_margin_uses_derived_q4_for_both_parts():
    ev = compute_metric(store(), "TST", "gross_margin", "quarter", "2025-12-31")
    gp_q4, rev_q4 = 224 - (44 + 53 + 62), 560 - 396                      # 65 / 164
    assert approx(ev.value, gp_q4 / rev_q4)


def test_free_cash_flow_and_margin():
    assert compute_metric(store(), "TST", "free_cash_flow", "annual").value == 150 - 40          # 110
    assert approx(compute_metric(store(), "TST", "fcf_margin", "annual").value, 110 / 560)
    q4 = compute_metric(store(), "TST", "free_cash_flow", "quarter", "2025-12-31")
    assert q4.value == (150 - 105) - (40 - 30)                           # derived Q4 OCF 45 - capex 10 = 35


def test_liquidity_and_leverage():
    s = store()
    assert approx(compute_metric(s, "TST", "current_ratio", "annual").value, 250 / 125)          # 2.0
    assert approx(compute_metric(s, "TST", "cash_ratio", "annual").value, 50 / 125)              # 0.4
    assert approx(compute_metric(s, "TST", "liabilities_to_equity", "annual").value, 300 / 200)  # 1.5


def test_liabilities_fallback():
    ev = compute_metric(store(), "TST", "liabilities_to_equity", "annual", "2024-12-31")
    assert approx(ev.value, (500 - 200) / 200)                           # Liabilities = L&E - equity = 300
    assert any("Liabilities not tagged" in w for w in ev.warnings)


def test_ratio_with_negative_denominator_is_refused():
    rows = [instant("TotalLiabilities", 300, "2025-12-31"), instant("ShareholdersEquity", -20, "2025-12-31"),
            F("Revenue", 1, "2025-12-31", "2025-01-01", "2026-02-01")]
    ev = compute_metric(store(rows), "TST", "liabilities_to_equity", "annual")
    assert ev.status == "insufficient_data" and ev.value is None


# ------------------------------------------------------------------ trailing twelve months
def test_ttm_at_year_end_equals_the_fiscal_year():
    ev = compute_ttm(store(), "TST", "Revenue", "2025-12-31")
    assert ev.value == 110 + 132 + 154 + 164 == 560 and ev.derived       # Q4 derived from FY - 9M


def test_ttm_mid_year_spans_two_fiscal_years():
    ev = compute_ttm(store(), "TST", "Revenue", "2025-09-30")
    assert ev.value == 140 + 110 + 132 + 154                             # Q4 2024 + Q1-Q3 2025 = 536


def test_ttm_margin_and_free_cash_flow_use_twelve_month_sums():
    gm = compute_ttm(store(), "TST", "gross_margin", "2025-12-31")
    assert approx(gm.value, (44 + 53 + 62 + 65) / 560)                   # 224 / 560
    fcf = compute_ttm(store(), "TST", "free_cash_flow", "2025-12-31")
    assert fcf.value == 150 - 40                                         # 110


def test_ttm_growth():
    ev = compute_ttm_growth(store(), "TST", "Revenue", "2025-12-31")
    assert approx(ev.value, 560 / 500 - 1)
    ev = compute_ttm_growth(store(), "TST", "Revenue", "2025-09-30")     # needs Q4 2023: not in the data
    assert ev.status == "insufficient_data"


def test_ttm_refuses_balance_sheet_items():
    assert compute_ttm(store(), "TST", "current_ratio", "2025-12-31").status == "insufficient_data"


# ------------------------------------------------------------------ cut-off dates (no look-ahead)
def test_cutoff_hides_later_filings():
    s = store()
    # 10-K for FY2025 is filed 2026-02-01: on 2026-01-31 the latest year is still FY2024
    assert get_financial_fact(s, "TST", "Revenue", "annual", cutoff_date=date(2026, 1, 31)).value == 500
    ev = get_financial_fact(s, "TST", "Revenue", "quarter", cutoff_date=date(2026, 1, 31))
    assert str(ev.period_end) == "2025-09-30" and ev.value == 154


def test_cutoff_is_inclusive():
    ev = get_financial_fact(store(), "TST", "Revenue", "annual", cutoff_date=date(2026, 2, 1))
    assert ev.value == 560


def test_restatement_respects_cutoff():
    rows = base_facts() + [F("Revenue", 505, "2024-12-31", "2024-01-01", "2026-02-01")]   # FY2024 restated
    s = store(rows)
    assert get_financial_fact(s, "TST", "Revenue", "annual", "2024-12-31", date(2025, 6, 1)).value == 500
    assert get_financial_fact(s, "TST", "Revenue", "annual", "2024-12-31").value == 505
    assert approx(compute_growth(s, "TST", "Revenue", "yoy", "annual").value, 560 / 505 - 1)


def test_no_inputs_filed_after_cutoff():
    cutoff = date(2025, 11, 30)
    for ev in (compute_growth(store(), "TST", "Revenue", "yoy", "quarter", cutoff_date=cutoff),
               compute_metric(store(), "TST", "gross_margin", "quarter", cutoff_date=cutoff),
               compute_trend(store(), "TST", "Revenue", 3, cutoff_date=cutoff)):
        assert ev.status == "ok"
        assert all(i.filed <= cutoff for i in ev.inputs), ev.inputs


# ------------------------------------------------------------------ trends
def test_revenue_trend():
    ev = compute_trend(store(), "TST", "Revenue", periods=4, span="quarter", period_end="2025-12-31")
    assert [p.value for p in ev.series] == [110, 132, 154, 164]
    assert ev.direction == "up" and ev.value == 164 - 110
    assert [p.derived for p in ev.series] == [False, False, False, True]


def test_yoy_trend():
    ev = compute_trend(store(), "TST", "Revenue", periods=4, basis="yoy", period_end="2025-12-31")
    expected = [110 / 100 - 1, 132 / 120 - 1, 154 / 140 - 1, 164 / 140 - 1]
    assert all(approx(p.value, e) for p, e in zip(ev.series, expected))
    assert len(ev.series) == 4


def test_trend_stops_at_gap_and_warns():
    ev = compute_trend(store(), "TST", "gross_margin", periods=8, period_end="2025-12-31")
    # GrossProfit is only tagged for 2025 but 2024 has Revenue and Cost, so the fallback fills 2024 too
    assert len(ev.series) == 8
    ev = compute_trend(store(), "TST", "operating_margin", periods=8, period_end="2025-12-31")
    assert len(ev.series) == 4 and any("only 4 of 8" in w for w in ev.warnings)


# ------------------------------------------------------------------ stated-value check (auditor)
def test_check_stated_value_ratio_uses_percentage_points():
    ev = compute_growth(store(), "TST", "Revenue", "yoy", "annual")       # 0.12
    assert check_stated_value(ev, 0.125).match                           # 0.5 pp off: within 1 pp
    assert not check_stated_value(ev, 0.14).match                        # 2 pp off
    assert not check_stated_value(ev, 12).match                          # "12" instead of 0.12 must not match


def test_check_stated_value_usd_is_relative():
    ev = get_financial_fact(store(), "TST", "Revenue", "annual")         # 560
    assert check_stated_value(ev, 565).match                             # 0.89% off
    assert not check_stated_value(ev, 570).match                         # 1.79% off


def test_check_stated_value_without_computed_value():
    ev = get_financial_fact(store(), "TST", "DilutedEPS", "quarter", "2025-12-31")
    assert not check_stated_value(ev, 0.15).match


# ------------------------------------------------------------------ evidence hygiene
def test_evidence_id_is_stable_and_specific():
    a = compute_growth(store(), "TST", "Revenue", "yoy", "annual")
    b = compute_growth(store(), "TST", "Revenue", "yoy", "annual")
    c = compute_growth(store(), "TST", "Revenue", "yoy", "annual", cutoff_date=date(2026, 1, 31))
    assert a.evidence_id == b.evidence_id != c.evidence_id


def test_unknown_ticker_and_concept():
    assert get_financial_fact(store(), "NOPE", "Revenue").status == "out_of_scope"
    assert get_financial_fact(store(), "TST", "Vibes").status == "insufficient_data"


def test_list_periods():
    p = list_periods(store(), "TST", date(2026, 1, 31))
    assert p["annual_ends"] == ["2024-12-31"] and p["quarter_ends"][-1] == "2025-09-30"


# ------------------------------------------------------------------ accounting checks (test_metrics.py)
def levels(rows, name):
    return {r["level"] for r in rows if r["check"] == name}


def test_checks_pass_on_clean_data():
    rows = checks.run(store())
    assert not [r for r in rows if r["level"] == "FAIL"], [r for r in rows if r["level"] == "FAIL"]
    assert levels(rows, "gross profit = revenue - cost of revenue") == {"PASS"}
    assert "PASS" in levels(rows, "Revenue: quarters add up to year-to-date")


def test_checks_catch_broken_identities():
    rows = base_facts()
    rows = [r for r in rows if not (r["concept"] == "LiabilitiesAndEquity" and r["period_end"] == "2025-12-31")]
    rows.append(instant("LiabilitiesAndEquity", 600, "2025-12-31"))     # assets 500 != L&E 600
    rows = [r for r in rows if not (r["concept"] == "GrossProfit" and r["period_class"] == "FY")]
    rows.append(F("GrossProfit", 300, "2025-12-31", "2025-01-01", "2026-02-01"))   # 560 - 336 = 224, not 300
    out = checks.run(store(rows))
    fails = {r["check"] for r in out if r["level"] == "FAIL"}
    assert "assets = liabilities + equity" in fails
    assert "gross profit = revenue - cost of revenue" in fails


def test_checks_catch_quarters_not_adding_up():
    rows = [r for r in base_facts() if not (r["concept"] == "Revenue" and r["period_class"] == "H1"
                                             and r["period_end"] == "2025-06-30")]
    rows.append(F("Revenue", 300, "2025-06-30", "2025-01-01", "2025-08-01"))   # Q1 110 + Q2 132 = 242, not 300
    out = checks.run(store(rows))
    assert any(r["check"] == "Revenue: quarters add up to year-to-date" and r["level"] == "FAIL"
               and "2025-06-30" in r["period"] for r in out)


def test_checks_eps_identity():
    rows = checks.run(store())
    # 0.56 x 100 shares = 56 = NetIncome FY2025
    assert levels(rows, "EPS x diluted shares = net income") == {"PASS"}


# ------------------------------------------------------------------ normalization (normalize.py)
def test_trailing_twelve_months_in_10q_is_not_a_fiscal_year():
    # Amazon's 10-Qs tag 12-month cash flow ending at each quarter. Those must not become "FY" facts,
    # or the latest fiscal year resolves to a quarter end (AMZN: 2025-03-31).
    from finsight.agents.financial.normalize import normalize_companyfacts
    ocf = "NetCashProvidedByUsedInOperatingActivities"
    raw = {"cik": 1, "facts": {"us-gaap": {ocf: {"units": {"USD": [
        {"start": "2024-01-01", "end": "2024-12-31", "val": 115, "form": "10-K", "accn": "a1", "filed": "2025-02-07"},
        {"start": "2024-04-01", "end": "2025-03-31", "val": 114, "form": "10-Q", "accn": "a2", "filed": "2025-05-02"},
        {"start": "2025-01-01", "end": "2025-03-31", "val": 17, "form": "10-Q", "accn": "a2", "filed": "2025-05-02"},
    ]}}}}}
    stats = {}
    rows = normalize_companyfacts(raw, "AMZN", stats)
    assert [(r["period_class"], r["period_end"]) for r in rows] == [("FY", "2024-12-31"), ("Q", "2025-03-31")]
    assert stats.get("trailing_twelve_months") == 1
    s = FactStore(rows)
    assert list_periods(s, "AMZN")["annual_ends"] == ["2024-12-31"]


# ------------------------------------------------------------------ runner without pytest
if __name__ == "__main__":
    import sys
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:                                                # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc(limit=3)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
