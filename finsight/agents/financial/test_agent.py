"""
Tests for the financial agent: how it reads a claim (claim_reader.py) and which figures it reports (agent.py).

The first group feeds post lines (and user prompts) to the claim reader and checks which measures and periods
it finds. The second group runs the agent on a small made-up company and checks the figures. The made-up
numbers mirror the eval items (e.g. 21.8% quarterly growth for item 02) so a failure points at the item.

The agent reports real figures only and every piece of evidence is NEUTRAL: judging the claim is the Auditor's
job. When the agent returns nothing, the tests check the reason it gives.

Run:  python -m finsight.agents.financial.test_agent
"""

from datetime import date
from itertools import count
from types import SimpleNamespace

from finsight.agents.financial.agent import (LEVERAGE_REASON, XbrlFinancialAgent,
                                             company_display_name, named_period)
from finsight.agents.financial.claim_reader import read_claim
from finsight.agents.financial.normalize import period_class
from finsight.agents.financial.store import FactStore
from finsight.contracts.schemas import Stance

# Sub-claim types by name, as in knowledge_layer.yaml v0.6
GROWTH, NET_INCOME, PROFIT, CASH, LIQUIDITY, LEVERAGE = (
    "HISTORICAL_GROWTH", "NET_INCOME_GROWTH", "PROFITABILITY", "CASH_GENERATION", "LIQUIDITY", "LEVERAGE")
TAGS = {"Revenue": "Revenues", "OperatingIncome": "OperatingIncomeLoss", "GrossProfit": "GrossProfit",
        "NetIncome": "NetIncomeLoss", "OperatingCashFlow": "NetCashProvidedByUsedInOperatingActivities",
        "CapitalExpenditure": "PaymentsToAcquirePropertyPlantAndEquipment",
        "CashAndEquivalents": "CashAndCashEquivalentsAtCarryingValue", "TotalAssets": "Assets",
        "CurrentAssets": "AssetsCurrent", "CurrentLiabilities": "LiabilitiesCurrent"}
M = 1e6
_accn, _accn_by_filing = count(1), {}


def make_fact(ticker, concept, value, end, start, filed, form="10-Q"):
    return dict(ticker=ticker, cik=1, concept=concept, xbrl_tag=TAGS[concept], value=value * M, unit="USD",
                period_start=start or "", period_end=end, period_class=period_class(start, end), form=form,
                accession=_accn_by_filing.setdefault((ticker, filed), f"0000000000-00-{next(_accn):06d}"),
                filed=filed, source_url="https://example.com")


# ------------------------------------------------------------------ made-up companies
# GOOGL (stand-in for Alphabet): calendar fiscal year. Each report is filed about a month after the quarter ends; the Q1 2026 10-Q
# is filed on 1 May 2026, the day AFTER a 30 Apr cut-off. Values in $m.
CAL_QUARTERS = [  # (start, end, filed, revenue, op. income, gross profit, net income, op. cash flow, capex)
    ("2024-01-01", "2024-03-31", "2024-05-01", 80.0, 40.0, 60.0, 20.0, 20, 10),
    ("2024-04-01", "2024-06-30", "2024-08-01", 85.0, 42.0, 64.0, 21.0, 20, 10),
    ("2024-07-01", "2024-09-30", "2024-11-01", 90.0, 45.0, 68.0, 22.0, 20, 10),
    ("2024-10-01", "2024-12-31", "2025-02-01", 95.0, 48.0, 72.0, 23.0, 20, 10),
    ("2025-01-01", "2025-03-31", "2025-05-01", 100.0, 62.0, 76.0, 25.0, 20, 10),
    ("2025-04-01", "2025-06-30", "2025-08-01", 105.0, 65.52, 80.0, 26.0, 20, 10),   # op. margin 62.4%
    ("2025-07-01", "2025-09-30", "2025-11-01", 110.0, 68.0, 84.0, 27.0, 20, 10),
    ("2025-10-01", "2025-12-31", "2026-02-01", 115.0, 71.0, 88.0, 28.0, 20, 10),
    ("2026-01-01", "2026-03-31", "2026-05-01", 121.8, 75.0, 93.0, 30.0, 25, 10),    # +21.8%; filed 1 May
    ("2026-04-01", "2026-06-30", "2026-07-23", 127.89, 77.25, 97.0, 31.2, 10, 15),  # +21.8%; op. margin 60.4%;
]                                                                                   # net income +20%; FCF -5
# Fiscal years (10-K): FY2024 revenue 350, FY2025 430 (+22.9%); FY2025 operating margin 266.52 / 430 = 62.0%
# Current assets / current liabilities ($m): 2.5 at FY2024, 2.0 at FY2025 and a year before the latest
# quarter, 1.5 at the latest quarter end
CAL_BALANCE = {"2024-12-31": (300, 120), "2025-06-30": (300, 150), "2025-09-30": (300, 150),
               "2025-12-31": (300, 150), "2026-03-31": (300, 180), "2026-06-30": (300, 200)}
# MSFT (stand-in for Microsoft): fiscal year ends 30 June (like Microsoft). Q1 of fiscal 2026 is the quarter ended 30 Sep 2025.
FIS_QUARTERS = [
    ("2024-07-01", "2024-09-30", "2024-10-30", 60.0), ("2024-10-01", "2024-12-31", "2025-01-30", 62.0),
    ("2025-01-01", "2025-03-31", "2025-04-30", 64.0), ("2025-04-01", "2025-06-30", "2025-07-30", 66.0),
    ("2025-07-01", "2025-09-30", "2025-10-30", 72.0), ("2025-10-01", "2025-12-31", "2026-01-30", 74.4),
    ("2026-01-01", "2026-03-31", "2026-04-30", 76.8),
]
_CAL_FLOWS = ("Revenue", "OperatingIncome", "GrossProfit", "NetIncome", "OperatingCashFlow", "CapitalExpenditure")


def make_store():
    rows = []
    for start, end, filed, *values in CAL_QUARTERS:
        for concept, value in zip(_CAL_FLOWS, values):
            rows.append(make_fact("GOOGL", concept, value, end, start, filed))
        rows.append(make_fact("GOOGL", "TotalAssets", 500, end, None, filed))
        if end in CAL_BALANCE:
            assets, liabilities = CAL_BALANCE[end]
            rows += [make_fact("GOOGL", "CurrentAssets", assets, end, None, filed),
                     make_fact("GOOGL", "CurrentLiabilities", liabilities, end, None, filed)]
    for year, filed in ((2024, "2025-02-01"), (2025, "2026-02-01")):        # the 10-K: full-year sums
        quarters = [q for q in CAL_QUARTERS if q[1].startswith(str(year))]
        for i, concept in enumerate(_CAL_FLOWS):
            rows.append(make_fact("GOOGL", concept, round(sum(q[3 + i] for q in quarters), 2),
                                  f"{year}-12-31", f"{year}-01-01", filed, "10-K"))
    for start, end, filed, rev in FIS_QUARTERS:
        rows.append(make_fact("MSFT", "Revenue", rev, end, start, filed))
    rows.append(make_fact("MSFT", "Revenue", 252, "2025-06-30", "2024-07-01", "2025-07-30", "10-K"))
    return FactStore(rows)


STORE = make_store()
AGENT = XbrlFinancialAgent(STORE)


def ask(text, ticker="GOOGL", kind="HISTORICAL_GROWTH", cutoff=None, prompt=None):
    """The agent's answer to one claim, given as the agreed sub-claim: id, ticker, text (+ optional prompt).
    The type is a stand-in with only a name, like the team's enum; the agent does not use it."""
    sub = SimpleNamespace(sub_claim_id="sc1", text=text, company_ticker=ticker,
                          sub_claim_type=SimpleNamespace(name=kind), user_prompt=prompt)
    return AGENT.gather(sub, cutoff, None)


def reason():
    return AGENT.reasons.get("sc1", "")


def period_of(item):
    """The period end in the evidence id (XBRL-<ticker>-<metric>-<YYYY-MM-DD>)."""
    return "-".join(item.evidence_id.split("-")[-3:])


def names(items):
    return [i.metric_name for i in items]


def measures(line, prompt=None, ticker="GOOGL"):
    return [p.measure for p in read_claim(line, ticker, prompt).points]


def close(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


# ================================================================== reading the claim (claim_reader.py)
def test_reads_every_measure_in_a_line():
    assert measures("Revenues are at all-time highs, operating profits at all-time highs, operating margins "
                    "are at 41%") == ["Revenue", "OperatingIncome", "operating_margin"]
    assert measures("Free cash flow fell while operating cash flow rose") == ["free_cash_flow", "OperatingCashFlow"]
    assert measures("Diluted EPS rose 25%") == ["DilutedEPS"]
    assert measures("the current ratio and the cash ratio both improved") == ["current_ratio", "cash_ratio"]


def test_growth_without_a_measure_word_is_revenue():
    assert measures("Google grew by 24% year-over-year") == ["Revenue"]
    assert measures("That was nearly 25% growth.") == []                 # no company, no prompt: nothing
    assert measures("Google's stock is up 25% this year") == []          # a share price


def test_valuation_and_event_words_are_not_measures():
    assert measures("Nvidia trades at a next 12 months earnings multiple of 20 times.") == []
    assert measures("Meta just reported earnings on July 29th.") == []
    assert measures("I booked a 300% return, taking profits late last year.") == []
    assert measures("Palantir has the highest price-to-sales ratio in the S&P 500.") == []


def test_another_companys_figures_are_skipped():
    assert measures("Last quarter Nebius generated $582 million in revenue, up 454%.", ticker="MSFT") == []
    assert measures("While Tesla's revenues fell, Microsoft has grown revenue by 15%.", ticker="TSLA") == ["Revenue"]
    assert measures("Alphabet's operating margins are 34% and AMD's are around 17%.") == ["operating_margin"]
    assert measures("Alphabet's free cash flow turned negative and Meta's free cash flow fell 91%.") \
        == ["free_cash_flow"]


def test_segments_and_non_calculated_measures_go_to_the_narrative_agent():
    r = read_claim("Google Cloud revenues rose to 20 billion on the quarter.", "GOOGL")
    assert r.points == [] and "segment" in r.notes[0]
    assert measures("AWS reported 24% growth with a $142 billion revenue run rate", ticker="AMZN") == []
    assert measures("Services margin sat at 40% last year.") == []
    assert measures("We also have subscription revenues slowing down to 22%.") == []
    r = read_claim("Tesla's revenues, gross profits, and earnings per share have all been in decline", "TSLA")
    assert [p.measure for p in r.points] == ["Revenue", "DilutedEPS"] and "gross profit" in r.notes[0]
    assert read_claim("Adjusted EBITDA rose 40%", "GOOGL").points == []
    r = read_claim("Net income fell once you exclude the investment gains", "GOOGL")
    assert r.points == [] and "removes gains" in r.notes[0]


def test_the_prompt_keeps_only_what_the_user_asked_about():
    line = "Compare it to Apple, which gives you less than 10% earnings growth and much lower margins."
    assert measures(line, "Mark Roussin said Apple has less than 10% earnings growth.", "AAPL") == ["NetIncome"]
    line = "You're seeing stronger commercial growth, AI platform demand, expanding profitability."
    assert measures(line, "Mark Roussin said Palantir's profitability is expanding.", "PLTR") \
        == ["operating_margin", "net_margin"]
    line = "On top of that, they have record-level margins."
    assert measures(line, "Alex Reffett said AMD's operating margin is around 17%.", "AMD") == ["operating_margin"]


def test_a_prompt_about_a_segment_keeps_nothing():
    r = read_claim("Up 6.7 billion, over 100% year-over-year growth.", "AMD",
                   "Mark Roussin said AMD's latest quarter showed over 100% data center growth.")
    assert r.points == [] and "segment" in r.notes[-1]


def test_a_line_without_the_asked_measure_keeps_nothing():
    r = read_claim("So Elon Musk owns 42% of the shares, but controls 85% of the votes.", "META",
                   "Alex Reffett said Meta's free cash flow is down 91%.")
    assert r.points == [] and "does not state the measure" in r.notes[-1]


def test_whole_company_phrase_expands_into_four_checks():
    line = "I still see Meta as having high operating leverage and a very strong financial position."
    assert measures(line, ticker="META") == ["operating_margin", "free_cash_flow", "current_ratio", "leverage"]


def test_period_order_prompt_then_words_then_field():
    line = "Alphabet's revenue growing 22% year-over-year to $110 in Q1."
    [p] = read_claim(line, "GOOGL", "Joseph Carlson said Alphabet's revenue grew 22% last quarter.").points
    assert p.period == ("latest", "quarter")                              # the prompt's period comes first
    [p] = read_claim(line, "GOOGL").points
    assert p.period == ("cal_q", None, 1)                                 # then the line's
    [p] = read_claim("Revenue grew 22%", "GOOGL", period_field="trailing 12-month").points
    assert p.period == ("ttm",)                                           # then Claim extraction's field
    [p] = read_claim("Revenue grew 22%", "GOOGL").points
    assert p.period is None                                               # none: the latest fiscal year


def test_a_level_stated_in_the_present_is_the_latest_quarter():
    [p] = read_claim("Alphabet's operating margins are 34%", "GOOGL").points
    assert p.period == ("latest", "quarter")
    [p] = read_claim("Operating margins expanded", "GOOGL").points
    assert p.period is None


def test_spoken_periods():
    assert named_period("22% this most recent quarter") == ("latest", "quarter")
    assert named_period("revenue grew 18% last quarter") == ("latest", "quarter")
    assert named_period("revenue recently grew 18%") == ("latest", "quarter")
    assert named_period("full year revenue grew 18%") == ("latest", "annual")
    assert named_period("grew revenue by about 15% in a year") == ("latest", "annual")
    assert named_period("revenue grew 18%") is None
    assert named_period("revenue grew 18% in Q4 of 2025") == ("cal_q", 2025, 4)
    assert named_period("revenue grew 18% in Q4'25") == ("cal_q", 2025, 4)
    assert named_period("revenue grew in the first quarter") == ("cal_q", None, 1)
    assert named_period("revenue over the trailing twelve months") == ("ttm",)
    assert named_period("revenue grew in the last two quarters") == ("last_n", 2)


# ================================================================== the figures (agent.py)
def test_item02_both_influencers_get_the_same_real_number():
    a = ask("The revenue growth of 20% for the company", prompt="Mark Roussin said Alphabet's revenue grew 20% "
            "last quarter, but Joseph Carlson said 22%.", cutoff=date(2026, 7, 31))
    b = ask("Google grew by 21%. In fact, 22% this most recent quarter.", cutoff=date(2026, 7, 31))
    assert len(a) == len(b) == 1 and a[0].computed_value == b[0].computed_value
    assert close(a[0].computed_value, 127.89 / 105 - 1)                   # +21.8%


def test_agent_never_takes_a_side():
    for text in ("Revenue grew 20% last quarter", "Revenue fell", "Margins are expanding",
                 "Free cash flow dipped into the red", "Net income grew 20%", "Liquidity is strong",
                 "Operating income shot up 124%", "Revenue grew double digits for four straight quarters"):
        items = ask(text, cutoff=date(2026, 7, 31))
        assert items and all(i.stance is Stance.NEUTRAL for i in items), text


def test_type_is_only_a_hint():
    for kind in ("OTHER", "PROFITABILITY_CASH_LEVERAGE", "CONTINUATION", "HISTORICAL_GROWTH"):
        [item] = ask("Revenue grew 22% last quarter", kind=kind)
        assert item.metric_name == "revenue_growth_yoy", kind


def test_growth_carries_the_amount_for_the_period():
    [item] = ask("Revenue grew 22% to $128 million last quarter")
    assert close(item.comparison_value, 127.89 * M) and item.comparison_period == "quarter ended 2026-06-30"


def test_level_carries_the_figure_a_year_earlier():
    [item] = ask("Operating margin was 60% last quarter")
    assert close(item.computed_value, 0.604, 5e-4) and close(item.comparison_value, 0.624)
    assert item.comparison_period == "quarter ended 2025-06-30"


def test_every_growth_series_in_the_yaml():
    for text, name, value in (("Net income grew 20% last quarter", "net_income_growth_yoy", 31.2 / 26 - 1),
                              ("Operating income rose last quarter", "operating_income_growth_yoy", 77.25 / 65.52 - 1),
                              ("Operating cash flow fell last quarter", "operating_cash_flow_growth_yoy", 10 / 20 - 1)):
        [item] = ask(text)
        assert item.metric_name == name and close(item.computed_value, value), text
    assert ask("Diluted EPS rose 25% last quarter") == [] and "Diluted EPS" in reason()    # no EPS in the test data


def test_no_period_named_uses_the_latest_fiscal_year_with_ttm_context_in_the_notes():
    [item] = ask("Revenue grew strongly")
    assert item.metric_name == "revenue_growth_yoy"
    assert period_of(item) == "2025-12-31" and close(item.computed_value, 430 / 350 - 1)
    # the twelve-month figure is context only: in the notes, not the evidence (team ruling, 18 Oct)
    assert "Context only (not evidence)" in reason() and "twelve months ended 2026-06-30" in reason()


def test_no_context_when_the_10k_is_the_latest_filing():
    [item] = ask("Revenue grew strongly", cutoff=date(2026, 2, 15))
    assert period_of(item) == "2025-12-31"


def test_trailing_twelve_months_is_the_main_figure_when_the_post_says_so():
    for text in ("Revenue grew 22% over the trailing 12 months", "TTM revenue grew 22%"):
        [item] = ask(text)
        assert item.metric_name == "revenue_growth_yoy" and item.period == "twelve months ended 2026-06-30", text
        assert close(item.computed_value, 474.69 / 390 - 1) and close(item.comparison_value, 474.69 * M)
    [item] = ask("Free cash flow went from 40 million to 30 million over the last 12 months")
    assert item.metric_name == "free_cash_flow" and item.computed_value == 30 * M
    assert item.comparison_value == 40 * M


def test_last_two_quarters_each_against_a_year_earlier():
    items = ask("Revenue grew about 22% in the last two quarters")
    assert [period_of(i) for i in items] == ["2026-06-30", "2026-03-31"]


def test_prompt_period_wins_over_the_line():
    # item 12: the post says Q1, the user says "last quarter"; with no cut-off that is the June quarter
    [item] = ask("Alphabet's revenue growing 22% year-over-year to $110 in Q1.",
                 prompt="Joseph Carlson said Alphabet's revenue grew 22% to $110 billion last quarter.")
    assert period_of(item) == "2026-06-30"
    [item] = ask("Alphabet's revenue growing 22% year-over-year to $110 in Q1.")
    assert period_of(item) == "2026-03-31"


def test_profitability_margins():
    assert names(ask("Gross margins expanded last quarter")) == ["gross_margin"]
    assert names(ask("Net margin rose last quarter")) == ["net_margin"]
    items = ask("The company is very profitable")
    assert names(items) == ["operating_margin", "net_margin"]
    assert close(items[0].computed_value, 266.52 / 430) and close(items[0].comparison_value, 175 / 350)


def test_item10_generic_margins_report_operating_then_gross():
    items = ask("Alphabet has very high margins this quarter.")
    assert names(items) == ["operating_margin", "gross_margin"]
    assert "60.4%" in items[0].content and "62.4%" in items[0].content


def test_present_tense_margin_is_the_latest_quarter():
    [item] = ask("For context, Alphabet's operating margins are 34% and AMD's are around 17%.")
    assert period_of(item) == "2026-06-30" and close(item.computed_value, 0.604, 5e-4)


def test_free_cash_flow():
    [item] = ask("The company's free cash flow dipped into the red last quarter")
    assert item.metric_name == "free_cash_flow" and item.computed_value == -5 * M      # 10 - 15
    assert item.comparison_value == 10 * M
    [item] = ask("Cash flow is strong")
    assert item.metric_name == "free_cash_flow" and item.computed_value == 40 * M


def test_free_cash_flow_growth_rate_when_the_words_state_one():
    items = ask("Free cash flow is down 91% last quarter")
    assert names(items) == ["free_cash_flow", "free_cash_flow_growth_yoy"]
    assert close(items[1].computed_value, -5 / 10 - 1)                   # -5m vs 10m a year earlier: -150%


def test_liquidity():
    [item] = ask("Liquidity is strong")                                  # balance sheet: no 12-month context
    assert item.metric_name == "current_ratio" and close(item.computed_value, 2.0)
    assert close(item.comparison_value, 2.5) and item.period == "as at 2025-12-31"
    [item] = ask("The current ratio fell last quarter")
    assert close(item.computed_value, 1.5) and close(item.comparison_value, 2.0)


def test_each_part_of_an_expanded_phrase_names_its_measure():
    # "financially strong" arrives as four claims, each naming its measure (agreed sub-claim rule 2)
    assert names(ask("The company's current ratio is strong")) == ["current_ratio"]
    assert names(ask("The company's free cash flow is strong")) == ["free_cash_flow"]
    assert names(ask("The company's operating margin is strong")) == ["operating_margin"]
    assert ask("The company's debt to equity is healthy") == [] and LEVERAGE_REASON in reason()


def test_whole_company_phrase_gives_three_figures_and_a_leverage_note():
    items = ask("The company has a very strong financial position")
    assert names(items) == ["operating_margin", "free_cash_flow", "current_ratio"]
    assert LEVERAGE_REASON in reason()


def test_segment_claim_gets_nothing():
    assert ask("Google Cloud revenue grew 28% in Q1 2026.") == [] and "segment" in reason()
    assert ask("AWS sales grew 28% in Q1 2026", ticker="GOOGL") == []


def test_debt_gets_nothing_with_a_reason():
    assert ask("Debt to equity is low") == [] and LEVERAGE_REASON in reason()
    assert ask("Its balance sheet carries a lot of debt") == [] and "debt amount" in reason()


def test_line_without_a_figure_gets_nothing():
    assert ask("Starbucks is using artificial intelligence to reduce its reliance on IBM software.") == []
    assert reason()


def test_history_and_a_reported_figure():
    items = ask("Their free cash flow turned negative last quarter for the first time since Google went public")
    assert names(items) == ["free_cash_flow"] and "history longer than the window" in reason()


def test_company_name_comes_from_the_thesaurus():
    assert company_display_name("GOOGL") == "Alphabet"
    assert company_display_name("XYZ") == "XYZ"                          # not in the thesaurus: the ticker


def test_q4_of_year_is_used():
    [item] = ask("Revenue grew 21% in Q4 of 2025")
    assert period_of(item) == "2025-12-31" and close(item.computed_value, 115 / 95 - 1)


# ------------------------------------------------------------------ newest quarter announced, not filed
def test_latest_quarter_unclear_between_announcement_and_filing():
    # Q1 2026 ended 31 Mar; it files about 31 days after a quarter ends, so by 30 Apr results are likely out
    assert ask("Revenue grew 22% last quarter", cutoff=date(2026, 4, 30)) == []
    assert "2026-03-31" in reason() and "could mean either quarter" in reason()
    [item] = ask("Revenue grew strongly", cutoff=date(2026, 4, 30))     # fiscal year: not affected
    assert period_of(item) == "2025-12-31"


def test_latest_filed_quarter_used_well_before_the_next_filing():
    [item] = ask("Revenue grew 21% last quarter", cutoff=date(2026, 4, 10))
    assert period_of(item) == "2025-12-31"
    [item] = ask("Revenue grew 22% last quarter", cutoff=date(2026, 5, 1))
    assert period_of(item) == "2026-03-31"


def test_quarter_filed_after_cutoff_gets_no_evidence():
    assert ask("Revenue grew 21.8% in Q1 2026", cutoff=date(2026, 4, 30)) == []
    [item] = ask("Revenue grew 21.8% in Q1 2026", cutoff=date(2026, 5, 1))
    assert period_of(item) == "2026-03-31"


# ------------------------------------------------------------------ window = latest 10-K + four quarters
def test_named_quarter_outside_the_window_gets_nothing():
    assert ask("Revenue grew 25% in Q1 2025") == [] and "outside the window" in reason()
    [item] = ask("Revenue grew 18% in Q3 2025")
    assert period_of(item) == "2025-09-30"


def test_named_year_outside_the_window_gets_nothing():
    assert ask("Revenue grew in fiscal 2024") == [] and "outside the window" in reason()
    [item] = ask("Revenue grew 23% in fiscal 2025")
    assert period_of(item) == "2025-12-31" and close(item.computed_value, 430 / 350 - 1)


def test_item01_streak_longer_than_the_window():
    items = ask("Google grew by 24% year-over-year, marking its twelfth consecutive quarter of double-digit growth")
    assert names(items) == ["revenue_growth_yoy"] and "12 quarters" in reason()
    assert ask("Google's revenue had its twelfth straight quarter of double-digit growth") == []


def test_streak_within_the_window_is_one_growth_item_per_quarter():
    items = ask("Revenue grew double digits for four straight quarters")
    assert names(items) == ["revenue_growth_yoy"] * 4
    assert [period_of(i) for i in items] == ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"]
    assert close(items[2].computed_value, 115 / 95 - 1)                  # the Auditor checks each against 10%
    assert "4 of 4 quarters reported" in reason() and "threshold 10%" in reason()


def test_item05_month_quarter_is_used_instead_of_latest():
    [item] = ask("Revenues of 82.9 billion, which was an 18.3% year-over-year growth in the March quarter")
    assert period_of(item) == "2026-03-31"


def test_item07_quarter_not_ended_by_cutoff_gets_no_evidence():
    assert ask("Google's revenue grew 24% in the June 2026 quarter.", cutoff=date(2026, 4, 1)) == []
    assert "had not ended" in reason()


def test_fiscal_quarter_uses_the_company_fiscal_calendar():
    [item] = ask("Revenue grew 20% in Q1 FY2026", ticker="MSFT")
    assert period_of(item) == "2025-09-30" and close(item.computed_value, 0.20)   # 72 / 60 - 1


def test_company_not_in_data_gets_nothing():
    assert ask("Revenue grew 10%", ticker="NOPE") == [] and "not in the SEC data" in reason()


# ------------------------------------------------------------------ output contract (agreed with the Auditor)
def test_growth_content_shape():
    [item] = ask("Revenue grew 22% last quarter")
    assert item.content == ("Revenue was $128m in the quarter ended 2026-06-30 vs $105m in the quarter ended "
                            "2025-06-30 (+21.8% YoY).")
    assert close(item.computed_value, 127.89 / 105 - 1) and item.metric_name == "revenue_growth_yoy"


def test_margin_content_shape():
    [item] = ask("Operating margin was 60% last quarter")
    assert item.content == ("Operating margin was 60.4% in the quarter ended 2026-06-30 vs 62.4% in the quarter "
                            "ended 2025-06-30.")


def test_ratio_content_shape():
    [item] = ask("The current ratio fell last quarter")
    assert item.content == "Current ratio was 1.50x as at 2026-06-30 vs 2.00x as at 2025-06-30."


def test_free_cash_flow_growth_content_shape():
    items = ask("Free cash flow is down 91% last quarter")
    assert items[1].content == ("Free cash flow was -$5m in the quarter ended 2026-06-30 vs $10m in the quarter "
                                "ended 2025-06-30 (-150.0% YoY).")


def test_evidence_fields_for_the_auditor():
    [item] = ask("Revenue grew 22% last quarter")
    assert item.stance is Stance.NEUTRAL and item.doc_type == "10-Q" and item.company == "GOOGL"
    assert item.source_tier.name == "TIER_1" and str(item.publication_date) == "2026-07-23"
    assert item.url and item.citation_span.startswith("XBRL us-gaap:Revenues; filings: 10-Q")
    assert item.period == "quarter ended 2026-06-30" and item.sub_claim_id == "sc1"


def test_amendments_are_reported_as_their_base_form():
    from finsight.agents.financial.agent import _doc_type
    assert _doc_type("10-Q/A") == "10-Q" and _doc_type("10-K/A") == "10-K" and _doc_type("10-K") == "10-K"


def test_no_context_streak_or_ttm_metric_names_reach_the_auditor():
    allowed = {"revenue_growth_yoy", "operating_income_growth_yoy", "net_income_growth_yoy",
               "operating_cash_flow_growth_yoy", "diluted_eps_growth_yoy", "free_cash_flow_growth_yoy",
               "gross_margin", "operating_margin", "net_margin", "free_cash_flow", "fcf_margin", "current_ratio",
               "cash_ratio"}
    for text in ("Revenue grew strongly", "TTM revenue grew 22%", "Revenue grew double digits for four straight "
                 "quarters", "Margins expanded", "Free cash flow is down 91%", "Liquidity is strong",
                 "The company has a very strong financial position"):
        assert set(names(ask(text))) <= allowed, text


def test_forecasts_get_nothing():
    assert ask("Earnings are set to grow 75% next year") == [] and "forecast" in reason()
    [item] = ask("Revenue grew 22% last quarter and earnings are set to grow 75% next year")
    assert item.metric_name == "revenue_growth_yoy" and "forecast" in reason()


# ------------------------------------------------------------------ runner without pytest
if __name__ == "__main__":
    import sys
    import traceback
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and callable(f)]
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
