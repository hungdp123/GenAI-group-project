"""Financial agent: the rule-based reading decides what to calculate first; the LLM is only the fallback."""

import json
from datetime import date

import pytest

from finsight.agents.financial.agent import XbrlFinancialAgent
from finsight.agents.financial.store import FactStore
from finsight.config import CORPUS_COMPANIES
from finsight.contracts.schemas import SubClaim, SubClaimType
from finsight.llm.client import set_mock_responder


def _fact(end, start, cls, value, filed, form, acc):
    return dict(ticker="NVDA", concept="Revenue", xbrl_tag="Revenues", value=str(value), cik="1045810",
                period_start=start, period_end=end, period_class=cls, filed=filed, accession=acc, form=form,
                source_url="https://www.sec.gov/x")


FACTS = [
    _fact("2025-01-26", "2024-01-29", "FY", 130_497e6, "2025-02-26", "10-K", "a1"),
    _fact("2026-01-25", "2025-01-27", "FY", 215_938e6, "2026-02-25", "10-K", "a2"),
    _fact("2025-01-26", "2024-01-29", "FY", 130_497e6, "2026-02-25", "10-K", "a2"),
    _fact("2025-04-27", "2025-01-27", "Q", 44_062e6, "2025-05-28", "10-Q", "a3"),
    _fact("2025-07-27", "2025-04-28", "Q", 46_743e6, "2025-08-27", "10-Q", "a4"),
    _fact("2025-10-26", "2025-07-28", "Q", 57_006e6, "2025-11-19", "10-Q", "a5"),
    _fact("2026-04-26", "2026-01-26", "Q", 70_000e6, "2026-05-27", "10-Q", "a6"),
    _fact("2026-07-26", "2026-04-27", "Q", 96_221e6, "2026-08-26", "10-Q", "a7"),
    _fact("2025-07-27", "2025-04-28", "Q", 46_743e6, "2026-08-26", "10-Q", "a7"),
]
FILINGS = [dict(ticker="NVDA", accession=r["accession"], form=r["form"], filed=r["filed"],
                period_end=r["period_end"], cik="1045810") for r in FACTS]
CUTOFF = date(2026, 9, 1)


@pytest.fixture
def model(monkeypatch):
    """A scripted model that reads 'brought in more money' as revenue for the latest quarter, and counts its calls."""
    monkeypatch.setitem(CORPUS_COMPANIES, "NVDA", ["Nvidia", "NVIDIA", "NVDA"])
    calls = []
    script = [
        {"thought": "money brought in is revenue", "action": "understand", "args": {"parts": [
            {"words": "brought in 106% more money", "measure": "Revenue", "period": {"type": "latest_quarter"},
             "stated": "106%", "direction": "up", "in_scope": True, "why": "money brought in = revenue"}]}},
        {"thought": "compute it", "action": "report_growth",
         "args": {"measure": "Revenue", "period": {"type": "latest_quarter"}}},
        {"thought": "done", "action": "finish", "args": {}},
    ]

    def responder(system, prompt):
        calls.append(prompt)
        return json.dumps(script[min(len(calls) - 1, len(script) - 1)])

    set_mock_responder(responder)
    yield calls
    set_mock_responder(None)


def _sub(text, measure=None, kind=SubClaimType.HISTORICAL_GROWTH):
    return SubClaim(sub_claim_id="s", text=text, sub_claim_type=kind, company_ticker="NVDA",
                    is_quantitative=True, measure=measure)


def _agent():
    return XbrlFinancialAgent(FactStore(FACTS, FILINGS))


def test_rules_find_the_measure_so_no_model_call(model, monkeypatch):
    monkeypatch.delenv("FINSIGHT_FINANCIAL_PLANNER", raising=False)
    agent = _agent()
    items = agent.gather(_sub("Nvidia reported revenues of 96.2 billion, up 106% year-over-year."), CUTOFF, None)
    assert items and items[0].metric_name == "revenue_growth_yoy"
    assert agent.modes["s"] == "rules"
    assert agent.traces["s"].startswith("Rule-based reading: Revenue")
    assert model == []                                    # the model was never asked


def test_rules_find_nothing_so_the_model_reads_it(model, monkeypatch):
    monkeypatch.delenv("FINSIGHT_FINANCIAL_PLANNER", raising=False)
    agent = _agent()
    items = agent.gather(_sub("Nvidia brought in 106% more money than a year ago."), CUTOFF, None)
    assert model, "the model should be the fallback when the rules find no measure"
    assert agent.modes["s"] == "llm"
    assert "the rules found no measure" in agent.traces["s"]
    assert items and "quarter ended 2026-07-26" in items[0].content


def test_retry_with_feedback_goes_to_the_model(model, monkeypatch):
    monkeypatch.delenv("FINSIGHT_FINANCIAL_PLANNER", raising=False)
    agent = _agent()
    agent.gather(_sub("Nvidia revenue grew 106%."), CUTOFF, "period is the year but the claim means the quarter")
    assert model and agent.modes["s"] == "llm"
    assert "retry" in agent.traces["s"]

    model.clear()
    monkeypatch.setenv("FINSIGHT_FINANCIAL_LLM_ON_RETRY", "0")
    agent.gather(_sub("Nvidia revenue grew 106%."), CUTOFF, "same feedback")
    assert model == [] and agent.modes["s"] == "rules"


def test_model_unavailable_keeps_the_rules_result(monkeypatch):
    monkeypatch.delenv("FINSIGHT_FINANCIAL_PLANNER", raising=False)
    monkeypatch.setitem(CORPUS_COMPANIES, "NVDA", ["Nvidia", "NVIDIA", "NVDA"])
    agent = _agent()                                      # default mock: never answers in the action format
    items = agent.gather(_sub("Nvidia brought in 106% more money than a year ago."), CUTOFF, None)
    assert items == [] and agent.modes["s"] == "rules"
    assert "not used" in agent.traces["s"]


def test_rules_mode_never_calls_the_model(model, monkeypatch):
    monkeypatch.setenv("FINSIGHT_FINANCIAL_PLANNER", "rules")
    agent = _agent()
    agent.gather(_sub("Nvidia brought in 106% more money than a year ago."), CUTOFF, None)
    assert model == [] and agent.modes["s"] == "rules"


def test_llm_mode_keeps_the_old_model_first_order(model, monkeypatch):
    monkeypatch.setenv("FINSIGHT_FINANCIAL_PLANNER", "llm")
    agent = _agent()
    agent.gather(_sub("Nvidia revenue grew 106%."), CUTOFF, None)
    assert model and agent.modes["s"] == "llm"


@pytest.mark.parametrize("measure", ["segment", "guidance", "gross_profit"])
def test_segment_or_guidance_measure_computes_nothing(model, measure):
    agent = _agent()
    items = agent.gather(_sub("Nvidia data center revenue grew 106%.", measure=measure), CUTOFF, None)
    assert items == [] and model == [] and agent.modes["s"] == "rules"
    assert agent.reasons["s"]


def test_single_measure_split_row_reports_only_its_measure(model):
    agent = _agent()
    sub = _sub("Revenue grew 106% and net income doubled", measure="revenue")
    items = agent.gather(sub, CUTOFF, None)
    assert {i.metric_name for i in items} == {"revenue_growth_yoy"}
    assert "outside this sub-claim's scope" in agent.reasons["s"]


def test_rules_that_explain_a_skip_do_not_call_the_model(model):
    agent = _agent()
    items = agent.gather(_sub("Nvidia data center revenue grew 106%."), CUTOFF, None)
    assert items == [] and model == [] and agent.modes["s"] == "rules"
    assert "segment" in agent.reasons["s"]
