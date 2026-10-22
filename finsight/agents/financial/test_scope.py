"""
Checks that the agent reads the text to decide what to compute, within the scope of each sub-claim.

Sub-claims are built the way the pipeline builds them (claim_extraction.build + extract._with_growth), so the
same sentence can arrive several times with a different type / measure. The model is scripted with
set_mock_responder; the data is a small synthetic fact store, so no SEC pull is needed.

Run from the repo root:  python -m finsight.agents.financial.test_scope   (or pytest on this file)
"""

from __future__ import annotations

import json
import os
from datetime import date

from finsight.agents.claim_extraction.build import claims_from_transcript
from finsight.contracts.schemas import SubClaim
from finsight.llm.client import set_mock_responder

from .agent import XbrlFinancialAgent
from .store import FactStore

CUTOFF = date(2026, 6, 1)


# ------------------------------------------------------------------ synthetic data
def _store() -> FactStore:
    """NVDA, calendar quarters 2024-Q1 to 2026-Q1, fiscal year = calendar year. Values in USD."""
    rows, n = [], 0

    def add(concept, tag, value, start, end, cls, form, filed, unit="USD"):
        nonlocal n
        n += 1
        rows.append(dict(ticker="NVDA", cik=1045810, concept=concept, xbrl_tag=tag, value=value, unit=unit,
                         period_start=start, period_end=end, period_class=cls, form=form,
                         accession=f"0000000000-26-{n:06d}", filed=filed, source_url="https://www.sec.gov/x"))

    quarters = [("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-06-30"), ("2024-07-01", "2024-09-30"),
                ("2024-10-01", "2024-12-31"), ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
                ("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"), ("2026-01-01", "2026-03-31")]
    for i, (start, end) in enumerate(quarters):
        rev = 20e9 * (1.1 ** i)
        year_end = end.endswith("12-31")
        form = "10-K" if year_end else "10-Q"
        filed = date.fromordinal(date.fromisoformat(end).toordinal() + (50 if year_end else 30)).isoformat()
        for concept, tag, v in [("Revenue", "Revenues", rev), ("OperatingIncome", "OperatingIncomeLoss", rev * 0.6),
                                ("NetIncome", "NetIncomeLoss", rev * 0.5), ("GrossProfit", "GrossProfit", rev * 0.75),
                                ("OperatingCashFlow", "NetCashProvidedByUsedInOperatingActivities", rev * 0.55),
                                ("CapitalExpenditure", "PaymentsToAcquirePropertyPlantAndEquipment", rev * 0.05)]:
            add(concept, tag, v, start, end, "Q", form, filed)
        add("CurrentAssets", "AssetsCurrent", 80e9 + i * 5e9, "", end, "I", form, filed)
        add("CurrentLiabilities", "LiabilitiesCurrent", 20e9, "", end, "I", form, filed)
        add("CashAndEquivalents", "CashAndCashEquivalentsAtCarryingValue", 10e9, "", end, "I", form, filed)
        if year_end:
            year = [q for q in quarters if q[1][:4] == end[:4]]
            idx = [quarters.index(q) for q in year]
            fy = sum(20e9 * (1.1 ** j) for j in idx)
            for concept, tag, k in [("Revenue", "Revenues", 1), ("OperatingIncome", "OperatingIncomeLoss", 0.6),
                                    ("NetIncome", "NetIncomeLoss", 0.5), ("GrossProfit", "GrossProfit", 0.75),
                                    ("OperatingCashFlow", "NetCashProvidedByUsedInOperatingActivities", 0.55),
                                    ("CapitalExpenditure", "PaymentsToAcquirePropertyPlantAndEquipment", 0.05)]:
                add(concept, tag, fy * k, year[0][0], end, "FY", "10-K", filed)
    return FactStore(rows)


def _sub_claims(sentence: str) -> list[SubClaim]:
    """The sub-claims the pipeline builds for one spoken sentence (stored row -> SubClaim, plus growth children)."""
    from finsight.agents.claim_extraction.extract import _with_growth
    out = []
    for r in claims_from_transcript(sentence, video_id="vid", influencer="x"):
        out += _with_growth(SubClaim(sub_claim_id=r["sub_claim_id"], text=r["text"],
                                     sub_claim_type=r["sub_claim_type"], company_ticker=r["company_ticker"],
                                     measure=r.get("measure"), parent_span=r.get("parent_span"),
                                     expansion_rule=r.get("expansion_rule")))
    return out


def _metrics(items) -> list[str]:
    return sorted(i.metric_name for i in items)


# ------------------------------------------------------------------ a scripted model
class ScriptedModel:
    """Replies with the given actions in order, one per call; records every prompt it was shown."""

    def __init__(self, *actions: dict):
        self.actions, self.prompts = list(actions), []

    def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.actions.pop(0) if self.actions else {"thought": "done", "action": "finish", "args": {}})


def _run(model: ScriptedModel, sub: SubClaim):
    os.environ.pop("FINSIGHT_FINANCIAL_PLANNER", None)
    set_mock_responder(model)
    try:
        agent = XbrlFinancialAgent(_store())
        return agent, agent.gather(sub, CUTOFF, None)
    finally:
        set_mock_responder(None)


# ------------------------------------------------------------------ tests
def test_model_reads_paraphrase_the_rules_miss():
    """'the top line jumped 21%' has no measure word the rules know; the model reads it as revenue growth."""
    sub = SubClaim(sub_claim_id="p1", text="Nvidia's top line jumped 21% last quarter.",
                   sub_claim_type="HISTORICAL_GROWTH", company_ticker="NVDA")
    model = ScriptedModel(
        {"thought": "'top line' is revenue; 'jumped 21%' is growth; 'last quarter' is the latest quarter",
         "action": "understand", "args": {"parts": [{"words": "top line jumped 21% last quarter",
                                                     "measure": "Revenue", "period": {"type": "latest_quarter"},
                                                     "stated": "21%", "direction": "up", "in_scope": True}]}},
        {"thought": "compute it", "action": "report_growth",
         "args": {"measure": "Revenue", "period": {"type": "latest_quarter"}}})
    agent, items = _run(model, sub)
    assert agent.modes["p1"] == "llm"
    assert _metrics(items) == ["revenue_growth_yoy"], _metrics(items)
    assert agent.readings["p1"][0]["stated"] == "21%"


def test_first_action_must_be_understand():
    sub = SubClaim(sub_claim_id="u1", text="Nvidia's revenue grew 21% last quarter.",
                   sub_claim_type="HISTORICAL_GROWTH", company_ticker="NVDA")
    model = ScriptedModel(
        {"thought": "skip ahead", "action": "report_growth",
         "args": {"measure": "Revenue", "period": {"type": "latest_quarter"}}},
        {"thought": "read first", "action": "understand",
         "args": {"parts": [{"words": "revenue grew 21%", "measure": "Revenue", "in_scope": True}]}},
        {"thought": "now compute", "action": "report_growth",
         "args": {"measure": "Revenue", "period": {"type": "latest_quarter"}}})
    agent, items = _run(model, sub)
    assert "REFUSED: your first action must be `understand`" in agent.traces["u1"]
    assert _metrics(items) == ["revenue_growth_yoy"]


def test_rule_hint_only_after_understand():
    """The model reads the words on its own first; the rule-based reading is shown only as a cross-check."""
    sub = SubClaim(sub_claim_id="h1", text="Nvidia's revenue grew 21% last quarter.",
                   sub_claim_type="HISTORICAL_GROWTH", company_ticker="NVDA")
    model = ScriptedModel({"thought": "read", "action": "understand",
                           "args": {"parts": [{"words": "revenue grew 21%", "measure": "Revenue", "in_scope": True}]}})
    _run(model, sub)
    assert "rule-based reading" not in model.prompts[0]
    assert "rule-based reading" in model.prompts[1]


def test_expansion_children_each_report_their_own_measure():
    """'financially strong' arrives as four sub-claims with the same text; each reports only its measure."""
    subs = _sub_claims("Nvidia is financially strong.")
    assert [s.measure for s in subs] == ["profitability", "cash_generation", "liquidity", "leverage"]
    os.environ["FINSIGHT_FINANCIAL_PLANNER"] = "rules"
    try:
        agent = XbrlFinancialAgent(_store())
        got = {s.measure: _metrics(agent.gather(s, CUTOFF, None)) for s in subs}
        reasons = {s.measure: agent.reasons.get(s.sub_claim_id, "") for s in subs}
    finally:
        os.environ.pop("FINSIGHT_FINANCIAL_PLANNER", None)
    assert got["profitability"] == ["operating_margin"], got
    assert got["cash_generation"] == ["free_cash_flow"], got
    assert got["liquidity"] == ["current_ratio"], got
    assert got["leverage"] == [] and "Debt is not mapped" in reasons["leverage"], (got, reasons)


def test_model_refused_outside_scope():
    """The LEVERAGE child: the model may read the whole phrase, but computing a margin there is refused."""
    lev = [s for s in _sub_claims("Nvidia is financially strong.") if s.measure == "leverage"][0]
    model = ScriptedModel(
        {"thought": "a company-wide phrase; this child is leverage", "action": "understand",
         "args": {"parts": [{"words": "financially strong", "measure": "leverage", "in_scope": True},
                            {"words": "financially strong", "measure": "operating_margin", "in_scope": False}]}},
        {"thought": "try a margin anyway", "action": "report_level",
         "args": {"measure": "operating_margin", "period": None}},
        {"thought": "leverage", "action": "report_level", "args": {"measure": "leverage", "period": None}})
    agent, items = _run(model, lev)
    assert items == []
    assert "REFUSED: operating_margin is outside this sub-claim's scope" in agent.traces[lev.sub_claim_id]
    assert "Debt is not mapped" in agent.reasons[lev.sub_claim_id]
    assert "Scope of this sub-claim: LEVERAGE / leverage" in model.prompts[0]


def test_split_sentence_margin_and_growth():
    """A margin sentence that names operating-income growth: the growth child (measure set) reports only its
    figure; the parent (type only) is not limited by its type, so it reports both (rules path)."""
    subs = _sub_claims("Nvidia's operating margin expanded to 60% and operating income grew 21% year over year.")
    # NOTE (upstream): build.py stores the growth row AND extract._with_growth adds it again at load time, so the
    # pipeline currently sends the HISTORICAL_GROWTH / operating_income sub-claim twice. Checked once here.
    subs = list({(s.sub_claim_type.value, s.measure): s for s in subs}.values())
    assert [(s.sub_claim_type.value, s.measure) for s in subs] == [("PROFITABILITY", None),
                                                                   ("HISTORICAL_GROWTH", "operating_income")]
    os.environ["FINSIGHT_FINANCIAL_PLANNER"] = "rules"
    try:
        agent = XbrlFinancialAgent(_store())
        got = [_metrics(agent.gather(s, CUTOFF, None)) for s in subs]
    finally:
        os.environ.pop("FINSIGHT_FINANCIAL_PLANNER", None)
    assert got == [["operating_income_growth_yoy", "operating_margin"], ["operating_income_growth_yoy"]], got


def _rules(sub):
    os.environ["FINSIGHT_FINANCIAL_PLANNER"] = "rules"
    try:
        return _metrics(XbrlFinancialAgent(_store()).gather(sub, CUTOFF, None))
    finally:
        os.environ.pop("FINSIGHT_FINANCIAL_PLANNER", None)


def test_type_does_not_drop_a_stated_measure():
    """Real stored lines whose loaded type does not match the measure the eval checks (claims_dev_final 26, 34):
    the type must not filter the measure out."""
    eps = SubClaim(sub_claim_id="r26", sub_claim_type="PROFITABILITY", company_ticker="NVDA",
                   text="Compare it to a company like Nvidia per se, which gives you less than 10% earnings growth.")
    assert "net_income_growth_yoy" in _rules(eps)
    op = SubClaim(sub_claim_id="r34", sub_claim_type="HISTORICAL_GROWTH", company_ticker="NVDA",
                  text="As a result, Nvidia's operating income shot up 124% giving them a whopping 66.2% operating "
                       "margin.")
    assert {"operating_income_growth_yoy", "operating_margin"} <= set(_rules(op)), _rules(op)


def test_claim_extraction_period_is_used_when_the_words_name_none():
    """claims_updated.jsonl stores fields.period ("trailing 12-month"); the agent uses it as the last resort."""
    from types import SimpleNamespace
    sub = SimpleNamespace(sub_claim_id="f1", text="Nvidia's revenue grew 21%.", company_ticker="NVDA",
                          sub_claim_type=SimpleNamespace(name="HISTORICAL_GROWTH"), measure=None,
                          fields={"period": "trailing 12-month"})
    os.environ["FINSIGHT_FINANCIAL_PLANNER"] = "rules"
    try:
        [item] = XbrlFinancialAgent(_store()).gather(sub, CUTOFF, None)
    finally:
        os.environ.pop("FINSIGHT_FINANCIAL_PLANNER", None)
    assert item.period.startswith("twelve months ended"), item.period


def test_unscoped_sub_claim_reports_everything():
    """A hand-built OTHER sub-claim (e.g. from run_eval) is not limited."""
    sub = SubClaim(sub_claim_id="o1", text="Nvidia's revenue grew 21% and net income grew 21% last quarter.",
                   sub_claim_type="OTHER", company_ticker="NVDA")
    os.environ["FINSIGHT_FINANCIAL_PLANNER"] = "rules"
    try:
        items = XbrlFinancialAgent(_store()).gather(sub, CUTOFF, None)
    finally:
        os.environ.pop("FINSIGHT_FINANCIAL_PLANNER", None)
    assert _metrics(items) == ["net_income_growth_yoy", "revenue_growth_yoy"], _metrics(items)


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:                                      # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} pass")
