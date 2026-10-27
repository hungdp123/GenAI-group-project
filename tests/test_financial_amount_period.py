"""A stated dollar amount picks the period when the line names none (NVDA, Mark Roussin, dev claim #27)."""

from datetime import date

from finsight.agents.auditor.agent import AuditorAgent
from finsight.agents.financial.agent import XbrlFinancialAgent
from finsight.agents.financial.store import FactStore
from finsight.config import CORPUS_COMPANIES
from finsight.contracts.schemas import SubClaim, SubClaimType, Verdict


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


def _gather(text):
    agent = XbrlFinancialAgent(FactStore(FACTS, FILINGS))
    sc = SubClaim(sub_claim_id="s", text=text, sub_claim_type=SubClaimType.HISTORICAL_GROWTH,
                  company_ticker="NVDA", is_quantitative=True)
    return sc, agent.gather(sc, CUTOFF, None), agent


def test_quarterly_amount_uses_the_latest_quarter(monkeypatch):
    monkeypatch.setitem(CORPUS_COMPANIES, "NVDA", ["Nvidia", "NVIDIA", "NVDA"])
    sc, items, agent = _gather("Nvidia reported revenues of 96.2 billion, up 106% year-over-year.")
    assert items and "quarter ended 2026-07-26" in items[0].content
    assert abs(items[0].computed_value - 1.0585) < 1e-3
    assert "latest quarter's figure" in agent.reasons.get("s", "")
    assert AuditorAgent(use_llm=False).audit(sc, items).verdict is Verdict.SUPPORTED


def test_annual_amount_or_no_amount_keeps_the_fiscal_year():
    for text in ("Nvidia reported revenues of 216 billion, up 65% year-over-year.",
                 "Nvidia revenue grew 65% year-over-year."):
        _sc, items, _agent = _gather(text)
        assert items and "year ended 2026-01-25" in items[0].content
