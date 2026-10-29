"""Period clues: the sentences before a line and the user's own words help pick the period, but the line is
always what is checked, and a mismatch with the user's prompt is reported (schema 0.9)."""

import json
from datetime import date

from finsight import execute
from finsight.agents.financial.agent import XbrlFinancialAgent
from finsight.agents.financial.store import FactStore
from finsight.agents.financial.stub import StubFinancialAgent
from finsight.agents.orchestrator import clues
from finsight.contracts.schemas import SubClaim, SubClaimType

from stored_demo import demo_registry
from test_financial_amount_period import CUTOFF, FACTS, FILINGS

TRANSCRIPT = ("Company X just reported its second quarter. The quarter was strong. "
              "Company X revenue grew 22% year over year. Margins held up.")
LINE = "Company X revenue grew 22% year over year."


def _sub(text, context=None, user_text=None, ticker="NVDA"):
    return SubClaim(sub_claim_id="s", text=text, sub_claim_type=SubClaimType.HISTORICAL_GROWTH,
                    company_ticker=ticker, context=context, user_text=user_text)


# -- the order the financial agent uses -----------------------------------------------------------


def _period(monkeypatch, sub_claim):
    monkeypatch.setenv("FINSIGHT_FINANCIAL_PLANNER", "rules")
    items = XbrlFinancialAgent(FactStore(FACTS, FILINGS)).gather(sub_claim, CUTOFF, None)
    assert items, "no evidence"
    return items[0].period, items[0].period_source


def test_preceding_sentence_sets_the_period(monkeypatch):
    period, source = _period(monkeypatch, _sub("Nvidia revenue was up 106% year-over-year.",
                                               context="Nvidia just reported its second quarter results."))
    assert "2026-07-26" in period and source == "the sentences before the line"


def test_user_prompt_fills_the_gap_only(monkeypatch):
    period, source = _period(monkeypatch, _sub("Nvidia revenue was up 106% year-over-year.",
                                               user_text="Nvidia revenue grew 106% last quarter"))
    assert "2026-07-26" in period and source == "the user's prompt"


def test_line_period_beats_the_user_prompt(monkeypatch):
    period, source = _period(monkeypatch, _sub("In fiscal 2026 Nvidia revenue grew 65%.",
                                               user_text="Nvidia revenue grew 106% last quarter"))
    assert "2026-01-25" in period and source == "the line"


def test_no_clue_keeps_the_fiscal_year_default(monkeypatch):
    period, source = _period(monkeypatch, _sub("Nvidia revenue was up 65% year-over-year."))
    assert "2026-01-25" in period and source.startswith("the default")


# -- clues and conflicts -------------------------------------------------------------------------


def test_preceding_sentences_are_found_in_the_transcript():
    assert clues.preceding_sentences(TRANSCRIPT, LINE) == "Company X just reported its second quarter. " \
                                                         "The quarter was strong."
    assert clues.preceding_sentences(TRANSCRIPT, "a sentence that is not there") is None


def test_conflicts_are_reported_only_when_clear():
    line = [_sub("Revenue grew 28% year-over-year.", ticker="GOOGL")]
    assert clues.prompt_conflicts("revenue grew 22% to $110 billion last quarter", line, "GOOGL") == [
        "You said 22% and $110 billion; the post says 28%. The post's figure is what was checked."]
    same = [_sub("Revenue grew 22% year-over-year.", context="It just reported its second quarter.")]
    assert clues.prompt_conflicts("revenue grew 22% last quarter", same, "NVDA") == []   # may be the same quarter
    year = [_sub("Revenue grew 22% in Q1 2026.", ticker="GOOGL")]
    assert "the post refers to Q1" in clues.prompt_conflicts("revenue grew 22% last year", year, "GOOGL")[0]


# -- end to end: what the financial agent receives ------------------------------------------------


class _Recorder(StubFinancialAgent):
    def __init__(self):
        super().__init__()
        self.seen = []

    def gather(self, sub_claim, cutoff_date, feedback):
        self.seen.append(sub_claim)
        return super().gather(sub_claim, cutoff_date, feedback)


def test_pipeline_attaches_clues_and_reports_the_mismatch(monkeypatch, tmp_path):
    posts = tmp_path / "clue_posts"
    posts.mkdir()
    (posts / "posts.jsonl").write_text(json.dumps({
        "influencer": "joseph_carlson", "video_id": "demo-q2", "title": "Company X", "description": "Company X",
        "upload_date": "2025-07-31", "transcript": TRANSCRIPT, "channel": "Joseph Carlson"}) + "\n")
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "claims.jsonl").write_text(json.dumps({
        "video_id": "demo-q2", "company_ticker": "CMPX", "sub_claim_id": "c1", "text": LINE,
        "sub_claim_type": "HISTORICAL_GROWTH", "is_quantitative": True}) + "\n")
    monkeypatch.setenv("FINSIGHT_POSTS_DIR", str(posts))
    recorder = _Recorder()
    out = execute("Joseph Carlson said Company X revenue grew 25% last quarter.", date(2025, 7, 31),
                  registry=demo_registry(financial=recorder))
    assert not recorder.seen
    assert out.assessment.clarification_question == (
        "That line is not in this influencer's stored videos. "
        "Restate the line, or resubmit with the video id."
    )
