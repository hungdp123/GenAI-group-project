"""Plain-language output: summary first, readable sources, rule-built explanations (no LLM, no new numbers)."""

from datetime import date

from finsight import execute
from finsight.agents.advice_boundary.agent import AdviceBoundaryAgent
from finsight.agents.orchestrator.explain import source_label, summary, verdict_label
from finsight.agents.orchestrator.render import render_assessment
from finsight.contracts.schemas import (
    AdviceStage, EvidenceItem, SourceTier, Stance, SubClaim, SubClaimResult, SubClaimType, Verdict,
)

from demo import DEMO_CLAIM, DEMO_CUTOFF
from stored_demo import DEMO_PROMPT, demo_registry, prepared_demo


def _demo(monkeypatch, tmp_path, transcript: str | None = None, prompt: str | None = None):
    prepared_demo(monkeypatch, tmp_path, transcript=transcript or DEMO_CLAIM)
    return execute(prompt or DEMO_PROMPT, DEMO_CUTOFF, registry=demo_registry()).assessment


def test_summary_and_claim_are_returned_for_the_ui(monkeypatch, tmp_path):
    fa = _demo(monkeypatch, tmp_path)
    assert fa.claim == DEMO_PROMPT
    assert fa.summary == ("Overall, this claim is partly supported. FinSight split it into 4 points and checked "
                          "each against company filings and reports: 1 supported, 2 partly supported, "
                          "1 can't be confirmed.")
    assert all(r.explanation for r in fa.sub_claim_results)


def test_explanations_are_plain_and_add_no_numbers(monkeypatch, tmp_path):
    finished = _demo(monkeypatch, tmp_path)
    by_verdict = {r.verdict: r.explanation for r in finished.sub_claim_results}
    supported = by_verdict[Verdict.SUPPORTED]
    assert supported.startswith("Backed up by official filings (see the evidence below).")
    assert "Quarterly report (10-Q) filed with the SEC, 25 Jul 2025" in supported
    assert "Where: XBRL us-gaap:Revenues." in supported
    assert "competitors" in by_verdict[Verdict.INSUFFICIENT_EVIDENCE]
    for r in finished.sub_claim_results:
        assert "_" not in r.explanation, "no internal identifiers like PARTIALLY_SUPPORTED"


def test_explanations_never_trip_the_output_advice_gate(monkeypatch, tmp_path):
    gate = AdviceBoundaryAgent()
    for r in _demo(monkeypatch, tmp_path).sub_claim_results:
        assert not gate.check(r.explanation, AdviceStage.OUTPUT).flagged


def test_reader_view_puts_summary_before_details(monkeypatch, tmp_path):
    text = render_assessment(_demo(monkeypatch, tmp_path))
    assert text.index("SUMMARY") < text.index("DETAILS") < text.index("Evidence for:")
    assert "1. [Supported]" in text and "3. [Can't be confirmed]" in text
    flat = " ".join(text.split())  # source lines wrap in the terminal
    assert "Quarterly report (10-Q) filed with the SEC, 25 Jul 2025 (Tier 1: official filing, most reliable)" in flat
    assert "[T1 10-Q" not in text, "no cryptic source codes in the reader view"


def test_technical_view_is_still_available(monkeypatch, tmp_path):
    text = render_assessment(_demo(monkeypatch, tmp_path), technical=True)
    assert "Overall verdict: PARTIALLY_SUPPORTED" in text and "[T1 10-Q 2025-07-25]" in text


def test_refusal_and_clarification_summaries_are_readable():
    refused = execute("Should I put my savings into Company X?", DEMO_CUTOFF).assessment
    assert refused.summary.startswith("FinSight didn't analyse this request.")
    assert "\\b" not in refused.summary and "pattern" not in refused.summary
    asked = execute("Its revenue growth is just getting started.", DEMO_CUTOFF).assessment
    assert asked.summary.startswith("FinSight needs a bit more information") and "Which influencer" in asked.summary


def test_overall_sentence_reads_naturally_for_every_verdict(monkeypatch, tmp_path):
    ranking = SubClaimResult(
        sub_claim=SubClaim(sub_claim_id="c", text="biggest winner", sub_claim_type=SubClaimType.COMPARATIVE),
        verdict=Verdict.INSUFFICIENT_EVIDENCE, rationale="A ranking is not checked against peer filings.",
    )
    text = summary([ranking], Verdict.INSUFFICIENT_EVIDENCE)
    assert text.startswith("Overall, this claim can't be confirmed from the available evidence.")
    margins = "Joseph Carlson said Company X is getting more profitable: its margins keep expanding."
    single = _demo(monkeypatch, tmp_path, transcript=margins, prompt=margins)
    assert single.summary == ("Overall, the evidence contradicts this claim. "
                              "FinSight checked it against company filings and reports.")
    assert "claim is can't" not in text
    assert "checked each" not in single.summary  # one point: "checked it"


def test_company_given_separately_is_not_shown_as_unknown(monkeypatch, tmp_path):
    sentence = "Its revenue growth is just getting started."
    prepared_demo(monkeypatch, tmp_path, transcript=sentence)
    fa = execute(sentence, DEMO_CUTOFF, company_ticker="CMPX", influencer="Joseph Carlson",
                 registry=demo_registry()).assessment
    assert fa.sub_claim_results and not any("UNKNOWN" in r.sub_claim.text for r in fa.sub_claim_results)


def test_a_partial_label_names_why():
    from finsight.agents.orchestrator.explain import explain, partial_reason

    forward = SubClaimResult(
        sub_claim=SubClaim(sub_claim_id="a", text="Growth is just getting started.",
                           sub_claim_type=SubClaimType.CONTINUATION),
        verdict=Verdict.PARTIALLY_SUPPORTED, rationale="test",
    )
    causes = SubClaimResult(
        sub_claim=SubClaim(sub_claim_id="b", text="Growth is from AI demand.",
                           sub_claim_type=SubClaimType.DEMAND_ATTRIBUTION),
        verdict=Verdict.PARTIALLY_SUPPORTED, rationale="test",
    )
    assert partial_reason(forward) == "forward-looking"
    assert partial_reason(causes) == "more than one cause"
    assert "about the future" in explain(forward, date(2026, 9, 29), None)
    assert "more than one cause" in explain(causes, date(2026, 9, 29), None)


def test_labels():
    assert verdict_label(Verdict.PARTIALLY_SUPPORTED) == "Partly supported"
    assert verdict_label(None) == "Not assessed"
    news = EvidenceItem(evidence_id="n", sub_claim_id="sc1", stance=Stance.SUPPORTING, content="x",
                        source_tier=SourceTier.TIER_3, doc_type="news", company="Company X",
                        publication_date=date(2025, 7, 28), produced_by="t")
    assert source_label(news) == "News article, 28 Jul 2025 (Tier 3: news, can't confirm a claim on its own)"


def test_a_lowered_verdict_names_the_auditor_proposal():
    from finsight.agents.orchestrator.explain import explain

    result = SubClaimResult(
        sub_claim=SubClaim(sub_claim_id="c", text="Growth is just getting started.",
                           sub_claim_type=SubClaimType.CONTINUATION),
        verdict=Verdict.PARTIALLY_SUPPORTED,
        capped_by="SUPPORTED -> PARTIALLY_SUPPORTED: tier rule: Tier 2 only",
        rationale="test",
    )
    text = explain(result, date(2026, 9, 29), None)
    assert "The auditor proposed Supported." in text
    assert "lowered that to Partly supported." in text
    assert "_" not in text


def test_a_measured_comparison_uses_the_two_figures():
    from finsight.agents.orchestrator.explain import comparison_line
    from finsight.agents.orchestrator.routing import measurable_comparison
    from finsight.config import get_settings

    companies = get_settings().corpus_companies
    pair = measurable_comparison("Mark Roussin said Nvidia's revenue grew faster than AMD's.", companies)
    assert pair is not None
    assert pair[0] == ["NVDA", "AMD"]
    assert pair[1] == "revenue growth"
    assert measurable_comparison("Joseph Carlson said Company X is the biggest winner.", companies) is None
    sentence, verdict = comparison_line(
        "Nvidia's revenue grew faster than AMD.", "NVIDIA", "AMD",
        "NVIDIA revenue growth was 24%.", "AMD revenue growth was 10%.", 0.24, 0.10,
    )
    assert verdict is Verdict.SUPPORTED
    assert "NVIDIA is higher than AMD" in sentence
    wrong, contradicted = comparison_line(
        "Nvidia's revenue grew faster than AMD.", "NVIDIA", "AMD", None, None, 0.10, 0.24,
    )
    assert contradicted is Verdict.CONTRADICTED
    assert "lower than" in wrong


def test_two_company_runs_get_one_summary_not_two():
    from finsight.agents.orchestrator.graph import RunOutput, _combine
    from finsight.contracts.schemas import AssessmentStatus, DecisionLog, FinalAssessment

    def row(cid: str, verdict: Verdict) -> SubClaimResult:
        return SubClaimResult(sub_claim=SubClaim(sub_claim_id=cid, text=f"Point {cid}.",
                                                 sub_claim_type=SubClaimType.HISTORICAL_GROWTH),
                              verdict=verdict, rationale="test")

    def run(rows: list[SubClaimResult], headline: list[str], text: str) -> RunOutput:
        final = FinalAssessment(request_id="r", status=AssessmentStatus.COMPLETED, summary=text,
                                sub_claim_results=rows, headline_sub_claim_ids=headline,
                                boundary_disclaimer="d")
        return RunOutput(final, DecisionLog(request_id="r", input="x"))

    nvda = run([row("n1", Verdict.SUPPORTED), row("n2", Verdict.INSUFFICIENT_EVIDENCE)], ["n1", "n2"],
               "Overall, this claim is partly supported. FinSight split it into 2 points and checked each "
               "against company filings and reports: 1 supported, 1 can't be confirmed.")
    amd = run([row("a1", Verdict.CONTRADICTED), row("a2", Verdict.SUPPORTED)], ["a1"],
              "Overall, the evidence contradicts this claim. FinSight checked it against company filings and "
              "reports. In the same line the influencer also said 1 other thing; those are checked below but "
              "not counted in this verdict.")
    merged = _combine([nvda, amd]).assessment
    assert merged.summary.count("Overall,") == 1
    assert "split it into 3 points" in merged.summary
    assert merged.summary.count("also said 1 other thing") == 1
    assert merged.headline_sub_claim_ids == ["n1", "n2", "a1"]


def test_an_unconfirmed_point_says_what_was_looked_for():
    from finsight.agents.orchestrator.explain import explain

    def unconfirmed(sub_type: SubClaimType, reason: str | None = None) -> str:
        result = SubClaimResult(sub_claim=SubClaim(sub_claim_id="c", text="x", sub_claim_type=sub_type),
                                verdict=Verdict.INSUFFICIENT_EVIDENCE, rationale="test")
        return explain(result, date(2026, 7, 24), reason)

    growth = unconfirmed(SubClaimType.HISTORICAL_GROWTH, "the financial agent returned no figure: a missing tag")
    assert "SEC filings and earnings releases, but none reports this exact measure and period." in growth
    assert "filings only report what has already happened" in unconfirmed(SubClaimType.CONTINUATION)
