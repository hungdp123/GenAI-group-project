"""Plain-language output: summary first, readable sources, rule-built explanations (no LLM, no new numbers)."""

import re
from datetime import date

from finsight import execute
from finsight.agents.advice_boundary.stub import StubAdviceBoundary
from finsight.agents.orchestrator.explain import source_label, summary, verdict_label
from finsight.agents.orchestrator.render import render_assessment
from finsight.contracts.schemas import (
    AdviceStage, EvidenceItem, SourceTier, Stance, SubClaim, SubClaimResult, SubClaimType, Verdict,
)

from demo import DEMO_CLAIM, DEMO_CUTOFF


def _demo():
    return execute(DEMO_CLAIM, DEMO_CUTOFF).assessment


def test_summary_and_claim_are_returned_for_the_ui():
    fa = _demo()
    assert fa.claim == DEMO_CLAIM
    assert fa.summary == ("Overall, this claim is partly supported. FinSight split it into 4 points and checked "
                          "each against company filings and reports: 1 supported, 2 partly supported, "
                          "1 can't be confirmed.")
    assert all(r.explanation for r in fa.sub_claim_results)


def test_explanations_are_plain_and_add_no_numbers():
    by_verdict = {r.verdict: r.explanation for r in _demo().sub_claim_results}
    assert by_verdict[Verdict.SUPPORTED] == "Backed up by official filings (see the evidence below)."
    assert "competitors" in by_verdict[Verdict.INSUFFICIENT_EVIDENCE]
    for r in _demo().sub_claim_results:
        assert not re.search(r"\d", r.explanation), "explanations must not introduce numbers"
        assert "_" not in r.explanation, "no internal identifiers like PARTIALLY_SUPPORTED"


def test_explanations_never_trip_the_output_advice_gate():
    gate = StubAdviceBoundary()
    for r in _demo().sub_claim_results:
        assert not gate.check(r.explanation, AdviceStage.OUTPUT).flagged


def test_reader_view_puts_summary_before_details():
    text = render_assessment(_demo())
    assert text.index("SUMMARY") < text.index("DETAILS") < text.index("Evidence for:")
    assert "1. [Supported]" in text and "3. [Can't be confirmed]" in text
    flat = " ".join(text.split())  # source lines wrap in the terminal
    assert "Quarterly report (10-Q) filed with the SEC, 25 Jul 2025 (Tier 1: official filing, most reliable)" in flat
    assert "[T1 10-Q" not in text, "no cryptic source codes in the reader view"


def test_technical_view_is_still_available():
    text = render_assessment(_demo(), technical=True)
    assert "Overall verdict: PARTIALLY_SUPPORTED" in text and "[T1 10-Q 2025-07-25]" in text


def test_refusal_and_clarification_summaries_are_readable():
    refused = execute("Should I put my savings into Company X?", DEMO_CUTOFF).assessment
    assert refused.summary.startswith("FinSight didn't analyse this request.")
    assert "\\b" not in refused.summary and "pattern" not in refused.summary
    asked = execute("Its revenue growth is just getting started.", DEMO_CUTOFF).assessment
    assert asked.summary.startswith("FinSight needs a bit more information") and "Which influencer" in asked.summary


def test_overall_sentence_reads_naturally_for_every_verdict():
    ranking = SubClaimResult(
        sub_claim=SubClaim(sub_claim_id="c", text="biggest winner", sub_claim_type=SubClaimType.COMPARATIVE),
        verdict=Verdict.INSUFFICIENT_EVIDENCE, rationale="A ranking is not checked against peer filings.",
    )
    text = summary([ranking], Verdict.INSUFFICIENT_EVIDENCE)
    assert text.startswith("Overall, this claim can't be confirmed from the available evidence.")
    single = execute(
        "Joseph Carlson said Company X is getting more profitable: its margins keep expanding.",
        DEMO_CUTOFF).assessment
    assert single.summary == ("Overall, the evidence contradicts this claim. "
                              "FinSight checked it against company filings and reports.")
    assert "claim is can't" not in text
    assert "checked each" not in single.summary  # one point: "checked it"


def test_company_given_separately_is_not_shown_as_unknown():
    fa = execute("Its revenue growth is just getting started.", DEMO_CUTOFF, company_ticker="CMPX",
                 influencer="Joseph Carlson").assessment
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
