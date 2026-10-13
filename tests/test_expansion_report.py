"""An expanded phrase is one report item, with the tests and the section 6 rollup written out."""

from datetime import date

from finsight.agents.orchestrator.explain import expansion_block, explain, summary
from finsight.agents.orchestrator.expansion import report_units
from finsight.agents.orchestrator.render import render_assessment
from finsight.contracts.schemas import (
    AssessmentStatus,
    EvidenceItem,
    FinalAssessment,
    SourceTier,
    Stance,
    SubClaim,
    SubClaimResult,
    SubClaimType,
    Verdict,
)

S = Verdict.SUPPORTED
C = Verdict.CONTRADICTED
I = Verdict.INSUFFICIENT_EVIDENCE
P = Verdict.PARTIALLY_SUPPORTED


def _child(measure: str, verdict: Verdict, figure: str | None = None) -> SubClaimResult:
    evidence = []
    if figure:
        evidence.append(EvidenceItem(
            evidence_id=f"e-{measure}", sub_claim_id=measure, stance=Stance.SUPPORTING, content=figure,
            source_tier=SourceTier.TIER_1, doc_type="10-K", company="AAPL", period="FY2025 vs FY2024",
            publication_date=date(2025, 10, 31), produced_by="financial",
        ))
    return SubClaimResult(
        sub_claim=SubClaim(
            sub_claim_id=measure, text=measure, sub_claim_type=SubClaimType.PROFITABILITY_CASH_LEVERAGE,
            parent_span="financially strong", expansion_rule="financially_strong", measure=measure,
            quote_time="02:14",
        ),
        verdict=verdict, supporting=evidence, rationale="test",
    )


def test_three_of_four_rolls_up_to_partly_supported():
    results = [
        _child("profitability", S, "Operating margin was 31% against 29%."),
        _child("cash_generation", S, "Free cash flow was $98bn against $108bn."),
        _child("liquidity", C, "Current ratio was 0.9 against 1.0."),
        _child("leverage", S, "Debt-to-equity did not rise."),
    ]
    units = report_units(results)
    assert len(units) == 1
    assert units[0].verdict is P
    text = expansion_block(units[0])
    assert 'The post said "financially strong" (02:14).' in text
    assert "Operating margin was above zero and did not fall." in text
    assert "Operating margin was 31% against 29%." in text
    assert "3 of 4 met the test, so the phrase is Partly supported." in text
    assert "Leverage was tested on direction only." in text
    assert "favorable" not in text.lower()


def test_two_missing_tests_are_insufficient():
    results = [_child("profitability", S), _child("cash_generation", S)]
    assert report_units(results)[0].verdict is I


def test_a_weak_phrase_reads_the_tests_the_other_way():
    children = []
    for measure in ("profitability", "cash_generation", "liquidity", "leverage"):
        child = _child(measure, C)
        child = child.model_copy(update={
            "sub_claim": child.sub_claim.model_copy(update={
                "parent_span": "in trouble", "expansion_rule": "in_trouble",
            }),
        })
        children.append(child)
    assert report_units(children)[0].verdict is S


def test_summary_counts_the_parent_once():
    results = [
        _child("profitability", S), _child("cash_generation", S),
        _child("liquidity", S), _child("leverage", C),
    ]
    text = summary(results, P)
    assert "split it into 1 points" not in text
    assert "split it into 4 points" not in text
    assert "checked it against company filings" in text


def test_reader_view_shows_one_block_and_the_limits():
    results = [
        _child("profitability", S, "Operating margin was 31% against 29%."),
        _child("cash_generation", S, "Free cash flow stayed positive."),
        _child("liquidity", S, "Current ratio was 1.2."),
        _child("leverage", C, "Debt-to-equity rose."),
    ]
    fa = FinalAssessment(
        request_id="fs-test", status=AssessmentStatus.COMPLETED, claim="Check the post.",
        summary=summary(results, P), overall_verdict=P, sub_claim_results=results,
        boundary_disclaimer="Not a personal recommendation.",
    )
    text = render_assessment(fa)
    assert text.count('The post said "financially strong"') == 2  # summary line and the detail block
    assert "1. [Partly supported: partial match]" in text
    assert "2. [" not in text
    assert "direction only" in text


def test_strong_growth_says_the_size_was_not_tested():
    result = SubClaimResult(
        sub_claim=SubClaim(
            sub_claim_id="g", text="Revenue growth is strong.", sub_claim_type=SubClaimType.HISTORICAL_GROWTH,
            expansion_rule="strong_growth",
        ),
        verdict=S, rationale="test",
    )
    text = explain(result, date(2026, 9, 29), None)
    assert text.startswith("The size of the change was not tested.")
    assert report_units([result])[0].is_parent is False
