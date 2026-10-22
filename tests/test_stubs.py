"""Demo-data expectations for the placeholder stubs (delete a test when its stub is replaced)."""
from finsight.agents.auditor.stub import StubAuditor
from finsight.agents.decomposition.stub import StubDecomposer
from finsight.agents.financial.stub import StubFinancialAgent
from finsight.agents.narrative.demo.stub import StubNarrativeAgent
from finsight.contracts.schemas import SubClaimType, SourceTier, Stance, Verdict

from demo import DEMO_CUTOFF, DEMO_CLAIM


def _sub_claims():
    return StubDecomposer().decompose(DEMO_CLAIM, DEMO_CUTOFF)


def test_demo_decomposition():
    sub_claims = _sub_claims()
    assert [c.sub_claim_type for c in sub_claims] == [
        SubClaimType.HISTORICAL_GROWTH, SubClaimType.DEMAND_ATTRIBUTION,
        SubClaimType.COMPARATIVE, SubClaimType.CONTINUATION,
    ]
    assert sub_claims[0].is_quantitative and all(c.company_ticker == "CMPX" for c in sub_claims)


def test_financial_revenue_growth():
    [item] = StubFinancialAgent().gather(_sub_claims()[0], DEMO_CUTOFF, None)
    assert item.computed_value == 0.14 and item.source_tier is SourceTier.TIER_1 and item.doc_type == "10-Q"


def test_narrative_returns_both_stances():
    items = StubNarrativeAgent().gather(_sub_claims()[1], DEMO_CUTOFF, None)
    assert {i.stance for i in items} == {Stance.SUPPORTING, Stance.CHALLENGING}
    assert any(i.source_tier is SourceTier.TIER_3 for i in items)


def test_auditor_forced_failure(monkeypatch):
    monkeypatch.setenv("FINSIGHT_STUB_FAIL_SUB_CLAIMS", "sc1")
    auditor, sub_claim = StubAuditor(), _sub_claims()[0]
    assert not auditor.audit(sub_claim, []).passed
    second = auditor.audit(sub_claim, StubFinancialAgent().gather(sub_claim, DEMO_CUTOFF, None))
    assert second.passed and second.verdict is Verdict.SUPPORTED
