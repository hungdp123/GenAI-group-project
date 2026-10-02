"""Demo-data expectations for the placeholder stubs (delete a test when its stub is replaced)."""
from finsight.agents.auditor.stub import StubAuditor
from finsight.agents.decomposition.stub import StubDecomposer
from finsight.agents.financial.stub import StubFinancialAgent
from finsight.agents.narrative.stub import StubNarrativeAgent
from finsight.contracts.schemas import ClaimType, SourceTier, Stance, Verdict

from demo import DEMO_CUTOFF, DEMO_THESIS


def _claims():
    return StubDecomposer().decompose(DEMO_THESIS, DEMO_CUTOFF)


def test_demo_decomposition():
    claims = _claims()
    assert [c.claim_type for c in claims] == [
        ClaimType.HISTORICAL_GROWTH, ClaimType.DEMAND_ATTRIBUTION, ClaimType.COMPARATIVE, ClaimType.CONTINUATION,
    ]
    assert claims[0].is_quantitative and all(c.company_ticker == "CMPX" for c in claims)


def test_financial_revenue_growth():
    [item] = StubFinancialAgent().gather(_claims()[0], DEMO_CUTOFF, None)
    assert item.computed_value == 0.14 and item.source_tier is SourceTier.TIER_1 and item.doc_type == "10-Q"


def test_narrative_returns_both_stances():
    items = StubNarrativeAgent().gather(_claims()[1], DEMO_CUTOFF, None)
    assert {i.stance for i in items} == {Stance.SUPPORTING, Stance.CHALLENGING}
    assert any(i.source_tier is SourceTier.TIER_3 for i in items)


def test_auditor_forced_failure(monkeypatch):
    monkeypatch.setenv("FINSIGHT_STUB_FAIL_CLAIMS", "c1")
    auditor, claim = StubAuditor(), _claims()[0]
    assert not auditor.audit(claim, []).passed
    second = auditor.audit(claim, StubFinancialAgent().gather(claim, DEMO_CUTOFF, None))
    assert second.passed and second.verdict is Verdict.SUPPORTED
