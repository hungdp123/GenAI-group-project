"""Contract tests run against whatever implementation is registered in finsight/config.py.

When you replace your stub, these must still pass (see CONTRIBUTING.md).
"""
from finsight.config import AgentRegistry
from finsight.contracts.schemas import AdviceStage, Claim, ClaimType, EvidenceItem, KnowledgeRule, Verdict

from demo import DEMO_CUTOFF, DEMO_THESIS


def _registry():
    return AgentRegistry.from_config()


def test_decomposer_contract():
    claims = _registry().decomposer.decompose(DEMO_THESIS, DEMO_CUTOFF)
    assert claims and all(isinstance(c, Claim) for c in claims)
    assert len({c.claim_id for c in claims}) == len(claims), "claim_ids must be unique"


def _collector_contract(agent):
    claims = _registry().decomposer.decompose(DEMO_THESIS, DEMO_CUTOFF)
    for claim in claims:
        for item in agent.gather(claim, DEMO_CUTOFF, None):
            assert isinstance(item, EvidenceItem)
            assert item.claim_id == claim.claim_id
            assert item.publication_date <= DEMO_CUTOFF, "never return post-cutoff evidence"


def test_financial_contract():
    _collector_contract(_registry().financial)


def test_narrative_contract():
    _collector_contract(_registry().narrative)


def test_retriever_contract():
    items = _registry().retriever.retrieve("Company X AI demand", 3, DEMO_CUTOFF)
    assert len(items) <= 3
    assert all(i.publication_date <= DEMO_CUTOFF for i in items)


def test_auditor_contract():
    reg = _registry()
    claim = reg.decomposer.decompose(DEMO_THESIS, DEMO_CUTOFF)[0]
    result = reg.auditor.audit(claim, reg.financial.gather(claim, DEMO_CUTOFF, None))
    assert result.claim_id == claim.claim_id


def test_advice_boundary_contract():
    adv = _registry().advice_boundary
    assert adv.check("Should I put my savings into Company X?", AdviceStage.INPUT).flagged
    assert not adv.check(DEMO_THESIS, AdviceStage.INPUT).flagged
    assert adv.check("x", AdviceStage.OUTPUT).stage is AdviceStage.OUTPUT


def test_tier_policy_never_raises():
    rule = KnowledgeRule(claim_type=ClaimType.OTHER, collectors=["narrative"],
                         highest_verdict=Verdict.PARTIALLY_SUPPORTED, tier_requirement=[])
    claim = Claim(claim_id="c1", text="x", claim_type=ClaimType.OTHER)
    verdict, _ = _registry().tier_policy.apply(claim, Verdict.INSUFFICIENT_EVIDENCE, [], rule)
    assert verdict.support_rank <= Verdict.INSUFFICIENT_EVIDENCE.support_rank
