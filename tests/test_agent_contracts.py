"""Contract tests run against whatever implementation is registered in finsight/config.py.

When you replace your stub, these must still pass (see CONTRIBUTING.md).
"""
from finsight.config import AgentRegistry
from finsight.contracts.schemas import AdviceStage, SubClaim, SubClaimType, EvidenceItem, KnowledgeRule, Verdict

from demo import DEMO_CUTOFF, DEMO_CLAIM


def _registry():
    return AgentRegistry.from_config()


def test_decomposer_contract():
    sub_claims = _registry().decomposer.decompose(DEMO_CLAIM, DEMO_CUTOFF)
    assert sub_claims and all(isinstance(c, SubClaim) for c in sub_claims)
    assert len({c.sub_claim_id for c in sub_claims}) == len(sub_claims), "sub_claim_ids must be unique"


def _collector_contract(agent):
    sub_claims = _registry().decomposer.decompose(DEMO_CLAIM, DEMO_CUTOFF)
    for sub_claim in sub_claims:
        for item in agent.gather(sub_claim, DEMO_CUTOFF, None):
            assert isinstance(item, EvidenceItem)
            assert item.sub_claim_id == sub_claim.sub_claim_id
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
    sub_claim = reg.decomposer.decompose(DEMO_CLAIM, DEMO_CUTOFF)[0]
    result = reg.auditor.audit(sub_claim, reg.financial.gather(sub_claim, DEMO_CUTOFF, None))
    assert result.sub_claim_id == sub_claim.sub_claim_id


def test_advice_boundary_contract():
    adv = _registry().advice_boundary
    assert adv.check("Should I put my savings into Company X?", AdviceStage.INPUT).flagged
    assert not adv.check(DEMO_CLAIM, AdviceStage.INPUT).flagged
    assert adv.check("x", AdviceStage.OUTPUT).stage is AdviceStage.OUTPUT


def test_tier_policy_never_raises():
    rule = KnowledgeRule(sub_claim_type=SubClaimType.OTHER, collectors=["narrative"],
                         highest_verdict=Verdict.PARTIALLY_SUPPORTED, tier_requirement=[])
    sub_claim = SubClaim(sub_claim_id="sc1", text="x", sub_claim_type=SubClaimType.OTHER)
    verdict, _ = _registry().tier_policy.apply(sub_claim, Verdict.INSUFFICIENT_EVIDENCE, [], rule)
    assert verdict.support_rank <= Verdict.INSUFFICIENT_EVIDENCE.support_rank
