from datetime import date

import pytest

from finsight.agents.orchestrator import router
from finsight.agents.orchestrator.nodes import Nodes, RunContext
from finsight.config import get_settings
from finsight.contracts.schemas import ClaimRequest, FinSightState, SubClaim, SubClaimType
from finsight.knowledge.loader import default_knowledge_layer
from finsight.llm import client

KB = default_knowledge_layer()
CUTOFF = date(2026, 4, 30)


def _sub_claim(sub_claim_type=SubClaimType.HISTORICAL_GROWTH, sub_claim_id="q11-1",
               text="Alphabet's revenue grew 22% year over year to $110 billion last quarter"):
    return SubClaim(sub_claim_id=sub_claim_id, company_ticker="GOOGL", text=text, sub_claim_type=sub_claim_type)


@pytest.fixture
def no_llm(monkeypatch):
    """Paid mode, but any LLM call fails the test. Routing must not need one."""
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")

    def fail(*_args, **_kwargs):
        raise AssertionError("routing must not call the LLM")

    monkeypatch.setattr(client, "complete", fail)


def test_routes_by_the_knowledge_layer(no_llm):
    growth = router.route_sub_claim(_sub_claim(), KB.rule_for(SubClaimType.HISTORICAL_GROWTH))
    demand = router.route_sub_claim(_sub_claim(SubClaimType.DEMAND_ATTRIBUTION),
                                    KB.rule_for(SubClaimType.DEMAND_ATTRIBUTION))
    other = router.route_sub_claim(_sub_claim(SubClaimType.OTHER, text="Google grew 22%"),
                                   KB.rule_for(SubClaimType.OTHER))
    assert growth.target == "financial"
    assert demand.target == "narrative"
    assert other.target == "narrative"


def test_same_sub_claim_always_gets_the_same_route(no_llm):
    rule = KB.rule_for(SubClaimType.OTHER)
    routes = {tuple(router.route_sub_claim(_sub_claim(SubClaimType.OTHER), rule).collectors) for _ in range(5)}
    assert routes == {("narrative",)}


def test_agent_input_carries_cutoff_and_feedback(no_llm):
    claim = _sub_claim(text="Alphabet's revenue grew 22% because of AI demand")
    decision = router.route_sub_claim(claim, KB.rule_for(SubClaimType.HISTORICAL_GROWTH),
                                      cutoff_date=CUTOFF, feedback="AAAAA - placeholder")
    assert decision.agent_input() == {
        "sub_claim_id": "q11-1", "company_ticker": "GOOGL", "text": "Alphabet's revenue grew 22% because of AI demand",
        "cutoff_date": CUTOFF, "feedback": "AAAAA - placeholder",
    }


def test_refused_type_gets_no_collector(no_llm):
    decision = router.route_sub_claim(_sub_claim(SubClaimType.PERSONALIZED_ADVICE),
                                      KB.rule_for(SubClaimType.PERSONALIZED_ADVICE))
    assert decision.collectors == [] and decision.reason.startswith("refused type")


def test_route_sub_claims_keeps_order_and_feedback(no_llm):
    claims = [_sub_claim(sub_claim_id="a"), _sub_claim(SubClaimType.DEMAND_ATTRIBUTION, sub_claim_id="b")]
    decisions = router.route_sub_claims(claims, KB, cutoff_date=CUTOFF, feedback={"b": "retry"})
    assert [(d.sub_claim_id, d.target, d.feedback) for d in decisions] == [
        ("a", "financial", None), ("b", "narrative", "retry"),
    ]


def _plan(sub_claims):
    ctx = RunContext(registry=None, settings=get_settings(), kb=KB)  # type: ignore[arg-type]
    ctx.recorder.start("t-1", "claim", CUTOFF)
    state = FinSightState(
        request=ClaimRequest(request_id="t-1", raw_claim="claim", company_ticker="GOOGL", cutoff_date=CUTOFF),
        sub_claims=sub_claims,
    )
    return Nodes(ctx).plan(state), ctx.recorder.log


def test_plan_node_uses_the_rule_route_and_cap(no_llm):
    update, log = _plan([_sub_claim(SubClaimType.DEMAND_ATTRIBUTION)])
    entry = update["routing_plan"]["q11-1"]
    assert entry.collectors == ["narrative"]
    assert entry.verdict_cap == KB.rule_for(SubClaimType.DEMAND_ATTRIBUTION).highest_verdict
    assert update["pending_sub_claim_ids"] == ["q11-1"]
    assert any("narrative - knowledge-layer rule for DEMAND_ATTRIBUTION" in e.message for e in log.events)


def test_plan_node_marks_a_refused_type_insufficient(no_llm):
    update, _log = _plan([_sub_claim(SubClaimType.PERSONALIZED_ADVICE, text="Buy Alphabet now")])
    assert update["pending_sub_claim_ids"] == []
    assert "q11-1" in update["insufficient_reasons"]
