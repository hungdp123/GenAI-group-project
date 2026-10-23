"""Knowledge layer section 13: when the orchestrator gives a sub-claim a second look."""

from datetime import date

from finsight.agents.orchestrator import routing
from finsight.agents.orchestrator.nodes import MAX_EXTRA_SEARCHES, MAX_SECOND_LOOKS, Nodes, RunContext
from finsight.config import get_settings
from finsight.contracts.schemas import (
    AuditResult,
    ClaimRequest,
    EvidenceItem,
    FinSightState,
    SourceTier,
    Stance,
    SubClaim,
    SubClaimType,
    Verdict,
)
from finsight.knowledge.loader import default_knowledge_layer

KB = default_knowledge_layer()
CUTOFF = date(2026, 4, 30)


def _claim(cid: str, sub_claim_type: SubClaimType) -> SubClaim:
    return SubClaim(sub_claim_id=cid, company_ticker="GOOGL", text=f"claim {cid}", sub_claim_type=sub_claim_type)


def _item(cid: str) -> EvidenceItem:
    return EvidenceItem(evidence_id=f"e-{cid}", sub_claim_id=cid, stance=Stance.SUPPORTING, content="passage",
                        source_tier=SourceTier.TIER_1, doc_type="10-Q", company="GOOGL",
                        publication_date=CUTOFF, produced_by="test")


def _retry(claims: list[SubClaim], evidence: dict[str, list[EvidenceItem]]):
    """Every sub-claim failed its audit. Returns retry_or_accept's update."""
    ctx = RunContext(registry=None, settings=get_settings(), kb=KB)  # type: ignore[arg-type]
    ctx.recorder.start("t-1", "claim", CUTOFF)
    state = FinSightState(
        request=ClaimRequest(request_id="t-1", raw_claim="claim", company_ticker="GOOGL", cutoff_date=CUTOFF),
        sub_claims=claims,
        routing_plan={c.sub_claim_id: routing.plan_entry(c, KB.rule_for(c.sub_claim_type)) for c in claims},
        pending_sub_claim_ids=[c.sub_claim_id for c in claims],
        audits={c.sub_claim_id: AuditResult(sub_claim_id=c.sub_claim_id, passed=False,
                                            verdict=Verdict.INSUFFICIENT_EVIDENCE, failure_reasons=["bad quote"])
                for c in claims},
        evidence=evidence,
        retries_used={c.sub_claim_id: 0 for c in claims},
    )
    return Nodes(ctx).retry_or_accept(state)


def test_narrative_claim_with_a_missing_passage_gets_one_second_look():
    update = _retry([_claim("a", SubClaimType.DEMAND_ATTRIBUTION)], {"a": []})
    assert update["pending_sub_claim_ids"] == ["a"]
    assert update["retries_used"]["a"] == 1


def test_financial_claim_with_no_figure_gets_no_second_look():
    update = _retry([_claim("a", SubClaimType.HISTORICAL_GROWTH)], {"a": []})
    assert update["pending_sub_claim_ids"] == []
    assert "no second look: the financial agent returned no figure" in update["insufficient_reasons"]["a"]


def test_financial_claim_with_a_figure_gets_a_second_look():
    update = _retry([_claim("a", SubClaimType.HISTORICAL_GROWTH)], {"a": [_item("a")]})
    assert update["pending_sub_claim_ids"] == ["a"]


def test_comparative_claim_gets_no_second_look():
    update = _retry([_claim("a", SubClaimType.COMPARATIVE)], {"a": [_item("a")]})
    assert update["pending_sub_claim_ids"] == []
    assert "no second look: a comparative claim" in update["insufficient_reasons"]["a"]


def test_request_stops_after_the_extra_search_cap():
    claims = [_claim(f"n{i}", SubClaimType.DEMAND_ATTRIBUTION) for i in range(MAX_EXTRA_SEARCHES + 2)]
    update = _retry(claims, {c.sub_claim_id: [] for c in claims})
    assert len(update["pending_sub_claim_ids"]) == MAX_EXTRA_SEARCHES
    assert "cap of 4 extra searches was hit" in update["insufficient_reasons"][claims[-1].sub_claim_id]


def test_request_stops_after_the_second_look_cap():
    claims = [_claim(f"f{i}", SubClaimType.HISTORICAL_GROWTH) for i in range(MAX_SECOND_LOOKS + 2)]
    update = _retry(claims, {c.sub_claim_id: [_item(c.sub_claim_id)] for c in claims})
    assert len(update["pending_sub_claim_ids"]) == MAX_SECOND_LOOKS
    assert "cap of 8 second looks was hit" in update["insufficient_reasons"][claims[-1].sub_claim_id]
