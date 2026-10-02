from datetime import date

from finsight.agents.orchestrator import routing
from finsight.agents.orchestrator.aggregation import overall_verdict
from finsight.contracts.schemas import ClaimType, EvidenceItem, KnowledgeRule, SourceTier, Stance, Verdict

S, P, C, I = (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED, Verdict.CONTRADICTED,
              Verdict.INSUFFICIENT_EVIDENCE)


def test_overall_verdict_rule():
    assert overall_verdict([S, S]) is S
    assert overall_verdict([I, I]) is I
    assert overall_verdict([]) is I
    assert overall_verdict([C, C, P, I, I]) is C  # 2 of 3 analyzable contradicted
    assert overall_verdict([C, P]) is P  # exactly half is not a majority
    assert overall_verdict([S, P, I, P]) is P


def test_resolve_cutoff():
    default = date(2025, 7, 31)
    assert routing.resolve_cutoff(date(2025, 1, 1), "Cut-off date: 31 Jul 2025", default) == (date(2025, 1, 1), False)
    assert routing.resolve_cutoff(None, "blah. Cut-off date: 30 June 2025", default) == (date(2025, 6, 30), False)
    assert routing.resolve_cutoff(None, "cutoff 2025-03-31", default) == (date(2025, 3, 31), False)
    assert routing.resolve_cutoff(None, "no date here", default) == (default, True)


def _item(stance, tier):
    return EvidenceItem(evidence_id="e", claim_id="c", stance=stance, content="x", source_tier=tier,
                        doc_type="d", company="X", publication_date=date(2025, 1, 1), produced_by="t")


def test_guards():
    rule = KnowledgeRule(claim_type=ClaimType.BUSINESS_RISK, collectors=["narrative"], highest_verdict=S,
                         tier_requirement=[SourceTier.TIER_1, SourceTier.TIER_3], counter_evidence_only=True)
    # Tier 3 never counts, even if the YAML lists it.
    assert routing.enforce_tier_requirement(S, [_item(Stance.SUPPORTING, SourceTier.TIER_3)], rule)[0] is I
    assert routing.enforce_tier_requirement(C, [], rule)[0] is C
    ev = [_item(Stance.SUPPORTING, SourceTier.TIER_1), _item(Stance.CHALLENGING, SourceTier.TIER_1)]
    assert routing.enforce_counter_evidence(S, ev, rule)[0] is P
    capped = rule.model_copy(update={"highest_verdict": P})
    assert routing.cap_verdict(S, capped)[0] is P and routing.cap_verdict(C, capped)[0] is C
    assert routing.is_raise(I, S) and not routing.is_raise(S, I) and not routing.is_raise(I, C)


def test_scope():
    companies = {"CMPX": ["Company X"]}
    assert routing.detect_company("company x is great", companies) == "CMPX"
    assert routing.scope_screen("Company X revenue grew", "CMPX", None, companies) == (None, None)
    refuse, _ = routing.scope_screen("The Fed will cut interest rates", None, None, companies)
    assert "Macro" in refuse
    refuse, _ = routing.scope_screen("Acme will grow", None, "ACME", companies)
    assert "not in the frozen corpus" in refuse
    refuse, question = routing.scope_screen("This stock will keep growing", None, None, companies)
    assert refuse is None and "Which company" in question


def test_tier_shortfall_and_replan():
    rule = KnowledgeRule(claim_type=ClaimType.HISTORICAL_GROWTH, collectors=["financial"],
                         retry_collectors=["narrative", "financial"], highest_verdict=S,
                         tier_requirement=[SourceTier.TIER_1])
    assert routing.tier_shortfall(S, [_item(Stance.SUPPORTING, SourceTier.TIER_1)], rule) is None
    assert "Tier 3" in routing.tier_shortfall(S, [_item(Stance.SUPPORTING, SourceTier.TIER_3)], rule)
    assert routing.tier_shortfall(I, [], rule).startswith("No evidence")
    assert routing.tier_shortfall(C, [_item(Stance.CHALLENGING, SourceTier.TIER_1)], rule) is None
    entry = routing.plan_entry(_claim(), rule)
    assert routing.replan_entry(entry, rule).collectors == ["financial", "narrative"]


def _claim():
    from finsight.contracts.schemas import Claim

    return Claim(claim_id="c1", text="x", claim_type=ClaimType.HISTORICAL_GROWTH)
