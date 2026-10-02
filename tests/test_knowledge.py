import pytest

from finsight.contracts.schemas import ClaimType, SourceTier, Verdict
from finsight.knowledge.loader import KnowledgeLayerError, load_knowledge_layer


def test_seed_table_loads():
    kb = load_knowledge_layer()
    assert kb.rule_for(ClaimType.HISTORICAL_GROWTH).tier_requirement == [SourceTier.TIER_1]
    assert kb.rule_for(ClaimType.DEMAND_ATTRIBUTION).highest_verdict is Verdict.PARTIALLY_SUPPORTED
    assert kb.rule_for(ClaimType.COMPARATIVE).collectors == []
    assert kb.rule_for(ClaimType.PERSONALIZED_ADVICE).refuse


def test_other_uses_fallback():
    rule = load_knowledge_layer().rule_for(ClaimType.OTHER)
    assert rule.collectors == ["financial", "narrative"]
    assert rule.highest_verdict is Verdict.PARTIALLY_SUPPORTED


def test_unknown_claim_type_fails_loudly(tmp_path):
    bad = tmp_path / "kb.yaml"
    bad.write_text(
        "version: '0.1'\nrules:\n  - {claim_type: MOON_PHASE, collectors: [], highest_verdict: SUPPORTED}\n"
        "fallback: {claim_type: OTHER, collectors: [narrative], highest_verdict: PARTIALLY_SUPPORTED}\n"
    )
    with pytest.raises(KnowledgeLayerError, match="MOON_PHASE"):
        load_knowledge_layer(bad)


def test_duplicate_rule_fails(tmp_path):
    bad = tmp_path / "kb.yaml"
    bad.write_text(
        "version: '0.1'\nrules:\n"
        "  - {claim_type: CONTINUATION, highest_verdict: SUPPORTED}\n"
        "  - {claim_type: CONTINUATION, highest_verdict: SUPPORTED}\n"
        "fallback: {claim_type: OTHER, highest_verdict: PARTIALLY_SUPPORTED}\n"
    )
    with pytest.raises(KnowledgeLayerError, match="duplicate"):
        load_knowledge_layer(bad)
