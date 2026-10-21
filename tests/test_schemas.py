from datetime import date

import pytest
from pydantic import ValidationError

from finsight.contracts.schemas import (
    SCHEMA_VERSION,
    SubClaim,
    SubClaimType,
    DecisionLog,
    EvidenceItem,
    FinSightState,
    SourceTier,
    Stance,
    Verdict,
)


def test_schema_version():
    assert SCHEMA_VERSION == "0.6"


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        SubClaim(sub_claim_id="sc1", text="x", sub_claim_type=SubClaimType.OTHER, bogus=1)


def test_evidence_roundtrip():
    item = EvidenceItem(
        evidence_id="e1", sub_claim_id="sc1", stance=Stance.SUPPORTING, content="Revenue +14%",
        source_tier=SourceTier.TIER_1, doc_type="10-Q", company="CMPX",
        publication_date=date(2025, 7, 25), produced_by="financial", computed_value=0.14,
    )
    assert EvidenceItem.model_validate_json(item.model_dump_json()) == item


def test_verdict_and_tier_ordering():
    assert Verdict.SUPPORTED.support_rank > Verdict.PARTIALLY_SUPPORTED.support_rank
    assert Verdict.INSUFFICIENT_EVIDENCE.support_rank > Verdict.CONTRADICTED.support_rank
    assert SourceTier.TIER_1.rank < SourceTier.TIER_3.rank


def test_state_and_log_defaults():
    assert FinSightState().sub_claims == []
    assert DecisionLog(request_id="r", input="x").total_tokens == 0
