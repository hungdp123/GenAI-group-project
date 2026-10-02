from datetime import date

from finsight.config import get_settings
from finsight.contracts.interfaces import Decomposer, FinancialAgent, Retriever
from finsight.contracts.schemas import SubClaim, SubClaimType


class _Dec:
    def decompose(self, claim: str, cutoff_date: date) -> list[SubClaim]:
        return [SubClaim(sub_claim_id="sc1", text=claim, sub_claim_type=SubClaimType.OTHER)]


def test_protocol_structural_match():
    assert isinstance(_Dec(), Decomposer)
    assert not isinstance(_Dec(), FinancialAgent)
    assert not isinstance(object(), Retriever)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("FINSIGHT_MAX_RETRIES", "2")
    assert get_settings().max_retries == 2
    assert get_settings().llm_provider == "mock"
