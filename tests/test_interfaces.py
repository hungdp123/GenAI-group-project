from datetime import date

from finsight.config import get_settings
from finsight.contracts.interfaces import Decomposer, FinancialAgent, Retriever
from finsight.contracts.schemas import Claim, ClaimType


class _Dec:
    def decompose(self, thesis: str, cutoff_date: date) -> list[Claim]:
        return [Claim(claim_id="c1", text=thesis, claim_type=ClaimType.OTHER)]


def test_protocol_structural_match():
    assert isinstance(_Dec(), Decomposer)
    assert not isinstance(_Dec(), FinancialAgent)
    assert not isinstance(object(), Retriever)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("FINSIGHT_MAX_RETRIES", "2")
    assert get_settings().max_retries == 2
    assert get_settings().llm_provider == "mock"
