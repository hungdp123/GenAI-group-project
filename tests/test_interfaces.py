from datetime import date

from finsight.config import FALLBACK_CORPUS_AS_OF, corpus_as_of_date, get_settings, latest_corpus_date
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


def test_default_cutoff_is_the_latest_filing(tmp_path, monkeypatch):
    filings = tmp_path / "filings.csv"
    filings.write_text(
        "ticker,filed\nAMD,2026-07-31\nAMD,2026-10-21\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FINSIGHT_CORPUS_AS_OF", raising=False)
    assert latest_corpus_date(filings) == date(2026, 10, 21)
    monkeypatch.setattr("finsight.config.FILINGS_CSV", filings)
    assert corpus_as_of_date() == date(2026, 10, 21)
    assert get_settings().corpus_as_of_date == date(2026, 10, 21)
    monkeypatch.setenv("FINSIGHT_CORPUS_AS_OF", "2024-01-01")
    assert corpus_as_of_date() == date(2024, 1, 1)
    monkeypatch.delenv("FINSIGHT_CORPUS_AS_OF", raising=False)
    monkeypatch.setattr("finsight.config.FILINGS_CSV", tmp_path / "missing.csv")
    assert corpus_as_of_date() == FALLBACK_CORPUS_AS_OF
