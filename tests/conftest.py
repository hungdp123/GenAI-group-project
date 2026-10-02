import pytest


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch, tmp_path):
    """Every test runs in MOCK mode with no real keys, temp logs/ledger, and no stub failures."""
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "mock")
    monkeypatch.setenv("FINSIGHT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("FINSIGHT_STUB_FAIL_CLAIMS", raising=False)
    monkeypatch.delenv("FINSIGHT_MAX_RETRIES", raising=False)
    monkeypatch.delenv("FINSIGHT_LLM_MODEL", raising=False)
    monkeypatch.setenv("FINSIGHT_USAGE_LEDGER", str(tmp_path / "llm_usage.jsonl"))
    monkeypatch.setenv("FINSIGHT_SECRETS_FILE", str(tmp_path / "no_secrets.yaml"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from finsight.llm.spend import GUARD

    GUARD.reset()
