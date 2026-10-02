import pytest


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch, tmp_path):
    """Every test runs in MOCK mode, writes logs to a temp dir, and starts with no stub failures."""
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "mock")
    monkeypatch.setenv("FINSIGHT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("FINSIGHT_STUB_FAIL_CLAIMS", raising=False)
    monkeypatch.delenv("FINSIGHT_MAX_RETRIES", raising=False)
