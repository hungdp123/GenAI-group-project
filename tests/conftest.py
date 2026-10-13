import json

import pytest

from demo import FIXTURE_POSTS
from finsight.config import STUB_IMPLEMENTATIONS

# These tests run against whatever agents are registered in finsight/config.py (real or stub).
# Every other test module is pinned to the stubs so its demo expectations stay stable.
REGISTERED_AGENT_MODULES = {"test_agent_contracts", "test_auditor_checks"}


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch, tmp_path, request):
    """Every test runs in MOCK mode with no real keys, temp logs/ledger, and no stub failures."""
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "mock")
    monkeypatch.setenv("FINSIGHT_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("FINSIGHT_STUB_FAIL_SUB_CLAIMS", raising=False)
    monkeypatch.delenv("FINSIGHT_MAX_RETRIES", raising=False)
    monkeypatch.delenv("FINSIGHT_LLM_MODEL", raising=False)
    monkeypatch.setenv("FINSIGHT_USAGE_LEDGER", str(tmp_path / "llm_usage.jsonl"))
    monkeypatch.setenv("FINSIGHT_SECRETS_FILE", str(tmp_path / "no_secrets.yaml"))
    posts = tmp_path / "posts"
    posts.mkdir()
    lines = [
        json.dumps({
            "influencer": "joseph_carlson",
            "video_id": f"demo-{index}",
            "title": "Company X",
            "description": "Company X",
            "upload_date": "2025-07-31",
            "transcript": text,
            "channel": "Joseph Carlson",
        })
        for index, text in enumerate(FIXTURE_POSTS)
    ]
    (posts / "posts.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setenv("FINSIGHT_POSTS_DIR", str(posts))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if request.module.__name__.rsplit(".", 1)[-1] not in REGISTERED_AGENT_MODULES:
        for slot, path in STUB_IMPLEMENTATIONS.items():
            monkeypatch.setenv(f"FINSIGHT_IMPL_{slot.upper()}", path)
    from finsight.llm.spend import GUARD

    GUARD.reset()
