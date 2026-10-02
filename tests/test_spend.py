"""Real-provider plumbing tested against a fake OpenAI SDK: no network, no spend."""
import sys
import types

import pytest

from finsight.cli import main as cli_main
from finsight.llm import client
from finsight.llm.spend import GUARD, MissingAPIKey, SpendLimitExceeded, api_key, ledger_summary


class _FakeCompletions:
    calls: list[dict] = []

    def create(self, **kwargs):
        _FakeCompletions.calls.append(kwargs)
        msg = types.SimpleNamespace(content="SUB-CLAIM: x | VERDICT: Supported")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
            usage=types.SimpleNamespace(prompt_tokens=1_000, completion_tokens=2_000),
        )


@pytest.fixture
def fake_openai(monkeypatch, tmp_path):
    _FakeCompletions.calls = []
    fake = types.ModuleType("openai")
    fake.OpenAI = lambda api_key: types.SimpleNamespace(chat=types.SimpleNamespace(completions=_FakeCompletions()))
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    secrets = tmp_path / "config.yaml"
    secrets.write_text('openai_api_key: "sk-test"\n')
    monkeypatch.setenv("FINSIGHT_SECRETS_FILE", str(secrets))
    return _FakeCompletions


def test_key_from_secrets_file_and_env(fake_openai, monkeypatch):
    assert api_key("openai") == "sk-test"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert api_key("openai") == "sk-env"


def test_missing_key(monkeypatch):
    with pytest.raises(MissingAPIKey):
        api_key("openai")


def test_gpt5_uses_max_completion_tokens_and_no_temperature(fake_openai):
    resp = client.complete("hi", model="gpt-5.4")
    [kwargs] = fake_openai.calls
    assert kwargs["max_completion_tokens"] == 4096 and "temperature" not in kwargs and "max_tokens" not in kwargs
    assert resp.cost_usd == pytest.approx((1_000 * 2.50 + 2_000 * 15.00) / 1e6)
    assert resp.finish_reason == "stop"


def test_older_model_gets_temperature(fake_openai):
    client.complete("hi", model="gpt-4o-mini")
    assert fake_openai.calls[0]["temperature"] == 0.0


def test_ledger_and_usage_command(fake_openai, capsys):
    client.complete("a", model="gpt-5.4-mini")
    client.complete("b", model="gpt-5.4-mini")
    totals = ledger_summary()
    assert totals["ALL"]["calls"] == 2 and totals["gpt-5.4-mini"]["input_tokens"] == 2_000
    assert cli_main(["usage"]) == 0
    assert "course credit" in capsys.readouterr().out


def test_spend_limit_stops_calls(fake_openai, monkeypatch):
    monkeypatch.setenv("FINSIGHT_SPEND_LIMIT_USD", "0.04")
    client.complete("a", model="gpt-5.4")  # $0.0325
    client.complete("b", model="gpt-5.4")  # total $0.065 >= limit
    with pytest.raises(SpendLimitExceeded):
        client.complete("c", model="gpt-5.4")
    assert GUARD.calls == 2


def test_unpriced_model_refused(fake_openai):
    with pytest.raises(ValueError, match="PRICE_TABLE"):
        client.complete("a", model="gpt-9-ultra")
    assert fake_openai.calls == []


def test_mode_switch_in_config_file(monkeypatch, tmp_path, capsys):
    from finsight.config import get_settings, read_local_config

    monkeypatch.delenv("FINSIGHT_LLM_PROVIDER")  # conftest forces mock via env; test the file instead
    secrets = tmp_path / "config.yaml"
    secrets.write_text('openai_api_key: "sk-test"\n')
    monkeypatch.setenv("FINSIGHT_SECRETS_FILE", str(secrets))
    assert get_settings().llm_provider == "mock"  # no mode set -> free

    assert cli_main(["mode", "paid"]) == 0
    assert (get_settings().llm_provider, get_settings().llm_model) == ("openai", "gpt-5.4-nano")
    assert cli_main(["mode", "paid", "--model", "gpt-5.4"]) == 0
    assert get_settings().llm_model == "gpt-5.4"
    assert cli_main(["mode", "mock"]) == 0
    assert get_settings().llm_provider == "mock" and get_settings().llm_model == "mock-model"
    assert read_local_config()["openai_api_key"] == "sk-test", "switching mode must keep the key"
    assert "MOCK (free" in capsys.readouterr().out

    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")  # env var still wins for one-off runs
    assert get_settings().llm_provider == "openai"


def test_mock_mode_does_not_touch_ledger():
    client.complete("x")
    assert ledger_summary() == {}
