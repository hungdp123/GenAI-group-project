"""Local evaluation CLI defaults and explicit API opt-in; no model calls."""
from argparse import Namespace

from pydantic import BaseModel
import pytest

from finsight.llm.client import LLMResponse
from scripts.narrative.evaluation import evaluate_ragas, run_rag_comparison
from scripts.narrative.evaluation.model_options import DEFAULT_OLLAMA_MODEL, configure_cli_model


@pytest.mark.parametrize("module,required", [
    (run_rag_comparison, ["--limit", "1", "--output-dir", "local"]),
    (evaluate_ragas, ["--limit", "1", "--run-dir", "saved", "--output-dir", "local"]),
])
def test_cli_defaults_to_local_even_with_shared_openai_config(monkeypatch, module, required):
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FINSIGHT_LLM_MODEL", "gpt-5.4")
    args = module.build_parser().parse_args(required)
    configure_cli_model(args)
    assert args.ollama_model == DEFAULT_OLLAMA_MODEL
    assert args.ollama_host == "http://localhost:11434"
    assert args.ollama_context == 20000
    assert not args.shared_llm


@pytest.mark.parametrize("module,required", [
    (run_rag_comparison, ["--all-cases", "--output-dir", "api"]),
    (evaluate_ragas, ["--all-cases", "--run-dir", "saved", "--output-dir", "api"]),
])
def test_api_provider_requires_explicit_opt_in(module, required):
    args = module.build_parser().parse_args(required + ["--shared-llm"])
    configure_cli_model(args)
    assert args.ollama_model is None and args.shared_llm
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(required + ["--shared-llm", "--ollama-model", "local"])


def test_explicit_judge_model_keeps_existing_shared_judge_commands():
    args = evaluate_ragas.build_parser().parse_args([
        "--run-dir", "saved", "--output-dir", "api", "--limit", "1", "--judge-model", "gpt-4o-mini"])
    configure_cli_model(args)
    assert args.ollama_model is None and args.judge_model == "gpt-4o-mini"
    with pytest.raises(SystemExit):
        evaluate_ragas.build_parser().parse_args([
            "--run-dir", "saved", "--output-dir", "invalid", "--limit", "1",
            "--judge-model", "gpt-4o-mini", "--ollama-model", "local"])


def test_retrieval_only_remains_model_free_and_custom_local_model_is_retained():
    parser = run_rag_comparison.build_parser()
    args = parser.parse_args(["--output-dir", "retrieval", "--retrieval-only"])
    configure_cli_model(args)
    assert args.ollama_model is None
    args = parser.parse_args(["--output-dir", "local", "--ollama-model", "another-local-model"])
    configure_cli_model(args)
    assert args.ollama_model == "another-local-model"


def test_local_judge_uses_ollama_and_never_falls_back_to_shared_api(tmp_path, monkeypatch):
    from finsight.agents.narrative.rag import verdict_baseline

    class Answer(BaseModel):
        value: int

    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(evaluate_ragas, "complete", lambda *a, **kw: pytest.fail("Unexpected shared API call"))
    calls = []

    def local(**kwargs):
        calls.append(kwargs)
        return LLMResponse('{"value": 1}', 10, 5, 0., 1., kwargs["model"], "ollama", "stop")

    monkeypatch.setattr(verdict_baseline, "_complete_ollama", local)
    args = Namespace(output_dir=tmp_path, retry_errors=False, ollama_model=DEFAULT_OLLAMA_MODEL,
                     ollama_host="http://localhost:11434", ollama_context=20000, max_tokens=3000)
    transport = evaluate_ragas.Transport(args)
    assert transport.generate("evaluate", Answer).value == 1
    assert calls[0]["model"] == DEFAULT_OLLAMA_MODEL

    def unavailable(**kwargs):
        raise RuntimeError("Could not reach Ollama")

    monkeypatch.setattr(verdict_baseline, "_complete_ollama", unavailable)
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        transport.generate("another evaluation", Answer)
