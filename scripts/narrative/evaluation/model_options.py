"""Local-by-default CLI model selection for Narrative evaluation runners."""
from __future__ import annotations

import argparse

DEFAULT_OLLAMA_MODEL = "qwen3-vl:4b-instruct"


def add_model_options(parser: argparse.ArgumentParser, *, judge: bool = False) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--ollama-model", help=f"Local model; default: {DEFAULT_OLLAMA_MODEL}")
    mode.add_argument("--shared-llm", action="store_true",
                      help="Explicitly use the shared provider/model and credentials; may incur API charges")
    if judge:
        mode.add_argument("--judge-model",
                          help="Explicitly use this model with the shared provider/credentials instead of Ollama")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-context", type=int, default=20000)


def configure_cli_model(args: argparse.Namespace) -> None:
    """Resolve CLI defaults once; an unavailable local service never falls back to an API."""
    if getattr(args, "retrieval_only", False):
        args.ollama_model = None
    elif args.shared_llm or getattr(args, "judge_model", None):
        args.ollama_model = None
    else:
        args.ollama_model = args.ollama_model or DEFAULT_OLLAMA_MODEL


def describe_cli_model(args: argparse.Namespace) -> str:
    if getattr(args, "retrieval_only", False):
        return "Retrieval only: no generation or judge model calls."
    if args.ollama_model:
        return f"Local Ollama: {args.ollama_model} at {args.ollama_host}; no shared API calls."
    from finsight.config import get_settings

    settings = get_settings()
    model = getattr(args, "judge_model", None) or settings.llm_model
    return f"Explicit shared LLM: {settings.llm_provider} / {model}."
