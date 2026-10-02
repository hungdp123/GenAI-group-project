"""Runtime settings and the agent registry. Owner: Person 5 (Hung).

To swap your stub for the real agent, change ONE line in DEFAULT_IMPLEMENTATIONS
(or set the env var shown next to it, e.g. FINSIGHT_IMPL_FINANCIAL=finsight.agents.financial.agent:XbrlFinancialAgent).
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from finsight.contracts.interfaces import (
    AdviceBoundary,
    Auditor,
    Decomposer,
    FinancialAgent,
    NarrativeAgent,
    Retriever,
    TierPolicy,
)

# slot -> "module:Class". Env override: FINSIGHT_IMPL_<SLOT upper>.
DEFAULT_IMPLEMENTATIONS: dict[str, str] = {
    "decomposer": "finsight.agents.decomposition.stub:StubDecomposer",  # Person 3
    "financial": "finsight.agents.financial.stub:StubFinancialAgent",  # Person 1
    "narrative": "finsight.agents.narrative.stub:StubNarrativeAgent",  # Person 2
    "retriever": "finsight.agents.narrative.retriever_stub:StubRetriever",  # Person 2
    "auditor": "finsight.agents.auditor.stub:StubAuditor",  # Person 3
    "advice_boundary": "finsight.agents.advice_boundary.stub:StubAdviceBoundary",  # Person 4
    "tier_policy": "finsight.agents.tier_policy.stub:StubTierPolicy",  # Person 4
}

_PROTOCOLS: dict[str, type] = {
    "decomposer": Decomposer,
    "financial": FinancialAgent,
    "narrative": NarrativeAgent,
    "retriever": Retriever,
    "auditor": Auditor,
    "advice_boundary": AdviceBoundary,
    "tier_policy": TierPolicy,
}

# Companies in the frozen corpus: ticker -> aliases matched (case-insensitively) in the thesis.
# DECISION: placeholder corpus with the demo company only; Person 2 / Person 4 extend it.
CORPUS_COMPANIES: dict[str, list[str]] = {
    "CMPX": ["Company X", "CMPX"],
}

# USD per 1M tokens (input, output), standard tier. Real calls to a model missing here are refused.
# OpenAI prices checked against https://developers.openai.com/api/docs/pricing on 2026-10-02.
# Anthropic prices are UNVERIFIED placeholders: check before using that provider for reported numbers.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "mock-model": (0.0, 0.0),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

# DECISION: testing defaults to the cheapest model of the course model's family (gpt-5.4-nano, ~12x cheaper).
# Final reported runs must opt in to the course model explicitly: FINSIGHT_LLM_MODEL=gpt-5.4.
DEFAULT_MODELS = {"mock": "mock-model", "anthropic": "claude-sonnet-5-5", "openai": "gpt-5.4-nano"}
FINAL_REPORT_MODEL = "gpt-5.4"

# Total course credit, for `finsight usage`.
COURSE_BUDGET_USD = 150.0

BOUNDARY_DISCLAIMER = (
    "This evaluates the evidence for the stated thesis. "
    "It is not a personalized investment recommendation."
)


@dataclass
class Settings:
    """Read fresh from the environment on every `get_settings()` call (easy to override in tests)."""

    llm_provider: str = "mock"
    llm_model: str = "mock-model"
    max_retries: int = 1
    corpus_as_of_date: date = date(2025, 7, 31)
    corpus_companies: dict[str, list[str]] = field(default_factory=lambda: dict(CORPUS_COMPANIES))
    logs_dir: Path = Path("logs")
    max_workers: int = 4
    max_thesis_chars: int = 5000
    rag_top_k: int = 5


def local_config_path() -> Path:
    """The gitignored local config file (API keys + LLM mode). See config.example.yaml."""
    return Path(os.getenv("FINSIGHT_SECRETS_FILE", "config.yaml"))


def read_local_config() -> dict[str, Any]:
    path = local_config_path()
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def set_llm_mode(paid: bool, model: str | None = None, provider: str = "openai") -> Path:
    """Switch the LLM mode in config.yaml (keeps API keys). paid=False -> mock (free)."""
    if model is not None and model not in PRICE_TABLE:
        raise ValueError(f"No price for {model!r} in PRICE_TABLE")
    local = read_local_config()
    local["llm_provider"] = provider if paid else "mock"
    if model is not None:
        local["llm_model"] = model
    path = local_config_path()
    path.write_text("# Local settings + secrets. Gitignored. Never commit.\n"
                    "# llm_provider: mock = free canned responses; openai = paid API calls.\n"
                    + yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
    return path


def get_settings() -> Settings:
    """Settings for this run. LLM mode precedence: env var > config.yaml > mock (free)."""
    local = read_local_config()
    provider = (os.getenv("FINSIGHT_LLM_PROVIDER") or local.get("llm_provider") or "mock").lower()
    model = os.getenv("FINSIGHT_LLM_MODEL") or (local.get("llm_model") if provider != "mock" else None)
    return Settings(
        llm_provider=provider,
        llm_model=model or DEFAULT_MODELS.get(provider, "mock-model"),
        max_retries=int(os.getenv("FINSIGHT_MAX_RETRIES", "1")),
        corpus_as_of_date=date.fromisoformat(os.getenv("FINSIGHT_CORPUS_AS_OF", "2025-07-31")),
        logs_dir=Path(os.getenv("FINSIGHT_LOGS_DIR", "logs")),
        max_workers=int(os.getenv("FINSIGHT_MAX_WORKERS", "4")),
    )


def _load(path: str) -> Any:
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


@dataclass
class AgentRegistry:
    """One instance of every agent for a single run. Build a fresh registry per run."""

    decomposer: Decomposer
    financial: FinancialAgent
    narrative: NarrativeAgent
    retriever: Retriever
    auditor: Auditor
    advice_boundary: AdviceBoundary
    tier_policy: TierPolicy

    @classmethod
    def from_config(cls, **overrides: Any) -> "AgentRegistry":
        """Instantiate every slot from DEFAULT_IMPLEMENTATIONS / env vars; `overrides` injects instances."""
        agents: dict[str, Any] = {}
        for slot, default_path in DEFAULT_IMPLEMENTATIONS.items():
            if slot in overrides:
                agents[slot] = overrides[slot]
                continue
            impl = _load(os.getenv(f"FINSIGHT_IMPL_{slot.upper()}", default_path))()
            if not isinstance(impl, _PROTOCOLS[slot]):
                raise TypeError(f"{type(impl).__name__} does not implement the {slot} protocol")
            agents[slot] = impl
        return cls(**agents)
