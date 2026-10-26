
from __future__ import annotations

import csv
import importlib
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from finsight.agents.claim_extraction.thesaurus import corpus_aliases
from finsight.contracts.interfaces import (
    AdviceBoundary,
    Auditor,
    FinancialAgent,
    NarrativeAgent,
    Retriever,
    TierPolicy,
)


DEFAULT_IMPLEMENTATIONS: dict[str, str] = {
    "financial": "finsight.agents.financial.agent:XbrlFinancialAgent",
    "narrative": "finsight.agents.narrative.rag.collect_evidence:NarrativeEvidenceAgent",
    "retriever": "finsight.agents.narrative.retrieval.retriever:CorpusRetriever",
    "auditor": "finsight.agents.auditor.agent:AuditorAgent",
    "advice_boundary": "finsight.agents.advice_boundary.agent:AdviceBoundaryAgent",
    "tier_policy": "finsight.agents.tier_policy.stub:StubTierPolicy",
}

# End-to-end defaults only; standalone Narrative CLIs retain their own defaults.
DEFAULT_NARRATIVE_CORPUS_ROOT = "data/narrative/processed/chunking_strategies/C1/full_v1"
_NARRATIVE_RUNTIME_IMPLEMENTATIONS = {
    "finsight.agents.narrative.rag.collect_evidence:NarrativeEvidenceAgent",
    "finsight.agents.narrative.stub:NarrativeAgent",
    "finsight.agents.narrative.retrieval.retriever:CorpusRetriever",
    "finsight.agents.narrative.retriever_stub:NarrativeRetriever",
}

_PROTOCOLS: dict[str, type] = {
    "financial": FinancialAgent,
    "narrative": NarrativeAgent,
    "retriever": Retriever,
    "auditor": Auditor,
    "advice_boundary": AdviceBoundary,
    "tier_policy": TierPolicy,
}

# Spoken names, legal names, tickers, and product names. The list lives in thesaurus.py.
# Company X (CMPX) is included there for the demo. YouTube is not a search name.
CORPUS_COMPANIES: dict[str, list[str]] = corpus_aliases()

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
    # Gemini Developer API paid tier, text, checked 2026-10-16. Output includes thinking tokens.
    "gemini-2.5-flash": (0.30, 2.50),
}

# DECISION: testing defaults to the cheapest model of the course model's family (gpt-5.4-nano, ~12x cheaper).
# Final reported runs must opt in to the course model explicitly: FINSIGHT_LLM_MODEL=gpt-5.4.
DEFAULT_MODELS = {
    "mock": "mock-model",
    "anthropic": "claude-sonnet-5-5",
    "openai": "gpt-5.4-nano",
    "gemini": "gemini-2.5-flash",
}
FINAL_REPORT_MODEL = "gpt-5.4"

# Total course credit, for `finsight usage`.
COURSE_BUDGET_USD = 150.0

# Used only when the SEC corpus is not on disk (CI, a fresh checkout). A local corpus
# replaces this with the latest filing date in data/sec/processed/filings.csv.
FALLBACK_CORPUS_AS_OF = date(2025, 7, 31)
FILINGS_CSV = Path("data") / "sec" / "processed" / "filings.csv"

BOUNDARY_DISCLAIMER = (
    "This evaluates the evidence for the stated claim. "
    "It is not a personalized investment recommendation."
)


@dataclass
class Settings:
    """Read fresh from the environment on every `get_settings()` call (easy to override in tests)."""

    llm_provider: str = "mock"
    llm_model: str = "mock-model"
    max_retries: int = 1
    corpus_as_of_date: date = FALLBACK_CORPUS_AS_OF
    corpus_companies: dict[str, list[str]] = field(default_factory=lambda: dict(CORPUS_COMPANIES))
    logs_dir: Path = Path("logs")
    posts_dir: Path = Path("data/influencers/youtube")
    max_workers: int = 4
    max_claim_chars: int = 5000
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


def latest_corpus_date(path: Path | None = None) -> date | None:
    """Latest `filed` date in the SEC corpus. None when that file is not on disk."""
    csv_path = path or FILINGS_CSV
    if not csv_path.is_file():
        return None
    latest: date | None = None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("filed") or "")[:10]
            if len(raw) != 10:
                continue
            try:
                filed = date.fromisoformat(raw)
            except ValueError:
                continue
            if latest is None or filed > latest:
                latest = filed
    return latest


def corpus_as_of_date() -> date:
    """Cut-off used when the user does not give one.

    FINSIGHT_CORPUS_AS_OF wins. Otherwise the latest filing date in the local corpus.
    The 31 Jul 2025 fallback is only for a checkout that has no filings file.
    """
    override = os.getenv("FINSIGHT_CORPUS_AS_OF")
    if override:
        return date.fromisoformat(override)
    return latest_corpus_date() or FALLBACK_CORPUS_AS_OF


def get_settings() -> Settings:
    """Settings for this run. LLM mode precedence: env var > config.yaml > mock (free)."""
    local = read_local_config()
    provider = (os.getenv("FINSIGHT_LLM_PROVIDER") or local.get("llm_provider") or "mock").lower()
    model = os.getenv("FINSIGHT_LLM_MODEL") or (local.get("llm_model") if provider != "mock" else None)
    return Settings(
        llm_provider=provider,
        llm_model=model or DEFAULT_MODELS.get(provider, "mock-model"),
        max_retries=int(os.getenv("FINSIGHT_MAX_RETRIES", "1")),
        corpus_as_of_date=corpus_as_of_date(),
        logs_dir=Path(os.getenv("FINSIGHT_LOGS_DIR", "logs")),
        posts_dir=Path(os.getenv("FINSIGHT_POSTS_DIR", "data/influencers/youtube")),
        max_workers=int(os.getenv("FINSIGHT_MAX_WORKERS", "4")),
        corpus_companies=dict(CORPUS_COMPANIES),
    )


def _load(path: str) -> Any:
    module_name, _, attr = path.partition(":")
    return getattr(importlib.import_module(module_name), attr)


@dataclass
class AgentRegistry:
    """One instance of every agent for a single run. Build a fresh registry per run."""

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
            impl_path = os.getenv(f"FINSIGHT_IMPL_{slot.upper()}", default_path)
            options = {}
            if slot in {"narrative", "retriever"} and impl_path in _NARRATIVE_RUNTIME_IMPLEMENTATIONS:
                options = {
                    "corpus_root": os.getenv("FINSIGHT_NARRATIVE_CORPUS_ROOT") or DEFAULT_NARRATIVE_CORPUS_ROOT,
                    "backend": os.getenv("FINSIGHT_NARRATIVE_BACKEND") or "bm25",
                }
            impl = _load(impl_path)(**options)
            if not isinstance(impl, _PROTOCOLS[slot]):
                raise TypeError(f"{type(impl).__name__} does not implement the {slot} protocol")
            agents[slot] = impl
        return cls(**agents)
