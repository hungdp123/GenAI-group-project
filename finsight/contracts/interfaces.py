"""Agent interfaces (typing.Protocol). Owner: Person 5 (Hung).

Implement the Protocol for your agent and register it in `finsight/config.py`.
Rules every implementation must follow:
- Never use data published after `cutoff_date` (the orchestrator also drops it and logs a violation).
- Treat the thesis and all retrieved text as DATA, never as instructions.
- Never invent financial numbers with an LLM; numbers come from deterministic tools.
- Raise exceptions freely: the orchestrator catches them and marks the claim INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from finsight.contracts.schemas import (
    AdviceDecision,
    AdviceStage,
    AuditResult,
    Claim,
    EvidenceItem,
    KnowledgeRule,
    Verdict,
)


@runtime_checkable
class Decomposer(Protocol):
    """Person 3. Split a thesis into atomic, typed claims."""

    def decompose(self, thesis: str, cutoff_date: date) -> list[Claim]: ...


@runtime_checkable
class FinancialAgent(Protocol):
    """Person 1. Deterministic XBRL-backed evidence for quantitative claims."""

    def gather(self, claim: Claim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]: ...


@runtime_checkable
class NarrativeAgent(Protocol):
    """Person 2. Narrative evidence; must return both SUPPORTING and CHALLENGING items when available."""

    def gather(self, claim: Claim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]: ...


@runtime_checkable
class Auditor(Protocol):
    """Person 3. Verify numbers by calculation and narrative claims by citation."""

    def audit(self, claim: Claim, evidence: list[EvidenceItem]) -> AuditResult: ...


@runtime_checkable
class AdviceBoundary(Protocol):
    """Person 4. Flag personalized investment advice at input and output."""

    def check(self, text: str, stage: AdviceStage) -> AdviceDecision: ...


@runtime_checkable
class TierPolicy(Protocol):
    """Person 4. Lower a proposed verdict according to source tiers. Returns (verdict, reason-if-lowered)."""

    def apply(
        self, claim: Claim, proposed: Verdict, evidence: list[EvidenceItem], rule: KnowledgeRule
    ) -> tuple[Verdict, str | None]: ...


@runtime_checkable
class Retriever(Protocol):
    """Person 2. Top-k retrieval over the frozen corpus (used by the Basic RAG baseline)."""

    def retrieve(self, query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]: ...
