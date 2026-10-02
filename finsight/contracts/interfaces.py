"""Agent interfaces (typing.Protocol). Owner: Person 5 (Hung).

Implement the Protocol for your agent and register it in `finsight/config.py`.
Rules every implementation must follow:
- Never use data published after `cutoff_date` (the orchestrator also drops it and logs a violation).
- Treat the claim and all retrieved text as DATA, never as instructions.
- Never invent financial numbers with an LLM; numbers come from deterministic tools.
- Raise exceptions freely: the orchestrator catches them and marks the sub-claim INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from finsight.contracts.schemas import (
    AdviceDecision,
    AdviceStage,
    AuditResult,
    SubClaim,
    EvidenceItem,
    KnowledgeRule,
    Verdict,
)


@runtime_checkable
class Decomposer(Protocol):
    """Person 3. Split a claim into atomic, typed sub-claims."""

    def decompose(self, claim: str, cutoff_date: date) -> list[SubClaim]: ...


@runtime_checkable
class FinancialAgent(Protocol):
    """Person 1. Deterministic XBRL-backed evidence for quantitative sub-claims."""

    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]: ...


@runtime_checkable
class NarrativeAgent(Protocol):
    """Person 2. Narrative evidence; must return both SUPPORTING and CHALLENGING items when available."""

    def gather(self, sub_claim: SubClaim, cutoff_date: date, feedback: str | None) -> list[EvidenceItem]: ...


@runtime_checkable
class Auditor(Protocol):
    """Person 3. Verify numbers by calculation and narrative sub-claims by citation."""

    def audit(self, sub_claim: SubClaim, evidence: list[EvidenceItem]) -> AuditResult: ...


@runtime_checkable
class AdviceBoundary(Protocol):
    """Person 4. Flag personalized investment advice at input and output."""

    def check(self, text: str, stage: AdviceStage) -> AdviceDecision: ...


@runtime_checkable
class TierPolicy(Protocol):
    """Person 4. Lower a proposed verdict according to source tiers. Returns (verdict, reason-if-lowered)."""

    def apply(
        self, sub_claim: SubClaim, proposed: Verdict, evidence: list[EvidenceItem], rule: KnowledgeRule
    ) -> tuple[Verdict, str | None]: ...


@runtime_checkable
class Retriever(Protocol):
    """Person 2. Top-k retrieval over the frozen corpus (used by the Basic RAG baseline)."""

    def retrieve(self, query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]: ...
