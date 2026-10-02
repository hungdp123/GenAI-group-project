"""Shared Pydantic contracts for every FinSight component.

Owner: Person 5 (Hung). Every agent imports its input/output types from here.
Do NOT change this file without telling Hung: a field change breaks every teammate.
Bump SCHEMA_VERSION whenever a field is added, removed or renamed.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# 0.2: NEEDS_CLARIFICATION status + FinalAssessment.clarification_question, KnowledgeRule.retry_collectors,
#      "clarification" gate.
SCHEMA_VERSION = "0.2"


class _Strict(BaseModel):
    """Base for all contracts: unknown fields are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SourceTier(str, Enum):
    """Tier 1 = SEC filings / XBRL, Tier 2 = earnings releases / IR, Tier 3 = frozen news."""

    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"

    @property
    def rank(self) -> int:
        """Lower is more authoritative (TIER_1 -> 1)."""
        return int(self.value[-1])


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

    @property
    def support_rank(self) -> int:
        """How much the verdict endorses the claim. Used for caps ("never raise above X").

        # DECISION: CONTRADICTED ranks lowest so a cap never turns a contradiction
        # into something else; INSUFFICIENT_EVIDENCE sits just above it.
        """
        return {
            Verdict.SUPPORTED: 3,
            Verdict.PARTIALLY_SUPPORTED: 2,
            Verdict.INSUFFICIENT_EVIDENCE: 1,
            Verdict.CONTRADICTED: 0,
        }[self]

    @property
    def is_positive(self) -> bool:
        return self in (Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED)


class ClaimType(str, Enum):
    HISTORICAL_GROWTH = "HISTORICAL_GROWTH"
    PROFITABILITY_CASH_LEVERAGE = "PROFITABILITY_CASH_LEVERAGE"
    DEMAND_ATTRIBUTION = "DEMAND_ATTRIBUTION"
    CONTINUATION = "CONTINUATION"
    COMPARATIVE = "COMPARATIVE"
    BUSINESS_RISK = "BUSINESS_RISK"
    PERSONALIZED_ADVICE = "PERSONALIZED_ADVICE"
    OTHER = "OTHER"


class Stance(str, Enum):
    SUPPORTING = "SUPPORTING"
    CHALLENGING = "CHALLENGING"
    NEUTRAL = "NEUTRAL"


class AdviceStage(str, Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class AssessmentStatus(str, Enum):
    COMPLETED = "COMPLETED"
    REFUSED = "REFUSED"
    FAILED = "FAILED"
    # The thesis is unclear or missing information: show `clarification_question` to the user and
    # resubmit with their answer (e.g. run_finsight(thesis, company_ticker="CMPX")).
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


CollectorName = Literal["financial", "narrative"]


# ---------------------------------------------------------------------------
# Request / claims / evidence
# ---------------------------------------------------------------------------


class ThesisRequest(_Strict):
    request_id: str
    raw_thesis: str
    company_ticker: str | None = None
    cutoff_date: date
    cutoff_was_defaulted: bool = False


class Claim(_Strict):
    claim_id: str
    text: str
    claim_type: ClaimType
    company_ticker: str | None = None
    is_quantitative: bool = False


class EvidenceItem(_Strict):
    evidence_id: str
    claim_id: str
    stance: Stance
    content: str
    source_tier: SourceTier
    doc_type: str = Field(description="e.g. 10-K, 10-Q, earnings_release, ir_remarks, news")
    company: str
    period: str | None = None
    publication_date: date
    url: str | None = None
    citation_span: str | None = None
    produced_by: str = Field(description="Name of the agent that produced this item")
    computed_value: float | None = None
    metric_name: str | None = None


# ---------------------------------------------------------------------------
# Audit / policy / advice
# ---------------------------------------------------------------------------


class NumericCheck(_Strict):
    metric_name: str | None = None
    stated: float | None = None
    computed: float | None = None
    matches: bool


class CitationCheck(_Strict):
    evidence_id: str
    entails: bool


class AuditResult(_Strict):
    claim_id: str
    passed: bool
    verdict: Verdict
    failure_reasons: list[str] = Field(default_factory=list)
    numeric_check: NumericCheck | None = None
    citation_check: CitationCheck | None = None
    feedback_for_retry: str | None = None


class AdviceDecision(_Strict):
    stage: AdviceStage
    flagged: bool
    reason: str | None = None


class KnowledgeRule(_Strict):
    """One row of the knowledge layer (knowledge/knowledge_layer.yaml). Content owned by Person 4."""

    claim_type: ClaimType
    collectors: list[CollectorName] = Field(default_factory=list)
    highest_verdict: Verdict | None = Field(
        default=None, description="Verdict cap. None only when refuse=True."
    )
    tier_requirement: list[SourceTier] = Field(
        default_factory=list, description="Tiers whose SUPPORTING evidence may justify a positive verdict."
    )
    counter_evidence_only: bool = Field(
        default=False, description="Evidence for this type can only block SUPPORTED, never add support."
    )
    refuse: bool = Field(default=False, description="Do not analyze; refuse the request.")
    retry_collectors: list[CollectorName] = Field(
        default_factory=list,
        description="Extra collectors the orchestrator adds when it re-plans a claim for its retry.",
    )
    note: str | None = None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class ClaimResult(_Strict):
    claim: Claim
    verdict: Verdict
    highest_tier_used: SourceTier | None = None
    supporting: list[EvidenceItem] = Field(default_factory=list)
    challenging: list[EvidenceItem] = Field(default_factory=list)
    rationale: str
    retries_used: int = 0
    capped_by: str | None = None


class FinalAssessment(_Strict):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    status: AssessmentStatus
    overall_verdict: Verdict | None = None
    claim_results: list[ClaimResult] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    as_of_date: date | None = None
    # DECISION: added so a defaulted cut-off is stated in the output, not only the log.
    cutoff_was_defaulted: bool = False
    boundary_disclaimer: str
    refusal_reason: str | None = None
    clarification_question: str | None = Field(
        default=None, description="Set when status is NEEDS_CLARIFICATION: ask the user this.")


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------


class AgentCall(_Strict):
    agent: str
    started_at: datetime
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    success: bool
    error: str | None = None
    claim_id: str | None = None  # DECISION: added so per-claim cost can be traced


class GateDecision(_Strict):
    """Any gate in the pipeline: advice (input/output), scope, or a clarification request."""

    gate: Literal["advice_input", "scope", "clarification", "advice_output"]
    flagged: bool
    reason: str | None = None


class RoutePlanEntry(_Strict):
    claim_id: str
    claim_type: ClaimType
    collectors: list[CollectorName]
    verdict_cap: Verdict | None
    tier_requirement: list[SourceTier]
    note: str | None = None


class RetryRecord(_Strict):
    claim_id: str
    attempt: int
    feedback: str | None = None
    exhausted: bool = Field(default=False, description="True when no retries remain and the claim is downgraded")


class TemporalViolation(_Strict):
    claim_id: str
    evidence_id: str
    publication_date: date
    cutoff_date: date
    produced_by: str


class LogEvent(_Strict):
    at: datetime
    node: str
    message: str


class DecisionLog(_Strict):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    input: str
    cutoff_date: date | None = None
    gate_decisions: list[GateDecision] = Field(default_factory=list)
    agent_calls: list[AgentCall] = Field(default_factory=list)
    routing_plan: list[RoutePlanEntry] = Field(default_factory=list)
    audit_results: list[AuditResult] = Field(default_factory=list)
    retries: list[RetryRecord] = Field(default_factory=list)
    temporal_violations: list[TemporalViolation] = Field(default_factory=list)
    events: list[LogEvent] = Field(default_factory=list)
    final_verdicts: dict[str, Verdict] = Field(default_factory=dict)
    status: AssessmentStatus | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class FinSightState(_Strict):
    """State carried through the orchestrator graph. Nodes return partial updates.

    The live decision log is kept by a thread-safe recorder in the run context and
    copied into `decision_log` by the finalize node.
    """

    # inputs
    raw_input: str = ""
    cutoff_input: date | None = None
    company_hint: str | None = None
    # progress
    request: ThesisRequest | None = None
    status: AssessmentStatus | None = None  # set only when the run stops early
    refusal_reason: str | None = None
    clarification_question: str | None = None  # set by a gate; the clarification node then stops the run
    advice_decisions: list[AdviceDecision] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    routing_plan: dict[str, RoutePlanEntry] = Field(default_factory=dict)
    pending_claim_ids: list[str] = Field(default_factory=list)
    feedback: dict[str, str] = Field(default_factory=dict)
    retries_used: dict[str, int] = Field(default_factory=dict)
    evidence: dict[str, list[EvidenceItem]] = Field(default_factory=dict)
    audits: dict[str, AuditResult] = Field(default_factory=dict)
    insufficient_reasons: dict[str, str] = Field(default_factory=dict)
    tier_shortfalls: dict[str, str] = Field(default_factory=dict)  # claims whose evidence tier is not enough
    final_verdicts: dict[str, Verdict] = Field(default_factory=dict)
    capped_by: dict[str, str] = Field(default_factory=dict)
    # outputs
    final: FinalAssessment | None = None
    decision_log: DecisionLog | None = None
