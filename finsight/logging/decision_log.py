"""Append-only decision log recorder + tracked agent calls + trace printer. Owner: Person 5 (Hung)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from finsight.contracts.schemas import (
    AgentCall,
    AssessmentStatus,
    AuditResult,
    DecisionLog,
    GateDecision,
    LogEvent,
    RetryRecord,
    RoutePlanEntry,
    TemporalViolation,
    Verdict,
)
from finsight.llm.usage import track_usage


def _now() -> datetime:
    return datetime.now(timezone.utc)


def tracked_call(
    agent: str, fn: Callable[..., Any], *args: Any, claim_id: str | None = None, **kwargs: Any
) -> tuple[Any, AgentCall, Exception | None]:
    """Run `fn`, never raising. Returns (result or None, AgentCall record, exception or None).

    Tokens/cost come from any LLM calls made inside `fn`; tool calls from `record_tool_call()`.
    """
    started, t0 = _now(), time.perf_counter()
    result, error = None, None
    with track_usage() as meter:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - one failed agent must not crash the run
            error = exc
    call = AgentCall(
        agent=agent, started_at=started, latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        input_tokens=meter.input_tokens, output_tokens=meter.output_tokens,
        cost_usd=round(meter.cost_usd, 6), tool_calls=meter.tool_calls,
        success=error is None, error=f"{type(error).__name__}: {error}" if error else None, claim_id=claim_id,
    )
    return result, call, error


class DecisionLogRecorder:
    """Thread-safe, append-only recorder. One per run; every orchestrator node writes through it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self.log: DecisionLog | None = None

    def start(self, request_id: str, raw_input: str, cutoff: date | None) -> None:
        with self._lock:
            self._t0 = time.perf_counter()
            self.log = DecisionLog(request_id=request_id, input=raw_input, cutoff_date=cutoff)

    def _append(self, field: str, item: Any) -> None:
        with self._lock:
            if self.log is None:
                raise RuntimeError("DecisionLogRecorder.start() must be called first")
            getattr(self.log, field).append(item)

    def event(self, node: str, message: str) -> None:
        self._append("events", LogEvent(at=_now(), node=node, message=message))

    def gate(self, gate: str, flagged: bool, reason: str | None = None) -> None:
        self._append("gate_decisions", GateDecision(gate=gate, flagged=flagged, reason=reason))

    def agent_call(self, call: AgentCall) -> None:
        self._append("agent_calls", call)

    def plan(self, entry: RoutePlanEntry) -> None:
        self._append("routing_plan", entry)

    def audit(self, result: AuditResult) -> None:
        self._append("audit_results", result)

    def retry(self, record: RetryRecord) -> None:
        self._append("retries", record)

    def temporal_violation(self, violation: TemporalViolation) -> None:
        self._append("temporal_violations", violation)

    def finish(self, status: AssessmentStatus, final_verdicts: dict[str, Verdict]) -> DecisionLog:
        """Close the log: compute totals and return a frozen copy."""
        with self._lock:
            if self.log is None:
                raise RuntimeError("DecisionLogRecorder.start() must be called first")
            log = self.log
            log.status = status
            log.final_verdicts = dict(final_verdicts)
            log.total_tokens = sum(c.input_tokens + c.output_tokens for c in log.agent_calls)
            log.total_cost_usd = round(sum(c.cost_usd for c in log.agent_calls), 6)
            # DECISION: total latency is wall-clock for the run (agents run in parallel, so summing overcounts).
            log.total_latency_ms = round((time.perf_counter() - self._t0) * 1000, 3)
            return log.model_copy(deep=True)


def write_log(log: DecisionLog, logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{log.request_id}.json"
    path.write_text(log.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_log(request_id: str, logs_dir: Path) -> DecisionLog:
    return DecisionLog.model_validate_json((logs_dir / f"{request_id}.json").read_text(encoding="utf-8"))


def format_trace(log: DecisionLog) -> str:
    """Human-readable trace for the demo and case studies."""
    lines = [
        f"Request {log.request_id}  status={log.status.value if log.status else '?'}  cutoff={log.cutoff_date}",
        f"Input: {log.input[:200]}{'...' if len(log.input) > 200 else ''}",
        f"Totals: {log.total_tokens} tokens, ${log.total_cost_usd:.4f}, {log.total_latency_ms:.0f} ms, "
        f"{len(log.agent_calls)} agent calls",
        "",
        "Gates:",
        *[f"  {g.gate:<14} {'FLAGGED' if g.flagged else 'pass'}  {g.reason or ''}" for g in log.gate_decisions],
        "Routing plan:",
        *[f"  {p.claim_id:<4} {p.claim_type.value:<28} collectors={p.collectors or '-'} "
          f"cap={p.verdict_cap.value if p.verdict_cap else '-'}" for p in log.routing_plan],
        "Agent calls:",
        *[f"  {c.started_at:%H:%M:%S.%f} {c.agent:<16} claim={c.claim_id or '-':<4} {c.latency_ms:>8.1f} ms "
          f"tok={c.input_tokens + c.output_tokens:<6} tools={c.tool_calls} {'ok' if c.success else 'ERROR ' + (c.error or '')}"
          for c in log.agent_calls],
        "Audits:",
        *[f"  {a.claim_id:<4} passed={a.passed} verdict={a.verdict.value} {'; '.join(a.failure_reasons)}"
          for a in log.audit_results],
        "Retries:",
        *[f"  {r.claim_id:<4} attempt={r.attempt} {'EXHAUSTED' if r.exhausted else 'retry'} {r.feedback or ''}"
          for r in log.retries],
        "Temporal violations:",
        *[f"  {v.claim_id:<4} dropped {v.evidence_id} published {v.publication_date} > cutoff {v.cutoff_date}"
          for v in log.temporal_violations],
        "Events:",
        *[f"  [{e.node}] {e.message}" for e in log.events],
        "Final verdicts:",
        *[f"  {cid:<4} {v.value}" for cid, v in log.final_verdicts.items()],
    ]
    return "\n".join(lines)
