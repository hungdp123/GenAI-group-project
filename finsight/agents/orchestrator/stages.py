"""Stage-by-stage view of a run (`finsight run --stages`). Owner: Person 5 (Hung).

`StageReporter` is passed to `execute(..., on_stage=...)` and receives every graph node's
state update as it happens. It only reads; it never changes the run.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Callable
from typing import Any

from finsight.logging.decision_log import DecisionLogRecorder

# Diagram role of each node, shown next to its name.
ROLES = {
    "intake": "coordination", "advice_gate_input": "guardrail", "scope_check": "guardrail",
    "clarification": "clarification", "collect_posts": "claim source", "extract_claims": "claim extraction",
    "plan": "coordination",
    "gather_evidence": "specialists", "audit": "verification", "retry_or_accept": "retry check",
    "apply_tier_policy": "verification", "assemble": "coordination", "advice_gate_output": "guardrail",
    "finalize": "coordination", "record": "record",
}


def _tier(t) -> str:
    return t.value.replace("TIER_", "T")


class StageReporter:
    """Formats each stage's update into a few readable lines and hands them to `emit`."""

    def __init__(self, emit: Callable[[str], None] = print, width: int = 100) -> None:
        self.emit = emit
        self.width = width
        self.step = 0
        self._calls_seen = 0
        self._active: set[str] | None = None  # sub-claims gathered / audited in the current round

    def __call__(self, node: str, update: dict[str, Any], recorder: DecisionLogRecorder) -> None:
        self.step += 1
        log = recorder.log if recorder is not None else None
        calls = log.agent_calls[self._calls_seen:] if log else []
        self._calls_seen += len(calls)
        lines = getattr(self, f"_{node}", self._default)(update)
        head = f"[{self.step:>2}] {node:<19} ({ROLES.get(node, '')})"
        self.emit(head)
        for line in lines:
            for i, part in enumerate(textwrap.wrap(line, self.width) or [""]):
                self.emit(("     " if i == 0 else "          ") + part)
        if calls:
            names = [c.agent + (f"({c.sub_claim_id})" if c.sub_claim_id else "") + ("" if c.success else " FAILED")
                     for c in calls]
            self.emit(f"     calls: {', '.join(names)}")

    # -- one formatter per node ------------------------------------------------------------

    def _default(self, update: dict) -> list[str]:
        return []

    def _stopped(self, update: dict) -> list[str]:
        if update.get("status") is not None:
            return [f"STOP -> {update['status'].value}: {update.get('refusal_reason') or ''}".rstrip(": ")]
        if update.get("clarification_question"):
            return [f"needs clarification: {update['clarification_question']}"]
        return []

    def _intake(self, u: dict) -> list[str]:
        r = u["request"]
        src = ("no cut-off given: posts up to the corpus as-of date; a stored post's date becomes the evidence "
               "cut-off" if r.cutoff_was_defaulted else "from input")
        return [f"request {r.request_id}, cut-off {r.cutoff_date} ({src})", *self._stopped(u)]

    def _advice_gate_input(self, u: dict) -> list[str]:
        flagged = bool(u.get("advice_decisions")) and u["advice_decisions"][-1].flagged
        return self._stopped(u) or ["not personalized advice -> continue" if not flagged else "flagged"]

    def _scope_check(self, u: dict) -> list[str]:
        stopped = self._stopped(u)
        if stopped:
            return stopped
        ticker = u["request"].company_ticker
        return [f"company {ticker} is in the corpus -> continue"]

    def _clarification(self, u: dict) -> list[str]:
        return ["run stops here and returns the question to the user"]

    def _record(self, u: dict) -> list[str]:
        return [f"checking post {u.get('index')} of {u.get('count')}: {u.get('post_id')}"]

    def _collect_posts(self, u: dict) -> list[str]:
        if u.get("status") is not None or u.get("clarification_question"):
            return self._stopped(u)
        request = u.get("request")
        if request is None:
            return []
        why = {"post_date": "the post date", "corpus_default": "corpus as-of date; the post has no date",
               "user": "your date"}[request.cutoff_source]
        return [f"stored post loaded, evidence cut-off {request.cutoff_date} ({why})"]

    def _extract_claims(self, u: dict) -> list[str]:
        lines = [f"{c.sub_claim_id:<4} {c.sub_claim_type.value:<28} {c.text}" for c in u.get("sub_claims", [])]
        label = u.get("checked_post") or ""
        lines += [f"note: {m.group(0).strip()}" for m in re.finditer(r"You said .+? is what was checked\.", label)]
        return lines + self._stopped(u)

    def _plan(self, u: dict) -> list[str]:
        self._active = set(u.get("pending_sub_claim_ids", []))
        lines = []
        for sid, p in u["routing_plan"].items():
            if sid in u.get("insufficient_reasons", {}):
                lines.append(f"{sid:<4} no collector -> INSUFFICIENT_EVIDENCE ({u['insufficient_reasons'][sid]})")
            else:
                tiers = "/".join(_tier(t) for t in p.tier_requirement) or "-"
                cap = p.verdict_cap.value if p.verdict_cap else "-"
                lines.append(f"{sid:<4} -> {'+'.join(p.collectors)}, needs {tiers}, cap {cap}")
        return lines

    def _gather_evidence(self, u: dict) -> list[str]:
        lines = []
        reports = u.get("narrative_reports") or {}
        evidence = u.get("evidence") or {}
        for sid in list(dict.fromkeys([*evidence, *reports])):
            if self._active is not None and sid not in self._active:
                continue
            report = reports.get(sid)
            if report:
                lines.extend(self._narrative_report_lines(sid, report))
            items = evidence.get(sid) or []
            if not items:
                lines.append(f"{sid:<4} no evidence")
            for e in items:
                period = f"  [period from {e.period_source}]" if getattr(e, "period_source", None) else ""
                lines.append(f"{sid:<4} {_tier(e.source_tier)} {e.content[:110]}{period}")
        for sid, why in u.get("insufficient_reasons", {}).items():
            if "failed" in why:
                lines.append(f"{sid:<4} {why}")
        return lines

    @staticmethod
    def _narrative_report_lines(sid: str, report: dict) -> list[str]:
        stance = report.get("stance") or "not produced"
        code = report.get("stance_reason_code")
        label = f"{stance} ({code})" if code else str(stance)
        lines = [f"{sid:<4} narrative stance {label}"]
        reason = report.get("stance_reason")
        if reason:
            lines.append(f"{sid:<4} narrative reason {reason}")
        for sentence in report.get("summary") or []:
            lines.append(f"{sid:<4} summary {sentence}")
        if report.get("status") == "FAILED" and not report.get("stance"):
            lines.append(f"{sid:<4} narrative preparation failed; retrieved passages were kept")
        return lines

    def _audit(self, u: dict) -> list[str]:
        lines = []
        for sid, a in u.get("audits", {}).items():
            if self._active is not None and sid not in self._active:
                continue
            state = "pass" if a.passed else "FAIL"
            extra = f" ({'; '.join(a.failure_reasons)})" if a.failure_reasons else ""
            if a.numeric_check and a.numeric_check.computed is not None:
                from finsight.agents.orchestrator.render import fmt_numeric_check
                extra += f" numeric {fmt_numeric_check(a.numeric_check)}"
            lines.append(f"{sid:<4} {state}, proposes {a.verdict.value}{extra}")
            lines.extend(self._stance_lines(sid, a))
        return lines

    @staticmethod
    def _stance_lines(sid: str, audit) -> list[str]:
        """The auditor's stance for each passage. Gather only lists the text; this is the judgment."""
        reasons = {}
        for reason in audit.failure_reasons:
            evidence_id, _, why = reason.partition(": ")
            if why:
                reasons[evidence_id] = why
        lines = []
        for evidence_id, stance in audit.evidence_stances.items():
            why = reasons.get(evidence_id)
            note = f" — {why}" if why else ""
            lines.append(f"{sid:<4} stance {stance.value}  {evidence_id}{note}")
        return lines

    def _retry_or_accept(self, u: dict) -> list[str]:
        retry = u.get("pending_sub_claim_ids", [])
        self._active = set(retry)
        gave_up = [f"{sid:<4} GAVE UP -> INSUFFICIENT_EVIDENCE: {why}"
                   for sid, why in u.get("insufficient_reasons", {}).items()
                   if "(still failing after" in why or "(no second look" in why or "the cap of" in why]
        if not retry:
            if gave_up:
                return [*gave_up, "no more retries -> continue"]
            return ["all sub-claims accepted (audit passed, source tier enough) -> continue"]
        return [f"{sid:<4} RETRY with collectors {'+'.join(u['routing_plan'][sid].collectors)}; "
                f"feedback: {u['feedback'].get(sid, '')}" for sid in retry]

    def _apply_tier_policy(self, u: dict) -> list[str]:
        capped = u.get("capped_by", {})
        return [f"{sid:<4} {v.value}" + (f"  (capped: {capped[sid]})" if sid in capped else "")
                for sid, v in u.get("final_verdicts", {}).items()]

    def _assemble(self, u: dict) -> list[str]:
        fa = u["final"]
        return [f"overall {fa.overall_verdict.value if fa.overall_verdict else '-'}",
                *[f"gap: {g}" for g in fa.evidence_gaps]]

    def _advice_gate_output(self, u: dict) -> list[str]:
        return self._stopped(u) or ["no advice language -> release"]

    def _finalize(self, u: dict) -> list[str]:
        fa, log = u["final"], u["decision_log"]
        return [f"status {fa.status.value}; {len(log.agent_calls)} agent calls, "
                f"{sum(c.tool_calls for c in log.agent_calls)} tool calls, "
                f"{sum(not r.exhausted for r in log.retries)} retries, "
                f"{log.total_latency_ms:.0f} ms"]
