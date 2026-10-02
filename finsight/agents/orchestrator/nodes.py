"""LangGraph nodes for the FinSight orchestrator. Owner: Person 5 (Hung).

Each node takes FinSightState and returns a partial update dict. Agent calls go through
`Nodes._call`, which records latency/tokens/cost/tool calls and never raises.

Safety rules enforced here:
- The thesis and all retrieved text are DATA. The orchestrator never builds prompts from them
  and never reads verdicts from them; verdicts come only from audit + tier policy + guards.
- The orchestrator never computes or rewrites financial numbers.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from finsight.agents.orchestrator import routing
from finsight.agents.orchestrator.aggregation import highest_tier, overall_verdict
from finsight.agents.orchestrator.render import render_assessment
from finsight.config import BOUNDARY_DISCLAIMER, AgentRegistry, Settings
from finsight.contracts.schemas import (
    AdviceDecision,
    AdviceStage,
    AssessmentStatus,
    AuditResult,
    Claim,
    ClaimResult,
    ClaimType,
    EvidenceItem,
    FinalAssessment,
    FinSightState,
    RetryRecord,
    Stance,
    TemporalViolation,
    ThesisRequest,
    Verdict,
)
from finsight.knowledge.loader import KnowledgeLayer
from finsight.logging.decision_log import DecisionLogRecorder, tracked_call, write_log


@dataclass
class RunContext:
    """Per-run dependencies. Not part of the graph state (agents are not serialisable data)."""

    registry: AgentRegistry
    settings: Settings
    kb: KnowledgeLayer
    recorder: DecisionLogRecorder = field(default_factory=DecisionLogRecorder)


def new_request_id() -> str:
    return f"fs-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _stop(status: AssessmentStatus, reason: str) -> dict[str, Any]:
    return {"status": status, "refusal_reason": reason}


class Nodes:
    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    # -- helpers ------------------------------------------------------------

    @property
    def rec(self) -> DecisionLogRecorder:
        return self.ctx.recorder

    def _call(self, agent: str, fn: Any, *args: Any, claim_id: str | None = None) -> tuple[Any, Exception | None]:
        result, call, error = tracked_call(agent, fn, *args, claim_id=claim_id)
        self.rec.agent_call(call)
        return result, error

    def _pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=max(1, self.ctx.settings.max_workers))

    # -- 1. intake ----------------------------------------------------------

    def intake(self, state: FinSightState) -> dict[str, Any]:
        settings = self.ctx.settings
        thesis = state.raw_input.strip()
        cutoff, defaulted = routing.resolve_cutoff(state.cutoff_input, thesis, settings.corpus_as_of_date)
        request = ThesisRequest(request_id=new_request_id(), raw_thesis=thesis, company_ticker=state.company_hint,
                                cutoff_date=cutoff, cutoff_was_defaulted=defaulted)
        self.rec.start(request.request_id, state.raw_input, cutoff)
        self.rec.event("intake", f"cutoff={cutoff} defaulted={defaulted}")
        update: dict[str, Any] = {"request": request}
        # DECISION: invalid input is REFUSED (the user's request cannot be served), not FAILED.
        if not thesis:
            update |= _stop(AssessmentStatus.REFUSED, "Empty thesis.")
        elif len(thesis) > settings.max_thesis_chars:
            update |= _stop(AssessmentStatus.REFUSED, f"Thesis longer than {settings.max_thesis_chars} characters.")
        return update

    # -- 2. advice gate (input) --------------------------------------------

    def advice_gate_input(self, state: FinSightState) -> dict[str, Any]:
        decision, error = self._call("advice_boundary", self.ctx.registry.advice_boundary.check,
                                     state.request.raw_thesis, AdviceStage.INPUT)
        if error or not isinstance(decision, AdviceDecision):
            # DECISION: fail closed - if the advice gate is down we do not analyze.
            self.rec.gate("advice_input", True, f"advice boundary unavailable: {error}")
            return _stop(AssessmentStatus.FAILED, "Advice boundary check failed; request not analyzed.")
        self.rec.gate("advice_input", decision.flagged, decision.reason)
        update: dict[str, Any] = {"advice_decisions": [*state.advice_decisions, decision]}
        if decision.flagged:
            update |= _stop(AssessmentStatus.REFUSED, decision.reason or "Personalized investment advice request.")
        return update

    # -- 3. scope check -----------------------------------------------------

    def scope_check(self, state: FinSightState) -> dict[str, Any]:
        thesis = state.request.raw_thesis
        ticker = routing.detect_company(thesis, self.ctx.settings.corpus_companies, state.request.company_ticker)
        reason = routing.scope_refusal(thesis, ticker)
        self.rec.gate("scope", reason is not None, reason or f"company={ticker}")
        request = state.request.model_copy(update={"company_ticker": ticker})
        update: dict[str, Any] = {"request": request}
        if reason:
            update |= _stop(AssessmentStatus.REFUSED, reason)
        return update

    # -- 4. decompose -------------------------------------------------------

    def decompose(self, state: FinSightState) -> dict[str, Any]:
        req = state.request
        claims, error = self._call("decomposer", self.ctx.registry.decomposer.decompose,
                                   req.raw_thesis, req.cutoff_date)
        if error or not isinstance(claims, list) or not all(isinstance(c, Claim) for c in claims):
            return _stop(AssessmentStatus.FAILED, f"Thesis decomposition failed: {error or 'invalid output'}")
        if len({c.claim_id for c in claims}) != len(claims):
            # DECISION: renumber duplicate claim_ids instead of failing the run.
            claims = [c.model_copy(update={"claim_id": f"c{i + 1}"}) for i, c in enumerate(claims)]
            self.rec.event("decompose", "duplicate claim_ids from decomposer; renumbered")
        claims = [c if c.company_ticker else c.model_copy(update={"company_ticker": req.company_ticker})
                  for c in claims]
        self.rec.event("decompose", f"{len(claims)} claims: " + ", ".join(
            f"{c.claim_id}={c.claim_type.value}" for c in claims))
        refusing = [c for c in claims if c.claim_type is ClaimType.PERSONALIZED_ADVICE
                    or self.ctx.kb.rule_for(c.claim_type).refuse]
        update: dict[str, Any] = {"claims": claims}
        if refusing:
            note = self.ctx.kb.rule_for(refusing[0].claim_type).note or "Personalized advice is out of scope."
            update |= _stop(AssessmentStatus.REFUSED, f"Claim {refusing[0].claim_id}: {note}")
        return update

    # -- 5. plan ------------------------------------------------------------

    def plan(self, state: FinSightState) -> dict[str, Any]:
        plan, pending, insufficient = {}, [], dict(state.insufficient_reasons)
        for claim in state.claims:
            rule = self.ctx.kb.rule_for(claim.claim_type)
            entry = routing.plan_entry(claim, rule)
            plan[claim.claim_id] = entry
            self.rec.plan(entry)
            if entry.collectors:
                pending.append(claim.claim_id)
            else:
                insufficient[claim.claim_id] = rule.note or f"No evidence collector for {claim.claim_type.value}."
        return {"routing_plan": plan, "pending_claim_ids": pending, "insufficient_reasons": insufficient,
                "retries_used": {c.claim_id: 0 for c in state.claims}}

    # -- 6. gather evidence (claims in parallel) ----------------------------

    def gather_evidence(self, state: FinSightState) -> dict[str, Any]:
        req = state.request
        claims = {c.claim_id: c for c in state.claims}

        def collect(claim_id: str) -> tuple[str, list[EvidenceItem], list[str]]:
            items, errors = [], []
            for collector in state.routing_plan[claim_id].collectors:
                agent = getattr(self.ctx.registry, collector)
                result, error = self._call(collector, agent.gather, claims[claim_id], req.cutoff_date,
                                           state.feedback.get(claim_id), claim_id=claim_id)
                if error is None and not (isinstance(result, list)
                                          and all(isinstance(i, EvidenceItem) for i in result)):
                    error = TypeError("collector did not return list[EvidenceItem]")
                if error:
                    errors.append(f"{collector} agent failed ({error})")
                else:
                    items.extend(result)
            return claim_id, items, errors

        with self._pool() as pool:
            results = list(pool.map(collect, state.pending_claim_ids))

        evidence, insufficient = dict(state.evidence), dict(state.insufficient_reasons)
        for claim_id, items, errors in results:
            if errors:
                # DECISION: any collector failure makes the claim INSUFFICIENT_EVIDENCE (no partial evidence).
                insufficient[claim_id] = "; ".join(errors)
                evidence[claim_id] = []
            else:
                evidence[claim_id] = self._screen(claim_id, items, req)
        pending = [c for c in state.pending_claim_ids if c not in insufficient]
        return {"evidence": evidence, "insufficient_reasons": insufficient, "pending_claim_ids": pending}

    def _screen(self, claim_id: str, items: list[EvidenceItem], req: ThesisRequest) -> list[EvidenceItem]:
        """Hard temporal filter, thesis-is-not-evidence filter, claim_id consistency."""
        kept, dropped = routing.split_temporal(items, req.cutoff_date)
        for item in dropped:
            self.rec.temporal_violation(TemporalViolation(
                claim_id=claim_id, evidence_id=item.evidence_id, publication_date=item.publication_date,
                cutoff_date=req.cutoff_date, produced_by=item.produced_by))
        screened = []
        for item in kept:
            if routing.is_thesis_echo(item, req.raw_thesis):
                self.rec.event("gather_evidence", f"{claim_id}: dropped {item.evidence_id} (thesis text is not evidence)")
            elif item.claim_id != claim_id:
                self.rec.event("gather_evidence", f"{claim_id}: dropped {item.evidence_id} (claim_id {item.claim_id})")
            else:
                screened.append(item)
        return screened

    # -- 7. audit -----------------------------------------------------------

    def audit(self, state: FinSightState) -> dict[str, Any]:
        claims = {c.claim_id: c for c in state.claims}

        def run(claim_id: str) -> tuple[str, AuditResult | None, Exception | None]:
            result, error = self._call("auditor", self.ctx.registry.auditor.audit, claims[claim_id],
                                       state.evidence.get(claim_id, []), claim_id=claim_id)
            if error is None and not (isinstance(result, AuditResult) and result.claim_id == claim_id):
                error = TypeError("auditor did not return an AuditResult for this claim")
            return claim_id, result, error

        with self._pool() as pool:
            results = list(pool.map(run, state.pending_claim_ids))

        audits, insufficient = dict(state.audits), dict(state.insufficient_reasons)
        for claim_id, result, error in results:
            if error:
                insufficient[claim_id] = f"auditor failed ({error})"
                continue
            audits[claim_id] = result
            self.rec.audit(result)
        return {"audits": audits, "insufficient_reasons": insufficient}

    # -- 8. retry or accept -------------------------------------------------

    def retry_or_accept(self, state: FinSightState) -> dict[str, Any]:
        max_retries = self.ctx.settings.max_retries
        retry, feedback = [], dict(state.feedback)
        retries_used, insufficient = dict(state.retries_used), dict(state.insufficient_reasons)
        for claim_id in state.pending_claim_ids:
            audit = state.audits.get(claim_id)
            if claim_id in insufficient or audit is None or audit.passed:
                continue
            used = retries_used.get(claim_id, 0)
            if used < max_retries:
                retries_used[claim_id] = used + 1
                feedback[claim_id] = audit.feedback_for_retry or "; ".join(audit.failure_reasons)
                retry.append(claim_id)
                self.rec.retry(RetryRecord(claim_id=claim_id, attempt=used + 1, feedback=feedback[claim_id]))
            else:
                insufficient[claim_id] = (f"audit failed after {used} retr{'y' if used == 1 else 'ies'}: "
                                          + ("; ".join(audit.failure_reasons) or "no reason given"))
                self.rec.retry(RetryRecord(claim_id=claim_id, attempt=used, exhausted=True))
        return {"pending_claim_ids": retry, "feedback": feedback, "retries_used": retries_used,
                "insufficient_reasons": insufficient}

    # -- 9. tier policy + knowledge-layer caps ------------------------------

    def apply_tier_policy(self, state: FinSightState) -> dict[str, Any]:
        verdicts, capped = {}, {}
        for claim in state.claims:
            cid, rule = claim.claim_id, self.ctx.kb.rule_for(claim.claim_type)
            if cid in state.insufficient_reasons or cid not in state.audits:
                verdicts[cid] = Verdict.INSUFFICIENT_EVIDENCE
                if not rule.collectors:
                    capped[cid] = f"knowledge layer: {rule.note or claim.claim_type.value + ' has no collector'}"
                continue
            evidence = state.evidence.get(cid, [])
            proposed = state.audits[cid].verdict
            reasons: list[str] = []

            result, error = self._call("tier_policy", self.ctx.registry.tier_policy.apply,
                                       claim, proposed, evidence, rule, claim_id=cid)
            verdict = proposed
            if error or not (isinstance(result, tuple) and isinstance(result[0], Verdict)):
                self.rec.event("apply_tier_policy", f"{cid}: tier policy unavailable ({error}); guards only")
            elif routing.is_raise(proposed, result[0]):
                self.rec.event("apply_tier_policy", f"{cid}: tier policy tried to raise {proposed.value} "
                                                    f"to {result[0].value}; ignored")
            else:
                verdict = result[0]
                if verdict is not proposed:
                    reasons.append(f"tier policy: {result[1] or 'lowered'}")

            verdict, why = routing.cap_verdict(verdict, rule)
            if why:
                reasons.append(why)
            for guard in (routing.enforce_tier_requirement, routing.enforce_counter_evidence):
                verdict, why = guard(verdict, evidence, rule)
                if why:
                    reasons.append(why)

            verdicts[cid] = verdict
            if reasons:
                capped[cid] = f"{proposed.value} -> {verdict.value}: " + "; ".join(reasons)
                self.rec.event("apply_tier_policy", f"{cid}: {capped[cid]}")
        return {"final_verdicts": verdicts, "capped_by": capped}

    # -- 10. assemble -------------------------------------------------------

    def assemble(self, state: FinSightState) -> dict[str, Any]:
        results, gaps = [], []
        for claim in state.claims:
            cid = claim.claim_id
            evidence = state.evidence.get(cid, [])
            verdict = state.final_verdicts[cid]
            results.append(ClaimResult(
                claim=claim, verdict=verdict, highest_tier_used=highest_tier(evidence),
                supporting=[e for e in evidence if e.stance is Stance.SUPPORTING],
                challenging=[e for e in evidence if e.stance is Stance.CHALLENGING],
                rationale=self._rationale(state, claim, evidence),
                retries_used=state.retries_used.get(cid, 0), capped_by=state.capped_by.get(cid),
            ))
            if cid in state.insufficient_reasons:
                gaps.append(f"{cid}: {state.insufficient_reasons[cid]}")
            elif verdict is Verdict.INSUFFICIENT_EVIDENCE:
                gaps.append(f"{cid}: no supporting evidence from a required source tier before the cut-off.")
        if not state.claims:
            gaps.append("No checkable claims were found in the thesis.")
        final = FinalAssessment(
            request_id=state.request.request_id, status=AssessmentStatus.COMPLETED,
            overall_verdict=overall_verdict([r.verdict for r in results]), claim_results=results,
            evidence_gaps=gaps, as_of_date=state.request.cutoff_date,
            cutoff_was_defaulted=state.request.cutoff_was_defaulted, boundary_disclaimer=BOUNDARY_DISCLAIMER,
        )
        return {"final": final}

    @staticmethod
    def _rationale(state: FinSightState, claim: Claim, evidence: list[EvidenceItem]) -> str:
        """Deterministic rationale built from checks only. Numbers are copied from evidence, never computed."""
        cid = claim.claim_id
        if cid in state.insufficient_reasons:
            return f"Not assessed: {state.insufficient_reasons[cid]}"
        audit = state.audits[cid]
        n_sup = sum(e.stance is Stance.SUPPORTING for e in evidence)
        n_chal = sum(e.stance is Stance.CHALLENGING for e in evidence)
        parts = [f"Audit passed with proposed verdict {audit.verdict.value}; "
                 f"{n_sup} supporting and {n_chal} challenging evidence items."]
        if audit.numeric_check and audit.numeric_check.computed is not None:
            nc = audit.numeric_check
            parts.append(f"Numeric check: {nc.metric_name or 'value'} computed = {nc.computed} "
                         f"({'matches' if nc.matches else 'does not match'} the claim).")
        if audit.citation_check:
            parts.append(f"Citation check on {audit.citation_check.evidence_id}: "
                         f"{'entails' if audit.citation_check.entails else 'does not entail'} the claim.")
        return " ".join(parts)

    # -- 11. advice gate (output) -------------------------------------------

    def advice_gate_output(self, state: FinSightState) -> dict[str, Any]:
        text = render_assessment(state.final)
        decision, error = self._call("advice_boundary", self.ctx.registry.advice_boundary.check,
                                     text, AdviceStage.OUTPUT)
        if error or not isinstance(decision, AdviceDecision):
            self.rec.gate("advice_output", True, f"advice boundary unavailable: {error}")
            # DECISION: fail closed on the output gate too.
            return {"final": None} | _stop(AssessmentStatus.FAILED,
                                           "Output advice check failed; response withheld.")
        self.rec.gate("advice_output", decision.flagged, decision.reason)
        update: dict[str, Any] = {"advice_decisions": [*state.advice_decisions, decision]}
        if decision.flagged:
            update |= {"final": None} | _stop(
                AssessmentStatus.REFUSED,
                f"The generated response was withheld because it contained advice-like language. {decision.reason or ''}".strip())
        return update

    # -- 12. finalize -------------------------------------------------------

    def finalize(self, state: FinSightState) -> dict[str, Any]:
        final = state.final or FinalAssessment(
            request_id=state.request.request_id if state.request else "unknown",
            status=state.status or AssessmentStatus.FAILED, refusal_reason=state.refusal_reason,
            as_of_date=state.request.cutoff_date if state.request else None,
            cutoff_was_defaulted=state.request.cutoff_was_defaulted if state.request else False,
            boundary_disclaimer=BOUNDARY_DISCLAIMER,
        )
        self.rec.event("finalize", f"status={final.status.value} reason={final.refusal_reason or '-'}")
        log = self.rec.finish(final.status, {r.claim.claim_id: r.verdict for r in final.claim_results})
        write_log(log, self.ctx.settings.logs_dir)
        return {"final": final, "decision_log": log}
