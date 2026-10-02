"""LangGraph nodes for the FinSight orchestrator. Owner: Person 5 (Hung).

Each node takes FinSightState and returns a partial update dict. Agent calls go through
`Nodes._call`, which records latency/tokens/cost/tool calls and never raises.

Safety rules enforced here:
- The claim and all retrieved text are DATA. The orchestrator never builds prompts from them
  and never reads verdicts from them; verdicts come only from audit + tier policy + guards.
- The orchestrator never computes or rewrites financial numbers.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from finsight.agents.orchestrator import explain, routing
from finsight.agents.orchestrator.aggregation import highest_tier, overall_verdict
from finsight.agents.orchestrator.render import render_assessment
from finsight.config import BOUNDARY_DISCLAIMER, AgentRegistry, Settings
from finsight.contracts.schemas import (
    AdviceDecision,
    AdviceStage,
    AssessmentStatus,
    AuditResult,
    SubClaim,
    SubClaimResult,
    SubClaimType,
    EvidenceItem,
    FinalAssessment,
    FinSightState,
    RetryRecord,
    Stance,
    TemporalViolation,
    ClaimRequest,
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

    def _call(self, agent: str, fn: Any, *args: Any, sub_claim_id: str | None = None) -> tuple[Any, Exception | None]:
        result, call, error = tracked_call(agent, fn, *args, sub_claim_id=sub_claim_id)
        self.rec.agent_call(call)
        return result, error

    def _pool(self) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=max(1, self.ctx.settings.max_workers))

    # -- 1. intake ----------------------------------------------------------

    def intake(self, state: FinSightState) -> dict[str, Any]:
        settings = self.ctx.settings
        claim = state.raw_input.strip()
        cutoff, defaulted = routing.resolve_cutoff(state.cutoff_input, claim, settings.corpus_as_of_date)
        request = ClaimRequest(request_id=new_request_id(), raw_claim=claim, company_ticker=state.company_hint,
                                cutoff_date=cutoff, cutoff_was_defaulted=defaulted)
        self.rec.start(request.request_id, state.raw_input, cutoff)
        self.rec.event("intake", f"cutoff={cutoff} defaulted={defaulted}")
        update: dict[str, Any] = {"request": request}
        # DECISION: missing claim -> ask for it (Clarification); overlong input -> REFUSED, not FAILED.
        if not claim:
            update["clarification_question"] = routing.CLARIFY_EMPTY
        elif len(claim) > settings.max_claim_chars:
            update |= _stop(AssessmentStatus.REFUSED, f"Claim longer than {settings.max_claim_chars} characters.")
        return update

    # -- 2. advice gate (input) --------------------------------------------

    def advice_gate_input(self, state: FinSightState) -> dict[str, Any]:
        decision, error = self._call("advice_boundary", self.ctx.registry.advice_boundary.check,
                                     state.request.raw_claim, AdviceStage.INPUT)
        if error or not isinstance(decision, AdviceDecision):
            # DECISION: fail closed - if the advice gate is down we do not analyze.
            self.rec.gate("advice_input", True, f"advice boundary unavailable: {error}")
            return _stop(AssessmentStatus.FAILED, "Advice boundary check failed; request not analyzed.")
        self.rec.gate("advice_input", decision.flagged, decision.reason)
        update: dict[str, Any] = {"advice_decisions": [*state.advice_decisions, decision]}
        if decision.flagged:
            update |= _stop(AssessmentStatus.REFUSED, decision.reason or "Personalized investment advice request.")
        return update

    # -- 3. scope check (second half of the diagram's "initial screening") -------

    def scope_check(self, state: FinSightState) -> dict[str, Any]:
        claim, hint = state.request.raw_claim, state.request.company_ticker
        companies = self.ctx.settings.corpus_companies
        ticker = routing.detect_company(claim, companies, hint)
        reason, question = routing.scope_screen(claim, ticker, hint, companies)
        self.rec.gate("scope", reason is not None, reason or question or f"company={ticker}")
        request = state.request.model_copy(update={"company_ticker": ticker})
        update: dict[str, Any] = {"request": request}
        if reason:
            update |= _stop(AssessmentStatus.REFUSED, reason)
        elif question:
            update["clarification_question"] = question
        return update

    # -- clarification (diagram: "Unclear or missing info" -> ask the user) ------

    def clarification(self, state: FinSightState) -> dict[str, Any]:
        """Stop the run and return the question to the user. The UI resubmits with the answer."""
        self.rec.gate("clarification", True, state.clarification_question)
        return {"status": AssessmentStatus.NEEDS_CLARIFICATION}

    # -- 4. decompose -------------------------------------------------------

    def decompose(self, state: FinSightState) -> dict[str, Any]:
        req = state.request
        sub_claims, error = self._call("decomposer", self.ctx.registry.decomposer.decompose,
                                   req.raw_claim, req.cutoff_date)
        if error or not isinstance(sub_claims, list) or not all(isinstance(c, SubClaim) for c in sub_claims):
            return _stop(AssessmentStatus.FAILED, f"Claim decomposition failed: {error or 'invalid output'}")
        if len({c.sub_claim_id for c in sub_claims}) != len(sub_claims):
            # DECISION: renumber duplicate sub_claim_ids instead of failing the run.
            sub_claims = [c.model_copy(update={"sub_claim_id": f"c{i + 1}"}) for i, c in enumerate(sub_claims)]
            self.rec.event("decompose", "duplicate sub_claim_ids from decomposer; renumbered")
        sub_claims = [c if c.company_ticker else c.model_copy(update={"company_ticker": req.company_ticker})
                  for c in sub_claims]
        self.rec.event("decompose", f"{len(sub_claims)} sub-claims: " + ", ".join(
            f"{c.sub_claim_id}={c.sub_claim_type.value}" for c in sub_claims))
        refusing = [c for c in sub_claims if c.sub_claim_type is SubClaimType.PERSONALIZED_ADVICE
                    or self.ctx.kb.rule_for(c.sub_claim_type).refuse]
        update: dict[str, Any] = {"sub_claims": sub_claims}
        if refusing:
            note = self.ctx.kb.rule_for(refusing[0].sub_claim_type).note or "Personalized advice is out of scope."
            update |= _stop(AssessmentStatus.REFUSED, f"Sub-claim {refusing[0].sub_claim_id}: {note}")
        elif all(c.sub_claim_type is SubClaimType.OTHER for c in sub_claims):
            # DECISION: no recognisable sub-claim (none, or only OTHER) counts as unclear -> ask what to check.
            update["clarification_question"] = routing.CLARIFY_NO_SUB_CLAIM
        return update

    # -- 5. plan ------------------------------------------------------------

    def plan(self, state: FinSightState) -> dict[str, Any]:
        plan, pending, insufficient = {}, [], dict(state.insufficient_reasons)
        for sub_claim in state.sub_claims:
            rule = self.ctx.kb.rule_for(sub_claim.sub_claim_type)
            entry = routing.plan_entry(sub_claim, rule)
            plan[sub_claim.sub_claim_id] = entry
            self.rec.plan(entry)
            if entry.collectors:
                pending.append(sub_claim.sub_claim_id)
            else:
                insufficient[sub_claim.sub_claim_id] = (
                    rule.note or f"No evidence collector for {sub_claim.sub_claim_type.value}.")
        return {"routing_plan": plan, "pending_sub_claim_ids": pending, "insufficient_reasons": insufficient,
                "retries_used": {c.sub_claim_id: 0 for c in state.sub_claims}}

    # -- 6. gather evidence (sub-claims in parallel) ----------------------------

    def gather_evidence(self, state: FinSightState) -> dict[str, Any]:
        req = state.request
        sub_claims = {c.sub_claim_id: c for c in state.sub_claims}

        def collect(sub_claim_id: str) -> tuple[str, list[EvidenceItem], list[str]]:
            items, errors = [], []
            for collector in state.routing_plan[sub_claim_id].collectors:
                agent = getattr(self.ctx.registry, collector)
                result, error = self._call(collector, agent.gather, sub_claims[sub_claim_id], req.cutoff_date,
                                           state.feedback.get(sub_claim_id), sub_claim_id=sub_claim_id)
                if error is None and not (isinstance(result, list)
                                          and all(isinstance(i, EvidenceItem) for i in result)):
                    error = TypeError("collector did not return list[EvidenceItem]")
                if error:
                    errors.append(f"{collector} agent failed ({error})")
                else:
                    items.extend(result)
            return sub_claim_id, items, errors

        with self._pool() as pool:
            results = list(pool.map(collect, state.pending_sub_claim_ids))

        evidence, insufficient = dict(state.evidence), dict(state.insufficient_reasons)
        for sub_claim_id, items, errors in results:
            if errors:
                # DECISION: any collector failure makes the sub-claim INSUFFICIENT_EVIDENCE (no partial evidence).
                insufficient[sub_claim_id] = "; ".join(errors)
                evidence[sub_claim_id] = []
            else:
                evidence[sub_claim_id] = self._screen(sub_claim_id, items, req)
        pending = [c for c in state.pending_sub_claim_ids if c not in insufficient]
        return {"evidence": evidence, "insufficient_reasons": insufficient, "pending_sub_claim_ids": pending}

    def _screen(self, sub_claim_id: str, items: list[EvidenceItem], req: ClaimRequest) -> list[EvidenceItem]:
        """Hard temporal filter, claim-is-not-evidence filter, sub_claim_id consistency."""
        kept, dropped = routing.split_temporal(items, req.cutoff_date)
        for item in dropped:
            self.rec.temporal_violation(TemporalViolation(
                sub_claim_id=sub_claim_id, evidence_id=item.evidence_id, publication_date=item.publication_date,
                cutoff_date=req.cutoff_date, produced_by=item.produced_by))
        screened = []
        for item in kept:
            if routing.is_claim_echo(item, req.raw_claim):
                self.rec.event("gather_evidence",
                               f"{sub_claim_id}: dropped {item.evidence_id} (claim text is not evidence)")
            elif item.sub_claim_id != sub_claim_id:
                self.rec.event("gather_evidence",
                               f"{sub_claim_id}: dropped {item.evidence_id} (sub_claim_id {item.sub_claim_id})")
            else:
                screened.append(item)
        return screened

    # -- 7. audit -----------------------------------------------------------

    def audit(self, state: FinSightState) -> dict[str, Any]:
        sub_claims = {c.sub_claim_id: c for c in state.sub_claims}

        def run(sub_claim_id: str) -> tuple[str, AuditResult | None, Exception | None]:
            result, error = self._call("auditor", self.ctx.registry.auditor.audit, sub_claims[sub_claim_id],
                                       state.evidence.get(sub_claim_id, []), sub_claim_id=sub_claim_id)
            if error is None and not (isinstance(result, AuditResult) and result.sub_claim_id == sub_claim_id):
                error = TypeError("auditor did not return an AuditResult for this sub-claim")
            return sub_claim_id, result, error

        with self._pool() as pool:
            results = list(pool.map(run, state.pending_sub_claim_ids))

        audits, insufficient = dict(state.audits), dict(state.insufficient_reasons)
        for sub_claim_id, result, error in results:
            if error:
                insufficient[sub_claim_id] = f"auditor failed ({error})"
                continue
            audits[sub_claim_id] = result
            self.rec.audit(result)
        return {"audits": audits, "insufficient_reasons": insufficient}

    # -- 8. retry or accept -------------------------------------------------

    def retry_or_accept(self, state: FinSightState) -> dict[str, Any]:
        """The diagram's Auditor decision: "number + citation checks, source tier: enough?".

        Yes -> accept. No -> the orchestrator re-plans the sub-claim (adds the rule's retry_collectors)
        and re-gathers it with feedback, at most `max_retries` times. Still failing -> the sub-claim
        becomes INSUFFICIENT_EVIDENCE and the rest of the answer is kept.
        """
        max_retries = self.ctx.settings.max_retries
        retry, feedback, plan = [], dict(state.feedback), dict(state.routing_plan)
        retries_used, insufficient = dict(state.retries_used), dict(state.insufficient_reasons)
        shortfalls = dict(state.tier_shortfalls)
        sub_claims = {c.sub_claim_id: c for c in state.sub_claims}
        for sub_claim_id in state.pending_sub_claim_ids:
            audit = state.audits.get(sub_claim_id)
            if sub_claim_id in insufficient or audit is None:
                continue
            rule = self.ctx.kb.rule_for(sub_claims[sub_claim_id].sub_claim_type)
            if audit.passed:
                shortfall = routing.tier_shortfall(audit.verdict, state.evidence.get(sub_claim_id, []), rule)
                if shortfall is None:
                    shortfalls.pop(sub_claim_id, None)
                    continue  # accept
                problem = f"source tier not enough: {shortfall}"
                hint = shortfall
            else:
                problem = "audit failed: " + ("; ".join(audit.failure_reasons) or "no reason given")
                hint = audit.feedback_for_retry or "; ".join(audit.failure_reasons)

            used = retries_used.get(sub_claim_id, 0)
            if used < max_retries:
                retries_used[sub_claim_id] = used + 1
                feedback[sub_claim_id] = hint
                plan[sub_claim_id] = routing.replan_entry(plan[sub_claim_id], rule)
                if plan[sub_claim_id].collectors != state.routing_plan[sub_claim_id].collectors:
                    self.rec.plan(plan[sub_claim_id])
                    self.rec.event("retry_or_accept", f"{sub_claim_id}: re-planned collectors -> "
                                                      f"{plan[sub_claim_id].collectors}")
                retry.append(sub_claim_id)
                self.rec.retry(RetryRecord(sub_claim_id=sub_claim_id, attempt=used + 1, feedback=f"{problem} | {hint}"))
            else:
                insufficient[sub_claim_id] = f"{problem} (still failing after {used} retr{'y' if used == 1 else 'ies'})"
                if audit.passed:
                    shortfalls[sub_claim_id] = insufficient[sub_claim_id]
                self.rec.retry(RetryRecord(sub_claim_id=sub_claim_id, attempt=used, feedback=problem, exhausted=True))
        return {"pending_sub_claim_ids": retry, "feedback": feedback, "retries_used": retries_used,
                "insufficient_reasons": insufficient, "routing_plan": plan, "tier_shortfalls": shortfalls}

    # -- 9. tier policy + knowledge-layer caps ------------------------------

    def apply_tier_policy(self, state: FinSightState) -> dict[str, Any]:
        verdicts, capped = {}, {}
        for sub_claim in state.sub_claims:
            cid, rule = sub_claim.sub_claim_id, self.ctx.kb.rule_for(sub_claim.sub_claim_type)
            if cid in state.insufficient_reasons or cid not in state.audits:
                verdicts[cid] = Verdict.INSUFFICIENT_EVIDENCE
                if not rule.collectors:
                    note = rule.note or f"{sub_claim.sub_claim_type.value} has no collector"
                    capped[cid] = f"knowledge layer: {note}"
                elif cid in state.tier_shortfalls:
                    capped[cid] = f"tier rule: {state.tier_shortfalls[cid]}"
                continue
            evidence = state.evidence.get(cid, [])
            proposed = state.audits[cid].verdict
            reasons: list[str] = []

            result, error = self._call("tier_policy", self.ctx.registry.tier_policy.apply,
                                       sub_claim, proposed, evidence, rule, sub_claim_id=cid)
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
        for sub_claim in state.sub_claims:
            cid = sub_claim.sub_claim_id
            evidence = state.evidence.get(cid, [])
            verdict = state.final_verdicts[cid]
            results.append(SubClaimResult(
                sub_claim=sub_claim, verdict=verdict, highest_tier_used=highest_tier(evidence),
                supporting=[e for e in evidence if e.stance is Stance.SUPPORTING],
                challenging=[e for e in evidence if e.stance is Stance.CHALLENGING],
                rationale=self._rationale(state, sub_claim, evidence),
                retries_used=state.retries_used.get(cid, 0), capped_by=state.capped_by.get(cid),
            ))
            results[-1].explanation = explain.explain(results[-1], state.request.cutoff_date,
                                                      state.insufficient_reasons.get(cid))
            if cid in state.insufficient_reasons:
                gaps.append(f"{cid}: {state.insufficient_reasons[cid]}")
            elif verdict is Verdict.INSUFFICIENT_EVIDENCE:
                gaps.append(f"{cid}: no supporting evidence from a required source tier before the cut-off.")
        if not state.sub_claims:
            gaps.append("No checkable sub-claims were found in the claim.")
        overall = overall_verdict([r.verdict for r in results])
        final = FinalAssessment(
            request_id=state.request.request_id, status=AssessmentStatus.COMPLETED,
            claim=state.request.raw_claim, summary=explain.summary(results, overall),
            overall_verdict=overall, sub_claim_results=results,
            evidence_gaps=gaps, as_of_date=state.request.cutoff_date,
            cutoff_was_defaulted=state.request.cutoff_was_defaulted, boundary_disclaimer=BOUNDARY_DISCLAIMER,
        )
        return {"final": final}

    @staticmethod
    def _rationale(state: FinSightState, sub_claim: SubClaim, evidence: list[EvidenceItem]) -> str:
        """Deterministic rationale built from checks only. Numbers are copied from evidence, never computed."""
        cid = sub_claim.sub_claim_id
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
                         f"({'matches' if nc.matches else 'does not match'} the sub-claim).")
        if audit.citation_check:
            parts.append(f"Citation check on {audit.citation_check.evidence_id}: "
                         f"{'entails' if audit.citation_check.entails else 'does not entail'} the sub-claim.")
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
                ("The generated response was withheld because it contained advice-like language. "
                 f"{decision.reason or ''}").strip())
        return update

    # -- 12. finalize -------------------------------------------------------

    def finalize(self, state: FinSightState) -> dict[str, Any]:
        status = state.status or AssessmentStatus.FAILED
        question = state.clarification_question if status is AssessmentStatus.NEEDS_CLARIFICATION else None
        final = state.final or FinalAssessment(
            request_id=state.request.request_id if state.request else "unknown",
            status=status, refusal_reason=state.refusal_reason, clarification_question=question,
            claim=state.request.raw_claim if state.request else state.raw_input,
            summary=explain.stop_summary(status, state.refusal_reason, question),
            as_of_date=state.request.cutoff_date if state.request else None,
            cutoff_was_defaulted=state.request.cutoff_was_defaulted if state.request else False,
            boundary_disclaimer=BOUNDARY_DISCLAIMER,
        )
        self.rec.event("finalize", f"status={final.status.value} reason={final.refusal_reason or '-'}")
        log = self.rec.finish(final.status, {r.sub_claim.sub_claim_id: r.verdict for r in final.sub_claim_results})
        write_log(log, self.ctx.settings.logs_dir)
        return {"final": final, "decision_log": log}
