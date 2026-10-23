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
from datetime import date, datetime
from typing import Any

from finsight.agents.advice_boundary.agent import without_advice
from finsight.agents.claim_extraction.build import claims_catalog_path
from finsight.agents.claim_extraction.extract import extract_claims as run_extraction
from finsight.agents.claim_extraction.extract import (
    extraction_from_quotes,
    load_claim_rows,
    load_stored_claims,
    load_stored_reasons,
)
from finsight.agents.claim_extraction.prepare import post_limits
from finsight.agents.claim_extraction.select import MATCH_NOTE, choose_line, miss_text, stated_claim
from finsight.agents.orchestrator import explain, pending, posts, router, routing
from finsight.agents.orchestrator.aggregation import highest_tier, overall_verdict
from finsight.agents.orchestrator.expansion import report_units
from finsight.agents.orchestrator.render import render_assessment
from finsight.config import BOUNDARY_DISCLAIMER, AgentRegistry, Settings
from finsight.contracts.schemas import (
    AdviceDecision,
    AdviceStage,
    AssessmentStatus,
    AuditResult,
    SubClaim,
    SubClaimResult,
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


def _with_note(state: FinSightState, text: str) -> str:
    if state.influencer_note:
        return f"{state.influencer_note} {text}"
    return text


def _stop(status: AssessmentStatus, reason: str) -> dict[str, Any]:
    return {"status": status, "refusal_reason": reason}


def _clean_evidence(item: EvidenceItem) -> EvidenceItem:
    cleaned = without_advice(item.content)
    if cleaned == item.content.strip():
        return item
    content = cleaned or "This passage told you to buy or sell, so that wording was removed. The source is unchanged."
    return item.model_copy(update={"content": content})


def _without_advice(final: FinalAssessment) -> FinalAssessment:
    """Remove buy or sell sentences from the draft. The claim and each label stay."""
    note = "A sentence that told you to buy or sell was removed. The claim and the labels are unchanged."
    results = []
    for row in final.sub_claim_results:
        results.append(row.model_copy(update={
            "explanation": without_advice(row.explanation or "") or row.explanation,
            "supporting": [_clean_evidence(item) for item in row.supporting],
            "challenging": [_clean_evidence(item) for item in row.challenging],
        }))
    summary = without_advice(final.summary or "")
    summary = f"{note} {summary}".strip()
    return final.model_copy(update={"summary": summary, "sub_claim_results": results})


def _said_and_verified(state: FinSightState, results: list[SubClaimResult], text: str, overall: Verdict | None):
    """Separate what was said from what can be verified on the user's cut-off."""
    request = state.request
    published = state.post_published
    if (
        request is None or not state.cutoff_from_prompt or state.cutoff_input is not None
        or published is None or published <= request.cutoff_date
    ):
        return text, overall
    said = "Said: yes." if results else "Said: no."
    note = (
        f"{said} Verified: insufficient evidence as of the cut-off. "
        f"The video was published on {published.isoformat()}, after the cut-off, "
        "so the influencer had not said this by then."
    )
    for row in results:
        row.verdict = Verdict.INSUFFICIENT_EVIDENCE
        row.explanation = note
    return note, Verdict.INSUFFICIENT_EVIDENCE if results else overall


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
        claim = pending.claim_with_saved_cutoff(settings.logs_dir, state.raw_input.strip()) or state.raw_input.strip()
        bad_date = routing.invalid_cutoff_text(claim)
        if bad_date:
            pending.remember_cutoff(settings.logs_dir, claim)
            self.rec.start(new_request_id(), claim, settings.corpus_as_of_date)
            return {
                "raw_input": claim,
                "clarification_question": (
                    f"{bad_date} is not a real date. Resubmit with a real cut-off date."
                ),
            }
        if routing.has_cutoff(claim):
            pending.remember_cutoff(settings.logs_dir, claim)
        cutoff, defaulted = routing.resolve_cutoff(state.cutoff_input, claim, settings.corpus_as_of_date)
        request = ClaimRequest(request_id=new_request_id(), raw_claim=claim, company_ticker=state.company_hint,
                                cutoff_date=cutoff, cutoff_was_defaulted=defaulted)
        self.rec.start(request.request_id, state.raw_input, cutoff)
        self.rec.event("intake", f"cutoff={cutoff} defaulted={defaulted}")
        update: dict[str, Any] = {
            "request": request,
            "raw_input": claim,
            "cutoff_from_prompt": routing.has_cutoff(claim),
        }
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
        hard = routing.price_or_macro_refusal(claim, ticker)
        if hard:
            self.rec.gate("scope", True, hard)
            return _stop(AssessmentStatus.REFUSED, hard)
        influencer, influencer_refusal, influencer_question = routing.influencer_screen(
            claim, state.influencer_hint)
        if influencer_refusal or influencer_question:
            self.rec.gate("scope", True, influencer_refusal or influencer_question)
            update: dict[str, Any] = {}
            if influencer_refusal:
                update |= _stop(AssessmentStatus.REFUSED, influencer_refusal)
            else:
                update["clarification_question"] = influencer_question
            return update
        ticker = routing.detect_company(claim, companies, hint)
        reason, question = routing.scope_screen(claim, ticker, hint, companies)
        self.rec.gate("scope", reason is not None, reason or question or f"influencer={influencer} company={ticker}")
        request = state.request.model_copy(update={"company_ticker": ticker})
        update = {
            "request": request,
            "influencer": influencer,
            "influencer_note": routing.typo_note(claim),
        }
        if reason:
            update |= _stop(AssessmentStatus.REFUSED, reason)
        elif question:
            update["clarification_question"] = question
        return update

    # -- collect posts (pipeline: stored blog, social, or video; not an agent) ------

    def collect_posts(self, state: FinSightState) -> dict[str, Any]:
        """Read one stored post. The cut-off is that post's date. Nothing is downloaded."""
        req = state.request
        aliases = self.ctx.settings.corpus_companies.get(req.company_ticker or "", [])
        display = routing.INFLUENCERS[state.influencer][0]
        company = aliases[0] if aliases else req.company_ticker
        stored = posts.posts_for(posts.load_posts(self.ctx.settings.posts_dir), state.influencer, aliases)
        # The Company X demo is outside the project window. Real companies stay inside it.
        if req.company_ticker != "CMPX":
            stored = [row for row in stored if posts.in_window(row)]
        # A post published after the cut-off had not been said yet. It is not a candidate.
        matched = [
            row for row in stored
            if (uploaded := posts.post_date(row)) is not None and uploaded <= req.cutoff_date
        ]
        with_text = [row for row in matched if row.get("transcript")]
        listed_only = [row for row in matched if not row.get("transcript")]
        self.rec.event(
            "collect_posts",
            f"{display}: {len(with_text)} posts with text, {len(listed_only)} listed without text, "
            f"about {req.company_ticker}, on or before {req.cutoff_date}",
        )
        if not stored and req.company_ticker == "CMPX":
            self.rec.event("collect_posts", "placeholder company; no stored post, so the submitted sentence is checked")
            return {}
        pasted = state.post_id or posts.video_id_from_text(state.raw_input)
        if pasted and not any(row.get("video_id") == pasted for row in matched):
            later = next((row for row in stored if row.get("video_id") == pasted), None)
            published = posts.post_date(later) if later else None
            if published is not None and published > req.cutoff_date:
                return _stop(
                    AssessmentStatus.REFUSED,
                    f"That post was published on {published.isoformat()}, "
                    f"after the cut-off {req.cutoff_date.isoformat()}.",
                )
            return _stop(AssessmentStatus.REFUSED, "That post is not in the store.")
        if not matched and any(
            (uploaded := posts.post_date(row)) is not None and uploaded > req.cutoff_date for row in stored
        ):
            return _stop(
                AssessmentStatus.REFUSED,
                f"No stored post from {display} about {company} was published on or before "
                f"{req.cutoff_date.isoformat()}.",
            )
        if not with_text and listed_only:
            return _stop(
                AssessmentStatus.REFUSED,
                f"A {display} video about {company} is listed, and the transcript is not available yet.",
            )
        if not with_text:
            return _stop(
                AssessmentStatus.REFUSED,
                f"No stored post from {display} about {company} is available to check.",
            )
        # Consistency and accuracy ask about the whole window, not one video.
        explicit = state.post_id or posts.video_id_from_text(state.raw_input)
        kinds = posts.record_kinds(state.raw_input)
        if kinds and not explicit:
            ordered = sorted(
                with_text,
                key=lambda row: (posts.post_date(row) or date.min, str(row.get("video_id") or "")),
            )
            self.rec.event("collect_posts", f"record over {len(ordered)} posts: {', '.join(sorted(kinds))}")
            return {"clarification_question": posts.record_marker(kinds, [str(row.get("video_id")) for row in ordered])}
        focus_quotes: list[str] = []
        focus = stated_claim(state.raw_input, state.influencer, aliases)
        claims_path = claims_catalog_path(self.ctx.settings.posts_dir)
        if focus and claims_path.exists():
            video_ids = {str(row.get("video_id") or "") for row in with_text}
            choice = choose_line(
                focus,
                load_claim_rows(claims_path, video_ids, req.company_ticker),
                with_text,
                confirmed_video=explicit,
                ticker=req.company_ticker,
            )
            if choice.question:
                self.rec.event("collect_posts", "asked which spoken line")
                return {"clarification_question": choice.question}
            chosen = next((row for row in with_text if row.get("video_id") == choice.video_id), None)
            if chosen is None:
                return _stop(AssessmentStatus.REFUSED, "That post is not in the store.")
            focus_quotes = choice.quotes
        else:
            chosen = posts.choose_post(with_text, state.raw_input, state.influencer, state.post_id)
            if chosen is None:
                return _stop(AssessmentStatus.REFUSED, "That post is not in the store.")
            if isinstance(chosen, list):
                return {"clarification_question": posts.which_post_question(chosen)}
        transcript = posts.about_company(
            str(chosen.get("transcript") or "").strip(),
            req.company_ticker or "",
            self.ctx.settings.corpus_companies,
        )
        uploaded = posts.post_date(chosen)
        fields: dict[str, Any] = {"raw_claim": transcript or str(chosen.get("transcript") or "").strip()}
        # A cut-off the user typed stays the review date. Otherwise the post date is the cut-off.
        if uploaded is not None and (state.cutoff_input is not None or not state.cutoff_from_prompt):
            fields["cutoff_date"] = uploaded
            fields["cutoff_was_defaulted"] = False
            if self.rec.log is not None:
                self.rec.log.cutoff_date = uploaded
        request = req.model_copy(update=fields)
        when = uploaded.strftime("%d %b %Y").lstrip("0") if uploaded else "an unknown date"
        label = (
            f"Checked {display}, \"{chosen.get('title') or 'untitled'}\" "
            f"({chosen.get('video_id')}), published {when}."
        )
        label += " " + post_limits(str(chosen.get("transcript") or ""), str(chosen.get("video_id") or "") or None)
        self.rec.event(
            "collect_posts",
            f"post {chosen.get('video_id')} {chosen.get('upload_date')} cutoff={request.cutoff_date}",
        )
        return {
            "request": request,
            "from_stored_post": True,
            "checked_post": label,
            "post_id": str(chosen.get("video_id") or "") or None,
            "focus_quotes": focus_quotes,
            "post_published": uploaded,
        }

    # -- clarification (diagram: "Unclear or missing info" -> ask the user) ------

    def clarification(self, state: FinSightState) -> dict[str, Any]:
        """Stop the run and return the question to the user. The UI resubmits with the answer."""
        self.rec.gate("clarification", True, state.clarification_question)
        return {"status": AssessmentStatus.NEEDS_CLARIFICATION}

    # -- claim extraction (its own step; the orchestrator starts at plan) ---

    def extract_claims(self, state: FinSightState) -> dict[str, Any]:
        req = state.request
        claims_path = claims_catalog_path(self.ctx.settings.posts_dir)
        stored = load_stored_claims(claims_path, state.post_id, req.company_ticker) if state.from_stored_post else None
        reasons = load_stored_reasons(claims_path, state.post_id, req.company_ticker) if stored is not None else None
        aliases = self.ctx.settings.corpus_companies.get(req.company_ticker or "", [])
        focus = stated_claim(state.raw_input, state.influencer, aliases) if state.from_stored_post else None
        if state.focus_quotes:
            extracted = extraction_from_quotes(state.focus_quotes, stored, req.company_ticker)
        else:
            extracted = run_extraction(
                text=req.raw_claim,
                cutoff=req.cutoff_date,
                company_ticker=req.company_ticker,
                companies=self.ctx.settings.corpus_companies,
                from_stored_post=state.from_stored_post,
                knowledge=self.ctx.kb,
                call=self._call,
                stored=stored,
                stored_reasons=reasons,
                focus=focus,
            )
        for note in extracted.notes:
            self.rec.event("extract_claims", note)
        update: dict[str, Any] = {"sub_claims": extracted.sub_claims}
        if extracted.insufficient_reasons:
            update["insufficient_reasons"] = extracted.insufficient_reasons
        label = state.checked_post or ""
        if extracted.miss:
            label = f"{label} {extracted.miss}".strip()
        elif (focus or state.focus_quotes) and extracted.sub_claims:
            label = f"{label} {MATCH_NOTE}".strip()
        if extracted.dropped_by_cap:
            quoted = "; ".join(f'"{quote}"' for quote in extracted.dropped_by_cap)
            label = f"{label} Not checked, past the cap of 8: {quoted}.".strip()
        if label != (state.checked_post or ""):
            update["checked_post"] = label
        if extracted.status is not None:
            update |= _stop(extracted.status, extracted.refusal_reason or "Claim extraction failed.")
        elif extracted.clarification_question:
            update["clarification_question"] = extracted.clarification_question
        return update

    # -- 5. plan ------------------------------------------------------------

    def plan(self, state: FinSightState) -> dict[str, Any]:
        """The LLM router picks financial / narrative / both per sub-claim; caps and tiers stay from the rule."""
        plan, pending, insufficient = {}, [], dict(state.insufficient_reasons)
        cutoff = state.request.cutoff_date if state.request else None

        def decide(sub_claim: SubClaim) -> tuple[SubClaim, router.RouteDecision | None, Exception | None]:
            if sub_claim.sub_claim_id in insufficient:
                return sub_claim, None, None
            rule = self.ctx.kb.rule_for(sub_claim.sub_claim_type)
            decision, error = self._call(
                "router", lambda: router.route_sub_claim(sub_claim, rule, cutoff_date=cutoff),
                sub_claim_id=sub_claim.sub_claim_id)
            return sub_claim, decision, error

        with self._pool() as pool:
            decisions = list(pool.map(decide, state.sub_claims))

        for sub_claim, decision, error in decisions:
            rule = self.ctx.kb.rule_for(sub_claim.sub_claim_type)
            entry = routing.plan_entry(sub_claim, rule)
            if decision is not None:
                entry = entry.model_copy(update={"collectors": list(decision.collectors)})
                self.rec.event("plan", f"{sub_claim.sub_claim_id}: {decision.target} ({decision.source}) - "
                                       f"{decision.reason}")
            elif error is not None:
                # DECISION: a failed router call keeps the knowledge-layer route instead of dropping the sub-claim.
                self.rec.event("plan", f"{sub_claim.sub_claim_id}: router failed ({error}); "
                                       f"knowledge-layer default {entry.collectors}")
            plan[sub_claim.sub_claim_id] = entry
            self.rec.plan(entry)
            if sub_claim.sub_claim_id in insufficient:
                continue
            if entry.collectors:
                pending.append(sub_claim.sub_claim_id)
            elif decision is not None and decision.source == "llm":
                insufficient[sub_claim.sub_claim_id] = f"No evidence agent can check this: {decision.reason}"
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
        evidence = dict(state.evidence)
        for sub_claim_id, result, error in results:
            if error:
                insufficient[sub_claim_id] = f"auditor failed ({error})"
                continue
            audits[sub_claim_id] = result
            self.rec.audit(result)
            # The Auditor's stances replace the collectors' before retry_or_accept checks the tiers.
            if result.passed and result.evidence_stances:
                evidence[sub_claim_id] = [
                    e.model_copy(update={"stance": result.evidence_stances.get(e.evidence_id, e.stance)})
                    for e in evidence.get(sub_claim_id, [])]
        return {"audits": audits, "insufficient_reasons": insufficient, "evidence": evidence}

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
            for guard in (routing.enforce_tier_requirement, routing.cap_tier2_only_support,
                          routing.enforce_counter_evidence):
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
        missed = miss_text(state.checked_post)
        post_label = state.checked_post or ""
        if missed:
            post_label = post_label[:post_label.find(missed)].strip()
        if not results:
            overall = None
            text = missed or "Nothing in this post could be checked, so there is no overall label."
            gaps.append(text)
        else:
            overall = overall_verdict([unit.verdict for unit in report_units(results)])
            text = explain.summary(results, overall)
        text, overall = _said_and_verified(state, results, text, overall)
        if state.influencer_note:
            text = f"{state.influencer_note} {text}"
        # The report names what the user asked. The post text stays in the check, not in this line.
        shown_claim = state.raw_input if state.from_stored_post else state.request.raw_claim
        final = FinalAssessment(
            request_id=state.request.request_id, status=AssessmentStatus.COMPLETED,
            claim=shown_claim, checked_post=post_label or None, summary=text,
            overall_verdict=overall, sub_claim_results=results,
            evidence_gaps=gaps, as_of_date=state.request.cutoff_date,
            cutoff_was_defaulted=state.request.cutoff_was_defaulted, boundary_disclaimer=BOUNDARY_DISCLAIMER,
        )
        return {"final": final}

    def _rationale(self, state: FinSightState, sub_claim: SubClaim, evidence: list[EvidenceItem]) -> str:
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
        if decision.flagged and state.final is not None:
            # The claim text and the verdicts stay. Only the advice sentence is removed.
            update["final"] = _without_advice(state.final)
        return update

    # -- 12. finalize -------------------------------------------------------

    def finalize(self, state: FinSightState) -> dict[str, Any]:
        status = state.status or AssessmentStatus.FAILED
        question = state.clarification_question if status is AssessmentStatus.NEEDS_CLARIFICATION else None
        final = state.final or FinalAssessment(
            request_id=state.request.request_id if state.request else "unknown",
            status=status, refusal_reason=state.refusal_reason, clarification_question=question,
            claim=state.request.raw_claim if state.request else state.raw_input,
            summary=_with_note(state, explain.stop_summary(status, state.refusal_reason, question)),
            as_of_date=state.request.cutoff_date if state.request else None,
            cutoff_was_defaulted=state.request.cutoff_was_defaulted if state.request else False,
            boundary_disclaimer=BOUNDARY_DISCLAIMER,
        )
        self.rec.event("finalize", f"status={final.status.value} reason={final.refusal_reason or '-'}")
        log = self.rec.finish(final.status, {r.sub_claim.sub_claim_id: r.verdict for r in final.sub_claim_results})
        write_log(log, self.ctx.settings.logs_dir)
        return {"final": final, "decision_log": log}
