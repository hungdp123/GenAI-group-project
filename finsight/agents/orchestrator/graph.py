r"""FinSight orchestrator graph and the public entry point `run_finsight`. Owner: Person 5 (Hung).

Mirrors the team architecture diagram:

    intake -> advice_gate_input -> scope_check            (initial screening: advice + scope)
           -> collect_posts                              (stored post; cut-off = post date)
           -> extract_claims                              (claim extraction: keep the quote, one type, no label)
           -> plan                                        (orchestrator: route each claim, plan evidence)
           -> gather_evidence -> audit -> retry_or_accept (specialists in parallel; auditor checks,
                   ^                          |            "source tier: enough?")
                   +---- re-plan + retry -----+            (max 1 retry per sub-claim; still failing -> INSUFFICIENT)
           -> apply_tier_policy -> assemble -> advice_gate_output -> finalize
Refusals jump straight to finalize. Unclear or missing info (no claim, no company, no checkable
sub-claim) goes to `clarification`, which returns a question to the user.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from finsight.agents.orchestrator import explain, posts, routing
from finsight.agents.orchestrator.aggregation import overall_verdict
from finsight.agents.orchestrator.expansion import report_units
from finsight.agents.orchestrator.nodes import Nodes, RunContext, new_request_id
from finsight.config import BOUNDARY_DISCLAIMER, AgentRegistry, Settings, get_settings
from finsight.contracts.schemas import (
    AssessmentStatus,
    DecisionLog,
    FinalAssessment,
    FinSightState,
    SubClaimResult,
    Verdict,
)
from finsight.knowledge.loader import KnowledgeLayer, default_knowledge_layer
from finsight.logging.decision_log import DecisionLogRecorder, write_log

if TYPE_CHECKING:
    from finsight.agents.orchestrator.record import RecordResult


@dataclass
class RunOutput:
    assessment: FinalAssessment
    decision_log: DecisionLog
    record: RecordResult | None = None  # set for a consistency/accuracy record: one check per post


# -- conditional edges ---------------------------------------------------------


def _continue_or_stop(next_node: str):
    """Refused/failed -> finalize; unclear or missing info -> clarification; otherwise continue."""

    def route(state: FinSightState) -> str:
        if state.status is not None:
            return "finalize"
        if state.clarification_question:
            return "clarification"
        return next_node

    route.__name__ = f"to_{next_node}_or_stop"
    return route


def route_after_plan(state: FinSightState) -> str:
    return "gather_evidence" if state.pending_sub_claim_ids else "apply_tier_policy"


def route_after_retry_check(state: FinSightState) -> str:
    """retry_or_accept leaves only sub-claims to retry in pending_sub_claim_ids."""
    return "gather_evidence" if state.pending_sub_claim_ids else "apply_tier_policy"


def build_graph(ctx: RunContext):
    """Compile the StateGraph for one run (nodes close over the run context)."""
    nodes = Nodes(ctx)
    g = StateGraph(FinSightState)
    for name in ("intake", "advice_gate_input", "scope_check", "clarification", "collect_posts",
                 "extract_claims", "plan", "gather_evidence", "audit", "retry_or_accept", "apply_tier_policy",
                 "assemble", "advice_gate_output", "finalize"):
        g.add_node(name, getattr(nodes, name))

    g.set_entry_point("intake")
    for src, dst in (("intake", "advice_gate_input"), ("advice_gate_input", "scope_check"),
                     ("scope_check", "collect_posts"), ("collect_posts", "extract_claims"),
                     ("extract_claims", "plan")):
        paths = {dst: dst, "finalize": "finalize"}
        if src != "advice_gate_input":  # the advice gate only refuses; it never asks a question
            paths["clarification"] = "clarification"
        g.add_conditional_edges(src, _continue_or_stop(dst), paths)
    g.add_edge("clarification", "finalize")
    g.add_conditional_edges("plan", route_after_plan,
                            {"gather_evidence": "gather_evidence", "apply_tier_policy": "apply_tier_policy"})
    g.add_edge("gather_evidence", "audit")
    g.add_edge("audit", "retry_or_accept")
    g.add_conditional_edges("retry_or_accept", route_after_retry_check,
                            {"gather_evidence": "gather_evidence", "apply_tier_policy": "apply_tier_policy"})
    g.add_edge("apply_tier_policy", "assemble")
    g.add_edge("assemble", "advice_gate_output")
    g.add_edge("advice_gate_output", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


# -- entry points ----------------------------------------------------------------


def _coerce_date(value: date | str | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def execute(
    claim: str,
    cutoff: date | str | None = None,
    *,
    registry: AgentRegistry | None = None,
    settings: Settings | None = None,
    kb: KnowledgeLayer | None = None,
    company_ticker: str | None = None,
    influencer: str | None = None,
    post_id: str | None = None,
    on_stage: Callable[[str, dict, DecisionLogRecorder], None] | None = None,
) -> RunOutput:
    """Run the full pipeline and return the assessment plus its decision log. Never raises.

    `on_stage(node, update, recorder)` is called after every graph node (used by `finsight run --stages`).
    It only observes; an exception inside it is ignored so it can never change the run.
    """
    settings = settings or get_settings()
    names = routing.influencers_named(claim or "") if not influencer else []
    if len(names) >= 2 and influencer is None and not post_id:
        runs = [
            execute(
                routing.speaker_claim(claim, name), cutoff, registry=registry, settings=settings, kb=kb,
                company_ticker=company_ticker, influencer=name, on_stage=on_stage,
            )
            for name in names[:2]
        ]
        return _combine(runs)
    tickers = (
        routing.companies_with_own_claims(claim or "", settings.corpus_companies)
        if company_ticker is None and not post_id else []
    )
    if len(tickers) >= 2:
        runs = [
            execute(claim, cutoff, registry=registry, settings=settings, kb=kb,
                    company_ticker=ticker, influencer=influencer, on_stage=on_stage)
            for ticker in tickers[:2]
        ]
        return _combine(runs)
    compared = (
        routing.measurable_comparison(claim or "", settings.corpus_companies)
        if company_ticker is None and not post_id else None
    )
    if compared:
        tickers, measure = compared
        who = influencer or routing.detect_influencer(claim or "")
        runs = [
            execute(
                routing.comparison_focus(claim or "", ticker, settings.corpus_companies, measure),
                cutoff, registry=registry, settings=settings, kb=kb,
                company_ticker=ticker, influencer=who, on_stage=on_stage,
            )
            for ticker in tickers
        ]
        return _compare(runs, claim or "", tickers, settings.corpus_companies)
    ctx = RunContext(registry=registry, settings=settings, kb=kb)  # type: ignore[arg-type]
    try:
        # Inside the try: a mis-registered agent or bad knowledge layer must return FAILED, not crash the UI.
        ctx.registry = registry or AgentRegistry.from_config()
        ctx.kb = kb or default_knowledge_layer()
        initial = FinSightState(raw_input=claim or "", cutoff_input=_coerce_date(cutoff),
                                company_hint=company_ticker, influencer_hint=influencer, post_id=post_id)
        config = {"recursion_limit": 40 + 6 * settings.max_retries}
        graph = build_graph(ctx)
        if on_stage is None:
            out = graph.invoke(initial, config=config)
        else:
            out = None
            for mode, chunk in graph.stream(initial, config=config, stream_mode=["updates", "values"]):
                if mode == "values":
                    out = chunk
                    continue
                for node, update in chunk.items():
                    try:
                        on_stage(node, update or {}, ctx.recorder)
                    except Exception:  # noqa: BLE001 - observers must never break a run
                        pass
        result = RunOutput(FinalAssessment.model_validate(_field(out, "final")),
                           DecisionLog.model_validate(_field(out, "decision_log")))
        parsed = posts.parse_record_marker(result.assessment.clarification_question)
        if parsed and not post_id:
            kinds, video_ids = parsed
            from finsight.agents.orchestrator.record import assess_record

            record = assess_record(
                claim, kinds=kinds, video_ids=video_ids, cutoff=cutoff,
                company_ticker=company_ticker, influencer=influencer,
                posts_root=ctx.settings.posts_dir,
                run_one=lambda *args, **kwargs: execute(*args, registry=ctx.registry, settings=ctx.settings,
                                                        kb=ctx.kb, **kwargs),
                on_stage=on_stage,
            )
            return RunOutput(record.assessment, result.decision_log, record)
        return result
    except Exception as exc:  # noqa: BLE001 - last line of defence: the caller always gets an assessment
        return _crash_output(ctx, claim, exc)


def _combine(runs: list[RunOutput]) -> RunOutput:
    """Two influencers in one prompt become two entries in one result."""
    finals = [run.assessment for run in runs]
    results = [row for final in finals for row in final.sub_claim_results]
    names = []
    for final in finals:
        if final.checked_post:
            names.append(final.checked_post)
    # A run with no headline ids counts all of its rows, as it does on its own.
    asked = [
        row for final in finals for row in final.sub_claim_results
        if not final.headline_sub_claim_ids or row.sub_claim.sub_claim_id in final.headline_sub_claim_ids
    ]
    status =next((final.status for final in finals if final.status is AssessmentStatus.COMPLETED), finals[0].status)
    if asked:
        overall = overall_verdict([unit.verdict for unit in report_units(asked)])
        summary = _merged_summary(finals, asked, overall, len(results) - len(asked))
    else:
        overall = finals[0].overall_verdict
        summary = _unique_sentences(final.summary for final in finals if final.summary)
    question = next((final.clarification_question for final in finals if final.clarification_question), None)
    merged = finals[0].model_copy(update={
        "status": status,
        "sub_claim_results": results,
        "headline_sub_claim_ids": [row.sub_claim.sub_claim_id for row in asked],
        "summary": summary,
        "checked_post": " | ".join(names) or None,
        "overall_verdict": overall,
        "clarification_question": question if status is AssessmentStatus.NEEDS_CLARIFICATION else None,
    })
    return RunOutput(merged, runs[0].decision_log)


# Sentences each run writes about its own counts. The merged result says them once, over both runs.
_COUNT_SENTENCE = re.compile(
    r"^(?:Overall, |FinSight (?:split it|checked it)|In the same line the influencer also said)"
)


def _merged_summary(finals: list[FinalAssessment], asked: list[SubClaimResult], overall: Verdict | None,
                    also: int) -> str:
    """One verdict and one breakdown for both runs, then any other note a run added (said once)."""
    text = explain.summary(asked, overall)
    if also:
        text += (f" In the same line the influencer also said {also} other thing"
                 f"{'s' if also != 1 else ''}; those are checked below but not counted in this verdict.")
    notes = _unique_sentences(final.summary for final in finals if final.summary)
    extra = [s for s in re.split(r"(?<=[.!?])\s+", notes) if s and not _COUNT_SENTENCE.match(s)]
    return " ".join([text, *extra])


def _unique_sentences(summaries: Iterable[str]) -> str:
    """Join run summaries, dropping a sentence one run already said (two companies, same verdict)."""
    seen: list[str] = []
    for text in summaries:
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            if sentence and sentence not in seen:
                seen.append(sentence)
    return " ".join(seen)


def _computed(run: RunOutput) -> tuple[float | None, str | None]:
    for row in run.assessment.sub_claim_results:
        for item in [*row.supporting, *row.challenging]:
            if item.computed_value is not None:
                return item.computed_value, item.content
    return None, None


def _compare(
    runs: list[RunOutput], claim: str, tickers: list[str], companies: dict[str, list[str]],
) -> RunOutput:
    """Keep each company's check, then say how the two figures compare."""
    from finsight.agents.orchestrator.explain import comparison_line

    merged = _combine(runs)
    left_value, left_text = _computed(runs[0])
    right_value, right_text = _computed(runs[1])
    names = [(companies.get(ticker) or [ticker])[0] for ticker in tickers]
    sentence, verdict = comparison_line(claim, names[0], names[1], left_text, right_text, left_value, right_value)
    summary = f"{sentence} {merged.assessment.summary or ''}".strip()
    return RunOutput(
        merged.assessment.model_copy(update={"summary": summary, "overall_verdict": verdict}),
        merged.decision_log,
    )


def _field(state, name: str):
    """Graph output may be a dict or a FinSightState depending on the LangGraph call path."""
    return state[name] if isinstance(state, dict) else getattr(state, name)


def _crash_output(ctx: RunContext, claim: str, exc: Exception) -> RunOutput:
    rec = ctx.recorder
    if rec.log is None:
        rec.start(new_request_id(), claim or "", None)
    rec.event("orchestrator", f"unhandled {type(exc).__name__}: {exc}")
    log = rec.finish(AssessmentStatus.FAILED, {})
    write_log(log, ctx.settings.logs_dir)
    final = FinalAssessment(request_id=log.request_id, status=AssessmentStatus.FAILED, as_of_date=log.cutoff_date,
                            refusal_reason=f"Internal error: {type(exc).__name__}: {exc}",
                            boundary_disclaimer=BOUNDARY_DISCLAIMER)
    return RunOutput(final, log)


def run_finsight(claim: str, cutoff: date | None = None, company_ticker: str | None = None,
                 influencer: str | None = None, post_id: str | None = None, **kwargs) -> FinalAssessment:
    """Public API for the UI (Person 4).

    The user sentence selects an influencer and a company. The claim that is checked is the
    stored post, and the cut-off is that post's publication date. A date passed in `cutoff`
    does not replace it.

    If the result has status NEEDS_CLARIFICATION, show `clarification_question` and call again
    with the answer: the influencer's name, `company_ticker=` , or `post_id=` for "which post?".
    Pass `influencer=""` when the reply still names no influencer: that request is refused.
    """
    return execute(claim, cutoff, company_ticker=company_ticker, influencer=influencer,
                   post_id=post_id, **kwargs).assessment


def export_mermaid(path: str | Path = "docs/orchestrator_graph.md") -> Path:
    """Write the compiled graph as a Mermaid diagram (Markdown file)."""
    ctx = RunContext(registry=None, settings=get_settings(), kb=default_knowledge_layer())  # type: ignore[arg-type]
    mermaid = build_graph(ctx).get_graph().draw_mermaid()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# FinSight orchestrator graph\n\n"
        "Generated by `finsight graph` (finsight.agents.orchestrator.graph.export_mermaid). Do not edit by hand.\n\n"
        f"```mermaid\n{mermaid}```\n",
        encoding="utf-8",
    )
    return path

