r"""FinSight orchestrator graph and the public entry point `run_finsight`. Owner: Person 5 (Hung).

Mirrors the team architecture diagram:

    intake -> advice_gate_input -> scope_check            (initial screening: advice + scope)
           -> decompose -> plan                           (orchestrator: decompose, plan evidence needs)
           -> gather_evidence -> audit -> retry_or_accept (specialists in parallel; auditor checks,
                   ^                          |            "source tier: enough?")
                   +---- re-plan + retry -----+            (max 1 retry per sub-claim; still failing -> INSUFFICIENT)
           -> apply_tier_policy -> assemble -> advice_gate_output -> finalize
Refusals jump straight to finalize. Unclear or missing info (no claim, no company, no checkable
sub-claim) goes to `clarification`, which returns a question to the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from finsight.agents.orchestrator.nodes import Nodes, RunContext, new_request_id
from finsight.config import BOUNDARY_DISCLAIMER, AgentRegistry, Settings, get_settings
from finsight.contracts.schemas import (
    AssessmentStatus,
    DecisionLog,
    FinalAssessment,
    FinSightState,
)
from finsight.knowledge.loader import KnowledgeLayer, default_knowledge_layer
from finsight.logging.decision_log import write_log


@dataclass
class RunOutput:
    assessment: FinalAssessment
    decision_log: DecisionLog


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
    for name in ("intake", "advice_gate_input", "scope_check", "clarification", "decompose", "plan",
                 "gather_evidence", "audit", "retry_or_accept", "apply_tier_policy", "assemble",
                 "advice_gate_output", "finalize"):
        g.add_node(name, getattr(nodes, name))

    g.set_entry_point("intake")
    for src, dst in (("intake", "advice_gate_input"), ("advice_gate_input", "scope_check"),
                     ("scope_check", "decompose"), ("decompose", "plan")):
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
) -> RunOutput:
    """Run the full pipeline and return the assessment plus its decision log. Never raises."""
    settings = settings or get_settings()
    ctx = RunContext(registry=registry, settings=settings, kb=kb)  # type: ignore[arg-type]
    try:
        # Inside the try: a mis-registered agent or bad knowledge layer must return FAILED, not crash the UI.
        ctx.registry = registry or AgentRegistry.from_config()
        ctx.kb = kb or default_knowledge_layer()
        initial = FinSightState(raw_input=claim or "", cutoff_input=_coerce_date(cutoff),
                                company_hint=company_ticker)
        out = build_graph(ctx).invoke(initial, config={"recursion_limit": 40 + 6 * settings.max_retries})
        return RunOutput(FinalAssessment.model_validate(out["final"]),
                         DecisionLog.model_validate(out["decision_log"]))
    except Exception as exc:  # noqa: BLE001 - last line of defence: the caller always gets an assessment
        return _crash_output(ctx, claim, exc)


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
                 **kwargs) -> FinalAssessment:
    """Public API for the UI (Person 4): evaluate `claim` as of `cutoff` (None -> corpus as-of date).

    If the result has status NEEDS_CLARIFICATION, show `clarification_question` to the user and call
    again with their answer: an edited claim, or `company_ticker=` for "which company?".
    """
    return execute(claim, cutoff, company_ticker=company_ticker, **kwargs).assessment


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

