"""
Runs the dev test prompts through the pipeline up to and including the Financial Analysis Agent, and stops there.

Steps run (the same Nodes methods the LangGraph uses, in the same order):
    intake -> advice_gate_input -> scope_check -> collect_posts -> extract_claims -> plan -> gather_evidence
A step that refuses or asks a question stops the item, as in the graph. The prompt splitting done by
`execute()` (two influencers, two companies with their own claims, a measured comparison) is repeated here, and
follow-up answers from the test sheet (who said it, which post, a valid date) are replayed once.

gather_evidence is the real node: it hands each sub-claim to the agent with the request's cut-off, checks the
agent returned list[EvidenceItem], and screens the items (cut-off filter, claim-text echo, sub_claim_id). Only
sub-claims routed to the financial agent are gathered; the narrative agent, the Auditor and later steps are not
run. With --all-collectors the narrative collector is called too, as a stub that returns nothing.

Model: FINSIGHT_LLM_PROVIDER / FINSIGHT_LLM_MODEL as usual (default mock = the rule-based reading, no cost).
--rules forces the rules even when a real provider is set.
SEC data: data/sec/processed (run finsight.agents.financial.pull first), or FINSIGHT_SEC_DATA=<dir>.

Run from the repo root:
    python -m scripts.run_until_financial --items data/test_cases/claims_dev_final.json
    python -m scripts.run_until_financial --ids 01 11 12 36 --trace
Writes financial_run.json and financial_run.csv (or --out PREFIX).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

# Follow-up answers taken from the test sheet's expected results, applied once after the first pass asks.
FOLLOW_UPS = {
    "03": {"influencer": "Mark Roussin"},
    "04": {"influencer": ""},                      # the user dodges: "doesn't matter, just check it"
    "14": {"reply": "6 May 2026"},                 # a lone date continues the saved claim
    "20": {"influencer": "Joseph Carlson"},
    "21": {"influencer": "Mark Roussin"},
    "49": {"post_id": "csv55UtVMZM"},
}
BEFORE = ("intake", "advice_gate_input", "scope_check", "collect_posts", "extract_claims", "plan")


class NotRun:
    """Stands in for a collector this script does not test (returns no evidence)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def gather(self, sub_claim, cutoff_date, feedback):
        return []


class Watched:
    """The financial agent, plus what it returned before the orchestrator screened it."""

    def __init__(self, agent) -> None:
        self.agent, self.returned = agent, {}

    def gather(self, sub_claim, cutoff_date, feedback):
        items = self.agent.gather(sub_claim, cutoff_date, feedback)
        self.returned[sub_claim.sub_claim_id] = len(items)
        return items

    def reset(self) -> None:
        self.returned.clear()
        for log in (self.agent.reasons, self.agent.traces, self.agent.modes, self.agent.readings):
            log.clear()


@dataclass
class Run:
    label: str
    status: str = "REACHED_FINANCIAL"
    stopped_at: str | None = None
    message: str | None = None
    influencer: str | None = None
    ticker: str | None = None
    cutoff: str | None = None
    post: str | None = None
    sub_claims: list[dict] = field(default_factory=list)


def _apply(state, update: dict):
    return state.model_copy(update=update) if update else state


def run_steps(claim: str, ctx, fin: Watched, *, all_collectors=False, company=None, influencer=None, post_id=None,
              label="") -> Run:
    from finsight.agents.orchestrator.nodes import Nodes
    from finsight.contracts.schemas import FinSightState

    nodes = Nodes(ctx)
    ctx.recorder.__init__()                        # fresh decision log for every run
    fin.reset()                                    # sub-claim ids (e.g. q1, q2) repeat across runs
    state = FinSightState(raw_input=claim, company_hint=company, influencer_hint=influencer, post_id=post_id)
    out = Run(label=label)
    for step in BEFORE + ("gather_evidence",):
        if step == "gather_evidence" and not all_collectors:
            only = [s for s in state.pending_sub_claim_ids if "financial" in state.routing_plan[s].collectors]
            state = state.model_copy(update={"pending_sub_claim_ids": only})
        state = _apply(state, getattr(nodes, step)(state))
        if state.status is not None or state.clarification_question:
            out.stopped_at = step
            out.status = state.status.value if state.status is not None else "NEEDS_CLARIFICATION"
            out.message = state.refusal_reason or state.clarification_question
            break
    req = state.request
    out.influencer, out.post = state.influencer, state.post_id
    out.ticker = req.company_ticker if req else company
    out.cutoff = str(req.cutoff_date) if req else None
    agent = fin.agent
    for c in state.sub_claims:
        sid = c.sub_claim_id
        collectors = list(state.routing_plan[sid].collectors) if sid in state.routing_plan else []
        evidence = state.evidence.get(sid)
        out.sub_claims.append(dict(
            id=sid, type=c.sub_claim_type.value, measure=c.measure, collectors=collectors, text=c.text,
            parent_span=c.parent_span,
            reached_agent=sid in fin.returned, mode=agent.modes.get(sid),
            returned=fin.returned.get(sid), kept=None if evidence is None else len(evidence),
            evidence=[dict(metric=e.metric_name, value=e.computed_value,
                           comparison_value=getattr(e, "comparison_value", None),
                           comparison_period=getattr(e, "comparison_period", None), period=e.period,
                           doc=f"{e.doc_type} filed {e.publication_date}", content=e.content)
                      for e in (evidence or [])],
            agent_reason=agent.reasons.get(sid), insufficient=state.insufficient_reasons.get(sid),
            trace=agent.traces.get(sid), reading=agent.readings.get(sid)))
    return out


def run_prompt(claim: str, ctx, settings, fin, all_collectors=False, **kw) -> list[Run]:
    """The prompt splitting `execute()` does before the graph, then the steps for each part."""
    from finsight.agents.orchestrator import routing

    def again(**more):
        return run_prompt(claim, ctx, settings, fin, all_collectors, **{**kw, **more})

    if not kw.get("influencer") and not kw.get("post_id"):
        names = routing.influencers_named(claim)
        if len(names) >= 2:
            return [r for n in names[:2] for r in again(influencer=n)]
    if kw.get("company") is None and not kw.get("post_id"):
        tickers = routing.companies_with_own_claims(claim, settings.corpus_companies)
        if len(tickers) >= 2:
            return [r for t in tickers[:2] for r in again(company=t)]
        compared = routing.measurable_comparison(claim, settings.corpus_companies)
        if compared:
            tickers, measure = compared
            who = kw.get("influencer") or routing.detect_influencer(claim)
            return [run_steps(routing.comparison_focus(claim, t, settings.corpus_companies, measure), ctx, fin,
                              all_collectors=all_collectors, company=t, influencer=who, label=f"compare:{t}")
                    for t in tickers]
    label = " ".join(f"{k}={v!r}" for k, v in kw.items() if v is not None)
    return [run_steps(claim, ctx, fin, all_collectors=all_collectors, label=label, **kw)]


def _short(v):
    return "-" if v is None else f"{v:.4f}" if isinstance(v, float) and abs(v) < 100 else f"{v:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/test_cases/claims_dev_final.json")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--out", default="financial_run")
    ap.add_argument("--trace", action="store_true", help="print the planner's steps (real model only)")
    ap.add_argument("--rules", action="store_true", help="force the rule-based reading (no model calls)")
    ap.add_argument("--all-collectors", action="store_true",
                    help="also pass narrative-routed sub-claims through gather_evidence (narrative is a stub)")
    args = ap.parse_args()

    os.environ.setdefault("FINSIGHT_LOGS_DIR", tempfile.mkdtemp(prefix="finsight_logs_"))  # keep logs/ clean
    if args.rules:
        os.environ["FINSIGHT_FINANCIAL_PLANNER"] = "rules"
    from finsight.agents.advice_boundary.agent import AdviceBoundaryAgent
    from finsight.agents.claim_extraction.build import claims_catalog_path
    from finsight.agents.financial.agent import XbrlFinancialAgent
    from finsight.agents.orchestrator.nodes import RunContext
    from finsight.config import get_settings
    from finsight.knowledge.loader import default_knowledge_layer

    settings = get_settings()
    claims_file = claims_catalog_path(settings.posts_dir)
    fin = Watched(XbrlFinancialAgent())
    sec = os.getenv("FINSIGHT_SEC_DATA", "data/sec/processed")
    print(f"posts:  {settings.posts_dir} ({'found' if settings.posts_dir.exists() else 'MISSING'})\n"
          f"claims: {claims_file} ({'found' if claims_file.exists() else 'MISSING'})\n"
          f"SEC:    {sec} ({'found' if Path(sec, 'facts.csv').exists() else 'MISSING: only the demo fixture'}), "
          f"tickers {', '.join(fin.agent.store.tickers)}\n"
          f"model:  {settings.llm_provider} {settings.llm_model}"
          + (" (forced rules)" if args.rules else " (mock = rules)" if settings.llm_provider == "mock" else ""))
    ctx = RunContext(registry=SimpleNamespace(advice_boundary=AdviceBoundaryAgent(), financial=fin,
                                              narrative=NotRun("narrative")),
                     settings=settings, kb=default_knowledge_layer())

    items = json.loads(Path(args.items).read_text(encoding="utf-8"))["items"]
    report, rows, tally = [], [], Counter()
    for it in items:
        if args.ids and it["id"] not in args.ids:
            continue
        runs = run_prompt(it["prompt"], ctx, settings, fin, args.all_collectors)
        follow = FOLLOW_UPS.get(it["id"])
        if follow and any(r.status == "NEEDS_CLARIFICATION" for r in runs):
            if "reply" in follow:
                more = run_prompt(follow["reply"], ctx, settings, fin, args.all_collectors)
            else:
                kw = {k: v for k, v in follow.items() if k in ("influencer", "post_id")}
                more = run_prompt(it["prompt"], ctx, settings, fin, args.all_collectors, **kw)
            for r in more:
                r.label = f"after answer {follow}: " + r.label
            runs += more
        print(f"\n=== {it['id']} [{it['category']}] {it['prompt']}")
        print(f"    expected:   {it['expected_result'][:220]}")
        print(f"    checked by: {it['checked_by']}")
        for r in runs:
            print(f"  -> {r.status}" + (f" at {r.stopped_at}" if r.stopped_at else "") + f"  {r.label}".rstrip())
            if r.message:
                print(f"     says: {r.message[:300]}")
            if r.post:
                print(f"     post {r.post}, cut-off {r.cutoff}, company {r.ticker}")
            for c in r.sub_claims:
                route = "+".join(c["collectors"]) or "none"
                tally["financial sub-claims" if "financial" in c["collectors"] else f"routed to {route}"] += 1
                print(f"     {c['id']} {c['type']} measure={c['measure'] or '-'} -> {route}"
                      f"\n        \"{c['text'][:150]}\"")
                if not c["reached_agent"]:
                    print(f"        not sent to the financial agent"
                          + (f": {c['insufficient']}" if c["insufficient"] else ""))
                    continue
                tally[f"mode {c['mode'] or 'none (stopped before reading)'}"] += 1
                tally["got evidence" if c["evidence"] else "no evidence"] += 1
                dropped = (c["returned"] or 0) - (c["kept"] or 0)
                print(f"        [{c['mode'] or '-'}] returned {c['returned']}, kept {c['kept']}"
                      + (f" ({dropped} dropped by the orchestrator's screen)" if dropped else ""))
                for e in c["evidence"]:
                    print(f"        {e['metric']} = {_short(e['value'])} (vs {_short(e['comparison_value'])}) "
                          f"{e['period']}\n          {e['content'][:200]}")
                if c["agent_reason"]:
                    print(f"        note: {c['agent_reason'][:300]}")
                if c["insufficient"]:
                    print(f"        insufficient: {c['insufficient'][:200]}")
                if args.trace and c["trace"]:
                    print("        " + c["trace"].replace("\n", "\n        "))
                rows += [dict(id=it["id"], run=r.label, ticker=r.ticker or "", cutoff=r.cutoff or "",
                              sub_claim_id=c["id"], type=c["type"], measure=c["measure"] or "", mode=c["mode"],
                              text=c["text"], metric=e["metric"], value=e["value"],
                              comparison_value=e["comparison_value"], period=e["period"], content=e["content"],
                              note=c["agent_reason"] or "", expected=it["expected_result"])
                         for e in c["evidence"]] or [
                    dict(id=it["id"], run=r.label, ticker=r.ticker or "", cutoff=r.cutoff or "",
                         sub_claim_id=c["id"], type=c["type"], measure=c["measure"] or "", mode=c["mode"],
                         text=c["text"], metric="", value="", comparison_value="", period="", content="",
                         note=c["agent_reason"] or c["insufficient"] or "", expected=it["expected_result"])]
        report.append(dict(item=it, runs=[r.__dict__ for r in runs]))

    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if rows:
        with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(report)} items. " + ", ".join(f"{k}: {v}" for k, v in sorted(tally.items())))
    print(f"Details: {args.out}.json; one row per evidence item: {args.out}.csv")


if __name__ == "__main__":
    main()
