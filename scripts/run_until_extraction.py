"""
Runs the dev test prompts through the pipeline up to and including claim extraction, and stops there.

Steps run (the same Nodes methods the LangGraph uses, in the same order):
    intake -> advice_gate_input -> scope_check -> collect_posts -> extract_claims
A step that refuses or asks a question stops the item, as in the graph. The prompt splitting done by
`execute()` (two influencers, two companies with their own claims, a measured comparison) is repeated here.
Follow-up answers from the test sheet (who said it, which post, a valid date) are replayed once.

No specialist agent, Auditor or LLM is called, so nothing costs money. LangGraph is not needed.

Run from the repo root:
    python -m scripts.run_until_extraction --items data/test_cases/claims_dev_final.json
    python -m scripts.run_until_extraction --items ... --ids 01 02 30 --show-quotes
    python -m scripts.run_until_extraction --ids 16 30 --full       every SubClaim field, as the agents get it
Writes extraction_run.json and extraction_run.csv (or --out PREFIX) and prints one block per item.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

# Follow-up answers taken from the test sheet's expected results. Each is applied once, after the first pass
# asks a question. Keys: influencer (the name the user gives), post_id, reply (a whole new message), cutoff_text.
FOLLOW_UPS = {
    "03": {"influencer": "Mark Roussin"},
    "04": {"influencer": ""},                      # the user dodges: "doesn't matter, just check it"
    "14": {"reply": "6 May 2026"},                 # a lone date continues the saved claim
    "20": {"influencer": "Joseph Carlson"},
    "21": {"influencer": "Mark Roussin"},
    "49": {"post_id": "csv55UtVMZM"},
}
STEPS = ("intake", "advice_gate_input", "scope_check", "collect_posts", "extract_claims")


@dataclass
class Run:
    label: str
    status: str = "REACHED_EXTRACTION"
    stopped_at: str | None = None
    message: str | None = None
    influencer: str | None = None
    ticker: str | None = None
    cutoff: str | None = None
    post: str | None = None
    checked_post: str | None = None
    focus_quotes: list[str] = field(default_factory=list)
    sub_claims: list[dict] = field(default_factory=list)
    insufficient: dict = field(default_factory=dict)
    events: list[str] = field(default_factory=list)


def _apply(state, update: dict):
    return state.model_copy(update=update) if update else state


def run_steps(claim: str, ctx, *, company=None, influencer=None, post_id=None, label="") -> Run:
    from finsight.agents.orchestrator.nodes import Nodes
    from finsight.contracts.schemas import FinSightState

    nodes = Nodes(ctx)
    ctx.recorder.__init__()                        # fresh decision log for every run
    state = FinSightState(raw_input=claim, company_hint=company, influencer_hint=influencer, post_id=post_id)
    out = Run(label=label)
    for step in STEPS:
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
    out.checked_post, out.focus_quotes = state.checked_post, list(state.focus_quotes)
    out.insufficient = dict(state.insufficient_reasons)
    for c in state.sub_claims:
        rule = ctx.kb.rule_for(c.sub_claim_type)
        out.sub_claims.append(dict(id=c.sub_claim_id, type=c.sub_claim_type.value, measure=c.measure,
                                   collectors=list(rule.collectors), text=c.text,
                                   parent_span=c.parent_span, expansion_rule=c.expansion_rule,
                                   full=c.model_dump(mode="json")))   # the SubClaim exactly as an agent gets it
    log = ctx.recorder.log
    out.events = [f"{e.node}: {e.message}" for e in (log.events if log else [])]
    return out


def run_prompt(claim: str, ctx, settings, **kw) -> list[Run]:
    """The prompt splitting `execute()` does before the graph, then the steps for each part."""
    from finsight.agents.orchestrator import routing

    if not kw.get("influencer") and not kw.get("post_id"):
        names = routing.influencers_named(claim)
        if len(names) >= 2:
            return [r for n in names[:2] for r in run_prompt(claim, ctx, settings, **{**kw, "influencer": n})]
    if kw.get("company") is None and not kw.get("post_id"):
        tickers = routing.companies_with_own_claims(claim, settings.corpus_companies)
        if len(tickers) >= 2:
            return [r for t in tickers[:2] for r in run_prompt(claim, ctx, settings, **{**kw, "company": t})]
        compared = routing.measurable_comparison(claim, settings.corpus_companies)
        if compared:
            tickers, measure = compared
            who = kw.get("influencer") or routing.detect_influencer(claim)
            return [run_steps(routing.comparison_focus(claim, t, settings.corpus_companies, measure), ctx,
                              company=t, influencer=who, label=f"compare:{t}") for t in tickers]
    label = " ".join(f"{k}={v!r}" for k, v in kw.items() if v is not None)
    return [run_steps(claim, ctx, label=label, **kw)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/test_cases/claims_dev_final.json")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--out", default="extraction_run")
    ap.add_argument("--show-quotes", action="store_true", help="print every sub-claim's text in full")
    ap.add_argument("--events", action="store_true", help="print the decision-log events of each run")
    ap.add_argument("--full", action="store_true", help="print every SubClaim field, as the agents receive it")
    args = ap.parse_args()

    os.environ.setdefault("FINSIGHT_LOGS_DIR", tempfile.mkdtemp(prefix="finsight_logs_"))  # keep logs/ clean
    from finsight.agents.advice_boundary.agent import AdviceBoundaryAgent
    from finsight.agents.claim_extraction.build import claims_catalog_path
    from finsight.agents.orchestrator.nodes import RunContext
    from finsight.config import get_settings
    from finsight.knowledge.loader import default_knowledge_layer

    settings = get_settings()
    claims_file = claims_catalog_path(settings.posts_dir)
    print(f"posts: {settings.posts_dir} ({'found' if settings.posts_dir.exists() else 'MISSING'})   "
          f"claims: {claims_file} ({'found' if claims_file.exists() else 'MISSING'})")
    # Only the advice gate is called before extraction; the other agents are never touched.
    ctx = RunContext(registry=SimpleNamespace(advice_boundary=AdviceBoundaryAgent()), settings=settings,
                     kb=default_knowledge_layer())

    items = json.loads(Path(args.items).read_text(encoding="utf-8"))["items"]
    report, rows = [], []
    for it in items:
        if args.ids and it["id"] not in args.ids:
            continue
        runs = run_prompt(it["prompt"], ctx, settings)
        follow = FOLLOW_UPS.get(it["id"])
        if follow and any(r.status == "NEEDS_CLARIFICATION" for r in runs):
            if "reply" in follow:
                again = run_prompt(follow["reply"], ctx, settings)
            else:
                kw = {k: v for k, v in follow.items() if k in ("influencer", "post_id")}
                again = run_prompt(it["prompt"], ctx, settings, **kw)
            for r in again:
                r.label = f"after answer {follow}: " + r.label
            runs += again
        print(f"\n=== {it['id']} [{it['category']}] {it['prompt']}")
        print(f"    expected:   {it['expected_result'][:200]}")
        print(f"    checked by: {it['checked_by']}")
        for r in runs:
            head = f"  -> {r.status}" + (f" at {r.stopped_at}" if r.stopped_at else "")
            print(f"{head}  {r.label}".rstrip())
            if r.message:
                print(f"     says: {r.message[:300]}")
            if r.post:
                print(f"     post {r.post}, cut-off {r.cutoff}, influencer {r.influencer}, company {r.ticker}")
            for q in r.focus_quotes:
                print(f"     matched line: {q[:160]}")
            for c in r.sub_claims:
                text = c["text"] if args.show_quotes else c["text"][:110]
                why = f"  [{r.insufficient[c['id']]}]" if c["id"] in r.insufficient else ""
                print(f"     {c['id']:<26} {c['type']:<18} measure={c['measure'] or '-':<16} -> "
                      f"{'+'.join(c['collectors']) or 'none'}{why}\n        \"{text}\"")
                if args.full:
                    for k, v in c["full"].items():
                        if k != "text":
                            print(f"          {k}: {v}")
            if args.events:
                for e in r.events:
                    print(f"       . {e}")
            rows += [dict(id=it["id"], category=it["category"], run=r.label, status=r.status,
                          stopped_at=r.stopped_at or "", message=r.message or "", influencer=r.influencer or "",
                          ticker=r.ticker or "", post=r.post or "", cutoff=r.cutoff or "",
                          sub_claim_id=c["id"], type=c["type"], measure=c["measure"] or "",
                          collectors="+".join(c["collectors"]), text=c["text"],
                          parent_span=c["parent_span"] or "", expansion_rule=c["expansion_rule"] or "",
                          is_quantitative=c["full"].get("is_quantitative"),
                          quote_time=c["full"].get("quote_time") or "", source_span=c["full"].get("source_span") or "",
                          expected=it["expected_result"], checked_by=it["checked_by"])
                     for c in (r.sub_claims or [{}]) if c] or [
                dict(id=it["id"], category=it["category"], run=r.label, status=r.status,
                     stopped_at=r.stopped_at or "", message=r.message or "", influencer=r.influencer or "",
                     ticker=r.ticker or "", post=r.post or "", cutoff=r.cutoff or "", sub_claim_id="", type="",
                     measure="", collectors="", text="", parent_span="", expansion_rule="", is_quantitative="",
                     quote_time="", source_span="", expected=it["expected_result"],
                     checked_by=it["checked_by"])]
        report.append(dict(item=it, runs=[r.__dict__ for r in runs]))

    Path(f"{args.out}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if rows:
        with open(f"{args.out}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(report)} items. Details: {args.out}.json, one row per sub-claim: {args.out}.csv")


if __name__ == "__main__":
    main()
