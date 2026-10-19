"""Command-line entry point. Owner: Person 5 (Hung).

    finsight run "claim text" --cutoff 2025-07-31     # print the assessment table
    finsight trace <request_id>                         # print the decision log
    finsight eval --system all --split dev              # run the evaluation
    finsight graph                                      # regenerate docs/orchestrator_graph.md
    finsight usage                                      # real-LLM spend so far vs the course credit
    finsight mode                                       # show LLM mode (mock = free, paid = OpenAI)
    finsight mode paid [--model gpt-5.4] / finsight mode mock
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from finsight.agents.orchestrator.graph import execute, export_mermaid
from finsight.agents.orchestrator.render import render_assessment
from finsight.config import get_settings
from finsight.logging.decision_log import format_trace, load_log


def _financial_type(text: str):
    """Revenue, earnings, margins, cash, liquidity, or leverage. A debt amount is not this agent."""
    from finsight.agents.claim_extraction.build import _classify
    from finsight.contracts.schemas import SubClaimType

    kind = _classify(text)
    financial = {
        "HISTORICAL_GROWTH", "NET_INCOME_GROWTH", "PROFITABILITY",
        "CASH_GENERATION", "LIQUIDITY", "LEVERAGE",
    }
    if kind is not None and kind.type in financial and not kind.expansion:
        return SubClaimType(kind.type)
    return None


def _ask(label: str) -> str:
    try:
        return input(label).strip()
    except EOFError:
        return ""


def _cmd_financial(args: argparse.Namespace) -> int:
    """Check one sentence against stored filings. The narrative agent is not called."""
    text = (args.claim or "").strip()
    ticker = (args.company or "").strip().upper()
    cutoff = (args.cutoff or "").strip()
    influencer = (args.influencer or "").strip()
    if text and ticker and cutoff and influencer:
        return _print_financial(text, ticker, cutoff, influencer)
    print("Type a sentence, who said it, a ticker, and the post date. An empty sentence stops.")
    one_shot = bool(args.claim or args.company or args.cutoff or args.influencer)
    while True:
        if not text:
            text = _ask("Sentence: ")
        if not text:
            return 0
        if not influencer:
            influencer = _ask("Influencer: ")
        if not ticker:
            ticker = _ask("Ticker: ").upper()
        if not cutoff:
            cutoff = _ask("As of (YYYY-MM-DD): ")
        if not influencer or not ticker or not cutoff:
            print("An influencer, a ticker, and a date are required.")
            if not args.claim:
                text = ""
            if not args.influencer:
                influencer = ""
            continue
        code = _print_financial(text, ticker, cutoff, influencer)
        if one_shot:
            return code
        text, ticker, cutoff, influencer = "", "", "", ""


def _print_financial(text: str, ticker: str, cutoff: str, influencer: str) -> int:
    from finsight.agents.financial.agent import XbrlFinancialAgent
    from finsight.agents.orchestrator.routing import INFLUENCERS, detect_influencer, influencer_names
    from finsight.contracts.schemas import SubClaim

    who = detect_influencer("", influencer)
    if who is None:
        print(f"{influencer} is not an influencer FinSight can check.")
        print(f"FinSight checks posts from: {influencer_names()}.")
        return 1
    name = INFLUENCERS[who][0]
    kind = _financial_type(text)
    if kind is None:
        print("The financial agent does not check this sentence.")
        print("It checks revenue, earnings, margins, cash, liquidity, and leverage.")
        print("A cause, a forecast, a risk, or a ranking is left for the narrative agent.")
        return 0
    try:
        as_of = date.fromisoformat(cutoff)
    except ValueError:
        print("The date must be YYYY-MM-DD.")
        return 1
    claim = SubClaim(
        sub_claim_id="c1", text=text, sub_claim_type=kind,
        company_ticker=ticker, is_quantitative=True,
    )
    items = XbrlFinancialAgent().gather(claim, as_of, None)
    print(f"{name}  {ticker}  as of {as_of}  {kind.value}")
    print(f"  {text}")
    if not items:
        print("  No filing figure for this sentence on or before that date.")
        return 0
    for item in items:
        print(f"  {item.stance.value}: {item.content}")
    return 0


def _cmd_claims(args: argparse.Namespace) -> int:
    """Write the claim table from stored videos. Prints one video when --post is set."""
    from finsight.agents.claim_extraction.build import build_claims

    posts_dir = Path(args.posts_dir) if args.posts_dir else get_settings().posts_dir
    summary = build_claims(posts_dir)
    counts = summary["counts"]
    print(f"Wrote {counts['claims']} claims from {counts['videos']} videos to {summary['path']}")
    print(
        "Skipped "
        f"{counts['skipped_no_transcript']} with no transcript, "
        f"{counts['skipped_outside_window']} outside the window, "
        f"{counts['skipped_other_influencer']} from another influencer."
    )
    types = [f"{name.removeprefix('type:')} {counts[name]}" for name in sorted(counts) if name.startswith("type:")]
    print("Types: " + ", ".join(types))
    if not args.post:
        return 0
    shown = [row for row in summary["rows"] if row["video_id"] == args.post]
    if not shown:
        print(f"No stored claim for {args.post}.")
        return 1
    print()
    print(f"{shown[0].get('influencer')}  {shown[0].get('title')}  {shown[0].get('upload_date')}  {args.post}")
    for row in shown:
        extra = row["sub_claim_type"]
        if row.get("measure"):
            extra += f"  {row['measure']}"
        print(f"  {row['company_ticker']:<6} {extra}")
        print(f"    {row['text']}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    on_stage = None
    if args.stages:
        from finsight.agents.orchestrator.stages import StageReporter

        print("Stages (as they run):")
        on_stage = StageReporter()
    out = execute(args.claim, date.fromisoformat(args.cutoff) if args.cutoff else None,
                  company_ticker=args.company, influencer=args.influencer, post_id=args.post,
                  on_stage=on_stage)
    if args.stages and not args.json:
        print("\nResult:")
    if args.json:
        print(out.assessment.model_dump_json(indent=2))
    else:
        print(render_assessment(out.assessment, show_evidence=not args.no_evidence, technical=args.technical))
        rid = out.decision_log.request_id
        print(f"Technical log: {get_settings().logs_dir / (rid + '.json')}  (view it: finsight trace {rid})")
    return 0


def _cmd_trace(args: argparse.Namespace) -> int:
    logs_dir = Path(args.logs_dir) if args.logs_dir else get_settings().logs_dir
    try:
        log = load_log(args.request_id, logs_dir)
    except FileNotFoundError:
        print(f"No decision log for {args.request_id} in {logs_dir}", file=sys.stderr)
        return 1
    print(log.model_dump_json(indent=2) if args.json else format_trace(log))
    return 0


def _cmd_graph(args: argparse.Namespace) -> int:
    print(f"wrote {export_mermaid(args.out)}")
    return 0


def _cmd_usage(args: argparse.Namespace) -> int:
    from finsight.config import COURSE_BUDGET_USD
    from finsight.llm.spend import ledger_path, ledger_summary

    totals = ledger_summary()
    if not totals:
        print(f"No real LLM calls recorded in {ledger_path()}")
        return 0
    print(f"{'MODEL':<22} {'CALLS':>6} {'IN TOK':>10} {'OUT TOK':>10} {'COST USD':>10}")
    for model, t in sorted(totals.items(), key=lambda kv: kv[0] == "ALL"):
        print(f"{model:<22} {t['calls']:>6.0f} {t['input_tokens']:>10.0f} {t['output_tokens']:>10.0f} "
              f"{t['cost_usd']:>10.4f}")
    spent = totals["ALL"]["cost_usd"]
    print(f"\nSpent ${spent:.4f} of ${COURSE_BUDGET_USD:.2f} course credit "
          f"(${COURSE_BUDGET_USD - spent:.2f} left, ledger: {ledger_path()}).")
    print("This ledger covers runs on this machine only; check the OpenAI dashboard for the shared total.")
    return 0


def _cmd_mode(args: argparse.Namespace) -> int:
    import os

    from finsight.config import set_llm_mode

    if args.mode:
        path = set_llm_mode(paid=args.mode == "paid", model=args.model)
        print(f"Updated {path}")
    s = get_settings()
    label = "MOCK (free, no API calls)" if s.llm_provider == "mock" else f"PAID ({s.llm_provider}, {s.llm_model})"
    print(f"LLM mode: {label}")
    if os.getenv("FINSIGHT_LLM_PROVIDER"):
        print("Note: FINSIGHT_LLM_PROVIDER is set in this shell and overrides config.yaml.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finsight", description="Stress-test a company-level investment claim.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate a claim")
    run.add_argument("claim")
    run.add_argument("--cutoff", help="YYYY-MM-DD (default: from the claim text, else the corpus as-of date)")
    run.add_argument("--json", action="store_true", help="print the FinalAssessment JSON")
    run.add_argument("--no-evidence", action="store_true", help="hide evidence lines")
    run.add_argument("--stages", action="store_true",
                     help="print what each pipeline stage did, as it runs (intake, gates, extract_claims, ...)")
    run.add_argument("--technical", action="store_true",
                     help="developer view: sub-claim types, tiers, retries, audit rationale, caps")
    run.add_argument("--company", help="ticker, e.g. CMPX (the answer to a 'which company?' clarification)")
    run.add_argument("--influencer", help="who to check, e.g. 'Joseph Carlson' (blank refuses)")
    run.add_argument("--post", help="video id when several stored posts match")
    run.set_defaults(func=_cmd_run)

    financial = sub.add_parser(
        "financial",
        help="check one sentence you type against stored filings (no narrative agent)",
    )
    financial.add_argument("claim", nargs="?", help="the sentence to check; omit it to be asked")
    financial.add_argument("--influencer", help="who said it, e.g. 'Mark Roussin'")
    financial.add_argument("--company", help="ticker, e.g. NVDA")
    financial.add_argument("--cutoff", help="the post date, YYYY-MM-DD")
    financial.set_defaults(func=_cmd_financial)

    claims = sub.add_parser("claims", help="write stored claims from the videos already on disk")
    claims.add_argument("--post", help="video id to print after the table is written")
    claims.add_argument("--posts-dir", help="stored videos (default: data/influencers/youtube)")
    claims.set_defaults(func=_cmd_claims)

    trace = sub.add_parser("trace", help="print the decision log for a request")
    trace.add_argument("request_id")
    trace.add_argument("--logs-dir")
    trace.add_argument("--json", action="store_true")
    trace.set_defaults(func=_cmd_trace)

    # `eval` forwards everything after it to finsight.eval.run_eval (see main below).
    sub.add_parser("eval", help="run the evaluation: finsight eval --system all --split dev", add_help=False)

    graph = sub.add_parser("graph", help="export the orchestrator graph as Mermaid")
    graph.add_argument("--out", default="docs/orchestrator_graph.md")
    graph.set_defaults(func=_cmd_graph)

    usage = sub.add_parser("usage", help="real-LLM spend recorded in the usage ledger")
    usage.set_defaults(func=_cmd_usage)

    mode = sub.add_parser("mode", help="show or switch the LLM mode (mock = free, paid = OpenAI)")
    mode.add_argument("mode", nargs="?", choices=["mock", "paid"])
    mode.add_argument("--model", help="paid model, e.g. gpt-5.4-nano (testing) or gpt-5.4 (final)")
    mode.set_defaults(func=_cmd_mode)

    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:
        from finsight.eval.run_eval import main as eval_main

        return eval_main(argv[1:])
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
