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


def _cmd_run(args: argparse.Namespace) -> int:
    on_stage = None
    if args.stages:
        from finsight.agents.orchestrator.stages import StageReporter

        print("Stages (as they run):")
        on_stage = StageReporter()
    out = execute(args.claim, date.fromisoformat(args.cutoff) if args.cutoff else None,
                  company_ticker=args.company, on_stage=on_stage)
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
                     help="print what each pipeline stage did, as it runs (intake, gates, decompose, ...)")
    run.add_argument("--technical", action="store_true",
                     help="developer view: sub-claim types, tiers, retries, audit rationale, caps")
    run.add_argument("--company", help="ticker, e.g. CMPX (the answer to a 'which company?' clarification)")
    run.set_defaults(func=_cmd_run)

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
