"""Command-line entry point. Owner: Person 5 (Hung).

    finsight run "thesis text" --cutoff 2025-07-31     # print the assessment table
    finsight trace <request_id>                         # print the decision log
    finsight eval --system all --split dev              # run the evaluation
    finsight graph                                      # regenerate docs/orchestrator_graph.md
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
    out = execute(args.thesis, date.fromisoformat(args.cutoff) if args.cutoff else None)
    if args.json:
        print(out.assessment.model_dump_json(indent=2))
    else:
        print(render_assessment(out.assessment, show_evidence=not args.no_evidence))
        print(f"\nDecision log: {get_settings().logs_dir / (out.decision_log.request_id + '.json')}")
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


def _cmd_eval(args: argparse.Namespace) -> int:
    from finsight.eval.run_eval import main as eval_main

    return eval_main(args.rest)


def _cmd_graph(args: argparse.Namespace) -> int:
    print(f"wrote {export_mermaid(args.out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finsight", description="Stress-test a company-level investment thesis.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate a thesis")
    run.add_argument("thesis")
    run.add_argument("--cutoff", help="YYYY-MM-DD (default: from the thesis text, else the corpus as-of date)")
    run.add_argument("--json", action="store_true", help="print the FinalAssessment JSON")
    run.add_argument("--no-evidence", action="store_true", help="hide evidence lines")
    run.set_defaults(func=_cmd_run)

    trace = sub.add_parser("trace", help="print the decision log for a request")
    trace.add_argument("request_id")
    trace.add_argument("--logs-dir")
    trace.add_argument("--json", action="store_true")
    trace.set_defaults(func=_cmd_trace)

    ev = sub.add_parser("eval", help="run the evaluation (args passed to finsight.eval.run_eval)")
    ev.add_argument("rest", nargs=argparse.REMAINDER)
    ev.set_defaults(func=_cmd_eval)

    graph = sub.add_parser("graph", help="export the orchestrator graph as Mermaid")
    graph.add_argument("--out", default="docs/orchestrator_graph.md")
    graph.set_defaults(func=_cmd_graph)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
