"""
Runs the financial agent on the dev eval items, using the lines the earlier pipeline steps picked, and prints
what it returns next to what the eval expects.

Inputs (from the team):
  pre_agent_steps.json     per item: the post lines handed to the agents, their stored types, and the cut-off used
  claims_dev_final.json    per item: the user's prompt and the expected result

Run from the repo root (after the pull):
  python -m finsight.agents.financial.run_eval --steps pre_agent_steps.json --items claims_dev_final.json
  python -m finsight.agents.financial.run_eval ... --ids 11 12 39 40       only these items
  python -m finsight.agents.financial.run_eval ... --spec-cutoff           cut-off from the prompt, else the
                                                                           latest filing (claim_spec.md section 5)
  python -m finsight.agents.financial.run_eval ... --trace                 also print the planner's steps

The model comes from FINSIGHT_LLM_PROVIDER / FINSIGHT_LLM_MODEL (team LLM client). With the default mock
provider the agent uses the rule-based reading. The agent is rules-first: the model is only called when the rules
find no measure (or on a retry). Set FINSIGHT_FINANCIAL_PLANNER=rules to never call it.
"""

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

from .agent import XbrlFinancialAgent

try:
    from finsight.agents.claim_extraction.thesaurus import resolve_ticker
except ImportError:   # pragma: no cover
    def resolve_ticker(text):
        return None


def prompt_cutoff(prompt: str) -> date | None:
    """'Cut-off: 30 Apr 2026' in the prompt, or None (an invalid date such as 31 Feb is None too)."""
    m = re.search(r"cut-?off(?: date)?\s*[:=]?\s*(\d{1,2} \w{3,9} \d{4})", prompt, flags=re.IGNORECASE)
    if not m:
        return None
    for fmt in ("%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(m.group(1), fmt).date()
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="data/test_cases/pre_agent_steps.json")
    ap.add_argument("--items", default="data/test_cases/claims_dev_final.json")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--spec-cutoff", action="store_true")
    ap.add_argument("--trace", action="store_true", help="print the planner's thought / action / result steps")
    args = ap.parse_args()

    steps = json.loads(Path(args.steps).read_text(encoding="utf-8"))
    items = {x["id"]: x for x in json.loads(Path(args.items).read_text(encoding="utf-8"))["items"]}
    agent = XbrlFinancialAgent()
    for step in steps:
        if step.get("actual_step") != "check" or not step.get("lines") or (args.ids and step["id"] not in args.ids):
            continue
        item = items.get(step["id"], {})
        prompt = item.get("prompt", "")
        ticker = resolve_ticker(prompt)
        if args.spec_cutoff:
            cutoff = prompt_cutoff(prompt)
        else:
            cutoff = date.fromisoformat(step["as_of"]) if step.get("as_of") else None
        print(f"\n=== {step['id']} [{item.get('category')}] {ticker} | cut-off {cutoff or 'latest data'}")
        print(f"    prompt:   {prompt}")
        print(f"    expected: {item.get('expected_result')}")
        for i, (line, kind) in enumerate(zip(step["lines"], step["types"]), 1):
            sub = SimpleNamespace(sub_claim_id=f"{step['id']}-{i}", text=line, company_ticker=ticker,
                                  sub_claim_type=SimpleNamespace(name=kind), measure=None, user_prompt=prompt)
            found = agent.gather(sub, cutoff, None)
            print(f"  line {i} ({kind}) [{agent.modes.get(sub.sub_claim_id)}]: {line[:150]}")
            if args.trace and sub.sub_claim_id in agent.traces:
                print("    " + agent.traces[sub.sub_claim_id].replace("\n", "\n    "))
            for ev in found:
                print(f"    -> {ev.metric_name} = {ev.computed_value}  (comparison "
                      f"{getattr(ev, 'comparison_value', None)}, {getattr(ev, 'comparison_period', None)})"
                      f"\n       {ev.content[:220]}")
            if sub.sub_claim_id in agent.reasons:
                print(f"    note: {agent.reasons[sub.sub_claim_id][:300]}")


if __name__ == "__main__":
    main()
