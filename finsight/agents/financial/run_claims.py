"""
Runs the financial agent over the stored claims and writes one row per result.

Every stored line is read by the agent (the stored type is only a hint). Each line is checked as of its post's
upload date (the cut-off). There is no user prompt here, so every measure the line states is reported.

Run from the repo root (after the pull):
  python -m finsight.agents.financial.run_claims --claims data/influencers/claims/claims_updated.jsonl --with-number
  python -m finsight.agents.financial.run_claims --claims ... --ticker GOOGL --limit 200

Writes agent_run.csv (or --out) and prints a summary: how many lines got figures and why the rest got none.

There are tens of thousands of lines, so this uses the rule-based reading by default (no model calls).
Add --llm to let the model plan each line (costs money on a real provider; FINSIGHT_SPEND_LIMIT_USD applies).
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from .agent import XbrlFinancialAgent, period_field

# Reasons grouped for the summary: (words in the reason, group name)
_GROUPS = [("segment", "segment or product"), ("Debt is not mapped", "debt to equity"),
           ("debt amount", "debt amount"), ("outside the window", "outside the window"),
           ("could mean either", "latest quarter announced, not filed"),
           ("had not ended", "period not ended by the cut-off"), ("was filed by", "period not filed by the cut-off"),
           ("non-GAAP", "adjusted / non-GAAP"), ("gross profit", "gross profit"), ("run rate", "run rate"),
           ("removes gains", "excluding gains"), ("window allows", "streak longer than window"),
           ("not in the SEC data", "company not in the SEC data"), ("No company", "no company"),
           ("states no figure", "no financial figure in the line")]


def group(reason):
    return next((name for words, name in _GROUPS if words in (reason or "")), "data missing / not computable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", default="data/influencers/claims/claims_updated.jsonl")
    ap.add_argument("--out", default="agent_run.csv")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--type", default=None, help="only lines stored with this type, e.g. HISTORICAL_GROWTH")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with-number", action="store_true",
                    help="only lines that state a figure (a %% or an amount): the ones the Auditor can check")
    ap.add_argument("--llm", action="store_true", help="let the model plan each line (default: rules only)")
    args = ap.parse_args()
    if not args.llm:
        os.environ["FINSIGHT_FINANCIAL_PLANNER"] = "rules"

    agent = XbrlFinancialAgent()
    out_rows, answered, empty, seen = [], 0, Counter(), 0
    for line in Path(args.claims).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get("text") or ""
        if (
            (args.ticker and r.get("company_ticker") != args.ticker)
            or (args.type and r["sub_claim_type"] != args.type)
            or (args.with_number and not re.search(r"\d\s*(%|percent|billion|million|bn\b)", text))
        ):
            continue
        if args.limit and seen >= args.limit:
            break
        seen += 1
        sub = SimpleNamespace(sub_claim_id=r["sub_claim_id"], text=text, company_ticker=r.get("company_ticker"),
                              sub_claim_type=SimpleNamespace(name=r["sub_claim_type"]), measure=r.get("measure"),
                              fields=r.get("fields") or {})
        cutoff = date.fromisoformat(r["upload_date"][:10])
        found = agent.gather(sub, cutoff, None)
        why = agent.reasons.get(sub.sub_claim_id, "")
        base = dict(sub_claim_id=r["sub_claim_id"], ticker=r.get("company_ticker"), stored_type=r["sub_claim_type"],
                    cutoff=cutoff, period_field=period_field(sub) or "", text=text)
        if found:
            answered += 1
            for ev in found:
                out_rows.append({**base, "metric": ev.metric_name, "computed_value": ev.computed_value,
                                 "comparison_value": getattr(ev, "comparison_value", None),
                                 "period": ev.period, "filing": f"{ev.doc_type} filed {ev.publication_date}",
                                 "content": ev.content, "notes": why})
        else:
            empty[group(why)] += 1
            out_rows.append({**base, "metric": "", "computed_value": "", "comparison_value": "", "period": "",
                             "filing": "", "content": "", "notes": why})

    if not out_rows:
        print("No lines matched.")
        return
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"{seen} lines read: {answered} got figures, {seen - answered} got none:")
    for name, n in empty.most_common():
        print(f"    {n:6}  {name}")
    print(f"Every result: {args.out}")


if __name__ == "__main__":
    main()
