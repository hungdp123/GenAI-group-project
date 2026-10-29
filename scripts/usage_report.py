"""Latency and token usage of an eval run: totals, per item, per agent.

    python scripts/usage_report.py results/financial/<timestamp>
    python scripts/usage_report.py results/financial/<timestamp> --out usage.md

Reads records.jsonl (one row per system x item: wall-clock latency, input / output tokens, cost) and, for FinSight,
the decision logs in logs/ (one AgentCall per agent call: agent, latency, tokens, sub-claim).
- Item latency is wall-clock for the whole run (agents run in parallel, so per-agent latencies add up to more).
- Tokens are counted only for real model calls; in mock mode they are estimates of the canned replies.
- Writes usage_items.csv and usage_agents.csv next to the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    pos = (len(v) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def load(run: Path) -> tuple[list[dict], list[dict]]:
    records = [json.loads(line) for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run / "logs").glob("*.json"))] \
        if (run / "logs").is_dir() else []
    return records, logs


def report(run: Path) -> str:
    records, logs = load(run)
    out = [f"# Latency and tokens: `{run}`", ""]

    # -- per system ------------------------------------------------------------------------------------------
    out += ["| System | Items | Latency p50 (s) | p95 (s) | max (s) | total (s) | Input tokens | Output tokens | "
            "Total tokens | Tokens / item | Cost (USD) |", "|---" * 11 + "|"]
    by_system = defaultdict(list)
    for r in records:
        by_system[r["system"]].append(r)
    for system, rows in by_system.items():
        lat = [r["latency_ms"] / 1000 for r in rows]
        tin, tout = sum(r["input_tokens"] for r in rows), sum(r["output_tokens"] for r in rows)
        out.append(f"| {system} | {len(rows)} | {pct(lat, 50):.2f} | {pct(lat, 95):.2f} | {max(lat, default=0):.2f} | "
                   f"{sum(lat):.1f} | {tin:,} | {tout:,} | {tin + tout:,} | {(tin + tout) / (len(rows) or 1):,.0f} | "
                   f"${sum(r['cost_usd'] for r in rows):.4f} |")

    # -- per item ---------------------------------------------------------------------------------------------
    out += ["", "## Per item (slowest first)", "", "| System | Item | Latency (s) | Agent calls | Tokens in | Tokens out | "
            "Cost (USD) | Overall |", "|---" * 8 + "|"]
    for r in sorted(records, key=lambda r: -r["latency_ms"]):
        out.append(f"| {r['system']} | {r['item_id']} | {r['latency_ms'] / 1000:.2f} | {r['agent_calls']} | "
                   f"{r['input_tokens']:,} | {r['output_tokens']:,} | ${r['cost_usd']:.4f} | {r.get('overall_verdict') or '-'} |")
    with (run / "usage_items.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "item_id", "latency_ms", "agent_calls", "tool_calls", "input_tokens", "output_tokens",
                    "cost_usd", "overall_verdict"])
        for r in records:
            w.writerow([r["system"], r["item_id"], r["latency_ms"], r["agent_calls"], r["tool_calls"],
                        r["input_tokens"], r["output_tokens"], r["cost_usd"], r.get("overall_verdict")])

    # -- per agent (FinSight decision logs) ----------------------------------------------------------------------
    if logs:
        agg = defaultdict(lambda: {"calls": 0, "lat": [], "in": 0, "out": 0, "cost": 0.0, "fail": 0})
        for log in logs:
            for c in log.get("agent_calls", []):
                a = agg[c["agent"]]
                a["calls"] += 1
                a["lat"].append(c["latency_ms"] / 1000)
                a["in"] += c.get("input_tokens", 0)
                a["out"] += c.get("output_tokens", 0)
                a["cost"] += c.get("cost_usd", 0.0)
                a["fail"] += not c.get("success", True)
        total_tokens = sum(a["in"] + a["out"] for a in agg.values()) or 1
        out += ["", f"## Per agent (FinSight, {len(logs)} decision logs)", "",
                "| Agent | Calls | Mean latency (s) | p95 (s) | Total latency (s) | Input tokens | Output tokens | "
                "Share of tokens | Cost (USD) | Failed |", "|---" * 10 + "|"]
        for name, a in sorted(agg.items(), key=lambda kv: -(kv[1]["in"] + kv[1]["out"])):
            out.append(f"| {name} | {a['calls']} | {statistics.fmean(a['lat']):.2f} | {pct(a['lat'], 95):.2f} | "
                       f"{sum(a['lat']):.1f} | {a['in']:,} | {a['out']:,} | {(a['in'] + a['out']) / total_tokens:.0%} | "
                       f"${a['cost']:.4f} | {a['fail']} |")
        with (run / "usage_agents.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["request_id", "input", "agent", "sub_claim_id", "latency_ms", "input_tokens", "output_tokens",
                        "cost_usd", "tool_calls", "success"])
            for log in logs:
                for c in log.get("agent_calls", []):
                    w.writerow([log.get("request_id"), (log.get("input") or "")[:120], c["agent"], c.get("sub_claim_id"),
                                c["latency_ms"], c.get("input_tokens", 0), c.get("output_tokens", 0),
                                c.get("cost_usd", 0.0), c.get("tool_calls", 0), c.get("success", True)])
    out += ["", "Item latency is wall-clock per run; per-agent latencies overlap because agents run in parallel.",
            "Files: usage_items.csv (one row per item), usage_agents.csv (one row per agent call)."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="an eval run folder, e.g. results/financial/20261028-210000")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    text = report(args.run)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
