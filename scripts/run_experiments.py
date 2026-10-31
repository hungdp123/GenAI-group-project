"""Run the FinSight experiment suite on the DEV split and write one report: results/experiments/experiments.md.

    python scripts/run_experiments.py                       # everything (paid mode; a few dollars)
    python scripts/run_experiments.py --only matcher_rules,matcher_llm
    python scripts/run_experiments.py --reuse main=results/20261031-120000   # use a finished run
    python scripts/run_experiments.py --report-only         # rebuild experiments.md from what is already there

Every experiment runs on dev, never on held-out (held-out stays the one-time final test). Each one is an ordinary
`finsight eval` run with one setting changed; its full output stays in results/experiments/<name>/<timestamp>/.

  main             all five systems, gpt-5.4                  architecture, RAG vs RAG + quote, safety
  nano             FinSight, plain LLM, LLM + filings, nano   model strength (compare with main)
  retries_off      FinSight, gpt-5.4, FINSIGHT_MAX_RETRIES=0  does the auditor's retry help? (compare with main)
  no_period_rules  plain LLM, LLM + filings, gpt-5.4, old prompt without period rules (compare with main)
  matcher_rules    claim matching only, rule-based pick       (free)
  matcher_llm      claim matching only, LLM pick              (compare with matcher_rules)
  agents           each agent on its own (decomposer, financial, narrative, retriever, auditor, guardrails)

RAG and RAG + quote need the narrative corpus (data/narrative/processed/chunking_strategies/C2/full_v1).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "experiments"
BIG, SMALL = "gpt-5.4", "gpt-5.4-nano"
NAMES = {"finsight": "FinSight", "llm": "Plain LLM", "direct": "LLM + filings", "rag": "Basic RAG",
         "rag_quote": "RAG + quote"}

EXPERIMENTS: dict[str, dict] = {
    "main": {"args": ["--system", "finsight,llm,direct,rag,rag_quote"], "env": {"FINSIGHT_LLM_MODEL": BIG}},
    "nano": {"args": ["--system", "finsight,llm,direct"],
             "env": {"FINSIGHT_LLM_MODEL": SMALL, "FINSIGHT_AUDITOR_MODEL": SMALL}},
    "retries_off": {"args": ["--system", "finsight"],
                    "env": {"FINSIGHT_LLM_MODEL": BIG, "FINSIGHT_MAX_RETRIES": "0"}},
    "no_period_rules": {"args": ["--system", "llm,direct"],
                        "env": {"FINSIGHT_LLM_MODEL": BIG, "FINSIGHT_BASELINE_PERIOD_RULES": "0"}},
    "matcher_rules": {"args": ["--matcher"], "env": {"FINSIGHT_MATCHER": "rules"}},
    "matcher_llm": {"args": ["--matcher"], "env": {"FINSIGHT_MATCHER": "llm", "FINSIGHT_LLM_MODEL": BIG}},
    "agents": {"args": ["--agents", "all"], "env": {"FINSIGHT_LLM_MODEL": BIG}},
}


# -- running ---------------------------------------------------------------------------------------------


def run(name: str) -> None:
    spec = EXPERIMENTS[name]
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **spec["env"]}
    cmd = [sys.executable, "-m", "finsight.eval.run_eval", *spec["args"], "--split", "dev", "--out", str(out)]
    settings = " ".join(f"{k}={v}" for k, v in spec["env"].items())
    print(f"\n=== {name}: {' '.join(cmd[3:])}   [{settings}]", flush=True)
    result = subprocess.run(cmd, env=env, cwd=ROOT)
    if result.returncode != 0:
        print(f"!!! {name} exited with code {result.returncode}; the report will show it as missing.", flush=True)


def latest(name: str, reuse: dict[str, Path]) -> Path | None:
    if name in reuse:
        return reuse[name]
    folders = sorted((p for p in (OUT / name).glob("*") if p.is_dir()), key=lambda p: p.name)
    return folders[-1] if folders else None


# -- reading a finished run -----------------------------------------------------------------------------


def comparison(folder: Path | None) -> dict:
    path = folder / "comparison.json" if folder else None
    if path and not path.exists() and (folder / "raw").is_dir():
        # A one-system run (retries_off) writes no comparison; score it the same way as the others.
        subprocess.run([sys.executable, "-m", "finsight.eval.run_eval", "--compare", str(folder), "--split", "dev"],
                       cwd=ROOT)
    return json.loads(path.read_text(encoding="utf-8"))["summary"] if path and path.exists() else {}


def records(folder: Path | None) -> dict[str, list[dict]]:
    path = folder / "records.jsonl" if folder else None
    out: dict[str, list[dict]] = {}
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            out.setdefault(rec["system"], []).append(rec)
    return out


def matcher_rows(folder: Path | None) -> list[dict]:
    if folder is None:
        return []
    files = sorted(folder.glob("match-*/match_rows.jsonl")) or sorted(folder.glob("match_rows.jsonl"))
    if not files:
        return []
    return [json.loads(line) for line in files[-1].read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def frac(pair) -> str:
    return f"{pair[0]}/{pair[1]}" if pair else "-"


def cost_latency(recs: list[dict]) -> tuple[str, str]:
    if not recs:
        return "-", "-"
    cost = sum(r.get("cost_usd") or 0 for r in recs) / len(recs)
    lat = sum(r.get("latency_ms") or 0 for r in recs) / len(recs) / 1000
    return f"${cost:.4f}", f"{lat:.1f} s"


def system_row(label: str, s: dict, recs: list[dict]) -> str:
    cost, lat = cost_latency(recs)
    routes = s.get("by_route", {})
    return (f"| {label} | {pct(s.get('fact_accuracy'))} (n={s.get('fact_n', 0)}) | "
            f"{frac(routes.get('financial'))} | {frac(routes.get('narrative'))} | "
            f"{pct(s.get('headline_accuracy'))} | {pct(s.get('overclaim_rate'))} (n={s.get('overclaim_n', 0)}) | "
            f"{pct(s.get('made_up_rate'))} | {cost} | {lat} |")


HEAD = ("| System | Facts right | Financial facts | Narrative facts | Headline right | Overclaim | Made-up figures | "
        "Cost / query | Latency / query |\n|---|---|---|---|---|---|---|---|---|")


# -- the report -----------------------------------------------------------------------------------------


def report(reuse: dict[str, Path]) -> str:
    folders = {name: latest(name, reuse) for name in EXPERIMENTS}
    main_c, main_r = comparison(folders["main"]), records(folders["main"])
    lines = ["# FinSight experiments (dev split)", "",
             "All experiments run on the dev claims; the held-out claims are kept for the one final test. "
             "Fact accuracy counts labelled facts only; a fact a system did not answer counts as wrong. "
             "Small samples: treat differences of a few facts as noise (see the paired tests in each run's "
             "comparison.md).", "",
             "Runs used: " + ", ".join(f"{n} = `{f.relative_to(ROOT) if f and f.is_relative_to(ROOT) else f}`"
                                       for n, f in folders.items() if f) + ".", ""]

    # 1. Architecture
    lines += ["## 1. Does the architecture help? FinSight vs baselines (gpt-5.4)", ""]
    if main_c:
        lines += [HEAD] + [system_row(NAMES[s], main_c[s], main_r.get(s, [])) for s in NAMES if s in main_c]
        vs = [(NAMES[s], main_c[s]["vs_finsight"]) for s in NAMES if s in main_c and "vs_finsight" in main_c[s]]
        if vs:
            lines += ["", "Paired test against FinSight (exact McNemar, same facts):", "",
                      "| Baseline | FinSight right, baseline wrong | Baseline right, FinSight wrong | p |",
                      "|---|---|---|---|"]
            lines += [f"| {n} | {v['finsight_only_right']} | {v['baseline_only_right']} | {v['p_value']:.3f} |"
                      for n, v in vs]
    else:
        lines += ["_Not run yet: `python scripts/run_experiments.py --only main`._"]

    # 2. Model strength
    nano_c, nano_r = comparison(folders["nano"]), records(folders["nano"])
    lines += ["", "## 2. How does model strength change things? gpt-5.4-nano vs gpt-5.4 (same claims)", ""]
    if nano_c and main_c:
        lines += [HEAD]
        for s in ("finsight", "llm", "direct"):
            if s in nano_c:
                lines.append(system_row(f"{NAMES[s]}, nano", nano_c[s], nano_r.get(s, [])))
            if s in main_c:
                lines.append(system_row(f"{NAMES[s]}, gpt-5.4", main_c[s], main_r.get(s, [])))
        lines += ["", "Read: does the gap between FinSight and LLM + filings shrink with the stronger model? "
                  "If so, FinSight's advantage is grounding (cited, computed figures; no made-up numbers) more "
                  "than raw accuracy."]
    else:
        lines += ["_Needs `main` and `nano`._"]

    # 3. Design choices
    lines += ["", "## 3. Which design choices matter?", ""]
    rules, llm = matcher_rows(folders["matcher_rules"]), matcher_rows(folders["matcher_llm"])
    lines += ["### 3a. Claim matcher: rules vs LLM pick", ""]
    if rules or llm:
        lines += ["| Matcher | Right line / right question | Facts in the right bucket | Cost (all items) | "
                  "Latency / item |", "|---|---|---|---|---|"]
        for label, rows in (("Rules", rules), ("LLM pick (gpt-5.4)", llm)):
            if not rows:
                lines.append(f"| {label} | not run | | | |")
                continue
            facts = sum(r["alignment_total"] for r in rows)
            right = sum(r["alignment_correct"] for r in rows)
            lines.append(f"| {label} | {sum(r['correct'] for r in rows)}/{len(rows)} "
                         f"({sum(r['correct'] for r in rows) / len(rows):.0%}) | {right}/{facts} | "
                         f"${sum(r.get('cost_usd', 0) for r in rows):.4f} | "
                         f"{sum(r.get('latency_ms', 0) for r in rows) / len(rows):.0f} ms |")
    else:
        lines += ["_Needs `matcher_rules` and `matcher_llm`._"]

    off_c, off_r = comparison(folders["retries_off"]), records(folders["retries_off"])
    lines += ["", "### 3b. Auditor retry: on (normal) vs off", ""]
    if off_c and "finsight" in main_c:
        lines += [HEAD, system_row("FinSight, retry on", main_c["finsight"], main_r.get("finsight", [])),
                  system_row("FinSight, retry off", off_c["finsight"], off_r.get("finsight", []))]
        retried = [r for r in main_r.get("finsight", [])]
        if retried:
            share = sum(r.get("sub_claims_retried", 0) for r in retried) / max(
                1, sum(r.get("sub_claims_assessed", 0) for r in retried))
            lines += ["", f"With retry on, {share:.0%} of sub-claims were re-gathered once after the auditor "
                      "rejected the first evidence."]
    else:
        lines += ["_Needs `main` and `retries_off`._"]

    np_c, np_r = comparison(folders["no_period_rules"]), records(folders["no_period_rules"])
    lines += ["", "### 3c. Baseline prompt: with vs without the period rules", ""]
    if np_c and main_c:
        lines += ["| System | Facts right | Facts with no verdict | Items where it asked a question | "
                  "Headline right |", "|---|---|---|---|---|"]
        for s in ("llm", "direct"):
            for label, c, r in (("with period rules", main_c, main_r), ("without", np_c, np_r)):
                if s not in c:
                    continue
                asked = sum(1 for x in r.get(s, []) if x.get("clarified") and not x.get("expect_clarification"))
                lines.append(f"| {NAMES[s]}, {label} | {pct(c[s].get('fact_accuracy'))} | "
                             f"{c[s].get('unanswered', '-')} of {c[s].get('fact_n', 0)} | {asked} | "
                             f"{pct(c[s].get('headline_accuracy'))} |")
        lines += ["", "FinSight applies these rules itself (knowledge layer, sections 11-12); the baselines only "
                  "get them in the prompt. The comparison in section 1 gives the baselines the rules."]
    else:
        lines += ["_Needs `main` and `no_period_rules`._"]

    # 4. RAG vs RAG + quote
    lines += ["", "## 4. Retrieval: Basic RAG vs RAG + the influencer's line", ""]
    if "rag" in main_c and "rag_quote" in main_c:
        lines += [HEAD, system_row(NAMES["rag"], main_c["rag"], main_r.get("rag", [])),
                  system_row(NAMES["rag_quote"], main_c["rag_quote"], main_r.get("rag_quote", [])),
                  system_row(NAMES["finsight"], main_c["finsight"], main_r.get("finsight", [])), "",
                  "Same retriever (C2 chunks, hybrid search) for all three. RAG + quote adds only the influencer's "
                  "exact line; FinSight adds claim matching, routing, calculation and the auditor. The chunking "
                  "and search-method comparison (C1-C4 x BM25 / dense / hybrid / R4) is Stella's separate study."]
    else:
        lines += ["_Needs `main` with the narrative corpus installed._"]

    # Steps inside FinSight
    lines += ["", "## Steps inside FinSight", "",
              "End-to-end accuracy hides where an error starts, so each step is also scored on its own:", ""]
    fin = main_c.get("finsight", {})
    rec = main_r.get("finsight", [])
    matched = [r for r in rec if r.get("match_correct") is not None]
    facts = sum(r.get("alignment_total") or 0 for r in rec)
    lines += ["| Step | What is checked | Result (main run) |", "|---|---|---|",
              f"| 1. Claim matching | the right influencer line, or the right question | "
              f"{sum(1 for r in matched if r['match_correct'])}/{len(matched)} |",
              f"| 2. Fact alignment | each fact in the right bucket (same / different / also said / not said) | "
              f"{sum(r.get('alignment_correct') or 0 for r in rec)}/{facts} |",
              f"| 3. Routing | each labelled fact sent to the agent its gold route names | "
              f"{frac(fin.get('routing'))} |",
              f"| 4. Verdict | each labelled fact's verdict | {pct(fin.get('fact_accuracy'))} |",
              f"| 5. Headline | the overall verdict for the facts the user asked about | "
              f"{pct(fin.get('headline_accuracy'))} |"]
    agents_md = sorted((OUT / "agents").glob("*/agents/summary.md")) if not reuse.get("agents") else \
        sorted(Path(reuse["agents"]).glob("**/summary.md"))
    if agents_md:
        text = agents_md[-1].read_text(encoding="utf-8")
        table = text[text.index("| Agent |"):].split("\n\n")[0] if "| Agent |" in text else ""
        lines += ["", "Each agent on its own, with its own small test set (details: "
                  f"`{agents_md[-1].relative_to(ROOT) if agents_md[-1].is_relative_to(ROOT) else agents_md[-1]}`):",
                  "", table]
    else:
        lines += ["", "_Per-agent results: run `--only agents`._"]

    # 5. Safety
    lines += ["", "## 5. Is it safe? Refusals, injections and clarifications", ""]
    md = folders["main"] / "comparison.md" if folders["main"] else None
    if md and md.exists() and "Safety and scope" in md.read_text(encoding="utf-8"):
        text = md.read_text(encoding="utf-8")
        block = text[text.index("Safety and scope"):].split("\n\n")
        lines += [block[0], ""] + [b for b in block[1:2]]
    else:
        lines += ["_Needs `main` (the safety table is part of its comparison.md)._"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="comma list of experiments to run: " + ", ".join(EXPERIMENTS))
    parser.add_argument("--reuse", action="append", default=[],
                        help="NAME=RUN_DIR: use a finished `finsight eval` run instead of running NAME")
    parser.add_argument("--report-only", action="store_true", help="only rebuild experiments.md")
    args = parser.parse_args()
    reuse = {}
    for item in args.reuse:
        name, _, folder = item.partition("=")
        if name not in EXPERIMENTS or not Path(folder).is_dir():
            parser.error(f"--reuse {item}: expected one of {', '.join(EXPERIMENTS)} = an existing folder")
        reuse[name] = Path(folder).resolve()
    chosen = [n.strip() for n in args.only.split(",")] if args.only else list(EXPERIMENTS)
    unknown = [n for n in chosen if n not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(unknown)}")
    if not args.report_only:
        for name in chosen:
            if name not in reuse:
                run(name)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "experiments.md"
    path.write_text(report(reuse), encoding="utf-8")
    print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
