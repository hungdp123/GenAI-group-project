"""Step-by-step correctness of a FinSight eval run: where does each gold sub-claim first go wrong?

    python scripts/step_check.py results/financial/<timestamp> finsight/eval/datasets/claims_dev_financial.jsonl
    python scripts/step_check.py <run> <dataset> --out steps.md        (also writes steps.csv in the run folder)

For every gold sub-claim it checks, in pipeline order:
  1 found     the system checked a sub-claim matching the gold quote (token overlap >= --min-match)
  2 type      that sub-claim's type equals the gold type
  3 route     the routing plan sent it to the agent(s) the knowledge layer lists for the gold type
  4 figure    the evidence holds the gold figure: a percentage within --pct-tol points, or an amount within 1%.
              Gold with no evidence = the system should report no figure. Skipped when the gold names no number.
  5 period    the evidence is for the gold period (a "quarter ended YYYY-MM-DD" date, Qn YYYY or FY YYYY)
  6 cut-off   no evidence published after the cut-off, and no temporal violation in the decision log
  7 verdict   the final verdict equals the gold verdict
The first failed step is the one to fix: a wrong verdict after a wrong figure is a figure problem, not an
Auditor problem. Gold evidence is free text, so steps 4 and 5 are best-effort; read the "detail" column.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml

STEPS = ["found", "type", "route", "figure", "period", "cut-off", "verdict"]
_KL = Path(__file__).resolve().parents[1] / "finsight" / "knowledge" / "knowledge_layer.yaml"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower())) - {"the", "a", "is", "s", "its", "of", "from", "by"}


def _jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def routes() -> tuple[dict[str, list[str]], list[str]]:
    kl = yaml.safe_load(_KL.read_text(encoding="utf-8"))
    return {r["sub_claim_type"]: r.get("collectors") or [] for r in kl["rules"]}, kl["fallback"].get("collectors") or []


# -- reading the gold evidence (free text) --------------------------------------------------------------------
_PCT = re.compile(r"([+-−]?\d+(?:\.\d+)?)\s*%")
_AMOUNT = re.compile(r"(-|−)?\$\(?([\d,]+(?:\.\d+)?)\)?\s*(m|bn|b|billion|million)?", re.I)
_DATE = re.compile(r"\b(20\d\d-\d\d-\d\d)\b")
_QTR = re.compile(r"\bQ([1-4])\s*(?:FY)?\s*(20\d\d)\b", re.I)
_FY = re.compile(r"\b(?:FY|fiscal(?: year)?|full[- ]year)\s*(20\d\d)\b", re.I)


def gold_figures(evidence: list[str]) -> tuple[list[float], list[float], list[str]]:
    """(percentages as fractions, amounts in $ millions, period hints) named in the gold evidence."""
    text = " ".join(evidence)
    text_no_filed = re.sub(r"\(?\b(10-[KQ]s?|filed|release)\b[^)]*\)?", " ", text, flags=re.I)
    pcts = [float(m.group(1).replace("−", "-")) / 100 for m in _PCT.finditer(text_no_filed)]
    amounts = []
    for m in _AMOUNT.finditer(text_no_filed):
        value = float(m.group(2).replace(",", ""))
        unit = (m.group(3) or "m").lower()
        value *= 1000 if unit in ("bn", "b", "billion") else 1
        if m.group(0).strip().startswith(("-", "−")) or "(" in m.group(0):
            value = -value
        if abs(value) >= 1:
            amounts.append(value)
    periods = _DATE.findall(text_no_filed) + [f"Q{q} {y}" for q, y in _QTR.findall(text_no_filed)] \
        + [f"FY{y}" for y in _FY.findall(text_no_filed)]
    return pcts, amounts, periods


def _evidence_numbers(item: dict) -> tuple[list[float], list[float]]:
    """(fractions, $ millions) an evidence item states: computed_value, comparison_value, and amounts in its text."""
    fracs, amounts = [], []
    for key in ("computed_value", "comparison_value"):
        v = item.get(key)
        if isinstance(v, (int, float)):
            (fracs if abs(v) < 20 else amounts).append(v if abs(v) < 20 else v / 1e6)
    for m in re.finditer(r"(-)?\$([\d,]+(?:\.\d+)?)m\b", item.get("content") or ""):
        amounts.append((-1 if m.group(1) else 1) * float(m.group(2).replace(",", "")))
    for m in _PCT.finditer(item.get("content") or ""):
        fracs.append(float(m.group(1).replace("−", "-")) / 100)
    return fracs, amounts


def _period_ok(hints: list[str], items: list[dict]) -> tuple[bool | None, str]:
    if not hints:
        return None, "gold names no period"
    seen = " | ".join(f"{i.get('period') or ''} {i.get('comparison_period') or ''}" for i in items)
    for h in hints:
        if re.match(r"20\d\d-\d\d-\d\d", h) and h in seen:
            return True, f"period {h}"
        q = re.match(r"Q([1-4]) (20\d\d)", h)
        if q:
            # a calendar quarter: the period end month must fall in that quarter of that year (fiscal calendars vary)
            for d in _DATE.findall(seen):
                y, m = int(d[:4]), int(d[5:7])
                if (y == int(q.group(2)) and (m - 1) // 3 + 1 == int(q.group(1))) or \
                        (y == int(q.group(2)) and abs(m - int(q.group(1)) * 3) <= 1):
                    return True, f"period {d} ~ {h}"
        fy = re.match(r"FY(20\d\d)", h)
        if fy and "year ended" in seen and any(d[:4] in (fy.group(1), str(int(fy.group(1)) + 1))
                                                for d in _DATE.findall(seen)):
            return True, f"annual period ~ {h}"
    return False, f"gold period {', '.join(hints)}; evidence: {seen.strip()[:120] or 'none'}"


def check_item(item: dict, fa: dict | None, log: dict | None, route_of, fallback, min_match: float,
               pct_tol: float) -> list[dict]:
    rows = []
    results = (fa or {}).get("sub_claim_results") or []
    plan = (log or {}).get("routing_plan") or []
    plan_by_id = {p["sub_claim_id"]: p for p in plan}
    cutoff = (fa or {}).get("as_of_date") or (log or {}).get("cutoff_date")
    violations = (log or {}).get("temporal_violations") or []
    used = set()
    for gi, gold in enumerate(item.get("gold_sub_claims") or []):
        row = {"item_id": item["item_id"], "gold_idx": gi, "gold_text": gold["text"][:140],
               "gold_type": gold.get("sub_claim_type"), "gold_verdict": gold.get("gold_verdict")}
        detail = {}
        best = max(((i, _jaccard(gold["text"], r["sub_claim"]["text"])) for i, r in enumerate(results) if i not in used),
                   key=lambda t: t[1], default=(None, 0.0))
        if best[0] is None or best[1] < min_match:
            row.update({s: "-" for s in STEPS}, found="FAIL", first_fail="found",
                       detail=("no output for this item in raw/finsight" if fa is None else
                               f"no checked sub-claim matches (best overlap {best[1]:.2f}); run status {fa.get('status')}"
                               + (f", question: {fa.get('clarification_question')[:80]}" if fa.get('clarification_question') else "")
                               + (f", refusal: {fa.get('refusal_reason')[:80]}" if fa.get('refusal_reason') else "")))
            rows.append(row)
            continue
        used.add(best[0])
        res = results[best[0]]
        sub = res["sub_claim"]
        evidence = (res.get("supporting") or []) + (res.get("challenging") or [])
        row["predicted_text"] = sub["text"][:140]
        row["predicted_type"] = sub["sub_claim_type"]
        row["predicted_verdict"] = res.get("verdict")
        status = {"found": True}
        detail["found"] = f"overlap {best[1]:.2f}"
        status["type"] = sub["sub_claim_type"] == gold.get("sub_claim_type")
        detail["type"] = f"{sub['sub_claim_type']} vs gold {gold.get('sub_claim_type')}"
        want = route_of.get(gold.get("sub_claim_type"), fallback)
        got = plan_by_id.get(sub["sub_claim_id"], {}).get("collectors")
        status["route"] = None if got is None else sorted(got) == sorted(want)
        detail["route"] = f"{got} vs expected {want}"
        numeric = [e for e in evidence if e.get("computed_value") is not None]
        pcts, amounts, hints = gold_figures(gold.get("evidence") or [])
        if not gold.get("evidence"):
            status["figure"] = not numeric
            detail["figure"] = "gold expects no figure; " + (f"system reported {len(numeric)}" if numeric else "none reported")
        elif not (pcts or amounts):
            status["figure"] = None
            detail["figure"] = "gold names no number"
        else:
            fr, am = [], []
            for e in numeric:
                f, a = _evidence_numbers(e)
                fr += f
                am += a
            hit_p = [p for p in pcts if any(abs(p - x) <= pct_tol for x in fr)]
            hit_a = [a for a in amounts if any(abs(a - x) <= max(1.0, 0.01 * abs(a)) for x in am)]
            status["figure"] = bool(hit_p or hit_a)
            detail["figure"] = (f"matched {[f'{p:+.1%}' for p in hit_p] + [f'${a:,.0f}m' for a in hit_a]}" if status["figure"]
                                else f"gold {[f'{p:+.1%}' for p in pcts]} {[f'${a:,.0f}m' for a in amounts]}; "
                                     f"system {[e.get('metric_name') + '=' + str(round(e['computed_value'], 4)) for e in numeric] or 'no figure'}")
        status["period"], detail["period"] = _period_ok(hints, numeric) if numeric else (None, "no figure")
        late = [e["evidence_id"] for e in evidence if cutoff and e.get("publication_date") and e["publication_date"] > cutoff]
        mine = [v for v in violations if v.get("sub_claim_id") == sub["sub_claim_id"]]
        status["cut-off"] = not late and not mine
        detail["cut-off"] = f"cut-off {cutoff}; late evidence {late + [v['evidence_id'] for v in mine] or 'none'}"
        status["verdict"] = res.get("verdict") == gold.get("gold_verdict")
        detail["verdict"] = f"{res.get('verdict')} vs gold {gold.get('gold_verdict')}"
        for s in STEPS:
            row[s] = {True: "ok", False: "FAIL", None: "n/a"}[status.get(s)]
        row["first_fail"] = next((s for s in STEPS if status.get(s) is False), "")
        row["detail"] = " | ".join(f"{s}: {detail[s]}" for s in STEPS if s in detail and status.get(s) is not True) \
            or "all steps ok"
        rows.append(row)
    return rows


def load_run(run: Path) -> tuple[dict[str, dict], list[dict]]:
    raws = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (run / "raw" / "finsight").glob("*.json")}
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in (run / "logs").glob("*.json")] \
        if (run / "logs").is_dir() else []
    return raws, logs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--min-match", type=float, default=0.3)
    ap.add_argument("--pct-tol", type=float, default=0.005, help="percentage tolerance as a fraction (0.005 = 0.5 points)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    route_of, fallback = routes()
    raws, logs = load_run(args.run)
    logs_by_request = {lg.get("request_id"): lg for lg in logs}
    rows = []
    for line in args.dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        fa = raws.get(item["item_id"])
        log = logs_by_request.get((fa or {}).get("request_id"))
        rows += check_item(item, fa, log, route_of, fallback, args.min_match, args.pct_tol)

    fields = ["item_id", "gold_idx", "first_fail", *STEPS, "gold_type", "predicted_type", "gold_verdict",
              "predicted_verdict", "gold_text", "predicted_text", "detail"]
    with (args.run / "steps.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    n = len(rows) or 1
    out = [f"# Step check: `{args.run}`", "", f"{len(rows)} gold sub-claims from `{args.dataset}`.", "",
           "| Step | ok | FAIL | n/a | Pass rate (of checked) |", "|---|---|---|---|---|"]
    for s in STEPS:
        c = Counter(r.get(s, "-") for r in rows)
        checked = c["ok"] + c["FAIL"]
        out.append(f"| {s} | {c['ok']} | {c['FAIL']} | {c['n/a'] + c['-']} | "
                   f"{(c['ok'] / checked if checked else 0):.0%} |")
    first = Counter(r["first_fail"] for r in rows if r["first_fail"])
    out += ["", f"End to end (every checked step ok): {sum(not r['first_fail'] for r in rows)}/{len(rows)} "
                f"({sum(not r['first_fail'] for r in rows) / n:.0%})", "",
            "Where sub-claims first go wrong: " + (", ".join(f"{s} {first[s]}" for s in STEPS if first[s]) or "nowhere"),
            "", "## Sub-claims with a failed step", ""]
    by_step = defaultdict(list)
    for r in rows:
        if r["first_fail"]:
            by_step[r["first_fail"]].append(r)
    for s in STEPS:
        for r in by_step.get(s, []):
            out.append(f"- **{s}** `{r['item_id']}#{r['gold_idx']}` {r['gold_text'][:90]}  \n  {r['detail']}")
    text = "\n".join(out) + "\n\nPer sub-claim: `steps.csv` in the run folder.\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
