"""Verdict accuracy, Macro-F1 and a confusion matrix from an eval run's sub_claims.csv.

    python scripts/score_sub_claims.py results/<timestamp>/sub_claims.csv
    python scripts/score_sub_claims.py results/<timestamp>/sub_claims.csv --min-match 0.3 --out score.md

Scoring rules (match finsight/eval/metrics.py):
- One row = one gold sub-claim, paired with the system's closest predicted sub-claim (token Jaccard).
- A gold sub-claim the system did not produce is predicted MISSING and counts as wrong.
- --min-match treats a pairing below that similarity as MISSING (default 0: keep run_eval's pairing).
- UNLABELED gold rows are skipped.
- Macro-F1 is the unweighted mean of per-verdict F1 over the four gold verdicts.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

VERDICTS = ["SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"]
SHORT = {"SUPPORTED": "SUP", "PARTIALLY_SUPPORTED": "PART", "CONTRADICTED": "CONTR",
         "INSUFFICIENT_EVIDENCE": "INSUF", "MISSING": "MISS"}


def score(rows: list[dict], min_match: float = 0.0) -> dict:
    pairs = []
    for r in rows:
        gold = r["gold_verdict"]
        if gold not in VERDICTS:
            continue
        pred = r["predicted_verdict"] or "MISSING"
        if pred != "MISSING" and float(r.get("match_score") or 0) < min_match:
            pred = "MISSING"
        pairs.append((gold, pred, r))
    n = len(pairs)
    correct = sum(g == p for g, p, _ in pairs)
    per = {}
    for v in VERDICTS:
        tp = sum(g == v and p == v for g, p, _ in pairs)
        fp = sum(g != v and p == v for g, p, _ in pairs)
        fn = sum(g == v and p != v for g, p, _ in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[v] = dict(support=tp + fn, precision=prec, recall=rec, f1=f1)
    present = [v for v in VERDICTS if per[v]["support"]]
    macro = sum(per[v]["f1"] for v in present) / len(present) if present else 0.0
    confusion = Counter((g, p) for g, p, _ in pairs)
    # "Dangerous" errors: telling the user a claim is supported when gold says contradicted, and the reverse.
    flips = sum(1 for g, p, _ in pairs if {g, p} == {"SUPPORTED", "CONTRADICTED"})
    return dict(n=n, accuracy=correct / n if n else 0.0, macro_f1=macro, per=per, confusion=confusion,
                missing=sum(p == "MISSING" for _, p, _ in pairs), flips=flips,
                wrong=[r for g, p, r in pairs if g != p])


def report(path: Path, min_match: float) -> str:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_system = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
    out = [f"# Verdict scoring: `{path}`", "",
           "| System | Gold sub-claims | Accuracy | Macro-F1 | MISSING | SUPPORTED<->CONTRADICTED flips |",
           "|---|---|---|---|---|---|"]
    scored = {s: score(r, min_match) for s, r in by_system.items()}
    for s, m in scored.items():
        out.append(f"| {s} | {m['n']} | {m['accuracy']:.0%} | {m['macro_f1']:.2f} | {m['missing']} | {m['flips']} |")
    for s, m in scored.items():
        out += ["", f"## {s}", "", "| Gold verdict | Support | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        out += [f"| {v} | {m['per'][v]['support']} | {m['per'][v]['precision']:.2f} | {m['per'][v]['recall']:.2f} | "
                f"{m['per'][v]['f1']:.2f} |" for v in VERDICTS]
        cols = VERDICTS + ["MISSING"]
        out += ["", "Confusion (rows = gold, columns = predicted):", "",
                "| gold \\ pred | " + " | ".join(SHORT[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        out += [f"| {SHORT[g]} | " + " | ".join(str(m['confusion'].get((g, p), 0)) for p in cols) + " |"
                for g in VERDICTS]
        if m["wrong"]:
            out += ["", "Wrong or missing:", ""]
            out += [f"- `{r['item_id']}` gold {SHORT[r['gold_verdict']]} -> {SHORT.get(r['predicted_verdict'] or 'MISSING', r['predicted_verdict'])}"
                    f" (match {r['match_score']}): {r['gold_text'][:100]}" for r in m["wrong"]]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--min-match", type=float, default=0.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    text = report(args.csv, args.min_match)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
