"""System-level metrics (Person 5) + per-claim CSV export for Person 3's Accuracy / Macro-F1."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from finsight.contracts.schemas import Verdict


class GoldClaim(BaseModel):
    text: str
    gold_verdict: Verdict


class PredictedClaim(BaseModel):
    text: str
    verdict: Verdict | None


class RunRecord(BaseModel):
    """One (system, item) run, reduced to what the metrics need."""

    system: str
    item_id: str
    split: str
    category: str
    completed: bool
    schema_valid: bool
    all_claims_have_verdict: bool
    refused: bool
    expect_refusal: bool
    task_success: bool
    agent_calls: int
    tool_calls: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    claims_assessed: int = 0
    claims_retried: int = 0
    downgraded_after_retry: int = 0
    overall_verdict: Verdict | None = None
    predicted_claims: list[PredictedClaim] = Field(default_factory=list)
    gold_claims: list[GoldClaim] = Field(default_factory=list)
    error: str | None = None


def task_success(completed: bool, schema_valid: bool, all_have_verdict: bool, refused: bool,
                 expect_refusal: bool) -> bool:
    """Run completed, output validates, every claim has a verdict, refusal behaviour as expected."""
    return completed and schema_valid and all_have_verdict and refused == expect_refusal


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 100]); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(records: list[RunRecord]) -> dict[str, float]:
    """Aggregate metrics for one system."""
    n = len(records) or 1
    latencies = [r.latency_ms for r in records]
    assessed = sum(r.claims_assessed for r in records)
    return {
        "items": len(records),
        "task_success_rate": sum(r.task_success for r in records) / n,
        "refusal_match_rate": sum(r.refused == r.expect_refusal for r in records) / n,
        "avg_agent_calls": sum(r.agent_calls for r in records) / n,
        "avg_tool_calls": sum(r.tool_calls for r in records) / n,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "avg_tokens": sum(r.input_tokens + r.output_tokens for r in records) / n,
        "avg_cost_usd": sum(r.cost_usd for r in records) / n,
        # DECISION: retry rate = claims retried at least once / claims assessed.
        "retry_rate": sum(r.claims_retried for r in records) / assessed if assessed else 0.0,
        "downgraded_after_retry": sum(r.downgraded_after_retry for r in records),
    }


# -- per-claim alignment for Accuracy / Macro-F1 ------------------------------------------


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - {"the", "a", "is", "s", "its", "of", "from", "by"}


def align_claims(gold: list[GoldClaim], predicted: list[PredictedClaim]) -> list[tuple[GoldClaim, PredictedClaim | None, float]]:
    """Greedy one-to-one match of gold to predicted claims by token Jaccard similarity.

    # DECISION: unmatched gold claims are scored as predicted "MISSING" (counts as wrong).
    """
    pairs = sorted(
        ((len(_tokens(g.text) & _tokens(p.text)) / (len(_tokens(g.text) | _tokens(p.text)) or 1), gi, pi)
         for gi, g in enumerate(gold) for pi, p in enumerate(predicted)),
        reverse=True,
    )
    used_g, used_p, match = set(), set(), {}
    for score, gi, pi in pairs:
        if score > 0 and gi not in used_g and pi not in used_p:
            used_g.add(gi), used_p.add(pi)
            match[gi] = (predicted[pi], score)
    return [(g, *match.get(gi, (None, 0.0))) for gi, g in enumerate(gold)]


CLAIM_CSV_FIELDS = ["system", "item_id", "split", "category", "gold_idx", "gold_text", "gold_verdict",
                    "predicted_text", "predicted_verdict", "match_score"]


def write_claims_csv(records: list[RunRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CLAIM_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            for idx, (gold, pred, score) in enumerate(align_claims(r.gold_claims, r.predicted_claims)):
                writer.writerow({
                    "system": r.system, "item_id": r.item_id, "split": r.split, "category": r.category,
                    "gold_idx": idx, "gold_text": gold.text, "gold_verdict": gold.gold_verdict.value,
                    "predicted_text": pred.text if pred else "",
                    "predicted_verdict": (pred.verdict.value if pred and pred.verdict else "MISSING"),
                    "match_score": round(score, 3),
                })


_ROWS = [
    ("Items", "items", "{:.0f}"),
    ("Task success rate", "task_success_rate", "{:.0%}"),
    ("Refusal behaviour matches", "refusal_match_rate", "{:.0%}"),
    ("Avg agent calls / query", "avg_agent_calls", "{:.1f}"),
    ("Avg tool calls / query", "avg_tool_calls", "{:.1f}"),
    ("Latency p50 (ms)", "latency_p50_ms", "{:.0f}"),
    ("Latency p95 (ms)", "latency_p95_ms", "{:.0f}"),
    ("Latency mean (ms)", "latency_mean_ms", "{:.0f}"),
    ("Avg tokens / query", "avg_tokens", "{:.0f}"),
    ("Avg cost / query (USD)", "avg_cost_usd", "${:.5f}"),
    ("Retry rate (claims)", "retry_rate", "{:.0%}"),
    ("Claims downgraded after retry", "downgraded_after_retry", "{:.0f}"),
]


def summary_markdown(summaries: dict[str, dict[str, float]], header: str) -> str:
    systems = list(summaries)
    lines = [header, "", "| Metric | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for label, key, fmt in _ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt.format(summaries[s][key]) for s in systems) + " |")
    lines += ["", "Per-claim predicted vs gold verdicts: `claims.csv` (for Accuracy / Macro-F1).",
              "Retry metrics apply to FinSight only (baselines make a single LLM call)."]
    return "\n".join(lines) + "\n"
