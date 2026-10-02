"""Shared types for per-agent evaluation. Owner: Person 5 (Hung).

Every evaluator runs the agent that is REGISTERED in finsight/config.py (stub or real), so the
same command scores the placeholder today and the teammate's real agent after the swap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).parent / "datasets"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    detail: str = ""


@dataclass
class AgentReport:
    agent: str
    owner: str
    implementation: str
    metrics: dict[str, Any] = field(default_factory=dict)
    cases: list[CaseResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def implementation_name(agent: object) -> str:
    return f"{type(agent).__module__}.{type(agent).__name__}"


def load_jsonl(name: str) -> list[dict]:
    text = (DATASET_DIR / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def as_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def ratio(numerator: float, denominator: float) -> float | None:
    """None (shown as n/a) when there is nothing to measure, instead of a misleading 0 or 1."""
    return round(numerator / denominator, 3) if denominator else None


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    if precision is None or recall is None or precision + recall == 0:
        return precision, recall, (0.0 if precision == 0 or recall == 0 else None)
    return precision, recall, round(2 * precision * recall / (precision + recall), 3)


def corpus_note(dataset: str) -> str:
    return ("Cases are built on the placeholder Company X corpus, so the stub scores well by construction. "
            f"Add real-corpus cases to finsight/eval/agent_eval/datasets/{dataset} before reporting numbers.")
