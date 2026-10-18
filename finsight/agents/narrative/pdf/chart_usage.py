"""Local usage accounting for new OpenAI chart requests, excluding cache reuse."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from finsight.agents.narrative.pdf.pdf_enrichment import write_json

PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.4"
PRICING_CHECKED = "2026-10-18"
# Standard tier, USD per 1M tokens: uncached input, cached input, output.
RATES = {"gpt-5.4": (2.50, 0.25, 15.00)}


def usage_root(workspace: Path, experiment: str) -> Path:
    return workspace / "data/narrative/experiments" / experiment / "usage"


def summarize_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priced = [row for row in rows if row.get("estimated_cost_usd") is not None]
    return {
        "api_requests": len(rows),
        "responses_received": sum(row["outcome"] == "response" for row in rows),
        "failed_requests": sum(row["outcome"] == "error" for row in rows),
        "requests_without_usage": sum(row.get("input_tokens") is None or
                                      row.get("output_tokens") is None for row in rows),
        "requests_without_cost_estimate": len(rows) - len(priced),
        "input_tokens": sum(row.get("input_tokens") or 0 for row in rows),
        "cached_input_tokens": sum(row.get("cached_input_tokens") or 0 for row in rows),
        "output_tokens": sum(row.get("output_tokens") or 0 for row in rows),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in priced), 8),
        "cost_estimate_complete": len(priced) == len(rows),
        "companies": sorted({row["ticker"] for row in rows if row.get("ticker")}),
        "models": sorted({row["model"] for row in rows}),
        "scope": "New requests recorded by Task 3 usage tracking; cached extractions excluded.",
        "limitations": [
            "Not an account balance or authoritative OpenAI billing total; no dollar cap is enforced.",
            "Requests without returned usage or verified pricing have unknown cost, not zero cost.",
            "Earlier runs and other tools or teammates are not included.",
            "Force termination can prevent recording an in-flight request.",
        ],
    }


class OpenAIChartUsage:
    def __init__(self, workspace: Path, experiment: str, model: str) -> None:
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
        self.directory = usage_root(workspace, experiment) / self.run_id
        self.model = model
        self.ticker: str | None = None
        self.rows: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.status = "running"
        self._save()

    def _save(self) -> dict[str, Any]:
        summary = {"run_id": self.run_id, "started_at": self.started_at,
                   "updated_at": datetime.now(timezone.utc).isoformat(), "status": self.status,
                   "pricing_source": PRICING_SOURCE, "pricing_checked": PRICING_CHECKED,
                   "standard_rates_per_million": RATES, **summarize_usage(self.rows)}
        write_json(self.directory / "summary.json", summary)
        return summary

    def record(self, request: Any, *, response: Any = None, error: BaseException | None = None) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        cached_tokens = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None)
        tier = getattr(response, "service_tier", None)
        cost = None
        rates = RATES.get(self.model)
        # Avoid guessing pricing for other models, tiers or long-context requests.
        if (rates and tier == "default" and input_tokens is not None and output_tokens is not None
                and cached_tokens is not None and 0 <= cached_tokens <= input_tokens <= 272000):
            cost = ((input_tokens - cached_tokens) * rates[0] + cached_tokens * rates[1]
                    + output_tokens * rates[2]) / 1_000_000
        row = {"at": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id,
               "ticker": self.ticker, "model": self.model, "document": request.document_name,
               "page": request.page_number, "chart_id": request.chart_id,
               "outcome": "error" if error is not None else "response",
               "error_type": type(error).__name__ if error is not None else None,
               "error_code": getattr(error, "code", None),
               "response_id": getattr(response, "id", None), "service_tier": tier,
               "input_tokens": input_tokens, "cached_input_tokens": cached_tokens,
               "output_tokens": output_tokens, "estimated_cost_usd": cost}
        with self.lock:
            with (self.directory / "calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.rows.append(row)
            self._save()

    def finish(self, status: str) -> dict[str, Any]:
        with self.lock:
            self.status = status
            return self._save()


def read_usage(workspace: Path, experiment: str, tickers: set[str]) -> dict[str, Any]:
    rows = []
    for path in sorted(usage_root(workspace, experiment).glob("*/calls.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Incomplete usage ledger: {path}. Do not treat its total as complete.") from None
            if row.get("ticker") in tickers:
                rows.append(row)
    return {"tracked_runs_with_requests": len({row["run_id"] for row in rows}), **summarize_usage(rows)}
