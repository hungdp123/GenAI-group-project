"""API keys, spend guard and persistent usage ledger for real LLM calls. Owner: Person 5 (Hung).

- Keys: environment variable first (OPENAI_API_KEY / ANTHROPIC_API_KEY), else `config.yaml`
  (gitignored; see config.example.yaml), matching the course API template.
- Spend guard: a process stops making real calls once it has spent FINSIGHT_SPEND_LIMIT_USD.
- Ledger: every real call is appended to FINSIGHT_USAGE_LEDGER (JSONL) so the team can track
  cumulative spend against the course credit with `finsight usage`.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from finsight.config import local_config_path, read_local_config

_KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class MissingAPIKey(RuntimeError):
    pass


class SpendLimitExceeded(RuntimeError):
    pass


def api_key(provider: str) -> str:
    """Key for `provider` from the environment or the local secrets file. Never logged."""
    env_name = _KEY_ENV[provider]
    if os.getenv(env_name):
        return os.environ[env_name]
    if key := read_local_config().get(f"{provider}_api_key"):
        return key
    raise MissingAPIKey(f"No {provider} key: set {env_name} or add {provider}_api_key to "
                        f"{local_config_path()} (copy config.example.yaml)")


class SpendGuard:
    """Process-wide running total of real-LLM spend with a hard limit (thread-safe)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.spent_usd = 0.0
        self.calls = 0

    @staticmethod
    def limit_usd() -> float:
        return float(os.getenv("FINSIGHT_SPEND_LIMIT_USD", "2.00"))

    def check(self) -> None:
        """Raise before a call if this process has already reached its limit."""
        with self._lock:
            if self.spent_usd >= self.limit_usd():
                raise SpendLimitExceeded(
                    f"Spent ${self.spent_usd:.4f} in this process (limit ${self.limit_usd():.2f}); "
                    "raise FINSIGHT_SPEND_LIMIT_USD to continue.")

    def add(self, cost_usd: float) -> None:
        with self._lock:
            self.spent_usd += cost_usd
            self.calls += 1

    def reset(self) -> None:
        with self._lock:
            self.spent_usd, self.calls = 0.0, 0


GUARD = SpendGuard()


def ledger_path() -> Path:
    return Path(os.getenv("FINSIGHT_USAGE_LEDGER", "logs/llm_usage.jsonl"))


_ledger_lock = threading.Lock()


def append_ledger(provider: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float,
                  latency_ms: float) -> None:
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": datetime.now(timezone.utc).isoformat(), "provider": provider, "model": model,
           "input_tokens": input_tokens, "output_tokens": output_tokens, "cost_usd": round(cost_usd, 6),
           "latency_ms": latency_ms}
    with _ledger_lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def ledger_summary(path: Path | None = None) -> dict[str, dict[str, float]]:
    """Totals per model plus an `ALL` row."""
    path = path or ledger_path()
    totals: dict[str, dict[str, float]] = {}
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for key in (row["model"], "ALL"):
            t = totals.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            t["calls"] += 1
            t["input_tokens"] += row["input_tokens"]
            t["output_tokens"] += row["output_tokens"]
            t["cost_usd"] += row["cost_usd"]
    return totals
