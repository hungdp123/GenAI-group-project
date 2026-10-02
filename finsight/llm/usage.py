"""Per-call usage meter. Owner: Person 5 (Hung).

The orchestrator opens a meter around every agent call; anything inside that calls the
LLM client or `record_tool_call()` is attributed to that agent call in the decision log.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class UsageMeter:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0


_current: contextvars.ContextVar[UsageMeter | None] = contextvars.ContextVar("finsight_usage", default=None)


@contextmanager
def track_usage() -> Iterator[UsageMeter]:
    """Collect usage for everything executed inside the block (same thread/context)."""
    meter = UsageMeter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def record_llm_usage(input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    meter = _current.get()
    if meter is not None:
        meter.input_tokens += input_tokens
        meter.output_tokens += output_tokens
        meter.cost_usd += cost_usd


def record_tool_call(n: int = 1) -> None:
    """Call this from an agent each time it invokes a deterministic tool (XBRL lookup, vector search, ...)."""
    meter = _current.get()
    if meter is not None:
        meter.tool_calls += n
