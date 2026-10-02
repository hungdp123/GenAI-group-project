"""Direct LLM baseline: thesis + XBRL facts as text -> one LLM call. Owner: Person 5 (Hung)."""

from __future__ import annotations

import time
from datetime import date

from finsight.agents.orchestrator.routing import resolve_cutoff
from finsight.baselines.common import BaselineResult, baseline_system_prompt, parse_output, xbrl_facts_as_text
from finsight.config import get_settings
from finsight.llm.client import complete, data_block
from finsight.logging.decision_log import tracked_call


def run_direct_llm(thesis: str, cutoff: date | None = None) -> BaselineResult:
    t0 = time.perf_counter()
    cutoff, _ = resolve_cutoff(cutoff, thesis, get_settings().corpus_as_of_date)
    prompt = "\n\n".join([
        f"As-of date: {cutoff.isoformat()}",
        data_block("thesis", thesis),
        data_block("xbrl_facts", xbrl_facts_as_text(cutoff)),
    ])
    resp, call, error = tracked_call("llm", complete, prompt, system=baseline_system_prompt(cutoff))
    result = BaselineResult(system="direct", cutoff_date=cutoff, agent_calls=[call])
    if error:
        result.error = f"{type(error).__name__}: {error}"
    else:
        result.raw_output = resp.text
        result.refused, result.refusal_reason, result.claims, result.overall_verdict = parse_output(resp.text)
    result.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return result
