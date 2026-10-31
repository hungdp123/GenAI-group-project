"""Direct LLM baselines. Owner: Person 5 (Hung).

    run_direct_llm(claim, cutoff)                    LLM + filings: claim + the company's XBRL facts as text
    run_direct_llm(claim, cutoff, with_facts=False)  plain LLM: claim only, answered from what the model knows
"""

from __future__ import annotations

import time
from datetime import date

from finsight.agents.orchestrator.routing import detect_company, resolve_cutoff
from finsight.baselines.common import BaselineResult, apply_output, baseline_system_prompt, xbrl_facts_as_text
from finsight.config import get_settings
from finsight.llm.client import complete, data_block
from finsight.logging.decision_log import tracked_call


def run_direct_llm(claim: str, cutoff: date | None = None, *, with_facts: bool = True) -> BaselineResult:
    t0 = time.perf_counter()
    settings = get_settings()
    cutoff, _ = resolve_cutoff(cutoff, claim, settings.corpus_as_of_date)
    parts = [f"As-of date: {cutoff.isoformat()}", data_block("claim", claim)]
    sources = ""
    if with_facts:
        sources = xbrl_facts_as_text(cutoff, detect_company(claim, settings.corpus_companies))
        parts.append(data_block("xbrl_facts", sources))
    prompt = "\n\n".join(parts)
    resp, call, error = tracked_call("llm", complete, prompt, system=baseline_system_prompt(cutoff))
    result = BaselineResult(system="direct" if with_facts else "llm", cutoff_date=cutoff, agent_calls=[call],
                            sources_text=sources)
    if error:
        result.error = f"{type(error).__name__}: {error}"
    else:
        apply_output(result, resp.text)
    result.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return result
