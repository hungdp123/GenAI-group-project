"""Basic RAG baseline: whole-thesis top-k retrieval -> one LLM call. Owner: Person 5 (Hung) (wiring only).

The retriever is Person 2's, reached through the `Retriever` Protocol via the agent registry.
"""

from __future__ import annotations

import time
from datetime import date

from finsight.agents.orchestrator.routing import resolve_cutoff
from finsight.baselines.common import BaselineResult, apply_output, baseline_system_prompt
from finsight.config import AgentRegistry, get_settings
from finsight.contracts.interfaces import Retriever
from finsight.contracts.schemas import EvidenceItem
from finsight.llm.client import complete, data_block
from finsight.logging.decision_log import tracked_call


def _context(items: list[EvidenceItem]) -> str:
    return "\n\n".join(
        data_block(f"doc{i + 1}", f"[{e.source_tier.value} {e.doc_type} {e.publication_date} {e.url or ''}]\n{e.content}")
        for i, e in enumerate(items)
    )


def run_basic_rag(thesis: str, cutoff: date | None = None, *, retriever: Retriever | None = None,
                  top_k: int | None = None) -> BaselineResult:
    t0 = time.perf_counter()
    settings = get_settings()
    cutoff, _ = resolve_cutoff(cutoff, thesis, settings.corpus_as_of_date)
    retriever = retriever or AgentRegistry.from_config().retriever
    result = BaselineResult(system="rag", cutoff_date=cutoff)

    items, call, error = tracked_call("retriever", retriever.retrieve, thesis, top_k or settings.rag_top_k, cutoff)
    result.agent_calls.append(call)
    if error:
        items = []  # DECISION: a retriever failure degrades to no context rather than aborting the baseline
    prompt = "\n\n".join([f"As-of date: {cutoff.isoformat()}", data_block("thesis", thesis),
                          "Retrieved documents:", _context(items or [])])
    resp, call, error = tracked_call("llm", complete, prompt, system=baseline_system_prompt(cutoff))
    result.agent_calls.append(call)
    if error:
        result.error = f"{type(error).__name__}: {error}"
    else:
        apply_output(result, resp.text)
    result.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return result
