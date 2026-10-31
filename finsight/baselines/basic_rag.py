"""Basic RAG baseline: whole-claim top-k retrieval -> one LLM call. Owner: Person 5 (Hung) (wiring only).

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
        data_block(f"doc{i + 1}",
                   f"[{e.source_tier.value} {e.doc_type} {e.publication_date} {e.url or ''}]\n{e.content}")
        for i, e in enumerate(items)
    )


QUOTE_NOTE = ("The influencer_said block is what the influencer actually said; the claim block is the user's "
              "message that points to it. Check what the influencer said.")


def run_basic_rag(claim: str, cutoff: date | None = None, *, retriever: Retriever | None = None,
                  top_k: int | None = None, quote: str | None = None) -> BaselineResult:
    """Whole-claim top-k retrieval, then one LLM call. With `quote` (the RAG + quote baseline) the model is also
    given the influencer's actual line, and retrieval uses that line, so only verification differs from FinSight."""
    t0 = time.perf_counter()
    settings = get_settings()
    cutoff, _ = resolve_cutoff(cutoff, claim, settings.corpus_as_of_date)
    retriever = retriever or AgentRegistry.from_config().retriever
    result = BaselineResult(system="rag_quote" if quote else "rag", cutoff_date=cutoff)

    query = quote or claim
    items, call, error = tracked_call("retriever", retriever.retrieve, query, top_k or settings.rag_top_k, cutoff)
    result.agent_calls.append(call)
    if error:
        items = []  # DECISION: a retriever failure degrades to no context rather than aborting the baseline
    result.retrieved_after_cutoff = sum(1 for item in items or [] if item.publication_date > cutoff)
    context = _context(items or [])
    blocks = [f"As-of date: {cutoff.isoformat()}", data_block("claim", claim)]
    if quote:
        blocks.append(data_block("influencer_said", quote))
    prompt = "\n\n".join([*blocks, "Retrieved documents:", context])
    result.sources_text = "\n".join(x for x in (quote or "", context) if x)
    system = baseline_system_prompt(cutoff) + ("\n" + QUOTE_NOTE if quote else "")
    resp, call, error = tracked_call("llm", complete, prompt, system=system)
    result.agent_calls.append(call)
    if error:
        result.error = f"{type(error).__name__}: {error}"
    else:
        apply_output(result, resp.text)
    result.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    return result
