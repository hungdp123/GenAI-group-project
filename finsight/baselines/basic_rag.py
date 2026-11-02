"""Basic RAG baseline: one company-scoped search, then one LLM call. Owner: Person 5 (Hung) (wiring only).

The search uses the same C2 hybrid index as FinSight's narrative agent, limited to the company named
in the claim. The verdict is still the one LLM call. There is no supporting/challenging pair and no auditor.
"""

from __future__ import annotations

import os
import time
from datetime import date

from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
from finsight.agents.orchestrator.routing import detect_company, resolve_cutoff
from finsight.baselines.common import BaselineResult, apply_output, baseline_system_prompt
from finsight.config import DEFAULT_NARRATIVE_BACKEND, DEFAULT_NARRATIVE_CORPUS_ROOT, get_settings
from finsight.contracts.interfaces import Retriever
from finsight.contracts.schemas import EvidenceItem
from finsight.llm.client import complete, data_block
from finsight.llm.usage import record_tool_call
from finsight.logging.decision_log import tracked_call

# One retriever per company keeps the embedding model loaded across claims.
_retrievers: dict[tuple[str, str, str], CorpusRetriever] = {}
_embedder = None


def _context(items: list[EvidenceItem]) -> str:
    return "\n\n".join(
        data_block(f"doc{i + 1}",
                   f"[{e.source_tier.value} {e.doc_type} {e.publication_date} {e.url or ''}]\n{e.content}")
        for i, e in enumerate(items)
    )


QUOTE_NOTE = ("The influencer_said block is what the influencer actually said; the claim block is the user's "
              "message that points to it. Check what the influencer said.")


def _company(claim: str, quote: str | None) -> str | None:
    """The corpus company named in the claim. The quote is the fallback when the claim names none."""
    settings = get_settings()
    ticker = detect_company(claim, settings.corpus_companies)
    if ticker is None and quote:
        ticker = detect_company(quote, settings.corpus_companies)
    return ticker


def _retriever_for(ticker: str) -> CorpusRetriever:
    global _embedder
    root = os.getenv("FINSIGHT_NARRATIVE_CORPUS_ROOT") or DEFAULT_NARRATIVE_CORPUS_ROOT
    backend = os.getenv("FINSIGHT_NARRATIVE_BACKEND") or DEFAULT_NARRATIVE_BACKEND
    key = (ticker, root, backend)
    found = _retrievers.get(key)
    if found is None:
        found = CorpusRetriever(
            company_ticker=ticker, corpus_root=root, backend=backend, embedding_backend=_embedder,
        )
        _retrievers[key] = found
    return found


def _retrieve(query: str, top_k: int, cutoff: date, ticker: str | None) -> list[EvidenceItem]:
    """Top-k passages for this company. No named company means no passages, not the Nvidia index."""
    global _embedder
    if ticker is None:
        record_tool_call()
        return []
    retriever = _retriever_for(ticker)
    items = retriever.retrieve(query, top_k, cutoff)
    if _embedder is None and retriever._embedding_backend is not None:
        _embedder = retriever._embedding_backend
    return items


def run_basic_rag(claim: str, cutoff: date | None = None, *, retriever: Retriever | None = None,
                  top_k: int | None = None, quote: str | None = None) -> BaselineResult:
    """One top-k search, then one LLM call. With `quote` (RAG + quote) the model is also given the
    influencer's actual line, and retrieval uses that line. The search is still limited to the company
    named in the claim."""
    t0 = time.perf_counter()
    settings = get_settings()
    cutoff, _ = resolve_cutoff(cutoff, claim, settings.corpus_as_of_date)
    result = BaselineResult(system="rag_quote" if quote else "rag", cutoff_date=cutoff)

    query = quote or claim
    k = top_k or settings.rag_top_k
    if retriever is None:
        items, call, error = tracked_call("retriever", _retrieve, query, k, cutoff, _company(claim, quote))
    else:
        items, call, error = tracked_call("retriever", retriever.retrieve, query, k, cutoff)
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
