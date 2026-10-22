"""Narrative workflow and orchestrator integration surface.

Runtime imports:
    from finsight.agents.narrative import NarrativeAgent, NarrativeRetriever

Collector contract: gather(SubClaim, cutoff_date, feedback) -> list[EvidenceItem].
Retriever contract: retrieve(query, top_k, cutoff_date) -> list[EvidenceItem].
The basic_rag entry point prepares cited evidence with an LLM, without stance or
claim verdict. verdict_baseline retains
historical answer generation. Workflow modules
below cover source preparation, evaluation and inference without import-time work.
"""

from importlib import import_module

__all__ = ["NarrativeAgent", "NarrativeRetriever", "run_basic_narrative_rag", "WORKFLOW_MODULES"]

WORKFLOW_MODULES = {
    "acquire_core_documents": "finsight.agents.narrative.corpus.acquisition.download_corpus",
    "acquire_supplemental_documents": "finsight.agents.narrative.corpus.acquisition.download_supplemental_corpus",
    "catalog": "finsight.agents.narrative.corpus.maintenance.generate_catalog",
    "process_text_and_build_bm25": "finsight.agents.narrative.corpus.processing.process_corpus",
    "extract_pdf_charts_and_promote": "finsight.agents.narrative.corpus.pdf.pdf_chart_processing",
    "build_vectors": "finsight.agents.narrative.retrieval.build_vector_databases",
    "retrieve": "finsight.agents.narrative.retrieval.retriever",
    "collect_evidence": "finsight.agents.narrative.rag.collect_evidence",
    "retrieve_basic_rag_evidence": "finsight.agents.narrative.rag.basic_rag",
    "prepare_cited_evidence": "finsight.agents.narrative.rag.evidence_summary",
    "generate_basic_rag_answer": "finsight.agents.narrative.rag.verdict_baseline",
    "generate_92_claim_answers": "finsight.agents.narrative.evaluation.run_basic_rag_batch",
    "validate_production_provenance": "finsight.agents.narrative.evaluation.validate_retrieval_provenance",
    "validate_experiment_provenance": "finsight.agents.narrative.evaluation.validate_canonical_provenance",
    "verify_integration": "finsight.agents.narrative.evaluation.verify_agent_integration",
    "prepare_experiment": "scripts.narrative.experiments.prepare_narrative_experiment",
    "prepare_evidence_labels": "scripts.narrative.datasets.prepare_narrative_labels",
    "validate_chunking": "scripts.narrative.chunking.validate_narrative_chunking",
    "build_adaptive_chunks": "scripts.narrative.chunking.build_adaptive_narrative_chunks",
    "compare_chunking": "scripts.narrative.chunking.compare_narrative_chunking",
    "compare_retrieval_priority": "scripts.narrative.retrieval.compare_narrative_retrieval_priority",
    "compare_canonical_92": "scripts.narrative.retrieval.compare_canonical_development",
}

_EXPORTS = {
    "NarrativeAgent": (".stub", "NarrativeAgent"),
    "NarrativeRetriever": (".retriever_stub", "NarrativeRetriever"),
    "run_basic_narrative_rag": (".rag.basic_rag", "run_basic_narrative_rag"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value
