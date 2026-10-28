"""Narrative workflow and orchestrator integration surface.

Runtime imports:
    from finsight.agents.narrative import NarrativeAgent, NarrativeRetriever

Collector contract: gather(SubClaim, cutoff_date, feedback) -> list[EvidenceItem].
Retriever contract: retrieve(query, top_k, cutoff_date) -> list[EvidenceItem].
The basic_rag entry point prepares cited evidence and a reasoned claim stance with
an LLM. The Auditor retains final verdict authority. verdict_baseline retains historical answer generation and the
Ollama adapter used by current evidence preparation and evaluation. Workflow
modules below cover preparation, evaluation and inference without import-time work.
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
    "validate_production_provenance": "finsight.agents.narrative.evaluation.validate_retrieval_provenance",
    "prepare_experiment": "scripts.narrative.experiments.prepare_narrative_experiment",
    "prepare_evidence_labels": "scripts.narrative.datasets.prepare_narrative_labels",
    "build_full_chunking_indexes": "scripts.narrative.retrieval.build_full_chunking_indexes",
    "run_rag_comparison": "scripts.narrative.evaluation.run_rag_comparison",
    "evaluate_rag_comparison": "scripts.narrative.evaluation.evaluate_rag_comparison",
    "evaluate_ragas": "scripts.narrative.evaluation.evaluate_ragas",
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
