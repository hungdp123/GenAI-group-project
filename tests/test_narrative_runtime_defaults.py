"""End-to-end C2/hybrid defaults and explicit configuration overrides."""

from pathlib import Path
import shutil

import pytest

from finsight.config import AgentRegistry
from finsight.contracts.schemas import SubClaim, SubClaimType
from test_narrative_entrypoints import CUTOFF, corpus_root as corpus_root


@pytest.mark.parametrize("use_wrappers", [False, True])
def test_end_to_end_defaults_select_c2_hybrid_for_both_slots(corpus_root, monkeypatch, use_wrappers):
    from finsight.agents.narrative.corpus.processing.index import search_index
    from finsight.agents.narrative.retrieval.retriever import CorpusRetriever

    monkeypatch.chdir(corpus_root)
    for key in (
        "FINSIGHT_NARRATIVE_CORPUS_ROOT", "FINSIGHT_NARRATIVE_BACKEND",
        "FINSIGHT_NARRATIVE_INDEX_TEMPLATE", "FINSIGHT_NARRATIVE_QDRANT_TEMPLATE",
        "FINSIGHT_NARRATIVE_INDEX", "FINSIGHT_NARRATIVE_QDRANT",
        "FINSIGHT_IMPL_NARRATIVE", "FINSIGHT_IMPL_RETRIEVER",
    ):
        monkeypatch.delenv(key, raising=False)
    if use_wrappers:
        monkeypatch.setenv("FINSIGHT_IMPL_NARRATIVE", "finsight.agents.narrative.stub:NarrativeAgent")
        monkeypatch.setenv("FINSIGHT_IMPL_RETRIEVER", "finsight.agents.narrative.retriever_stub:NarrativeRetriever")
    monkeypatch.setenv("FINSIGHT_NARRATIVE_ANALYSIS", "retrieval")
    root = Path("data/narrative/processed/chunking_strategies/C2/full_v1")
    monkeypatch.setenv("FINSIGHT_NARRATIVE_TICKER", "META")
    shutil.copytree(corpus_root / "META", root / "META")
    searched = []

    def vector_matches(self, payload, query, top_k, cutoff_date):
        # Exercise registry/collector propagation without opening Qdrant or loading a model.
        assert self.backend == self.last_backend == "hybrid"
        assert self.index_path == root / "META/indexes/index.json"
        assert self.qdrant_dir == root / "META/indexes/qdrant"
        searched.append(query)
        return search_index(payload, query, top_k, cutoff_date, self.company_ticker)

    monkeypatch.setattr(CorpusRetriever, "_vector_matches", vector_matches)
    registry = AgentRegistry.from_config()
    claim = SubClaim(sub_claim_id="c2-default", text="Daily active people June 2024",
                     company_ticker="META", sub_claim_type=SubClaimType.OTHER)
    items = registry.narrative.gather(claim, CUTOFF, None)
    assert items and all(item.company == "META" for item in items)
    assert registry.narrative.last_search["backend"] == "hybrid"
    assert registry.narrative.last_search["index"]["path"] == str((root / "META/indexes/index.json").resolve())
    # Both registry slots search the same selected build.
    if not use_wrappers:
        assert registry.retriever.index_path == root / "META/indexes/index.json"
    assert registry.retriever.backend == "hybrid"
    before = len(searched)
    assert registry.retriever.retrieve("META daily active people", 3, CUTOFF)
    assert len(searched) == before + 1


@pytest.mark.parametrize("use_wrappers", [False, True])
def test_end_to_end_index_and_backend_overrides_remain_available(corpus_root, monkeypatch, use_wrappers):
    from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent
    from finsight.agents.narrative.retrieval.retriever import CorpusRetriever

    monkeypatch.setenv("FINSIGHT_NARRATIVE_CORPUS_ROOT", str(corpus_root))
    monkeypatch.setenv("FINSIGHT_NARRATIVE_BACKEND", "dense")
    monkeypatch.setenv("FINSIGHT_IMPL_NARRATIVE", (
        "finsight.agents.narrative.stub:NarrativeAgent" if use_wrappers
        else "finsight.agents.narrative.rag.collect_evidence:NarrativeEvidenceAgent"))
    monkeypatch.setenv("FINSIGHT_IMPL_RETRIEVER", (
        "finsight.agents.narrative.retriever_stub:NarrativeRetriever" if use_wrappers
        else "finsight.agents.narrative.retrieval.retriever:CorpusRetriever"))
    registry = AgentRegistry.from_config()
    assert registry.narrative.corpus_root == corpus_root
    assert registry.narrative.backend == registry.retriever.backend == "dense"
    if use_wrappers:
        assert registry.retriever.corpus_root == str(corpus_root)
    else:
        assert registry.retriever.index_path.parent.parent.parent == corpus_root
    explicit = NarrativeEvidenceAgent(corpus_root=corpus_root, backend="bm25")
    assert AgentRegistry.from_config(narrative=explicit).narrative is explicit
    monkeypatch.delenv("FINSIGHT_NARRATIVE_CORPUS_ROOT")
    monkeypatch.delenv("FINSIGHT_NARRATIVE_BACKEND")
    assert NarrativeEvidenceAgent().corpus_root == Path("data/narrative/processed")
    assert CorpusRetriever(company_ticker="META").index_path == Path("data/narrative/processed/META/indexes/index.json")


