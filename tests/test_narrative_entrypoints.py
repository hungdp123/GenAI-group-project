"""Real registry entry points with temporary indexes and no model requests."""

from datetime import date
import json
from types import SimpleNamespace

import pytest

from finsight.agents.narrative import NarrativeAgent, NarrativeRetriever, WORKFLOW_MODULES
from finsight.agents.narrative.corpus.processing.chunk import chunk_blocks
from finsight.agents.narrative.corpus.processing.index import build_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock
from finsight.config import AgentRegistry, _load
from finsight.contracts.interfaces import NarrativeAgent as NarrativeProtocol, Retriever
from finsight.contracts.schemas import EvidenceItem, Stance, SubClaim, SubClaimType


CUTOFF = date(2025, 7, 31)


@pytest.fixture
def corpus_root(tmp_path):
    blocks = []
    for ticker, text, publication in (
        ("META", "Daily active people were 3.27 billion in June 2024.", "2024-07-31"),
        ("NVDA", "Revenue growth reflected accelerated computing demand.", "2024-08-01"),
        ("META", "Future daily active people guidance increased.", "2026-01-01"),
    ):
        blocks.append(
            ProcessedBlock(
                document_id=ticker + publication,
                company_ticker=ticker,
                company_name=ticker,
                document_type="earnings_release",
                publication_date=publication,
                source_tier="TIER_2",
                source_url="https://example.test/" + ticker,
                local_path=str(tmp_path / "source.html"),
                period="2024-06-30",
                section="Results",
                page=None,
                citation_span="Results paragraph",
                text=text,
            )
        )
    chunks = chunk_blocks(blocks, 800, 120)
    for ticker in ("META", "NVDA"):
        build_index(
            chunks,
            tmp_path / ticker / "indexes/index.json",
            corpus_fingerprint="test",
            target_words=800,
            overlap_words=120,
        )
    return tmp_path


def test_registered_real_entries_return_scoped_contract_evidence(corpus_root, monkeypatch):
    monkeypatch.setenv("FINSIGHT_IMPL_NARRATIVE", "finsight.agents.narrative.stub:NarrativeAgent")
    monkeypatch.setenv("FINSIGHT_IMPL_RETRIEVER", "finsight.agents.narrative.retriever_stub:NarrativeRetriever")
    monkeypatch.setenv("FINSIGHT_NARRATIVE_CORPUS_ROOT", str(corpus_root))
    monkeypatch.setenv("FINSIGHT_NARRATIVE_BACKEND", "bm25")
    monkeypatch.setenv("FINSIGHT_NARRATIVE_STANCE_METHOD", "none")
    monkeypatch.setenv("FINSIGHT_NARRATIVE_ANALYSIS", "retrieval")
    monkeypatch.setenv("FINSIGHT_NARRATIVE_ANALYSIS_DIR", str(corpus_root / "reports"))
    monkeypatch.delenv("FINSIGHT_NARRATIVE_TICKER", raising=False)
    registry = AgentRegistry.from_config()
    assert isinstance(registry.narrative, NarrativeProtocol)
    assert isinstance(registry.retriever, Retriever)
    claim = SubClaim(
        sub_claim_id="meta-dap",
        text="Daily active people were 3.27 billion in June 2024.",
        company_ticker="META",
        sub_claim_type=SubClaimType.OTHER,
    )
    items = registry.narrative.gather(claim, CUTOFF, None)
    assert len(items) == 1
    assert isinstance(items[0], EvidenceItem)
    assert items[0].sub_claim_id == "meta-dap"
    assert items[0].stance == Stance.NEUTRAL  # Required contract placeholder, not a model judgment.
    assert items[0].publication_date <= CUTOFF
    assert items[0].url == "https://example.test/META"
    assert items[0].citation_span == "Results paragraph"
    assert items[0].produced_by == "narrative_retrieval_unassessed"
    assert "stance" not in registry.narrative.last_analysis["evidence"][0]


def test_registry_retriever_does_not_default_every_query_to_nvda(corpus_root, monkeypatch):
    monkeypatch.delenv("FINSIGHT_NARRATIVE_TICKER", raising=False)
    retriever = NarrativeRetriever(corpus_root=corpus_root, backend="bm25")
    for ticker in ("META", "NVDA"):
        items = retriever.retrieve(ticker + " daily active people revenue", 5, CUTOFF)
        assert items and all(item.company == ticker for item in items)
        assert all(item.publication_date <= CUTOFF for item in items)
    assert retriever.retrieve("unscoped revenue growth", 5, CUTOFF) == []
    with pytest.raises(ValueError, match="multiple companies"):
        retriever.retrieve("META versus NVDA", 5, CUTOFF)
    explicit = NarrativeRetriever(company_ticker="META", corpus_root=corpus_root, backend="bm25")
    scoped = explicit.retrieve("META versus NVDA daily active people", 5, CUTOFF)
    assert scoped and all(i.company == "META" for i in scoped)


def test_feedback_and_tuning_are_forwarded(corpus_root, monkeypatch):
    calls = []
    monkeypatch.setenv("FINSIGHT_NARRATIVE_TOP_K", "2")
    monkeypatch.setenv("FINSIGHT_NARRATIVE_CANDIDATE_K", "20")

    class RecordingRetriever:
        def retrieve(self, query, top_k, cutoff_date):
            calls.append((query, top_k, cutoff_date))
            return []

    agent = NarrativeAgent(
        corpus_root=corpus_root,
        backend="bm25",
        stance_method="none",
        analysis_mode="retrieval",
        analysis_dir=corpus_root / "reports",
        retriever_factory=lambda ticker: RecordingRetriever(),
    )
    claim = SubClaim(
        sub_claim_id="sc1", text="Daily active people", company_ticker="META", sub_claim_type=SubClaimType.OTHER
    )
    assert agent.gather(claim, CUTOFF, "June 2024") == []
    assert len(calls) == 2
    assert all("June 2024" in query and k == 2 and cutoff == CUTOFF for query, k, cutoff in calls)
    assert agent.candidate_k == 20


def test_demo_paths_remain_loadable_and_separate_from_real_entries():
    demo_agent = _load("finsight.agents.narrative.demo.stub:StubNarrativeAgent")()
    demo_retriever = _load("finsight.agents.narrative.demo.retriever_stub:StubRetriever")()
    assert demo_agent.name == "narrative_stub"
    assert demo_retriever.name == "retriever_stub"
    assert not isinstance(demo_agent, NarrativeAgent)
    assert not isinstance(demo_retriever, NarrativeRetriever)
    from finsight.agents.narrative import run_basic_narrative_rag

    assert callable(run_basic_narrative_rag)
    assert WORKFLOW_MODULES["extract_pdf_charts_and_promote"].endswith("pdf_chart_processing")
    assert WORKFLOW_MODULES["generate_92_claim_answers"].endswith("run_basic_rag_batch")


def test_registry_matches_new_main_slots_and_preserves_other_agents():
    from finsight.config import DEFAULT_IMPLEMENTATIONS

    assert set(DEFAULT_IMPLEMENTATIONS) == set(AgentRegistry.__dataclass_fields__)
    assert DEFAULT_IMPLEMENTATIONS["financial"] == "finsight.agents.financial.agent:XbrlFinancialAgent"
    assert DEFAULT_IMPLEMENTATIONS["advice_boundary"] == "finsight.agents.advice_boundary.agent:AdviceBoundaryAgent"
    assert DEFAULT_IMPLEMENTATIONS["narrative"] == (
        "finsight.agents.narrative.rag.collect_evidence:NarrativeEvidenceAgent"
    )
    assert DEFAULT_IMPLEMENTATIONS["retriever"] == (
        "finsight.agents.narrative.retrieval.retriever:CorpusRetriever"
    )


@pytest.fixture
def preparation_model(monkeypatch):
    """Replace only the external model boundary; real retrieval/validation still run."""
    from finsight.agents.narrative.rag import evidence_summary
    from finsight.llm.client import LLMResponse

    calls = []
    monkeypatch.delenv("FINSIGHT_NARRATIVE_OLLAMA_MODEL", raising=False)
    monkeypatch.setattr(
        evidence_summary, "get_settings", lambda: SimpleNamespace(llm_provider="openai", llm_model="fake-model")
    )

    def completion(*, system, prompt, max_tokens):
        calls.append((system, prompt, max_tokens))
        passages = json.loads(prompt.split('<data name="source_passages">\n', 1)[1].split("\n</data>", 1)[0])
        passage = passages[0]
        result = {
            "selected_evidence_ids": [passage["evidence_id"]],
            "facts": [{"evidence_id": passage["evidence_id"], "quote": passage["content"]}],
            "summary": [
                {
                    "text": "The source reports the daily active people figure and period.",
                    "evidence_ids": [passage["evidence_id"]],
                }
            ],
        }
        return LLMResponse(
            text=json.dumps(result),
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            latency_ms=1.0,
            model="fake-model",
            provider="openai",
            finish_reason="stop",
        )

    monkeypatch.setattr(evidence_summary, "complete", completion)
    return calls


def test_basic_rag_prepares_cited_evidence_without_stance_or_verdict(corpus_root, preparation_model):
    from finsight.agents.narrative.rag.basic_rag import run_basic_narrative_rag

    report = run_basic_narrative_rag("META", "Daily active people June 2024", CUTOFF, corpus_root=corpus_root, top_k=3)
    assert report["status"] == "OK"
    assert report["evidence_preparation_status"] == "COMPLETED"
    assert len(preparation_model) == 1
    assert report["llm"]["api_requests"] == 1
    assert report["llm"]["input_tokens"] == 100
    assert report["citation_validation"]["passed"] is True
    assert report["facts"][0]["quote"] == report["evidence"][0]["content"]
    assert report["summary"][0]["evidence_ids"] == report["selected_evidence_ids"]
    assert "answer" not in report and "verdict" not in report
    assert all("stance" not in item for item in report["evidence"])


def test_registry_reuses_candidates_for_preparation(corpus_root, preparation_model):
    claim = SubClaim(
        sub_claim_id="sc1",
        text="Daily active people June 2024",
        company_ticker="META",
        sub_claim_type=SubClaimType.DEMAND_ATTRIBUTION,
    )
    agent = NarrativeAgent(
        corpus_root=corpus_root,
        backend="bm25",
        analysis_mode="evidence",
        stance_method="none",
        analysis_dir=corpus_root / "reports",
    )
    items = agent.gather(claim, CUTOFF, "June 2024")
    assert items and items[0].sub_claim_id == "sc1"
    assert items[0].content == agent.last_analysis["facts"][0]["quote"]
    assert len(preparation_model) == 1
    assert agent.last_analysis_path.is_file()
    assert "verdict" not in agent.last_analysis


@pytest.mark.parametrize(
    "raw",
    [
        '{"verdict": "SUPPORTED"}',
        '{"selected_evidence_ids": ["unknown"], "facts": [], "summary": []}',
        '{"selected_evidence_ids": [], "facts": [], "summary": [{"text": "Uncited", "evidence_ids": []}]}',
    ],
)
def test_invalid_model_output_preserves_source_candidates(corpus_root, preparation_model, monkeypatch, raw):
    from finsight.agents.narrative.rag import evidence_summary
    from finsight.agents.narrative.rag.basic_rag import run_basic_narrative_rag
    from finsight.llm.client import LLMResponse

    monkeypatch.setattr(
        evidence_summary,
        "complete",
        lambda **kwargs: LLMResponse(
            text=raw,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0,
            latency_ms=1,
            model="fake-model",
            provider="openai",
            finish_reason="stop",
        ),
    )
    report = run_basic_narrative_rag("META", "Daily active people June 2024", CUTOFF, corpus_root=corpus_root)
    assert report["status"] == "INVALID_GENERATION"
    assert report["selection_applied"] is False
    assert report["evidence"] == report["candidate_evidence"]
    assert report["summary"] == [] and report["facts"] == []
    assert report["citation_validation"]["passed"] is False


def test_invented_quote_is_not_accepted(corpus_root, preparation_model, monkeypatch):
    from finsight.agents.narrative.rag import evidence_summary
    from finsight.agents.narrative.rag.basic_rag import run_basic_narrative_rag

    original = evidence_summary.complete

    def invented(**kwargs):
        response = original(**kwargs)
        data = json.loads(response.text)
        data["facts"][0]["quote"] = "This quote was never in the source."
        response.text = json.dumps(data)
        return response

    monkeypatch.setattr(evidence_summary, "complete", invented)
    report = run_basic_narrative_rag("META", "Daily active people June 2024", CUTOFF, corpus_root=corpus_root)
    assert report["status"] == "INVALID_GENERATION"
    assert report["evidence"] == report["candidate_evidence"]
    assert any("exact source quote" in error for error in report["citation_validation"]["errors"])
