from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from finsight.agents.narrative.corpus.processing.chunk import chunk_blocks
from finsight.agents.narrative.corpus.processing.corpus import CorpusDocument, validate_document
from finsight.agents.narrative.corpus.processing.index import build_index, load_index, search_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock, parse_document
from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    DenseVectorCache,
    FiveWayRetriever,
)
from finsight.contracts.schemas import SourceTier, Stance


def _document(path: Path, *, digest: str | None = None) -> CorpusDocument:
    return CorpusDocument(
        document_id="NVDA-test-10-K",
        company_ticker="NVDA",
        company_name="NVIDIA CORP",
        document_type="10-K",
        publication_date=date(2025, 2, 26),
        source_tier="TIER_1",
        source_url="https://www.sec.gov/example",
        local_path=path,
        sha256=digest or hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _block(text: str, publication_date: str = "2025-02-26") -> ProcessedBlock:
    return ProcessedBlock(
        document_id=f"NVDA-{publication_date}",
        company_ticker="NVDA",
        company_name="NVIDIA CORP",
        document_type="10-K",
        publication_date=publication_date,
        source_tier="TIER_1",
        source_url="https://www.sec.gov/example",
        local_path="sample.htm",
        period="2025-01-26",
        section="Item 7. Management's Discussion and Analysis",
        page=None,
        citation_span="Item 7. Management's Discussion and Analysis",
        text=text,
    )


def test_validate_document_detects_hash_mismatch(tmp_path):
    path = tmp_path / "filing.htm"
    path.write_text("filing", encoding="utf-8")
    validate_document(_document(path))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_document(_document(path, digest="0" * 64))


def test_html_ingestion_preserves_filing_sections(tmp_path):
    path = tmp_path / "filing.htm"
    path.write_text(
        "<html><body><p>Cover</p><h1>Item 1. Business</h1><p>Accelerated computing.</p>"
        "<h1>Item 1A. Risk Factors</h1><p>Export restrictions may reduce demand.</p></body></html>",
        encoding="utf-8",
    )
    blocks, warnings = parse_document(_document(path))
    assert not warnings
    assert [block.section for block in blocks][-2:] == ["Item 1. Business", "Item 1A. Risk Factors"]
    assert "Export restrictions" in blocks[-1].text


def test_html_ingestion_removes_page_furniture_and_formats_tables(tmp_path):
    path = tmp_path / "filing.htm"
    toc_rows = "".join(
        f"<tr><td>Item {number}.</td><td><a href='#item{number}'>Section {number}</a></td><td>{number}</td></tr>"
        for number in range(1, 11)
    )
    toc_table = f"<table><tr><td>PART I</td></tr>{toc_rows}<tr><td>PART II</td></tr></table>"
    path.write_text(
        "<html><body>"
        "<nav>Navigation links</nav><div class='page-header'>Repeated page title</div>"
        "<script>not narrative</script><style>.hidden { display: none; }</style>"
        "<p style='display:none'>Hidden content</p><p>Before the table.</p>"
        f"{toc_table}"
        "<h1>Item 7. Results</h1>"
        "<table><thead><tr><th>Metric</th><th>2025</th><th>2024</th></tr></thead>"
        "<tbody><tr><th scope='row'>Revenue</th><td>$</td><td>100</td><td>$</td><td>80</td></tr>"
        "<tr><th scope='row'>Gross margin</th><td>63.1</td><td>%</td><td>62.6</td><td>%</td></tr></tbody></table>"
        "<table><tr><td>Year Ended</td></tr><tr><td>2025</td><td>2024</td></tr>"
        "<tr><td>Net income</td><td>$</td><td>20</td><td>$</td><td>15</td></tr></table>"
        "<p>After the table.</p><footer>Repeated footer</footer></body></html>",
        encoding="utf-8",
    )

    blocks, warnings = parse_document(_document(path))
    assert not warnings
    text = "\n".join(block.text for block in blocks)
    assert "Navigation links" not in text
    assert "Repeated page title" not in text
    assert "not narrative" not in text
    assert "Hidden content" not in text
    assert "Repeated footer" not in text
    assert "Revenue — 2025: $100; 2024: $80" in text
    assert "Gross margin — 2025: 63.1%; 2024: 62.6%" in text
    assert "Net income — 2025: $20; 2024: $15" in text
    assert text.index("Before the table") < text.index("Revenue — 2025") < text.index("After the table")
    item_sections = [block.section for block in blocks if block.section.startswith("Item")]
    assert item_sections and set(item_sections) == {"Item 7. Results"}
    assert text.count("Item 7. Results") == 1


def test_chunk_ids_are_stable_and_do_not_cross_documents():
    first = _block("accelerated computing demand " * 30)
    second = ProcessedBlock(**{**first.to_json(), "document_id": "NVDA-other", "text": "supply risk " * 30})
    chunks_one = chunk_blocks([first, second], target_words=20, overlap_words=5)
    chunks_two = chunk_blocks([first, second], target_words=20, overlap_words=5)
    assert [chunk.chunk_id for chunk in chunks_one] == [chunk.chunk_id for chunk in chunks_two]
    assert all(chunk.document_id in chunk.chunk_id for chunk in chunks_one)


def test_index_retrieval_enforces_company_and_cutoff(tmp_path):
    blocks = [
        _block("Blackwell accelerated computing data center customer demand grew strongly."),
        _block("Blackwell demand after cutoff.", publication_date="2026-01-01"),
    ]
    chunks = chunk_blocks(blocks, target_words=50, overlap_words=5)
    index_path = tmp_path / "index.json"
    build_index(
        chunks,
        index_path,
        corpus_fingerprint="fingerprint",
        target_words=50,
        overlap_words=5,
    )
    payload = load_index(index_path)
    results = search_index(payload, "Blackwell data center demand", 5, date(2025, 7, 31), "NVDA")
    assert results
    assert all(item["company_ticker"] == "NVDA" for item in results)
    assert all(item["publication_date"] <= "2025-07-31" for item in results)
    assert search_index(payload, "Blackwell", 5, date(2025, 7, 31), "AAPL") == []

    evidence = CorpusRetriever(index_path=index_path, company_ticker="NVDA").retrieve(
        "Blackwell data center demand", 5, date(2025, 7, 31)
    )
    assert evidence[0].source_tier is SourceTier.TIER_1
    assert evidence[0].stance is Stance.NEUTRAL
    assert evidence[0].url == "https://www.sec.gov/example"
    assert evidence[0].citation_span


class _FakeEmbeddingBackend:
    model_name = "test-embedding"

    def encode_documents(self, texts):
        return [self._vector(text) for text in texts]

    def encode_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        value = text.lower()
        if "customer concentration" in value or "risk" in value:
            return [0.0, 1.0]
        return [1.0, 0.0]


class _FakeReranker:
    model_name = "test-reranker"

    def score(self, query, texts):
        query_terms = set(query.lower().split())
        return [float(len(query_terms & set(text.lower().split()))) for text in texts]


def test_five_retrieval_setups_share_filters_and_keep_provenance(tmp_path):
    blocks = [
        _block("Accelerated computing data center demand supports customer growth."),
        replace(
            _block("Customer concentration risk and supply limits could reduce growth."),
            document_id="NVDA-2025-10-Q",
            publication_date="2025-05-28",
            document_type="10-Q",
            citation_span="Item 1A. Risk Factors",
            section="Item 1A. Risk Factors",
        ),
        replace(
            _block("Accelerated computing demand at another company."),
            document_id="AAPL-2025-10-K",
            company_ticker="AAPL",
        ),
        replace(
            _block("Accelerated computing demand after cutoff."),
            document_id="NVDA-2026-10-Q",
            publication_date="2026-02-25",
        ),
    ]
    chunks = chunk_blocks(blocks, target_words=100, overlap_words=10)
    index_path = tmp_path / "index.json"
    payload = build_index(
        chunks,
        index_path,
        corpus_fingerprint="test-fingerprint",
        target_words=100,
        overlap_words=10,
    )
    embedding = _FakeEmbeddingBackend()
    cache = DenseVectorCache.load_or_build(payload, tmp_path / "dense.json.gz", embedding)
    retriever = FiveWayRetriever(payload, cache, embedding, _FakeReranker())

    results = retriever.search(
        "accelerated computing demand risk customer concentration",
        top_k=3,
        cutoff_date=date(2025, 7, 31),
        company_ticker="NVDA",
        candidate_k=10,
    )

    assert set(results) == {
        "bm25", "dense", "hybrid_rrf", "bm25_cross_encoder", "hybrid_cross_encoder"
    }
    assert all(results.values())
    for method, ranking in results.items():
        assert all(item["company_ticker"] == "NVDA" for item in ranking), method
        assert all(item["publication_date"] <= "2025-07-31" for item in ranking), method
        assert all(item["source_url"] and item["citation_span"] for item in ranking), method
        assert all(item["retrieval_method"] == method for item in ranking), method


def test_dense_cache_is_reused_and_invalidated_by_fingerprint(tmp_path):
    embedding = _FakeEmbeddingBackend()
    chunk = chunk_blocks([_block("accelerated computing demand")], target_words=50, overlap_words=5)
    payload = build_index(
        chunk,
        tmp_path / "index.json",
        corpus_fingerprint="first",
        target_words=50,
        overlap_words=5,
    )
    cache_path = tmp_path / "dense.json.gz"
    first = DenseVectorCache.load_or_build(payload, cache_path, embedding)
    assert first.model_name == embedding.model_name
    second = DenseVectorCache.load_or_build(payload, cache_path, embedding)
    assert second.vectors == first.vectors

    changed = dict(payload, corpus_fingerprint="second")
    rebuilt = DenseVectorCache.load_or_build(changed, cache_path, embedding)
    assert rebuilt.corpus_fingerprint == "second"
