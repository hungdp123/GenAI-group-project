"""Both narrative searches are written, in order, so a later change can be compared."""

import json
from datetime import date

from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent, SEARCH_RECORD_VERSION
from finsight.contracts.schemas import EvidenceItem, SourceTier, Stance, SubClaim, SubClaimType


def _item(chunk_id: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=chunk_id, sub_claim_id="claim", stance=Stance.NEUTRAL, content=text,
        source_tier=SourceTier.TIER_2, doc_type="earnings_release", company="NVIDIA",
        publication_date=date(2025, 5, 28), citation_span="p. 2", produced_by="test",
    )


def test_both_searches_are_appended(tmp_path, monkeypatch):
    claim_text = "NVIDIA revenue grew."
    index = tmp_path / "NVDA" / "indexes" / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text(json.dumps({
        "schema_version": "1.0",
        "chunks": [
            {"chunk_id": "support-hit", "company_ticker": "NVDA", "publication_date": "2025-05-28",
             "text": claim_text, "source_tier": "TIER_2", "document_type": "earnings_release",
             "company_name": "NVIDIA", "source_url": "https://example.com/a", "citation_span": "p. 2"},
            {"chunk_id": "counter-hit", "company_ticker": "NVDA", "publication_date": "2025-05-28",
             "text": "NVIDIA revenue declined.", "source_tier": "TIER_2", "document_type": "10-Q",
             "company_name": "NVIDIA", "source_url": "https://example.com/b", "citation_span": "p. 4"},
        ],
    }), encoding="utf-8")
    log_path = tmp_path / "narrative_searches.jsonl"
    monkeypatch.setenv("FINSIGHT_NARRATIVE_SEARCH_LOG", str(log_path))
    monkeypatch.setenv("FINSIGHT_NARRATIVE_SEARCH_LABEL", "friend")

    def retrieve(query: str, top_k: int, cutoff_date: date) -> list[EvidenceItem]:
        del top_k, cutoff_date
        if "contrary" in query:
            return [_item("counter-hit", "NVIDIA revenue declined.")]
        return [_item("support-hit", claim_text)]

    agent = NarrativeEvidenceAgent(
        corpus_root=tmp_path, backend="bm25", stance_method="exact",
        retriever_factory=lambda ticker: type("R", (), {"retrieve": staticmethod(retrieve)})(),
    )
    claim = SubClaim(
        sub_claim_id="sc1", text=claim_text, company_ticker="NVDA",
        sub_claim_type=SubClaimType.HISTORICAL_GROWTH,
    )
    found = agent.gather(claim, date(2025, 7, 31))
    assert [item.evidence_id for item in found] == ["support-hit", "counter-hit"]
    assert found[0].stance is Stance.SUPPORTING
    assert found[1].stance is Stance.NEUTRAL

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["version"] == SEARCH_RECORD_VERSION
    assert record["label"] == "friend"
    assert record["reason"] == "ok"
    assert "reported results drivers evidence" in record["support_query"]
    assert "contrary evidence" in record["counter_query"]
    assert record["support"][0]["chunk_id"] == "support-hit"
    assert record["support"][0]["stance"] == "SUPPORTING"
    assert record["counter"][0]["chunk_id"] == "counter-hit"
    assert record["counter"][0]["rank"] == 1
