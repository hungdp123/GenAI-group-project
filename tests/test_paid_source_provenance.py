"""Paid-source citation exceptions come from manifests and never promote source tiers."""
from datetime import date
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from finsight.agents.auditor.agent import AuditorAgent
from finsight.agents.narrative.corpus.processing.chunk import chunk_blocks
from finsight.agents.narrative.corpus.processing.index import build_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock
from finsight.agents.narrative.rag import source_provenance
from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent
from finsight.agents.orchestrator.render import _cite, _evidence
from finsight.agents.tier_policy.stub import StubTierPolicy
from finsight.contracts.schemas import (
    EvidenceItem, KnowledgeRule, PAID_SOURCE_URL_EXCEPTION, SourceTier, Stance, SubClaim, SubClaimType, Verdict,
)
from finsight.ui.payload import _evidence as ui_evidence

PAGE = "https://www.capitaliq.spglobal.com/"
CLAIM = SubClaim(sub_claim_id="aws", text="AWS revenue grew 28% year over year.",
                 company_ticker="AMZN", sub_claim_type=SubClaimType.OTHER)


def evidence(**changes):
    values = dict(
        evidence_id="AMZN-paid:001", sub_claim_id="aws", stance=Stance.NEUTRAL,
        content="AWS revenue grew 28% year over year in Q1 2026.", source_tier=SourceTier.TIER_3,
        doc_type="earnings_call_transcript", company="AMZN", publication_date=date(2026, 4, 29),
        url="", citation_span="PDF page 9", produced_by="narrative_retrieval_unassessed",
        source_page=PAGE, source_url_requirement=PAID_SOURCE_URL_EXCEPTION,
    )
    return EvidenceItem(**(values | changes))


def test_approved_platform_citation_passes_but_does_not_promote_tier():
    item = evidence()
    result = AuditorAgent(use_llm=False).audit(CLAIM, [item])
    assert result.passed and result.verdict is Verdict.SUPPORTED
    assert item.url == "" and item.source_tier is SourceTier.TIER_3
    used = item.model_copy(update={"stance": result.evidence_stances[item.evidence_id]})
    rule = KnowledgeRule(sub_claim_type=SubClaimType.OTHER, collectors=["narrative"],
                         highest_verdict=Verdict.SUPPORTED, tier_requirement=[SourceTier.TIER_2])
    verdict, _ = StubTierPolicy().apply(CLAIM, result.verdict, [used], rule)
    assert verdict is Verdict.INSUFFICIENT_EVIDENCE


def test_official_release_and_approved_paid_transcript_can_pass_together():
    paid = evidence()
    release = evidence(evidence_id="AMZN-release:001", doc_type="earnings_release",
                       source_tier=SourceTier.TIER_2, url="https://www.sec.gov/example",
                       citation_span="Results paragraph", source_page=None, source_url_requirement=None)
    result = AuditorAgent(use_llm=False).audit(CLAIM, [release, paid])
    assert result.passed and result.verdict is Verdict.SUPPORTED
    assert all(s is Stance.SUPPORTING for s in result.evidence_stances.values())
    assert [e.source_tier for e in [release, paid]] == [SourceTier.TIER_2, SourceTier.TIER_3]


@pytest.mark.parametrize("changes", [
    {"source_url_requirement": None},
    {"source_url_requirement": "unreviewed"},
    {"source_page": None},
    {"source_page": " "},
    {"source_page": "javascript:alert(1)"},
    {"source_page": "https://"},
    {"source_page": "https://[invalid"},
    {"citation_span": None},
    {"citation_span": " "},
])
def test_missing_or_unapproved_citations_still_fail(changes):
    result = AuditorAgent(use_llm=False).audit(CLAIM, [evidence(**changes)])
    assert not result.passed and result.verdict is Verdict.INSUFFICIENT_EVIDENCE
    assert any("no url or citation span" in reason for reason in result.failure_reasons)


def test_old_evidence_without_optional_fields_still_roundtrips():
    data = evidence().model_dump(exclude={"source_page", "source_url_requirement"})
    item = EvidenceItem.model_validate(data)
    assert item.source_page is None and item.source_url_requirement is None
    assert item.publisher_page is None
    assert EvidenceItem.model_validate_json(item.model_dump_json()) == item


def test_runtime_lookup_uses_manifest_not_retriever_and_keeps_index_unchanged(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    row = dict(document_id="AMZN-paid", company_ticker="AMZN", source_access="paid_subscription",
               source_page=PAGE, source_url_requirement=PAID_SOURCE_URL_EXCEPTION)
    manifest.write_text(json.dumps({"documents": [row]}), encoding="utf-8")
    monkeypatch.setattr(source_provenance, "CANONICAL_MANIFESTS", (manifest,))
    monkeypatch.setenv("FINSIGHT_NARRATIVE_SEARCH_LOG", "off")
    block = ProcessedBlock(
        document_id="AMZN-paid", company_ticker="AMZN", company_name="AMZN",
        document_type="earnings_call_transcript", publication_date="2026-04-29", source_tier="TIER_3",
        source_url="", local_path="transcript.pdf", period="FY2026_Q1", section="Results",
        page=9, citation_span="PDF page 9", text=evidence().content,
    )
    chunks = chunk_blocks([block], 300, 50)
    index = tmp_path / "AMZN/indexes/index.json"
    build_index(chunks, index, corpus_fingerprint="test", target_words=300, overlap_words=50)
    before = index.read_bytes()

    class ForgedRetriever:
        def retrieve(self, *_args):
            return [evidence(evidence_id=chunks[0].chunk_id, source_page="https://wrong.example/",
                             source_tier=SourceTier.TIER_2)]

    collector = NarrativeEvidenceAgent(corpus_root=tmp_path, backend="bm25",
                                       retriever_factory=lambda ticker: ForgedRetriever())
    item = collector.gather(CLAIM, date(2026, 4, 30))[0]
    assert item.source_page == PAGE and item.source_url_requirement == PAID_SOURCE_URL_EXCEPTION
    assert item.source_tier is SourceTier.TIER_3 and item.url == ""
    assert index.read_bytes() == before
    assert source_provenance.source_attribution("AMZN-paid", "PLTR") == {}

    # A manifest update takes effect without changing chunks or restarting the process.
    row["source_url_requirement"] = "unreviewed"
    manifest.write_text(json.dumps({"documents": [row]}), encoding="utf-8")
    unapproved = collector.gather(CLAIM, date(2026, 4, 30))[0]
    assert unapproved.source_page is None and unapproved.source_url_requirement is None
    assert index.read_bytes() == before


def test_missing_or_invalid_manifest_does_not_grant_an_exception(tmp_path, monkeypatch):
    manifest = tmp_path / "invalid.json"
    manifest.write_text("invalid json", encoding="utf-8")
    monkeypatch.setattr(source_provenance, "CANONICAL_MANIFESTS", (tmp_path / "missing.json", manifest))
    assert source_provenance.source_attribution("AMZN-paid", "AMZN") == {}


def test_cli_and_ui_distinguish_platform_links_from_document_links():
    item = evidence()
    reader = "\n".join(_evidence(item))
    assert "Publisher/platform: " + PAGE in reader
    assert "Publisher/platform: <" + PAGE + ">" in _cite(item)
    payload = ui_evidence(item)
    assert payload["url"] == "" and payload["publisher_page"] == PAGE
    assert payload["source_tier"] == "TIER_3"
    assert ui_evidence(evidence(source_url_requirement=None))["publisher_page"] is None
    direct = evidence(url="https://example.test/document.pdf")
    assert "Publisher/platform:" not in "\n".join(_evidence(direct))
    assert "Publisher/platform:" not in _cite(direct)


def test_browser_renders_only_approved_platform_fallback():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is optional; Python citation payload checks still run")
    script = Path("finsight/ui/static/app.js").read_text(encoding="utf-8").rsplit("\ninit();", 1)[0]
    items = [ui_evidence(evidence()), ui_evidence(evidence(source_url_requirement=None)),
             ui_evidence(evidence(url="https://example.test/document.pdf"))]
    program = (
        "const vm = require('vm'); const context = {}; vm.createContext(context);"
        f"vm.runInContext({json.dumps(script)}, context);"
        f"const items = {json.dumps(items)};"
        "const html = items.map(item => {context.item = item;"
        "return vm.runInContext('evidenceCard(item)', context)});"
        "process.stdout.write(JSON.stringify(html));"
    )
    result = subprocess.run([node, "-"], input=program, encoding="utf-8", capture_output=True, check=True)
    approved, unapproved, direct = json.loads(result.stdout)
    assert "Publisher/platform" in approved and f'href="{PAGE}"' in approved
    assert "Publisher/platform" not in unapproved and PAGE not in unapproved
    assert '<span>Link</span>' in direct and "Publisher/platform" not in direct
