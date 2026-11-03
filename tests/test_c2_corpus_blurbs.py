"""Claim-independent corpus coverage, provenance, cache reuse and resume guards."""

import json
from types import SimpleNamespace

import pytest

from scripts.narrative.chunking import build_c2_corpus_blurbs as builder


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("finsight.config.get_settings", lambda: SimpleNamespace(
        llm_provider="mock", llm_model="test-model",
    ))
    root = tmp_path / "metadata"
    for ticker in ("AAA", "BBB"):
        parsed = tmp_path / "data/narrative/processed" / ticker / "documents.jsonl"
        parsed.parent.mkdir(parents=True)
        parsed.write_text("{}\n", encoding="utf-8")
        chunks = [{
            "chunk_id": f"{ticker}-{i}", "document_id": f"{ticker}-doc",
            "company_ticker": ticker, "text": f"Original source passage {ticker} {i}.",
            "search_text": f"Metadata prefix\nOriginal source passage {ticker} {i}.",
            "citation_span": f"Page {i}", "source_url": "https://example.com/document",
        } for i in range(2)]
        company = root / ticker
        index = company / "indexes/index.json"
        builder.shared.write_json(index, {
            "schema_version": "1.0", "chunking": {"target_words": 300, "overlap_words": 50},
            "enrichment": {"mode": "metadata"}, "chunks": chunks,
            "corpus_fingerprint": f"source-{ticker}",
        })
        builder.shared.write_json(company / "build_report.json", {
            "status": "COMPLETE", "index_sha256": builder.shared.file_sha(index),
            "inputs": {"parsed_sha256": builder.shared.file_sha(parsed)},
            "embedding_model": "same-source-model",
        })
        maps = [{"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
                 "parsed_sha256": builder.shared.file_sha(parsed)} for chunk in chunks]
        (company / "source_map.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in maps), encoding="utf-8",
        )
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            text="A short description of the source passage.", finish_reason="stop",
            provider="mock", model="test-model", input_tokens=20, output_tokens=8,
            cost_usd=0.01, latency_ms=1,
        )

    monkeypatch.setattr("finsight.llm.client.complete", complete)
    args = builder.build_parser().parse_args([
        "--source-root", str(root), "--output-root", str(tmp_path / "output"), "--no-embed",
    ])
    return args, calls


def test_prepare_reads_no_cases_and_makes_no_generation_calls(corpus):
    args, calls = corpus
    args.prepare_only = True
    args.embed = True  # Preparation must not load/download the embedding model either.
    builder.run(args)
    plan = builder.shared.read_json(args.output_root / "build_plan.json")
    config = builder.shared.read_json(args.output_root / "enrichment_configuration.json")
    assert plan["total_chunks"] == plan["new_blurb_requests_needed"] == 4
    assert config["claim_independent"] is True
    assert "cases_sha256" not in config and "selected_case_ids" not in config
    assert "selection_backends" not in config
    assert not calls
    assert not (args.output_root / "blurb_cache.jsonl").exists()


def test_every_chunk_is_enriched_and_source_provenance_is_preserved(corpus):
    args, calls = corpus
    builder.run(args)
    assert len(calls) == 4
    for ticker in ("AAA", "BBB"):
        before = builder.shared.read_json(args.source_root / ticker / "indexes/index.json")
        after = builder.shared.read_json(args.output_root / ticker / "indexes/index.json")
        for source, enriched in zip(before["chunks"], after["chunks"], strict=True):
            assert enriched == {**source, "context_blurb": "A short description of the source passage.",
                                "search_text": "A short description of the source passage.\n" + source["search_text"]}
        assert after["corpus_fingerprint"] != before["corpus_fingerprint"]
        assert (args.output_root / ticker / "source_map.jsonl").read_bytes() == (
            args.source_root / ticker / "source_map.jsonl"
        ).read_bytes()
    assert builder.shared.read_json(args.output_root / "corpus_build_report.json")["status"] == "COMPLETE"
    args.resume = True
    builder.run(args)
    assert len(calls) == 4  # Resume must not regenerate completed responses or indexes.


def test_reuses_exact_prompt_keys_without_double_counting_cost(corpus, tmp_path):
    args, calls = corpus
    args.prepare_only = True
    builder.run(args)
    configuration = builder.shared.read_json(args.output_root / "enrichment_configuration.json")
    chunk = builder.shared.read_json(args.source_root / "AAA/indexes/index.json")["chunks"][0]
    cache = tmp_path / "previous_cache.jsonl"
    cache.write_text(json.dumps({
        "key": builder.shared.blurb_key(chunk, configuration), "chunk_id": chunk["chunk_id"],
        "status": "DONE", "blurb": "Previously saved description.", "usage": {"cost_usd": 99},
    }) + "\n", encoding="utf-8")
    original = cache.read_bytes()
    args.reuse_cache = [cache]
    args.prepare_only = False
    args.resume = True
    builder.run(args)
    assert len(calls) == 3
    rows = [json.loads(line) for line in (args.output_root / "blurb_cache.jsonl").read_text().splitlines()]
    reused = [row for row in rows if "reused_from" in row]
    assert len(reused) == 1 and reused[0]["usage"] == {}
    assert cache.read_bytes() == original
    assert sum(row.get("usage", {}).get("cost_usd", 0) for row in rows) == pytest.approx(0.03)


def test_changed_model_cannot_resume_an_existing_build(corpus):
    args, calls = corpus
    args.prepare_only = True
    builder.run(args)
    args.resume = True
    args.blurb_model = "changed-model"
    with pytest.raises(ValueError, match="Existing/changed build"):
        builder.run(args)
    assert not calls


def test_unresolved_request_requires_explicit_retry(corpus):
    args, calls = corpus
    args.prepare_only = True
    builder.run(args)
    config = builder.shared.read_json(args.output_root / "enrichment_configuration.json")
    chunk = builder.shared.read_json(args.source_root / "AAA/indexes/index.json")["chunks"][0]
    (args.output_root / "blurb_cache.jsonl").write_text(json.dumps({
        "key": builder.shared.blurb_key(chunk, config), "chunk_id": chunk["chunk_id"], "status": "STARTED",
    }) + "\n", encoding="utf-8")
    args.prepare_only = False
    args.resume = True
    with pytest.raises(ValueError, match="Unresolved prior API attempt"):
        builder.run(args)
    assert not calls


def test_mismatched_cache_key_is_not_reused(corpus, tmp_path):
    args, calls = corpus
    cache = tmp_path / "other-model-cache.jsonl"
    cache.write_text(json.dumps({
        "key": "different-model-or-prompt-key", "chunk_id": "AAA-0",
        "status": "DONE", "blurb": "Do not reuse this.",
    }) + "\n", encoding="utf-8")
    args.reuse_cache = [cache]
    builder.run(args)
    assert len(calls) == 4
