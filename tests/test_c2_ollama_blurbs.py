"""Ollama blurb requests, token accounting, scope isolation and safe resume."""

import csv
import json
from urllib.error import URLError

import pytest

from test_c2_corpus_blurbs import corpus as corpus  # Reuse the tiny metadata-only corpus fixture.
from scripts.narrative.chunking import build_c2_ollama_blurbs as builder


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


@pytest.fixture
def local_run(corpus, monkeypatch):
    old_args, shared_calls = corpus
    args = builder.build_parser().parse_args([
        "--source-root", str(old_args.source_root), "--output-root", str(old_args.output_root),
        "--all-chunks", "--ollama-model", "installed-instruct-model", "--no-embed",
    ])
    requests = []
    payload = {"message": {"content": "This source describes the company's business."},
               "done": True, "done_reason": "stop", "prompt_eval_count": 111, "eval_count": 12}

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response(payload)

    monkeypatch.setattr(builder.urllib.request, "urlopen", urlopen)

    def forbidden_settings():
        pytest.fail("Ollama blurb builds must not use shared provider settings")

    monkeypatch.setattr("finsight.config.get_settings", forbidden_settings)
    return args, requests, payload, shared_calls


def cache_rows(args):
    return [json.loads(line) for line in (args.output_root / "blurb_cache.jsonl").read_text().splitlines()]


def test_plain_text_ollama_payload_and_actual_token_counts(local_run):
    args, requests, _, shared_calls = local_run
    args.ollama_host = "http://localhost:11434/api/"
    builder.run(args)
    assert not shared_calls
    assert len(requests) == 4
    request, timeout = requests[0]
    body = json.loads(request.data)
    assert request.full_url == "http://localhost:11434/api/chat"
    assert timeout == 900
    assert body["model"] == "installed-instruct-model"
    assert body["stream"] is body["think"] is False
    assert body["messages"][0]["content"] == builder.shared.BLURB_SYSTEM
    assert body["options"] == {"num_ctx": 8192, "num_predict": 512, "temperature": 0}
    assert "format" not in body  # Blurbs are sentences, not verdict JSON.
    done = [row for row in cache_rows(args) if row["status"] == "DONE"]
    assert len(done) == 4
    assert sum(row["usage"]["input_tokens"] for row in done) == 444
    assert sum(row["usage"]["output_tokens"] for row in done) == 48
    assert all(row["usage"]["provider"] == "ollama" and row["usage"]["cost_usd"] == 0 for row in done)
    assert all(row["usage"]["token_usage_reported"] is True for row in done)
    source = builder.shared.read_json(args.source_root / "AAA/indexes/index.json")
    output = builder.shared.read_json(args.output_root / "AAA/indexes/index.json")
    for before, after in zip(source["chunks"], output["chunks"], strict=True):
        assert after["text"] == before["text"]
        assert after["chunk_id"] == before["chunk_id"]
        assert after["source_url"] == before["source_url"]
        assert after["search_text"].endswith(before["search_text"])
    args.resume = True
    builder.run(args)
    assert len(requests) == 4  # No repeated inference after completion.


def test_prepare_all_chunks_calls_neither_ollama_nor_shared_llm(local_run):
    args, requests, _, shared_calls = local_run
    args.prepare_only = True
    args.embed = True  # Must not load embeddings for an all-chunk preview.
    builder.run(args)
    assert not requests and not shared_calls
    config = builder.shared.read_json(args.output_root / "enrichment_configuration.json")
    assert config["provider"] == "ollama" and config["claim_independent"] is True
    assert "cases_sha256" not in config
    assert builder.shared.read_json(args.output_root / "build_plan.json")["selected_blurb_passages"] == 4


def test_query_selected_mode_includes_unsupported_cases_and_preserves_other_chunks(local_run, monkeypatch, tmp_path):
    args, requests, _, _ = local_run
    cases = tmp_path / "cases.csv"
    with cases.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "claims", "company", "cut_off", "type", "gold_answer"])
        writer.writeheader()
        for number, label in enumerate(("supported", "unsupported", "contradicting")):
            writer.writerow({"id": str(number), "claims": f"Claim {number}", "company": "AAA",
                             "cut_off": "2026-01-01", "type": label, "gold_answer": "Never read this"})
    args.all_chunks, args.cases, args.selection_backends = False, cases, ["bm25"]

    def shortlist(args, selected, backend, output, signature):
        assert [case["id"] for case in selected] == ["0", "1", "2"]
        assert all(set(case) == {"id", "claims", "company", "cut_off"} for case in selected)
        assert backend is None
        return {"AAA": ["AAA-0"]}

    monkeypatch.setattr(builder.shared, "shortlist", shortlist)
    builder.run(args)
    assert len(requests) == 1
    before = builder.shared.read_json(args.source_root / "AAA/indexes/index.json")["chunks"]
    after = builder.shared.read_json(args.output_root / "AAA/indexes/index.json")["chunks"]
    assert "context_blurb" in after[0]
    assert after[1] == before[1]
    assert builder.shared.read_json(args.output_root / "build_plan.json")["cases"] == 3


@pytest.mark.parametrize("change", ["model", "context", "host", "tokens"])
def test_model_or_request_settings_change_cannot_mix_into_old_output(local_run, change):
    args, requests, _, _ = local_run
    args.prepare_only = True
    builder.run(args)
    args.resume = True
    if change == "model":
        args.ollama_model = "other-model"
    elif change == "context":
        args.ollama_context += 1024
    elif change == "host":
        args.ollama_host = "http://localhost:11435"
    else:
        args.blurb_max_tokens += 100
    with pytest.raises(ValueError, match="Existing/changed build"):
        builder.run(args)
    assert not requests


def test_unavailable_ollama_has_no_api_fallback_or_automatic_retry(local_run, monkeypatch):
    args, requests, _, shared_calls = local_run

    def unreachable(request, timeout):
        requests.append((request, timeout))
        raise URLError("connection refused")

    monkeypatch.setattr(builder.urllib.request, "urlopen", unreachable)
    with pytest.raises(RuntimeError, match="Ollama request failed"):
        builder.run(args)
    assert len(requests) == 1 and not shared_calls
    assert cache_rows(args)[-1]["status"] == "FAILED"
    args.resume = True
    with pytest.raises(ValueError, match="Unresolved Ollama attempts"):
        builder.run(args)
    assert len(requests) == 1


@pytest.mark.parametrize("failure", ["length", "thinking_only", "not_done"])
def test_failed_generation_preserves_tokens_and_requires_explicit_retry(local_run, failure):
    args, requests, payload, _ = local_run
    if failure == "length":
        payload["done_reason"] = "length"
    elif failure == "thinking_only":
        payload["message"] = {"content": "", "thinking": "Internal reasoning"}
    else:
        payload["done"] = False
    with pytest.raises(ValueError):
        builder.run(args)
    failed = cache_rows(args)[-1]
    assert failed["status"] == "FAILED"
    assert failed["usage"]["input_tokens"] == 111 and failed["usage"]["output_tokens"] == 12
    assert failed["error"]
    args.resume = True
    with pytest.raises(ValueError, match="Unresolved Ollama attempts"):
        builder.run(args)
    assert len(requests) == 1
    payload.update(done=True, done_reason="stop", message={"content": "A valid final sentence."})
    args.retry_incomplete = True
    builder.run(args)
    assert len(requests) == 5  # Failed first attempt plus four successful requests.


def test_missing_token_counts_are_explicitly_marked_unknown(local_run):
    args, _, payload, _ = local_run
    del payload["prompt_eval_count"]
    del payload["eval_count"]
    builder.run(args)
    done = [row for row in cache_rows(args) if row["status"] == "DONE"]
    assert all(row["usage"]["token_usage_reported"] is False for row in done)


def test_all_chunks_rejects_claims_file(local_run):
    args, requests, _, _ = local_run
    args.cases = args.source_root / "nonexistent.csv"
    with pytest.raises(ValueError, match="does not accept"):
        builder.run(args)
    assert not requests


@pytest.mark.parametrize("host", ["ftp://localhost:11434", "http://user:password@localhost:11434",
                                  "http://localhost:11434/other", "http://localhost:11434?token=secret"])
def test_rejects_ambiguous_or_credential_bearing_host_urls(host):
    with pytest.raises(ValueError, match="Ollama host"):
        builder.normalize_host(host)
