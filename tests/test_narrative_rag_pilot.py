"""Offline pilot scoring and output-preservation regressions; no model calls."""
from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.narrative.evaluation.evaluate_rag_pilot import evaluate, score_direction, write_json


def test_source_span_bundle_is_not_single_hit_mrr():
    reference = {"block": {"document_id": "doc", "text": "alpha beta"},
                 "line": 1, "start": 0, "end": 10}
    maps = {
        "a": {"document_id": "doc", "block_line_1based": 1, "body_char_span": [0, 5]},
        "b": {"document_id": "doc", "block_line_1based": 1, "body_char_span": [6, 10]},
    }
    scores = score_direction([{"chunk_id": "a"}, {"chunk_id": "b"}], reference, maps)
    assert scores["known_reference_recall_at_5"] == 1
    assert scores["source_hit_at_5"] == 1
    assert scores["mrr_at_5"] == 0
    assert scores["mrr_at_10"] == 0
    assert scores["bundle_completion_rr_at_10"] == 0.5
    assert score_direction([], {}, maps)["mrr_at_10"] is None


@pytest.mark.parametrize("rank", [1, 5, 6, 10, 11])
def test_source_hits_and_mrr_respect_rank_cutoffs(rank):
    reference = {"block": {"document_id": "gold", "text": "alpha"},
                 "line": 1, "start": 0, "end": 5}
    maps = {"gold": {"document_id": "gold", "block_line_1based": 1,
                     "body_char_span": [0, 5]}}
    hits = []
    for n in range(1, rank):
        chunk_id = f"other-{n}"
        maps[chunk_id] = {"document_id": "other", "block_line_1based": 1,
                          "body_char_span": [0, 5]}
        hits.append({"chunk_id": chunk_id})
    hits.append({"chunk_id": "gold"})
    scores = score_direction(hits, reference, maps)
    for k in (5, 10):
        assert scores[f"source_hit_at_{k}"] == int(rank <= k)
        assert scores[f"mrr_at_{k}"] == pytest.approx(1 / rank if rank <= k else 0)


def test_source_hit_does_not_require_gold_passage_or_span():
    reference = {"block": {"document_id": "gold", "text": "alpha"},
                 "line": 1, "start": 0, "end": 5}
    maps = {"wrong-block": {"document_id": "gold", "block_line_1based": 2,
                            "body_char_span": [0, 5]}}
    scores = score_direction([{"chunk_id": "wrong-block"}], reference, maps)
    assert scores["source_hit_at_5"] == scores["source_hit_at_10"] == 1
    assert scores["known_reference_recall_at_5"] == scores["mrr_at_5"] == 0
    empty = score_direction([], reference, maps)
    assert empty["source_hit_at_5"] == empty["source_hit_at_10"] == empty["mrr_at_5"] == 0
    # No positive gold source exists for unsupported cases, even if background docs exist.
    unsupported = score_direction([{"chunk_id": "wrong-block"}], {}, maps)
    assert all(unsupported[k] is None for k in ("source_hit_at_5", "source_hit_at_10", "mrr_at_5"))


def saved_run(workspace: Path):
    run = workspace / "pilot"
    corpus = workspace / "strategy"
    parsed = workspace / "data/narrative/processed/AAPL/documents.jsonl"
    parsed.parent.mkdir(parents=True)
    text = "alpha\nbeta"
    block = {"document_id": "doc", "publication_date": "2024-01-01", "text": text}
    parsed.write_text(json.dumps(block) + "\n", encoding="utf-8")
    chunk = {"chunk_id": "c", "document_id": "doc", "text": text,
             "company_ticker": "AAPL", "publication_date": "2024-01-01"}
    index_path = corpus / "AAPL/indexes/index.json"
    write_json(index_path, {"corpus_fingerprint": "fingerprint", "chunks": [chunk]})
    mapping = {"chunk_id": "c", "document_id": "doc", "block_line_1based": 1,
               "body_char_span": [0, len(text)]}
    (corpus / "AAPL/source_map.jsonl").write_text(json.dumps(mapping) + "\n", encoding="utf-8")
    write_json(run / "configuration.json", {"corpus_root": str(corpus)})
    case = {"id": "AAPL-SUP-01", "company": "AAPL", "claims": "A claim", "cut_off": "2024-01-01",
            "type": "supported", "gold_snippet": text, "source_doc": "doc | test\nblock 1 | chars 0:10"}
    with (run / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case))
        writer.writeheader()
        writer.writerow(case)
    folder = run / "cases/AAPL-SUP-01"
    write_json(folder / "preflight.json", {
        "parsed_sha256": hashlib.sha256(parsed.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    })
    hit = {"chunk_id": "c", "text": text, "rank": 1}
    write_json(folder / "retrieval.json", {
        "claim": case["claims"], "cutoff": case["cut_off"],
        "index": {"corpus_fingerprint": "fingerprint"}, "support": [hit], "counter": [hit],
    })
    write_json(folder / "result.json", {
        "status": "INVALID_GENERATION", "facts": [], "summary": [],
        "candidate_evidence": [{"evidence_id": "c", "content": text}],
        "raw_model_output": json.dumps({"facts": [{"evidence_id": "c", "quote": "alpha beta"}]}),
        "citation_validation": {"passed": False, "errors": ["not an exact source quote"]},
        "llm": {"input_tokens": 10, "output_tokens": 5, "generation_ms": 100, "cost_usd": 0},
    })
    return run, parsed


def test_whitespace_diagnostic_does_not_accept_invalid_generation(tmp_path):
    run, _ = saved_run(tmp_path)
    original = (run / "cases/AAPL-SUP-01/result.json").read_bytes()
    report = evaluate(run, tmp_path)
    assert report["technical_passes"] == 0
    assert report["directional_retrieval"]["support"]["mrr_at_10"] == 1
    row = report["records"][0]
    assert row["exact_raw_quotes"] == 0
    assert row["whitespace_normalized_raw_quotes"] == 1
    assert row["accepted_summary_sentences"] == 0
    assert (run / "cases/AAPL-SUP-01/result.json").read_bytes() == original
    assert report["new_llm_requests"] == report["new_retrieval_requests"] == 0


def test_changed_source_is_not_scored_against_stale_trace(tmp_path):
    run, parsed = saved_run(tmp_path)
    parsed.write_text(parsed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Saved input hash differs"):
        evaluate(run, tmp_path)


def test_runner_preserves_existing_outputs(tmp_path):
    from scripts.narrative.evaluation.run_rag_pilot import run

    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "result.json"
    marker.write_text("keep", encoding="utf-8")
    args = Namespace(output_dir=output, corpus_root=tmp_path, top_k=10, candidate_k=50,
                     max_tokens=1500, ollama_context=20000)
    with pytest.raises(ValueError, match="not empty"):
        run(args)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_runner_captures_fresh_pipeline_without_sending_gold(tmp_path, monkeypatch):
    from finsight.agents.narrative.rag import verdict_baseline
    from finsight.llm.client import LLMResponse
    from scripts.narrative.evaluation.run_rag_pilot import run

    fixture, parsed = saved_run(tmp_path)
    corpus = tmp_path / "strategy"
    index_path = corpus / "AAPL/indexes/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["schema_version"] = "1.0"
    index["chunks"][0].update(
        source_tier="TIER_2", document_type="earnings_release", company_name="Apple Inc.",
        source_url="https://example.org/source", citation_span="block 1", local_path="source.html",
    )
    write_json(index_path, index)
    write_json(corpus / "AAPL/build_report.json", {
        "status": "COMPLETE", "corpus_fingerprint": "fingerprint",
        "inputs": {"parsed_sha256": hashlib.sha256(parsed.read_bytes()).hexdigest()},
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    })
    case = {"id": "AAPL-SUP-01", "company": "AAPL", "claims": "alpha beta",
            "cut_off": "2024-01-01", "type": "supported", "gold_snippet": "alpha\nbeta",
            "gold_answer": "GOLD ANSWER MUST NOT ENTER PROMPT",
            "source_doc": "doc | test\nblock 1 | chars 0:10"}
    csv_path = fixture / "cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case))
        writer.writeheader()
        writer.writerows([case, {**case, "id": "AAPL-SUP-02"}])
    for rel in ("agents/narrative/rag/basic_rag.py", "agents/narrative/rag/collect_evidence.py",
                "agents/narrative/rag/evidence_summary.py", "agents/narrative/retrieval/retriever.py",
                "agents/narrative/rag/verdict_baseline.py", "llm/client.py"):
        file = tmp_path / "finsight" / rel
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("# fixture identity only\n", encoding="utf-8")
    calls = []

    def fake_local(**kwargs):
        calls.append(kwargs)
        return LLMResponse(
            text=json.dumps({"selected_evidence_ids": ["c"],
                             "facts": [{"evidence_id": "c", "quote": "alpha\nbeta"}],
                             "summary": [{"text": "Source says alpha beta.", "evidence_ids": ["c"]}]}),
            input_tokens=20, output_tokens=10, cost_usd=0, latency_ms=1,
            model="fake-local", provider="ollama", finish_reason="stop",
        )

    monkeypatch.setattr(verdict_baseline, "_complete_ollama", fake_local)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "new-pilot"
    options = Namespace(
        cases=csv_path, limit=1, corpus_root=corpus, output_dir=output,
        backend="bm25", top_k=10, candidate_k=50, max_tokens=1500,
        ollama_model="fake-local", ollama_host="http://not-called", ollama_context=20000,
        rerank_overflow="error",
    )
    report = run(options)
    assert report["status"] == "COMPLETE" and len(calls) == 1
    folder = output / "cases/AAPL-SUP-01"
    capture = json.loads((folder / "llm_input.json").read_text(encoding="utf-8"))
    assert capture["prompt"] == calls[0]["prompt"]
    assert capture["system"] == calls[0]["system"]
    assert case["gold_answer"] not in capture["prompt"]
    trace = json.loads((folder / "retrieval.json").read_text(encoding="utf-8"))
    assert trace["support"][0]["rank"] == trace["counter"][0]["rank"] == 1
    assert evaluate(output, tmp_path)["technical_passes"] == 1
    options.resume = True
    resumed = run(options)
    assert len(calls) == 1 and resumed["records"][0]["reused"]
    assert resumed["records"][0]["elapsed_ms"] == report["records"][0]["elapsed_ms"]
    options.max_tokens = 1000
    with pytest.raises(ValueError, match="configuration/dataset/code differs"):
        run(options)
    assert len(calls) == 1
    options.max_tokens = 1500
    # Extend the selection within an unchanged CSV; reuse even a saved failed generation.
    first_result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    first_result["status"] = "INVALID_GENERATION"
    write_json(folder / "result.json", first_result)
    first_bytes = (folder / "result.json").read_bytes()
    options.limit = 2
    extended = run(options)
    assert len(calls) == 2  # Only the added case makes another model call.
    assert [r["reused"] for r in extended["records"]] == [True, False]
    assert extended["status"] == "COMPLETE_WITH_ERRORS"
    assert (folder / "result.json").read_bytes() == first_bytes
    assert extended["records"][0]["elapsed_ms"] == report["records"][0]["elapsed_ms"]
    assert len(evaluate(output, tmp_path)["records"]) == 2
    assert list((output / "history").glob("*/configuration.json"))
    options.limit = 1
    with pytest.raises(ValueError, match="configuration/dataset/code differs"):
        run(options)
    options.limit = 2
    (folder / "result.json").unlink()
    with pytest.raises(ValueError, match="Interrupted model attempt"):
        run(options)
    assert len(calls) == 2
    options.output_dir = tmp_path / "provider-failure"
    options.limit = 2
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case))
        writer.writeheader()
        writer.writerows([case, {**case, "id": "AAPL-SUP-02"}])

    def unavailable_local(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Could not reach Ollama")

    monkeypatch.setattr(verdict_baseline, "_complete_ollama", unavailable_local)
    with pytest.raises(ValueError, match="Model provider is unavailable"):
        run(options)
    assert len(calls) == 3  # Two earlier calls, then one service failure; no second-case call.
    failed_report = json.loads((options.output_dir / "report.json").read_text(encoding="utf-8"))
    assert failed_report["status"] == "STOPPED_PROVIDER_ERROR"
    assert not (options.output_dir / "cases/AAPL-SUP-02/result.json").exists()
    # Retrieval-only must never invoke a local or online completion, even with no real provider selected.
    options.output_dir = tmp_path / "retrieval-only"
    options.retrieval_only = True
    options.ollama_model = None
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "mock")
    retrieval_report = run(options)
    assert len(calls) == 3
    assert retrieval_report["status"] == "COMPLETE" and retrieval_report["retrieval_only"]
    assert retrieval_report["input_tokens"] == retrieval_report["output_tokens"] == 0
    assert not list(options.output_dir.glob("cases/*/llm_input.json"))
    evaluated = evaluate(options.output_dir, tmp_path)
    assert evaluated["retrieval_only_cases"] == 2 and evaluated["technical_passes"] == 0
    assert all(r["citation_validation"]["status"] == "NOT_RUN" for r in evaluated["records"])


def test_resume_accepts_only_append_and_audited_bookkeeping_migration():
    from copy import deepcopy

    from scripts.narrative.evaluation.evaluate_rag_pilot import case_identity
    from scripts.narrative.evaluation.resume import (
        PREVIOUS_CAPTURE_SHA, PREVIOUS_MATRIX_SHA, capture_compatible, matrix_compatible,
    )

    old_cases = [{"id": "one", "claims": "unchanged"}]
    new_cases = old_cases + [{"id": "two"}]
    old = {"dataset_sha256": "same-dataset", "model": "same-model", "case_ids": ["one"],
           "selected_cases_sha256": case_identity(old_cases),
           "code_hashes": {"run_rag_pilot.py": PREVIOUS_CAPTURE_SHA, "basic_rag.py": "same-code"}}
    new = {**old, "case_ids": ["one", "two"], "selected_cases_sha256": case_identity(new_cases),
           "code_hashes": {"run_rag_pilot.py": "current", "basic_rag.py": "same-code"}}
    assert capture_compatible(old, new, old_cases, new_cases)
    for key, value in (("dataset_sha256", "changed"), ("model", "changed"),
                       ("code_hashes", {"run_rag_pilot.py": "current", "basic_rag.py": "changed"})):
        changed = deepcopy(new)
        changed[key] = value
        assert not capture_compatible(old, changed, old_cases, new_cases)
    assert not capture_compatible(old, new, old_cases, [{"id": "one", "claims": "changed"}, {"id": "two"}])
    assert not capture_compatible(old, new, old_cases, list(reversed(new_cases)))
    corrupted = deepcopy(old)
    corrupted["selected_cases_sha256"] = "wrong"
    assert not capture_compatible(corrupted, new, old_cases, new_cases)
    corrupted["selected_cases_sha256"] = old["selected_cases_sha256"]
    corrupted["code_hashes"]["run_rag_pilot.py"] = "unknown-old-code"
    assert not capture_compatible(corrupted, new, old_cases, new_cases)

    old_matrix = {"dataset_sha256": "same-dataset", "selected_case_ids": ["one"],
                  "case_count_per_configuration": 1, "runner_sha256": PREVIOUS_MATRIX_SHA}
    new_matrix = {**old_matrix, "selected_case_ids": ["one", "two"],
                  "case_count_per_configuration": 2, "runner_sha256": "current"}
    assert matrix_compatible(old_matrix, new_matrix)
    assert not matrix_compatible(new_matrix, old_matrix)  # No shrinking.
    assert not matrix_compatible({**old_matrix, "runner_sha256": "unknown"}, new_matrix)


def test_output_folder_rename_migration_accepts_only_exact_verified_code_pairs():
    from scripts.narrative.evaluation.evaluate_rag_pilot import case_identity
    from scripts.narrative.evaluation.resume import OUTPUT_PATH_MIGRATIONS, capture_compatible

    cases = [{"id": "one", "claims": "unchanged"}]
    for filename, old_sha, new_sha in OUTPUT_PATH_MIGRATIONS:
        old = {"dataset_sha256": "unchanged", "case_ids": ["one"],
               "selected_cases_sha256": case_identity(cases), "code_hashes": {filename: old_sha}}
        new = {**old, "code_hashes": {filename: new_sha}}
        assert capture_compatible(old, new, cases, cases)
        assert not capture_compatible(old, {**new, "code_hashes": {filename: "other-code"}}, cases, cases)
        assert not capture_compatible({**old, "code_hashes": {filename: "other-code"}}, new, cases, cases)
        assert not capture_compatible(old, {**new, "dataset_sha256": "changed"}, cases, cases)


@pytest.mark.parametrize("prefix", ["", "C:/work/project/", "C:\\work\\project\\"])
@pytest.mark.parametrize("kind", ["matrix", "capture"])
def test_dataset_folder_rename_preserves_resume_guards(prefix, kind):
    from scripts.narrative.evaluation.evaluate_rag_pilot import case_identity
    from scripts.narrative.evaluation.resume import (
        DATASET_PATH_MIGRATIONS, capture_compatible, matrix_compatible,
    )

    cases = [{"id": "one", "claims": "unchanged"}]
    filename = "run_rag_comparison.py" if kind == "matrix" else "run_rag_pilot.py"
    _, old_sha, new_sha = next(p for p in DATASET_PATH_MIGRATIONS if p[0] == filename)
    old_path = prefix + "data/narrative/evaluation/narrative_claims_v1/cases.csv"
    new_path = prefix + "data/narrative/evaluation/test_claims_v1/cases.csv"
    old = {"dataset": old_path, "dataset_sha256": "a" * 64, "backend": "bm25"}
    if kind == "matrix":
        old.update(selected_case_ids=["one"], case_count_per_configuration=1, runner_sha256=old_sha)
        new = {**old, "dataset": new_path, "runner_sha256": new_sha}
        compatible = matrix_compatible
        changed_code = {**new, "runner_sha256": "unknown-code"}
    else:
        old.update(case_ids=["one"], selected_cases_sha256=case_identity(cases),
                   code_hashes={filename: old_sha, "evidence_summary.py": "same-code"})
        new = {**old, "dataset": new_path,
               "code_hashes": {filename: new_sha, "evidence_summary.py": "same-code"}}

        def compatible(previous, current):
            return capture_compatible(previous, current, cases, cases)

        changed_code = {**new, "code_hashes": {**new["code_hashes"], "evidence_summary.py": "changed"}}
    assert compatible(old, new)
    assert not compatible(old, changed_code)
    for change in (
        {"dataset_sha256": "b" * 64},
        {"dataset_sha256": None},
        {"dataset": prefix + "data/narrative/evaluation/other_claims_v1/cases.csv"},
        {"dataset": "D:/different/project/data/narrative/evaluation/test_claims_v1/cases.csv"},
        {"backend": "dense"},
    ):
        assert not compatible(old, {**new, **change})
    assert not compatible({k: v for k, v in old.items() if k != "dataset_sha256"},
                          {k: v for k, v in new.items() if k != "dataset_sha256"})


@pytest.mark.parametrize("retrieval_only", [False, True])
def test_matrix_preflight_expands_all_sixteen_configurations(tmp_path, monkeypatch, retrieval_only):
    from scripts.narrative.evaluation import run_rag_comparison as matrix

    dataset = tmp_path / "cases.csv"
    with dataset.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id"])
        writer.writeheader()
        writer.writerows({"id": f"case-{i}"} for i in range(126))
    options = Namespace(
        cases=dataset, all_cases=True, limit=None,
        strategies=["C1", "C2", "C3", "C4"], backends=["bm25", "dense", "hybrid", "r4"],
        threads=2, output_dir=tmp_path / "matrix", resume=False, processed_root=tmp_path / "builds",
        build_version="full_v1", top_k=10, candidate_k=50, max_tokens=1500,
        ollama_model="fake-local", ollama_host="http://not-called", ollama_context=20000,
        rerank_overflow="truncate", preflight_only=True,
        retrieval_only=retrieval_only,
    )
    jobs = []

    def fake_preflight(job):
        assert job.prepare_only
        assert job.retrieval_only is retrieval_only
        jobs.append((job.corpus_root.parent.name, job.backend, job.limit))
        return {"status": "PREFLIGHT_PASSED"}

    monkeypatch.setattr(matrix, "run_configuration", fake_preflight)
    report = matrix.run(options)
    assert len(set(jobs)) == 16 and all(job[2] == 126 for job in jobs)
    assert report["planned_case_runs"] == 2016
    assert not (options.output_dir / ".run.lock").exists()
    options.resume = True
    options.max_tokens = 1000
    with pytest.raises(ValueError, match="settings/dataset/code changed"):
        matrix.run(options)
    assert len(jobs) == 16
    assert not (options.output_dir / ".run.lock").exists()


def test_retrieval_only_matrix_does_not_report_generation_passes(tmp_path):
    from scripts.narrative.evaluation.evaluate_rag_comparison import score

    run, _ = saved_run(tmp_path)
    config = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
    config.update(backend="bm25", dataset_sha256="dataset", retrieval_only=True)
    write_json(run / "configuration.json", config)
    result_path = run / "cases/AAPL-SUP-01/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(status="OK", pipeline="narrative_evidence_retrieval", raw_model_output=None,
                  llm={"provider": "none", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
                  citation_validation={"passed": None, "status": "NOT_RUN", "errors": []})
    write_json(result_path, result)
    write_json(tmp_path / "matrix_configuration.json", {
        "selected_case_ids": ["AAPL-SUP-01"], "dataset_sha256": "dataset", "retrieval_only": True,
        "configurations": [{"strategy": "C4", "backend": "bm25", "folder": "pilot"}],
    })
    report = score(tmp_path, tmp_path)
    assert report["retrieval_only"] and report["recorded_cost_usd"] == 0
    assert all(r["ok_outputs"] == 0 and r["retrieval_only_outputs"] == 1
               for r in report["aggregates"] if r["stratum"] == "all")
    assert "generation not assessed" in (tmp_path / "pilot/evaluation.md").read_text(encoding="utf-8")
    assert "Technically accepted generated outputs: N/A" in (
        tmp_path / "pilot/evaluation.md").read_text(encoding="utf-8")
    assert "N/A (retrieval only)" in (tmp_path / "comparison.md").read_text(encoding="utf-8")


def test_matrix_scoring_keeps_missing_cases_and_unsupported_metrics_distinct(tmp_path):
    from scripts.narrative.evaluation.evaluate_rag_comparison import score

    run, _ = saved_run(tmp_path)
    config = json.loads((run / "configuration.json").read_text(encoding="utf-8"))
    config.update(backend="bm25", dataset_sha256="dataset")
    write_json(run / "configuration.json", config)
    manifest = {"selected_case_ids": ["AAPL-SUP-01"], "dataset_sha256": "dataset",
                "configurations": [{"strategy": "C4", "backend": "bm25", "folder": "pilot"}]}
    write_json(tmp_path / "matrix_configuration.json", manifest)
    report = score(tmp_path, tmp_path)
    assert report["case_configuration_runs"] == 1
    assert report["new_llm_requests"] == 0
    assert all(row["mrr_at_10"] is None for row in report["aggregates"] if row["stratum"] == "unsupported")
    assert all(row["mrr_at_10"] == 1 for row in report["aggregates"] if row["stratum"] == "all")
    assert all(row["source_hit_at_5"] == row["source_hit_at_10"] == row["mrr_at_5"] == 1
               for row in report["aggregates"] if row["stratum"] == "all")
    assert all(row["source_hit_at_5"] is None and row["mrr_at_5"] is None
               for row in report["aggregates"] if row["stratum"] == "unsupported")
    with (tmp_path / "per_case.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["source_hit_at_5"]) == float(rows[0]["mrr_at_5"]) == 1
    assert "Source Hit@5" in (tmp_path / "comparison.md").read_text(encoding="utf-8")
    manifest["selected_case_ids"].append("missing")
    write_json(tmp_path / "matrix_configuration.json", manifest)
    with pytest.raises(ValueError, match="Missing/different cases"):
        score(tmp_path, tmp_path)
