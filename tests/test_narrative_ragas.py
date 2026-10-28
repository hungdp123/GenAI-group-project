"""Saved-output evaluation regressions; no provider/model calls."""
from argparse import Namespace
import asyncio
from copy import deepcopy
import json

from pydantic import BaseModel
import pytest

from finsight.llm.client import LLMResponse, data_block
from scripts.narrative.evaluation import evaluate_ragas as evaluator
from scripts.narrative.evaluation.evaluate_rag_pilot import case_identity, write_json


def example():
    case = {"id": "AAPL-CTR-01", "claims": "Apple's product works only offline.",
            "company": "AAPL", "cut_off": "2024-08-01", "type": "contradicting",
            "gold_answer": "It supports both offline and cloud processing.",
            "gold_snippet": "both offline and cloud", "source_doc": "original document",
            "expected_behavior": "Preserve the conflicting capability."}
    sources = [{"evidence_id": "cloud", "content": "It supports both offline and cloud processing."},
               {"evidence_id": "unrelated", "content": "An unrelated product feature."}]
    passages = [{"evidence_id": p["evidence_id"], "company": "Apple", "publication_date": "2024-08-01",
                 "source_spans": [{"source_span_id": "s1", "text": p["content"]}]} for p in sources]
    captured = {"prompt": data_block("request", json.dumps({"ticker": "AAPL", "claim": case["claims"],
                                                           "cutoff_date": case["cut_off"]})) + "\n" +
                data_block("source_passages", json.dumps(passages))}
    result = {"pipeline": "narrative_evidence_rag", "status": "OK", "claim": case["claims"],
              "ticker": "AAPL", "cutoff_date": case["cut_off"],
              "evidence_preparation_protocol": "source_spans_v2", "candidate_evidence": sources,
              "citation_validation": {"passed": True}, "selected_evidence_ids": ["cloud"],
              "facts": [{"evidence_id": "cloud", "quote": sources[0]["content"]}],
              "summary": [{"text": "The product uses offline and cloud processing.", "evidence_ids": ["cloud"]}]}
    return case, result, captured


def test_actual_contexts_selected_contexts_and_cited_contexts_are_distinct():
    sample = evaluator.build_sample(*example())
    assert len(sample["retrieved_contexts"]) == 2
    assert len(sample["selected_contexts"]) == 1
    assert sample["reference"] == "It supports both offline and cloud processing."
    jobs = {name: (metric, kwargs) for name, metric, kwargs in evaluator.tasks(sample)}
    assert len(jobs["faithfulness"][1]["retrieved_contexts"]) == 2
    assert len(jobs["evidence_completeness"][1]["retrieved_contexts"]) == 1
    citation = jobs["citation_1"][1]["material"]
    assert "cloud" in citation and "unrelated" not in citation
    assert "gold_answer" not in jobs["relevance"][1]["material"]


def test_stance_protocol_keeps_ragas_contexts_and_checks_stance_references():
    case, result, captured = example()
    result.update(evidence_preparation_protocol="source_spans_stance_v3", assessment_status="ASSESSED",
                  stance="CONTRADICTING", stance_reason="The source explicitly includes cloud processing.",
                  stance_reason_code="DIRECT_CONTRADICTION", supporting_fact_indices=[],
                  contradicting_fact_indices=[0], missing_claim_parts=[])
    sample = evaluator.build_sample(case, result, captured)
    assert sample["narrative_stance"] == "CONTRADICTING"
    assert len(sample["retrieved_contexts"]) == 2
    assert sample["response"] == result["summary"][0]["text"]
    result["contradicting_fact_indices"] = [7]
    with pytest.raises(ValueError, match="Invalid saved stance assessment"):
        evaluator.build_sample(case, result, captured)


def stance_example():
    case, result, captured = example()
    result.update(evidence_preparation_protocol="source_spans_stance_v3", assessment_status="ASSESSED",
                  stance="CONTRADICTING", stance_reason="The source explicitly includes cloud processing.",
                  stance_reason_code="DIRECT_CONTRADICTION", supporting_fact_indices=[],
                  contradicting_fact_indices=[0], missing_claim_parts=[])
    return case, result, captured


def test_stance_judges_receive_label_reason_and_only_referenced_citation_context():
    sample = evaluator.build_sample(*stance_example())
    jobs = {name: (metric, kwargs) for name, metric, kwargs in evaluator.tasks(sample)}
    reason = jobs["stance_justification"][1]["material"]
    assert "CONTRADICTING" in reason and sample["narrative_stance_reason"] in reason
    assert "expected_stance" not in reason and '"reference"' not in reason
    citation = jobs["stance_citation_contradicting_0"][1]["material"]
    assert "cloud" in citation and "unrelated" not in citation
    assert len(jobs["retrieved_context_precision"][1]["retrieved_contexts"]) == 2
    assert len(jobs["retrieved_context_recall"][1]["retrieved_contexts"]) == 2
    assert len(jobs["evidence_selection_quality"][1]["retrieved_contexts"]) == 1


def test_unsupported_rubric_receives_stance_and_missing_parts():
    case, result, captured = stance_example()
    case.update(type="unsupported", gold_snippet="")
    result.update(stance="UNSUPPORTED", stance_reason="No supplied passage establishes the requested feature.",
                  stance_reason_code="NO_RELEVANT_EVIDENCE", selected_evidence_ids=[], facts=[], summary=[],
                  contradicting_fact_indices=[], missing_claim_parts=["The requested feature."])
    sample = evaluator.build_sample(case, result, captured)
    jobs = {name: (metric, kwargs) for name, metric, kwargs in evaluator.tasks(sample)}
    material = jobs["unsupported_behavior"][1]["material"]
    assert "UNSUPPORTED" in material and "missing_claim_parts" in material
    assert jobs["stance_justification"][0] == "stance_justification"
    assert jobs["retrieved_context_precision"][0] is None
    assert jobs["retrieved_context_recall"][0] is None
    assert not any(k.startswith("stance_citation_") for k in jobs)


def test_partial_case_retains_reference_metrics_and_checks_refusal():
    case, result, captured = stance_example()
    case["type"] = "partially supported"
    result.update(stance="UNSUPPORTED", stance_reason="Only part of the claim is established.",
                  stance_reason_code="PARTIAL_SUPPORT", supporting_fact_indices=[0],
                  contradicting_fact_indices=[], missing_claim_parts=["The remaining assertion."])
    sample = evaluator.build_sample(case, result, captured)
    assert sample["expected_stance"] == "UNSUPPORTED" and sample["stance_correct"] is True
    jobs = {name: key for name, key, _ in evaluator.tasks(sample)}
    assert jobs["evidence_completeness"] == "evidence_completeness"
    assert jobs["retrieved_context_recall"] == "evidence_completeness"
    assert jobs["unsupported_behavior"] == "unsupported_behavior"


def test_empty_retrieval_refusal_is_assessed_but_not_a_perfect_evidence_score():
    case, result, _ = stance_example()
    result.update(status="NO_EVIDENCE", stance="UNSUPPORTED", stance_reason="No passages were retrieved.",
                  stance_reason_code="NO_RELEVANT_EVIDENCE", selected_evidence_ids=[], facts=[], summary=[],
                  candidate_evidence=[], contradicting_fact_indices=[], missing_claim_parts=[case["claims"]],
                  citation_validation={"passed": None})
    sample = evaluator.build_sample(case, result, None)
    assert sample["judge_eligible"] and sample["predicted_stance"] == "UNSUPPORTED"
    assert sample["stance_correct"] is False  # A valid refusal can still be wrong against the gold label.
    jobs = {name: (metric, kwargs) for name, metric, kwargs in evaluator.tasks(sample)}
    assert jobs["retrieved_context_recall"][0] == "zero"
    assert jobs["evidence_completeness"][0] == "zero"
    assert jobs["stance_justification"][0] == "stance_justification"


def test_ragas_stance_report_normalizes_rubric_and_preserves_citation_reasons(tmp_path, monkeypatch):
    sample = {**evaluator.build_sample(*stance_example()), "configuration": "C1/bm25",
              "dataset_sha256": "dataset", "source_result": "saved.json"}
    monkeypatch.setattr(evaluator, "prepare", lambda _: [sample])
    monkeypatch.setattr(evaluator, "version", lambda _: evaluator.RAGAS_VERSION)
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    names = ("faithfulness", "evidence_completeness", "evidence_selection_quality", "summary_completeness",
             "relevance", "citation", "stance_justification", "stance_citation")
    monkeypatch.setattr(evaluator, "make_metrics", lambda _: {n: n for n in names})
    calls = []
    async def score(metric, kwargs):
        calls.append(metric)
        return {"status": "SCORED", "value": 2 if metric in {"relevance", "stance_justification"} else 1,
                "reason": "The cited source supports this relation."}
    monkeypatch.setattr(evaluator, "score_task", score)
    args = Namespace(run_dir=tmp_path / "saved", output_dir=tmp_path / "judge", limit=1, all_cases=False,
                     max_tokens=100, ollama_context=20000, ollama_model=None, judge_model="test",
                     resume=False, prepare_only=False, retry_errors=False, ollama_host="http://localhost:11434")
    report = asyncio.run(evaluator.run(args))
    row = report["records"][0]
    assert row["metrics"]["stance_justification"]["value"] == 1
    assert row["metrics"]["stance_citation_correctness"]["value"] == 1
    assert report["stance_metrics"]["C1/bm25"]["accuracy"] == 1
    assert not any(r["metric"].startswith("stance_citation_contradicting") for r in report["aggregates"])
    assert (args.output_dir / "samples/C1/bm25" / sample["id"] / "stance_citation_contradicting_0.json").exists()
    assert "stance_justification" in (args.output_dir / "ragas_per_case.csv").read_text(encoding="utf-8-sig")
    count = len(calls)
    args.resume = True
    asyncio.run(evaluator.run(args))
    assert len(calls) == count  # New metrics follow the existing checkpoint/resume rules.


def test_reject_mismatched_captured_sources_and_unknown_citations():
    case, result, captured = example()
    wrong = deepcopy(result)
    wrong["candidate_evidence"][0]["content"] = "A different source"
    with pytest.raises(ValueError, match="source spans differ"):
        evaluator.build_sample(case, wrong, captured)
    wrong = deepcopy(result)
    wrong["summary"][0]["evidence_ids"] = ["unrelated"]
    with pytest.raises(ValueError, match="unselected evidence"):
        evaluator.build_sample(case, wrong, captured)
    with pytest.raises(ValueError, match="request differs"):
        evaluator.build_sample({**case, "cut_off": "2024-08-02"}, result, captured)


def test_source_spans_allow_only_the_generators_blank_segment_omissions():
    from finsight.agents.narrative.rag.evidence_summary import _source_spans

    class Passage:
        evidence_id = "source"
        content = "\n \nRevenue was 34%.\n\n \nIt includes  cloud services.  \n"

        def model_dump(self, **kwargs):
            return {"evidence_id": self.evidence_id}

    original = Passage()
    passages, _ = _source_spans([original])
    spans = passages[0]["source_spans"]
    text = evaluator.source_span_text(original.content, spans)
    assert text != original.content  # Blank-only pieces are intentionally absent from the real prompt.
    assert "Revenue was 34%" in text and "includes  cloud" in text
    for bad in (spans[1:], list(reversed(spans)), spans + spans[:1],
                [{**s, "text": s["text"].replace("34%", "36%")} for s in spans],
                [{**s, "text": s["text"].replace("includes  cloud", "includes cloud")} for s in spans],
                [{**s, "source_span_id": "s99"} for s in spans]):
        with pytest.raises(ValueError, match="source spans differ"):
            evaluator.source_span_text(original.content, bad)


def test_evaluator_resume_admits_only_audited_input_validator_change():
    from scripts.narrative.evaluation.resume import EVALUATOR_INPUT_MIGRATIONS, evaluator_compatible

    old, new = next(iter(EVALUATOR_INPUT_MIGRATIONS))
    previous = {"code_sha256": old, "judge_model": "same", "client_sha256": "same"}
    current = {**previous, "code_sha256": new}
    assert evaluator_compatible(previous, current)
    assert not evaluator_compatible(previous, {**current, "judge_model": "different"})
    assert not evaluator_compatible(previous, {**current, "client_sha256": "different"})
    assert not evaluator_compatible(previous, {**current, "code_sha256": "unreviewed-change"})


def test_empty_unsupported_output_does_not_get_perfect_faithfulness_or_reference_recall():
    case, result, captured = example()
    case.update(type="unsupported", gold_snippet="")
    result.update(selected_evidence_ids=[], facts=[], summary=[])
    jobs = {name: (metric, kwargs) for name, metric, kwargs in evaluator.tasks(
        evaluator.build_sample(case, result, captured))}
    assert jobs["faithfulness"][0] is None
    assert jobs["evidence_completeness"][0] is None
    assert jobs["evidence_selection_quality"][0] is None
    assert jobs["unsupported_behavior"][0] == "unsupported_behavior"
    assert jobs["relevance"][0] is None


def test_generation_failure_is_not_a_semantic_zero():
    case, result, _ = example()
    result["status"] = "GENERATION_ERROR"
    sample = evaluator.build_sample(case, result, None)
    assert not sample["judge_eligible"]
    assert all(metric is None for _, metric, _ in evaluator.tasks(sample))
    result["pipeline"] = "narrative_evidence_retrieval"
    with pytest.raises(ValueError, match="Retrieval-only"):
        evaluator.build_sample(case, result, None)


def test_judge_checkpoint_preserves_usage_and_requires_explicit_retry(tmp_path, monkeypatch):
    class Output(BaseModel):
        value: int

    args = Namespace(output_dir=tmp_path, retry_errors=False, ollama_model=None,
                     judge_model="test", max_tokens=100)
    calls = []

    def complete(prompt, **kwargs):
        calls.append(prompt)
        return LLMResponse('{"value": 1}', 20, 5, .001, 3, "test", "test", "stop")

    monkeypatch.setattr(evaluator, "complete", complete)
    transport = evaluator.Transport(args)
    assert transport.generate("task", Output).value == 1
    assert transport.generate("task", Output).value == 1
    assert len(calls) == 1

    def invalid(*a, **kw):
        return LLMResponse("invalid", 20, 5, .001, 3, "test", "test", "length")

    monkeypatch.setattr(evaluator, "complete", invalid)
    with pytest.raises(ValueError, match="output limit"):
        transport.generate("other task", Output)
    with pytest.raises(RuntimeError, match="retry-errors"):
        transport.generate("other task", Output)
    args.retry_errors = True
    monkeypatch.setattr(evaluator, "complete", complete)
    assert transport.generate("other task", Output).value == 1
    histories = [json.loads(p.read_text()) for p in (tmp_path / "calls").glob("*.json")]
    assert sum(len(h["attempts"]) for h in histories) == 3
    assert all("response" in a for h in histories for a in h["attempts"])


def test_resume_extension_reuses_metrics_and_rejects_changed_inputs(tmp_path, monkeypatch):
    case, result, captured = example()
    sample = {**evaluator.build_sample(case, result, captured), "configuration": "C1/bm25",
              "dataset_sha256": "unchanged", "source_result": "saved.json"}
    samples = [sample]
    monkeypatch.setattr(evaluator, "prepare", lambda args: samples)
    monkeypatch.setattr(evaluator, "version", lambda name: evaluator.RAGAS_VERSION)
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(evaluator, "make_metrics", lambda args: {k: k for k in (
        "faithfulness", "evidence_completeness", "evidence_selection_quality", "summary_completeness",
        "relevance", "citation")})
    judged = []

    async def score(metric, kwargs):
        judged.append(metric)
        return {"status": "SCORED", "value": 2 if metric == "relevance" else 1., "reason": "mock judge"}

    monkeypatch.setattr(evaluator, "score_task", score)
    args = Namespace(run_dir=tmp_path / "rag", output_dir=tmp_path / "judge", limit=1, all_cases=False,
                     max_tokens=100, ollama_context=20000, ollama_model=None, judge_model="test",
                     resume=False, prepare_only=False, retry_errors=False, ollama_host="http://localhost:11434")
    report = asyncio.run(evaluator.run(args))
    assert report["records"][0]["metrics"]["relevance"]["value"] == 1
    count = len(judged)
    args.resume = True
    samples.append({**sample, "id": "AAPL-CTR-02"})
    args.all_cases = True
    args.limit = None
    report = asyncio.run(evaluator.run(args))
    assert len(judged) == count * 2  # Only the added sample is scored.
    assert report["saved_runs"] == 2
    samples[0] = {**sample, "reference": "Changed gold answer"}
    with pytest.raises(ValueError, match="reference changed"):
        asyncio.run(evaluator.run(args))
    assert len(judged) == count * 2


def test_prepare_exports_without_importing_ragas(tmp_path, monkeypatch):
    import csv

    case, result, captured = example()
    root = tmp_path / "saved"
    write_json(root / "configuration.json", {"selected_cases_sha256": case_identity([case]),
                                             "dataset_sha256": "dataset"})
    with (root / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(case))
        writer.writeheader()
        writer.writerow(case)
    write_json(root / "cases" / case["id"] / "result.json", result)
    write_json(root / "cases" / case["id"] / "llm_input.json", captured)
    args = Namespace(run_dir=root, output_dir=tmp_path / "judge", limit=1, all_cases=False,
                     max_tokens=100, ollama_context=20000, ollama_model=None, judge_model="test",
                     resume=False, prepare_only=True, retry_errors=False, ollama_host="http://localhost:11434")
    monkeypatch.setattr(evaluator, "make_metrics", lambda _: pytest.fail("Must not load RAGAS"))
    assert asyncio.run(evaluator.run(args))["status"] == "PREPARED_NO_MODEL_CALLS"
    assert (args.output_dir / "samples/single" / case["id"] / "input.json").exists()


def test_real_ragas_metric_interfaces_with_mocked_structured_judge(tmp_path, monkeypatch):
    """Optional dependency smoke test: real metrics and schemas, zero provider requests."""
    monkeypatch.setenv("RAGAS_DO_NOT_TRACK", "true")
    pytest.importorskip("ragas")
    responses = {
        "StatementGeneratorOutput": {"statements": ["The product uses cloud processing."]},
        "NLIStatementOutput": {"statements": [{"statement": "The product uses cloud processing.",
                                               "reason": "Mock entailment", "verdict": 1}]},
        "ContextRecallOutput": {"classifications": [{"statement": "The product uses cloud processing.",
                                                     "reason": "Mock coverage", "attributed": 1}]},
        "ContextPrecisionOutput": {"reason": "Mock relevance", "verdict": 1},
        "ClaimDecompositionOutput": {"claims": ["The product uses cloud processing."]},
        "DiscreteResponseModel": {"reason": "Mock criterion", "value": 1},
    }
    calls = []

    def generate(self, prompt, response_model):
        calls.append(response_model.__name__)
        return response_model.model_validate(responses[response_model.__name__])

    monkeypatch.setattr(evaluator.Transport, "generate", generate)
    metrics = evaluator.make_metrics(Namespace(output_dir=tmp_path))
    sample = evaluator.build_sample(*stance_example())

    async def score_all():
        return [await evaluator.score_task(metrics[key], kwargs)
                for _, key, kwargs in evaluator.tasks(sample) if key]

    results = asyncio.run(score_all())
    assert len(results) == 10 and all(r["status"] == "SCORED" for r in results)
    assert all(0 <= r["value"] <= 1 for r in results)
    assert {"ContextRecallOutput", "ContextPrecisionOutput", "DiscreteResponseModel"} <= set(calls)
