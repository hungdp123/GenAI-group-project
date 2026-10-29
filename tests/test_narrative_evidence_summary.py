"""Exact source-span extraction and bounded output regressions; no real LLM calls."""
from datetime import date
import json
from types import SimpleNamespace

import pytest

from finsight.agents.narrative.rag import evidence_summary as summary
from finsight.contracts.schemas import EvidenceItem, SubClaim, SubClaimType
from finsight.llm.client import LLMResponse


def item(eid="source", content="Product scope.\nSupports multiple clouds, but excludes offline devices."):
    return EvidenceItem(evidence_id=eid, sub_claim_id="sc1", stance="NEUTRAL", content=content,
                        source_tier="TIER_2", doc_type="earnings_release", company="Example",
                        publication_date=date(2024, 1, 1), produced_by="narrative_retrieval_unassessed")


def stance_fields():
    return {"stance": "SUPPORTED", "stance_reason": "The cited source states the claim.",
            "stance_reason_code": "ALL_MATERIAL_PARTS_SUPPORTED", "supporting_fact_indices": [0],
            "contradicting_fact_indices": [], "missing_claim_parts": []}


def selection(eid="source", span="s2"):
    return {**stance_fields(), "selected_evidence_ids": [eid], "facts": [{"evidence_id": eid, "source_span_id": span}],
            "summary": [{"text": "The source describes multi-cloud support with an offline limitation.",
                         "evidence_ids": [eid]}]}


def linked_selection(eid="source", span="s2"):
    return {**stance_fields(), "facts": [{"evidence_id": eid, "source_span_id": span}],
            "summary": [{"text": "The source describes multi-cloud support with an offline limitation.",
                         "fact_indices": [0]}]}


def test_selection_and_public_citations_are_derived_only_from_provided_facts():
    items = [item(), item("second", "Qualification: online access required.")]
    _, catalog = summary._source_spans(items)
    value = linked_selection()
    value["facts"].append({"evidence_id": "second", "source_span_id": "s1"})
    value["summary"][0]["fact_indices"] = [1, 0, 1]
    resolved, _, errors = summary._resolve_spans(json.dumps(value), items, catalog)
    parsed, checks = summary._validate(resolved, items)
    assert not errors and not checks
    assert parsed["selected_evidence_ids"] == ["source", "second"]
    assert parsed["summary"][0]["evidence_ids"] == ["second", "source"]
    assert all(f["quote"] in next(i.content for i in items if i.evidence_id == f["evidence_id"])
               for f in parsed["facts"])


@pytest.mark.parametrize("refs", [[], [1], [-1], [True], ["0"], None])
def test_missing_fact_references_are_rejected_without_adding_facts(refs):
    items = [item()]
    _, catalog = summary._source_spans(items)
    value = linked_selection()
    value["summary"][0]["fact_indices"] = refs
    _, anchors, errors = summary._resolve_spans(json.dumps(value), items, catalog)
    assert errors and len(anchors) == 1
    assert len(value["facts"]) == 1  # No invented replacement fact or dropped summary.


def test_legacy_unbacked_selected_passage_still_fails():
    items = [item(), item("unbacked", "A separate passage.")]
    _, catalog = summary._source_spans(items)
    value = selection()
    value["selected_evidence_ids"].append("unbacked")
    value["summary"].append({"text": "Unbacked assertion.", "evidence_ids": ["unbacked"]})
    resolved, _, _ = summary._resolve_spans(json.dumps(value), items, catalog)
    parsed, errors = summary._validate(resolved, items)
    assert parsed is None
    assert any("Each selected passage" in e for e in errors)
    assert any("without an exact-source fact" in e for e in errors)


def test_derived_selection_still_enforces_passage_budget_and_exact_quote_checks():
    items = [item(str(n), "Original statement.") for n in range(summary.MAX_SELECTED + 1)]
    _, catalog = summary._source_spans(items)
    value = {"facts": [{"evidence_id": i.evidence_id, "source_span_id": "s1"} for i in items],
             "summary": [{"text": "Sources report the original statements.", "fact_indices": [0]}]}
    _, _, errors = summary._resolve_spans(json.dumps(value), items, catalog)
    assert any("selected_evidence_ids exceeds" in e for e in errors)
    value = linked_selection(eid="0")
    value["facts"] = [{"evidence_id": "0", "quote": "Invented statement."}]
    resolved, _, _ = summary._resolve_spans(json.dumps(value), items, catalog)
    parsed, errors = summary._validate(resolved, items)
    assert parsed is None and any("exact source quote" in e for e in errors)


def test_source_span_text_is_lossless_with_tables_unicode_and_line_breaks():
    text = 'Scope — US only.\r\nPeriod | 2024 | 2025\nCustomers | 32 | 34\nDo not round 3.27 to 3.3.'
    evidence = item(content=text)
    passages, catalog = summary._source_spans([evidence])
    assert "content" not in passages[0] and "stance" not in passages[0]
    assert "".join(s["text"] for s in passages[0]["source_spans"]) == text
    for span in passages[0]["source_spans"]:
        start, end = catalog[evidence.evidence_id, span["source_span_id"]]
        assert text[start:end] == span["text"]
    # Decimal values are not split at the decimal point.
    assert any("3.27" in span["text"] for span in passages[0]["source_spans"])


def test_compact_prompt_metadata_retains_attribution_and_estimated_date_qualifications():
    evidence = item().model_copy(update={
        "period": "FY2024_Q1", "publication_date_is_estimated": True,
        "publication_date_precision": "estimated_day", "publication_date_basis": "Call date proxy; review required",
        "url": "https://example.test/official.pdf", "citation_span": "PDF page 3",
    })
    passages, _ = summary._source_spans([evidence])
    passage = passages[0]
    for key in ("period", "publication_date_is_estimated", "publication_date_precision",
                "publication_date_basis", "url", "citation_span", "company", "doc_type"):
        assert passage[key] == getattr(evidence, key)
    assert not {"sub_claim_id", "produced_by", "computed_value", "comparison_value"}.intersection(passage)
    assert "".join(s["text"] for s in passage["source_spans"]) == evidence.content


def test_span_resolution_copies_exact_text_and_preserves_original_evidence():
    evidence = item(content='Apple\n®\ntoday announced results.\nScope: device and server, excluding offline use.')
    original = evidence.model_dump()
    _, catalog = summary._source_spans([evidence])
    value = selection(span="s4")
    resolved, anchors, errors = summary._resolve_spans(json.dumps(value), [evidence], catalog)
    parsed, validation = summary._validate(resolved, [evidence])
    assert not errors and not validation
    assert parsed["facts"][0]["quote"] == 'Scope: device and server, excluding offline use.'
    anchor = anchors[0]
    assert evidence.content[anchor["char_start"]:anchor["char_end"]] == parsed["facts"][0]["quote"]
    assert evidence.model_dump() == original
    assert "source_span_id" in value["facts"][0]  # Original raw response was not rewritten.


@pytest.mark.parametrize("fact", [
    {"evidence_id": "unknown", "source_span_id": "s1"},
    {"evidence_id": "source", "source_span_id": "s999"},
    {"evidence_id": "source", "source_span_id": ["s1"]},
    {"evidence_id": "source", "source_span_id": "s1", "quote": "fabricated"},
])
def test_invented_or_malformed_span_references_are_rejected(fact):
    evidence = item()
    _, catalog = summary._source_spans([evidence])
    raw = selection()
    raw["facts"] = [fact]
    _, _, errors = summary._resolve_spans(json.dumps(raw), [evidence], catalog)
    assert errors


def test_literal_quote_fallback_still_rejects_changed_words_and_stitched_sentences():
    evidence = item(content="Supports public cloud. Does not support offline use. Requires internet.")
    _, catalog = summary._source_spans([evidence])
    for quote in ("Supports private cloud.", "Supports public cloud. Requires internet."):
        raw = selection()
        raw["facts"] = [{"evidence_id": "source", "quote": quote}]
        resolved, _, _ = summary._resolve_spans(json.dumps(raw), [evidence], catalog)
        parsed, errors = summary._validate(resolved, [evidence])
        assert parsed is None and any("exact source quote" in e for e in errors)


@pytest.mark.parametrize("field,limit", [
    ("selected_evidence_ids", summary.MAX_SELECTED), ("facts", summary.MAX_FACTS),
    ("summary", summary.MAX_SUMMARY_SENTENCES),
])
def test_over_budget_response_is_rejected_not_silently_trimmed(field, limit):
    evidence = item()
    _, catalog = summary._source_spans([evidence])
    raw = selection()
    raw[field] = raw[field] * (limit + 1)
    _, _, errors = summary._resolve_spans(json.dumps(raw), [evidence], catalog)
    assert any("exceeds" in e for e in errors)


@pytest.mark.parametrize("finish,expected", [("stop", "OK"), ("length", "INVALID_GENERATION")])
def test_single_llm_call_resolves_facts_but_never_accepts_truncated_output(monkeypatch, finish, expected):
    evidence = item()
    claim = SubClaim(sub_claim_id="sc1", text="Product supports multiple clouds.", company_ticker="AAPL",
                     sub_claim_type=SubClaimType.OTHER)
    monkeypatch.setattr(summary, "get_settings", lambda: SimpleNamespace(llm_provider="openai", llm_model="fake"))
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        assert "source_spans" in kwargs["prompt"] and "at most 4 passages" in kwargs["system"]
        return LLMResponse(text=json.dumps(linked_selection()), input_tokens=100, output_tokens=80,
                           cost_usd=0.001, latency_ms=1, model="fake", provider="openai", finish_reason=finish)

    monkeypatch.setattr(summary, "complete", complete)
    selected, report = summary.prepare_narrative_evidence(claim, date(2024, 1, 2), [evidence])
    assert len(calls) == 1 and report["llm"]["api_requests"] == 1
    assert report["status"] == expected
    assert report["evidence_preparation_protocol"] == "source_spans_stance_v3"
    assert "source_span_id" in json.loads(report["raw_model_output"])["facts"][0]
    assert selected == [evidence] and "verdict" not in report
    if expected == "OK":
        assert report["facts"][0]["quote"] == "Supports multiple clouds, but excludes offline devices."
        assert report["citation_validation"]["passed"] is True
        assert report["assessment_status"] == "ASSESSED" and report["stance"] == "SUPPORTED"
    else:
        assert not report["facts"] and not report["summary"]
        assert report["selection_applied"] is False
        assert report["assessment_status"] == "ERROR" and report["stance"] is None


@pytest.mark.parametrize("label,code,support,against,missing", [
    ("SUPPORTED", "ALL_MATERIAL_PARTS_SUPPORTED", [0], [], []),
    ("CONTRADICTING", "DIRECT_CONTRADICTION", [0], [1], []),
    ("UNSUPPORTED", "PARTIAL_SUPPORT", [0], [], ["Global availability is not established."]),
    ("UNSUPPORTED", "UNRESOLVED_CONFLICT", [0], [1], ["The sources disagree on launch scope."]),
])
def test_stance_and_both_sides_keep_exact_source_references(monkeypatch, label, code, support, against, missing):
    first = (
        "The platform launched globally in June." if code == "UNRESOLVED_CONFLICT" else "The platform launched in June."
    )
    second = "Availability was not specified." if code == "PARTIAL_SUPPORT" else "The June launch was US-only."
    items = [item("date", first), item("scope", second)]
    claim_text = (
        "The platform launched in the US in June."
        if label == "SUPPORTED" else "The platform launched globally in June."
    )
    claim = SubClaim(sub_claim_id="sc1", text=claim_text,
                     company_ticker="AAPL", sub_claim_type=SubClaimType.OTHER)
    output = {"facts": [{"evidence_id": i.evidence_id, "source_span_id": "s1"} for i in items],
              "summary": [{"text": "The source describes a June launch in the US.", "fact_indices": [0, 1]}],
              "stance": label, "stance_reason": "The launch date is established; assess availability separately.",
              "stance_reason_code": code, "supporting_fact_indices": support,
              "contradicting_fact_indices": against, "missing_claim_parts": missing}
    monkeypatch.setattr(summary, "get_settings", lambda: SimpleNamespace(llm_provider="openai", llm_model="fake"))
    monkeypatch.setattr(summary, "complete", lambda **kwargs: LLMResponse(
        text=json.dumps(output), input_tokens=10, output_tokens=10, cost_usd=0,
        latency_ms=1, model="fake", provider="openai", finish_reason="stop"))
    selected, report = summary.prepare_narrative_evidence(claim, date(2024, 1, 2), items)
    assert report["status"] == "OK" and report["stance"] == label
    assert report["supporting_fact_indices"] == support
    assert report["contradicting_fact_indices"] == against
    assert [i.content for i in selected] == [i.content for i in items]
    assert all(i.stance.value == "NEUTRAL" for i in selected)  # Auditor owns passage judgments.


@pytest.mark.parametrize("changes", [
    {"stance": "NEUTRAL"},
    {"stance_reason": " "},
    {"supporting_fact_indices": [True]},
    {"supporting_fact_indices": [2]},
    {"supporting_fact_indices": [0, 0]},
    {"contradicting_fact_indices": [0]},
    {"missing_claim_parts": ["Missing assertion"]},
    {"stance": "CONTRADICTING", "stance_reason_code": "DIRECT_CONTRADICTION"},
    {"stance": "UNSUPPORTED", "stance_reason_code": "PARTIAL_SUPPORT"},
    {"stance": "UNSUPPORTED", "stance_reason_code": "DIRECT_CONTRADICTION"},
])
def test_unbacked_inconsistent_or_malformed_stance_is_rejected(changes):
    items = [item()]
    _, catalog = summary._source_spans(items)
    output = {**linked_selection(), **changes}
    resolved, _, errors = summary._resolve_spans(json.dumps(output), items, catalog)
    result, validation = summary._validate(resolved, items)
    assert not errors and result is None and validation


def test_no_candidates_abstains_without_calling_a_model(monkeypatch):
    def unexpected_call(**kwargs):
        pytest.fail("Empty retrieval must not call a model")
    monkeypatch.setattr(summary, "complete", unexpected_call)
    claim = SubClaim(sub_claim_id="sc1", text="A product launched globally.", company_ticker="AAPL",
                     sub_claim_type=SubClaimType.OTHER)
    selected, report = summary.prepare_narrative_evidence(claim, date(2024, 1, 2), [])
    assert selected == [] and report["stance"] == "UNSUPPORTED"
    assert report["stance_reason_code"] == "NO_RELEVANT_EVIDENCE"
    assert report["assessment_status"] == "ASSESSED" and report["llm"]["api_requests"] == 0


def test_no_relevant_selection_abstains_with_a_reason():
    output = {"facts": [], "summary": [], "stance": "UNSUPPORTED",
              "stance_reason": "The supplied passage discusses another product.",
              "stance_reason_code": "NO_RELEVANT_EVIDENCE", "supporting_fact_indices": [],
              "contradicting_fact_indices": [], "missing_claim_parts": ["The claimed launch."]}
    items = [item()]
    _, catalog = summary._source_spans(items)
    resolved, _, errors = summary._resolve_spans(json.dumps(output), items, catalog)
    result, checks = summary._validate(resolved, items)
    assert not errors and not checks and result["selected_evidence_ids"] == []


def test_provider_failure_is_error_not_unsupported(monkeypatch):
    claim = SubClaim(sub_claim_id="sc1", text="Product launch.", company_ticker="AAPL",
                     sub_claim_type=SubClaimType.OTHER)
    monkeypatch.setattr(summary, "get_settings", lambda: SimpleNamespace(llm_provider="openai", llm_model="fake"))
    def fail(**kwargs):
        raise RuntimeError("Provider unavailable")
    monkeypatch.setattr(summary, "complete", fail)
    items = [item()]
    selected, report = summary.prepare_narrative_evidence(claim, date(2024, 1, 2), items)
    assert selected == items and report["assessment_status"] == "ERROR"
    assert report["stance"] is None and report["stance_reason"] is None
