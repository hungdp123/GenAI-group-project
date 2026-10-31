"""The local testing page: form checks, citation payload, and one refused run."""

import json
import threading
from datetime import date
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from finsight.agents.orchestrator.graph import RunOutput
from finsight.agents.orchestrator.posts import which_post_question
from finsight.cli import main
from finsight.contracts.schemas import (
    AssessmentStatus,
    DecisionLog,
    EvidenceItem,
    FinalAssessment,
    SourceTier,
    Stance,
    SubClaim,
    SubClaimResult,
    SubClaimType,
    Verdict,
)
from finsight.ui.payload import answer_slot, build_view, cli_command, post_choices
from finsight.ui.server import STATIC, Handler, StageCapture, log_file, parse_run_body, run_check


def test_ui_help_exits_cleanly():
    with pytest.raises(SystemExit) as caught:
        main(["ui", "--help"])
    assert caught.value.code == 0


def test_page_keeps_steps_and_citations_visible():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "Steps" in html
    assert "Checks and citations" in script or "Citation" in script
    assert "Cut-off leak" in script


def test_clarification_answers_map_to_the_right_field():
    assert answer_slot("Which company is this claim about? FinSight covers: Nvidia.") == "company"
    assert answer_slot("Which influencer should this claim be checked against?") == "influencer"
    question = which_post_question([
        {"upload_date": "2025-07-31", "video_id": "demo-0", "title": "Company X"},
        {"upload_date": "2026-08-31", "video_id": "abcdefghijk", "title": "Nvidia update"},
    ])
    assert answer_slot(question) == "post"
    choices = post_choices(question)
    assert [row["video_id"] for row in choices] == ["demo-0", "abcdefghijk"]
    assert choices[1]["title"] == "Nvidia update"
    assert answer_slot("Your claim has nothing specific FinSight can check.") == "claim"


def test_cli_command_quotes_the_claim():
    command = cli_command({
        "claim": "Joseph Carlson said Nvidia's revenue grew.",
        "cutoff": "2026-08-31",
        "company": "NVDA",
        "influencer": "Joseph Carlson",
        "post": "demo-0",
    })
    assert command.startswith("finsight run 'Joseph Carlson said Nvidia'\"'\"'s revenue grew.' --stages")
    assert "--cutoff 2026-08-31" in command
    assert "--company NVDA" in command
    assert "--post demo-0" in command


def test_rejects_a_log_path_that_leaves_the_folder(tmp_path):
    with pytest.raises(ValueError):
        log_file(tmp_path, "../config")
    with pytest.raises(ValueError):
        log_file(tmp_path, "fs-ok/../../secrets")
    path = log_file(tmp_path, "fs-20261026-120000-abcdef")
    assert path.parent == tmp_path.resolve()


def test_run_body_is_checked():
    parsed = parse_run_body({"claim": "  Nvidia revenue grew. ", "company": "nvda"})
    assert parsed["company"] == "NVDA"
    with pytest.raises(ValueError):
        parse_run_body({"claim": "x", "cutoff": "31-08-2026"})


def test_view_keeps_the_source_label_and_citation():
    evidence = EvidenceItem(
        evidence_id="E1", sub_claim_id="sc1", stance=Stance.SUPPORTING,
        content="Revenue was $96.2 billion.", source_tier=SourceTier.TIER_1, doc_type="10-Q",
        company="NVIDIA", publication_date=date(2026, 8, 27), citation_span="Item 2, page 4",
        produced_by="financial", url="https://www.sec.gov/example",
    )
    result = SubClaimResult(
        sub_claim=SubClaim(sub_claim_id="sc1", text="Nvidia revenue was $96.2 billion.",
                           sub_claim_type=SubClaimType.HISTORICAL_GROWTH, company_ticker="NVDA"),
        verdict=Verdict.SUPPORTED, supporting=[evidence], rationale="number matched",
    )
    assessment = FinalAssessment(
        request_id="fs-test", status=AssessmentStatus.COMPLETED, claim="Nvidia revenue was $96.2 billion.",
        overall_verdict=Verdict.SUPPORTED, sub_claim_results=[result], as_of_date=date(2026, 8, 31),
        boundary_disclaimer="This evaluates the evidence for the stated claim.",
    )
    log = DecisionLog(request_id="fs-test", input=assessment.claim, cutoff_date=date(2026, 8, 31))
    view = build_view(RunOutput(assessment, log))
    item = view["assessment"]["sub_claim_results"][0]["supporting"][0]
    assert item["citation_span"] == "Item 2, page 4"
    assert "Tier 1" in item["source_label"]
    assert "10-Q" in item["source_label"]
    assert view["units"][0]["ids"] == ["sc1"]
    assert view["assessment"]["boundary_disclaimer"] == "This evaluates the evidence for the stated claim."


def test_refused_advice_run_records_the_gate():
    capture = StageCapture()
    view = run_check({
        "claim": "Should I put my savings into Nvidia?",
        "cutoff": None, "company": None, "influencer": None, "post": None,
    }, capture)
    assert view["assessment"]["status"] == "REFUSED"
    assert view["assessment"]["boundary_disclaimer"]
    nodes = [stage["node"] for stage in capture.snapshot()]
    assert "intake" in nodes
    assert "advice_gate_input" in nodes
    assert "gather_evidence" not in nodes


def test_meta_and_index_are_served():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/meta") as response:
            body = json.load(response)
        assert body["mode"]["provider"] == "mock"
        assert body["mode"]["paid"] is False
        assert any(row["ticker"] == "NVDA" for row in body["companies"])
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read().decode()
        assert "FinSight" in page
        assert "Run check" in page
    finally:
        server.shutdown()
        server.server_close()


def test_model_level_is_checked_and_routes_each_call(monkeypatch):
    from finsight.llm import client
    from finsight.ui.server import MODEL_LEVELS, level_model

    with pytest.raises(ValueError):
        parse_run_body({"claim": "Nvidia revenue grew.", "level": "ultra"})
    assert parse_run_body({"claim": "Nvidia revenue grew.", "level": "high"})["level"] == "high"
    assert level_model("high") is None  # mock mode keeps its own model
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    assert level_model("high") == MODEL_LEVELS["high"]["model"]
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "mock")
    with client.use_model("gpt-5.4-mini"):
        assert client.complete("hello").model == "gpt-5.4-mini"
    assert client.complete("hello").model == "mock-model"


def test_model_choice_reaches_orchestrator_worker_threads():
    from finsight.agents.orchestrator.nodes import _ModelPool
    from finsight.llm.client import model_override, use_model

    with use_model("gpt-5.4"), _ModelPool(max_workers=2) as pool:
        seen = list(pool.map(lambda _: model_override(), range(3)))
    assert seen == ["gpt-5.4"] * 3


def test_suggestions_are_stored_claims_and_never_raise():
    from finsight.ui.suggest import suggestions

    view = {"assessment": {"claim": "Kevin Paffrath said Tesla sales fell.", "sub_claim_results": []}}
    picked = suggestions(view, {"claim": view["assessment"]["claim"], "company": "TSLA"})
    assert len(picked) <= 3
    assert all(text.startswith("Kevin Paffrath said ") for text in picked)
    assert suggestions({}, {}) == []
