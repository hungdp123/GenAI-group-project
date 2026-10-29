"""`finsight run --stages` and the on_stage hook of execute()."""

from finsight import execute
from finsight.agents.orchestrator.stages import StageReporter
from finsight.cli import main as cli_main
from finsight.contracts.schemas import AssessmentStatus

from demo import DEMO_CUTOFF
from stored_demo import DEMO_PROMPT, prepared_demo

DEMO_STAGES = ["intake", "advice_gate_input", "scope_check", "collect_posts", "extract_claims", "plan",
               "gather_evidence", "audit", "retry_or_accept", "apply_tier_policy", "assemble",
               "advice_gate_output", "finalize"]


def test_on_stage_sees_every_node_and_result_is_unchanged(monkeypatch, tmp_path):
    prepared_demo(monkeypatch, tmp_path, pin_impl=True)
    seen = []
    staged = execute(DEMO_PROMPT, DEMO_CUTOFF, on_stage=lambda node, update, rec: seen.append(node))
    plain = execute(DEMO_PROMPT, DEMO_CUTOFF)
    assert seen == DEMO_STAGES
    assert [r.verdict for r in staged.assessment.sub_claim_results] == \
           [r.verdict for r in plain.assessment.sub_claim_results]
    assert staged.decision_log.final_verdicts == plain.decision_log.final_verdicts


def test_retry_loop_shows_up_as_repeated_stages(monkeypatch, tmp_path):
    prepared_demo(monkeypatch, tmp_path, pin_impl=True)
    monkeypatch.setenv("FINSIGHT_STUB_FAIL_SUB_CLAIMS", "sc1")
    seen = []
    execute(DEMO_PROMPT, DEMO_CUTOFF, on_stage=lambda node, update, rec: seen.append(node))
    assert seen.count("gather_evidence") == 2 and seen.count("audit") == 2 and seen.count("retry_or_accept") == 2


def test_broken_observer_cannot_break_the_run(monkeypatch, tmp_path):
    prepared_demo(monkeypatch, tmp_path, pin_impl=True)

    def boom(node, update, rec):
        raise RuntimeError("observer bug")

    out = execute(DEMO_PROMPT, DEMO_CUTOFF, on_stage=boom)
    assert out.assessment.status is AssessmentStatus.COMPLETED


def test_audit_stage_lists_each_passage_stance():
    from finsight.contracts.schemas import AuditResult, Stance, Verdict

    reporter = StageReporter(emit=lambda _line: None)
    lines = reporter._audit({
        "audits": {
            "aws": AuditResult(
                sub_claim_id="aws", passed=False, verdict=Verdict.INSUFFICIENT_EVIDENCE,
                failure_reasons=["doc-1: only 22% of the sub-claim's key words appear"],
                evidence_stances={"doc-1": Stance.NEUTRAL},
            ),
        },
    })
    assert any("stance NEUTRAL" in line and "doc-1" in line and "22%" in line for line in lines)


def test_prepared_narrative_output_is_what_the_auditor_receives(monkeypatch):
    from datetime import date

    from finsight.agents.orchestrator.nodes import _prepared_narrative
    from finsight.contracts.schemas import EvidenceItem, SourceTier, Stance, SubClaim, SubClaimType

    claim = SubClaim(sub_claim_id="aws", text="AWS grew 28%.", company_ticker="AMZN",
                     sub_claim_type=SubClaimType.OTHER)
    kept = EvidenceItem(
        evidence_id="release", sub_claim_id="aws", stance=Stance.NEUTRAL, content="AWS sales increased 28%.",
        source_tier=SourceTier.TIER_2, doc_type="EX-99.1", company="Amazon", period="Q1 2026",
        publication_date=date(2026, 4, 29), produced_by="narrative",
    )
    dropped = kept.model_copy(update={"evidence_id": "other", "content": "Unrelated caption."})

    def prepare(sub_claim, cutoff, items, **_kwargs):
        assert items == [kept, dropped]
        return [kept], {
            "evidence_preparation_status": "COMPLETED",
            "stance": "SUPPORTED",
            "stance_reason": "The filing says AWS sales increased 28%.",
            "stance_reason_code": "ALL_MATERIAL_PARTS_SUPPORTED",
            "summary": [{"text": "AWS sales increased 28% year over year.", "evidence_ids": ["release"]}],
        }

    monkeypatch.setattr(
        "finsight.agents.narrative.rag.evidence_summary.prepare_narrative_evidence", prepare,
    )
    selected, report = _prepared_narrative(object(), claim, date(2026, 4, 30), [kept, dropped])
    assert selected == [kept]
    assert report["stance"] == "SUPPORTED"
    assert report["summary"] == ["AWS sales increased 28% year over year."]


def test_gather_stage_shows_the_narrative_stance_and_summary():
    reporter = StageReporter(emit=lambda _line: None)
    lines = reporter._gather_evidence({
        "narrative_reports": {
            "aws": {
                "stance": "SUPPORTED",
                "stance_reason_code": "ALL_MATERIAL_PARTS_SUPPORTED",
                "stance_reason": "The filing says AWS sales increased 28%.",
                "summary": ["AWS sales increased 28% year over year."],
                "status": "COMPLETED",
            },
        },
        "evidence": {},
    })
    text = "\n".join(lines)
    assert "narrative stance SUPPORTED (ALL_MATERIAL_PARTS_SUPPORTED)" in text
    assert "narrative reason The filing says AWS sales increased 28%." in text
    assert "summary AWS sales increased 28% year over year." in text


def test_reporter_lines_for_retry_round_only_show_retried_sub_claim(monkeypatch, tmp_path):
    prepared_demo(monkeypatch, tmp_path, pin_impl=True)
    monkeypatch.setenv("FINSIGHT_STUB_FAIL_SUB_CLAIMS", "sc1")
    lines: list[str] = []
    execute(DEMO_PROMPT, DEMO_CUTOFF, on_stage=StageReporter(emit=lines.append))
    text = "\n".join(lines)
    second_gather = text.split("[10] gather_evidence")[1].split("[11]")[0]
    assert "sc1" in second_gather and "sc2" not in second_gather and "sc4" not in second_gather
    assert "RETRY with collectors financial;" in text
    assert "1 retries" in text


def test_cli_stages_demo(capsys, monkeypatch, tmp_path):
    prepared_demo(monkeypatch, tmp_path, pin_impl=True)
    assert cli_main(["run", DEMO_PROMPT, "--cutoff", "2025-07-31", "--stages", "--no-evidence"]) == 0
    out = capsys.readouterr().out
    for i, node in enumerate(DEMO_STAGES, start=1):
        assert f"[{i:>2}] {node}" in out
    assert "sc3  no collector -> INSUFFICIENT_EVIDENCE" in out
    assert "Result:" in out and "Overall, this claim is partly supported" in out


def test_cli_stages_stops_at_refusal_and_clarification(capsys):
    cli_main(["run", "Should I put my savings into Company X?", "--stages"])
    refused = capsys.readouterr().out
    assert "STOP -> REFUSED" in refused and "extract_claims" not in refused

    cli_main(["run", "Its revenue growth is just getting started.", "--stages"])
    asked = capsys.readouterr().out
    assert "needs clarification: Which influencer" in asked and "[ 4] clarification" in asked


def test_cli_company_answers_clarification(capsys, monkeypatch, tmp_path):
    sentence = "Its revenue growth is just getting started."
    prepared_demo(monkeypatch, tmp_path, transcript=sentence, pin_impl=True)
    cli_main(["run", sentence, "--cutoff", "2025-07-31", "--company", "CMPX", "--influencer", "Joseph Carlson"])
    assert "Overall, this claim is" in capsys.readouterr().out
