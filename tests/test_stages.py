"""`finsight run --stages` and the on_stage hook of execute()."""

from finsight import execute
from finsight.agents.orchestrator.stages import StageReporter
from finsight.cli import main as cli_main
from finsight.contracts.schemas import AssessmentStatus

from demo import DEMO_CLAIM, DEMO_CUTOFF

DEMO_STAGES = ["intake", "advice_gate_input", "scope_check", "decompose", "plan", "gather_evidence", "audit",
               "retry_or_accept", "apply_tier_policy", "assemble", "advice_gate_output", "finalize"]


def test_on_stage_sees_every_node_and_result_is_unchanged():
    seen = []
    staged = execute(DEMO_CLAIM, DEMO_CUTOFF, on_stage=lambda node, update, rec: seen.append(node))
    plain = execute(DEMO_CLAIM, DEMO_CUTOFF)
    assert seen == DEMO_STAGES
    assert [r.verdict for r in staged.assessment.sub_claim_results] == \
           [r.verdict for r in plain.assessment.sub_claim_results]
    assert staged.decision_log.final_verdicts == plain.decision_log.final_verdicts


def test_retry_loop_shows_up_as_repeated_stages(monkeypatch):
    monkeypatch.setenv("FINSIGHT_STUB_FAIL_SUB_CLAIMS", "sc1")
    seen = []
    execute(DEMO_CLAIM, DEMO_CUTOFF, on_stage=lambda node, update, rec: seen.append(node))
    assert seen.count("gather_evidence") == 2 and seen.count("audit") == 2 and seen.count("retry_or_accept") == 2


def test_broken_observer_cannot_break_the_run():
    def boom(node, update, rec):
        raise RuntimeError("observer bug")

    out = execute(DEMO_CLAIM, DEMO_CUTOFF, on_stage=boom)
    assert out.assessment.status is AssessmentStatus.COMPLETED


def test_reporter_lines_for_retry_round_only_show_retried_sub_claim(monkeypatch):
    monkeypatch.setenv("FINSIGHT_STUB_FAIL_SUB_CLAIMS", "sc1")
    lines: list[str] = []
    execute(DEMO_CLAIM, DEMO_CUTOFF, on_stage=StageReporter(emit=lines.append))
    text = "\n".join(lines)
    second_gather = text.split("[ 9] gather_evidence")[1].split("[10]")[0]
    assert "sc1" in second_gather and "sc2" not in second_gather and "sc4" not in second_gather
    assert "RETRY with collectors financial+narrative" in text
    assert "1 retries" in text


def test_cli_stages_demo(capsys):
    assert cli_main(["run", DEMO_CLAIM, "--cutoff", "2025-07-31", "--stages", "--no-evidence"]) == 0
    out = capsys.readouterr().out
    for i, node in enumerate(DEMO_STAGES, start=1):
        assert f"[{i:>2}] {node}" in out
    assert "sc3  no collector -> INSUFFICIENT_EVIDENCE" in out
    assert "Result:" in out and "Overall, this claim is partly supported" in out


def test_cli_stages_stops_at_refusal_and_clarification(capsys):
    cli_main(["run", "Should I put my savings into Company X?", "--stages"])
    refused = capsys.readouterr().out
    assert "STOP -> REFUSED" in refused and "decompose" not in refused

    cli_main(["run", "Its revenue growth is just getting started.", "--stages"])
    asked = capsys.readouterr().out
    assert "needs clarification: Which company" in asked and "[ 4] clarification" in asked


def test_cli_company_answers_clarification(capsys):
    cli_main(["run", "Its revenue growth is just getting started.", "--cutoff", "2025-07-31", "--company", "CMPX"])
    assert "Overall, this claim is" in capsys.readouterr().out
