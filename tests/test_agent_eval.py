"""Per-agent evaluation harness (finsight eval --agents). Pinned to the stubs by conftest."""

import json

import pytest

from finsight.config import AgentRegistry
from finsight.contracts.schemas import AuditResult, Verdict
from finsight.eval.agent_eval import auditor, decomposer, financial, guardrails, narrative
from finsight.eval.agent_eval.runner import EVALUATORS, parse_names, run_agent_evals
from finsight.eval.run_eval import main


def _reg():
    return AgentRegistry.from_config()


def test_every_evaluator_runs_without_errors_on_the_stubs():
    for name, evaluate in EVALUATORS.items():
        report = evaluate(_reg())
        assert report.metrics["errors"] == 0, name
        assert report.cases, name


def test_stub_scores_reflect_what_the_stubs_actually_do():
    from finsight.agents.financial.stub import StubFinancialAgent
    from finsight.agents.narrative.demo.retriever_stub import StubRetriever
    from finsight.agents.narrative.demo.stub import StubNarrativeAgent

    stubs = AgentRegistry.from_config(
        financial=StubFinancialAgent(),
        narrative=StubNarrativeAgent(),
        retriever=StubRetriever(),
    )
    assert financial.evaluate(stubs).metrics["number correct (within 0.005)"] == 1.0
    assert narrative.evaluate_narrative(stubs).metrics["post-cut-off items returned"] == 0
    assert narrative.evaluate_retriever(stubs).metrics["recall@3"] == 1.0
    audit = auditor.evaluate(stubs).metrics
    assert audit["catch rate (broken evidence rejected)"] == 0.0  # the stub checks nothing...
    assert audit["false-alarm rate (correct evidence rejected)"] == 0.0  # ...but approves correct evidence
    assert guardrails.evaluate_tier_policy(stubs).metrics["illegal raises (verdict made stronger)"] == 0
    advice = guardrails.evaluate_advice_boundary(stubs).metrics
    assert advice["input (user's request): recall (advice caught)"] == 1.0
    assert advice["input (user's request): precision (flags that were right)"] == 1.0
    assert advice["output (FinSight's answer): recall (advice caught)"] == 1.0
    assert advice["output (FinSight's answer): precision (flags that were right)"] == 1.0
    assert decomposer.evaluate(_reg()).metrics["advice requests flagged (PERSONALIZED_ADVICE)"] == 1.0


class _PerfectAuditor:
    """Rejects anything that differs from the clean base cases: proves the harness can score 100%."""

    def audit(self, sub_claim, evidence):
        clean = {(b.sub_claim.text, tuple(e.model_dump_json() for e in b.evidence)) for b in auditor.BASES}
        ok = (sub_claim.text, tuple(e.model_dump_json() for e in evidence)) in clean
        return AuditResult(sub_claim_id=sub_claim.sub_claim_id, passed=ok,
                           verdict=Verdict.SUPPORTED if ok else Verdict.INSUFFICIENT_EVIDENCE,
                           failure_reasons=[] if ok else ["not the clean evidence"])


def test_auditor_harness_rewards_a_good_auditor():
    metrics = auditor.evaluate(AgentRegistry.from_config(auditor=_PerfectAuditor())).metrics
    assert metrics["catch rate (broken evidence rejected)"] == 1.0
    assert metrics["false-alarm rate (correct evidence rejected)"] == 0.0


class _BrokenFinancial:
    def gather(self, sub_claim, cutoff_date, feedback):
        raise RuntimeError("xbrl down")


def test_agent_exceptions_are_scored_not_crashing():
    report = financial.evaluate(AgentRegistry.from_config(financial=_BrokenFinancial()))
    assert report.metrics["errors"] == len(report.cases) and all(not c.passed for c in report.cases)


def test_runner_writes_summary_and_case_files(tmp_path):
    out = run_agent_evals(["auditor", "tier_policy"], tmp_path)
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## auditor (Person 3 (Amelia))" in summary and "catch rate" in summary
    cases = json.loads((out / "auditor.json").read_text(encoding="utf-8"))["cases"]
    assert any(c["case_id"].startswith("units / scale") for c in cases)


def test_cli_agents_flag(tmp_path, capsys):
    assert main(["--agents", "financial,retriever", "--out", str(tmp_path)]) == 0
    assert "## financial" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--agents", "astrologer"])
    assert parse_names("all") == list(EVALUATORS)
