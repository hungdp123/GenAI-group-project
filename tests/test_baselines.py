from finsight.baselines.basic_rag import run_basic_rag
from finsight.baselines.common import parse_output
from finsight.baselines.direct_llm import run_direct_llm
from finsight.contracts.schemas import Verdict
from finsight.llm import client

from demo import DEMO_CUTOFF, DEMO_THESIS


def test_direct_llm_baseline_logs_usage():
    r = run_direct_llm(DEMO_THESIS, DEMO_CUTOFF)
    assert r.error is None and not r.refused and r.claims
    [call] = r.agent_calls
    assert call.agent == "llm" and call.input_tokens > 0 and call.success


def test_direct_llm_prompt_contains_xbrl_and_wraps_thesis():
    seen = {}
    client.set_mock_responder(lambda system, prompt: seen.update(system=system, prompt=prompt) or "OVERALL: Supported")
    try:
        run_direct_llm(DEMO_THESIS, DEMO_CUTOFF)
    finally:
        client.set_mock_responder(None)
    assert "Revenues Q2 FY2025 = 4560.0" in seen["prompt"]
    assert '<data name="thesis">' in seen["prompt"] and "finfluencer" not in seen["system"]


def test_basic_rag_baseline_uses_retriever():
    r = run_basic_rag(DEMO_THESIS, DEMO_CUTOFF)
    assert [c.agent for c in r.agent_calls] == ["retriever", "llm"]
    assert r.agent_calls[0].tool_calls == 1


def test_baseline_refusal():
    assert run_direct_llm("Should I put my savings into Company X?", DEMO_CUTOFF).refused


def test_parser_structured_and_free_form():
    refused, _, claims, overall = parse_output(
        "CLAIM: growth is strong | VERDICT: Supported\nCLAIM: biggest winner | VERDICT: Insufficient Evidence\n"
        "OVERALL: Partially Supported")
    assert not refused and overall is Verdict.PARTIALLY_SUPPORTED
    assert [c.verdict for c in claims] == [Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE]
    _, _, free, _ = parse_output("The growth claim is partially supported by the 10-Q.")
    assert free[0].verdict is Verdict.PARTIALLY_SUPPORTED
