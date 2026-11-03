from finsight.baselines.basic_rag import run_basic_rag
from finsight.baselines.common import parse_output
from finsight.baselines.direct_llm import run_direct_llm
from finsight.contracts.schemas import Verdict
from finsight.llm import client

from demo import DEMO_CUTOFF, DEMO_CLAIM


def test_direct_llm_baseline_logs_usage():
    r = run_direct_llm(DEMO_CLAIM, DEMO_CUTOFF)
    assert r.error is None and not r.refused and r.sub_claims
    [call] = r.agent_calls
    assert call.agent == "llm" and call.input_tokens > 0 and call.success


def test_direct_llm_prompt_contains_xbrl_and_wraps_claim():
    seen = {}
    client.set_mock_responder(lambda system, prompt: seen.update(system=system, prompt=prompt) or "OVERALL: Supported")
    try:
        run_direct_llm(DEMO_CLAIM, DEMO_CUTOFF)
    finally:
        client.set_mock_responder(None)
    assert "Revenues Q2 FY2025 = 4560.0" in seen["prompt"]
    assert '<data name="claim">' in seen["prompt"] and "finfluencer" not in seen["system"]


def test_basic_rag_baseline_uses_retriever():
    r = run_basic_rag(DEMO_CLAIM, DEMO_CUTOFF)
    assert [c.agent for c in r.agent_calls] == ["retriever", "llm"]
    assert r.agent_calls[0].tool_calls == 1


def test_basic_rag_searches_the_company_named_in_the_claim(monkeypatch):
    from finsight.baselines import basic_rag as rag

    seen = {}

    class FakeRetriever:
        _embedding_backend = None

        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def retrieve(self, query, top_k, cutoff):
            seen["query"] = query
            seen["top_k"] = top_k
            return []

    rag._retrievers.clear()
    monkeypatch.setattr(rag, "CorpusRetriever", FakeRetriever)
    claim = "Joseph Carlson said Netflix generates $12 billion in free cash flow."
    quote = "It also generates $12 billion in free cash flow"
    try:
        run_basic_rag(claim, DEMO_CUTOFF, quote=quote)
    finally:
        rag._retrievers.clear()
        rag._embedder = None
    assert seen["init"]["company_ticker"] == "NFLX"
    assert seen["init"]["backend"] == "hybrid"
    assert str(seen["init"]["corpus_root"]).endswith("chunking_strategies/C2/full_v1")
    assert seen["query"] == quote
    assert seen["top_k"] == 5


def test_basic_rag_without_a_company_does_not_open_the_nvidia_index(monkeypatch):
    from finsight.baselines import basic_rag as rag

    def refuse(**kwargs):
        raise AssertionError(kwargs)

    monkeypatch.setattr(rag, "CorpusRetriever", refuse)
    r = run_basic_rag("Its revenue growth is just getting started.", DEMO_CUTOFF)
    assert r.error is None
    assert r.agent_calls[0].tool_calls == 1


def test_baseline_refusal():
    assert run_direct_llm("Should I put my savings into Company X?", DEMO_CUTOFF).refused


def test_parser_structured_and_free_form():
    refused, _, sub_claims, overall = parse_output(
        "SUB-CLAIM: growth is strong | VERDICT: Supported\nCLAIM: biggest winner | VERDICT: Insufficient Evidence\n"
        "OVERALL: Partially Supported")
    assert not refused and overall is Verdict.PARTIALLY_SUPPORTED
    assert [c.verdict for c in sub_claims] == [Verdict.SUPPORTED, Verdict.INSUFFICIENT_EVIDENCE]
    _, _, free, _ = parse_output("The growth sub-claim is partially supported by the 10-Q.")
    assert free[0].verdict is Verdict.PARTIALLY_SUPPORTED


def test_baseline_clarification_is_parsed():
    client.set_mock_responder(lambda system, prompt: "CLARIFY: Which company is this claim about?")
    try:
        r = run_direct_llm("Its growth is just getting started.", DEMO_CUTOFF)
    finally:
        client.set_mock_responder(None)
    assert r.clarification_question == "Which company is this claim about?"
    assert not r.refused and r.sub_claims == []


def test_parser_maps_mixed_to_partial():
    _, _, _, overall = parse_output("SUB-CLAIM: a | VERDICT: Supported\nOVERALL: Mixed")
    assert overall is Verdict.PARTIALLY_SUPPORTED
