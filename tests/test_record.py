"""A consistency or accuracy question checks every stored post instead of asking which one."""

from finsight import execute
from finsight.agents.orchestrator.record import accuracy_sentence, consistency_sentence, directions
from finsight.contracts.schemas import Verdict

from stored_demo import demo_registry, store_fixture_claims


def test_consistency_does_not_ask_which_post():
    out = execute("Has Joseph Carlson been consistent about Company X?")
    text = out.assessment.summary or ""
    assert out.assessment.clarification_question is None
    assert "Which post" not in text
    assert "No measure flips" in text
    assert "does not compare price, rank" in text
    assert "demo-0" in text and "demo-5" in text
    assert out.assessment.overall_verdict is None


def test_accuracy_over_time_counts_labels():
    out = execute("What is the accuracy of Joseph Carlson's claims about Company X over time?")
    text = out.assessment.summary or ""
    assert "Accuracy:" in text
    assert "Which post" not in text
    assert out.assessment.overall_verdict is None


def test_a_named_post_stays_one_check():
    text = "Joseph Carlson said Company X is getting more profitable: its margins keep expanding."
    store_fixture_claims("demo-2", text)
    out = execute(
        "Has Joseph Carlson been consistent about Company X?",
        post_id="demo-2",
        registry=demo_registry(),
    )
    assert out.assessment.overall_verdict is not None
    assert "record across" not in (out.assessment.summary or "")


def test_a_direction_flip_is_not_consistent():
    sentence = consistency_sentence([
        ("a", directions("Revenue grew and margins are expanding.")),
        ("b", directions("Revenue fell.")),
    ])
    assert "not consistent on revenue" in sentence


def test_accuracy_does_not_treat_a_partial_as_wrong():
    text = accuracy_sentence([Verdict.SUPPORTED, Verdict.CONTRADICTED, Verdict.PARTIALLY_SUPPORTED])
    assert "1 of 2 were supported" in text
    assert "not counted as right or wrong" in text
    assert "not scored against what happened later" in text
    assert "does not match the filing" in text
