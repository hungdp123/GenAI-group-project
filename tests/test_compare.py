"""The comparison scorer: made-up numbers, overclaiming, the paired test, and a full mock run of every system."""

import json

from finsight.eval.compare import figures, grounded, made_up, mcnemar, summarize


def test_figures_and_grounding():
    assert figures("Revenue was $12,250m, up 16.2%, about 9.5 billion") == [
        ("usd", 12_250e6), ("pct", 16.2), ("usd", 9.5e9)]
    sources = figures("Revenue $12,250m in Q1 2026 vs $10,543m in Q1 2025")
    assert grounded(("pct", 16.2), sources)            # a percent change of two source amounts
    assert grounded(("usd", 12.25e9), sources)
    assert not grounded(("pct", 24.0), sources)        # not in, not calculable from, the sources
    assert made_up("Netflix revenue grew 16.2% to $12.25 billion; margins hit 31%.",
                   "Revenue $12,250m vs $10,543m") == (1, 3)


def test_bare_numbers_and_years_are_not_figures():
    assert figures("In 2026 the company had 3 segments") == []


def test_mcnemar_is_exact_and_two_sided():
    assert mcnemar(0, 0) == 1.0
    assert round(mcnemar(14, 3), 3) == 0.013
    assert mcnemar(5, 5) == 1.0


def test_overclaim_counts_confident_answers_on_unknowable_facts():
    row = {"item_id": "x", "golds": ["INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE", "SUPPORTED"],
           "preds": ["SUPPORTED", "INSUFFICIENT_EVIDENCE", "SUPPORTED"], "routes": ["financial"] * 3,
           "gold_overall": None, "overall": None, "made_up": 0, "figures": 0, "dates_after": 0, "leaks": 0}
    summary = summarize({"systems": ["llm"], "rows": {"llm": [row]}})["llm"]
    assert summary["overclaim_rate"] == 0.5 and summary["fact_accuracy"] == 2 / 3


def test_every_system_runs_and_the_comparison_is_written(tmp_path):
    from finsight.eval.run_eval import SYSTEMS, run_eval

    out = run_eval(list(SYSTEMS), "dev", tmp_path)
    assert {p.name for p in (out / "raw").iterdir()} == set(SYSTEMS)
    report = (out / "comparison.md").read_text(encoding="utf-8")
    for name in ("Plain LLM", "LLM + filings", "Basic RAG", "RAG + quote", "FinSight", "Overclaim rate",
                 "Made-up number rate", "McNemar"):
        assert name in report
    data = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert set(data["summary"]) == set(SYSTEMS)


def test_only_route_keeps_facts_on_that_route_and_whole_route_headlines():
    from finsight.eval.compare import only_route

    rows = [
        {"item_id": "a", "golds": ["SUPPORTED", "INSUFFICIENT_EVIDENCE"], "preds": ["SUPPORTED", "SUPPORTED"],
         "routes": ["financial", "narrative"], "gold_overall": "PARTIALLY_SUPPORTED", "overall": "SUPPORTED",
         "made_up": 0, "figures": 0, "dates_after": 0, "leaks": 0},
        {"item_id": "b", "golds": ["CONTRADICTED"], "preds": ["CONTRADICTED"], "routes": ["financial"],
         "gold_overall": "CONTRADICTED", "overall": "CONTRADICTED", "made_up": 0, "figures": 0, "dates_after": 0,
         "leaks": 0},
        {"item_id": "c", "golds": ["SUPPORTED"], "preds": ["MISSING"], "routes": ["narrative"],
         "gold_overall": "SUPPORTED", "overall": None, "made_up": 0, "figures": 0, "dates_after": 0, "leaks": 0},
    ]
    kept = only_route({"systems": ["llm"], "rows": {"llm": rows}}, "financial")["rows"]["llm"]
    assert [r["item_id"] for r in kept] == ["a", "b"]
    assert kept[0]["golds"] == ["SUPPORTED"] and kept[0]["gold_overall"] is None  # mixed item: no headline
    assert kept[1]["gold_overall"] == "CONTRADICTED"


def test_the_cli_takes_a_list_of_systems_and_a_route(tmp_path):
    from finsight.eval.run_eval import main

    out = tmp_path / "results"
    assert main(["--system", "finsight,llm", "--split", "dev", "--route", "financial", "--out", str(out)]) == 0
    run = next(out.iterdir())
    assert {p.name for p in (run / "raw").iterdir()} == {"finsight", "llm"}
    assert "financial facts only" in (run / "comparison.md").read_text(encoding="utf-8")


def test_a_paid_run_stops_before_it_starts_when_the_llm_call_fails(monkeypatch, tmp_path, capsys):
    from finsight.eval.run_eval import llm_preflight, main

    assert llm_preflight() is None  # mock mode: nothing to check
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("FINSIGHT_LLM_MODEL", "gpt-5.4-nano")
    assert llm_preflight()  # no key in the test environment: the error comes back as text
    assert main(["--system", "llm", "--split", "dev", "--out", str(tmp_path / "r")]) == 2
    assert "LLM call failed before the run started" in capsys.readouterr().out
    assert not (tmp_path / "r").exists()


def test_baseline_pairing_prefers_the_fact_over_the_attribution_line():
    from types import SimpleNamespace

    from finsight.eval.compare import _baseline_verdicts

    fact = SimpleNamespace(asked=True, gold_verdict="SUPPORTED", quote_span="Revenues of 82.9 billion",
                           user_span="Microsoft's revenue was $82.9 billion last quarter")
    item = SimpleNamespace(gold_alignment=[fact])
    raw = {"sub_claims": [
        {"text": "Mark Roussin said Microsoft's revenue was $82.9 billion last quarter",
         "verdict": "INSUFFICIENT_EVIDENCE"},
        {"text": "Microsoft revenue was $82.9 billion last quarter", "verdict": "SUPPORTED"},
    ]}
    assert _baseline_verdicts(item, raw) == ["SUPPORTED"]
