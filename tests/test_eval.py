import csv
import json

import pytest

from finsight.contracts.schemas import Verdict
from finsight.eval.metrics import GoldSubClaim, PredictedSubClaim, align_sub_claims, percentile
from finsight.eval.run_eval import load_dataset, main, run_eval, validate_datasets


def test_datasets_load_and_validate():
    dev = load_dataset("dev")
    assert load_dataset("heldout")
    assert {i.category for i in dev} >= {"normal", "adversarial_advice", "out_of_scope", "injection", "temporal"}
    assert validate_datasets() == []


def test_validator_catches_leak_and_bad_items(tmp_path, monkeypatch):
    from finsight.eval import run_eval

    dev = (run_eval.DATASET_DIR / "claims_dev.jsonl").read_text(encoding="utf-8")
    leaked = dev.splitlines()[0].replace('"dev-001"', '"ho-x"').replace('"split": "dev"', '"split": "heldout"')
    (tmp_path / "claims_dev.jsonl").write_text(dev, encoding="utf-8")
    (tmp_path / "claims_heldout.jsonl").write_text(leaked, encoding="utf-8")
    monkeypatch.setattr(run_eval, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "HELDOUT_LOCK", tmp_path / "claims_heldout.lock")
    assert any("leak" in p for p in validate_datasets())


def test_freeze_blocks_edits_and_unfrozen_final(tmp_path, monkeypatch):
    from finsight.eval import run_eval

    for split in ("dev", "heldout"):
        (tmp_path / f"claims_{split}.jsonl").write_text(
            (run_eval.DATASET_DIR / f"claims_{split}.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(run_eval, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "HELDOUT_LOCK", tmp_path / "claims_heldout.lock")
    with pytest.raises(SystemExit):  # not frozen yet
        main(["--split", "heldout", "--final", "--system", "finsight"])
    assert main(["--freeze"]) == 0
    (tmp_path / "claims_heldout.jsonl").write_text("tampered", encoding="utf-8")
    assert any("changed after it was frozen" in p for p in validate_datasets())


def test_heldout_requires_final():
    with pytest.raises(SystemExit):
        main(["--split", "heldout"])


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 95) == 0.0


def test_align_sub_claims():
    gold = [GoldSubClaim(text="revenue growth is strong", gold_verdict=Verdict.SUPPORTED),
            GoldSubClaim(text="biggest winner", gold_verdict=Verdict.INSUFFICIENT_EVIDENCE)]
    pred = [PredictedSubClaim(text="X is the biggest winner", verdict=Verdict.SUPPORTED),
            PredictedSubClaim(text="X recent revenue growth is strong", verdict=Verdict.SUPPORTED)]
    aligned = align_sub_claims(gold, pred)
    assert aligned[0][1].text.startswith("X recent") and aligned[1][1].text.endswith("winner")


def test_full_dev_eval(tmp_path):
    out = run_eval(["direct", "rag", "finsight"], "dev", tmp_path)
    summary = (out / "summary.md").read_text()
    assert "| Metric | direct | rag | finsight |" in summary
    rows = list(csv.DictReader((out / "sub_claims.csv").open()))
    finsight_rows = [r for r in rows if r["system"] == "finsight"]
    assert finsight_rows
    raw = json.loads((out / "raw" / "finsight" / "dev-001.json").read_text(encoding="utf-8"))
    assert raw["status"] == "NEEDS_CLARIFICATION"
    assert "stored claims" in raw["clarification_question"]
    assert len(list((out / "raw" / "finsight").glob("*.json"))) == len(load_dataset("dev"))
    assert list((out / "logs").glob("*.json"))
    assert "| Task success rate | " in summary


def test_final_run_requires_course_model(monkeypatch):
    monkeypatch.setenv("FINSIGHT_LLM_PROVIDER", "openai")  # default model is the cheap testing one
    with pytest.raises(SystemExit):
        main(["--split", "heldout", "--final", "--system", "direct"])


def test_cli_forwards_eval_flags():
    from finsight.cli import main as cli_main

    with pytest.raises(SystemExit):  # heldout guard proves --split reached run_eval
        cli_main(["eval", "--system", "direct", "--split", "heldout"])
