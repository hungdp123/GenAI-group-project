import csv

import pytest

from finsight.contracts.schemas import Verdict
from finsight.eval.metrics import GoldClaim, PredictedClaim, align_claims, percentile
from finsight.eval.run_eval import load_dataset, main, run_eval


def test_datasets_load():
    dev, heldout = load_dataset("dev"), load_dataset("heldout")
    assert 3 <= len(dev) <= 5 and 3 <= len(heldout) <= 5
    assert {i.category for i in dev} >= {"normal", "adversarial_advice", "out_of_scope", "injection", "temporal"}


def test_heldout_requires_final():
    with pytest.raises(SystemExit):
        main(["--split", "heldout"])


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5 and percentile([], 95) == 0.0


def test_align_claims():
    gold = [GoldClaim(text="revenue growth is strong", gold_verdict=Verdict.SUPPORTED),
            GoldClaim(text="biggest winner", gold_verdict=Verdict.INSUFFICIENT_EVIDENCE)]
    pred = [PredictedClaim(text="X is the biggest winner", verdict=Verdict.SUPPORTED),
            PredictedClaim(text="X recent revenue growth is strong", verdict=Verdict.SUPPORTED)]
    aligned = align_claims(gold, pred)
    assert aligned[0][1].text.startswith("X recent") and aligned[1][1].text.endswith("winner")


def test_full_dev_eval(tmp_path):
    out = run_eval(["direct", "rag", "finsight"], "dev", tmp_path)
    summary = (out / "summary.md").read_text()
    assert "| Metric | direct | rag | finsight |" in summary
    rows = list(csv.DictReader((out / "claims.csv").open()))
    finsight_rows = [r for r in rows if r["system"] == "finsight"]
    assert finsight_rows and all(r["predicted_verdict"] != "MISSING" for r in finsight_rows
                                 if r["item_id"] == "dev-001")
    assert len(list((out / "raw" / "finsight").glob("*.json"))) == 5
    assert list((out / "logs").glob("*.json"))
    assert "| Task success rate | " in summary
