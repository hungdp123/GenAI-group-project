"""Record items (consistency/accuracy over every stored post) and two-turn "which post?" items."""

import json
import os
from pathlib import Path

from finsight.contracts.schemas import Verdict
from finsight.eval.metrics import GoldSubClaim
from finsight.eval.run_eval import EvalItem, run_finsight_item

from stored_demo import store_fixture_claims

MARGINS = "Company X is getting more profitable: its margins keep expanding."


def _two_posts() -> None:
    """Two Joseph Carlson posts about Company X with the same sentence and different dates."""
    path = Path(os.environ["FINSIGHT_POSTS_DIR"]) / "posts.jsonl"
    rows = [
        {"influencer": "joseph_carlson", "video_id": vid, "title": title, "description": "Company X",
         "upload_date": when, "transcript": MARGINS, "channel": "Joseph Carlson"}
        for vid, title, when in (("postAAAAAAA", "Company X in May", "2025-05-30"),
                                 ("postBBBBBBB", "Company X in July", "2025-07-31"))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    store_fixture_claims("postAAAAAAA", MARGINS)
    claims = Path(os.environ["FINSIGHT_POSTS_DIR"]).parent / "claims" / "claims.jsonl"
    first = claims.read_text(encoding="utf-8")
    store_fixture_claims("postBBBBBBB", MARGINS)
    claims.write_text(first + claims.read_text(encoding="utf-8"), encoding="utf-8")


RECORD_CLAIM = "Has Joseph Carlson been consistent about Company X's margins?"
POSTS = (("postAAAAAAA", "2025-05-30"), ("postBBBBBBB", "2025-07-31"))


def _record_item(verdicts: dict[str, Verdict | None]) -> EvalItem:
    return EvalItem(
        item_id="rec-1", split="dev", category="record", claim=RECORD_CLAIM,
        gold_sub_claims=[
            GoldSubClaim(text=f"{when} post: Company X margins keep expanding", gold_verdict=verdicts[vid],
                         post_id=vid)
            for vid, when in POSTS
        ],
    )


def _stub_verdicts() -> dict[str, Verdict | None]:
    """What the stub agents give each post, so the gold can match or miss on purpose."""
    from finsight import execute

    record = execute(RECORD_CLAIM).record
    return {post.video_id: post.verdict for post in record.posts}


def test_record_item_scores_the_four_checks(tmp_path):
    _two_posts()
    record, raw = run_finsight_item(_record_item(_stub_verdicts()), tmp_path)
    data = json.loads(raw)
    assert [p["video_id"] for p in data["posts"]] == ["postAAAAAAA", "postBBBBBBB"]
    assert [p["cutoff_date"] for p in data["posts"]] == ["2025-05-30", "2025-07-31"]
    assert data["assessment"]["overall_verdict"] is None
    assert record.failed_checks == [] and record.task_success
    assert record.predicted_sub_claims[0].text.startswith("2025-05-30 post: ")


def test_record_item_fails_when_the_count_differs_from_gold(tmp_path):
    _two_posts()
    flipped = {vid: Verdict.CONTRADICTED if v is Verdict.SUPPORTED else Verdict.SUPPORTED
               for vid, v in _stub_verdicts().items()}
    record, _ = run_finsight_item(_record_item(flipped), tmp_path)
    assert not record.task_success
    assert any("gold is" in check for check in record.failed_checks)


def test_record_item_fails_when_a_gold_post_is_not_checked(tmp_path):
    _two_posts()
    item = _record_item(_stub_verdicts())
    item = item.model_copy(update={"gold_sub_claims": [
        *item.gold_sub_claims, GoldSubClaim(text="other post", gold_verdict=Verdict.SUPPORTED, post_id="missingpost")]})
    record, _ = run_finsight_item(item, tmp_path)
    assert any("missing: missingpost" in check for check in record.failed_checks)


def _which_post_item(answer: str) -> EvalItem:
    return EvalItem(
        item_id="wp-1", split="dev", category="ask_which_post",
        claim="Joseph Carlson said Company X's margins keep expanding.",
        expect_clarification=True, expected_outcome="PROCESS_AFTER_CLARIFICATION",
        posts=[{"video_id": "postBBBBBBB"}],
        clarification_turns=[{"system": "Two posts fit: 2025-05-30, postAAAAAAA; 2025-07-31, postBBBBBBB. "
                                        "Which post should I check?", "user": answer}],
        gold_sub_claims=[GoldSubClaim(text="Company X margins keep expanding", gold_verdict=Verdict.SUPPORTED)],
    )


def test_which_post_item_runs_two_turns(tmp_path):
    _two_posts()
    record, raw = run_finsight_item(_which_post_item("postBBBBBBB"), tmp_path)
    data = json.loads(raw)
    assert data["turn_1"]["status"] == "NEEDS_CLARIFICATION"
    assert "postAAAAAAA" in data["turn_1"]["clarification_question"]
    assert data["turn_2"]["status"] == "COMPLETED"
    assert data["turn_2"]["as_of_date"] == "2025-07-31"
    assert record.clarified and record.failed_checks == [] and record.task_success


def test_which_post_item_fails_when_turn_two_is_not_a_stored_post(tmp_path):
    _two_posts()
    record, _ = run_finsight_item(_which_post_item("not-a-post"), tmp_path)
    assert not record.task_success
    assert any("turn 2" in check for check in record.failed_checks)


def test_validator_accepts_a_which_post_item(tmp_path, monkeypatch):
    from finsight.eval import run_eval

    dev = (run_eval.DATASET_DIR / "claims_dev.jsonl").read_text(encoding="utf-8").rstrip("\n")
    extra = _which_post_item("postBBBBBBB").model_dump_json(exclude_none=True)
    (tmp_path / "claims_dev.jsonl").write_text(dev + "\n" + extra + "\n", encoding="utf-8")
    (tmp_path / "claims_heldout.jsonl").write_text(
        (run_eval.DATASET_DIR / "claims_heldout.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(run_eval, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "HELDOUT_LOCK", tmp_path / "claims_heldout.lock")
    assert run_eval.validate_datasets() == []
