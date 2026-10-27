"""End-to-end evaluation runner. Owner: Person 5 (Hung).

    python -m finsight.eval.run_eval --system all --split dev
    python -m finsight.eval.run_eval --system all --split heldout --final   # frozen split: final run only
    python -m finsight.eval.run_eval --validate                            # check both dataset files
    python -m finsight.eval.run_eval --freeze                              # lock the heldout file (once)

Outputs results/{timestamp}/: raw/{system}/{item_id}.json, logs/ (FinSight decision logs),
records.jsonl, sub_claims.csv and summary.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from datetime import date, datetime
from importlib import resources
from pathlib import Path

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finsight.agents.orchestrator import posts
from finsight.agents.orchestrator.aggregation import overall_verdict
from finsight.agents.orchestrator.graph import RunOutput, execute
from finsight.agents.orchestrator.record import RecordResult
from finsight.baselines.basic_rag import run_basic_rag
from finsight.baselines.common import BaselineResult
from finsight.baselines.direct_llm import run_direct_llm
from finsight.config import FINAL_REPORT_MODEL, get_settings
from finsight.contracts.schemas import AssessmentStatus, DecisionLog, FinalAssessment, Verdict
from finsight.eval.metrics import (
    GoldSubClaim,
    PredictedSubClaim,
    RunRecord,
    summarize,
    summary_markdown,
    task_success,
    write_sub_claims_csv,
)

SYSTEMS = ("direct", "rag", "finsight")
DATASET_DIR = Path(str(resources.files("finsight.eval.datasets")))
HELDOUT_LOCK = DATASET_DIR / "claims_heldout.lock"


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    claim: str
    cutoff_date: date | None = None
    split: Literal["dev", "heldout"]
    gold_sub_claims: list[GoldSubClaim] = Field(default_factory=list)
    expect_refusal: bool = False
    expect_clarification: bool = False  # unclear or missing info: the system should ask a question
    category: Literal["normal", "adversarial_advice", "out_of_scope", "injection", "temporal",
                      "clarification", "comparison", "misremembered", "missing_name_answered",
                      "missing_name_refused", "cutoff_trap", "no_stored_post", "ask_which_post",
                      "record", "cutoff", "no_cutoff", "partly_found"] = "normal"
    notes: str | None = None
    # Optional fields from the influencer-post items. The runner reads `posts` and `clarification_turns`
    # (ask_which_post); the rest document the label for the labeler.
    cutoff_resolved_to: dict[str, Any] | None = None  # what a relative cut-off resolves to: rule, date, basis
    influencer: str | None = None
    influencer_after_clarification: str | None = None
    clarification_turns: list[dict[str, str]] = Field(default_factory=list)  # [{"system": "...", "user": "..."}]
    expected_outcome: str | None = None  # e.g. PROCESS_AFTER_CLARIFICATION
    posts: list[Any] = Field(default_factory=list)  # the influencer posts the claim is based on
    user_matches_post: bool | None = None  # False when the user misremembers the post
    mismatch_note: str | None = None
    gold_source: str | None = None


def dataset_path(split: str) -> Path:
    return DATASET_DIR / f"claims_{split}.jsonl"


def load_dataset(split: str, path: Path | None = None) -> list[EvalItem]:
    text = (path or dataset_path(split)).read_text("utf-8")
    return [EvalItem.model_validate_json(line) for line in text.splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    """Line-ending-insensitive hash, so a Windows checkout (CRLF) matches the committed file (LF)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_datasets() -> list[str]:
    """Problems with the dev/heldout files (empty list = OK)."""
    problems: list[str] = []
    items: dict[str, list[EvalItem]] = {}
    for split in ("dev", "heldout"):
        try:
            items[split] = load_dataset(split)
        except (ValidationError, ValueError) as exc:
            problems.append(f"{split}: invalid JSONL ({exc})")
            items[split] = []
    ids = [i.item_id for split_items in items.values() for i in split_items]
    problems += [f"duplicate item_id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    dev_claims = {i.claim.strip().lower() for i in items["dev"]}
    problems += [f"{i.item_id}: claim also appears in dev (leak)" for i in items["heldout"]
                 if i.claim.strip().lower() in dev_claims]
    for split, split_items in items.items():
        for i in split_items:
            if i.split != split:
                problems.append(f"{i.item_id}: split={i.split} but stored in {split} file")
            stops = i.expect_refusal or i.expect_clarification
            if i.expect_refusal and i.expect_clarification:
                problems.append(f"{i.item_id}: expect_refusal and expect_clarification cannot both be true")
            # A clarification item that is analyzed after the user answers may carry gold sub-claims.
            processed_later = i.expected_outcome == "PROCESS_AFTER_CLARIFICATION"
            if stops and i.gold_sub_claims and not processed_later:
                problems.append(f"{i.item_id}: refusal/clarification items must not have gold_sub_claims")
            if not stops and not i.gold_sub_claims:
                problems.append(f"{i.item_id}: items that are analyzed need gold_sub_claims")
            if i.category == "ask_which_post":
                # The question asks which post, so the answer is a post id, not an influencer name.
                if not i.expect_clarification or not processed_later:
                    problems.append(f"{i.item_id}: ask_which_post needs expect_clarification and "
                                    "expected_outcome PROCESS_AFTER_CLARIFICATION")
                if not i.clarification_turns or not (i.clarification_turns[0].get("user") or "").strip():
                    problems.append(f"{i.item_id}: ask_which_post needs a clarification turn answered with a post id")
                if not i.gold_sub_claims:
                    problems.append(f"{i.item_id}: ask_which_post needs gold_sub_claims for the second turn")
            if i.category == "record" and any(not g.post_id for g in i.gold_sub_claims):
                problems.append(f"{i.item_id}: every record gold sub-claim needs a post_id")
    if HELDOUT_LOCK.exists() and HELDOUT_LOCK.read_text().strip() != file_sha256(dataset_path("heldout")):
        problems.append("heldout file changed after it was frozen (hash mismatch with claims_heldout.lock)")
    return problems


def freeze_heldout() -> str:
    digest = file_sha256(dataset_path("heldout"))
    HELDOUT_LOCK.write_text(digest + "\n", encoding="utf-8")
    return digest


def _base(item: EvalItem, system: str) -> dict:
    return {"system": system, "item_id": item.item_id, "split": item.split, "category": item.category,
            "expect_refusal": item.expect_refusal, "expect_clarification": item.expect_clarification,
            "gold_sub_claims": item.gold_sub_claims}


def _success(completed: bool, valid: bool, have_verdicts: bool, refused: bool, clarified: bool,
             item: EvalItem) -> bool:
    return task_success(completed, valid, have_verdicts, refused, item.expect_refusal,
                        clarified, item.expect_clarification)


def _schema_valid(fa: FinalAssessment) -> bool:
    try:
        FinalAssessment.model_validate_json(fa.model_dump_json())
        return True
    except ValidationError:
        return False


def _all_verdicts(fa: FinalAssessment) -> bool:
    return bool(fa.sub_claim_results) and all(r.verdict for r in fa.sub_claim_results)


def _predicted(fa: FinalAssessment, prefix: str = "") -> list[PredictedSubClaim]:
    return [PredictedSubClaim(text=prefix + r.sub_claim.text, verdict=r.verdict) for r in fa.sub_claim_results]


def _finsight_record(item: EvalItem, fa: FinalAssessment, logs: list[DecisionLog], *, valid: bool,
                     have_verdicts: bool, clarified: bool, success: bool,
                     predicted: list[PredictedSubClaim], failed: list[str] | None = None) -> RunRecord:
    """One RunRecord. Usage is summed over every run the item needed (each post of a record, each turn)."""
    calls = [c for log in logs for c in log.agent_calls]
    retries = [r for log in logs for r in log.retries]
    return RunRecord(
        **_base(item, "finsight"), completed=fa.status is not AssessmentStatus.FAILED, schema_valid=valid,
        all_sub_claims_have_verdict=have_verdicts, refused=fa.status is AssessmentStatus.REFUSED,
        clarified=clarified, task_success=success,
        agent_calls=len(calls), tool_calls=sum(c.tool_calls for c in calls),
        latency_ms=sum(log.total_latency_ms for log in logs), input_tokens=sum(c.input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls), cost_usd=sum(log.total_cost_usd for log in logs),
        sub_claims_assessed=len(predicted),
        sub_claims_retried=len({r.sub_claim_id for r in retries if not r.exhausted}),
        downgraded_after_retry=sum(r.exhausted for r in retries), overall_verdict=fa.overall_verdict,
        predicted_sub_claims=predicted, failed_checks=failed or [],
        error=fa.refusal_reason if fa.status is AssessmentStatus.FAILED else None,
    )


def run_finsight_item(item: EvalItem, out_dir: Path) -> tuple[RunRecord, str]:
    settings = dataclasses.replace(get_settings(), logs_dir=out_dir / "logs")
    if item.category == "record":
        return _run_record_item(item, settings)
    if item.category == "ask_which_post":
        return _run_which_post_item(item, settings)
    run = execute(item.claim, item.cutoff_date, settings=settings)
    fa = run.assessment
    valid = _schema_valid(fa)
    refused = fa.status is AssessmentStatus.REFUSED
    clarified = fa.status is AssessmentStatus.NEEDS_CLARIFICATION
    completed = fa.status is not AssessmentStatus.FAILED
    have_verdicts = refused or clarified or _all_verdicts(fa)
    record = _finsight_record(
        item, fa, [run.decision_log], valid=valid, have_verdicts=have_verdicts, clarified=clarified,
        success=_success(completed, valid, have_verdicts, refused, clarified, item), predicted=_predicted(fa),
    )
    return record, fa.model_dump_json(indent=2)


# -- record items (consistency / accuracy across every stored post) ----------------------------


def _supported_count(verdicts: list[Verdict | None]) -> tuple[int, int]:
    """(supported, supported + contradicted): the "x of y supported" count in the record."""
    supported = sum(v is Verdict.SUPPORTED for v in verdicts)
    return supported, supported + sum(v is Verdict.CONTRADICTED for v in verdicts)


def record_checks(item: EvalItem, fa: FinalAssessment, record: RecordResult | None) -> list[str]:
    """The four record checks; returns the ones that failed (empty = pass).

    1. every stored post in gold was checked and finished, 2. each post used its own date as the
    cut-off, 3. no single overall label, 4. a consistency sentence, and the supported count over the
    gold posts matches gold.
    """
    if record is None:
        return ["no record: the posts were not checked one by one"]
    failed = []
    gold_posts = list(dict.fromkeys(g.post_id for g in item.gold_sub_claims if g.post_id))
    checked = {p.video_id: p for p in record.posts}
    missing = [pid for pid in gold_posts if pid not in checked]
    unfinished = [p.video_id for p in record.posts
                  if p.status is not AssessmentStatus.COMPLETED or not _all_verdicts(p.assessment)]
    if missing or unfinished:
        failed.append(f"not every stored post was checked (missing: {', '.join(missing) or '-'}; "
                      f"unfinished: {', '.join(unfinished) or '-'})")
    wrong_cutoff = [p.video_id for p in record.posts if p.upload_date is None or p.cutoff_date != p.upload_date]
    if wrong_cutoff:
        failed.append(f"cut-off is not the post's own date for {', '.join(wrong_cutoff)}")
    if fa.overall_verdict is not None:
        failed.append(f"the record has a single overall label ({fa.overall_verdict.value})")
    if not record.consistency or record.consistency not in (fa.summary or ""):
        failed.append("no consistency sentence")
    gold = _supported_count([
        overall_verdict([g.gold_verdict for g in item.gold_sub_claims if g.post_id == pid and g.gold_verdict])
        for pid in gold_posts
    ])
    got = _supported_count([checked[pid].verdict if pid in checked else None for pid in gold_posts])
    if got != gold:
        failed.append(f"{got[0]} of {got[1]} posts supported; gold is {gold[0]} of {gold[1]}")
    return failed


def _run_record_item(item: EvalItem, settings) -> tuple[RunRecord, str]:
    """Scored by the four record checks. The whole result has no sub-claims of its own, so the usual
    "every sub-claim has a verdict" check runs per post instead."""
    run = execute(item.claim, item.cutoff_date, settings=settings)
    fa, record = run.assessment, run.record
    failed = record_checks(item, fa, record)
    checked = record.posts if record else []
    valid = _schema_valid(fa)
    refused = fa.status is AssessmentStatus.REFUSED
    clarified = fa.status is AssessmentStatus.NEEDS_CLARIFICATION
    completed = fa.status is not AssessmentStatus.FAILED
    have_verdicts = bool(checked) and all(_all_verdicts(p.assessment) for p in checked)
    # Gold texts start "2026-05-31 post: ...", so the prefix helps the sub-claim alignment.
    predicted = [pc for p in checked for pc in _predicted(p.assessment, f"{p.upload_date or 'undated'} post: ")]
    logs = [run.decision_log, *(p.decision_log for p in checked if p.decision_log is not None)]
    out = _finsight_record(
        item, fa, logs, valid=valid, have_verdicts=have_verdicts, clarified=clarified,
        success=_success(completed, valid, have_verdicts, refused, clarified, item) and not failed,
        predicted=predicted, failed=failed,
    )
    raw = {
        "assessment": fa.model_dump(mode="json"),
        "posts": [{"video_id": p.video_id, "upload_date": p.upload_date.isoformat() if p.upload_date else None,
                   "cutoff_date": p.cutoff_date.isoformat() if p.cutoff_date else None,
                   "status": p.status.value, "verdict": p.verdict.value if p.verdict else None}
                  for p in checked],
        "failed_checks": failed,
    }
    return out, json.dumps(raw, indent=2)


# -- ask_which_post items (two turns) -----------------------------------------------------------


def expected_post_ids(item: EvalItem, settings) -> list[str]:
    """Stored post ids named in the gold "which post?" turn, plus the item's own posts."""
    stored = {str(row.get("video_id")) for row in posts.load_posts(settings.posts_dir)}
    gold_turn = item.clarification_turns[0].get("system", "") if item.clarification_turns else ""
    named = [word for word in re.findall(r"[\w-]+", gold_turn) if word in stored]
    own = [str(p["video_id"]) for p in item.posts if isinstance(p, dict) and p.get("video_id")]
    return list(dict.fromkeys(named + own))


def _run_which_post_item(item: EvalItem, settings) -> tuple[RunRecord, str]:
    """Turn 1 must ask which post and name every candidate post. A question about the post counts as
    a clarification, not only a question about the name. Turn 2 reruns with the user's post id and is
    scored like a normal item."""
    first = execute(item.claim, item.cutoff_date, settings=settings)
    asked = first.assessment.status is AssessmentStatus.NEEDS_CLARIFICATION
    failed = []
    if not asked:
        failed.append(f"turn 1 returned {first.assessment.status.value}, not NEEDS_CLARIFICATION")
    else:
        question = first.assessment.clarification_question or ""
        unnamed = [pid for pid in expected_post_ids(item, settings) if pid not in question]
        if unnamed:
            failed.append(f"turn 1 question does not name post {', '.join(unnamed)}")
    answer = (item.clarification_turns[0].get("user") or "").strip() if item.clarification_turns else ""
    second: RunOutput | None = None
    if answer:
        second = execute(item.claim, item.cutoff_date, settings=settings, post_id=answer)
    else:
        failed.append("no clarification answer to run turn 2 with")
    fa = second.assessment if second else first.assessment
    valid = _schema_valid(fa)
    refused = fa.status is AssessmentStatus.REFUSED
    completed = fa.status is not AssessmentStatus.FAILED
    have_verdicts = _all_verdicts(fa)
    turn_two_clarified = second is not None and fa.status is AssessmentStatus.NEEDS_CLARIFICATION
    # Turn 2 is a normal item: completed, valid, every sub-claim labeled, no refusal, no second question.
    turn_two_ok = second is not None and task_success(completed, valid, have_verdicts, refused, False,
                                                      turn_two_clarified, False)
    if second is not None and not turn_two_ok:
        failed.append(f"turn 2 did not finish as a normal item (status {fa.status.value})")
    logs = [first.decision_log] + ([second.decision_log] if second else [])
    out = _finsight_record(
        item, fa, logs, valid=valid, have_verdicts=have_verdicts, clarified=asked,
        success=asked and turn_two_ok and not failed, predicted=_predicted(fa) if second else [], failed=failed,
    )
    raw = {"turn_1": first.assessment.model_dump(mode="json"),
           "turn_2": second.assessment.model_dump(mode="json") if second else None,
           "failed_checks": failed}
    return out, json.dumps(raw, indent=2)


def run_baseline_item(system: str, item: EvalItem) -> tuple[RunRecord, str]:
    try:
        result = (run_direct_llm if system == "direct" else run_basic_rag)(item.claim, item.cutoff_date)
    except Exception as exc:  # noqa: BLE001 - a crashing baseline is a failed item, not a failed eval
        result = BaselineResult(system=system, cutoff_date=item.cutoff_date or get_settings().corpus_as_of_date,
                                error=f"{type(exc).__name__}: {exc}")
    raw = result.model_dump_json(indent=2)
    completed = result.error is None
    clarified = result.clarification_question is not None
    stopped = result.refused or clarified
    valid = completed and (stopped or bool(result.sub_claims))
    have_verdicts = stopped or (bool(result.sub_claims) and all(c.verdict for c in result.sub_claims))
    calls = result.agent_calls
    record = RunRecord(
        **_base(item, system), completed=completed, schema_valid=valid, all_sub_claims_have_verdict=have_verdicts,
        refused=result.refused, clarified=clarified,
        task_success=_success(completed, valid, have_verdicts, result.refused, clarified, item),
        agent_calls=len(calls), tool_calls=sum(c.tool_calls for c in calls), latency_ms=result.latency_ms,
        input_tokens=sum(c.input_tokens for c in calls), output_tokens=sum(c.output_tokens for c in calls),
        cost_usd=sum(c.cost_usd for c in calls), sub_claims_assessed=len(result.sub_claims),
        overall_verdict=result.overall_verdict,
        predicted_sub_claims=[PredictedSubClaim(text=c.text, verdict=c.verdict) for c in result.sub_claims],
        error=result.error,
    )
    return record, raw


def run_eval(systems: list[str], split: str, out_root: Path = Path("results"),
             dataset_path: Path | None = None) -> Path:
    items = load_dataset(split, dataset_path)
    out_dir = out_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    records: list[RunRecord] = []
    for system in systems:
        raw_dir = out_dir / "raw" / system
        raw_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            record, raw = (run_finsight_item(item, out_dir) if system == "finsight"
                           else run_baseline_item(system, item))
            (raw_dir / f"{item.item_id}.json").write_text(raw, encoding="utf-8")
            records.append(record)
            print(f"[{system:<8}] {item.item_id:<8} success={record.task_success} "
                  f"overall={record.overall_verdict.value if record.overall_verdict else '-'}"
                  + (f" failed: {'; '.join(record.failed_checks)}" if record.failed_checks else ""))

    (out_dir / "records.jsonl").write_text("\n".join(r.model_dump_json() for r in records) + "\n", encoding="utf-8")
    write_sub_claims_csv(records, out_dir / "sub_claims.csv")
    summaries = {s: summarize([r for r in records if r.system == s]) for s in systems}
    settings = get_settings()
    header = (f"# FinSight evaluation: split={split}, {len(items)} items\n\n"
              f"Run {out_dir.name}, provider={settings.llm_provider}, model={settings.llm_model}")
    (out_dir / "summary.md").write_text(summary_markdown(summaries, header), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finsight-eval", description="Run the FinSight evaluation.")
    parser.add_argument("--system", choices=[*SYSTEMS, "all"], default="all")
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--final", action="store_true", help="required to run the frozen heldout split")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=Path, default=None, help="override the JSONL file for the split")
    parser.add_argument("--validate", action="store_true", help="validate both dataset files and exit")
    parser.add_argument("--freeze", action="store_true", help="write claims_heldout.lock (do this once)")
    parser.add_argument("--agents", metavar="NAMES",
                        help="evaluate agents one by one instead of the full system: 'all' or a comma list of "
                             "decomposer, financial, narrative, retriever, auditor, advice_boundary, tier_policy")
    args = parser.parse_args(argv)
    if args.agents:
        from finsight.eval.agent_eval.runner import parse_names, run_agent_evals

        try:
            names = parse_names(args.agents)
        except ValueError as exc:
            parser.error(str(exc))
        out_dir = run_agent_evals(names, args.out)
        print((out_dir / "summary.md").read_text(encoding="utf-8"))
        print(f"Results written to {out_dir}")
        return 0
    if args.validate or args.freeze:
        problems = validate_datasets()
        print("\n".join(problems) or "datasets OK")
        if args.freeze and not problems:
            print(f"heldout frozen: sha256 {freeze_heldout()}")
        return 1 if problems else 0
    if args.split == "heldout" and not args.final:
        parser.error("the heldout split is frozen: pass --final only for the final reported run")
    settings = get_settings()
    if args.final and settings.llm_provider == "openai" and settings.llm_model != FINAL_REPORT_MODEL:
        parser.error(f"final runs must use {FINAL_REPORT_MODEL} (got {settings.llm_model}); "
                     f"set FINSIGHT_LLM_MODEL={FINAL_REPORT_MODEL}")
    if args.split == "heldout" and args.dataset is None:
        if not HELDOUT_LOCK.exists():
            parser.error("freeze the heldout split first: finsight eval --freeze")
        if HELDOUT_LOCK.read_text().strip() != file_sha256(dataset_path("heldout")):
            parser.error("heldout file changed since it was frozen; results would not be comparable")
    systems = list(SYSTEMS) if args.system == "all" else [args.system]
    out_dir = run_eval(systems, args.split, args.out, args.dataset)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
