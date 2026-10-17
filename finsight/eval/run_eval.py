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
from datetime import date, datetime
from importlib import resources
from pathlib import Path

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finsight.agents.orchestrator.graph import execute
from finsight.baselines.basic_rag import run_basic_rag
from finsight.baselines.common import BaselineResult
from finsight.baselines.direct_llm import run_direct_llm
from finsight.config import FINAL_REPORT_MODEL, get_settings
from finsight.contracts.schemas import AssessmentStatus, FinalAssessment
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
                      "missing_name_refused"] = "normal"
    notes: str | None = None
    # Optional fields from the influencer-post items. None of them is used by the runner yet.
    cutoff_resolved_to: str | None = None  # what a relative cut-off ("last quarter") resolves to
    influencer: str | None = None
    influencer_after_clarification: str | None = None
    clarification_prompt: str | None = None  # the question the system should ask
    clarification_reply: str | None = None  # the user's answer to that question
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


def run_finsight_item(item: EvalItem, out_dir: Path) -> tuple[RunRecord, str]:
    settings = dataclasses.replace(get_settings(), logs_dir=out_dir / "logs")
    run = execute(item.claim, item.cutoff_date, settings=settings)
    fa, log = run.assessment, run.decision_log
    raw = fa.model_dump_json(indent=2)
    try:
        FinalAssessment.model_validate_json(raw)
        valid = True
    except ValidationError:
        valid = False
    refused = fa.status is AssessmentStatus.REFUSED
    clarified = fa.status is AssessmentStatus.NEEDS_CLARIFICATION
    completed = fa.status is not AssessmentStatus.FAILED
    results = fa.sub_claim_results
    have_verdicts = refused or clarified or (bool(results) and all(r.verdict for r in results))
    retried = {r.sub_claim_id for r in log.retries if not r.exhausted}
    record = RunRecord(
        **_base(item, "finsight"), completed=completed, schema_valid=valid, all_sub_claims_have_verdict=have_verdicts,
        refused=refused, clarified=clarified,
        task_success=_success(completed, valid, have_verdicts, refused, clarified, item),
        agent_calls=len(log.agent_calls), tool_calls=sum(c.tool_calls for c in log.agent_calls),
        latency_ms=log.total_latency_ms, input_tokens=sum(c.input_tokens for c in log.agent_calls),
        output_tokens=sum(c.output_tokens for c in log.agent_calls), cost_usd=log.total_cost_usd,
        sub_claims_assessed=len(fa.sub_claim_results), sub_claims_retried=len(retried),
        downgraded_after_retry=sum(r.exhausted for r in log.retries), overall_verdict=fa.overall_verdict,
        predicted_sub_claims=[PredictedSubClaim(text=r.sub_claim.text, verdict=r.verdict) for r in results],
        error=fa.refusal_reason if fa.status is AssessmentStatus.FAILED else None,
    )
    return record, raw


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
                  f"overall={record.overall_verdict.value if record.overall_verdict else '-'}")

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
