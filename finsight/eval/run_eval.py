"""End-to-end evaluation runner. Owner: Person 5 (Hung).

    python -m finsight.eval.run_eval --system all --split dev
    python -m finsight.eval.run_eval --system all --split heldout --final   # frozen split: final run only
    python -m finsight.eval.run_eval --validate                            # check both dataset files
    python -m finsight.eval.run_eval --freeze                              # lock the heldout file (once)

Outputs results/{timestamp}/: raw/{system}/{item_id}.json, logs/ (FinSight decision logs),
records.jsonl, claims.csv and summary.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
from datetime import date, datetime
from importlib import resources
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finsight.agents.orchestrator.graph import execute
from finsight.baselines.basic_rag import run_basic_rag
from finsight.baselines.common import BaselineResult
from finsight.baselines.direct_llm import run_direct_llm
from finsight.config import FINAL_REPORT_MODEL, get_settings
from finsight.contracts.schemas import AssessmentStatus, FinalAssessment, Verdict
from finsight.eval.metrics import (
    GoldClaim,
    PredictedClaim,
    RunRecord,
    summarize,
    summary_markdown,
    task_success,
    write_claims_csv,
)

SYSTEMS = ("direct", "rag", "finsight")
DATASET_DIR = Path(str(resources.files("finsight.eval.datasets")))
HELDOUT_LOCK = DATASET_DIR / "claims_heldout.lock"


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    thesis: str
    cutoff_date: date | None = None
    split: Literal["dev", "heldout"]
    gold_claims: list[GoldClaim] = Field(default_factory=list)
    expect_refusal: bool = False
    category: Literal["normal", "adversarial_advice", "out_of_scope", "injection", "temporal"] = "normal"
    notes: str | None = None


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
    dev_theses = {i.thesis.strip().lower() for i in items["dev"]}
    problems += [f"{i.item_id}: thesis also appears in dev (leak)" for i in items["heldout"]
                 if i.thesis.strip().lower() in dev_theses]
    for split, split_items in items.items():
        for i in split_items:
            if i.split != split:
                problems.append(f"{i.item_id}: split={i.split} but stored in {split} file")
            if i.expect_refusal and i.gold_claims:
                problems.append(f"{i.item_id}: refusal items must not have gold_claims")
            if not i.expect_refusal and not i.gold_claims:
                problems.append(f"{i.item_id}: non-refusal items need gold_claims")
    if HELDOUT_LOCK.exists() and HELDOUT_LOCK.read_text().strip() != file_sha256(dataset_path("heldout")):
        problems.append("heldout file changed after it was frozen (hash mismatch with claims_heldout.lock)")
    return problems


def freeze_heldout() -> str:
    digest = file_sha256(dataset_path("heldout"))
    HELDOUT_LOCK.write_text(digest + "\n", encoding="utf-8")
    return digest


def _base(item: EvalItem, system: str) -> dict:
    return {"system": system, "item_id": item.item_id, "split": item.split, "category": item.category,
            "expect_refusal": item.expect_refusal, "gold_claims": item.gold_claims}


def run_finsight_item(item: EvalItem, out_dir: Path) -> tuple[RunRecord, str]:
    settings = dataclasses.replace(get_settings(), logs_dir=out_dir / "logs")
    run = execute(item.thesis, item.cutoff_date, settings=settings)
    fa, log = run.assessment, run.decision_log
    raw = fa.model_dump_json(indent=2)
    try:
        FinalAssessment.model_validate_json(raw)
        valid = True
    except ValidationError:
        valid = False
    refused = fa.status is AssessmentStatus.REFUSED
    completed = fa.status is not AssessmentStatus.FAILED
    have_verdicts = refused or (bool(fa.claim_results) and all(r.verdict for r in fa.claim_results))
    retried = {r.claim_id for r in log.retries if not r.exhausted}
    record = RunRecord(
        **_base(item, "finsight"), completed=completed, schema_valid=valid, all_claims_have_verdict=have_verdicts,
        refused=refused, task_success=task_success(completed, valid, have_verdicts, refused, item.expect_refusal),
        agent_calls=len(log.agent_calls), tool_calls=sum(c.tool_calls for c in log.agent_calls),
        latency_ms=log.total_latency_ms, input_tokens=sum(c.input_tokens for c in log.agent_calls),
        output_tokens=sum(c.output_tokens for c in log.agent_calls), cost_usd=log.total_cost_usd,
        claims_assessed=len(fa.claim_results), claims_retried=len(retried),
        downgraded_after_retry=sum(r.exhausted for r in log.retries), overall_verdict=fa.overall_verdict,
        predicted_claims=[PredictedClaim(text=r.claim.text, verdict=r.verdict) for r in fa.claim_results],
        error=fa.refusal_reason if fa.status is AssessmentStatus.FAILED else None,
    )
    return record, raw


def run_baseline_item(system: str, item: EvalItem) -> tuple[RunRecord, str]:
    try:
        result = (run_direct_llm if system == "direct" else run_basic_rag)(item.thesis, item.cutoff_date)
    except Exception as exc:  # noqa: BLE001 - a crashing baseline is a failed item, not a failed eval
        result = BaselineResult(system=system, cutoff_date=item.cutoff_date or get_settings().corpus_as_of_date,
                                error=f"{type(exc).__name__}: {exc}")
    raw = result.model_dump_json(indent=2)
    completed = result.error is None
    valid = completed and (result.refused or bool(result.claims))
    have_verdicts = result.refused or (bool(result.claims) and all(c.verdict for c in result.claims))
    calls = result.agent_calls
    record = RunRecord(
        **_base(item, system), completed=completed, schema_valid=valid, all_claims_have_verdict=have_verdicts,
        refused=result.refused, task_success=task_success(completed, valid, have_verdicts, result.refused,
                                                           item.expect_refusal),
        agent_calls=len(calls), tool_calls=sum(c.tool_calls for c in calls), latency_ms=result.latency_ms,
        input_tokens=sum(c.input_tokens for c in calls), output_tokens=sum(c.output_tokens for c in calls),
        cost_usd=sum(c.cost_usd for c in calls), claims_assessed=len(result.claims),
        overall_verdict=result.overall_verdict,
        predicted_claims=[PredictedClaim(text=c.text, verdict=c.verdict) for c in result.claims],
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
    write_claims_csv(records, out_dir / "claims.csv")
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
    args = parser.parse_args(argv)
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
