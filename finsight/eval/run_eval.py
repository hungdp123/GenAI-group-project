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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from finsight.agents.orchestrator import posts
from finsight.agents.orchestrator.aggregation import overall_verdict
from finsight.agents.orchestrator.graph import RunOutput, execute
from finsight.agents.orchestrator.record import RecordResult
from finsight.baselines.basic_rag import run_basic_rag
from finsight.baselines.common import BaselineResult
from finsight.baselines.direct_llm import run_direct_llm
from finsight.config import FINAL_REPORT_MODEL, get_settings
from finsight.contracts.schemas import AssessmentStatus, DecisionLog, FinalAssessment, Verdict
from finsight.eval.match_eval import fact_verdicts, match_correct, score_alignment
from finsight.eval.metrics import (
    GoldFact,
    GoldMatch,
    GoldSubClaim,
    PredictedSubClaim,
    RunRecord,
    summarize,
    summary_markdown,
    task_success,
    write_sub_claims_csv,
)

# llm: plain LLM (no data). direct: LLM + the company's XBRL facts. rag: Basic RAG. rag_quote: RAG + the
# influencer's actual line (from the gold match), so only verification differs from FinSight. See
# docs/evaluation.md.
SYSTEMS = ("llm", "direct", "rag", "rag_quote", "finsight")
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
    # Claim matching gold (docs/claim_matching.md). gold_sub_claims is derived from gold_alignment when empty.
    match_case: str | None = None
    gold_match: GoldMatch | None = None
    gold_alignment: list[GoldFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _sub_claims_from_alignment(self) -> "EvalItem":
        if self.gold_alignment and not self.gold_sub_claims:
            seen: set[str] = set()
            for fact in self.gold_alignment:
                if fact.bucket == "user_only" or not fact.quote_span or fact.quote_span in seen:
                    continue
                seen.add(fact.quote_span)
                self.gold_sub_claims.append(GoldSubClaim(
                    text=fact.quote_span, gold_verdict=fact.gold_verdict, sub_claim_type=fact.sub_claim_type,
                    evidence=fact.evidence, influencer=self.influencer, rule_note=fact.rule_note))
        return self

    @property
    def gold_overall(self) -> Verdict | None:
        """The headline verdict: the facts the user asked about, when every one of them is labeled."""
        asked = [fact.gold_verdict for fact in self.gold_alignment if fact.asked]
        if not asked or any(verdict is None for verdict in asked):
            return None
        return overall_verdict(asked)


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
            if i.gold_match is not None:
                problems += _match_problems(i)
            if i.category == "ask_which_post" and i.gold_match is None:
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


def _match_problems(i: EvalItem) -> list[str]:
    gold, out = i.gold_match, []
    if gold.status == "MATCHED":
        if not gold.quote_ids:
            out.append(f"{i.item_id}: a MATCHED gold_match needs quote_ids")
        if not i.gold_alignment:
            out.append(f"{i.item_id}: a MATCHED item needs gold_alignment")
        if i.expect_clarification or i.expect_refusal:
            out.append(f"{i.item_id}: a MATCHED item is analyzed, so it cannot expect a stop")
        for fact in i.gold_alignment:
            if fact.bucket != "user_only" and fact.quote_id not in set(gold.quote_ids) | set(gold.acceptable_quote_ids):
                out.append(f"{i.item_id}: fact quote_id {fact.quote_id} is not one of the gold lines")
            if fact.bucket == "user_only" and (fact.quote_span or fact.gold_verdict):
                out.append(f"{i.item_id}: a user_only fact has no quote and no verdict (it is never checked)")
    else:
        if not i.expect_clarification:
            out.append(f"{i.item_id}: a {gold.status} gold_match asks the user, so expect_clarification is true")
        if gold.status == "ASK" and len(gold.video_ids) < 2:
            out.append(f"{i.item_id}: an ASK gold_match needs at least two video_ids")
    return out


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
    if item.category == "ask_which_post" and item.gold_match is None:
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
    if item.gold_match is not None:
        _score_match(record, item, fa)
    return record, fa.model_dump_json(indent=2)


def _score_match(record: RunRecord, item: EvalItem, fa: FinalAssessment) -> None:
    """Right line, right fact buckets, right verdict per fact: three scores, never mixed."""
    predicted = fa.match if fa.match is not None and fa.match.status == "MATCHED" else None
    record.match_case = item.match_case
    record.predicted_match_status = fa.match.status if fa.match else (
        "ASK" if fa.clarification_question else fa.status.value)
    record.match_correct = match_correct(item.gold_match, fa.match, fa.clarification_question)
    alignment = score_alignment(item.gold_alignment, predicted)
    record.alignment_correct, record.alignment_total = alignment.correct, alignment.total
    record.fact_verdicts = fact_verdicts(item.gold_alignment, fa) if record.match_correct else []
    record.failed_checks += [f"alignment {miss}" for miss in alignment.misses]
    if not record.match_correct:
        record.failed_checks.append(f"match: gold {item.gold_match.status}, got {record.predicted_match_status}")


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


def baseline_cutoff(item: EvalItem) -> date | None:
    """The same as-of date FinSight judges on: the user's cut-off, else the matched video's date."""
    if item.cutoff_date:
        return item.cutoff_date
    for post in item.posts:
        raw = post.get("upload_date") if isinstance(post, dict) else None
        if raw:
            return date.fromisoformat(str(raw)[:10])
    return None


def gold_quote(item: EvalItem) -> str | None:
    """The influencer's actual line(s) for the RAG + quote baseline (an oracle: no matching errors)."""
    if item.gold_match is None or item.gold_match.status != "MATCHED":
        return None
    from finsight.agents.claim_extraction.build import claims_catalog_path
    from finsight.agents.claim_matching.catalog import load_catalog

    wanted = set(item.gold_match.quote_ids)
    lines = [line for line in load_catalog(claims_catalog_path(get_settings().posts_dir)) if line.line_id in wanted]
    return " ".join(line.text for line in lines) or None


def run_baseline_item(system: str, item: EvalItem) -> tuple[RunRecord, str]:
    cutoff = baseline_cutoff(item)
    try:
        if system == "llm":
            result = run_direct_llm(item.claim, cutoff, with_facts=False)
        elif system == "direct":
            result = run_direct_llm(item.claim, cutoff)
        elif system == "rag_quote":
            result = run_basic_rag(item.claim, cutoff, quote=gold_quote(item))
            result.system = "rag_quote"
        else:
            result = run_basic_rag(item.claim, cutoff)
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


def llm_preflight() -> str | None:
    """One tiny real call before a paid run. Returns the error text, or None when the model answers.

    A paid run where every call fails still "finishes": FinSight falls back to its rules and every baseline item
    fails in a few milliseconds with 0 tokens. This stops that before it wastes a run.
    """
    settings = get_settings()
    if settings.llm_provider == "mock":
        return None
    from finsight.llm.client import complete

    try:
        response = complete("Reply with the single word OK.", max_tokens=200)
    except Exception as exc:  # noqa: BLE001 - reported to the user, not raised
        return f"{type(exc).__name__}: {exc}"
    if not (response.text or "").strip():
        return (f"the model returned an empty answer (finish_reason={response.finish_reason}); "
                "a reasoning model may need a larger max_tokens")
    return None


def run_eval(systems: list[str], split: str, out_root: Path = Path("results"),
             dataset_path: Path | None = None, route: str | None = None) -> Path:
    """`route` ("financial" / "narrative"): every system runs only on items with a labeled fact whose gold route is
    that agent, and comparison.md scores only those facts. The set comes from the gold labels, not from a run."""
    items = load_dataset(split, dataset_path)
    out_dir = out_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []
    if route:
        from finsight.eval.compare import items_on_route

        items = items_on_route(items, route)
        (out_dir / f"items_{route}.txt").write_text("\n".join(i.item_id for i in items) + "\n", encoding="utf-8")
        print(f"[route] {len(items)} items have a labeled {route} fact: " + ", ".join(i.item_id for i in items))
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
                  + (f" failed: {'; '.join(record.failed_checks)}" if record.failed_checks else "")
                  + (f" ERROR: {record.error}" if record.error else ""))

    (out_dir / "records.jsonl").write_text("\n".join(r.model_dump_json() for r in records) + "\n", encoding="utf-8")
    write_sub_claims_csv(records, out_dir / "sub_claims.csv")
    summaries = {s: summarize([r for r in records if r.system == s]) for s in systems}
    settings = get_settings()
    scope = f", {route} items only" if route else ""
    header = (f"# FinSight evaluation: split={split}, {len(items)} items{scope}\n\n"
              f"Run {out_dir.name}, provider={settings.llm_provider}, model={settings.llm_model}")
    (out_dir / "summary.md").write_text(summary_markdown(summaries, header), encoding="utf-8")
    if len(systems) > 1:
        from finsight.eval.compare import write_comparison

        write_comparison(out_dir, items, systems, route)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finsight-eval", description="Run the FinSight evaluation.")
    parser.add_argument("--system", default="all", metavar="NAMES",
                        help=f"all, or a comma list of: {', '.join(SYSTEMS)} (e.g. finsight,llm,direct)")
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--final", action="store_true", help="required to run the frozen heldout split")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--dataset", type=Path, default=None, help="override the JSONL file for the split")
    parser.add_argument("--validate", action="store_true", help="validate both dataset files and exit")
    parser.add_argument("--freeze", action="store_true", help="write claims_heldout.lock (do this once)")
    parser.add_argument("--route", choices=["financial", "narrative"],
                        help="only items with a labeled fact the router sends to this agent; score only those facts")
    parser.add_argument("--compare", type=Path, metavar="RUN_DIR",
                        help="recompute comparison.md for a finished run directory (all its systems)")
    parser.add_argument("--matcher", action="store_true",
                        help="score claim matching only (no agents): match and alignment accuracy per case")
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
    if args.compare:
        from finsight.eval.compare import write_comparison

        found = {p.name for p in (args.compare / "raw").iterdir() if p.is_dir()}
        systems = [s for s in SYSTEMS if s in found] + sorted(found - set(SYSTEMS))
        path = write_comparison(args.compare, load_dataset(args.split, args.dataset), systems, args.route)
        print(path.read_text(encoding="utf-8"))
        return 0
    if args.matcher:
        from finsight.eval.match_eval import matcher_report, run_matcher_only
        from finsight.agents.claim_matching.matcher import matcher_mode

        if args.split == "heldout" and not args.final:
            parser.error("the heldout split is frozen: pass --final only for the final reported run")
        rows = run_matcher_only(load_dataset(args.split, args.dataset))
        out_dir = args.out / f"match-{datetime.now():%Y%m%d-%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = matcher_report(rows, f"# Claim matching: split={args.split}, mode={matcher_mode()}")
        (out_dir / "match_report.md").write_text(report, encoding="utf-8")
        (out_dir / "match_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(report)
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
    # The reported run is gpt-5.4. gpt-5.4-nano is allowed for the model-strength check on the same split.
    allowed_final = {FINAL_REPORT_MODEL, "gpt-5.4-nano"}
    if args.final and settings.llm_provider == "openai" and settings.llm_model not in allowed_final:
        parser.error(f"final runs must use {FINAL_REPORT_MODEL} or gpt-5.4-nano (got {settings.llm_model}); "
                     f"set FINSIGHT_LLM_MODEL={FINAL_REPORT_MODEL}")
    if args.split == "heldout" and args.dataset is None:
        if not HELDOUT_LOCK.exists():
            parser.error("freeze the heldout split first: finsight eval --freeze")
        if HELDOUT_LOCK.read_text().strip() != file_sha256(dataset_path("heldout")):
            parser.error("heldout file changed since it was frozen; results would not be comparable")
    problem = llm_preflight()
    if problem:
        print(f"The LLM call failed before the run started ({settings.llm_provider}, {settings.llm_model}):\n"
              f"  {problem}\n"
              "Check: `pip install -e \".[dev,openai]\"` in this venv, the API key in config.yaml, "
              "FINSIGHT_SPEND_LIMIT_USD, and `finsight usage`. Or run in mock mode: `finsight mode mock`.")
        return 2
    systems = list(SYSTEMS) if args.system == "all" else [s.strip() for s in args.system.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"unknown system(s) {', '.join(unknown)}; choose from {', '.join(SYSTEMS)} or all")
    out_dir = run_eval(systems, args.split, args.out, args.dataset, route=args.route)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"Results written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
