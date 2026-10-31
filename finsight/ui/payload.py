"""JSON the testing UI renders. Built only from a finished run; it does not call agents."""

from __future__ import annotations

import re
from typing import Any

from finsight.agents.orchestrator.explain import source_label, unit_heading, unit_partial_reason, verdict_label
from finsight.agents.orchestrator.expansion import report_units
from finsight.agents.orchestrator.graph import RunOutput
from finsight.agents.orchestrator.render import REPORT_LIMITS, cutoff_note, fmt_figure, fmt_numeric_check
from finsight.contracts.schemas import AuditResult, EvidenceItem, FinalAssessment, SubClaimResult
from finsight.logging.decision_log import format_trace

_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{4,64}")


def answer_slot(question: str | None) -> str | None:
    """Which form field should receive the user's answer to a clarification question."""
    if not question:
        return None
    text = question.lower()
    if "which post" in text or "video id" in text:
        return "post"
    if "which influencer" in text:
        return "influencer"
    if "which company" in text:
        return "company"
    return "claim"


def post_choices(question: str | None) -> list[dict[str, str]]:
    """Video ids named in a 'which post?' question, so the UI can offer them as buttons."""
    if not question or "which post" not in question.lower():
        return []
    found: list[dict[str, str]] = []
    for line in question.splitlines()[1:]:
        parts = [part.strip() for part in line.split("  ") if part.strip()]
        if len(parts) < 2 or _VIDEO_ID.fullmatch(parts[1]) is None:
            continue
        found.append({"date": parts[0], "video_id": parts[1], "title": " ".join(parts[2:])})
    return found


def cli_command(body: dict[str, Any]) -> str:
    """The same check as a shell command, so a UI run can be repeated in the terminal."""
    parts = ["finsight run", _quote(str(body.get("claim") or "")), "--stages"]
    if body.get("cutoff"):
        parts += ["--cutoff", str(body["cutoff"])]
    if body.get("company"):
        parts += ["--company", str(body["company"])]
    if body.get("influencer"):
        parts += ["--influencer", _quote(str(body["influencer"]))]
    if body.get("post"):
        parts += ["--post", str(body["post"])]
    return " ".join(parts)


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def build_view(out: RunOutput) -> dict[str, Any]:
    """Assessment, grouped checks, citations, and the decision log for one run."""
    assessment = out.assessment
    audits = {item.sub_claim_id: _audit(item) for item in out.decision_log.audit_results}
    return {
        "assessment": _assessment(assessment, audits),
        "units": _units(assessment),
        "decision_log": out.decision_log.model_dump(mode="json"),
        "trace_text": format_trace(out.decision_log),
        "stats": _stats(out.decision_log),
        "record": _record(out.record),
        "answer_slot": answer_slot(assessment.clarification_question),
        "post_choices": post_choices(assessment.clarification_question),
        "cutoff_note": cutoff_note(assessment),
        "overall_label": verdict_label(assessment.overall_verdict),
        "limits": REPORT_LIMITS,
        "match": match_view(assessment),
    }


BUCKET_LABELS = {
    "both": ("Same", "You and the influencer state the same fact. It counts toward the verdict."),
    "conflict": ("Different", "Same fact, different figure or direction. The influencer's version is checked."),
    "quote_only": ("Also said", "The influencer said this too. It is checked but not counted in the verdict."),
    "user_only": ("Not said", "You typed this, but the matched line does not say it. It is not checked."),
}


def match_view(assessment: FinalAssessment) -> dict[str, Any] | None:
    """You asked / they said, with each fact's bucket and (when checked) its verdict."""
    match = assessment.match
    if match is None:
        return None
    verdicts = {row.sub_claim.sub_claim_id: row for row in assessment.sub_claim_results}
    facts = []
    for fact in match.facts:
        label, hint = BUCKET_LABELS.get(fact.bucket, (fact.bucket, ""))
        row = verdicts.get(fact.sub_claim_id or "")
        facts.append({
            **fact.model_dump(mode="json"),
            "label": label,
            "hint": hint,
            "verdict": row.verdict.value if row else None,
            "verdict_label": verdict_label(row.verdict) if row else None,
        })
    return {
        **match.model_dump(mode="json"),
        "facts": facts,
        "url": f"https://www.youtube.com/watch?v={match.video_id}" if match.video_id else None,
        "counts": {key: sum(fact["bucket"] == key for fact in facts) for key in BUCKET_LABELS},
    }


def _assessment(assessment: FinalAssessment, audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    data = assessment.model_dump(mode="json")
    data["sub_claim_results"] = [
        _result(row, audits.get(row.sub_claim.sub_claim_id)) for row in assessment.sub_claim_results
    ]
    data["verdict_label"] = verdict_label(assessment.overall_verdict)
    return data


def _result(row: SubClaimResult, audit: dict[str, Any] | None) -> dict[str, Any]:
    data = row.model_dump(mode="json")
    data["verdict_label"] = verdict_label(row.verdict)
    data["supporting"] = [_evidence(item) for item in row.supporting]
    data["challenging"] = [_evidence(item) for item in row.challenging]
    data["audit"] = audit
    return data


def _evidence(item: EvidenceItem) -> dict[str, Any]:
    data = item.model_dump(mode="json")
    data["publisher_page"] = item.publisher_page
    data["source_label"] = source_label(item)
    data["computed_label"] = fmt_figure(item.computed_value) if item.computed_value is not None else None
    if item.comparison_value is not None:
        data["comparison_label"] = fmt_figure(item.comparison_value)
    return data


def _audit(audit: AuditResult) -> dict[str, Any]:
    data = audit.model_dump(mode="json")
    check = audit.numeric_check
    data["numeric_label"] = fmt_numeric_check(check) if check is not None and check.computed is not None else None
    return data


def _units(assessment: FinalAssessment) -> list[dict[str, Any]]:
    units = []
    headline = set(assessment.headline_sub_claim_ids)
    for unit in report_units(assessment.sub_claim_results):
        ids = [row.sub_claim.sub_claim_id for row in unit.results]
        units.append({
            "group": "also" if headline and not (set(ids) & headline) else "asked",
            "alignment": next((row.sub_claim.alignment for row in unit.results if row.sub_claim.alignment), None),
            "heading": unit_heading(unit),
            "verdict": unit.verdict.value,
            "label": verdict_label(unit.verdict),
            "partial_reason": unit_partial_reason(unit),
            "is_parent": unit.is_parent,
            "ids": [row.sub_claim.sub_claim_id for row in unit.results],
        })
    return units


def _stats(log) -> dict[str, Any]:
    return {
        "agent_calls": len(log.agent_calls),
        "tool_calls": sum(call.tool_calls for call in log.agent_calls),
        "retries": sum(not row.exhausted for row in log.retries),
        "tokens": log.total_tokens,
        "cost_usd": log.total_cost_usd,
        "latency_ms": log.total_latency_ms,
        "temporal_violations": len(log.temporal_violations),
    }


def _record(record) -> dict[str, Any] | None:
    if record is None:
        return None
    posts = []
    for post in record.posts:
        posts.append({
            "video_id": post.video_id,
            "title": post.title,
            "upload_date": post.upload_date.isoformat() if post.upload_date else None,
            "cutoff_date": post.cutoff_date.isoformat() if post.cutoff_date else None,
            "status": post.status.value,
            "verdict": post.verdict.value if post.verdict else None,
            "label": verdict_label(post.verdict) if post.verdict else post.status.value,
            "view": build_view(RunOutput(post.assessment, post.decision_log)) if post.decision_log else None,
        })
    return {
        "kinds": sorted(record.kinds),
        "consistency": record.consistency,
        "accuracy": record.accuracy,
        "unfinished": list(record.unfinished),
        "posts": posts,
    }
