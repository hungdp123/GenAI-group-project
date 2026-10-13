"""A consistency or accuracy question checks every stored post, then counts.

Each post keeps its own publication date as the cut-off. Consistency compares the
influencer's directions with each other. Accuracy counts labels against filings.
A Supported post and a Contradicted post are an accuracy result, not a wording flip.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from finsight.agents.orchestrator import posts
from finsight.agents.orchestrator.explain import verdict_label
from finsight.config import BOUNDARY_DISCLAIMER
from finsight.contracts.schemas import AssessmentStatus, FinalAssessment, Verdict
from finsight.logging.decision_log import DecisionLogRecorder

_MEASURES = (
    ("earnings", re.compile(r"\b(earnings|profits?|net income)\b", flags=re.IGNORECASE)),
    ("margins", re.compile(r"\bmargins?\b", flags=re.IGNORECASE)),
    ("revenue", re.compile(r"\b(revenue|sales)\b", flags=re.IGNORECASE)),
)
_UP = re.compile(r"\b(grew|grown|growth|rising|rose|higher|expanding|expanded|strong|up)\b", flags=re.IGNORECASE)
_DOWN = re.compile(
    r"\b(fell|falling|fallen|declined|decline|shrinking|shrank|lower|weak|down)\b", flags=re.IGNORECASE,
)


def directions(text: str) -> dict[str, set[str]]:
    """Direction words for each measure in `text`. A sentence with both is skipped."""
    found: dict[str, set[str]] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        up, down = bool(_UP.search(sentence)), bool(_DOWN.search(sentence))
        if up == down:
            continue
        side = "up" if up else "down"
        for name, pattern in _MEASURES:
            if pattern.search(sentence):
                found.setdefault(name, set()).add(side)
    return found


def consistency_sentence(per_post: list[tuple[str, dict[str, set[str]]]]) -> str:
    """`per_post` is (label, directions). A measure that rises in one post and falls in another flips."""
    by_measure: dict[str, set[str]] = {}
    for _label, found in per_post:
        for measure, sides in found.items():
            by_measure.setdefault(measure, set()).update(sides)
    flips = [measure for measure, sides in by_measure.items() if sides == {"up", "down"}]
    if flips:
        names = ", ".join(flips)
        return f"The posts are not consistent on {names}: one post has it rising and another has it falling."
    limit = " This does not compare price, rank, or a shift from bullish to cautious."
    if not by_measure:
        return ("The stored posts do not state a direction for the same measure, so consistency cannot be judged."
                + limit)
    stated = ", ".join(sorted(by_measure))
    return f"No measure flips across the posts. The posts that state a direction agree on {stated}.{limit}"


def accuracy_sentence(verdicts: list[Verdict | None]) -> str:
    counts = Counter(verdict for verdict in verdicts if verdict is not None)
    supported = counts[Verdict.SUPPORTED]
    contradicted = counts[Verdict.CONTRADICTED]
    decided = supported + contradicted
    tally = ", ".join(
        f"{counts[verdict]} {verdict_label(verdict).lower()}"
        for verdict in (
            Verdict.SUPPORTED,
            Verdict.PARTIALLY_SUPPORTED,
            Verdict.CONTRADICTED,
            Verdict.INSUFFICIENT_EVIDENCE,
        )
        if counts[verdict]
    ) or "none could be labeled"
    if decided == 0:
        scored = "No post was supported or contradicted, so accuracy is not scored."
    else:
        scored = f"Of the posts that were supported or contradicted, {supported} of {decided} were supported."
    held = "Partly supported and unconfirmed posts are listed and are not counted as right or wrong."
    as_of = ("Each label uses filings filed on or before that post's date. "
             "A forecast is not scored against what happened later. "
             "A contradicted label means that sentence does not match the filing.")
    return f"Accuracy: {tally}. {scored} {held} {as_of}"


def assess_record(
    claim: str,
    *,
    kinds: frozenset[str],
    video_ids: list[str],
    cutoff,
    company_ticker: str | None,
    influencer: str | None,
    posts_root: Path,
    run_one: Callable[..., FinalAssessment],
    on_stage: Callable[[str, dict, DecisionLogRecorder], None] | None = None,
) -> FinalAssessment:
    """Check each stored post, then write the record. `run_one` is one normal check.

    Each post uses the one retry already allowed. That retry repeats the same check.
    It does not start a second agent. A post that cannot be finished leaves the record incomplete.
    """
    checked = []
    unfinished: list[str] = []
    for index, video_id in enumerate(video_ids, start=1):
        if on_stage is not None:
            try:
                on_stage("record", {"post_id": video_id, "index": index, "count": len(video_ids)}, None)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001 - showing progress must not stop the record
                pass
        assessment = run_one(
            claim, cutoff, company_ticker=company_ticker, influencer=influencer,
            post_id=video_id, on_stage=on_stage,
        )
        checked.append(assessment)
        if assessment.status is not AssessmentStatus.COMPLETED:
            unfinished.append(video_id)
    rows = {str(row.get("video_id")): row for row in posts.load_posts(posts_root)}
    lines = [
        f"This is the record across {len(video_ids)} stored posts. "
        "Each post uses its own publication date as the cut-off. There is no single overall label. "
        "Each post keeps the one retry already allowed. "
        "That retry repeats the same check and does not start a second agent."
    ]
    if unfinished:
        lines.append(
            "The record is incomplete. These posts could not be finished: " + ", ".join(unfinished) + "."
        )
    per_post = []
    verdicts = []
    post_lines = []
    for video_id, assessment in zip(video_ids, checked):
        row = rows.get(video_id, {})
        transcript = str(row.get("transcript") or "")
        found = directions(transcript)
        per_post.append((video_id, found))
        verdicts.append(assessment.overall_verdict)
        label = verdict_label(assessment.overall_verdict) if assessment.overall_verdict else assessment.status.value
        when = row.get("upload_date") or "undated"
        title = row.get("title") or "untitled"
        post_lines.append(f"{when}  {video_id}  {label}  {title}")
    if "consistency" in kinds:
        lines.append(consistency_sentence(per_post))
    if "accuracy" in kinds:
        lines.append(accuracy_sentence(verdicts))
    lines.append("Posts:")
    lines.extend(post_lines)
    return FinalAssessment(
        request_id=checked[-1].request_id if checked else "record",
        status=AssessmentStatus.COMPLETED,
        claim=claim,
        summary="\n".join(lines),
        boundary_disclaimer=BOUNDARY_DISCLAIMER,
    )
