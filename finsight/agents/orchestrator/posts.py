"""Load stored influencer posts and pick the one a request is about.

This is a pipeline step, not an agent. It reads posts already on disk. It does not
call YouTube, X, or Transcribe. The financial and narrative agents are not involved.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from finsight.agents.claim_extraction.thesaurus import find_mentions, has_alias
from finsight.agents.orchestrator.routing import INFLUENCER_CHANNELS, INFLUENCERS, parse_date, without_cutoff

# The project window. A question does not fetch a post outside it.
WINDOW_START = date(2026, 4, 1)
WINDOW_END = date(2026, 9, 30)
_VIDEO_URL = re.compile(
    r"(?:v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
)
_WRITTEN_DATE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
)

_SKIP = frozenset({
    "said", "says", "claim", "check", "about", "this", "that", "with", "from", "just",
    "have", "been", "will", "into", "your", "their", "what", "when", "where",
})


def load_posts(root: Path) -> list[dict]:
    """Every saved post under `root` that has a video id.

    A row with no transcript is kept so a listed video can be told apart from a
    company the influencer never covered. The same video id is kept once, and the
    copy with the transcript wins.
    """
    if not root.exists():
        return []
    by_id: dict[str, dict] = {}
    order: list[str] = []
    for path in sorted(root.rglob("*.jsonl")):
        if path.name == "missing_transcripts.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = row.get("video_id")
            if not video_id:
                continue
            previous = by_id.get(video_id)
            if previous is None:
                by_id[video_id] = row
                order.append(video_id)
            elif row.get("transcript") and (
                not previous.get("transcript")
                or len(row.get("transcript") or "") > len(previous.get("transcript") or "")
            ):
                by_id[video_id] = row
    return [by_id[video_id] for video_id in order]


def in_window(row: dict) -> bool:
    """True when the post was published from 1 Apr 2026 through 30 Sep 2026."""
    uploaded = post_date(row)
    return uploaded is not None and WINDOW_START <= uploaded <= WINDOW_END


def video_id_from_text(text: str) -> str | None:
    """A YouTube video id pasted in the prompt. A bare word is not treated as an id."""
    match = _VIDEO_URL.search(text or "")
    return match.group(1) if match else None


def dates_in_text(text: str) -> set[date]:
    found: set[date] = set()
    for match in _WRITTEN_DATE.finditer(without_cutoff(text)):
        raw = match.group(0)
        if len(raw) == 10 and raw[4] == "-":
            try:
                found.add(date.fromisoformat(raw))
            except ValueError:
                continue
        elif (parsed := parse_date(raw)):
            found.add(parsed)
    return found


def posts_for(rows: list[dict], influencer: str, aliases: list[str]) -> list[dict]:
    """Posts by this influencer that mention the company."""
    channels = INFLUENCER_CHANNELS.get(influencer, frozenset({influencer}))
    kept = []
    for row in rows:
        if row.get("influencer") not in channels:
            continue
        blob = " ".join(str(row.get(key) or "") for key in ("title", "description", "transcript"))
        if any(has_alias(blob, alias) for alias in aliases):
            kept.append(row)
    return kept


def choose_post(rows: list[dict], user_text: str, influencer: str, post_id: str | None) -> dict | None | list[dict]:
    """One post, None when `post_id` misses, or the list when several posts fit the request.

    A pasted video id, a named publication date, or a named title is used before asking
    which post. None of those start a download.
    """
    pasted = post_id or video_id_from_text(user_text)
    if pasted:
        match = [row for row in rows if row.get("video_id") == pasted]
        return match[0] if match else None
    named_dates = dates_in_text(user_text)
    if named_dates:
        dated = [row for row in rows if post_date(row) in named_dates]
        if not dated:
            return None
        if len(dated) == 1:
            return dated[0]
        rows = dated
    titled = [
        row for row in rows
        if (title := (row.get("title") or "").strip()) and len(title) > 12 and title.lower() in user_text.lower()
    ]
    if len(titled) == 1:
        return titled[0]
    if len(rows) <= 1:
        return rows[0] if rows else None
    scored = sorted(
        (
            (score(user_text, row, influencer), -len(row.get("transcript") or ""), str(row.get("video_id") or ""), row)
            for row in rows
        ),
        reverse=True,
    )
    best, second = scored[0], scored[1]
    if best[0] > second[0]:
        return best[3]
    # Same overlap: the shorter post is the closer quote.
    if best[0] > 0 and len(best[3].get("transcript") or "") < len(second[3].get("transcript") or ""):
        return best[3]
    return [item[3] for item in scored]


def score(user_text: str, row: dict, influencer: str) -> int:
    """How many of the user's content words appear in the post. The company name is ignored."""
    names = {alias.lower() for alias in INFLUENCERS.get(influencer, ())}
    words = [
        word for word in re.findall(r"[a-z0-9']+", user_text.lower())
        if len(word) > 3 and word not in _SKIP and word not in names
    ]
    if not words:
        return 0
    blob = f"{row.get('title') or ''} {row.get('transcript') or ''}".lower()
    return sum(1 for word in words if word in blob)


def about_company(text: str, ticker: str, companies: dict[str, list[str]]) -> str:
    """Sentences about the named company. A sentence that only names another company is dropped.

    Tesla, TSLA, and Tesla, Inc. are the same company. AWS is Amazon. A name after
    "than" or "versus" does not by itself make the sentence about that company.
    """
    del companies
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        mentioned = {found for found, _surface, _start in find_mentions(sentence)}
        if mentioned and ticker not in mentioned:
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def post_date(row: dict) -> date | None:
    raw = row.get("upload_date") or ""
    if len(raw) == 10 and raw[4] == "-" and raw[7] == "-":
        return date.fromisoformat(raw)
    return None


_RECORD_PREFIX = "[[record]]\n"


def record_kinds(text: str) -> frozenset[str]:
    """Whether the prompt asks for the record across posts, not one post.

    `consistency` compares the influencer's own directions. `accuracy` tallies
    each post's label against filings. An explicit post id still selects one post.
    """
    kinds: set[str] = set()
    if re.search(r"\b(consistent|consistency)\b", text, flags=re.IGNORECASE):
        kinds.add("consistency")
    if re.search(r"\b(accuracy|accurate|track record)\b", text, flags=re.IGNORECASE):
        kinds.add("accuracy")
    if re.search(r"\bclaims?\b", text, flags=re.IGNORECASE) and re.search(r"\bover time\b", text, flags=re.IGNORECASE):
        kinds.add("accuracy")
    return frozenset(kinds)


def record_marker(kinds: frozenset[str], video_ids: list[str]) -> str:
    """Internal handoff. Not a question for the user."""
    return _RECORD_PREFIX + ",".join(sorted(kinds)) + "\n" + "\n".join(video_ids)


def parse_record_marker(text: str | None) -> tuple[frozenset[str], list[str]] | None:
    if not text or not text.startswith(_RECORD_PREFIX):
        return None
    lines = text[len(_RECORD_PREFIX):].splitlines()
    if len(lines) < 2:
        return None
    kinds = frozenset(part for part in lines[0].split(",") if part)
    return kinds, [line for line in lines[1:] if line]


def which_post_question(rows: list[dict]) -> str:
    lines = []
    for row in rows[:8]:
        when = row.get("upload_date") or "undated"
        lines.append(f"{when}  {row.get('video_id')}  {row.get('title') or ''}".rstrip())
    extra = f"\n{len(rows) - 8} more posts are stored." if len(rows) > 8 else ""
    return (
        "Which post should FinSight check?\n"
        + "\n".join(lines)
        + extra
        + "\nResubmit with that video id."
    )
