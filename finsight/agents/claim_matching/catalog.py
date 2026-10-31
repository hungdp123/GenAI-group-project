"""The stored claim catalog: one row per line an influencer said, extracted once when the corpus is built.

File: data/influencers/claims/claim_update_v3.jsonl (see claims_catalog_path). A row has video_id,
influencer (channel key), title, upload_date, company_ticker, text, sub_claim_type, measure and fields.
This module only reads it. The match step filters it by influencer, company, date and video.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

from finsight.agents.claim_extraction.thesaurus import find_mentions
from finsight.agents.orchestrator.routing import INFLUENCER_CHANNELS


@dataclass(frozen=True)
class Line:
    line_id: str
    video_id: str
    channel: str
    title: str
    upload_date: date | None
    ticker: str
    text: str
    row: dict


def _date(raw: object) -> date | None:
    text = str(raw or "")
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None
    return None


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> tuple[Line, ...]:
    del mtime  # part of the cache key only: an edited file is read again
    lines: list[Line] = []
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle):
            if not raw.strip():
                continue
            row = json.loads(raw)
            text = str(row.get("text") or row.get("source_span") or "").strip()
            if not text:
                continue
            lines.append(Line(
                line_id=str(row.get("sub_claim_id") or f"row{number}"),
                video_id=str(row.get("video_id") or ""),
                channel=str(row.get("influencer") or ""),
                title=str(row.get("title") or ""),
                upload_date=_date(row.get("upload_date")),
                ticker=str(row.get("company_ticker") or "NONE"),
                text=text,
                row=row,
            ))
    return tuple(lines)


def load_catalog(path: Path) -> list[Line]:
    if not path.exists():
        return []
    return list(_load(str(path), path.stat().st_mtime))


def about_company(line: Line, ticker: str | None) -> bool:
    """The line is stored under this company, or is stored under none and names it (AWS names Amazon)."""
    if not ticker:
        return True
    if line.ticker == ticker:
        return True
    # Stored under no company, or under another one but naming this one ("while Tesla's revenues declined,
    # Microsoft has grown all three"): the facts still have to line up before the line is used.
    return any(found == ticker for found, _surface, _start in find_mentions(line.text))


def lines_for(
    lines: list[Line],
    *,
    influencer: str | None,
    ticker: str | None,
    cutoff: date | None = None,
    video_id: str | None = None,
    video_ids: set[str] | None = None,
) -> list[Line]:
    """Candidate lines: this influencer, this company, said on or before the cut-off."""
    channels = INFLUENCER_CHANNELS.get(influencer or "", frozenset({influencer})) if influencer else None
    out = []
    for line in lines:
        # An older catalog row has no channel; the video filter then keeps it to this influencer's posts.
        if channels is not None and line.channel and line.channel not in channels:
            continue
        if video_id and line.video_id != video_id:
            continue
        if video_ids is not None and line.video_id not in video_ids:
            continue
        if cutoff and line.upload_date and line.upload_date > cutoff:
            continue
        if not about_company(line, ticker):
            continue
        out.append(line)
    return out


def posts_from_catalog(lines: list[Line]) -> list[dict]:
    """Post rows built from the catalog, for a machine that has the catalog but not the transcripts.

    The "transcript" is the stored lines of that video in order. It is enough to choose and label a post;
    it is not the full spoken text.
    """
    by_video: dict[str, dict] = {}
    for line in lines:
        post = by_video.get(line.video_id)
        if post is None:
            post = by_video[line.video_id] = {
                "influencer": line.channel,
                "video_id": line.video_id,
                "title": line.title,
                "description": "",
                "upload_date": line.upload_date.isoformat() if line.upload_date else "",
                "transcript": "",
                "from_catalog": True,
            }
        post["transcript"] = f"{post['transcript']} {line.text}".strip()
    return list(by_video.values())


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def transcript_lines(posts: list[dict], ticker: str | None) -> list[Line]:
    """Sentences of stored transcripts about this company, as Lines (ids t1, t2, ...). Used when the catalog
    has no line that fits: the speaker may have said it in a sentence the catalog did not store."""
    from finsight.agents.claim_extraction.fields import company_for_sentences

    out: list[Line] = []
    for post in posts:
        if post.get("from_catalog"):
            continue  # built from the catalog itself: nothing new to search
        sentences = [part.strip() for part in _SENTENCE.split(str(post.get("transcript") or "")) if part.strip()]
        companies = company_for_sentences(sentences)
        for sentence, company in zip(sentences, companies):
            if ticker and company not in (ticker, None):
                continue
            out.append(Line(
                line_id=f"{post.get('video_id')}-t{len(out) + 1}",
                video_id=str(post.get("video_id") or ""),
                channel=str(post.get("influencer") or ""),
                title=str(post.get("title") or ""),
                upload_date=_date(post.get("upload_date")),
                ticker=company or (ticker or "NONE"),
                text=sentence,
                row={},
            ))
    return out
