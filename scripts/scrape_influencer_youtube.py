"""Collect public YouTube listings and spoken transcripts for influencer channels.

The channel tab only has titles and rough dates. Pass --transcripts to open each
listed video and save the English caption track (the spoken words) plus the
description and the real upload date.

YouTube's Data API will not download another creator's captions. This uses the
public caption track the video page already exposes. When that download returns
HTTP 429, the same captions are requested through youtube-transcript-api, which
uses a different YouTube endpoint. transcript_via records which one was used.

Requires yt-dlp and youtube-transcript-api. From the repo root:

    python3 -m venv .venv
    .venv/bin/pip install yt-dlp youtube-transcript-api
    .venv/bin/python scripts/scrape_influencer_youtube.py
    .venv/bin/python scripts/scrape_influencer_youtube.py --transcripts

Output goes to data/influencers/youtube/<run-date>/, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yt_dlp

# Three calendar months back from the run date is computed below. These are the
# two channels chosen for the influencer-claim experiment.
CHANNELS = {
    "joseph_hogue": "https://www.youtube.com/@josephhogue/videos",
    "joseph_carlson": "https://www.youtube.com/@JosephCarlsonShow/videos",
    # The main Carlson channel is nearly idle. Recent videos are posted here.
    "joseph_carlson_after_hours": "https://www.youtube.com/@JosephCarlsonAfterHours/videos",
    "mark_roussin": "https://www.youtube.com/@MarkRoussinCPA/videos",
    "alex_reffett": "https://www.youtube.com/@TickerSymbolYOU/videos",
    # Kevin Paffrath posts several videos a day: pass --limit 600 for a six-month window.
    "meet_kevin": "https://www.youtube.com/@MeetKevin/videos",
    "stock_curry": "https://www.youtube.com/@StockCurry/videos",
}

# Names of the corpus companies, for --company-titles. Kept in step with COMPANY_NAMES and
# PRODUCTS in finsight/agents/claim_extraction/thesaurus.py. Titles only: descriptions carry
# affiliate boilerplate (a Tesla referral link, "Apple Store" app links) on nearly every video.
_COMPANY_TITLE = re.compile(
    r"\b(Apple|AAPL|iPhone|Alphabet|Google|GOOGL?|Microsoft|MSFT|Azure|Netflix|NFLX|NVIDIA|NVDA"
    r"|Tesla|TSLA|Amazon|AMZN|AWS|Meta|Facebook|Instagram|MercadoLibre|Mercado Libre|MELI"
    r"|Starbucks|SBUX|Palantir|PLTR|Intel|INTC|Intuit|INTU|Salesforce|CRM|Oracle|ORCL|AMD)\b",
    flags=re.IGNORECASE,
)

# How far down a newest-first tab to read before stopping. Both channels post
# well under this in a three-month window.
_MAX_LISTED = 80
_ROOT = Path(__file__).resolve().parents[1]


def _cutoff(now: datetime) -> datetime:
    """Three calendar months before now, keeping the day when it exists."""
    month = now.month - 3
    year = now.year
    if month <= 0:
        month += 12
        year -= 1
    day = min(now.day, _days_in_month(year, month))
    return now.replace(year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = datetime(year + 1, 1, 1, tzinfo=UTC)
    else:
        nxt = datetime(year, month + 1, 1, tzinfo=UTC)
    return (nxt - timedelta(days=1)).day


def list_channel(url: str, limit: int = _MAX_LISTED) -> tuple[str | None, str | None, list[dict]]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": limit,
        "extractor_args": {"youtubetab": {"approximate_date": [""]}},
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
    channel = info.get("channel") or info.get("uploader") or info.get("title")
    channel_id = info.get("channel_id") or info.get("uploader_id")
    return channel, channel_id, list(info.get("entries") or [])


def _precision(published: datetime, now: datetime) -> str:
    """YouTube's tab says '3 weeks ago' or '2 months ago', not a calendar day.

    A result that lands on today's day-of-month in an earlier month came from a
    'N months ago' label. A result an exact number of weeks back came from a
    'N weeks ago' label. Newer rows are usually a specific day.
    """
    if published.day == now.day and (published.year, published.month) != (now.year, now.month):
        return "month"
    age_days = (now.date() - published.date()).days
    if age_days >= 7 and age_days % 7 == 0:
        return "week"
    return "day"


def _row(influencer: str, entry: dict, now: datetime, channel: str | None, channel_id: str | None) -> dict | None:
    timestamp = entry.get("timestamp")
    if timestamp is None:
        return None
    published = datetime.fromtimestamp(timestamp, UTC)
    return {
        "influencer": influencer,
        "video_id": entry.get("id"),
        "title": entry.get("title"),
        "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
        "published_at": published.date().isoformat(),
        "date_precision": _precision(published, now),
        "duration_seconds": entry.get("duration"),
        "view_count": entry.get("view_count"),
        "channel": entry.get("channel") or entry.get("uploader") or channel,
        "channel_id": entry.get("channel_id") or channel_id,
    }


def collect(
    now: datetime | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    names: list[str] | None = None,
    limit: int | None = None,
) -> tuple[datetime, dict[str, list[dict]]]:
    now = now or datetime.now(UTC)
    cutoff = since or _cutoff(now)
    until_s = (until + timedelta(days=7)).date().isoformat() if until else None
    # Relative labels such as "7 months ago" can place an early-April video in March.
    floor = (since - timedelta(days=45)).date().isoformat() if since else cutoff.date().isoformat()
    limit = limit or (200 if since else _MAX_LISTED)
    selected = {name: CHANNELS[name] for name in (names or list(CHANNELS))}
    by_influencer: dict[str, list[dict]] = {}
    for name, url in selected.items():
        channel, channel_id, entries = list_channel(url, limit)
        kept = []
        for entry in entries:
            row = _row(name, entry, now, channel, channel_id)
            if row is None:
                continue
            if until_s and row["published_at"] > until_s:
                continue
            if row["published_at"] < floor:
                # Newest-first. Once dates fall below the floor, the rest are older.
                break
            kept.append(row)
        by_influencer[name] = kept
    return cutoff, by_influencer


def write_run(
    cutoff: datetime,
    by_influencer: dict[str, list[dict]],
    now: datetime,
    until: datetime | None = None,
    channels: dict[str, str] | None = None,
) -> Path:
    out_dir = _ROOT / "data" / "influencers" / "youtube" / now.date().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    kept_fields = (
        "description", "upload_date", "transcript", "transcript_source", "transcript_via", "transcript_chars",
        "transcript_error",
    )
    counts = {}
    for name, rows in by_influencer.items():
        path = out_dir / f"{name}.jsonl"
        previous = {}
        if path.exists():
            previous = {
                row["video_id"]: row
                for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            }
        merged = []
        for row in rows:
            old = previous.get(row["video_id"]) or {}
            for field in kept_fields:
                if old.get(field) not in (None, ""):
                    row[field] = old[field]
            merged.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in merged), encoding="utf-8")
        counts[name] = len(merged)
    summary = {
        "collected_at": now.isoformat(timespec="seconds"),
        "cutoff_date": cutoff.date().isoformat(),
        "window": (
            f"{cutoff.date().isoformat()} through {until.date().isoformat()}, inclusive"
            if until
            else "3 calendar months, inclusive of the cutoff date"
        ),
        "until_date": until.date().isoformat() if until else None,
        "date_note": (
            "published_at comes from YouTube's relative label. "
            "date_precision 'day' is a recent label such as '4 days ago'. "
            "'week' is 'N weeks ago'. 'month' is 'N months ago', so every video in that "
            "month bucket shares one date and the July bucket can include a video from early June."
        ),
        "source": "Channel Videos tab via yt-dlp. Video pages were not opened.",
        "counts": counts,
        "channels": channels or {name: CHANNELS[name] for name in by_influencer},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out_dir


def _caption_url(info: dict) -> tuple[str | None, str | None]:
    """English caption URL. A creator-uploaded track wins over the automatic one."""
    manual = (info.get("subtitles") or {}).get("en") or []
    auto = (info.get("automatic_captions") or {}).get("en") or []
    if manual:
        tracks, source = manual, "manual"
    elif auto:
        tracks, source = auto, "automatic"
    else:
        return None, None
    json3 = next((t.get("url") for t in tracks if t.get("ext") == "json3" and t.get("url")), None)
    if json3:
        return json3, source
    any_url = next((t.get("url") for t in tracks if t.get("url")), None)
    return any_url, source


def _plain_text(payload: bytes) -> str:
    data = json.loads(payload)
    parts: list[str] = []
    for event in data.get("events") or []:
        for seg in event.get("segs") or []:
            parts.append(seg.get("utf8") or "")
    return " ".join("".join(parts).split())


def reuse_saved_transcripts(run_dir: Path) -> int:
    """Copy transcripts already stored under an earlier run directory."""
    saved: dict[str, dict] = {}
    root = run_dir.parent
    for path in sorted(root.glob("*/*.jsonl")):
        if path.parent == run_dir or path.name == "missing_transcripts.jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("video_id") and row.get("transcript"):
                saved[row["video_id"]] = row
    copied = 0
    fields = ("description", "upload_date", "transcript", "transcript_source", "transcript_via", "transcript_chars")
    for path in sorted(run_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = False
        for row in rows:
            old = saved.get(row.get("video_id"))
            if not old or row.get("transcript"):
                continue
            for field in fields:
                if old.get(field) not in (None, ""):
                    row[field] = old[field]
            row.pop("transcript_error", None)
            copied += 1
            changed = True
        if changed:
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return copied


def fetch_video_details(video_id: str) -> dict:
    """Title, description, and the real upload date. Does not request captions."""
    options = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    upload = info.get("upload_date")
    exact = None
    if upload and len(upload) == 8:
        exact = f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}"
    return {
        "title": info.get("title"),
        "description": info.get("description") or "",
        "upload_date": exact,
    }


def add_details(run_dir: Path, pause_seconds: float = 2.0) -> None:
    """Open each video page that does not yet have an exact upload date."""
    for path in sorted(run_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index, row in enumerate(rows, start=1):
            if row.get("upload_date"):
                continue
            video_id = row["video_id"]
            print(f"{path.stem} details {index}/{len(rows)} {video_id}", flush=True)
            try:
                details = fetch_video_details(video_id)
            except Exception as exc:
                error = str(exc).splitlines()[0][:300]
                print(f"  details failed {error}", flush=True)
                row["transcript_error"] = error
                path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
                if "429" in error:
                    return
                time.sleep(pause_seconds)
                continue
            for key, value in details.items():
                if value not in (None, ""):
                    row[key] = value
            path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
            print(f"  {row.get('upload_date')} {row.get('title')}", flush=True)
            time.sleep(pause_seconds)


def keep_window(run_dir: Path, since: str, until: str) -> None:
    """Drop rows whose real upload date is outside the inclusive window."""
    slack = (datetime.fromisoformat(since) - timedelta(days=31)).date().isoformat()
    for path in sorted(run_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = []
        for row in rows:
            exact = row.get("upload_date")
            approx = row.get("published_at") or ""
            if exact:
                if since <= exact <= until:
                    kept.append(row)
            elif slack <= approx <= until:
                kept.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
        print(f"{path.stem}: kept {len(kept)} of {len(rows)}", flush=True)


def _transcript_api(video_id: str) -> tuple[str, str]:
    """English captions through youtube-transcript-api. A creator-uploaded track wins.

    Raises with "429" in the message when YouTube blocks this endpoint too, so a
    run that stops on rate limits still stops.
    """
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import NoTranscriptFound, RequestBlocked

    try:
        listing = YouTubeTranscriptApi().list(video_id)
        try:
            track, source = listing.find_manually_created_transcript(["en"]), "manual"
        except NoTranscriptFound:
            track, source = listing.find_generated_transcript(["en"]), "automatic"
        text = " ".join(" ".join(snippet.text for snippet in track.fetch()).split())
    except RequestBlocked as exc:
        blocked = type(exc).__name__
        raise RuntimeError(f"HTTP Error 429 on captions; youtube-transcript-api also blocked ({blocked})") from exc
    return text, source


def fetch_spoken_content(video_id: str) -> dict:
    """Title, description, exact upload date, and the spoken transcript."""
    options = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    url, source = _caption_url(info)
    transcript = None
    via = None
    if url:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                transcript = _plain_text(response.read())
            via = "caption_track"
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            transcript, source = _transcript_api(video_id)
            via = "youtube_transcript_api"
    upload = info.get("upload_date")
    exact = None
    if upload and len(upload) == 8:
        exact = f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}"
    return {
        "title": info.get("title"),
        "description": info.get("description") or "",
        "upload_date": exact,
        "transcript": transcript,
        "transcript_source": source,
        "transcript_via": via,
        "transcript_chars": len(transcript) if transcript else 0,
    }


def add_transcripts(
    run_dir: Path,
    pause_seconds: float = 1.5,
    stop_on_rate_limit: bool = False,
    company_titles: bool = False,
    window: tuple[str, str] | None = None,
) -> None:
    """Fill transcripts for the jsonl files already written by a listing run.

    With `company_titles`, only videos whose title names a corpus company are fetched.
    With `window` (inclusive since, until), a video whose real upload date is known and
    outside it is skipped, and so is one whose rough tab date is far outside it. Fetching
    a transcript also records the real upload date, so no separate details pass is needed.
    The skipped videos stay listed without a transcript.
    """
    if window:
        slack_low = (datetime.fromisoformat(window[0]) - timedelta(days=45)).date().isoformat()
        slack_high = (datetime.fromisoformat(window[1]) + timedelta(days=7)).date().isoformat()
    names = [path.stem for path in sorted(run_dir.glob("*.jsonl"))]
    halted = False
    for name in names:
        if halted:
            break
        path = run_dir / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        done = 0
        wanted = len(rows)
        for index, row in enumerate(rows, start=1):
            video_id = row["video_id"]
            # A title can name a company without being about it (an "Amazon Prime" cargo-jet crash).
            # Such rows are marked off_topic by hand and never fetched.
            if row.get("off_topic") or (company_titles and not _COMPANY_TITLE.search(row.get("title") or "")):
                wanted -= 1
                continue
            if window:
                exact = row.get("upload_date")
                rough = row.get("published_at") or ""
                if (exact and not window[0] <= exact <= window[1]) or (
                    not exact and not slack_low <= rough <= slack_high
                ):
                    wanted -= 1
                    continue
            if row.get("transcript"):
                done += 1
                continue
            print(f"{name} {index}/{len(rows)} {video_id}", flush=True)
            spoken = None
            error = None
            attempts = 1 if stop_on_rate_limit else 4
            for attempt in range(attempts):
                try:
                    spoken = fetch_spoken_content(video_id)
                    break
                except Exception as exc:
                    error = str(exc).splitlines()[0][:300]
                    if stop_on_rate_limit and "429" in error:
                        break
                    wait = pause_seconds * (2 ** attempt)
                    print(f"  attempt {attempt + 1} failed {error}; waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
            if spoken is None:
                row["transcript_error"] = error or "unknown error"
                if stop_on_rate_limit and error and "429" in error:
                    halted = True
            else:
                row.pop("transcript_error", None)
                row.update(spoken)
                done += 1
                print(
                    f"  {row.get('upload_date')} {row.get('transcript_source')} {row.get('transcript_chars')} chars",
                    flush=True,
                )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            if halted:
                print("stopping caption downloads; YouTube returned 429", flush=True)
                break
            time.sleep(pause_seconds)
        print(f"{name}: transcripts {done}/{wanted} ({len(rows)} listed)", flush=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["source"] = (
            "Channel Videos tab for the list. Spoken words from the public English caption track, "
            "or from youtube-transcript-api when that track returned HTTP 429 (see transcript_via). "
            "upload_date is the video's real upload date."
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="List influencer videos and, with --transcripts, save what was said.")
    parser.add_argument(
        "--transcripts",
        action="store_true",
        help="After listing, download the English caption track for each video in the window.",
    )
    parser.add_argument("--since", help="Inclusive start date, YYYY-MM-DD. Overrides the 3-month cutoff.")
    parser.add_argument("--until", help="Inclusive end date, YYYY-MM-DD.")
    parser.add_argument("--channels", help="Comma-separated channel keys. Defaults to every channel.")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds between video-page requests.")
    parser.add_argument(
        "--limit", type=int, help="Videos to read from each channel tab, newest first (default 200 with --since)."
    )
    parser.add_argument(
        "--company-titles",
        action="store_true",
        help="With --transcripts, fetch only videos whose title names a corpus company.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Skip listing and fill transcripts in this existing run folder (use with --transcripts).",
    )
    args = parser.parse_args()
    if args.run_dir:
        window = (args.since, args.until) if args.since and args.until else None
        add_transcripts(
            args.run_dir, pause_seconds=max(args.pause, 8.0), stop_on_rate_limit=True,
            company_titles=args.company_titles, window=window,
        )
        if window:
            keep_window(args.run_dir, *window)
        return 0
    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
    until = datetime.fromisoformat(args.until).replace(tzinfo=UTC) if args.until else None
    names = [part.strip() for part in args.channels.split(",") if part.strip()] if args.channels else None
    if names:
        unknown = [name for name in names if name not in CHANNELS]
        if unknown:
            raise SystemExit(f"unknown channel key: {', '.join(unknown)}")
    now = datetime.now(UTC)
    cutoff, by_influencer = collect(now, since=since, until=until, names=names, limit=args.limit)
    selected = {name: CHANNELS[name] for name in by_influencer}
    out_dir = write_run(cutoff, by_influencer, now, until=until, channels=selected)
    print(f"cutoff {cutoff.date().isoformat()}")
    for name, rows in by_influencer.items():
        print(f"{name}: {len(rows)} videos")
        for row in rows:
            print(f"  {row['published_at']}  {row['date_precision']:5}  {row['title']}")
    print(f"wrote {out_dir}")
    if args.since and args.until:
        copied = reuse_saved_transcripts(out_dir)
        print(f"reused {copied} transcripts")
        add_details(out_dir, pause_seconds=args.pause)
        keep_window(out_dir, args.since, args.until)
    if args.transcripts:
        add_transcripts(
            out_dir, pause_seconds=max(args.pause, 8.0), stop_on_rate_limit=True, company_titles=args.company_titles
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
