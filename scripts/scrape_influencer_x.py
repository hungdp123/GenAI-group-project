"""Collect influencer posts from X (Twitter) for the project window.

X's free API cannot read tweets, so this uses twscrape, which signs in with a
normal X account's browser cookies. Use a spare account, not a personal one:
scraping is against X's terms and the account can be locked.

One-time setup. In a browser logged in to the spare account, copy the auth_token
and ct0 cookies (DevTools > Application > Cookies > x.com), then:

    .venv/bin/pip install twscrape
    export X_USERNAME=<spare account handle>
    export X_COOKIES="auth_token=<value>; ct0=<value>"
    .venv/bin/python scripts/scrape_influencer_x.py --add-account

Then collect (handles in HANDLES below, or --handle key=handle):

    .venv/bin/python scripts/scrape_influencer_x.py --dry-run
    .venv/bin/python scripts/scrape_influencer_x.py

Retweets and replies to other people are dropped. A thread the influencer
replies to themselves is merged into one post. The text goes in `transcript`, so
the post loader and claim extraction read it the same way as a YouTube caption
track. A tweet with a video only has its caption text here: `needs_transcription`
marks those so the spoken words can be added later.

Output goes to data/influencers/x/<run-date>/<influencer>/<YYYY-MM>.jsonl, one file
per month, written as soon as that month is searched. Rerunning with the same
--run-date skips months already saved. --from-raw splits a raw tweet file from an
earlier run into month files without searching again. The folder is gitignored.
Dates are UTC.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from twscrape import API

# Influencer key (the same keys as the YouTube scraper and the orchestrator) to X handle, without the @.
# An empty handle is skipped. Fill these in, or pass --handle key=handle.
HANDLES: dict[str, str] = {
    "joseph_carlson": "",
    "mark_roussin": "",
    "alex_reffett": "",
    "zack_morris": "MrZackMorris",
}

WINDOW_START = date(2026, 4, 1)
WINDOW_END = date(2026, 9, 30)

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "data" / "influencers" / "x"
_ACCOUNTS_DB = _OUT / "accounts.db"


def month_chunks(since: date, until: date) -> list[tuple[date, date]]:
    """[start, end) pairs, one per calendar month. X's until: is exclusive, so the last end is until + 1."""
    chunks = []
    start = since
    stop = until + timedelta(days=1)
    while start < stop:
        nxt = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
        chunks.append((start, min(nxt, stop)))
        start = nxt
    return chunks


def query(handle: str, start: date, end: date) -> str:
    """X does not cut since:/until: on UTC days, so the search runs a day wider on each side.

    The tweets are then kept by their UTC date, so a tweet near a month boundary is not lost.
    """
    since = (start - timedelta(days=1)).isoformat()
    until = (end + timedelta(days=1)).isoformat()
    return f"from:{handle} since:{since} until:{until} -filter:retweets"


def _is_own(tweet, handle: str) -> bool:
    """A standalone tweet, or a reply to the influencer's own tweet (a thread)."""
    if tweet.retweetedTweet is not None:
        return False
    if tweet.inReplyToTweetId is None:
        return True
    replied_to = tweet.inReplyToUser.username if tweet.inReplyToUser else tweet.inReplyToScreenName
    return (replied_to or "").lower() == handle.lower()


def _video_urls(tweet) -> list[str]:
    """The highest-bitrate mp4 for each video in the tweet."""
    urls = []
    for video in (tweet.media.videos if tweet.media else []):
        mp4 = [v for v in video.variants if v.contentType == "video/mp4"]
        if mp4:
            urls.append(max(mp4, key=lambda v: v.bitrate).url)
    return urls


def _title(text: str) -> str:
    first = " ".join(text.split())
    return first if len(first) <= 100 else first[:97].rstrip() + "..."


def to_posts(influencer: str, handle: str, tweets: list) -> list[dict]:
    """One row per post. Tweets in one self-thread become one post, keyed by the earliest tweet."""
    threads: dict[int, list] = {}
    for tweet in tweets:
        threads.setdefault(tweet.conversationId or tweet.id, []).append(tweet)
    rows = []
    for parts in threads.values():
        parts.sort(key=lambda t: t.id)
        root = parts[0]
        text = "\n\n".join(t.rawContent for t in parts)
        quoted = [t.quotedTweet.rawContent for t in parts if t.quotedTweet is not None]
        videos = [url for t in parts for url in _video_urls(t)]
        rows.append({
            "influencer": influencer,
            "platform": "x",
            "handle": handle,
            # Named video_id so the shared post loader keys X posts the same way as videos.
            "video_id": root.id_str,
            "post_id": root.id_str,
            "tweet_ids": [t.id_str for t in parts],
            "title": _title(root.rawContent),
            "url": root.url,
            "upload_date": root.date.astimezone(UTC).date().isoformat(),
            "published_at": root.date.astimezone(UTC).date().isoformat(),
            "posted_at_utc": root.date.astimezone(UTC).isoformat(timespec="seconds"),
            "date_precision": "day",
            "description": "",
            # The quoted tweet is someone else's words, so it stays out of the transcript.
            "transcript": text,
            "transcript_source": "x_text",
            "transcript_chars": len(text),
            "quoted_text": quoted,
            "cashtags": sorted({tag for t in parts for tag in (t.cashtags or [])}),
            "is_thread": len(parts) > 1,
            "has_video": bool(videos),
            "needs_transcription": bool(videos),
            "video_urls": videos,
            "like_count": root.likeCount,
            "retweet_count": root.retweetCount,
            "view_count": root.viewCount,
        })
    rows.sort(key=lambda r: r["posted_at_utc"])
    return rows


def _month(day: date) -> str:
    return day.isoformat()[:7]


def _write_atomic(path: Path, lines: list[str]) -> None:
    """A month file only appears once the month is complete, so a rerun can trust it."""
    part = path.with_suffix(".part")
    part.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    part.replace(path)


async def collect(api: API, influencer: str, handle: str, since: date, until: date, out_dir: Path) -> dict[str, int]:
    """Search one month at a time and write that month's posts as soon as it finishes.

    A month whose posts file already exists is skipped, so an interrupted run resumes
    where it stopped. A thread that crosses a month boundary is split at the boundary.
    """
    posts_dir = out_dir / influencer
    raw_dir = out_dir / "raw" / influencer
    posts_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for start, end in month_chunks(since, until):
        month = _month(start)
        posts_path = posts_dir / f"{month}.jsonl"
        if posts_path.exists():
            counts[month] = sum(1 for line in posts_path.read_text(encoding="utf-8").splitlines() if line.strip())
            print(f"{influencer} {month}: already saved, {counts[month]} posts", flush=True)
            continue
        seen: dict[int, object] = {}
        async for tweet in api.search(query(handle, start, end)):
            seen.setdefault(tweet.id, tweet)
        _write_atomic(raw_dir / f"{month}.jsonl", [t.json() for t in seen.values()])
        rows = month_posts(influencer, handle, list(seen.values()), start, end)
        _write_atomic(posts_path, [json.dumps(row) for row in rows])
        counts[month] = len(rows)
        print(f"{influencer} {month}: {len(seen)} tweets, {len(rows)} posts", flush=True)
    return counts


def month_posts(influencer: str, handle: str, tweets: list, start: date, end: date) -> list[dict]:
    """Posts from the influencer's own tweets dated in [start, end)."""
    kept = [t for t in tweets if _is_own(t, handle) and start <= t.date.astimezone(UTC).date() < end]
    return to_posts(influencer, handle, kept)


def _ns(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ns(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_ns(item) for item in value]
    return value


def split_raw(raw_path: Path, influencer: str, since: date, until: date, out_dir: Path) -> dict[str, int]:
    """Rebuild per-month post files from a raw tweet file saved by an earlier run.

    That run searched month by month, newest first, so each search shows up as a run
    of falling dates and the next search starts with a jump forward in time. Every
    search except the last one finished. The last one counts as finished only if the
    whole run did, which it marks by writing <run-date>/<influencer>.jsonl. A month
    that is not finished is not written, so a normal run with the same --run-date
    searches it again.
    """
    tweets = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tweet = _ns(json.loads(line))
            tweet.date = datetime.fromisoformat(tweet.date)
            tweets.append(tweet)
    if not tweets:
        return {}
    handle = tweets[0].user.username
    posts_dir = out_dir / influencer
    posts_dir.mkdir(parents=True, exist_ok=True)
    # Search also returns a few old tweets (e.g. a pinned one), so only tweets near the window count.
    floor = since - timedelta(days=7)
    searches: list[list[str]] = []
    previous = None
    for tweet in tweets:
        day = tweet.date.astimezone(UTC).date()
        if not floor <= day <= until + timedelta(days=7):
            continue
        if previous is None or day > previous + timedelta(days=1):
            searches.append([])
        searches[-1].append(_month(day))
        previous = day
    months = [Counter(search).most_common(1)[0][0] for search in searches]
    complete = set(months[:-1])
    if months and (raw_path.parents[1] / f"{influencer}.jsonl").exists():
        complete.add(months[-1])
    counts = {}
    for start, end in month_chunks(since, until):
        if _month(start) not in complete:
            print(f"{influencer} {_month(start)}: not complete in the raw file, not saved", flush=True)
            continue
        rows = month_posts(influencer, handle, tweets, start, end)
        _write_atomic(posts_dir / f"{_month(start)}.jsonl", [json.dumps(row) for row in rows])
        counts[_month(start)] = len(rows)
        print(f"{influencer} {_month(start)}: {len(rows)} posts", flush=True)
    return counts


def write_summary(counts: dict[str, dict[str, int]], handles: dict[str, str], since: date, until: date, out_dir: Path) -> None:
    path = out_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summary.update({
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": f"{since.isoformat()} through {until.isoformat()}, inclusive, UTC dates",
        "layout": "<influencer>/<YYYY-MM>.jsonl posts; raw/<influencer>/<YYYY-MM>.jsonl tweets as returned",
        "source": "X search (Latest) via twscrape, one query per month. Retweets and replies to others dropped; "
        "self-threads merged into one post, split at month boundaries.",
    })
    summary.setdefault("posts_per_month", {}).update(counts)
    summary.setdefault("handles", {}).update({name: handles.get(name, "") for name in counts})
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


async def add_account(api: API) -> None:
    username = os.getenv("X_USERNAME")
    cookies = os.getenv("X_COOKIES")
    if not username or not cookies:
        sys.exit("Set X_USERNAME and X_COOKIES (\"auth_token=...; ct0=...\") first.")
    # With cookies, twscrape skips the login flow, so the password and email fields are unused.
    await api.pool.add_account(username, "-", "-", "-", cookies=cookies)
    for info in await api.pool.accounts_info():
        print(info)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--add-account", action="store_true", help="store the account from X_USERNAME / X_COOKIES")
    parser.add_argument("--handle", action="append", default=[], metavar="KEY=HANDLE", help="set or override a handle")
    parser.add_argument("--names", nargs="+", help="only these influencer keys")
    parser.add_argument("--since", type=date.fromisoformat, default=WINDOW_START)
    parser.add_argument("--until", type=date.fromisoformat, default=WINDOW_END, help="inclusive")
    parser.add_argument("--dry-run", action="store_true", help="print the search queries and stop")
    parser.add_argument("--run-date", default=datetime.now(UTC).date().isoformat(),
                        help="output folder name; reuse an earlier date to resume that run (default: today, UTC)")
    parser.add_argument("--from-raw", type=Path, metavar="RAW_JSONL",
                        help="split a raw tweet file from an earlier run into month files, then stop")
    args = parser.parse_args()

    handles = dict(HANDLES)
    for item in args.handle:
        key, _, handle = item.partition("=")
        if not handle:
            parser.error(f"--handle needs KEY=HANDLE, got {item!r}")
        handles[key.strip()] = handle.strip().lstrip("@")
    selected = {k: v for k, v in handles.items() if v and (not args.names or k in args.names)}
    missing = [k for k in (args.names or handles) if not handles.get(k)]
    if missing:
        print(f"no X handle for: {', '.join(missing)} (skipped)", flush=True)

    if args.dry_run:
        for name, handle in selected.items():
            for start, end in month_chunks(args.since, args.until):
                print(f"{name}: {query(handle, start, end)}")
        return

    out_dir = _OUT / args.run_date
    if args.from_raw:
        influencer = args.names[0] if args.names else args.from_raw.stem
        counts = split_raw(args.from_raw, influencer, args.since, args.until, out_dir)
        write_summary({influencer: counts}, handles, args.since, args.until, out_dir)
        print(f"wrote {out_dir / influencer}", flush=True)
        return

    _OUT.mkdir(parents=True, exist_ok=True)
    api = API(str(_ACCOUNTS_DB))
    if args.add_account:
        await add_account(api)
        return
    if not selected:
        sys.exit("No handles to collect. Fill HANDLES or pass --handle key=handle.")

    for name, handle in selected.items():
        counts = await collect(api, name, handle, args.since, args.until, out_dir)
        write_summary({name: counts}, handles, args.since, args.until, out_dir)
        print(f"{name}: {sum(counts.values())} posts", flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
