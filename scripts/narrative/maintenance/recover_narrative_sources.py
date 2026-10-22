"""Stage missing public calls and recover exact-hash official URLs.

Run without --apply to download/review candidates; --apply imports verified rows.
The cached stage report is a review snapshot, not proof that indexes were rebuilt.
Existing source files and document IDs are preserved. No model calls are made.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "tmp/source_gap_recovery"
STAGE.mkdir(parents=True, exist_ok=True)
PREFIX = "https://www.fool.com/earnings/call-transcripts/"
CALLS = [
    ("AAPL", "FY2024_Q1", "2024-02-01", "https://sixcolors.com/post/2024/02/this-is-tim-transcript-of-apples-q1-2024-analyst-call/"),
    ("AAPL", "FY2024_Q2", "2024-05-02", "https://sixcolors.com/post/2024/05/this-is-tim-transcript-of-apples-q2-2024-analyst-call/"),
    ("AAPL", "FY2024_Q3", "2024-08-01", "https://sixcolors.com/post/2024/08/this-is-tim-q3-2024-analyst-call-transcript/"),
    ("AMZN", "FY2023_Q4", "2024-02-01", "2024/02/01/amazoncom-amzn-q4-2023-earnings-call-transcript/"),
    ("AMZN", "FY2024_Q1", "2024-04-30", "2024/04/30/amazoncom-amzn-q1-2024-earnings-call-transcript/"),
    ("AMZN", "FY2024_Q2", "2024-08-01", "2024/08/01/amazoncom-amzn-q2-2024-earnings-call-transcript/"),
    ("INTC", "FY2023_Q4", "2024-01-25", "2024/01/25/intel-intc-q4-2023-earnings-call-transcript/"),
    ("MELI", "FY2023_Q4", "2024-02-22", "2024/02/23/mercadolibre-meli-q4-2023-earnings-call-transcript/"),
    ("MELI", "FY2024_Q1", "2024-05-02", "2024/05/03/mercadolibre-meli-q1-2024-earnings-call-transcript/"),
    ("MELI", "FY2024_Q2", "2024-08-01", "2024/08/02/mercadolibre-meli-q2-2024-earnings-call-transcript/"),
    ("PLTR", "FY2023_Q4", "2024-02-05", "2024/02/05/palantir-technologies-pltr-q4-2023-earnings-call-t/"),
    ("PLTR", "FY2024_Q1", "2024-05-06", "2024/05/06/palantir-technologies-pltr-q1-2024-earnings-call-t/"),
    ("PLTR", "FY2024_Q2", "2024-08-05", "2024/08/05/palantir-technologies-pltr-q2-2024-earnings-call-t/"),
    ("TSLA", "FY2023_Q4", "2024-01-24", "2024/01/24/tesla-tsla-q4-2023-earnings-call-transcript/"),
]


def fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(20_000_000), response.geturl()


def stage_call(item: tuple) -> dict:
    ticker, period, call_date, suffix = item
    url = suffix if suffix.startswith("https://") else PREFIX + suffix
    sixcolors = url.startswith("https://sixcolors.com/")
    row = dict(company_ticker=ticker, period=period, earnings_call_date=call_date,
               source_url=url, status="failed")
    cache = STAGE / f"{ticker}_{period}.html"
    try:
        if cache.exists():
            data, final = cache.read_bytes(), url
        else:
            data, final = fetch(url)
        cache.write_bytes(data)
        soup = BeautifulSoup(data, "html.parser")
        title = soup.h1.get_text(" ", strip=True) if soup.h1 else ""
        body = soup.select_one(".column2" if sixcolors else ".article-body.transcript-content")
        text = body.get_text(" ", strip=True) if body else ""
        year, quarter = re.fullmatch(r"FY(\d{4})_Q([1-4])", period).groups()
        correct_title = ("Apple" in soup.get_text(" ", strip=True)[:15000] and f"Q{quarter} {year}" in title and "transcript" in title.lower()) if sixcolors else (ticker in title and bool(re.search(rf"Q{quarter}\s+{year}\s+Earnings Call Transcript", title, re.I)))
        if not correct_title:
            raise ValueError(f"Wrong transcript identity: {title}")
        if len(text) < 15000 or not re.search(r"Questions and Answers|Q&A", text, re.I):
            raise ValueError("Full transcript body / Q&A not confirmed")
        head_date = re.search(r"Earnings Call\s+([A-Z][a-z]+ \d{1,2}, \d{4})", text[:800])
        if not sixcolors and (not head_date or datetime.strptime(head_date[1], "%b %d, %Y").date().isoformat() != call_date):
            raise ValueError("Printed call date differs from expected call date")
        published = soup.select_one('meta[property="article:published_time"]')
        if not published:
            raise ValueError("Publisher publication timestamp missing")
        timestamp = published["content"]
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)
        modified = soup.select_one('meta[property="article:modified_time"]') if sixcolors else None
        modified_timestamp = modified.get("content") if modified else None
        available = max(dt, datetime.fromisoformat(modified_timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)) if modified_timestamp else dt
        day = datetime.fromisoformat(call_date).date()
        if sixcolors and dt.date() != day:
            raise ValueError("Publisher article date differs from reviewed call date")
        if not day <= dt.date() <= day + timedelta(days=7):
            raise ValueError("Unexpected publisher date")
        cache.write_bytes(data)
        row.update(status="verified", title=title, transcript_characters=len(text),
                   publisher_publication_timestamp=timestamp, publication_date=available.date().isoformat(),
                   publisher_modified_timestamp=modified_timestamp,
                   publisher="Six Colors" if sixcolors else "The Motley Fool",
                   source_url=final, local_path=cache.relative_to(ROOT).as_posix(),
                   sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    print(ticker, period, row["status"], row.get("error", ""), flush=True)
    return row


def official_recoveries() -> list[dict]:
    matches = []
    for name in ["remote_report.json", "netflix_final_matches.json"]:
        path = ROOT / "tmp/url_recovery" / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            matches.extend(payload.get("matches", []) if isinstance(payload, dict) else payload)
    choices = {}
    for match in sorted(matches, key=lambda r: ("q4cdn.com" in r["url"], r["url"])):
        choices.setdefault(match["sha256"], match)
    manifest = json.loads((ROOT / "data/narrative/raw/manifest_official_ir.json").read_text(encoding="utf-8"))
    result = []
    for record in manifest["documents"]:
        if record.get("source_url"):
            continue
        match = choices.get(record["sha256"])
        if not match:
            continue
        local_hash = hashlib.sha256((ROOT / record["local_path"]).read_bytes()).hexdigest()
        cache = ROOT / "tmp/url_recovery" / (hashlib.sha256(match["url"].encode()).hexdigest() + ".bin")
        # Earlier remote downloads are evidence only if the actual cached bytes also match.
        if local_hash != record["sha256"] or not cache.exists() or hashlib.sha256(cache.read_bytes()).hexdigest() != local_hash:
            raise ValueError(f"Unverified cached URL recovery: {record['document_id']}")
        result.append(dict(document_id=record["document_id"], sha256=local_hash,
                           source_url=match["url"], source_page=match["source_page"],
                           method="remote_pdf_exact_sha256_from_cached_download"))
    return result


def apply(report: dict) -> None:
    raw = ROOT / "data/narrative/raw"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "data/narrative/metadata_backups"
    backup.mkdir(parents=True, exist_ok=True)
    official_path = raw / "manifest_official_ir.json"
    third_path = raw / "manifest_third_party_transcripts.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    third = json.loads(third_path.read_text(encoding="utf-8"))
    recovered = {r["document_id"]: r for r in report["official_url_recoveries"]}
    for record in official["documents"]:
        if record["document_id"] in recovered:
            match = recovered[record["document_id"]]
            record.update(source_url=match["source_url"], source_page=match["source_page"],
                          document_url_status="Recovered from exact SHA-256 match to official remote PDF",
                          source_url_recovery_method=match["method"])
    known_periods = {(r["company_ticker"], r.get("period")) for r in third["documents"]}
    imported = []
    for row in report["calls"]:
        if row["status"] != "verified" or (row["company_ticker"], row["period"]) in known_periods:
            continue
        ticker, digest = row["company_ticker"], row["sha256"]
        source = ROOT / row["local_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError("Staged transcript hash changed")
        dest = raw / ticker / "third_party_transcripts" / f"{row['earnings_call_date']}_earnings_call_transcript_{ticker}_{digest[:12]}.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
            raise ValueError("Import destination collision")
        shutil.copy2(source, dest)
        record = {k: row[k] for k in ["company_ticker", "period", "earnings_call_date", "publication_date", "source_url", "sha256", "publisher_publication_timestamp", "publisher_modified_timestamp"]}
        record.update(document_id=f"{ticker}-TP-{digest[:16]}", document_type="earnings_call_transcript",
                      source_tier=3, publisher=row["publisher"], source_origin="Public third-party transcript, user-authorized",
                      source_page=row["source_url"], local_path=dest.relative_to(ROOT).as_posix(),
                      original_local_path=row["local_path"], publication_date_is_estimated=False,
                      date_precision="day", retrieval_eligible=True, imported_on="2026-10-17",
                      publication_date_basis="Later of publisher article:published_time and article:modified_time (UTC date) for this saved revision; original article timestamp and call date retained separately.",
                      document_url_status="Verified full public transcript; company, fiscal period, printed call date and Q&A confirmed")
        third["documents"].append(record)
        known_periods.add((ticker, row["period"]))
        imported.append(record["document_id"])
    third["documents"].sort(key=lambda r: (r["company_ticker"], r.get("earnings_call_date", "")))
    for path, payload in [(official_path, official), (third_path, third)]:
        shutil.copy2(path, backup / f"{path.stem}.pre_source_recovery_{stamp}.json")
        staged = path.with_suffix(".recovery_tmp")
        staged.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged.replace(path)
    report["imported_document_ids"] = imported


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Import previously staged verified records")
    args = parser.parse_args()
    path = STAGE / "report.json"
    if args.apply:
        report = json.loads(path.read_text(encoding="utf-8"))
        apply(report)
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            calls = list(pool.map(stage_call, CALLS))
        report = {"created_at": datetime.now(timezone.utc).isoformat(), "calls": calls,
                  "official_url_recoveries": official_recoveries()}
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified_calls": sum(r["status"] == "verified" for r in report["calls"]),
                      "failed_calls": [r for r in report["calls"] if r["status"] != "verified"],
                      "official_urls": len(report["official_url_recoveries"]),
                      "imported": len(report.get("imported_document_ids", []))}, indent=2))
