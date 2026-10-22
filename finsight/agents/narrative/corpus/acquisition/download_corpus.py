"""Download FinSight's frozen SEC narrative corpus.

Downloads, per ticker and as of a cutoff date:
- latest Form 10-K;
- latest two Forms 10-Q;
- latest earnings release filed as an EX-99.1 exhibit to an Item 2.02 Form 8-K.

The output directory is intentionally under the gitignored top-level ``data/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_TICKERS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "NKE",
    "NVDA",
    "MCD",
    "TSLA",
    "NFLX",
    "OPEN",
    "KSS",
    "DNUT",
    "GPRO",
    "BIRD",
    "GME",
    "AMC",
    "AMZN",
    "META",
    "MELI",
    "SBUX",
    "PLTR",
    "INTC",
    "INTU",
    "CRM",
    "ORCL",
    "AMD",
]
SEC_DATA = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"


@dataclass
class ManifestEntry:
    document_id: str
    company_name: str
    company_ticker: str
    cik: str
    document_type: str
    form: str
    period: str | None
    publication_date: str
    accession_number: str
    source_tier: int
    source_url: str
    local_path: str
    sha256: str


class SecClient:
    def __init__(
        self,
        user_agent: str,
        delay: float = 0.15,
        timeout: float = 60,
        attempts: int = 5,
    ) -> None:
        self.user_agent = user_agent
        self.delay = delay
        self.timeout = timeout
        self.attempts = attempts

    def get(self, url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"},
        )
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                time.sleep(self.delay)
                return payload
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"Unable to download {url}: {last_error}")

    def json(self, url: str) -> dict[str, Any]:
        return json.loads(self.get(url).decode("utf-8"))


def _filings(submissions: dict[str, Any]) -> list[dict[str, Any]]:
    recent = submissions.get("filings", {}).get("recent", submissions)
    keys = ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument", "items")
    return [dict(zip(keys, values, strict=True)) for values in zip(*(recent[k] for k in keys), strict=True)]


def _all_filings(client: SecClient, submissions: dict[str, Any], from_date: date | None) -> list[dict[str, Any]]:
    """Return recent filings plus SEC historical submission shards needed for the range."""
    result = _filings(submissions)
    if from_date is None:
        return result
    for shard in submissions.get("filings", {}).get("files", []):
        date.fromisoformat(shard["filingFrom"])  # Validate the range start without retaining an unused value.
        shard_end = date.fromisoformat(shard["filingTo"])
        if shard_end < from_date:
            continue
        payload = client.json(f"{SEC_DATA}/submissions/{shard['name']}")
        result.extend(_filings(payload))
    seen: set[str] = set()
    return [f for f in result if not (f["accessionNumber"] in seen or seen.add(f["accessionNumber"]))]


def _archive_base(cik: str, accession: str) -> str:
    return f"{SEC_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}"


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _save_document(
    client: SecClient,
    root: Path,
    company: dict[str, str],
    filing: dict[str, Any],
    document_type: str,
    source_url: str,
    filename: str,
    corpus_folder: str | None = None,
) -> ManifestEntry:
    ticker = company["ticker"]
    payload = client.get(source_url)
    target_dir = root / ticker
    if corpus_folder:
        target_dir /= corpus_folder
    target_dir.mkdir(parents=True, exist_ok=True)
    accession = filing["accessionNumber"]
    target = target_dir / _safe_name(f"{filing['filingDate']}_{document_type}_{accession}_{filename}")
    target.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    return ManifestEntry(
        document_id=f"{ticker}-{accession}-{document_type}",
        company_name=company["name"],
        company_ticker=ticker,
        cik=company["cik"],
        document_type=document_type,
        form=filing["form"],
        period=filing.get("reportDate") or None,
        publication_date=filing["filingDate"],
        accession_number=accession,
        source_tier=1 if filing["form"] in {"10-K", "10-Q"} else 2,
        source_url=source_url,
        local_path=target.as_posix(),
        sha256=digest,
    )


def _earnings_exhibit(client: SecClient, cik: str, filing: dict[str, Any]) -> tuple[str, str] | None:
    base = _archive_base(cik, filing["accessionNumber"])
    index = client.json(f"{base}/index.json")
    items = index.get("directory", {}).get("item", [])
    candidates: list[tuple[int, int, str]] = []
    for item in items:
        name = item.get("name", "")
        if not name.lower().endswith((".htm", ".html", ".txt", ".pdf")):
            continue
        lowered = name.lower()
        if name == filing.get("primaryDocument") or lowered.startswith("r") and lowered[1:2].isdigit():
            continue
        score = 0
        if re.search(r"(?:ex|exhibit)[-_]?99[-_.]?1(?:\D|$)", lowered):
            score += 6
        if "ex99" in lowered or "ex-99" in lowered or "exhibit99" in lowered:
            score += 3
        if "earn" in lowered or "release" in lowered or "press" in lowered:
            score += 2
        if lowered.endswith((".htm", ".html")):
            score += 1
        size_text = str(item.get("size") or "0")
        candidates.append((score, int(size_text) if size_text.isdigit() else 0, name))
    if not candidates:
        return None
    filename = sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))[0][2]
    return filename, f"{base}/{filename}"


def download_company(
    client: SecClient,
    company: dict[str, str],
    cutoff: date,
    root: Path,
    from_date: date | None = None,
    corpus_folder: str | None = None,
) -> tuple[list[ManifestEntry], list[str]]:
    submissions = client.json(f"{SEC_DATA}/submissions/CIK{int(company['cik']):010d}.json")
    filings = [
        f
        for f in _all_filings(client, submissions, from_date)
        if (from_date is None or date.fromisoformat(f["filingDate"]) >= from_date)
        and date.fromisoformat(f["filingDate"]) <= cutoff
    ]
    filings.sort(key=lambda f: (f["filingDate"], f["accessionNumber"]), reverse=True)
    selected = []
    annual = [f for f in filings if f["form"] == "10-K"]
    quarters = [f for f in filings if f["form"] == "10-Q"]
    if from_date is None:
        if annual:
            selected.append((annual[0], "10-K"))
        selected.extend((f, "10-Q") for f in quarters[:2])
    else:
        selected.extend((f, "10-K") for f in annual)
        selected.extend((f, "10-Q") for f in quarters)

    entries: list[ManifestEntry] = []
    warnings: list[str] = []
    for filing, doc_type in selected:
        filename = filing["primaryDocument"]
        url = f"{_archive_base(company['cik'], filing['accessionNumber'])}/{filename}"
        entries.append(_save_document(client, root, company, filing, doc_type, url, filename, corpus_folder))

    earnings_filings = [f for f in filings if f["form"] == "8-K" and "2.02" in (f.get("items") or "")]
    for filing in earnings_filings:
        exhibit = _earnings_exhibit(client, company["cik"], filing)
        if exhibit:
            filename, url = exhibit
            entries.append(
                _save_document(client, root, company, filing, "earnings_release", url, filename, corpus_folder)
            )
            if from_date is None:
                break
    if not any(entry.document_type == "earnings_release" for entry in entries):
        warnings.append(f"{company['ticker']}: no Item 2.02 earnings-release exhibit found")

    if not annual:
        warnings.append(f"{company['ticker']}: no 10-K found")
    if len(quarters) < 2:
        warnings.append(f"{company['ticker']}: only {len(quarters)} 10-Q filing(s) found")
    return entries, warnings


def company_map(client: SecClient, tickers: list[str]) -> list[dict[str, str]]:
    raw = client.json("https://www.sec.gov/files/company_tickers.json")
    by_ticker = {
        row["ticker"].upper(): {"ticker": row["ticker"].upper(), "name": row["title"], "cik": str(row["cik_str"])}
        for row in raw.values()
    }
    missing = [ticker for ticker in tickers if ticker not in by_ticker]
    if missing:
        raise ValueError(f"Tickers not found in SEC mapping: {', '.join(missing)}")
    return [by_ticker[ticker] for ticker in tickers]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2026, 9, 30))
    parser.add_argument(
        "--from-date",
        type=date.fromisoformat,
        help="When set, download every 10-K, 10-Q and earnings release in the inclusive date range.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Corpus folder under data/narrative/raw (e.g. "
            "data/narrative/raw/core_sec). For canonical raw output, "
            "files go in raw/<TICKER>/<folder>/ and the manifest is written "
            "directly in raw/ with a dataset-specific name."
        ),
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--user-agent",
        default=os.getenv("SEC_USER_AGENT", "FinSight academic project finsight-project@example.com"),
        help="SEC-compliant application/contact identifier; SEC_USER_AGENT overrides the default.",
    )
    args = parser.parse_args()
    client = SecClient(args.user_agent)
    tickers = [ticker.upper() for ticker in args.tickers]
    companies = company_map(client, tickers)
    output_text = args.output.as_posix().rstrip("/")
    if args.output.parent.name == "raw":
        raw_root = args.output.parent
        corpus_folder = args.output.name
    elif output_text == "data/narrative":
        raw_root = args.output / "raw"
        corpus_folder = "aggregate"
    else:
        raw_root = args.output / "raw"
        corpus_folder = None
    entries: list[ManifestEntry] = []
    warnings: list[str] = []
    for company in companies:
        print(f"Downloading {company['ticker']} ({company['name']})...")
        company_entries, company_warnings = download_company(
            client, company, args.cutoff, raw_root, args.from_date, corpus_folder
        )
        entries.extend(company_entries)
        warnings.extend(company_warnings)

    manifest = {
        "schema_version": "1.0",
        "cutoff_date": args.cutoff.isoformat(),
        "from_date": args.from_date.isoformat() if args.from_date else None,
        "generated_on": date.today().isoformat(),
        "documents": [asdict(entry) for entry in entries],
        "warnings": warnings,
    }
    if args.output.parent.name == "raw" and args.output.parent.parent.name == "narrative":
        manifest_name = {
            "core_sec": "manifest_core_sec.json",
        }.get(args.output.name, f"manifest_{args.output.name}.json")
        manifest_path = args.output.parent / manifest_name
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(entries)} documents; manifest: {manifest_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
