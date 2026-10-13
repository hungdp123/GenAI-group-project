"""Build import metadata for manually downloaded official IR files.

The helper reads Windows ``Zone.Identifier`` download provenance, excludes files
already represented by the canonical manifests, assigns publication dates from
the project's verified quarterly-results calendar, and writes one metadata file
per populated ticker inbox.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path("data/narrative")
INBOX = ROOT / "manual_inbox"
MANIFESTS = (
    ROOT / "raw" / "from_2021_final" / "manifest.json",
    ROOT / "raw" / "supplemental_from_2021_official_final" / "manifest.json",
    ROOT / "raw" / "ir_from_2021_official_final" / "manifest.json",
)
FROM_DATE = date(2021, 1, 1)
CUTOFF_DATE = date(2026, 10, 4)

SOURCE_PAGES = {
    "BIRD": "https://ir.smartbird.ai/financial-information/quarterly-results",
    "DNUT": "https://investors.krispykreme.com/financial-information/quarterly-results",
    "GPRO": "https://investor.gopro.com/events-and-presentations/default.aspx",
    "MCD": "https://investor.mcdonalds.com/events-and-presentations/default.aspx",
    "MSFT": "https://www.microsoft.com/en-us/Investor/earnings/FY-{fy}-Q{quarter}/press-release-webcast",
    "NFLX": "https://ir.netflix.net/investor-news-and-events/investor-events/default.aspx",
    "NKE": "https://investors.nike.com/investors/news-events-and-reports/default.aspx",
    "OPEN": "https://investor.opendoor.com/financials-filings/quarterly-reports",
}

QUARTER_DATES: dict[str, dict[tuple[int, int], str]] = {
    "BIRD": {
        (2021, 3): "2021-11-30", (2021, 4): "2022-02-23",
        (2022, 1): "2022-05-10", (2022, 2): "2022-08-08",
        (2022, 3): "2022-11-08", (2022, 4): "2023-03-09",
        (2023, 1): "2023-05-09", (2023, 2): "2023-08-08",
        (2023, 3): "2023-11-08", (2023, 4): "2024-03-12",
        (2024, 1): "2024-05-08", (2024, 2): "2024-08-07",
        (2024, 3): "2024-11-06", (2024, 4): "2025-03-11",
        (2025, 1): "2025-05-08", (2025, 2): "2025-08-07",
        (2025, 3): "2025-11-06",
    },
    "DNUT": {
        (2021, 2): "2021-08-17", (2021, 3): "2021-11-09",
        (2024, 1): "2024-05-09", (2024, 2): "2024-08-08",
        (2024, 3): "2024-11-07", (2024, 4): "2025-02-25",
        (2025, 1): "2025-05-08", (2025, 2): "2025-08-07",
        (2025, 3): "2025-11-06", (2025, 4): "2026-02-26",
        (2026, 1): "2026-05-07", (2026, 2): "2026-08-06",
    },
    "GPRO": {
        (2021, 1): "2021-05-06", (2021, 2): "2021-08-05",
        (2021, 3): "2021-11-04", (2021, 4): "2022-02-03",
        (2022, 1): "2022-05-05", (2022, 2): "2022-08-04",
        (2022, 3): "2022-11-03", (2022, 4): "2023-02-02",
        (2023, 1): "2023-05-09", (2023, 2): "2023-08-03",
        (2023, 3): "2023-11-07", (2023, 4): "2024-02-07",
        (2024, 1): "2024-05-07", (2024, 2): "2024-08-06",
        (2024, 3): "2024-11-07", (2024, 4): "2025-02-06",
        (2025, 1): "2025-05-12", (2025, 2): "2025-08-11",
        (2025, 3): "2025-11-06", (2025, 4): "2026-03-05",
        (2026, 1): "2026-05-11", (2026, 2): "2026-08-10",
    },
    "NFLX": {
        (2020, 4): "2021-01-19", (2025, 4): "2026-01-20",
        (2026, 1): "2026-04-16", (2026, 2): "2026-07-16",
    },
    "OPEN": {
        (2021, 1): "2021-05-11", (2021, 2): "2021-08-11",
        (2021, 3): "2021-11-10", (2021, 4): "2022-02-24",
        (2022, 1): "2022-05-05", (2022, 2): "2022-08-04",
        (2022, 3): "2022-11-03", (2022, 4): "2023-02-23",
        (2023, 1): "2023-05-04", (2023, 2): "2023-08-03",
        (2023, 3): "2023-11-02", (2023, 4): "2024-02-15",
        (2024, 1): "2024-05-02", (2024, 2): "2024-08-01",
        (2024, 3): "2024-11-07", (2024, 4): "2025-02-27",
        (2025, 1): "2025-05-06", (2025, 2): "2025-08-05",
    },
}

NKE_DATES = {
    (2021, 1): "2020-09-22", (2021, 2): "2020-12-18",
    (2021, 3): "2021-03-18", (2021, 4): "2021-06-24",
    (2022, 2): "2021-12-20", (2022, 3): "2022-03-21",
    (2022, 4): "2022-06-27", (2023, 1): "2022-09-29",
    (2023, 2): "2022-12-20", (2023, 3): "2023-03-21",
    (2024, 2): "2023-12-21", (2024, 3): "2024-03-21",
    (2024, 4): "2024-06-27", (2025, 1): "2024-10-01",
    (2025, 2): "2024-12-19", (2025, 3): "2025-03-20",
    (2025, 4): "2025-06-26", (2026, 1): "2025-09-30",
    (2026, 2): "2025-12-18", (2026, 3): "2026-03-31",
    (2026, 4): "2026-06-30",
}


def _known_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in MANIFESTS:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        hashes.update(d["sha256"].lower() for d in manifest["documents"] if d.get("sha256"))
    return hashes


def _zone_url(path: Path) -> str:
    try:
        text = Path(f"{path}:Zone.Identifier").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    match = re.search(r"^HostUrl=(.+)$", text, re.MULTILINE)
    return match.group(1).strip().rstrip("\x00") if match else ""


def _quarter_year(name: str) -> tuple[int, int] | None:
    patterns = (
        r"(?i)\bQ([1-4])[- _]?(20\d{2})(?!\d)",
        r"(?i)\b([1-4])Q(20\d{2})(?!\d)",
        r"(?i)\bQ([1-4])[- _]?(\d{2})(?!\d)",
        r"(?i)\b([1-4])Q(\d{2})(?!\d)",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            quarter, year = int(match.group(1)), int(match.group(2))
            return (2000 + year if year < 100 else year, quarter)
    return None


def _date_for(ticker: str, name: str) -> str:
    prefix = re.match(r"(20\d{2})-(\d{2})-(\d{2})", name)
    if prefix:
        return prefix.group(0)

    if ticker == "MSFT":
        return name[:10]
    if ticker == "MCD":
        return name[:10]
    if ticker == "NKE":
        match = re.search(r"Q([1-4])FY(\d{2})", name, re.IGNORECASE)
        if not match:
            raise ValueError(f"Cannot infer Nike fiscal quarter: {name}")
        return NKE_DATES[(2000 + int(match.group(2)), int(match.group(1)))]
    if ticker == "NFLX":
        dated = re.search(r"(20\d{2})-(\d{2})-(\d{2})", name)
        if dated:
            return dated.group(0)

    quarter_year = _quarter_year(name)
    if quarter_year and quarter_year in QUARTER_DATES.get(ticker, {}):
        return QUARTER_DATES[ticker][quarter_year]

    overrides = {
        ("BIRD", "2022-May-10-BIRD Transcript.pdf"): "2022-05-10",
        ("BIRD", "2022-Nov-08-BIRD TRANSCRIPT.pdf"): "2022-11-08",
        ("BIRD", "Prepared Remarks.pdf"): "2024-11-06",
        ("BIRD", "Strategic Transformation Plan Update.pdf"): "2024-03-12",
        ("DNUT", "Earning presentations - kk2q21.pdf"): "2021-08-17",
        ("DNUT", "Earning presentations - DNUT_3Q21-Earnings-Deck_Final.pdf"): "2021-11-09",
        ("GPRO", "GPRO-2025-06-30-Management-Commentary.pdf"): "2025-08-11",
        ("GPRO", "GPRO-2025-09-30-Management-s-Commentary.pdf"): "2025-11-06",
        ("GPRO", "GPRO-2025-12-31-Management-Commentary.pdf"): "2026-03-05",
        ("GPRO", "GPRO_2026-6-30_Management_Commentary.pdf"): "2026-08-10",
    }
    try:
        return overrides[(ticker, name)]
    except KeyError as exc:
        raise ValueError(f"Cannot infer publication date: {ticker}/{name}") from exc


def _document_type(name: str) -> str:
    lowered = name.lower()
    if "investor_day" in lowered or "investor day" in lowered:
        return "investor_day_presentation"
    if (
        "strategic" in lowered
        or "investor_update" in lowered
        or "investor overview" in lowered
        or "investor_overview" in lowered
    ):
        return "special_presentation"
    if "presentation" in lowered or "slides" in lowered or "infographic" in lowered:
        return "earnings_presentation"
    if "transcript" in lowered or "earnings-call" in lowered or "earnings_call" in lowered:
        return "earnings_call_transcript"
    if "commentary" in lowered or "prepared remarks" in lowered or "shareholder" in lowered:
        return "prepared_remarks"
    raise ValueError(f"Cannot classify document: {name}")


def _source_page(ticker: str, name: str) -> str:
    if ticker != "MSFT":
        return SOURCE_PAGES[ticker]
    match = re.search(r"FY(\d{4})-Q([1-4])", name)
    if not match:
        raise ValueError(f"Cannot infer Microsoft source page: {name}")
    return SOURCE_PAGES[ticker].format(fy=match.group(1), quarter=match.group(2))


def main() -> int:
    known = _known_hashes()
    skipped: list[dict[str, str]] = []
    total = 0
    for ticker_dir in sorted(path for path in INBOX.iterdir() if path.is_dir()):
        ticker = ticker_dir.name.upper()
        if ticker in {"NVDA", "TSLA"}:
            continue
        records: list[dict[str, Any]] = []
        for path in sorted(ticker_dir.iterdir()):
            if not path.is_file() or path.name == "import_metadata.json":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in known:
                skipped.append({"ticker": ticker, "filename": path.name, "reason": "canonical duplicate"})
                continue
            if path.suffix.lower() == ".htm":
                skipped.append({"ticker": ticker, "filename": path.name, "reason": "Office viewer wrapper"})
                continue
            publication_date = _date_for(ticker, path.name)
            if not FROM_DATE <= date.fromisoformat(publication_date) <= CUTOFF_DATE:
                skipped.append({"ticker": ticker, "filename": path.name, "reason": "outside date window"})
                continue
            source_page = _source_page(ticker, path.name)
            source_url = _zone_url(path) or source_page
            records.append({
                "filename": path.name,
                "company_ticker": ticker,
                "document_type": _document_type(path.name),
                "publication_date": publication_date,
                "source_page": source_page,
                "source_url": source_url,
                "link_text": path.stem,
                "sha256": digest,
            })
            known.add(digest)
        if records:
            payload = {
                "source_policy": (
                    "Official company investor-relations downloads; URLs recovered from browser provenance."
                ),
                "documents": records,
            }
            (ticker_dir / "import_metadata.json").write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            total += len(records)
            print(f"Prepared {ticker}: {len(records)} records")
    (INBOX / "manual_import_skipped.json").write_text(
        json.dumps({"documents": skipped}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {total} unique in-window records; skipped {len(skipped)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
