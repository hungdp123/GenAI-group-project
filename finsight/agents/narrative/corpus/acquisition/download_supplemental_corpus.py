"""Download official supplemental narrative materials from SEC-filed exhibits.

This complements the core 10-K/10-Q/earnings-release corpus with presentation
decks, prepared remarks or transcripts, investor-day decks, and special investor
presentations. Only official SEC-hosted exhibits are accepted by this downloader.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from finsight.agents.narrative.corpus.acquisition.download_corpus import (
    DEFAULT_TICKERS,
    SEC_DATA,
    SecClient,
    _all_filings,
    _archive_base,
    _safe_name,
    company_map,
)


@dataclass
class SupplementalEntry:
    document_id: str
    company_name: str
    company_ticker: str
    cik: str
    document_type: str
    form: str
    period: str | None
    publication_date: str
    accession_number: str
    form_items: str
    source_tier: int
    source_origin: str
    source_url: str
    local_path: str
    sha256: str
    classification_basis: str


def _plain_text(payload: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        return ""
    text = payload.decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _classify(filename: str, text: str, form_items: str) -> tuple[str, str] | None:
    sample = f"{filename} {text[:20000]}".lower()
    name = filename.lower()

    rules: list[tuple[str, tuple[str, ...]]] = [
        ("earnings_call_transcript", ("earnings call transcript", "conference call transcript", "call transcript")),
        ("prepared_remarks", (
            "prepared remarks", "earnings call remarks", "conference call remarks", "shareholder letter",
        )),
        ("investor_day_presentation", (
            "investor day", "investors day", "analyst day", "capital markets day", "investor meeting presentation",
        )),
        ("earnings_presentation", (
            "earnings presentation", "earnings slides", "results presentation", "quarterly presentation",
            "quarterly results presentation", "financial results presentation",
        )),
    ]
    for document_type, phrases in rules:
        for phrase in phrases:
            if phrase in sample:
                return document_type, f"matched phrase: {phrase}"

    presentation_name = any(token in name for token in ("presentation", "slides", "deck"))
    special_phrase = next((phrase for phrase in (
        "investor presentation", "business update", "strategic update", "company presentation",
        "investor update", "financial outlook presentation",
    ) if phrase in sample), None)

    if "2.02" in form_items and presentation_name:
        return "earnings_presentation", "Item 2.02 exhibit with presentation/slide filename"
    if "7.01" in form_items and (presentation_name or special_phrase):
        basis = f"matched phrase: {special_phrase}" if special_phrase else "Item 7.01 presentation exhibit"
        return "special_presentation", basis
    return None


def _candidate_exhibits(client: SecClient, cik: str, filing: dict[str, Any]) -> list[tuple[str, str, bytes]]:
    base = _archive_base(cik, filing["accessionNumber"])
    index = client.json(f"{base}/index.json")
    results = []
    for item in index.get("directory", {}).get("item", []):
        filename = item.get("name", "")
        lowered = filename.lower()
        if filename == filing.get("primaryDocument"):
            continue
        if "-index" in lowered or lowered.endswith("index.html") or lowered.endswith("index.htm"):
            continue
        accession_text = filing["accessionNumber"]
        if lowered in {f"{accession_text}.txt", f"{accession_text.replace('-', '')}.txt"}:
            continue
        if not lowered.endswith((".htm", ".html", ".txt", ".pdf")):
            continue
        if lowered.startswith("r") and lowered[1:2].isdigit():
            continue
        if any(token in lowered for token in ("_cal.", "_def.", "_lab.", "_pre.", "_ins.")):
            continue
        url = f"{base}/{filename}"
        results.append((filename, url, client.get(url)))
    return results


def download_company(
    client: SecClient,
    company: dict[str, str],
    from_date: date,
    cutoff: date,
    raw_root: Path,
    excluded_hashes: set[str],
    corpus_folder: str | None = None,
) -> list[SupplementalEntry]:
    submissions = client.json(f"{SEC_DATA}/submissions/CIK{int(company['cik']):010d}.json")
    filings = [
        filing for filing in _all_filings(client, submissions, from_date)
        if filing["form"] in {"8-K", "8-K/A"}
        and from_date <= date.fromisoformat(filing["filingDate"]) <= cutoff
        and ("2.02" in (filing.get("items") or "") or "7.01" in (filing.get("items") or ""))
    ]
    filings.sort(key=lambda filing: (filing["filingDate"], filing["accessionNumber"]), reverse=True)
    entries: list[SupplementalEntry] = []
    seen_hashes: set[str] = set()
    for filing in filings:
        for filename, url, payload in _candidate_exhibits(client, company["cik"], filing):
            classification = _classify(filename, _plain_text(payload, filename), filing.get("items") or "")
            if classification is None:
                continue
            document_type, basis = classification
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen_hashes or digest in excluded_hashes:
                continue
            seen_hashes.add(digest)
            target_dir = raw_root / company["ticker"]
            if corpus_folder:
                target_dir /= corpus_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / _safe_name(
                f"{filing['filingDate']}_{document_type}_{filing['accessionNumber']}_{filename}"
            )
            target.write_bytes(payload)
            entries.append(SupplementalEntry(
                document_id=f"{company['ticker']}-{filing['accessionNumber']}-{document_type}-{filename}",
                company_name=company["name"],
                company_ticker=company["ticker"],
                cik=company["cik"],
                document_type=document_type,
                form=filing["form"],
                period=filing.get("reportDate") or None,
                publication_date=filing["filingDate"],
                accession_number=filing["accessionNumber"],
                form_items=filing.get("items") or "",
                source_tier=2,
                source_origin="SEC EDGAR official filing exhibit",
                source_url=url,
                local_path=target.as_posix(),
                sha256=digest,
                classification_basis=basis,
            ))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2021, 10, 2))
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2026, 10, 2))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/narrative/raw/supplemental_sec"),
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        nargs="+",
        default=[
            Path("data/narrative/raw/manifest_core_sec.json"),
            Path("data/narrative/raw/manifest_official_ir.json"),
        ],
        help="Manifests whose document hashes should not be duplicated.",
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument(
        "--user-agent",
        default=os.getenv("SEC_USER_AGENT", "FinSight academic project finsight-project@example.com"),
    )
    args = parser.parse_args()
    client = SecClient(args.user_agent)
    companies = company_map(client, [ticker.upper() for ticker in args.tickers])
    entries: list[SupplementalEntry] = []
    excluded_hashes: set[str] = set()
    for manifest_path in args.exclude_manifest:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            excluded_hashes.update(document["sha256"] for document in manifest.get("documents", []))
    if args.output.parent.name == "raw" and args.output.parent.parent.name == "narrative":
        raw_root = args.output.parent
        corpus_folder = args.output.name
    else:
        raw_root = args.output / "raw"
        corpus_folder = None
    for company in companies:
        print(f"Scanning {company['ticker']} ({company['name']})...")
        entries.extend(download_company(
            client, company, args.from_date, args.cutoff, raw_root, excluded_hashes, corpus_folder
        ))

    counts = Counter(entry.document_type for entry in entries)
    coverage = {}
    for company in companies:
        company_entries = [entry for entry in entries if entry.company_ticker == company["ticker"]]
        present = {entry.document_type for entry in company_entries}
        coverage[company["ticker"]] = {
            "company_name": company["name"],
            "counts": dict(Counter(entry.document_type for entry in company_entries)),
            "gaps": [kind for kind in (
                "earnings_presentation", "earnings_call_transcript", "prepared_remarks",
                "investor_day_presentation", "special_presentation",
            ) if kind not in present],
        }
    manifest = {
        "schema_version": "1.0",
        "source_policy": "Official SEC EDGAR filing exhibits only; no third-party transcripts.",
        "from_date": args.from_date.isoformat(),
        "cutoff_date": args.cutoff.isoformat(),
        "documents": [asdict(entry) for entry in entries],
        "counts_by_type": dict(counts),
        "coverage": coverage,
    }
    if args.output.parent.name == "raw" and args.output.parent.parent.name == "narrative":
        manifest_name = {
            "supplemental_sec": "manifest_supplemental_sec.json",
        }.get(args.output.name, f"manifest_{args.output.name}.json")
        manifest_path = args.output.parent / manifest_name
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(entries)} supplemental documents; manifest: {manifest_path}")
    print(json.dumps(dict(counts), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
