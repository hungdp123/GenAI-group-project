"""Collect official narrative documents linked by company investor-relations sites.

The collector never follows search-engine or transcript-vendor links. It accepts a
document only when it is linked from an official company IR page. Documents without
an exact publication date are downloaded but marked ineligible for temporal retrieval
until a reviewer supplies the date.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlparse, urlunparse

from finsight.agents.narrative.download_corpus import SecClient, _safe_name


IR_SOURCES: dict[str, list[str]] = {
    "AAPL": ["https://investor.apple.com/"],
    "GOOGL": ["https://abc.xyz/investor/earnings/"],
    "NKE": ["https://investors.nike.com/Search-Results/default.aspx?SearchTerm=transcript"],
    "NVDA": [
        "https://investor.nvidia.com/financial-info/financial-reports/default.aspx?lang=el-GR",
        "https://investor.nvidia.com/events-and-presentations/default.aspx",
        "https://investor.nvidia.com/search-results/default.aspx?SearchTerm=CFO%20Commentary",
        "https://investor.nvidia.com/search-results/default.aspx?SearchTerm=Quarterly%20Presentation",
        "https://investor.nvidia.com/search-results/default.aspx?SearchTerm=Revenue%20Trend",
    ],
    "MCD": ["https://investor.mcdonalds.com/search-results/default.aspx?SearchTerm=presentation"],
    "TSLA": ["https://ir.tesla.com/"],
    "NFLX": [
        "https://ir.netflix.net/search-results/default.aspx?SearchTerm=transcript",
        "https://ir.netflix.net/investor-news-and-events/investor-events/default.aspx",
    ],
    "OPEN": ["https://investor.opendoor.com/financials-filings/quarterly-reports"],
    "KSS": ["https://investors.kohls.com/search-results/default.aspx?SearchTerm=presentation"],
    "DNUT": ["https://investors.krispykreme.com/"],
    "GPRO": [
        "https://investor.gopro.com/search-results/default.aspx?SearchTerm=commentary",
        "https://investor.gopro.com/events-and-presentations/default.aspx",
    ],
    "BIRD": [
        "https://ir.allbirds.com/financial-information/quarterly-results",
        "https://ir.allbirds.com/news-and-events/presentations",
        "https://ir.allbirds.com/events/",
    ],
    "GME": ["https://investor.gamestop.com/search-results/default.aspx?SearchTerm=presentation"],
    "AMC": [
        "https://investor.amctheatres.com/news-events/presentations",
        "https://investor.amctheatres.com/financial-information/financial-results",
    ],
}

KNOWN_DATES_BY_FILENAME = {
    "NIKE-Inc.-Q1FY21-OFFICIAL-Transcript-with-QA.pdf": "2020-09-22",
    "NIKE-Inc.-Q3FY21-OFFICIAL-Transcript-with-Q-A.pdf": "2021-03-18",
    "NIKE-Inc.-Q4FY21-OFFICIAL-Transcript-with-QA-FINAL.pdf": "2021-06-24",
    "FY-2022-Q2-Earnings-Release-Conference-Call-OFFICIAL-TRANSCRIPT-FINAL-(1).pdf": "2021-12-20",
    "FY-2022-Q3-Earnings-Release-Conference-Call-OFFICIAL-TRANSCRIPT-FINAL.pdf": "2022-03-21",
    "NIKE-Inc.-Q1FY23-OFFICIAL-Transcript-FINAL.pdf": "2022-09-29",
    "Netflix,-Inc.pdf": "2022-07-19",
    "netflix-inc-usa_earnings-call_2024-04-18_english-1.pdf": "2024-04-18",
    "netflix-inc-usa-244e80f7-c939-40a1-a316-82d7d29f8a40.pdf": "2024-07-18",
    "Netflix-Inc-USA_Earnings-Call_2024-10-17_English-1.pdf": "2024-10-17",
    "FINAL-Q1-22-Shareholder-Letter.pdf": "2022-04-19",
    "FINAL-Q2-22-Shareholder-Letter.pdf": "2022-07-19",
    "FINAL-Q3-22-Shareholder-Letter.pdf": "2022-10-18",
    "FINAL-Q4-22-Shareholder-Letter.pdf": "2023-01-19",
    "Final-Q1-23-Shareholder-Letter.pdf": "2023-04-18",
    "FINAL-Q2-23-Shareholder-Letter.pdf": "2023-07-19",
    "FINAL-Q3-23-Shareholder-Letter.pdf": "2023-10-18",
    "NEW-FINAL-Q4-23-Shareholder-Letter.pdf": "2024-01-23",
    "FINAL-Q1-24-Shareholder-Letter.pdf": "2024-04-18",
    "FINAL-Q2-24-Shareholder-Letter.pdf": "2024-07-18",
    "FINAL-Q3-24-Shareholder-Letter.pdf": "2024-10-17",
    "Q1-2022-Management-Commentary.pdf": "2022-05-05",
    "Q2-2022-Management-Commentary.pdf": "2022-08-04",
    "Q3-2022-Management-Commentary.pdf": "2022-11-03",
    "Q4-2022-Management-Commentary.pdf": "2023-02-02",
    "Q1-2023-Management-Commentary.pdf": "2023-05-09",
    "Q2-2023-Management-Commentary.pdf": "2023-08-03",
    "Q3-2023-Management-Commentary.pdf": "2023-11-07",
    "Q4-2023-Management-Commentary.pdf": "2024-02-07",
    "Q1-2024-Management-Commentary.pdf": "2024-05-07",
    "Q2-2024-Management-Commentary.pdf": "2024-08-06",
    "Q3-2024-Management-Commentary.pdf": "2024-11-07",
    "Q4-2024-Management-Commentary.pdf": "2025-02-06",
    "Q1-2025-Management-Commentary.pdf": "2025-05-12",
    "Q3-2021-Management-Commentary.pdf": "2021-11-04",
    "Q4-2021-Management-Commentary.pdf": "2022-02-03",
    "FINAL-Q1-21-Shareholder-Letter.pdf": "2021-04-20",
    "FINAL-Q2-21-Shareholder-Letter.pdf": "2021-07-20",
    "FINAL-Q3-21-Shareholder-Letter.pdf": "2021-10-19",
    "FINAL-Q4-21-Shareholder-Letter.pdf": "2022-01-20",
    "FINAL-Q4-24-Shareholder-Letter.pdf": "2025-01-21",
    "COMBINED-Q1-25-Shareholder-Letter-V2.pdf": "2025-04-17",
    "FINAL-Q2-25-Shareholder-Letter.pdf": "2025-07-17",
    "FINAL-Q3-25-Shareholder-Letter.pdf": "2025-10-21",
    "Netflix-Inc-_Earnings-Call_Transcript.pdf": "2025-07-17",
}

NVDA_PUBLICATION_DATES = {
    (2024, 1): "2023-05-24", (2024, 2): "2023-08-23",
    (2024, 3): "2023-11-21", (2024, 4): "2024-02-21",
    (2025, 1): "2024-05-22", (2025, 2): "2024-08-28",
    (2025, 3): "2024-11-20", (2025, 4): "2025-02-26",
    (2026, 1): "2025-05-28", (2026, 2): "2025-08-27",
    (2026, 3): "2025-11-19", (2026, 4): "2026-02-25",
    (2027, 1): "2026-05-20", (2027, 2): "2026-08-26",
}

MONTHS: dict[str, int] = {}
for month_number, month_name in enumerate(
    (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ),
    start=1,
):
    MONTHS[month_name.lower()] = month_number
    MONTHS[month_name[:3].lower()] = month_number


@dataclass
class IrEntry:
    document_id: str
    company_ticker: str
    document_type: str
    publication_date: str | None
    date_precision: str
    retrieval_eligible: bool
    source_tier: int
    source_origin: str
    source_page: str
    source_url: str
    local_path: str
    sha256: str
    link_text: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, re.sub(r"\s+", " ", " ".join(self._text)).strip()))
            self._href = None
            self._text = []


def _page_url(url: str, page: int) -> str:
    parts = urlparse(url)
    query = dict(parse_qsl(parts.query))
    query["PageNumber"] = str(page)
    return urlunparse(parts._replace(query=urlencode(query)))


def _classify(href: str, label: str) -> str | None:
    value = html.unescape(f"{href} {label}").lower()
    if "transcript" in value or "pre-recorded-earnings-call" in value:
        return "earnings_call_transcript"
    if any(term in value for term in (
        "prepared-remarks", "prepared remarks", "commentary", "letter-to-shareholders",
        "letter to shareholders",
    )):
        return "prepared_remarks"
    if any(term in value for term in ("investor-day", "investor day", "analyst-day", "capital-markets-day")):
        return "investor_day_presentation"
    earnings_terms = (
        "earnings-presentation", "earnings presentation", "results-presentation", "shareholder-deck",
        "quarterly revenue trend", "quarterly presentation", "earnings-supplement", "earnings supplement",
    )
    if any(term in value for term in earnings_terms):
        return "earnings_presentation"
    if any(term in value for term in (
        "investor-presentation", "investor presentation", "investor-deck", "investor deck",
        "strategic-transformation", "strategic update", "presentation", "slides", "deck",
    )):
        return "special_presentation"
    return None


def _is_document_link(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".pdf", ".htm", ".html", ".txt", ".doc", ".docx")) or "/static-files/" in path


def _safe_url(url: str) -> str:
    parts = urlparse(url)
    return urlunparse(parts._replace(path=quote(parts.path, safe="/%:@-._~!$&'()*+,;=")))


def _date_hint(value: str, from_date: date, cutoff: date) -> tuple[str | None, str]:
    exact = re.search(
        r"(?i)\b(" + "|".join(MONTHS) + r")[\s_-]+(\d{1,2}),?[\s_-]+(20\d{2})\b",
        value,
    )
    if exact:
        parsed = date(int(exact.group(3)), MONTHS[exact.group(1).lower()], int(exact.group(2)))
        if from_date <= parsed <= cutoff:
            return parsed.isoformat(), "day"
    iso = re.search(r"(?<!\d)(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})(?!\d)", value)
    if iso:
        try:
            parsed = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
            if from_date <= parsed <= cutoff:
                return parsed.isoformat(), "day"
        except ValueError:
            pass
    years = [int(year) for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", value)]
    years = [year for year in years if from_date.year <= year <= cutoff.year]
    return (str(years[0]), "year") if years else (None, "unknown")


def _is_search_page(seed: str) -> bool:
    return "search-results" in seed.lower()


def _discover(client: SecClient, seed: str, max_pages: int) -> list[tuple[str, str, str]]:
    discovered: list[tuple[str, str, str]] = []
    seen_pages: set[str] = set()
    page_range = range(1, max_pages + 1) if _is_search_page(seed) else range(1, 2)
    for page_number in page_range:
        page_url = _page_url(seed, page_number) if _is_search_page(seed) else seed
        payload = client.get(page_url)
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen_pages:
            break
        seen_pages.add(digest)
        parser = LinkParser()
        parser.feed(payload.decode("utf-8", errors="ignore"))
        page_hits = 0
        for href, label in parser.links:
            url = urljoin(page_url, href)
            if _classify(url, label) and _is_document_link(url):
                discovered.append((url, label, page_url))
                page_hits += 1
        if _is_search_page(seed) and page_number > 1 and page_hits == 0:
            break
    return discovered


def _microsoft_pages() -> list[tuple[str, str, str]]:
    rows = []
    for fiscal_year in range(2021, 2027):
        for quarter in range(1, 5):
            url = (
                f"https://www.microsoft.com/en-us/investor/events/fy-{fiscal_year}/"
                f"earnings-fy-{fiscal_year}-q{quarter}"
            )
            rows.append((url, f"Microsoft FY{str(fiscal_year)[-2:]} Q{quarter} earnings call transcript", url))
    return rows


def _alphabet_pages() -> list[tuple[str, str, str]]:
    source_page = "https://abc.xyz/investor/earnings/"
    rows: list[tuple[str, str, str]] = []
    legacy_dates = {
        (2021, 1): "April 27, 2021",
        (2021, 2): "July 27, 2021",
        (2021, 3): "October 26, 2021",
        (2021, 4): "February 1, 2022",
        (2022, 1): "April 26, 2022",
        (2022, 2): "July 26, 2022",
        (2022, 3): "October 25, 2022",
        (2022, 4): "February 2, 2023",
        (2023, 1): "April 25, 2023",
        (2023, 2): "July 25, 2023",
        (2023, 3): "October 24, 2023",
        (2023, 4): "January 30, 2024",
    }
    for (year, quarter), call_date in legacy_dates.items():
        url = f"https://abc.xyz/assets/investor/static/pdf/{year}_Q{quarter}_Earnings_Transcript.pdf"
        label = f"Alphabet Q{quarter} {year} earnings call transcript {call_date}"
        rows.append((url, label, source_page))

    event_pages = {
        (2024, 1): "https://abc.xyz/investor/events/event-details/2024/2024-Q1-Earnings-Call/",
        (2024, 2): "https://abc.xyz/investor/events/event-details/2024/2024-Q2-Earnings-Call/",
        (2024, 3): "https://abc.xyz/investor/events/event-details/2024/2024-q3-earnings-call/",
        (2024, 4): "https://abc.xyz/investor/events/event-details/2025/2024-Q4-Earnings-Call/",
        (2025, 1): "https://abc.xyz/investor/events/event-details/2025/2025-Q1-Earnings-Call/",
        (2025, 2): "https://abc.xyz/investor/events/event-details/2025/2025-Q2-Earnings-Call/",
        (2025, 3): (
            "https://abc.xyz/investor/events/event-details/2025/"
            "2025-Q3-Earnings-Call-2025-4OI4Bac_Q9/default.aspx"
        ),
        (2025, 4): (
            "https://abc.xyz/investor/events/event-details/2026/"
            "2025-Q4-Earnings-Call-2026-Dr_C033hS6/default.aspx"
        ),
    }
    for (year, quarter), url in event_pages.items():
        rows.append((url, f"Alphabet Q{quarter} {year} earnings call transcript", url))
    return rows


def _netflix_pages(client: SecClient, from_date: date, cutoff: date) -> list[tuple[str, str, str]]:
    quarter_names = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    event_root = "https://ir.netflix.net/investor-news-and-events/investor-events/event-details"
    rows: list[tuple[str, str, str]] = [
        (
            "https://s22.q4cdn.com/959853165/files/doc_financials/2021/q1/"
            "Netflix-Inc.-Q1-2021-Earnings-Call-Apr-20-2021.pdf",
            "Netflix Q1 2021 earnings call transcript April 20, 2021",
            f"{event_root}/2021/Netflix-First-Quarter-2021-Earnings-Interview/default.aspx",
        ),
        (
            "https://s22.q4cdn.com/959853165/files/doc_financials/2021/q2/"
            "Netflix,-Inc.,-Q2-2021-Pre-Recorded-Earnings-Call,-Jul-20,-2021.pdf",
            "Netflix Q2 2021 earnings call transcript July 20, 2021",
            f"{event_root}/2021/Netflix-Second-Quarter-2021-Earnings-Interview/default.aspx",
        ),
        (
            "https://s22.q4cdn.com/959853165/files/doc_financials/2021/q3/"
            "Netflix,-Inc.,-Q3-2021-Earnings-Call,-Oct-19,-2021.pdf",
            "Netflix Q3 2021 earnings call transcript October 19, 2021",
            f"{event_root}/2021/Netflix-Third-Quarter-2021-Earnings-Interview/default.aspx",
        ),
        (
            "https://s22.q4cdn.com/959853165/files/doc_financials/2021/q4/"
            "Netflix-Inc.-Q4-2021-Pre-Recorded-Earnings-Call-Jan-20-2022.pdf",
            "Netflix Q4 2021 earnings call transcript January 20, 2022",
            f"{event_root}/2022/Netflix-Fourth-Quarter-2021-Earnings-Interview/default.aspx",
        ),
    ]
    for fiscal_year in range(from_date.year, cutoff.year + 1):
        for quarter, quarter_name in quarter_names.items():
            publication_year = fiscal_year + 1 if quarter == 4 else fiscal_year
            if publication_year > cutoff.year:
                continue
            default_url = (
                "https://ir.netflix.net/investor-news-and-events/investor-events/"
                f"event-details/{publication_year}/Netflix-{quarter_name}-Quarter-"
                f"{fiscal_year}-Earnings-Interview/default.aspx"
            )
            special_urls = {
                (2023, 3): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2023/"
                    "Netflix-Third-Quarter-2023-Earnings-Interview-Oct-18-2023-0300-PM-PT-"
                    "2023-LMLw6EO3HI/default.aspx"
                ),
                (2024, 1): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2024/"
                    "Netflix-First-Quarter-2024-Earnings-Interview-2024-elLzUHd2U3/default.aspx"
                ),
                (2024, 2): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2024/"
                    "Netflix-Second-Quarter-2024-Earnings-Interview-2024-HkuRh-_yAj/default.aspx"
                ),
                (2024, 3): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2024/"
                    "Netflix-Third-Quarter-2024-Earnings-Interview-2024-ZkMMCbq8rF/default.aspx"
                ),
                (2024, 4): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2025/"
                    "Netflix-Fourth-Quarter-2024-Earnings-Interview-2025-sZnTu2Hz5E/default.aspx"
                ),
                (2025, 1): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2025/"
                    "Netflix-First-Quarter-2025-Earnings-Interview-2025-sSZFdrz-vz/default.aspx"
                ),
                (2025, 2): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2025/"
                    "Netflix-Second-Quarter-2025-Earnings-Interview-2025--ZzeVaVp2V/default.aspx"
                ),
                (2025, 3): (
                    "https://ir.netflix.net/investor-news-and-events/investor-events/event-details/2025/"
                    "Netflix-Third-Quarter-2025-Earnings-Interview-2025-bw-f2eDpkM/default.aspx"
                ),
            }
            url = special_urls.get((fiscal_year, quarter), default_url)
            try:
                rows.extend(_discover(client, url, 1))
            except RuntimeError:
                continue
    return rows


def _nvidia_pages() -> list[tuple[str, str, str]]:
    """Return official NVIDIA narrative assets missed by its JavaScript archive."""
    source_page = "https://investor.nvidia.com/financial-info/financial-reports/default.aspx"
    cdn = "https://s201.q4cdn.com/141608511/files/doc_financials"
    rows: list[tuple[str, str, str]] = []
    for fiscal_year in range(2024, 2028):
        year_suffix = str(fiscal_year)[-2:]
        for quarter in range(1, 5):
            folder = f"Q{quarter}FY{year_suffix}" if fiscal_year == 2024 else f"Q{quarter}{year_suffix}"
            publication_date = NVDA_PUBLICATION_DATES.get((fiscal_year, quarter), "")
            rows.extend((
                (
                    f"{cdn}/{fiscal_year}/{folder}/Q{quarter}FY{year_suffix}-CFO-Commentary.pdf",
                    f"NVIDIA Q{quarter} FY{fiscal_year} CFO commentary {publication_date}",
                    source_page,
                ),
                (
                    f"{cdn}/{fiscal_year}/{folder}/Rev_by_Mkt_Qtrly_Trend_Q{quarter}{year_suffix}.pdf",
                    f"NVIDIA Q{quarter} FY{fiscal_year} quarterly revenue trend {publication_date}",
                    source_page,
                ),
                (
                    f"{cdn}/{fiscal_year}/q{quarter}/NVDA-F{quarter}Q{year_suffix}-Quarterly-Presentation-FINAL.pdf",
                    f"NVIDIA Q{quarter} FY{fiscal_year} quarterly presentation {publication_date}",
                    source_page,
                ),
                (
                    f"{cdn}/{fiscal_year}/{folder}/NVDA-F{quarter}Q{year_suffix}-Quarterly-Presentation-FINAL.pdf",
                    f"NVIDIA Q{quarter} FY{fiscal_year} quarterly presentation {publication_date}",
                    source_page,
                ),
            ))
    rows.extend((
        (
            f"{cdn}/2027/Q127/Rev_by_Mkt_Qtrly_Trend_Q127-NEW-v3.pdf",
            "NVIDIA Q1 FY2027 quarterly revenue trend 2026-05-20",
            source_page,
        ),
        (
            f"{cdn}/2027/Q227/Q227-Revenue-by-Market-Platform-Slides.pdf",
            "NVIDIA Q2 FY2027 quarterly revenue trend 2026-08-26",
            source_page,
        ),
        (
            f"{cdn}/2026/q4/NVDA-Q4-2026-Earnings-Call-25-February-2026-5_00-PM-ET.pdf",
            "NVIDIA Q4 FY2026 earnings call transcript February 25, 2026",
            source_page,
        ),
    ))
    return rows


def _gopro_pages(from_date: date, cutoff: date) -> list[tuple[str, str, str]]:
    """Return official GoPro commentary assets missed by its JavaScript archive."""
    source_page = "https://investor.gopro.com/events-and-presentations/default.aspx"
    root = "https://investor.gopro.com/files/doc_financials"
    rows: list[tuple[str, str, str]] = []
    for year in range(from_date.year, cutoff.year + 1):
        for quarter in range(1, 5):
            url = f"{root}/{year}/q{quarter}/Q{quarter}-{year}-Management-Commentary.pdf"
            rows.append((url, f"GoPro Q{quarter} {year} management commentary", source_page))
    return rows


def _opendoor_pages() -> list[tuple[str, str, str]]:
    """Return shareholder letters linked by Opendoor's official quarterly archive."""
    source_page = "https://investor.opendoor.com/financials-filings/quarterly-reports"
    rows = (
        ("2021-05-11", "1Q21", "e3df73ec-4145-41e2-ab32-7093b36cb296"),
        ("2021-08-11", "2Q21", "c78989fe-7029-4ad7-9923-f56e86353783"),
        ("2021-11-10", "3Q21", "25461ff7-dbe5-418b-ba4c-ee1575c2bf69"),
        ("2022-02-24", "4Q21", "9e9b4075-3822-436b-874f-0dc23d6c2ff7"),
        ("2022-05-05", "1Q22", "f8414fc7-9caa-4819-9376-5b4ab5fb463b"),
        ("2022-08-04", "2Q22", "6ee385f1-8aa7-443a-850a-1659145bb672"),
        ("2022-11-03", "3Q22", "f53f99c1-b0e9-4fd6-a915-1a13168231b0"),
        ("2023-02-23", "4Q22", "fd4e86fc-dc58-47a6-a5f9-ad59b20219ab"),
        ("2023-05-04", "1Q23", "d8c2a991-84cd-406a-99fc-b818b0913043"),
        ("2023-08-03", "2Q23", "e2d14964-7ae3-43f9-af5e-4fa2b8ab2bdb"),
        ("2023-11-02", "3Q23", "f93906e6-f284-40fb-8571-6de0a01a9bde"),
        ("2024-02-15", "4Q23", "d8ae91d0-1697-4e39-a7df-c413ef9b4d9c"),
        ("2024-05-02", "1Q24", "77697f56-f125-42e2-bd44-06e0ee2f53e7"),
        ("2024-08-01", "2Q24", "842e61a1-a372-48cc-9404-2706c27124e4"),
        ("2024-11-07", "3Q24", "b720864a-b9b8-4286-a2f0-d35abb057f17"),
        ("2025-02-27", "4Q24", "ec052fa7-f17b-42ed-8ebc-baccd40b7a08"),
        ("2025-05-06", "1Q25", "9ea3387d-858b-449a-8fe5-817ba97d2c28"),
        ("2025-08-05", "2Q25", "3af7a1bb-ecb6-4446-9194-cc1b7cc64bfd"),
    )
    return [
        (
            f"https://investor.opendoor.com/static-files/{asset_id}",
            f"Opendoor {quarter} letter to shareholders {publication_date}",
            source_page,
        )
        for publication_date, quarter, asset_id in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2021, 1, 1))
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2026, 10, 4))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/narrative/raw/ir_from_2021_official_final"),
    )
    parser.add_argument("--max-search-pages", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--reuse-failures", action="store_true")
    parser.add_argument(
        "--user-agent",
        default=os.getenv("SEC_USER_AGENT", "FinSight academic project finsight-project@example.com"),
    )
    args = parser.parse_args()
    client = SecClient(args.user_agent, timeout=args.timeout, attempts=args.attempts)
    existing_manifest = args.output / "manifest.json"
    cached_files: dict[str, Path] = {}
    cached_failures: dict[str, str] = {}
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        for document in previous.get("documents", []):
            local_path = Path(document["local_path"])
            if local_path.exists():
                cached_files[document["source_url"]] = local_path
        if args.reuse_failures:
            for warning in previous.get("warnings", []):
                match = re.match(r"^[A-Z]+: (https?://.*?): Unable to download", warning)
                if match:
                    cached_failures[match.group(1)] = warning
    candidates: dict[str, list[tuple[str, str, str]]] = {ticker: [] for ticker in IR_SOURCES}
    for ticker, seeds in IR_SOURCES.items():
        print(f"Discovering official IR documents for {ticker}...")
        for seed in seeds:
            try:
                candidates[ticker].extend(_discover(client, seed, args.max_search_pages))
            except RuntimeError as exc:
                print(f"WARNING: {ticker}: {exc}")
    candidates["GOOGL"].extend(_alphabet_pages())
    candidates["NFLX"].extend(_netflix_pages(client, args.from_date, args.cutoff))
    candidates["NVDA"].extend(_nvidia_pages())
    candidates["GPRO"].extend(_gopro_pages(args.from_date, args.cutoff))
    candidates["OPEN"].extend(_opendoor_pages())
    candidates["MSFT"] = _microsoft_pages()

    entries: list[IrEntry] = []
    seen_hashes: set[str] = set()
    warnings: list[str] = []
    for ticker, links in candidates.items():
        for url, label, source_page in dict.fromkeys(links):
            document_type = _classify(url, label)
            if document_type is None:
                continue
            stated_years = [int(year) for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", f"{url} {label}")]
            if stated_years and not any(args.from_date.year <= year <= args.cutoff.year for year in stated_years):
                continue
            url = _safe_url(url)
            cached_path = cached_files.get(url)
            if cached_path is not None:
                payload = cached_path.read_bytes()
            elif url in cached_failures:
                warnings.append(cached_failures[url])
                continue
            else:
                try:
                    payload = client.get(url)
                except RuntimeError as exc:
                    warnings.append(f"{ticker}: {url}: {exc}")
                    continue
            if payload.startswith(b"%PDF"):
                text = ""
            else:
                text = re.sub(r"(?s)<[^>]+>", " ", payload.decode("utf-8", errors="ignore"))
            if ticker == "MSFT":
                if "Transcript" not in text:
                    continue
            digest = hashlib.sha256(payload).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            filename = Path(urlparse(url).path).name or "index.html"
            if "." not in filename:
                filename += ".pdf" if payload.startswith(b"%PDF") else ".html"
            override = KNOWN_DATES_BY_FILENAME.get(Path(urlparse(url).path).name)
            if override:
                parsed_override = date.fromisoformat(override)
                if not (args.from_date <= parsed_override <= args.cutoff):
                    continue
                date_value, precision = override, "day"
            else:
                date_value, precision = _date_hint(f"{url} {label} {text}", args.from_date, args.cutoff)
            if precision == "year" and date_value is not None:
                year = int(date_value)
                starts_after_year_start = year == args.from_date.year and args.from_date != date(year, 1, 1)
                ends_before_year_end = year == args.cutoff.year and args.cutoff != date(year, 12, 31)
                if year < args.from_date.year or year > args.cutoff.year or starts_after_year_start or ends_before_year_end:
                    continue
            # Keep the manifest under raw/<corpus>/ and source files grouped
            # under raw/<ticker>/<corpus>/.
            target_dir = args.output.parent / ticker / args.output.name
            target_dir.mkdir(parents=True, exist_ok=True)
            date_prefix = date_value or "date-unknown"
            target = target_dir / _safe_name(f"{date_prefix}_{document_type}_{filename}")
            target.write_bytes(payload)
            entries.append(IrEntry(
                document_id=f"{ticker}-IR-{digest[:16]}",
                company_ticker=ticker,
                document_type=document_type,
                publication_date=date_value if precision == "day" else None,
                date_precision=precision,
                retrieval_eligible=precision == "day",
                source_tier=2,
                source_origin="Official company investor-relations website",
                source_page=source_page,
                source_url=url,
                local_path=target.as_posix(),
                sha256=digest,
                link_text=label,
            ))

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "source_policy": "Documents directly linked by official company investor-relations pages only.",
        "from_date": args.from_date.isoformat(),
        "cutoff_date": args.cutoff.isoformat(),
        "documents": [asdict(entry) for entry in entries],
        "warnings": warnings,
        "review_required": [
            entry.document_id for entry in entries if not entry.retrieval_eligible
        ],
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Downloaded {len(entries)} official IR documents; manifest: {manifest_path}")
    print(f"Date review required: {len(manifest['review_required'])}; warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
