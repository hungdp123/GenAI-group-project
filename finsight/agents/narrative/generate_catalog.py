"""Generate a Markdown catalog from a downloaded narrative-corpus manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


def _display_path(path: str) -> str:
    return path.replace("\\", "/")


def _unmanifested_count(manifest_path: Path | None, documents: list[dict[str, Any]]) -> int:
    if manifest_path is None or not documents:
        return 0
    data_root = manifest_path.parent.parent
    corpus_name = manifest_path.parent.name
    represented = {Path(document["local_path"]).resolve() for document in documents}
    total = 0
    for ticker in {document["company_ticker"] for document in documents}:
        corpus_root = data_root / ticker / corpus_name
        if corpus_root.exists():
            total += sum(
                1 for path in corpus_root.rglob("*") if path.is_file() and path.resolve() not in represented
            )
    return total


def generate(
    manifest_path: Path,
    output_path: Path,
    supplemental_manifest_path: Path | None = None,
    ir_manifest_path: Path | None = None,
) -> None:
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        grouped[document["company_ticker"]].append(document)

    supplemental: dict[str, Any] = {}
    if supplemental_manifest_path and supplemental_manifest_path.exists():
        supplemental = json.loads(supplemental_manifest_path.read_text(encoding="utf-8"))
    supplemental_documents = supplemental.get("documents", [])

    ir_manifest: dict[str, Any] = {}
    if ir_manifest_path and ir_manifest_path.exists():
        ir_manifest = json.loads(ir_manifest_path.read_text(encoding="utf-8"))
    ir_documents = ir_manifest.get("documents", [])
    ir_eligible = sum(bool(document.get("retrieval_eligible")) for document in ir_documents)
    ir_review_count = len(ir_documents) - ir_eligible
    ir_review_phrase = "file requires" if ir_review_count == 1 else "files require"
    core_type_counts = Counter(document["document_type"] for document in documents)
    supplemental_type_counts = Counter(document["document_type"] for document in supplemental_documents)
    ir_type_counts = Counter(document["document_type"] for document in ir_documents)
    core_dates = [document["publication_date"] for document in documents if document.get("publication_date")]
    supplemental_dates = [
        document["publication_date"] for document in supplemental_documents if document.get("publication_date")
    ]
    ir_dates = [document["publication_date"] for document in ir_documents if document.get("publication_date")]
    core_date_range = f"{min(core_dates)} to {max(core_dates)}" if core_dates else "No exact dates"
    supplemental_date_range = (
        f"{min(supplemental_dates)} to {max(supplemental_dates)}" if supplemental_dates else "No exact dates"
    )
    ir_date_range = f"{min(ir_dates)} to {max(ir_dates)}" if ir_dates else "No exact dates"
    supplemental_by_ticker = Counter(document["company_ticker"] for document in supplemental_documents)
    ir_by_ticker = Counter(document["company_ticker"] for document in ir_documents)
    core_unmanifested = _unmanifested_count(manifest_path, documents)
    supplemental_unmanifested = _unmanifested_count(supplemental_manifest_path, supplemental_documents)
    ir_unmanifested = _unmanifested_count(ir_manifest_path, ir_documents)
    total_unmanifested = core_unmanifested + supplemental_unmanifested + ir_unmanifested

    core_root = _display_path(str(manifest_path.parent))
    supplemental_root = (
        _display_path(str(supplemental_manifest_path.parent)) if supplemental_manifest_path else "-"
    )
    ir_root = _display_path(str(ir_manifest_path.parent)) if ir_manifest_path else "-"
    total_downloads = len(documents) + len(supplemental_documents) + len(ir_documents)

    lines = [
        "# FinSight Narrative Corpus - Download Guide and File Catalog",
        "",
        "> Latest status: generated from the canonical local manifests. Raw files are gitignored and",
        "> remain on the machine that ran the download; this Markdown inventory is safe to commit.",
        "",
        "## Current status at a glance",
        "",
        f"- Catalog generated from the local manifests on: **{date.today().isoformat()}**.",
        f"- Collection window: **{manifest.get('from_date')} through {manifest['cutoff_date']}**, inclusive.",
        "- The start boundary is 1 January 2021; there is no October-2021 start filter.",
        f"- Companies in scope: **{len(grouped)}**.",
        f"- Files downloaded across all three canonical datasets: **{total_downloads}**.",
        f"- Core SEC filings and earnings releases: **{len(documents)}**.",
        f"- Supplemental SEC presentations: **{len(supplemental_documents)}**.",
        f"- Official IR files: **{len(ir_documents)}**; **{ir_eligible}** have exact dates and",
        f"  **{ir_review_count}** {ir_review_phrase} date review before retrieval.",
        f"- Unmanifested raw files excluded from the corpus: **{total_unmanifested}**",
        "  (leftovers from earlier collector reruns; index manifest entries only).",
        "",
        "Canonical datasets:",
        "",
        "| Dataset | Raw files | Manifest | Status |",
        "|---|---|---|---|",
        f"| Core SEC | `data/narrative/raw/<TICKER>/{Path(core_root).name}/` | `{core_root}/manifest.json` | Complete for the defined SEC rules |",
        (
            f"| Supplemental SEC | `data/narrative/raw/<TICKER>/{Path(supplemental_root).name}/` | `{supplemental_root}/manifest.json` | "
            "Downloaded and deduplicated against core |"
        ),
        f"| Official IR | `data/narrative/raw/<TICKER>/{Path(ir_root).name}/` | `{ir_root}/manifest.json` | Partial; official-site access gaps remain |",
        "",
        "Document summary by dataset and type:",
        "",
        "| Dataset | Type breakdown | Observed publication dates |",
        "|---|---|---|",
        f"| Core SEC | 10-K: {core_type_counts['10-K']}; 10-Q: {core_type_counts['10-Q']}; "
        f"earnings releases: {core_type_counts['earnings_release']} | {core_date_range} |",
        f"| Supplemental SEC | earnings presentations: {supplemental_type_counts['earnings_presentation']}; "
        f"investor-day presentations: {supplemental_type_counts['investor_day_presentation']}; "
        f"special presentations: {supplemental_type_counts['special_presentation']} | "
        f"{supplemental_date_range} |",
        f"| Official IR | earnings-call transcripts: {ir_type_counts['earnings_call_transcript']}; "
        f"prepared remarks/shareholder letters: {ir_type_counts['prepared_remarks']}; "
        f"earnings presentations: {ir_type_counts['earnings_presentation']}; "
        f"investor-day presentations: {ir_type_counts['investor_day_presentation']}; "
        f"special presentations: {ir_type_counts['special_presentation']} | "
        f"{ir_date_range} for exact-dated files; {ir_review_count} file(s) require date review |",
        "",
        "Do not use earlier folders named `history_5y_final`, `supplemental_5y_official_final`,",
        "`ir_5y_official_final`, `ir_5y_official_final_v2`, or `ir_5y_official_final_v3` for new",
        "indexing. They are older snapshots with the previous October-2021 boundary.",
        "",
        "## Corpus scope",
        "",
        "In-scope companies: Apple (`AAPL`), Alphabet/Google (`GOOGL`), Microsoft (`MSFT`),",
        "Nike (`NKE`), NVIDIA (`NVDA`), McDonald's (`MCD`), Tesla (`TSLA`), Netflix (`NFLX`),",
        "Opendoor (`OPEN`), Kohl's (`KSS`), Krispy Kreme (`DNUT`), GoPro (`GPRO`),",
        "Smartbird/formerly Allbirds (`BIRD`), GameStop (`GME`), and AMC Entertainment (`AMC`).",
        "",
        "Included document classes:",
        "",
        "- every Form 10-K and 10-Q filed in the inclusive date window;",
        "- earnings/results exhibits attached to Item 2.02 Form 8-K filings;",
        "- earnings, investor-day, capital-markets-day, and special investor presentations found",
        "  in official SEC filings; and",
        "- official earnings-call transcripts, prepared remarks, and presentations directly linked",
        "  by company investor-relations sites when the site exposed a downloadable asset.",
        "",
        "Excluded classes include Forms 3/4/5, proxy and registration statements, unrelated 8-K",
        "exhibits, webcast-only audio/video, unofficial transcript aggregators, investment blogs,",
        "social posts, and search-engine caches. The collection is based on publication/filing date,",
        "not the fiscal period printed inside a document.",
        "",
        "## Why these documents are necessary",
        "",
        "| Document class | Narrative-agent value |",
        "|---|---|",
        "| 10-K | Audited annual strategy, risks, business model, segment discussion, and long-term context |",
        "| 10-Q | Quarterly changes in performance, liquidity, risks, and management discussion |",
        "| Earnings release | Timely headline results, management framing, KPIs, and guidance |",
        "| Earnings presentation | Concise charts, operating metrics, segment trends, and management priorities |",
        (
            "| Earnings-call transcript/prepared remarks | Management explanations, analyst questions, "
            "tone, uncertainty, and forward guidance |"
        ),
        (
            "| Investor-day/special presentation | Multi-year targets, capital allocation, strategic "
            "initiatives, and market assumptions |"
        ),
        "",
        "Together they let the Narrative Agent compare audited facts with management's current",
        "explanation and forward-looking story. Keeping provenance, publication dates, and hashes",
        "allows time-aware retrieval, citations, deduplication, and later audit of an answer.",
        "",
        "## Source policy and where to download",
        "",
        "Only official SEC EDGAR and company-controlled investor-relations sources are allowed.",
        "A company CDN file is allowed only when an official IR page links to it. Search engines may",
        "help locate an official page during manual research, but a search result or cached copy is",
        "never stored as corpus evidence.",
        "",
        "Official SEC endpoints used:",
        "",
        "1. SEC ticker-to-CIK mapping:",
        "   <https://www.sec.gov/files/company_tickers.json>",
        "2. Company submission history:",
        "   `https://data.sec.gov/submissions/CIK##########.json`",
        "3. Historical submission shards listed by the company submission response:",
        "   `https://data.sec.gov/submissions/<historical-file-name>.json`",
        "4. Filing directory metadata used to identify Exhibit 99.1:",
        "   `https://www.sec.gov/Archives/edgar/data/<CIK>/<ACCESSION>/index.json`",
        "5. Original filing or exhibit:",
        "   `https://www.sec.gov/Archives/edgar/data/<CIK>/<ACCESSION>/<DOCUMENT>`",
        "",
        "In archive URLs, the CIK has no leading zeros and the accession number has no hyphens.",
        "",
        "Official company IR entry points are configured in",
        "`finsight/agents/narrative/download_ir_archive.py` under `IR_SOURCES`. Those URLs are the",
        "approved starting pages for Apple, Alphabet, Nike, NVIDIA, McDonald's, Tesla, Netflix,",
        "Opendoor, Kohl's, Krispy Kreme, GoPro, Smartbird/Allbirds, GameStop, and AMC. Microsoft",
        "official earnings transcript pages are generated from its fiscal-year/quarter URL pattern.",
        "",
        "## Data-download flow",
        "",
        "```text",
        "Ticker list",
        "    -> resolve SEC CIK",
        "    -> load current + historical SEC submission metadata",
        "    -> filter publication dates from 2021-01-01 through 2026-10-04",
        "    -> download core 10-K, 10-Q, and Item 2.02 earnings exhibits",
        "    -> scan Item 2.02/7.01 exhibits for supplemental presentations",
        "    -> crawl official IR entry pages for transcripts/remarks/presentations",
        "    -> stage browser-only downloads under manual_inbox/<TICKER>/",
        "    -> recover official download provenance and prepare import metadata",
        "    -> import unique, in-window manual files into the canonical IR corpus",
        "    -> classify + deduplicate + hash every file",
        "    -> write manifest.json",
        "    -> validate dates, paths, hashes, warnings, and retrieval eligibility",
        "    -> generate this Markdown catalog",
        "```",
        "",
        "The core downloader performs these steps:",
        "",
        "1. Resolve each ticker to its SEC CIK.",
        "2. Load recent and historical submission metadata.",
        "3. Filter filings using the inclusive filing-date range.",
        "4. Select every 10-K and 10-Q in the range.",
        "5. Find Item 2.02 8-K filings and select the Exhibit 99.1 earnings/results document.",
        "6. Download the original SEC HTML, Inline XBRL, PDF, or text file without converting it.",
        "7. Save it under the ticker directory using a traceable filename.",
        "8. Compute a SHA-256 hash and write all metadata to `manifest.json`.",
        "",
        "Run all collectors from the repository root:",
        "",
        "```powershell",
        "$env:SEC_USER_AGENT=\"FinSight academic project your-contact@example.com\"",
        "python -m finsight.agents.narrative.download_corpus `",
        "  --from-date 2021-01-01 `",
        "  --cutoff 2026-10-04 `",
        "  --output data/narrative/raw/from_2021_final",
        "",
        "python -m finsight.agents.narrative.download_supplemental_corpus `",
        "  --from-date 2021-01-01 `",
        "  --cutoff 2026-10-04 `",
        "  --output data/narrative/raw/supplemental_from_2021_official_final `",
        "  --exclude-manifest data/narrative/raw/from_2021_final/manifest.json `",
        "    data/narrative/raw/ir_from_2021_official_final/manifest.json",
        "",
        "python -m finsight.agents.narrative.download_ir_archive `",
        "  --from-date 2021-01-01 `",
        "  --cutoff 2026-10-04 `",
        "  --output data/narrative/raw/ir_from_2021_official_final `",
        "  --max-search-pages 2",
        "",
        "# After placing browser-only official files in manual_inbox/<TICKER>/:",
        "python -m finsight.agents.narrative.prepare_manual_ir_metadata",
        "",
        "$tickers = 'BIRD','DNUT','GPRO','MCD','MSFT','NFLX','NKE','NVDA','OPEN'",
        "foreach ($ticker in $tickers) {",
        "  python -m finsight.agents.narrative.import_manual_ir `",
        "    \"data/narrative/manual_inbox/$ticker/import_metadata.json\" `",
        "    --manifest data/narrative/raw/ir_from_2021_official_final/manifest.json `",
        "    --exclude-manifest data/narrative/raw/from_2021_final/manifest.json `",
        "      data/narrative/raw/supplemental_from_2021_official_final/manifest.json `",
        "    --skip-duplicates",
        "}",
        "",
        "python -m finsight.agents.narrative.generate_catalog `",
        "  data/narrative/raw/from_2021_final/manifest.json `",
        "  docs/narrative_corpus_download_guide.md `",
        "  --supplemental-manifest data/narrative/raw/supplemental_from_2021_official_final/manifest.json `",
        "  --ir-manifest data/narrative/raw/ir_from_2021_official_final/manifest.json",
        "```",
        "",
        "Replace the example address with a monitored contact address. SEC requests should use an",
        "identifying User-Agent and remain rate-limited.",
        "",
        "## Naming convention",
        "",
        "Each local filename follows:",
        "",
        "```text",
        "<filing-date>_<document-type>_<accession-number>_<original-SEC-filename>",
        "```",
        "",
        "Example:",
        "",
        "```text",
        "2026-07-30_earnings_release_0000320193-26-000018_a8-kex991q3202606272026.htm",
        "```",
        "",
        "The corresponding manifest record preserves the company, CIK, form, period, publication",
        "date, accession number, source tier, original SEC URL, local path, and SHA-256 hash.",
        "",
        "## Validation performed",
        "",
        "The final corpus was checked for:",
        "",
        "- publication dates outside the requested interval;",
        "- manifest entries with missing local files;",
        "- local files not represented by the manifest;",
        "- SHA-256 mismatches;",
        "- downloader warnings; and",
        "- correct preference for Exhibit 99.1 when supplemental Exhibit 99.2 also exists.",
        "",
        "The latest audit found 0 missing manifest files, 0 SHA-256 mismatches, 0 out-of-range",
        "publication dates, and 0 duplicate-hash groups across the three manifests. The core and",
        "supplemental manifests have no warnings. The official IR manifest contains site-access",
        "warnings and keeps any file without an exact date out of retrieval.",
        "Manual inbox files already present in a canonical manifest are deliberately not re-imported;",
        "viewer wrappers and files outside the collection window are recorded in",
        "`data/narrative/manual_inbox/manual_import_skipped.json`.",
        "",
        f"There are **{total_unmanifested}** unmanifested raw files: core {core_unmanifested},",
        f"supplemental {supplemental_unmanifested}, and IR {ir_unmanifested}. They are stale outputs",
        f"from earlier reruns, are not counted in the {total_downloads}-file corpus, and must not be indexed.",
        "",
        "## Complete and not-complete checklist",
        "",
        "### Complete or verified",
        "",
        f"- **{len(documents)}** core SEC documents covering every matching 10-K, 10-Q, and selected",
        "  Item 2.02 earnings-results exhibit returned by SEC metadata for all 15 companies;",
        f"- **{len(supplemental_documents)}** additional SEC-filed presentations, deduplicated against core; and",
        f"- **{len(ir_documents)}** files discovered through accessible official IR pages, of which",
        f"  **{ir_eligible}** are exact-dated and retrieval-eligible.",
        "",
        "The word **complete** above applies to the implemented SEC selection rules. For the",
        "company-level IR assessment below, **Complete for published quarterly materials** means the",
        "corpus contains the official downloadable quarterly narrative series the publisher actually",
        "offers through the latest in-window result. A company is not marked incomplete merely because",
        "it uses prepared remarks instead of transcripts or does not publish a separate earnings deck.",
        "",
        "### Company-level completeness",
        "",
        "- **Complete for published downloadable materials:** 4 companies (MSFT, TSLA, NFLX, GPRO).",
        "- **Partial:** 8 companies (GOOGL, NKE, NVDA, MCD, OPEN, KSS, DNUT, BIRD).",
        "- **Not complete or availability unverified:** 3 companies (AAPL, GME, AMC).",
        "",
        "| Company | Captured beyond core SEC | Status | Missing or uncertain coverage |",
        "|---|---:|---|---|",
        f"| Apple (AAPL) | {supplemental_by_ticker['AAPL'] + ir_by_ticker['AAPL']} | Not complete / availability unverified | "
        "No official downloadable earnings transcript, earnings deck, or investor-day deck was located; Apple provides earnings-call webcasts |",
        f"| Alphabet (GOOGL) | {supplemental_by_ticker['GOOGL'] + ir_by_ticker['GOOGL']} | Partial | "
        "20 official call transcripts cover 2021-2025; 2026 Q1-Q2 transcripts and official earnings/investor-day presentations remain missing or unverified |",
        f"| Microsoft (MSFT) | {supplemental_by_ticker['MSFT'] + ir_by_ticker['MSFT']} | Complete for published quarterly materials | "
        "22 official transcripts cover FY2021 Q3 through FY2026 Q4; four official legacy earnings slide decks were also imported |",
        f"| Nike (NKE) | {supplemental_by_ticker['NKE'] + ir_by_ticker['NKE']} | Partial | "
        "18 official transcripts captured; FY2022 Q1 and FY2023 Q3-Q4/FY2024 Q1 remain missing, and no strategy-presentation series is captured |",
        f"| NVIDIA (NVDA) | {supplemental_by_ticker['NVDA'] + ir_by_ticker['NVDA']} | Partial | "
        "Older investor presentations, commentary/trends, FY2024 Q3 materials, and 2021/2022 Investor Day decks are now imported; FY2022 Q1 commentary/trend, FY2023 Q2 commentary, FY2023 Q4 trend, and most transcripts remain missing |",
        (
            f"| McDonald's (MCD) | {supplemental_by_ticker['MCD'] + ir_by_ticker['MCD']} | Partial | "
            "Five official/SEC presentations are captured, including the 2026 Investor Day deck; quarterly earnings decks, remarks, and transcripts remain missing |"
        ),
        (
            f"| Tesla (TSLA) | {supplemental_by_ticker['TSLA'] + ir_by_ticker['TSLA']} | Complete for published downloadable materials | "
            "All 22 quarterly shareholder decks from 2021 Q1 through 2026 Q2 and the 2023 Investor Day "
            "deck are captured; earnings Q&A is webcast-only with no official text transcript |"
        ),
        (
            f"| Netflix (NFLX) | {supplemental_by_ticker['NFLX'] + ir_by_ticker['NFLX']} | Complete for published quarterly materials | "
            "Official shareholder letters and transcripts are captured through 2026 Q2; one May 2025 special deck still lacks an exact day |"
        ),
        (
            f"| Opendoor (OPEN) | {supplemental_by_ticker['OPEN'] + ir_by_ticker['OPEN']} | Partial | "
            "18 official shareholder letters cover 2021 Q1 through 2025 Q2; later 2025-2026 narrative materials and special presentations remain unverified |"
        ),
        (
            f"| Kohl's (KSS) | {supplemental_by_ticker['KSS'] + ir_by_ticker['KSS']} | Partial | "
            "21 SEC-filed presentations captured; official transcript coverage remains missing |"
        ),
        (
            f"| Krispy Kreme (DNUT) | {supplemental_by_ticker['DNUT'] + ir_by_ticker['DNUT']} | Partial | "
            "Twelve earnings decks, two 2026 transcripts, and one SEC investor-day deck are captured; 2022-2023 decks and most transcripts remain missing |"
        ),
        (
            f"| GoPro (GPRO) | {supplemental_by_ticker['GPRO'] + ir_by_ticker['GPRO']} | Complete for published quarterly materials | "
            "23 management commentaries cover Q4 2020 through Q2 2026 and 16 supplemental slide decks cover Q1 2021 through Q4 2024; no separate special deck was located |"
        ),
        (
            f"| Smartbird/Allbirds (BIRD) | {supplemental_by_ticker['BIRD'] + ir_by_ticker['BIRD']} | Partial | "
            "Ten transcripts, seven prepared-remarks files, one infographic, and one strategic deck are captured; older investor decks and post-Q3 2025 quarterly coverage remain incomplete |"
        ),
        (
            f"| GameStop (GME) | {supplemental_by_ticker['GME'] + ir_by_ticker['GME']} | Not complete | "
            "No requested IR files captured; configured official search URL returned HTTP 404 |"
        ),
        f"| AMC Entertainment (AMC) | {supplemental_by_ticker['AMC'] + ir_by_ticker['AMC']} | Not complete | "
        "One special presentation captured; earnings presentations and transcript files remain missing |",
        "",
        "Cross-company unresolved items:",
        "",
        "- webcast-only events are not transcribed locally because generated text would not be an official-source transcript;",
        "- JavaScript-only archives, HTTP 403 responses, DNS failures, and timeouts can hide official files; and",
        "- a zero supplemental/IR count means **not captured**, not proof that the company never published a document.",
        "",
        "For official links and manual staging instructions, use",
        "`docs/narrative_manual_download_checklist.md`.",
        "",
        "## Summary by company",
        "",
        "| Ticker | Company | 10-K | 10-Q | Earnings releases | Supplemental | IR | All files |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for ticker in sorted(grouped):
        company_documents = grouped[ticker]
        counts = Counter(d["document_type"] for d in company_documents)
        company = company_documents[0]["company_name"].replace("|", "\\|")
        supplemental_count = supplemental_by_ticker[ticker]
        ir_count = ir_by_ticker[ticker]
        all_count = len(company_documents) + supplemental_count + ir_count
        lines.append(
            f"| {ticker} | {company} | {counts['10-K']} | {counts['10-Q']} | "
            f"{counts['earnings_release']} | {supplemental_count} | {ir_count} | {all_count} |"
        )

    lines.extend([
        "",
        "## Complete file catalog",
        "",
        "Each filename links to its original SEC document. `Local path` is relative to the",
        "repository root.",
        "",
    ])

    for ticker in sorted(grouped):
        company_documents = sorted(
            grouped[ticker],
            key=lambda d: (d["publication_date"], d["document_type"], d["accession_number"]),
            reverse=True,
        )
        lines.extend([
            f"### {ticker} - {company_documents[0]['company_name']}",
            "",
            "| Filed | Type | Period | Local file | Original SEC document |",
            "|---|---|---|---|---|",
        ])
        for document in company_documents:
            local_path = _display_path(document["local_path"])
            filename = Path(local_path).name
            period = document.get("period") or "-"
            source_url = document["source_url"]
            lines.append(
                f"| {document['publication_date']} | {document['document_type']} | {period} | "
                f"`{local_path}` | [{filename}]({source_url}) |"
            )
        lines.append("")

    if supplemental_documents:
        lines.extend([
            "## Supplemental SEC exhibits",
            "",
            "These documents were selected from official Item 2.02 and Item 7.01 SEC filing exhibits.",
            "They exclude documents already present in the core corpus.",
            "",
            f"Documents: **{len(supplemental_documents)}**",
            "",
            "| Ticker | Filed | Type | Local file | Original SEC document |",
            "|---|---|---|---|---|",
        ])
        for document in sorted(
            supplemental_documents,
            key=lambda d: (d["company_ticker"], d["publication_date"], d["document_type"]),
        ):
            local_path = _display_path(document["local_path"])
            filename = Path(local_path).name
            lines.append(
                f"| {document['company_ticker']} | {document['publication_date']} | {document['document_type']} | "
                f"`{local_path}` | [{filename}]({document['source_url']}) |"
            )
        lines.append("")

    if ir_documents:
        lines.extend([
            "## Official investor-relations materials",
            "",
            "These files were linked directly by company-controlled investor-relations pages. A CDN asset is",
            "accepted only when discovered through its official IR page. No search-engine or transcript-vendor",
            "copy is used as the source.",
            "",
            f"- Documents: **{len(ir_documents)}**",
            f"- Temporally verified and retrieval-eligible: **{ir_eligible}**",
            f"- Awaiting exact-date review: **{len(ir_documents) - ir_eligible}**",
            "",
            "| Ticker | Published | Type | Retrieval eligible | Local file | Official source |",
            "|---|---|---|---|---|---|",
        ])
        for document in sorted(
            ir_documents,
            key=lambda d: (d["company_ticker"], d.get("publication_date") or "", d["document_type"]),
        ):
            local_path = _display_path(document["local_path"])
            filename = Path(local_path).name
            published = document.get("publication_date") or f"review required ({document['date_precision']})"
            eligible_label = "Yes" if document.get("retrieval_eligible") else "No"
            lines.append(
                f"| {document['company_ticker']} | {published} | {document['document_type']} | "
                f"{eligible_label} | `{local_path}` | [{filename}]({document['source_url']}) |"
            )
        lines.extend([
            "",
            "Documents awaiting exact-date review must not enter the retrieval index. Official webcasts without",
            "a downloadable transcript are recorded as coverage gaps rather than converted from third-party text.",
            "",
        ])

    lines.extend([
        "## Important handling notes",
        "",
        "- Do not commit `data/`; it is intentionally excluded by `.gitignore`.",
        "- Share the raw corpus through the team's approved shared storage if another machine needs it.",
        "- Use `publication_date`, not period end, for FinSight cut-off-date filtering.",
        "- Treat retrieved filing text as untrusted data, never as model instructions.",
        "- Do not use the pasted user claim as evidence.",
        "- Rebuild the index whenever the manifest or chunking implementation changes.",
        "- The manifest is the authoritative machine-readable inventory; this Markdown file is the",
        "  human-readable catalog.",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--supplemental-manifest", type=Path)
    parser.add_argument("--ir-manifest", type=Path)
    args = parser.parse_args()
    generate(args.manifest, args.output, args.supplemental_manifest, args.ir_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
