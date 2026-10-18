"""Generate a Markdown catalog from a downloaded narrative-corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


def _display_path(path: str) -> str:
    return path.replace("\\", "/")


CANONICAL_MANIFEST_FOLDERS = {
    "manifest_core_sec.json": "core_sec",
    "manifest_supplemental_sec.json": "supplemental_sec",
    "manifest_official_ir.json": "official_ir",
    "manifest_event_reports.json": "event_ir",
}


def _corpus_folder(manifest_path: Path) -> str:
    return CANONICAL_MANIFEST_FOLDERS.get(manifest_path.name, manifest_path.parent.name)


def _unmanifested_count(manifest_path: Path | None, documents: list[dict[str, Any]]) -> int:
    if manifest_path is None or not documents:
        return 0
    if manifest_path.name in CANONICAL_MANIFEST_FOLDERS and manifest_path.parent.name == "raw":
        data_root = manifest_path.parent
        corpus_name = _corpus_folder(manifest_path)
    else:
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


def _manual_inbox_summary(
    raw_root: Path,
    documents: list[dict[str, Any]],
) -> str:
    """Summarize which staged manual files are represented in canonical manifests."""
    inbox = raw_root.parent / "manual_inbox"
    if not inbox.exists():
        return "Manual inbox is not present locally; no staged-file reconciliation was run."

    manifest_hashes = {
        str(document.get("sha256", "")).lower()
        for document in documents
        if document.get("sha256")
    }
    staged_files = [
        path
        for path in inbox.rglob("*")
        if path.is_file() and path.name not in {"import_metadata.json", "manual_import_skipped.json"}
    ]
    included = 0
    unmatched_keys: set[tuple[str, str]] = set()
    for path in staged_files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest().lower()
        if digest in manifest_hashes:
            included += 1
        else:
            unmatched_keys.add((path.parent.name.upper(), path.name))
    excluded = len(staged_files) - included
    skipped_path = inbox / "manual_import_skipped.json"
    skipped_reasons: Counter[str] = Counter()
    if skipped_path.exists():
        skipped_payload = json.loads(skipped_path.read_text(encoding="utf-8"))
        skipped_reasons.update(
            str(item.get("reason", "unspecified"))
            for item in skipped_payload.get("documents", [])
            if (str(item.get("ticker", "")).upper(), str(item.get("filename", "")))
            in unmatched_keys
        )
    reason_text = ", ".join(
        f"{count} {reason}"
        for reason, count in sorted(skipped_reasons.items())
    )
    suffix = f"; skipped-file log: {reason_text}" if reason_text else ""
    return (
        f"Manual inbox reconciliation: {included} of {len(staged_files)} staged document files "
        f"match a canonical manifest; {excluded} are not included{suffix}. See "
        "`data/narrative/manual_inbox/manual_import_skipped.json` for exclusions."
    )


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
    ir_estimated_eligible = sum(
        bool(document.get("retrieval_eligible"))
        and bool(
            document.get("publication_date_is_estimated")
            or (not document.get("publication_date") and document.get("estimated_publication_date"))
        )
        for document in ir_documents
    )
    ir_exact_eligible = ir_eligible - ir_estimated_eligible
    ir_review_count = len(ir_documents) - ir_eligible
    ir_ineligible_with_date = sum(
        not document.get("retrieval_eligible")
        and bool(document.get("publication_date") or document.get("estimated_publication_date"))
        for document in ir_documents
    )
    ir_ineligible_without_date = ir_review_count - ir_ineligible_with_date
    ir_eligible_by_ticker = Counter(
        document["company_ticker"]
        for document in ir_documents
        if document.get("retrieval_eligible")
    )
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
    manual_inbox_summary = _manual_inbox_summary(
        manifest_path.parent,
        documents + supplemental_documents + ir_documents,
    )

    core_manifest = _display_path(str(manifest_path))
    supplemental_manifest = _display_path(str(supplemental_manifest_path)) if supplemental_manifest_path else "-"
    ir_manifest_display = _display_path(str(ir_manifest_path)) if ir_manifest_path else "-"
    core_folder = _corpus_folder(manifest_path)
    supplemental_folder = _corpus_folder(supplemental_manifest_path) if supplemental_manifest_path else "-"
    ir_folder = _corpus_folder(ir_manifest_path) if ir_manifest_path else "-"
    total_downloads = len(documents) + len(supplemental_documents) + len(ir_documents)
    scope_tickers = sorted(grouped)
    company_scope = ", ".join(
        f"{grouped[ticker][0]['company_name']} (`{ticker}`)" for ticker in scope_tickers
    )
    coverage_rows = []
    for ticker in scope_tickers:
        company = grouped[ticker][0]["company_name"].replace("|", "\\|")
        supplemental_count = supplemental_by_ticker[ticker]
        ir_count = ir_by_ticker[ticker]
        beyond_core = supplemental_count + ir_count
        status = (
            "Captured; coverage review still required"
            if beyond_core
            else "Not captured / availability unverified"
        )
        coverage_rows.append(
            f"| {company} ({ticker}) | {beyond_core} | {status} | "
            f"{supplemental_count} supplemental SEC; {ir_count} official IR; "
            f"{ir_eligible_by_ticker[ticker]} IR retrieval-eligible |"
        )
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
        f"- The start boundary is {manifest.get('from_date')}; records before it are out of scope.",
        f"- Companies in scope: **{len(grouped)}**.",
        f"- Files downloaded across all three canonical datasets: **{total_downloads}**.",
        f"- Core SEC filings and earnings releases: **{len(documents)}**.",
        f"- Supplemental SEC presentations: **{len(supplemental_documents)}**.",
        f"- Official IR files: **{len(ir_documents)}**; **{ir_eligible}** are retrieval-eligible",
        f"  (**{ir_exact_eligible}** exact-date and **{ir_estimated_eligible}** estimated-date);",
        f"  **{ir_review_count}** remain ineligible ({ir_ineligible_without_date} without a date;",
        f"  {ir_ineligible_with_date} have a date but still need eligibility review).",
        f"- Unmanifested raw files excluded from the corpus: **{total_unmanifested}**",
        "  (present in dataset folders but not represented in a canonical manifest; index manifest entries only).",
        "",
        "Canonical datasets:",
        "",
        "| Dataset | Raw files | Manifest | Status |",
        "|---|---|---|---|",
        f"| Core SEC | `data/narrative/raw/<TICKER>/{core_folder}/` | `{core_manifest}` | "
        "Complete for the defined SEC rules |",
        (
            f"| Supplemental SEC | `data/narrative/raw/<TICKER>/{supplemental_folder}/` | `{supplemental_manifest}` | "
            "Downloaded and deduplicated against core |"
        ),
        f"| Official IR | `data/narrative/raw/<TICKER>/{ir_folder}/` | `{ir_manifest_display}` | "
        "Partial; official-site access gaps remain |",
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
        f"{ir_date_range} for exact-dated files; {ir_estimated_eligible} eligible estimated-date file(s); "
        f"{ir_ineligible_without_date} without any date; {ir_ineligible_with_date} dated but ineligible |",
        "",
        "Do not use earlier folders named `history_5y_final`, `supplemental_5y_official_final`,",
        "`ir_5y_official_final`, `ir_5y_official_final_v2`, or `ir_5y_official_final_v3` for new",
        "indexing. They are older snapshots with the previous October-2021 boundary.",
        "",
        "## Corpus scope",
        "",
        f"In-scope companies, derived from the core manifest: {company_scope}.",
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
        "`finsight/agents/narrative/acquisition/download_ir_archive.py` under `IR_SOURCES`. Treat those",
        "company-controlled URLs as discovery starting points and retain the final official source",
        "URL in every accepted manifest record.",
        "",
        "## Data-download flow",
        "",
        "```text",
        "Ticker list",
        "    -> resolve SEC CIK",
        "    -> load current + historical SEC submission metadata",
        f"    -> filter publication dates from {manifest.get('from_date')} through {manifest['cutoff_date']}",
        "    -> download core 10-K, 10-Q, and Item 2.02 earnings exhibits",
        "    -> scan Item 2.02/7.01 exhibits for supplemental presentations",
        "    -> crawl official IR entry pages for transcripts/remarks/presentations",
        "    -> stage browser-only downloads under manual_inbox/<TICKER>/",
        "    -> recover official download provenance and prepare import metadata",
        "    -> import unique, in-window manual files into the canonical IR corpus",
        "    -> classify + deduplicate + hash every file",
        "    -> write dataset-named manifests directly in data/narrative/raw/",
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
        "8. Compute a SHA-256 hash and write metadata to the dataset manifest in `data/narrative/raw/`.",
        "",
        "Run all collectors from the repository root:",
        "",
        "```powershell",
        "$env:SEC_USER_AGENT=\"FinSight academic project your-contact@example.com\"",
        "python -m finsight.agents.narrative.acquisition.download_corpus `",
        f"  --from-date {manifest.get('from_date')} `",
        f"  --cutoff {manifest['cutoff_date']} `",
        "  --output data/narrative/raw/core_sec",
        "",
        "python -m finsight.agents.narrative.acquisition.download_supplemental_corpus `",
        f"  --from-date {manifest.get('from_date')} `",
        f"  --cutoff {manifest['cutoff_date']} `",
        "  --output data/narrative/raw/supplemental_sec `",
        "  --exclude-manifest data/narrative/raw/manifest_core_sec.json `",
        "    data/narrative/raw/manifest_official_ir.json",
        "",
        "python -m finsight.agents.narrative.acquisition.download_ir_archive `",
        f"  --from-date {manifest.get('from_date')} `",
        f"  --cutoff {manifest['cutoff_date']} `",
        "  --output data/narrative/raw/official_ir `",
        "  --max-search-pages 2",
        "",
        "# After placing browser-only official files in manual_inbox/<TICKER>/:",
        "python -m finsight.agents.narrative.acquisition.prepare_manual_ir_metadata",
        "",
        "python -m finsight.agents.narrative.acquisition.import_manual_ir `",
        "  \"data/narrative/manual_inbox/<TICKER>/import_metadata.json\" `",
        "  --manifest data/narrative/raw/manifest_official_ir.json `",
        "  --exclude-manifest data/narrative/raw/manifest_core_sec.json `",
        "    data/narrative/raw/manifest_supplemental_sec.json `",
        "  --skip-duplicates",
        "",
        "python -m finsight.agents.narrative.maintenance.generate_catalog `",
        "  data/narrative/raw/manifest_core_sec.json `",
        "  docs/narrative/corpus/narrative_corpus_download_guide.md `",
        "  --supplemental-manifest data/narrative/raw/manifest_supplemental_sec.json `",
        "  --ir-manifest data/narrative/raw/manifest_official_ir.json",
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
        "SEC: <filing-date>_<document-type>_<accession-number>_<original-SEC-filename>",
        "IR:  <publication-date-or-year>_<document-type>_<source-filename>",
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
        "Before indexing, run the corpus validation workflow to check missing files, SHA-256",
        "mismatches, out-of-range publication dates, and duplicate hashes. Files without a",
        "sufficiently supported date remain out of retrieval until reviewed.",
        "Manual inbox files already present in a canonical manifest are deliberately not re-imported;",
        "viewer wrappers and files outside the collection window are recorded in",
        "`data/narrative/manual_inbox/manual_import_skipped.json`.",
        manual_inbox_summary,
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
        "  Item 2.02 earnings-results exhibit returned by SEC metadata for all companies in scope;",
        f"- **{len(supplemental_documents)}** additional SEC-filed presentations, deduplicated against core; and",
        f"- **{len(ir_documents)}** files discovered through accessible official IR pages, of which",
        f"  **{ir_eligible}** are retrieval-eligible (**{ir_exact_eligible}** exact-date and",
        f"  **{ir_estimated_eligible}** estimated-date).",
        "",
        "Manifest counts establish what was captured, not that every official publication was found.",
        "Company-level supplemental and IR coverage therefore remains subject to a quarter-by-quarter",
        "review against each company's official investor-relations archive.",
        "",
        "### Company-level completeness",
        "",
        "| Company | Captured beyond core SEC | Status | Captured inventory |",
        "|---|---:|---|---|",
        *coverage_rows,
        "",
        "Cross-company unresolved items:",
        "",
        "- webcast-only events are not transcribed locally because generated text would not be an "
        "official-source transcript;",
        "- JavaScript-only archives, HTTP 403 responses, DNS failures, and timeouts can hide official files; and",
        "- a zero supplemental/IR count means **not captured**, not proof that the company never published a document.",
        "",
        "For official links and manual staging instructions, use",
        "`docs/narrative/corpus/narrative_manual_download_checklist.md`.",
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
        "Each SEC filename links to its original EDGAR document; official IR files link to the",
        "company-controlled source page or asset. `Local path` is relative to the repository root.",
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
            f"- Exact-date and retrieval-eligible: **{ir_exact_eligible}**",
            f"- Estimated-date and retrieval-eligible: **{ir_estimated_eligible}**",
            f"- Ineligible with no verified or estimated date: **{ir_ineligible_without_date}**",
            f"- Date present but still marked ineligible: **{ir_ineligible_with_date}**",
            "",
            "| Ticker | Verified publication date | Estimated publication date | Type | "
            "Retrieval eligible | Local file | Official source |",
            "|---|---|---|---|---|---|---|",
        ])
        for document in sorted(
            ir_documents,
            key=lambda d: (d["company_ticker"], d.get("publication_date") or "", d["document_type"]),
        ):
            local_path = _display_path(document["local_path"])
            filename = Path(local_path).name
            published = document.get("publication_date") or f"review required ({document['date_precision']})"
            estimated = document.get("estimated_publication_date") or "-"
            eligible_label = "Yes" if document.get("retrieval_eligible") else "No"
            lines.append(
                f"| {document['company_ticker']} | {published} | {estimated} | {document['document_type']} | "
                f"{eligible_label} | `{local_path}` | [{filename}]({document['source_url']}) |"
            )
        lines.extend([
            "",
            "Records without a verified or approved estimated date have no usable as-of date. Records "
            "with a date but `retrieval_eligible: false` remain excluded until that flag is reviewed. "
            "Official webcasts without",
            "a downloadable transcript are recorded as coverage gaps rather than converted from third-party text.",
            "Eligible estimated dates remain provisional and carry explicit provenance through processing "
            "and retrieval.",
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
