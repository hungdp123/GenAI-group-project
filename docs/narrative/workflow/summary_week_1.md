# Narrative Agent — Week 1 Summary

> **Current processing status (17 October 2026):** All 1,204 retrieval-eligible
> documents across 16 companies are represented in the saved blocks, chunks and
> BM25 indexes. All 16 companies pass the current document-coverage, provenance
> and freshness checks. Microsoft's rebuilt output includes all 66 documents,
> including its 11 converted presentation PDFs. The current totals are 9,225 blocks,
> 22,657 chunks and 21,972 deduplicated indexed passages. The remaining sections
> retain the original Week 1 snapshot. See the [updated Week 1 recap](recap_week_1.md)
> and [audit report](../../../data/narrative/experiments/week1_audit/report.json).

Workflow recap (tasks, inputs, scripts, outputs and status): [Week 1 recap](recap_week_1.md).

## Goal

Prepare official company documents for the Narrative Agent: collect SEC filings and
earnings-related materials, preserve source tier and date metadata, and turn the documents
into traceable chunks that can be retrieved as evidence.

## End-to-end workflow

```text
                     OFFICIAL DOCUMENT SOURCES
                               │
     ┌─────────────────┬───────┴────────┬──────────────────┐
     │                 │                │                  │
 SEC core         SEC supplemental   Company IR       Manual intake
 10-K / 10-Q /    presentations /    transcripts /    official files
 earnings         special exhibits   decks / remarks  (optional)
 releases
     │                 │                │                  │
     └─────────────────┴───────┬────────┴──────────────────┘
                               ▼
                  Raw files + canonical manifests
            (URL, company, source tier, date, hash, path)
                               │
                               ▼
                  Validate records and source files
                          [corpus.py]
                               │
                               ▼
                     Extract and clean text
                          [ingest.py]
                               │
                               ▼
                 Create traceable evidence chunks
                           [chunk.py]
                               │
                               ▼
                   Build and query BM25 index
                           [index.py]
                               │
                               ▼
               Return cited evidence to the agent
                        [retriever.py]
```

## Source collection

The corpus can be populated through four official-source paths:

- `download_corpus.py` collects core SEC 10-Ks, 10-Qs, and earnings-release exhibits.
- `download_supplemental_corpus.py` collects supplemental SEC exhibits such as earnings
  presentations, prepared remarks/transcripts, and investor-day or special presentations.
- `download_ir_archive.py` discovers materials linked from official company investor-relations
  sites, including presentations, remarks, and transcripts.
- **Manual official-source intake:** a person downloads a document from an official SEC or
  company IR page; `prepare_manual_ir_metadata.py` then prepares its provenance metadata, and
  `import_manual_ir.py` validates and imports the reviewed file. These scripts are part of the
  corpus acquisition workflow, but they do not themselves download the document. Use this path
  when automated collection cannot retrieve a source or when a manual download is preferred.

`generate_catalog.py` is separate from acquisition: it creates a readable manifest inventory
for corpus auditing.

Downloader success does not guarantee that every expected document was found. Site changes,
blocked requests, and missing or ambiguous dates can require manual review. The core SEC
downloader requires an explicit `--output` destination.

### Collection runbook

The canonical manifests are already available in this workspace. Run the collectors below
only when intentionally refreshing or rebuilding the source corpus: they contact SEC/company
sites and write files and manifests to the selected output folders. Run commands from the
repository root. Keep the date window consistent across collectors.

1. **Set an identifying SEC User-Agent** (replace the example contact with a monitored
   project contact):

   ```powershell
   $env:SEC_USER_AGENT="FinSight academic project your-contact@example.com"
   ```

2. **Collect core SEC filings and earnings releases** (all configured tickers by default):

   ```powershell
   python -m finsight.agents.narrative.download_corpus `
     --from-date 2021-01-01 `
     --cutoff 2026-10-04 `
     --output data/narrative/raw/from_2021_final
   ```

3. **Collect SEC supplemental exhibits**, excluding files already in the core corpus:

   ```powershell
   python -m finsight.agents.narrative.download_supplemental_corpus `
     --from-date 2021-01-01 `
     --cutoff 2026-10-04 `
     --output data/narrative/raw/supplemental_from_2021_official_final `
     --exclude-manifest data/narrative/raw/from_2021_final/manifest.json
   ```

4. **Collect downloadable materials from official company IR sites:**

   ```powershell
   python -m finsight.agents.narrative.download_ir_archive `
     --from-date 2021-01-01 `
     --cutoff 2026-10-04 `
     --output data/narrative/raw/ir_from_2021_official_final
   ```

   Review its warnings and manifest. HTTP 403, DNS failures, unavailable archives, or
   undated files may require manual review; a successful run does not mean every company has
   complete coverage.

5. **Optionally import manually downloaded official files** if automation could not retrieve
   them. Put only reviewed official-source documents under
   `data/narrative/manual_inbox/<TICKER>/`, then run:

   ```powershell
   python -m finsight.agents.narrative.prepare_manual_ir_metadata
   ```

   Inspect each generated `import_metadata.json` and verify dates/source pages before importing
   a ticker. The helper scans the inbox (it is not a `--help`/dry-run command) and currently
   skips NVDA and TSLA. For each ticker with reviewed metadata, run:

   ```powershell
   python -m finsight.agents.narrative.import_manual_ir `
     "data/narrative/manual_inbox/<TICKER>/import_metadata.json" `
     --manifest data/narrative/raw/ir_from_2021_official_final/manifest.json `
     --exclude-manifest `
       data/narrative/raw/from_2021_final/manifest.json `
       data/narrative/raw/supplemental_from_2021_official_final/manifest.json `
     --skip-duplicates
   ```

6. **Review the resulting corpus catalog and manifest status:**

   ```powershell
   python -m finsight.agents.narrative.generate_catalog `
     data/narrative/raw/from_2021_final/manifest.json `
     data/narrative/processed/corpus_catalog.md `
     --supplemental-manifest data/narrative/raw/supplemental_from_2021_official_final/manifest.json `
     --ir-manifest data/narrative/raw/ir_from_2021_official_final/manifest.json
   ```

   Check counts by company/type, warnings, source URLs, publication dates, local file
   existence, and duplicate handling. The corpus loader validates local paths and hashes
   when processing begins. Do not index an undated document for as-of retrieval until its
   publication date is verified.

## Available source documents

Counts below are grouped by company and source corpus, based on the three canonical manifests
currently in the workspace. They are manifest records, not guaranteed unique events or unique
text; the same earnings event may have both an SEC and an IR copy.

The collection window starts in **2021** and has a manifest cutoff of **2026-10-04**. Dated
records currently span 2021-01-04 through 2026-10-02 in the core SEC corpus, 2021-03-02 through
2026-09-23 in supplemental SEC, and 2021-01-19 through 2026-09-23 in official IR. The IR
manifest also contains one record without a publication date, so it cannot be assigned to a
year or used for as-of retrieval until its date is verified.

| Company | Core SEC | Supplemental SEC | Official IR | Total records |
| --- | ---: | ---: | ---: | ---: |
| AAPL | 46 | 0 | 0 | 46 |
| AMC | 53 | 1 | 0 | 54 |
| BIRD | 38 | 0 | 19 | 57 |
| DNUT | 44 | 1 | 14 | 59 |
| GME | 48 | 0 | 0 | 48 |
| GOOGL | 47 | 0 | 20 | 67 |
| GPRO | 46 | 0 | 39 | 85 |
| KSS | 49 | 21 | 0 | 70 |
| MCD | 46 | 2 | 3 | 51 |
| MSFT | 46 | 0 | 26 | 72 |
| NFLX | 46 | 0 | 46 | 92 |
| NKE | 48 | 0 | 18 | 66 |
| NVDA | 47 | 1 | 57 | 105 |
| OPEN | 47 | 0 | 18 | 65 |
| TSLA | 70 | 1 | 22 | 93 |
| **Total** | **721** | **27** | **282** | **1,030** |

The manifests contain 10-Ks, 10-Qs, earnings releases, earnings presentations, earnings-call
transcripts, prepared remarks, investor-day presentations, and special presentations. The
supplemental/IR columns show how many records each company has from those corpus sources;
zero means no records are currently listed in that manifest, not proof that no official
documents exist.

For the NVDA pilot, the available set is 47 core SEC records, 1 supplemental SEC record,
and 57 official IR records. Image-only pages in the 2021 and 2022 NVDA Investor Day decks
remain an extraction limitation; the manifest count does not mean every page is searchable.

## Processing and retrieval scripts

These scripts process the files and metadata produced by source collection:

| Script/module | What it does (plain language) | How to run / output |
| --- | --- | --- |
| `corpus.py` | Reads the three canonical manifests, selects records for a company, skips records explicitly marked ineligible, normalizes source-tier values, checks each source file exists and matches its SHA-256 hash, and computes a corpus fingerprint. It is a library module, not a command-line script. | Called by `pilot.py`; no standalone command. Missing files or hash mismatches stop processing with an error. |
| `ingest.py` | Opens each HTML or PDF file and extracts readable text blocks. HTML sections and readable tables are retained; PDF page numbers are preserved. Pages with no extractable text generate warnings; OCR is not performed. | Called by `pilot.py`; no standalone command. Its blocks are written to `documents.jsonl`. |
| `chunk.py` | Splits each text block into smaller overlapping passages (“chunks”) while retaining the company, date, source URL, tier, section/page citation, and source document ID. Current defaults are 800 words per chunk with 120 words of overlap. | Called by `pilot.py`; no standalone command. Its output is `chunks.jsonl`. |
| `index.py` | Builds and searches a local JSON BM25 index. BM25 is keyword-based ranking: it scores passages that share important words with the query. It removes exact duplicate chunk text and filters by company ticker and cutoff date. | Called by `pilot.py` and `retriever.py`; no standalone command. Writes `index.json` and supports loading/searching it. This is not a vector database. |
| `retriever.py` | Wraps search results as the project's standard `EvidenceItem` objects so other agents can consume the passage and its citation. Its current stance is neutral because retrieval alone does not prove or disprove a claim. | Provides `CorpusRetriever` to the application; no standalone command. It defaults to an NVDA index unless configured. The app still selects the stub retriever by default. |
| `pilot.py` | A runnable end-to-end example that calls the modules above in order, then runs a sample query. The default is a two-document NVDA pilot; `--all-documents` processes all eligible NVDA records into a separate output set. | Run commands below. It creates JSONL document/chunk files, a BM25 index, index metadata, a processing report, and query results under `data/narrative/processed/`. |

### Run the processing workflow

Run these from the repository root. The corpus documents must already be downloaded and
listed in the canonical manifests. These commands do not download source documents.

**First, run the small two-document NVDA pilot:**

```powershell
python -m finsight.agents.narrative.pilot
```

By default, the pilot uses an NVDA 10-K and an NVDA earnings presentation. It writes:

```text
data/narrative/processed/NVDA/
  documents.jsonl          # extracted text blocks
  chunks.jsonl             # chunked passages with source/citation metadata
  processing_report.json   # counts, fingerprint, and extraction warnings
  query_results.json       # results for the default sample query

data/narrative/processed/indexes/NVDA/
  index.json               # searchable local BM25 index
  index_metadata.json      # index version, chunk settings, fingerprint, counts
```

**Then, after reviewing the pilot, process the full NVDA manifest set:**

```powershell
python -m finsight.agents.narrative.pilot --all-documents
```

This creates separate outputs under `processed/NVDA_full/` and
`processed/indexes/NVDA_full/`; it preserves the smaller pilot outputs. To change the sample
query, cutoff date, or number of results, pass CLI options, for example:

```powershell
python -m finsight.agents.narrative.pilot `
  --all-documents `
  --query "NVDA data center growth risks and customer concentration" `
  --cutoff 2025-07-31 `
  --top-k 5
```

The cutoff is an **as-of date**: documents published after it must not appear in results.

### What to check in the output

- `processing_report.json`: selected document count, extracted block count, chunk counts,
  duplicate count, corpus fingerprint, and any PDF extraction warnings.
- `chunks.jsonl`: each line is one passage; check `chunk_id`, ticker, publication date,
  source tier, `source_url`, `citation_span`, and `text`.
- `query_results.json`: confirm each result is for NVDA, is not after the requested cutoff,
  and includes a useful passage and source citation.
- `index_metadata.json`: note the corpus fingerprint and chunking settings; a changed corpus
  or chunking configuration requires rebuilding the index.

### Worked example: one query and one result

Run this example against the full NVDA corpus:

```powershell
python -m finsight.agents.narrative.pilot `
  --all-documents `
  --query "NVDA data center growth risks and customer concentration" `
  --cutoff 2025-07-31 `
  --top-k 5
```

The workflow saves the question and up to five matching passages in
`data/narrative/processed/NVDA_full/query_results.json`. In the current pilot output, the
first result is an NVDA 10-K dated 2025-02-26, under the section “Item 7. Management's
Discussion and Analysis.” Its `source_tier` is `TIER_1`, and its `source_url` points to the
official SEC filing. Because 2025-02-26 is before the requested 2025-07-31 cutoff, the result
passes the as-of-date check.

When reviewing a result, use `text` as the passage returned by search, `citation_span` to
locate it within the filing, and `source_url` to verify it against the original. The result is
a relevant passage to inspect—not an automatic conclusion that the question's claim is true.

The chunking, metadata, and retrieval details are documented in
[`narrative_chunking_metadata_retrieval_schema.md`](../reference/narrative_chunking_metadata_retrieval_schema.md).

## Current gaps / next work

### Planned for Week 2

Build the first claim-level Narrative Evidence Agent on the existing NVDA corpus, add a
persistent vector-store path, and compare these retrieval experiments using the same queries,
company/date filters, corpus fingerprint, and top-k settings:

| Experiment | Method | Week 2 action |
| --- | --- | --- |
| E1 | BM25 keyword baseline | Keep as the control; save baseline rankings. The search implementation already exists. |
| E2 | Dense retrieval (embeddings) | Run the real embedding model and connect dense search to the persistent vector store. |
| E3 | Hybrid BM25 + dense | Run the existing fusion method under the same query and cutoff conditions. |
| E4 | Hybrid + cross-encoder reranking | Run the existing reranker and inspect whether it improves the hybrid ranking. |

Then implement the first claim-level workflow with separate support and challenge searches,
correct `sub_claim_id` and stance assignment, citations, ticker/date filtering, and tests. The
Week 2 comparison is an initial review—not a final choice of best method; that requires a
labeled evaluation set and retrieval metrics. See [`summary_week_2.md`](summary_week_2.md) for
the detailed tasks and commands.

After the NVDA workflow passes its acceptance checks, use the same validated NVIDIA flow as
the template for additional companies: load their canonical manifests, validate files and
metadata, process/chunk, build the company index, and test company/date filtering. Expand
company by company and record any source or extraction exceptions rather than changing the
pipeline silently.

### Longer-term follow-up

- **Evaluate retrieval on labeled development and held-out query sets.** Write realistic
  company questions and have a reviewer label which source passages are relevant. Use the
  development questions to tune settings. Keep the held-out questions unseen until the end,
  then compare BM25, dense, hybrid, and reranked search on them. This helps show whether one
  approach finds useful evidence on new questions, not just the examples used while building
  it. Compare relevance, counter-evidence, citation quality, and cutoff-date correctness.
- **Decide how to handle image-only pages in the NVDA 2021 and 2022 Investor Day decks.** The
  current PDF text extractor cannot read text that exists only inside slide images. OCR
  (optical character recognition) could convert those images into searchable text, but its
  output must be checked for errors. The other valid choice is to record that this content is
  excluded from retrieval. Until one choice is made, the indexed corpus has a known content
  gap for those decks.
- **Complete multi-company index selection and test ticker isolation.** Decide how the app
  chooses the correct company’s index for a query, then test that an NVDA question cannot
  return AAPL or another company’s evidence (and vice versa). Also verify every company’s
  as-of cutoff. Do these checks before enabling the workflow broadly, so evidence cannot be
  mixed across companies or time periods.
- **Continue auditing canonical manifests after source collection.** A manifest is the
  machine-readable list of accepted source files and their metadata. After a collection or
  manual import, check that expected IR/supplemental documents are present, each file exists
  and matches its recorded hash, publication dates and source URLs are valid, duplicate files
  are handled, and warnings or undated records are reviewed. This makes collection gaps and
  provenance problems visible before they affect retrieval.

