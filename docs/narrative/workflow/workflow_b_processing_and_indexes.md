# Workflow B: Prepare searchable company evidence

[Back to the full Narrative workflow](summary_full_narrative_agent_flow.md) | [Shared setup](summary_full_narrative_agent_flow.md#3-before-running)

For saved results, see the [main summary](summary_full_narrative_agent_flow.md).

## Scripts and outputs

Paths in this table are relative to `finsight/agents/narrative/` unless stated otherwise. Steps start at 1 within this workflow. Use workflow letters A-E for the overall flow. See the [script inventory](../reference/narrative_script_inventory.md) for per-file descriptions.

| Step | Workflow blocks | Entry point / supporting modules | Main output |
|---|---|---|---|
| 1. Parse, chunk and index text | W2-C, W2-J | `corpus/processing/process_corpus.py`, `ingest.py`, `chunk.py`, `index.py` | Parsed source blocks, canonical chunks and BM25 |
| 2. Extract PDFs/charts and promote | W2-D to W2-J | `corpus/pdf/pdf_chart_processing.py` and PDF extraction helpers | Extraction artifacts and promoted company content |
| 3. Build matching vectors/strategy artifacts | W2-K | `retrieval/build_vector_databases.py`; `scripts/narrative/retrieval/build_full_chunking_indexes.py` for C1–C4 | Selected chunks, BM25 and matching Qdrant vectors; no retrieval-quality scores |

## Workflow and responsibilities

```mermaid
flowchart LR
    Sources["Reviewed source files and manifests from Workflow A"] --> B["W2-B: load manifests and validate source files and date metadata"]
    B --> C["W2-C: parse native text into source blocks"]
    B --> D["W2-D: audit PDF pages when applicable"]
    D --> E["W2-E: extract native PDF text"]
    D --> F["W2-F: layout extraction and optional OCR"]
    F --> G["W2-G: tables and chart candidates"]
    G --> H["W2-H: optional chart vision extraction"]
    E --> I["W2-I: extraction reports and page coverage"]
    F --> I
    G --> I
    H --> I
    C --> J["W2-J: promote content, chunk and build BM25"]
    I --> J
    J --> V{"Dense, hybrid or R4 needed?"}
    V -->|Yes| K["W2-K: build matching embedding vectors"]
    V -->|BM25 only| Ready["Prepared artifacts ready for inference or evaluation"]
    K --> Ready
```

Use existing complete, matching builds when available. W2-K is required for
dense, hybrid and R4, but not for BM25. Each strategy has its own chunks and
matching indexes. A changed query or a new test dataset alone does not require
rechunking or rebuilding vectors.

## Steps

**Workflow B — preparation.** Run these steps when building/changing artifacts.
If matching chunks/indexes are already available, start directly with the
per-claim commands in [Section 5](summary_full_narrative_agent_flow.md#5-run-rag-for-a-claim).

### Step 1. Create source blocks, chunks and BM25

Start with one company:

```powershell
python -u -m finsight.agents.narrative.corpus.processing.process_corpus --ticker META
```

Use `--all` instead of `--ticker META` for every eligible company. Existing output folders are skipped; use `--rebuild` only when intentionally refreshing them. The build uses promoted PDF enrichment when available. Outputs are under `data/narrative/processed/META/`:

```text
documents.jsonl             parsed source blocks with provenance
chunks.jsonl                retrieval passages and stable evidence IDs
processing_report.json      processing counts and warnings
indexes/index.json          BM25 data and passages
indexes/index_metadata.json build metadata
```

### Step 2. Enrich PDFs/charts, then promote into company passages

```powershell
python -u -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --ticker META --ollama-model qwen3-vl:4b-instruct --chart-workers 1 --chart-timeout 900 --promote
```

For OpenAI extraction, replace `--ollama-model qwen3-vl:4b-instruct` with `--openai-model gpt-5.4`; requests may incur API charges. Use `--all` instead of the ticker for an all-company batch. The pipeline reuses valid source-matched artifacts; failures or changed inputs may require work. `--promote` merges applicable extracted content into canonical company blocks/chunks and BM25.

If extraction is already complete and only promotion is needed:

```powershell
python -u -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --ticker META --promote-only
```

Inspect saved status:

```powershell
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --status
```

Per-document artifacts are under `data/narrative/processed/<TICKER>/experiments/pdf_chart_processing/<DOCUMENT_ID>/`. Available files depend on page routing; not every PDF produces charts or enhanced Markdown. A final report establishes recorded extraction status, not independently verified financial accuracy.

### Step 3. Build or refresh vectors after final promotion

```powershell
python -u -m finsight.agents.narrative.retrieval.build_vector_databases --ticker META
```

Use `--all` for every company. The builder reuses current vector indexes and rebuilds stale ones; `--rebuild` forces regeneration. Outputs include `data/narrative/processed/<TICKER>/indexes/qdrant/qdrant_metadata.json` and `storage/`. BM25 and vectors must describe the same passages for dense/hybrid retrieval. Avoid simultaneous writers to the same local Qdrant storage.

## Source metadata and traceability

### From source files to indexed passages

The corpus loader selects the company's registered documents and checks local source paths and file SHA-256 hashes before processing. It reads the core SEC, supplemental SEC, official IR and event-report manifests; the third-party transcript manifest is optional. Keep third-party provenance distinct from official sources.

HTML parsing removes scripts, hidden content and common navigation, preserves document order and labels table rows with their headers. Native PDF text retains page references; PDFs needing richer extraction follow Step 2. These transformations do not prove that every table, chart or OCR value was extracted correctly.

Each parsed block is chunked within its source boundary. Canonical chunk IDs combine the document ID, section slug, ordinal and a text-hash prefix. The canonical chunker handles charts as whole units and tables by rows/headers, rather than treating all content as plain word windows. C1-C4 have their own saved representations; use the selected build's IDs and source map when comparing strategies.

### Metadata retained with chunks

These fields are defined in [`NarrativeChunk`](../../../finsight/agents/narrative/corpus/processing/chunk.py) and carried into the saved passage/index records:

| Fields | Purpose |
|---|---|
| `chunk_id`, `document_id` | Identify the retrieval passage and its source document. |
| `company_ticker`, `company_name`, `document_type` | Keep company and document scope explicit. |
| `publication_date` | Effective source publication date used for cutoff filtering; separate from the reporting period. |
| `publication_date_is_estimated`, `publication_date_precision`, `publication_date_basis` | Disclose estimated dates and the basis for the effective date. |
| `period` | Reporting period, when supplied; may be null. |
| `source_tier`, `source_url` | Source classification and original URL; missing URLs must not be invented. |
| `local_path`, `original_local_path` | Saved source path and an earlier original path where one exists, such as a converted presentation. |
| `section`, `page`, `citation_span` | Human-readable source location; PDF pages and HTML section locations are not necessarily exact character spans. |
| `text`, `text_sha256` | Stored passage text and its SHA-256. |
| `source_format`, `extraction_method`, `content_type` | How the content was obtained and whether it represents narrative, a table or a chart. |
| `chart_id`, `source_ref`, `validation_status` | Additional chart/source locators and recorded extraction/validation status, where present. |

Manifest-level `sha256`, `retrieval_eligible`, `estimated_publication_date` and `estimated_publication_date_basis` are intake fields, not all separate chunk fields. The loader chooses a recorded publication date or a supplied estimate when eligible under its rules, excludes records without an effective date, and carries the effective date plus its qualifications into processing. Review estimates during collection; an estimated date does not certify the actual release day.

### Source maps, hashes and duplicate handling

| Saved information | How it is used |
|---|---|
| Index `schema_version`, `corpus_fingerprint`, chunking settings and passage counts | Identify a compatible index build and its contents. |
| `text_sha256` versus manifest file `sha256` | The first identifies passage text; the second verifies the saved source file. They are different hashes. |
| Strategy `source_map.jsonl`: `chunk_id`, `document_id`, `block_line_1based` | Link each strategy passage to a parsed source block. |
| `body_char_span`, `context_char_spans` | Identify body and copied context within the parsed block using zero-based, end-exclusive offsets. They do not locate bytes in the original PDF/HTML. |
| `parsed_sha256` and build reports | Detect changed source text or mismatched chunks, mappings and indexes. |

Canonical index construction removes exact duplicate text by default. The full C1-C4 builders retain source duplicates (`deduplicate_text=False`) so that another source or an earlier publication date is not lost. Neither is semantic deduplication. Request-time deduplication of repeated retrieved IDs is a separate step.

BM25 and vector metadata must describe the same passage set. Matching IDs/hashes establish technical consistency, not source completeness or retrieval relevance. See [Workflow C's evidence mapping](workflow_c_rag.md#how-source-metadata-becomes-evidence) for the fields returned to callers.

## Different chunking strategies

**Canonical** means the existing company build stored under `data/narrative/processed/<TICKER>/`. **C1-C4** name four chunking strategies with separate builds. They can reuse the same parsed documents while dividing the text into different passages.

| Build | How it creates chunks |
|---|---|
| Canonical | Uses the existing chunker with an 800-word target and 120-word overlap setting, plus special handling for tables and charts. |
| C1 | Uses strict 800-word windows with 120-word overlap within each source block, including table/chart text. |
| C2 | Uses strict 300-word windows with 50-word overlap within each source block. |
| C3 | Uses strict 400-word windows with 60-word overlap within each source block. |
| C4 | Uses adaptive, structure-aware splitting with candidate token targets of 128, 192 and 256, subject to the embedding tokenizer's hard limit. |

Canonical and C1 share the 800/120 settings, but their table/chart handling can produce different passages. Do not treat them as identical builds. Fixed-window chunks can be shorter at the end of a source block; word counts do not guarantee compliance with an embedding token limit.

C1-C4 builds are stored under `data/narrative/processed/chunking_strategies/<STRATEGY>/<build-version>/<TICKER>/`. Each has its own chunks and matching indexes. Selecting a strategy chooses those saved artifacts; selecting BM25, dense, hybrid or R4 chooses how to search them. Full strategy builds can be used for ordinary RAG as well as evaluation.

Workflow B prepares these artifacts. [Workflow C](workflow_c_rag.md#step-1-select-chunks-and-indexes-for-a-rag-run) selects them for a request, and [Workflow D](workflow_d_evaluation.md) compares their quality. The table below compares the canonical build with the full C1-C4 builds.

## Recorded build counts: canonical and C1-C4

| Item | Canonical | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|---:|
| Chunking strategy | 800-word target / 120-word overlap; special table/chart handling | Fixed 800 words / 120 overlap | Fixed 300 words / 50 overlap | Fixed 400 words / 60 overlap | Adaptive; token targets 128 / 192 / 256 |
| Companies | 16 | 16 | 16 | 16 | 16 |
| Source documents | 1,201 | 1,201 | 1,201 | 1,201 | 1,201 |
| Processed source blocks | 10,424 | 10,424 | 10,424 | 10,424 | 10,424 |
| Generated chunks | 23,856 | 24,089 | 52,732 | 40,680 | 90,361 |
| Duplicate chunks excluded from BM25 | 874 | 0 | 0 | 0 | 0 |
| Final BM25 passages | 22,982 | 24,089 | 52,732 | 40,680 | 90,361 |
| Saved Qdrant vectors | 22,982 | 24,089 | 52,732 | 40,680 | 90,361 |
| Companies with matching BM25/vector metadata | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 |
| Embedding model | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| Vector dimensions | 384 | 384 | 384 | 384 | 384 |
| Vector similarity | Cosine | Cosine | Cosine | Cosine | Cosine |
| Embedding input limit (tokens) | 256 | 256 | 256 | 256 | 256 |
| Longest chunk before embedding (tokens) | Not recorded | 2,877 | 2,626 | 2,626 | 256 |
| Indexed chunks exceeding the embedding limit | Not recorded | 18,609 | 44,722 | 33,847 | 0 |
| Saved embedding overflow policy | Not recorded | Truncate embedding input | Truncate embedding input | Truncate embedding input | Enforce token limit during chunking |

All five builds use `sentence-transformers/all-MiniLM-L6-v2`; the model, vector dimensions and cosine similarity match across all 16 companies in each build. The effective 256-token limit is recorded in the C1-C4 configurations and confirmed in the cached model configuration. Canonical metadata does not record overflow counts or an explicit overflow policy; "Not recorded" does not mean zero overflow.

C1-C3 keep complete passages in BM25 and source files, but embedding inputs exceeding 256 tokens are truncated. Their vectors therefore represent only the model-visible beginning of oversized passages. C4 produces shorter token-valid chunks, so its saved reports show no embedding overflow and more vectors. This explains an embedding difference between the strategies; it does not establish which retrieves the best evidence. The embedding input limit is separate from the cross-encoder reranker's limit and the generation LLM's context/output limits.

Canonical removes exact duplicate passages during indexing. The C1-C4 builders retain exact-text duplicates with their separate source provenance, so zero excluded duplicates does not mean no repeated text exists. Different chunk sizes, overlaps, table/chart handling and duplicate policies all affect these counts; more chunks do not establish better retrieval quality.

All strategies cover the same 1,201 documents. Retrieval only searches documents published on or before the request's cutoff date.

Counts were checked by reading saved reports, counting `chunks.jsonl` entries and indexed passages, and comparing ordered chunk IDs, corpus fingerprints and vector counts against Qdrant metadata. Strategy totals also match their aggregate build reports. Physical Qdrant collections were not opened or recounted; no builds, retrieval runs or model calls were executed.

Source files:

- Canonical: `data/narrative/processed/<TICKER>/processing_report.json`, `chunks.jsonl`, `indexes/index.json` and `indexes/qdrant/qdrant_metadata.json`.
- C1-C4: `data/narrative/processed/chunking_strategies/<STRATEGY>/full_v1/build_report.json`, plus each `<TICKER>/build_report.json`, `chunks.jsonl`, `indexes/index.json` and `indexes/qdrant/qdrant_metadata.json`.
- Embedding settings and token statistics: each strategy's `configuration.json`, company `build_report.json` and `indexes/qdrant/qdrant_metadata.json`. The cached model's `sentence_bert_config.json` specifies its effective input limit.

Matching metadata does not establish that embedding inputs avoided truncation. Canonical and C1-C3 chunks can exceed the embedding model's 256-token limit; the saved C1-C3 builds explicitly use `truncate`. C4 enforces its tokenizer-based limit. See [Section 4.1 of the main summary](summary_full_narrative_agent_flow.md#41-saved-chunking-and-embedding-results) for canonical per-company counts and limitations.

Full C1-C4 strategy builds are described below. Run either the canonical vector builder above or the appropriate strategy builder for the artifacts you need; both are not required for every claim.

## Chunk and index storage

Full-corpus strategy outputs live under `data/narrative/processed/chunking_strategies/`.
The existing canonical company folders remain in place; canonical 800/120 is
not renamed to experimental C1. Historical experiments are preserved separately.

```text
data/narrative/processed/
├── <TICKER>/
│   ├── documents.jsonl                 shared parsed source blocks
│   ├── chunks.jsonl                    canonical 800/120 chunks
│   └── indexes/                        canonical BM25 and vectors
└── chunking_strategies/
    └── <C1|C2|C3|C4>/<build-version>/
        ├── configuration.json          strategy, model and processing identities
        └── <TICKER>/
            ├── chunks.jsonl            exact indexed passages and provenance
            ├── source_map.jsonl        source block/body/context locations
            ├── chunking_report.json
            ├── build_report.json       completed chunk/vector build report
            ├── layout_manifest.json    hashes for chunks/map/index
            ├── token_overflow.jsonl    fixed arms C1–C3
            ├── candidate_scores.jsonl.gz  C4 adaptive candidate diagnostics
            └── indexes/
                ├── index.json          BM25 data and complete passage text
                └── qdrant/
                    ├── qdrant_metadata.json
                    └── storage/
```

All four strategies have a full-corpus builder through
`scripts/narrative/retrieval/build_full_chunking_indexes.py`. C1 uses strict
800-word/120-overlap windows, C2 uses 300/50, C3 uses 400/60, and C4 delegates to
the existing adaptive builder. The table shows strategy-specific optional files;
not every strategy creates every diagnostic file. `full_v1` is a build-version
name, not a publication cutoff or a retrieval policy.

`experiments/` keeps frozen experiment artifacts, labels, configurations, metrics
and reports. Building full indexes does not refresh Recall/MRR results or rewrite
historical experiments. Full build coverage includes current eligible processed
documents; each retrieval request still enforces its own publication cutoff.

### Build full C1-C4 indexes

Use existing complete builds when their source/configuration identities match.
Run from the repository root after restoring canonical processed source blocks,
indexes and the cached local embedding model. These commands are instructions,
not a record of a new build executed while updating this guide.

All companies, all four strategies:

```powershell
python -u -m scripts.narrative.retrieval.build_full_chunking_indexes --all --strategies C1 C2 C3 C4 --build-version full_v1 --threads 4 --embedding-overflow truncate --resume
```

One company, all four strategies:

```powershell
python -u -m scripts.narrative.retrieval.build_full_chunking_indexes --ticker MSFT --strategies C1 C2 C3 C4 --build-version full_v1 --threads 2 --embedding-overflow truncate --resume
```

Replace `MSFT` with another valid ticker when needed. Select just one arm with
`--strategies C4` or another listed strategy. The output root is
`data/narrative/processed/chunking_strategies/`; the builder creates the strategy
and version folders underneath it.

`--embedding-overflow truncate` explicitly permits C1–C3 inputs longer than the
embedding model's token limit to be shortened for embedding. Complete chunks and
BM25 passage text remain stored. This limitation must accompany any dense/hybrid
comparison. The default `error` policy rejects overflow; it does not resize fixed
chunks. C4 applies its own tokenizer-based hard limit. Changing an existing
build's overflow/configuration/code/source identity requires a new build version.

`--resume` verifies compatible checkpoints and reuses completed work. It can
reject stale or interrupted artifacts; it does not mean every existing folder
is safe to reuse. Review the error and use a new version when required rather
than deleting saved data to bypass checks.

The C4-only entry point remains available and is also used by the unified builder:

```powershell
python -u -m scripts.narrative.retrieval.build_full_c4_indexes --ticker MSFT --output-dir data/narrative/processed/chunking_strategies/C4/full_v1 --threads 2 --resume
```

The builders use current eligible processed documents without the old benchmark
date filter. They reuse existing parsed text and do not reparse PDFs. They build
matching BM25 and vector indexes using local embedding inference, with no
generation LLM/API call. They do not run evaluation cases or generate new
Recall/MRR scores. The optional retrieval dependencies and cached model must
already be available on the laptop.

Initialize the shared destination with one invocation before launching separate
company processes. Then use different companies per process, with --threads 2
as a starting point. Do not run the same company twice or overlap --all with
individual company builds. A later compatible `--all --resume` run verifies
company checkpoints and writes aggregate build reports. The unified builder
writes a root `<build-version>_build_report.json` for `--all`, or
`<build-version>_<TICKER>_build_report.json` for a single company; each strategy
also retains its own company and aggregate reports.

After a build is complete, follow [Workflow C: select chunks and indexes for a RAG run](workflow_c_rag.md#step-1-select-chunks-and-indexes-for-a-rag-run). For comparing strategies and measuring quality, follow [Workflow D](workflow_d_evaluation.md#recorded-input-and-build-status).
