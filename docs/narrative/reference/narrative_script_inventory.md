# Narrative pipeline: Python file reference

Covers **42 implementation modules and 10 package initializers (52 Python files)** in `finsight/agents/narrative/`. For execution order, commands and results, use [the full workflow](../workflow/summary_full_narrative_agent_flow.md). Batch build and current CSV evaluation helpers are listed separately in [the scripts inventory](narrative_scripts.md).

## 1. How to read this inventory

Paths below are relative to `finsight/agents/narrative/`. Package initializers support imports; the root initializer exposes the real agent/retriever and the module directory.

| Folder | Responsibility | Implementation modules |
|---|---|---:|
| Root | Orchestrator Narrative Agent and scoped retriever adapters | 2 |
| `corpus/acquisition/` | SEC/IR downloads and manual source import | 7 |
| `corpus/processing/` | Corpus loading, parsing, fixed/adaptive chunking and BM25 | 6 |
| `corpus/pdf/` | PDF routing, OCR, charts and enrichment | 6 |
| `corpus/maintenance/` | Dates, catalogs and specialist build/pilot tools | 6 |
| `retrieval/` | Qdrant, retrieval backends, embeddings/reranking and priority helpers | 5 |
| `rag/` | Evidence collection, cited summaries and retained Ollama/baseline helpers | 4 |
| `evaluation/` | Optional JSONL retrieval comparisons and provenance tools | 4 |
| `demo/` | Demo agent/retriever compatibility used by tests | 2 |

The current 126-claim evaluation uses `scripts/narrative/evaluation/`; the optional JSONL commands in this package are separate tools. No deleted acquisition wrappers, FAISS module or historical chunking/priority runners are required for the current flow.

**CLI** identifies a command run from the repository root as `python -m finsight.agents.narrative.<group>.<name>`. **Library** identifies an imported helper, usually not run separately. **Maintenance** commands collect or update sources. **Diagnostic/compatibility** files preserve earlier experiments and demo behavior.

Helper modules, stubs and acquisition tools can be required by other modules even when they are not run directly. Keep credentials in local configuration and corpus/index artifacts under `data/`.

## Orchestrator entry points

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [stub.py](../../../finsight/agents/narrative/stub.py) | Registry adapter | Real Narrative Agent: collect candidates and prepare selected evidence, exact facts and a cited summary. | SubClaim/cutoff/feedback -> original EvidenceItems; preparation report saved separately. No stance/verdict. |
| [retriever_stub.py](../../../finsight/agents/narrative/retriever_stub.py) | Registry adapter | Real scoped retriever; infers one company or accepts an explicit ticker. | Query/top-K/cutoff -> original EvidenceItems. No generated summary. |

The filenames retain compatibility; these are active implementations. [Workflow E](../workflow/workflow_e_orchestrator_integration.md) documents registry selection, C1/BM25 end-to-end defaults and overrides.

## 2. Corpus preparation and BM25: 5 files

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [corpus/processing/corpus.py](../../../finsight/agents/narrative/corpus/processing/corpus.py) | Library | Defines `CorpusDocument`; loads canonical manifests for a ticker; normalizes source tiers/dates; checks files/hashes and calculates source fingerprints. | Manifests -> normalized source records and validation results. |
| [corpus/processing/ingest.py](../../../finsight/agents/narrative/corpus/processing/ingest.py) | Library | Parses HTML/inline-XBRL, PDF and Markdown into ordered `ProcessedBlock` records. Preserves sections/pages, cleans document furniture and makes tables readable. Rich OCR/chart extraction is separate. | Source file/metadata -> text/table/chart blocks and warnings. |
| [corpus/processing/chunk.py](../../../finsight/agents/narrative/corpus/processing/chunk.py) | Library | Defines `NarrativeChunk`; creates deterministic passages with evidence IDs and source metadata. Supports word-size and paragraph/table boundary handling; distinct from experimental C4 selection. | Blocks -> passages. Canonical default target: 800 words / 120 overlap. |
| [corpus/processing/index.py](../../../finsight/agents/narrative/corpus/processing/index.py) | Library | Builds/loads/searches the BM25 JSON index with company/date filtering and citation-location diversity. Does not require embeddings. | Passages -> index; query -> ranked passages. |
| [corpus/processing/process_corpus.py](../../../finsight/agents/narrative/corpus/processing/process_corpus.py) | CLI | Main company text build. Loads sources and promoted enrichment, extracts/reuses blocks, chunks and builds BM25 with staged replacement. | `--ticker`/`--all` -> company `documents.jsonl`, `chunks.jsonl`, `processing_report.json` and `indexes/`. Existing output is skipped unless `--rebuild` is supplied. |

## 3. PDF, OCR and charts: 6 files

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [corpus/pdf/pdf_chart_processing.py](../../../finsight/agents/narrative/corpus/pdf/pdf_chart_processing.py) | CLI | Main resumable PDF workflow: page audit/routing, native/Docling/OCR extraction, chart processing, checkpoints and optional canonical promotion/vector refresh. | Ticker/document selection -> company `experiments/pdf_chart_processing/` artifacts/reports. `--promote`/`--promote-only` merges applicable content; `--status` reads saved state; `--usage` reads tracked chart API usage. |
| [corpus/pdf/docling_charts.py](../../../finsight/agents/narrative/corpus/pdf/docling_charts.py) | Library + CLI | Lower-level single-PDF workflow. Creates/reuses Docling exports, detects/crops chart regions, tries positioned/vision extraction, validates/caches records and merges results into enhanced exports. | Positional PDF/output directory plus optional matching Docling exports -> crops, structured charts, enhanced Markdown/JSON and reports. Provider can be OpenAI, Ollama or none. |
| [corpus/pdf/chart_models.py](../../../finsight/agents/narrative/corpus/pdf/chart_models.py) | Library | Pydantic contracts for chart geometry, source provenance, legends, data points, extraction/validation results, usage and cache records. | Payloads -> typed structured records. No model calls by itself. |
| [corpus/pdf/chart_vision.py](../../../finsight/agents/narrative/corpus/pdf/chart_vision.py) | Library | Vision request/response interfaces and OpenAI/Ollama chart extraction adapters. | Cropped image/context -> structured chart response and usage; sends provider requests. |
| [corpus/pdf/chart_usage.py](../../../finsight/agents/narrative/corpus/pdf/chart_usage.py) | Library | Records new OpenAI chart requests, failures, returned tokens and available cost estimates; aggregates local tracked usage, excluding cache reuse. | Request events -> local usage records/summaries. Not account credit or complete team billing. |
| [corpus/pdf/pdf_enrichment.py](../../../finsight/agents/narrative/corpus/pdf/pdf_enrichment.py) | Library | Converts rich PDF exports to page-cited blocks, loads promoted enrichment and incorporates enrichment identity into fingerprints. | Enriched artifacts/promotion metadata -> source blocks and effective corpus fingerprint. |

## 4. Persistent vectors, retrieval and RAG: 7 files

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [retrieval/build_vector_databases.py](../../../finsight/agents/narrative/retrieval/build_vector_databases.py) | CLI | Builds/validates persistent local Qdrant per company from final BM25 passages. Reuses current stores and replaces stale ones; supports forced `--rebuild`. | `--ticker`/`--all` -> company `indexes/qdrant/storage/` and `qdrant_metadata.json`. |
| [retrieval/qdrant_index.py](../../../finsight/agents/narrative/retrieval/qdrant_index.py) | Library | Builds/loads Qdrant, validates model/dimension/fingerprint identity and implements dense/RRF/hybrid search. | Passages/embeddings -> persistent vectors; query -> rankings. One owner at a time for a local storage folder. |
| [retrieval/retriever.py](../../../finsight/agents/narrative/retrieval/retriever.py) | Library + CLI | `CorpusRetriever` adapts BM25/dense/hybrid results into shared `EvidenceItem` records with company/date scope and citation provenance. | Ticker/query/cutoff/backend -> neutral cited evidence, printed or saved with `--output`. No final answer generation. |
| [rag/collect_evidence.py](../../../finsight/agents/narrative/rag/collect_evidence.py) | Library + CLI | Raw `NarrativeEvidenceAgent.gather` builds support/counter queries, retrieves/deduplicates original passages and checks source provenance. Its CLI can also prepare a cited summary. | SubClaim/ticker/cutoff -> unassessed original evidence; no stance/verdict decisions. |
| [rag/basic_rag.py](../../../finsight/agents/narrative/rag/basic_rag.py) | Library + CLI | Retrieves candidates and invokes shared LLM evidence preparation. No stance/claim verdict. | Relevant original sources, exact facts, cited summary, quote validation and usage; explicit --retrieval-only opt-out. |
| [rag/evidence_summary.py](../../../finsight/agents/narrative/rag/evidence_summary.py) | Shared library | One LLM call for relevance selection, extractive facts and a cited evidence summary on existing passages. | Validates IDs, exact quotes and summary references. Invalid output retains original evidence and records failure. |
| [rag/verdict_baseline.py](../../../finsight/agents/narrative/rag/verdict_baseline.py) | Shared helper + historical CLI | Retains historical verdict generation and the Ollama completion adapter used by current evidence preparation, trace capture and local RAGAS. | Keep for current local-model execution even when the historical verdict CLI is not used. |

## 5. Shared chunking/retrieval helpers and optional validation: 7 files

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [corpus/processing/adaptive_chunking.py](../../../finsight/agents/narrative/corpus/processing/adaptive_chunking.py) | Shared library | Generates token-bounded C4 plans, validates content/provenance and records plan selection. | Source blocks/tokenizer/config -> selected chunks and trace. Called by `scripts/narrative/retrieval/build_full_c4_indexes.py`; no LLM. |
| [retrieval/source_priority.py](../../../finsight/agents/narrative/retrieval/source_priority.py) | Library; experimental | Assigns underlying document types High/Medium. Supports P0 traces, P1 expansion while sufficiency is unknown and bounded P2 near-tie promotions with period guards. Priority is separate from source tier. | Ranking/aliases/config -> priority assignments and policy/rank traces. Not activated in the Basic RAG CLI. |
| [retrieval/retrieval_experiments.py](../../../finsight/agents/narrative/retrieval/retrieval_experiments.py) | Library | Shared Sentence Transformers embedding/reranking backends, compressed dense cache and `FiveWayRetriever`: BM25, dense, hybrid, BM25+cross-encoder, hybrid+cross-encoder. | Existing BM25 payload/local models -> experimental rankings/caches. Experimental and canonical hybrid variants differ. |
| [evaluation/compare_retrieval.py](../../../finsight/agents/narrative/evaluation/compare_retrieval.py) | CLI; diagnostic | Inspects five methods for one query against one company index. This is not the complete scored benchmark. | Query/cutoff/index/model settings -> per-method results, optionally saved with `--output`. |
| [evaluation/run_retrieval_comparison.py](../../../finsight/agents/narrative/evaluation/run_retrieval_comparison.py) | CLI | Canonical multi-company E1-E4 runner: BM25, dense, hybrid and hybrid+cross-encoder using recorded support/counter queries and reference IDs. | Query JSONL/indexes -> combined `results.json` and company caches. `--output-dir` isolates reports, not all caches/inputs. |
| [evaluation/summarize_retrieval_comparison.py](../../../finsight/agents/narrative/evaluation/summarize_retrieval_comparison.py) | CLI | Audits/scores saved rankings and their source/company/cutoff/query identities. Does not rerun retrieval or load embedding models. | `--results`, `--queries` -> adjacent `evaluation.json` and `per_query.csv`. |
| [evaluation/validate_retrieval_provenance.py](../../../finsight/agents/narrative/evaluation/validate_retrieval_provenance.py) | CLI; validation | **Retrieval Provenance Validation**. Runs support/counter BM25/dense/hybrid searches and checks evidence mapping and temporal/company isolation. | Queries/indexes -> `data/narrative/experiments/basic_rag/task5_smoke.json` and evidence artifacts. No generated-answer evaluation. |

## 6. Acquisition and manual import: 7 files

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [corpus/acquisition/download_corpus.py](../../../finsight/agents/narrative/corpus/acquisition/download_corpus.py) | CLI; maintenance | Downloads core SEC annual/quarterly filings and earnings-release exhibits. Also provides the SEC client/acquisition helpers used by other collectors. | Tickers/date range/SEC identification and required `--output` -> source files/manifest. Makes network requests. |
| [corpus/acquisition/download_supplemental_corpus.py](../../../finsight/agents/narrative/corpus/acquisition/download_supplemental_corpus.py) | CLI; maintenance | Finds SEC-hosted supplemental exhibits: decks, prepared remarks, transcripts and other presentations; can exclude known documents. | Date range/tickers/destination/exclusion manifests -> supplemental files and manifest. |
| [corpus/acquisition/download_ir_archive.py](../../../finsight/agents/narrative/corpus/acquisition/download_ir_archive.py) | CLI; maintenance | Discovers documents linked on official company IR pages, with company-specific archive handling, dates and retry settings. | Official pages/tickers/date window/output -> downloads, IR manifest and discovery/review information. Unresolved dates require metadata follow-up. |
| [corpus/acquisition/download_event_reports.py](../../../finsight/agents/narrative/corpus/acquisition/download_event_reports.py) | CLI; maintenance | Discovers/downloads primary SEC 8-K/6-K bodies. Supports discovery-only and reuse of reviewed discovery. Requires appropriate SEC client identification. | Tickers/date range -> intake files/discovery/manifest. Canonical registration is separate. |
| [corpus/acquisition/download_ir_event_reports.py](../../../finsight/agents/narrative/corpus/acquisition/download_ir_event_reports.py) | CLI; maintenance | Downloads public Q4 filing-platform copies of main event reports and extracts primary content. Records third-party provenance, distinct from direct SEC acquisition. | Company feeds/date range -> intake discovery/files and `manifest_event_ir.json`. Canonical registration is separate. |
| [corpus/acquisition/prepare_manual_ir_metadata.py](../../../finsight/agents/narrative/corpus/acquisition/prepare_manual_ir_metadata.py) | CLI; maintenance | Reads the configured manual inbox, Windows download provenance, known hashes and project date mappings. Has no argument parser; uses module constants. | Inbox files -> per-ticker `import_metadata.json` and skipped-file report. Review before import. |
| [corpus/acquisition/import_manual_ir.py](../../../finsight/agents/narrative/corpus/acquisition/import_manual_ir.py) | CLI; maintenance | Checks manually prepared metadata, hashes, dates/source policy and duplicates; copies accepted files and updates the chosen manifest. | Positional inbox, `--manifest` and optional exclusions -> canonical file copies and changed manifest. |

## 7. Metadata maintenance and catalog: 4 files

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [corpus/maintenance/publication_dates.py](../../../finsight/agents/narrative/corpus/maintenance/publication_dates.py) | Library | Proposes dates from quarter/year labels and core filing calendars, retaining an explicitly estimated basis. | Source label/period/calendar -> proposed date and rationale, not verified publication date. |
| [corpus/maintenance/estimate_ir_publication_dates.py](../../../finsight/agents/narrative/corpus/maintenance/estimate_ir_publication_dates.py) | CLI; maintenance | Writes provisional IR date estimates and provenance; enables successfully estimated rows under current project policy. | `--ir-manifest`, `--core-manifest` -> edits IR manifest in place, including eligibility. Not a routine inference step. |
| [corpus/maintenance/update_intc_ir_metadata_from_workbook.py](../../../finsight/agents/narrative/corpus/maintenance/update_intc_ir_metadata_from_workbook.py) | CLI; Intel maintenance | Applies period/date annotations from the local Intel workbook by exact file stem. Reviewed populated dates verify records; blank dates remain ineligible. | Workbook/manifest options -> updated manifest and optionally workbook. Company-specific history/tool. |
| [corpus/maintenance/generate_catalog.py](../../../finsight/agents/narrative/corpus/maintenance/generate_catalog.py) | CLI; reporting | Builds a human-readable document catalog with source counts/status, paths and local inventory information. Does not parse/chunk/index source content. | Positional manifest/output, optional supplemental/IR manifests -> Markdown catalog. |

## 8. Earlier workflows and demo compatibility: 4 files

| File | Type | Purpose | Input and output |
|---|---|---|---|
| [corpus/maintenance/pilot.py](../../../finsight/agents/narrative/corpus/maintenance/pilot.py) | CLI; diagnostic | Original NVDA vertical slice; `--all-documents` expands its NVDA scope. Current multi-company builds use `corpus/processing/process_corpus.py`. | NVDA sources/query -> pilot or full-corpus outputs and query result. |
| [corpus/maintenance/build_markdown_index.py](../../../finsight/agents/narrative/corpus/maintenance/build_markdown_index.py) | CLI; alternate build | Combines enhanced Markdown and an optional base corpus, inheriting canonical source metadata; builds isolated BM25 plus **Qdrant**. Not the standard all-company promotion route. | Markdown/ticker/document/output/base-index -> `markdown_blocks.jsonl`, `markdown_chunks.jsonl`, combined indexes and `build_report.json`. |
| [demo/retriever_stub.py](../../../finsight/agents/narrative/demo/retriever_stub.py) | Library; demo | `StubRetriever`: word-overlap search over the demo corpus. Keep for mock tests/default registry compatibility. | Demo query -> example evidence, not real-company retrieval. |
| [demo/stub.py](../../../finsight/agents/narrative/demo/stub.py) | Library; demo | `StubNarrativeAgent`: example supporting/challenging evidence by demo subclaim topic. Keep beside the real agent. | Demo subclaim -> example evidence; no real source search. |

## 9. Which modules work together?

```text
Sources/manifests -> corpus/processing/corpus.py
  -> corpus/processing/process_corpus.py -> corpus/processing/ingest.py -> corpus/processing/chunk.py -> corpus/processing/index.py

PDF enrichment when needed -> corpus/pdf/pdf_chart_processing.py
  -> corpus/pdf/docling_charts.py + corpus/pdf/chart_models.py + corpus/pdf/chart_vision.py + corpus/pdf/chart_usage.py
  -> corpus/pdf/pdf_enrichment.py -> promoted company blocks/chunks/BM25

Final company passages -> retrieval/build_vector_databases.py -> retrieval/qdrant_index.py
  -> retrieval/retriever.py -> rag/collect_evidence.py -> rag/basic_rag.py

Parallel checks/research:
  evaluation/run_retrieval_comparison.py -> evaluation/summarize_retrieval_comparison.py
  evaluation/validate_retrieval_provenance.py         retrieval provenance checks
```

Current C1-C4 builders and 126-claim retrieval/RAG/RAGAS evaluation runners live under `scripts/narrative/`. They reuse this package's processing, retrieval and evidence-summary implementations. See [the scripts inventory](narrative_scripts.md) and [Workflow D](../workflow/workflow_d_evaluation.md). The historical P1/P2 runner has been retired; keeping `retrieval/source_priority.py` does not activate priority in the current pipeline.

## 10. Current entry-point examples

These illustrate the main commands. Processing/index commands write their documented artifacts. For prerequisites, extraction/promotion commands and the complete workflow, follow the [Full Narrative workflow summary](../workflow/summary_full_narrative_agent_flow.md).

```powershell
python -u -m finsight.agents.narrative.corpus.processing.process_corpus --ticker META
python -m finsight.agents.narrative.corpus.pdf.pdf_chart_processing --all --status
python -u -m finsight.agents.narrative.retrieval.build_vector_databases --ticker META
python -m finsight.agents.narrative.retrieval.retriever --ticker META --query "DAP June 2024" --cutoff 2025-07-31 --backend bm25 --top-k 3
python -u -m finsight.agents.narrative.rag.basic_rag --ticker META --claim "DAP was 3.27 billion on average for June 2024" --cutoff 2025-07-31 --backend bm25 --top-k 3 --max-tokens 1500 --ollama-context 20000 --ollama-model qwen3-vl:4b-instruct
```

Modules with argument parsers support `python -m finsight.agents.narrative.<group>.<name> --help`. **Do not use this pattern for `prepare_manual_ir_metadata`: it has no argument parser, so invoking it runs its work.** Library-only modules are used through their callers.

## 11. Maintenance and verification notes

Keep package initializers and imported helpers even when they are not separate workflow commands. The script inventory and Workflows A-E describe the current paths; corpus/index builds and model requests are separate execution steps.
