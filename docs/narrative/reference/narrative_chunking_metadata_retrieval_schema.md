# Narrative Agent: Chunking, Metadata, and Retrieval Schema

**Status: implementation note / draft for review.** This document describes the current code in `finsight/agents/narrative/`. It records what the pilot currently does; it is not evidence that the team has formally approved these choices as a project standard.

## 1. Processing flow

The current vertical slice follows this path:

```text
canonical corpus manifests
  -> load and validate source documents
  -> extract HTML sections or PDF pages into blocks
  -> split each block into overlapping word windows
  -> build a deterministic BM25 index
  -> retrieve evidence and map it to the shared EvidenceItem contract
```

The canonical loader currently reads the core SEC, supplemental SEC, and official IR manifests. It filters to the requested ticker and excludes documents explicitly marked `retrieval_eligible: false`; it validates each local file against the manifest SHA-256 before ingestion.

HTML ingestion strips scripts/styles, hidden content, and common navigation/header/footer elements. It keeps text in document order and serializes tables as labeled rows (for example, `Revenue — 2025: $100; 2024: $80`) rather than an unlabeled sequence of cells. SEC contents tables with many linked Item references are removed so their entries are not mistaken for filing sections. This is structural/heuristic cleanup; unusual site-specific page furniture may still need a targeted rule.

## 2. Chunking

Chunking is section/page-aware and deterministic (`processing/chunk.py`):

- Target size: **800 whitespace-separated words**.
- Overlap: **120 words**, so the next window starts 680 words after the previous start.
- Each extracted block is chunked independently. Windows do not cross block boundaries.
- HTML extraction makes blocks from SEC-style `Item N` sections; documents without matching headings are one `Document text` block. PDF extraction creates one block per page.
- Short blocks produce a single chunk. Empty blocks produce none.
- Chunk IDs combine the document ID, a normalized section slug, a per-document/section ordinal, and the first 12 hex characters of the chunk text SHA-256.

The index builder removes duplicate chunks by exact text SHA-256, retaining the first occurrence. This is exact-text deduplication, not semantic similarity deduplication.

## 3. Metadata schema

Each indexed chunk contains these fields:

| Field | Meaning |
| --- | --- |
| `chunk_id` | Stable ID derived from document, section, ordinal, and text hash. |
| `document_id` | Corpus manifest identifier for the source document. |
| `company_ticker`, `company_name` | Company identity. |
| `document_type` | Type from the corpus manifest, e.g. `10-K`, `earnings_presentation`, or `earnings_call_transcript`. |
| `publication_date` | Effective ISO date used for as-of filtering: the verified date when present, otherwise an approved estimate. |
| `publication_date_is_estimated` | `true` when the effective publication date came from `estimated_publication_date`; otherwise `false`. |
| `publication_date_precision` | Precision of the effective date, such as `day` or `estimated_day`. |
| `publication_date_basis` | Audit note describing either the verified-date source or the estimation method. |
| `estimated_publication_date` | Manifest-level heuristic estimate for a date-unknown IR file. During loading, an approved estimate becomes the effective `publication_date`. |
| `estimated_publication_date_basis` | Explains the estimate method and the SEC fiscal-period end it was derived from; null when no unambiguous quarter match exists. |
| `period` | Reporting period when supplied by the manifest; otherwise null. |
| `source_tier` | Normalized `TIER_1`, `TIER_2`, or `TIER_3`. |
| `source_url` | Official/source URL from the manifest. |
| `local_path` | Local source-file path used during processing. |
| `section` | HTML section heading, PDF page title, or fallback block label. |
| `page` | PDF page number; null for HTML. |
| `citation_span` | Current coarse locator: HTML heading/front matter/document text or PDF page number. |
| `text` | Extracted chunk text. |
| `text_sha256` | SHA-256 of the normalized chunk text. |

The loader's document-level metadata also includes `sha256` for validating the source file and `retrieval_eligible`. Those are used before chunking and are not copied as chunk fields. The index records schema version, corpus fingerprint, chunking parameters, chunk count, and duplicate count alongside the chunks.

For date-unknown IR records, the manifest keeps `publication_date` null and may store an `estimated_publication_date`. The estimate uses the final day of the calendar month after the matching SEC-reported fiscal-period end, with the SEC 10-Q/10-K period used to account for each company's fiscal calendar (including 52/53-week years). When such an estimate is explicitly approved for retrieval, the loader uses it as the effective `publication_date` and sets `publication_date_is_estimated: true`, `publication_date_precision: estimated_day`, and the estimation basis. Records with neither a verified date nor an approved estimate remain excluded. Consumers must preserve and disclose this provenance because an estimated date is not a verified public-release date.

## 4. Retrieval behavior

The current index is lexical BM25, not vector/embedding search (`processing/index.py`):

1. Tokenize query and chunk text into lowercase alphanumeric terms (internal apostrophes are retained).
2. Filter chunks to the configured company ticker and `publication_date <= cutoff_date`.
3. Score matching chunks with BM25 (`k1 = 1.5`, `b = 0.75`).
4. Sort by descending score, then chunk ID for deterministic ties.
5. Prefer different `(document_id, citation_span)` locations before returning a second result from the same location; then fill remaining slots if needed.

`top_k < 1`, an empty query after tokenization, or no matching terms returns an empty list. The retriever maps each result into the shared `EvidenceItem` contract, including the chunk text, source tier, document type, company, period, publication date, source URL, and citation span.

## 5. Current limitations to review

- Retrieval has no semantic/vector search, reranker, or query expansion.
- HTML citation locations are section-level rather than exact line, paragraph, or character offsets. PDF citations identify a page.
- Extracted PDF text can be incomplete or absent for scanned/image-only pages; ingestion emits a warning for pages with no extractable text.
- Deduplication is based only on identical chunk text. Identical text from different documents can therefore collapse to the first chunk and its metadata in the index.
- BM25 searches chunk text only; metadata is used for filtering and citations, not relevance boosting.
- The retrieval implementation currently emits neutral stance and the placeholder sub-claim ID `claim`; claim-specific stance assignment is not part of this schema yet.
- The fixed 800/120 chunk sizes and ranking choices are pilot defaults and should be evaluated against representative narrative questions before being treated as final.

## 6. Relevant implementation

- `finsight/agents/narrative/corpus/processing/corpus.py` — manifest loading, normalization, validation, and corpus fingerprint.
- `finsight/agents/narrative/corpus/processing/ingest.py` — HTML/PDF extraction and citation spans.
- `finsight/agents/narrative/corpus/processing/chunk.py` — chunk fields, IDs, and word-window chunking.
- `finsight/agents/narrative/corpus/processing/index.py` — index format, exact-text deduplication, BM25 scoring, and filtering.
- `finsight/agents/narrative/retrieval/retriever.py` — conversion to the shared evidence contract.
