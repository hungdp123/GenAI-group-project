# Basic RAG cited-passage baseline

Week 2 task 5 supplies the real `CorpusRetriever` with explicitly selectable `bm25`,
`dense`, and `hybrid` backends. It returns original cited passages as neutral `EvidenceItem`
objects. It does not generate answers, compute new financial values, or infer support or
challenge from similarity. The shared `Retriever.retrieve(query, top_k, cutoff_date)`
signature is unchanged.

## Select a backend

| Backend | Search | Requirements |
|---|---|---|
| `bm25` (default) | Lexical search of the canonical JSON index | Base project dependencies only |
| `dense` | Cosine search in the corresponding embedded Qdrant database | Retrieval extra, built Qdrant database, its embedding model |
| `hybrid` | BM25 and Qdrant results combined by the existing fusion function | Same requirements as dense |
| `auto` (optional) | Hybrid when Qdrant metadata exists, otherwise BM25 | Availability determines the selected method; inspect `backend_used` |

Explicit dense/hybrid requests raise an actionable error if vector metadata is missing,
stale, or incompatible. They never silently switch to BM25. Auto also rejects stale
vector metadata when a database is present. BM25 ignores vector metadata completely.

Hybrid uses the production Qdrant fusion function: reciprocal-rank fusion with constant
60 and retention of up to two leading BM25 candidates. This differs from the pure RRF
experiment helper; record the implementation and settings when comparing results.
The candidate depth defaults to 50 and expands to at least the requested output K.

## Retrieve passages

From the repository root:

```powershell
python -m finsight.agents.narrative.retrieval.retriever --ticker INTC --backend bm25 --query "Intel revenue Q1 2025" --cutoff 2025-07-31 --top-k 5
python -m finsight.agents.narrative.retrieval.retriever --ticker INTC --backend dense --query "Intel revenue Q1 2025" --cutoff 2025-07-31 --top-k 5
python -m finsight.agents.narrative.retrieval.retriever --ticker INTC --backend hybrid --query "Intel revenue Q1 2025" --cutoff 2025-07-31 --top-k 5 --candidate-k 50 --output results/intc_passages.json
```

The CLI prints JSON with the query, ticker, cutoff, requested/actual backend, corpus
fingerprint, parameters, exact passages, and citation gaps. Empty retrieval returns
`status=NO_EVIDENCE` and `evidence=[]`; configuration/index failures exit with code 2.
`--index`, `--qdrant-dir`, and `--corpus-root` support explicit local locations.

Build the vector database first if required:

```powershell
python -m finsight.agents.narrative.retrieval.build_vector_databases --ticker INTC
```

The retrieval extra is installed with `pip install -e ".[retrieval]"`. Dense and hybrid
load the embedding model named in the database metadata. An initial model load may
download its files; cached models can run with `HF_HUB_OFFLINE=1`.

## Python and application configuration

```python
from datetime import date
from finsight.agents.narrative.retrieval.retriever import CorpusRetriever

retriever = CorpusRetriever(company_ticker="INTC", backend="hybrid", candidate_k=50)
passages = retriever.retrieve("Intel revenue Q1 2025", 5, date(2025, 7, 31))
for passage in passages:
    print(passage.evidence_id, passage.content, passage.url, passage.citation_span)
```

The constructor binds one ticker. The calling application supplies the company it has
identified; the adapter does not infer or switch companies from words in the query.
An AMD request defaults to AMD's index, rather than the former hard-coded NVDA path.

| Environment variable | Default | Meaning |
|---|---|---|
| `FINSIGHT_NARRATIVE_BACKEND` | `bm25` | `bm25`, `dense`, `hybrid`, or `auto` |
| `FINSIGHT_NARRATIVE_TICKER` | `NVDA` | Company for this retriever instance |
| `FINSIGHT_NARRATIVE_CORPUS_ROOT` | `data/narrative/processed` | Parent of company directories |
| `FINSIGHT_NARRATIVE_INDEX` | `<root>/<TICKER>/indexes/index.json` | Override the JSON index |
| `FINSIGHT_NARRATIVE_QDRANT` | `<index directory>/qdrant` | Override the Qdrant directory |
| `FINSIGHT_NARRATIVE_CANDIDATE_K` | `50` | Hybrid candidate depth |

Constructor arguments take precedence over their corresponding environment variables.
An explicit index override takes precedence over deriving a path from the corpus root.

For an application trial, select the implementation through the existing registry override:

```powershell
$env:FINSIGHT_IMPL_RETRIEVER="finsight.agents.narrative.retrieval.retriever:CorpusRetriever"
$env:FINSIGHT_NARRATIVE_TICKER="INTC"
$env:FINSIGHT_NARRATIVE_BACKEND="bm25"
```

The repository's shared application defaults still select the demo stubs, as required by
the integration plan. The existing `run_basic_rag(..., retriever=retriever)` benchmark
wrapper can also consume this adapter. That wrapper invokes an LLM; its generated text
is outside this passage baseline's exact-source guarantee and needs its own citation review.

## Empty results, provenance, and chart coverage

- Blank queries, nonpositive K, empty indexes, absent canonical company indexes, and requests
  with no company/date-eligible chunks return an empty list before loading models.
- A missing explicitly configured index raises `FileNotFoundError`; it is a configuration
  problem rather than evidence that a claim is false.
- BM25 returns no results when no query terms match. Dense search returns semantic nearest
  neighbors when eligible chunks exist; a nonempty ranking is not a correctness judgment.
- Each returned item preserves its chunk ID, original text, company name, publication-date
  metadata, document type, source tier, period, URL, and citation span. Stance stays neutral.
- Missing URLs remain missing and are listed in the CLI's `citation_gaps`; no source URL is
  guessed. Source IDs and citation text remain available for local review.
- Already promoted chart and table chunks use the same indexes and retrieval path. Their
  exact extracted text and page/chart citation spans survive conversion to `EvidenceItem`.
  The shared evidence contract has no separate chart-ID field; its stable chunk ID and
  citation span carry that reference.
- Task 3 has now promoted real PDF chart/table content into the canonical indexes. The
  original Step 5 checks below predate that promotion and used fixtures for chart/table
  adapter coverage. On 16 October, 24 live retrieval checks were completed on the enriched
  corpus. Chart-specific reviewed experiment labels remain later work.

Vector metadata must match the BM25 corpus fingerprint and ordered chunk IDs. Injected
embedding backends must match the stored model name. The adapter closes each Qdrant client
after retrieval, including failed queries, and caches the model and loaded JSON snapshot
for reuse. Create a new instance after rebuilding the corpus.

## Validation

```powershell
python -m pytest tests/test_corpus_retriever.py tests/test_narrative_pilot.py tests/test_baselines.py -q
python -m ruff check finsight/agents/narrative/retrieval/retriever.py tests/test_corpus_retriever.py
```

Coverage includes NVDA/AMD/INTC isolation for all three backends, cutoff filtering, exact
citations, chart/table and estimated-date metadata, empty/out-of-scope requests, missing
indexes, stale fingerprints, embedding-model mismatch, storage-lock release after failure,
configuration precedence, JSON output, and injection into the existing Basic RAG wrapper.

Before Task 3 promotion on 15 October 2026: **47 focused tests passed**, and lint passed for the adapter
and its new tests. Nine additional real-index checks (AMD/INTC/NVDA, each with BM25/dense/
hybrid) returned passages with matching company, cutoff, and exact indexed text using the
cached real embedding model. The original smoke-run record is archived at
`data/narrative/experiments/basic_rag/archives/pre_pdf_20261016_003732/task5_smoke.json`.
Those historical checks establish functioning
backends and provenance; relevance-quality comparison belongs to the experiment evaluation.

## Post-PDF refresh: 16 October 2026

The refreshed run uses the frozen v1 support/counter queries across AMD, INTC, NVDA, and
TSLA: 24 actual retrieval checks across three backends, K=10 and candidate K=50. All source
text, company/date filters, citation mapping, duplicate-ID, and source-file checks passed.
Eleven missing-URL occurrences remain recorded. The report includes actual chart retrieval.
Relevance/financial accuracy and a reviewed chart-specific query set remain later work.

```powershell
python -m finsight.agents.narrative.evaluation.validate_retrieval_provenance --queries docs/narrative/evaluation/retrieval_queries_v1.jsonl --top-k 10 --candidate-k 50
```

Archive existing outputs before repeating. The summary is
`data/narrative/experiments/basic_rag/task5_smoke.json`; each check lists its full-evidence
`output_path`. See the [post-PDF refresh report](../results/2026-10-16_retrieval_refresh.md)
for before/after comparison, archive locations, and limitations. The historical nine-check
report lacked query strings; this refresh records explicit frozen queries. Unit/regression
test suites were not rerun in this refresh.
