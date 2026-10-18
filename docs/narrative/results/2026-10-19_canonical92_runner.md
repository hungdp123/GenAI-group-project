# Canonical chunking comparison on 92 development claims

Date: 19 October 2026.

## Implementation

Runner: `scripts/narrative/retrieval/compare_canonical_development.py`.

The runner applies canonical 800-word / 120-word overlap chunking rules to the same frozen parsed sources used by the C4 experiment. It reuses the 92 development questions, source-span evidence labels, cutoff, P0 retrieval implementations and evidence-bundle scoring. It preserves table/chart handling, literal source mappings and duplicate-source aliases. This compares representations under shared conditions; it does not reproduce the historical canonical 16-claim benchmark's parser outputs or directional queries.

Company indexes, vector storage, checkpoints and reports are isolated. Canonical production artifacts and prior C4 results are preserved. Checkpoints validate configuration/code identities and artifact hashes before reuse.

## Run all 92 claims

From the project root:

```powershell
python -u -m scripts.narrative.retrieval.compare_canonical_development --run-id 20261018_all16_v1 --experiment-id canonical92_v1 --allow-draft --overflow-policy truncate --resume
```

Scope: 16 companies, 92 development claims, R1-R4 with P0, **368 cases**. The runner uses pinned local models and makes no paid API requests. Existing frozen inputs, C4 comparison artifacts and cached model snapshots are prerequisites. Heldout claims are untouched.

The explicit `truncate` policy permits native embedding/reranker truncation and records overflow counts and token lengths. Oversized queries are rejected. Stored passages retain their full source text, so evidence can occur outside the encoded prefix; this limitation must accompany results. The default policy rejects overflow. No final strategy is selected or promoted.

## Output

Base directory: `data/narrative/experiments/narrative_rag/20261018_all16_v1/retrieval_canonical/canonical92_v1/`.

- `configuration.json`: reproducible run settings and input/code identities.
- `comparison.json`: aggregate results and C4 reference metrics for the same claims/methods.
- `per_query.jsonl` and `per_query.csv`: claim/method metrics.
- `report.md`: readable result summary.
- `companies/`: isolated indexes, mappings, aliases, rankings, overflow reports and resumable checkpoints.

## Completed checks and current status

Six focused regression cases passed for narrative windows/overlap, source coverage, repeated table headers, atomic charts, distinct block IDs and explicit embedding/reranker overflow handling. The CLI help command succeeded.

A real BM25-only smoke run completed on all six META development claims using experiment ID `canonical92_smoke_20261019_r1_v1`:

| Metric | Value |
|---|---:|
| Claims / method cases | 6 / 6 |
| Recall@5 | 91.67% |
| Recall@10 | 91.67% |
| MRR@10 | 0.6667 |

Smoke outputs are under the same base parent with that smoke experiment ID. No embedding, reranker or generation calls were made during this smoke run. The subsequent full 92-claim/four-method run is complete; results follow. Draft labels make these development diagnostics, not final benchmark acceptance.

## Completed full development run

Saved status: `development_diagnostic_complete`; 92 claims across all 16 companies, 368 method cases, P0, zero heldout claims evaluated. Figures below were read from saved artifacts, without rerunning retrieval or models.

| Method | Recall@5 | Recall@10 | Support Recall@10 | Direct-counter Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| R1 BM25 | 79.89% | 85.33% | 88.73% | 85.71% | 0.7121 |
| R2 Dense | 27.17% | 36.41% | 38.03% | 42.86% | 0.1968 |
| R3 Hybrid with lexical anchors | 46.20% | 59.78% | 63.38% | 66.67% | 0.3926 |
| R4 Hybrid + cross-encoder | 48.37% | 64.67% | 71.83% | 61.90% | 0.4356 |

| Method | Canonical Recall@10 | C4 Recall@10 | Canonical MRR@10 | C4 MRR@10 |
|---|---:|---:|---:|---:|
| R1 | 85.33% | 77.17% | 0.7121 | 0.6958 |
| R2 | 36.41% | 35.87% | 0.1968 | 0.1921 |
| R3 | 59.78% | 62.50% | 0.3926 | 0.3727 |
| R4 | 64.67% | 63.59% | 0.4356 | 0.5317 |

BM25 leads overall recall in this run. C4 retains higher BM25 direct-counter Recall@10 (100% versus 85.71%), and higher R4 MRR@10 (0.5317 versus 0.4356); the overall recall comparison alone does not establish a universally better representation.

Saved files contain 368 per-query metric rows and 16 company indexing reports, covering 40,046 indexed passages. The reports record 8,621 over-limit embedding passage inputs and 1,679 over-limit reranker pairs under explicit truncation. Zero paid API requests were recorded. Timing values include cache effects and are not independent cold deployment measurements. Support/direct-counter metrics use 71/21 claims respectively. No final strategy is selected or promoted.

Separate provenance and Narrative Evidence Agent integration refreshes on these 92 claims remain pending. Existing 96-check/48-case reports belong to the earlier dataset.
