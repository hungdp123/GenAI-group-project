# Task 4: Full 16-company E1-E4 retrieval comparison

**Status:** Complete and scored  
**Saved evaluation timestamp:** 18 October 2026, 12:08 Singapore time  
**Scope:** approved v2 development queries across all 16 companies

## Run scope

- 16 claims, one per company; 32 directional support/counter queries.
- Four retrieval methods: BM25, dense, hybrid reciprocal-rank fusion (RRF), and
  hybrid with cross-encoder reranking.
- Top K = 10; candidate K = 50; query cutoff = 2025-07-31.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- 64 query/method result rows and 128 directional rankings.
- Evaluation includes Recall, Precision, MRR and nDCG at K = 1, 3, 5 and 10.

## Latest Recall@10

| Method | Support Recall@10 | Counter Recall@10 |
|---|---:|---:|
| E1 BM25 | 93.75% | 87.50% |
| E2 dense | 37.50% | 31.25% |
| E3 hybrid RRF | 68.75% | 43.75% |
| E4 hybrid + cross-encoder | 93.75% | 75.00% |

These are the current full-corpus scores for the saved run. The older table in the
all-company historical report used an earlier corpus and should not be cited as the
latest result. The five-company subset is a separate scope and is not directly
comparable to this 16-company aggregate.

## Saved files

- [Rankings](../../../data/narrative/experiments/retrieval_comparison/ready_16_v2/results.json)
- [Scored evaluation and audit](../../../data/narrative/experiments/retrieval_comparison/ready_16_v2/evaluation.json)
- [Per-query metrics](../../../data/narrative/experiments/retrieval_comparison/ready_16_v2/per_query.csv)
- [Approved v2 queries and references](../evaluation/retrieval_queries_v2.jsonl)

`evaluation.json` was generated after the rankings. Its results hash matches
`results.json`, its query hash matches the approved query set, and its 16 recorded
company corpus fingerprints match the current company indexes at the saved readiness
snapshot. The run used the approved cutoff and retrieval settings.

## Saved ranking audit

Each method returned 320 passage occurrences (16 claims x two directions x 10
results). The saved audit reports:

| Method | Returned occurrences | Missing `source_url` | Cutoff leaks | Wrong company | Duplicates | Index mismatches | Missing source files |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 320 | 31 | 0 | 0 | 0 | 0 | 0 |
| Dense | 320 | 50 | 0 | 0 | 0 | 0 | 0 |
| Hybrid RRF | 320 | 59 | 0 | 0 | 0 | 0 | 0 |
| Hybrid + cross-encoder | 320 | 39 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **1,280** | **179** | **0** | **0** | **0** | **0** | **0** |

The 179 missing URL entries are returned-passage occurrences across methods, not 179
distinct documents. In the recorded audit, `source_url` was the only missing
citation field.

## Interpretation limits

The 16 claims are a small development set with sparse curated reference passages.
Unlabeled passages were not judged irrelevant, and the references are not exhaustive.
These retrieval metrics measure ranking against those references; they do not
establish financial accuracy, semantic entailment, or a final production method
selection. The experiment's pure RRF method also differs from production hybrid
retrieval, which retains lexical anchors.
