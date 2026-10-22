# All-company chunking and retrieval comparison results

Saved experiment: `20261018_all16_v1`, 18 October 2026. This summary exports the recorded local reports for GitHub review. Creating it did not rerun experiments or independently revalidate artifact hashes.

## 1. Chunking technical results

The four arms use the same frozen extracted sources across 16 companies. All cover 12,093,992/12,093,992 source words and have zero recorded provenance errors or unresolved reference mappings.

| Arm | Strategy | Chunks | Chunks over 256 embedding tokens |
|---|---|---:|---:|
| C1 | Fixed 800 words / 120 overlap | 92,975 | 16,270 |
| C2 | Fixed 300 words / 50 overlap | 109,033 | 26,985 |
| C3 | Fixed 400 words / 60 overlap | 101,537 | 22,481 |
| C4 | Adaptive, candidate token targets 128/192/256 | 139,956 | 0 |

C4 considers complete segmentation plans, rejects hard-rule failures and scores valid plans using structural integrity, context completeness, lexical semantic coherence and length efficiency. Initial configurable weights are 0.25 each. The context/coherence features are proxies; passing technical checks does not establish semantic quality or correct original PDF extraction.

For each source block, C4 considers nine plans: maximum token targets 128/192/256 x structure/sentence/token boundary policies. Actual chunks can be shorter. Each plan must preserve the complete source body, literal source traceability and a final embedding-input limit of 256 tokens, including context and special tokens. Context copied from the same block is budgeted at most 64 tokens or half the candidate target, whichever is smaller. Blocks are not combined across source boundaries.

Only hard-valid plans receive the four soft scores. Selection takes the highest weighted score; ties prefer fewer chunks, larger target and candidate ID. If none passes, the block is explicitly reported/quarantined. Targets and weights are initial settings, not validated optima. Full feature definitions and configuration links are in [summary_week_2.md, Section 8.1](../workflow/summary_week_2.md#81-c1-c4-chunking-strategies).

## 2. Chunking retrieval results

Scope: 92 development claims x four arms = 368 cases; 16 companies, identical BM25/P0 backend and unchanged claim queries, top K=5/10, cutoff 2025-07-31. The cutoff admits 679 of the 1,201 frozen documents. The 36 provisional heldout claims were not evaluated.

Claim origins: 16 reused project-curated claims from `docs/narrative/evaluation/retrieval_queries_v2.jsonl`, 55 deterministic source-fact drafts and 21 synthetic numerical counterexamples created by `scripts/prepare_narrative_labels.py`. Evidence derives from frozen filings, earnings releases, presentations and call transcripts. The [92-row claim source inventory](2026-10-18_development_claim_sources.csv) records each claim's origin, source document IDs/types, available URLs, evidence locations and any numerical transformation. All rows remain pending review in this dataset; no heldout claim text is exported.

| Arm | Seed evidence-bundle Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| C1 | 79.89% | 85.33% | 0.7112 |
| C2 | 82.07% | 84.78% | 0.7514 |
| C3 | 83.15% | 87.50% | 0.7334 |
| C4 | 72.83% | 77.17% | 0.6958 |

C3 has the highest provisional Recall@10; C2 has the highest MRR. C3's difference from C1 is +2.17 percentage points with an exploratory company-cluster interval of -4.26 to +7.53 points. C4 is token-valid but has lower recall. No strategy was selected or promoted. Fixed C1-C3 can be searched by BM25 but cannot be embedded without overflow under the current model/representation.

Recall measures retrieved source fact bundles, including required context spans; MRR measures the first result containing a complete fact bundle. Labels are unreviewed and sparse. Sixteen claims come from the original approved set and 76 are new source-derived drafts, including synthetic numerical counterexamples. Shared wording can inflate scores. Unjudged passages are unknown, not automatically irrelevant; semantic context completeness and citation entailment remain unmeasured.

## 3. Retrieval and document-priority results

Scope: the same 92 development claims, C4 as a temporary token-valid representation, four retrieval methods x three policies = 1,104 cases. All 16 checkpoints completed, with 62,553 cutoff-filtered vectors and 384 separate latency probes. C4 was an engineering input, not the accepted chunking winner.

- R1: BM25; R2: dense Qdrant retrieval.
- R3: hybrid RRF with two lexical anchors; R4: R3 plus cross-encoder reranking.
- P0: no source priority; P1: High-first with counter probe and Medium expansion; P2: bounded priority promotions near relevance ties.

| Arm | Recall@5 | Recall@10 | Direct-counter Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|
| R1/P0 | 72.83% | 77.17% | 100.00% | 0.6958 |
| R1/P1 | 72.83% | 77.17% | 100.00% | 0.6958 |
| R1/P2 | 72.83% | 77.17% | 100.00% | 0.6956 |
| R2/P0 | 28.26% | 35.87% | 38.10% | 0.1921 |
| R2/P1 | 28.26% | 35.87% | 38.10% | 0.1921 |
| R2/P2 | 28.26% | 35.87% | 38.10% | 0.1934 |
| R3/P0 | 52.72% | 62.50% | 76.19% | 0.3727 |
| R3/P1 | 52.72% | 62.50% | 76.19% | 0.3727 |
| R3/P2 | 52.72% | 62.50% | 76.19% | 0.3771 |
| R4/P0 | 57.61% | 63.59% | 76.19% | 0.5317 |
| R4/P1 | 57.61% | 63.59% | 76.19% | 0.5317 |
| R4/P2 | 57.61% | 63.59% | 76.19% | 0.5314 |

BM25 leads these provisional metrics. P1 produced identical top-10 rankings to P0 in 368/368 cases because it always expanded when semantic sufficiency was unknown; it demonstrated no selective stopping savings. P2 changed ordering in 132/368 cases with small MRR differences and unchanged Recall@5/10 at reported precision. No retrieval or priority winner was accepted.

No paid API calls were made; local compute cost in dollars was not measured. P0 cold-feature/warm-feature median query times in milliseconds were R1 845.9/2.0, R2 213.3/1.9, R3 1123.7/2.2 and R4 4003.6/2936.9. Cold-feature means cleared query caches with models already loaded, not a fresh machine/process. These probes cover one selected query per company; they are not general deployment latency estimates. Precision@5 and nDCG were unmeasured in this source-bundle diagnostic.

## 4. Separate canonical E1-E4 evaluation

The earlier canonical run uses 16 approved v2 claims and 32 directional support/counter queries, top K=10/candidate K=50 and the same date cutoff, but different chunking, query formulation and reference scoring. Its hybrid is pure RRF, unlike the anchored R3 above. Do not compare percentages directly across these experiments.

| Canonical method | Support Recall@10 | Counter Recall@10 |
|---|---:|---:|
| E1 BM25 | 93.75% | 87.50% |
| E2 Dense | 37.50% | 31.25% |
| E3 Hybrid RRF | 68.75% | 43.75% |
| E4 Hybrid + cross-encoder | 93.75% | 75.00% |

See the [canonical evaluation report](2026-10-18_task4_retrieval_comparison.md). Retrieval scores do not measure final generated-verdict accuracy; the three Basic RAG smoke outcomes are documented in the [consistency report](2026-10-18_basic_narrative_rag_consistency.md).

## 5. Original local evidence and next decision

Paths below are relative to the repository root and require the separately shared `data/` snapshot:

```text
data/narrative/experiments/narrative_rag/20261018_all16_v1/validation/c4_v1/step_03_validation_mapping.md
data/narrative/experiments/narrative_rag/20261018_all16_v1/reports/step_04_chunking_comparison.md
data/narrative/experiments/narrative_rag/20261018_all16_v1/comparisons/bm25_dev_v1/comparison.json
data/narrative/experiments/narrative_rag/20261018_all16_v1/comparisons/bm25_dev_v1/per_query.csv
data/narrative/experiments/narrative_rag/20261018_all16_v1/reports/step_05_retrieval_priority.md
data/narrative/experiments/narrative_rag/20261018_all16_v1/retrieval_priority/c4_dev_v1/comparison.json
data/narrative/experiments/narrative_rag/20261018_all16_v1/retrieval_priority/c4_dev_v1/per_query.csv
```

Before final selection, review/freeze labels and required context, judge pooled candidates, resolve token representation and source-structure issues, then evaluate under the predeclared criteria. Promote only accepted artifacts and refresh dependent indexes/checks. Execution commands and scripts to commit are in [summary_week_2.md](../workflow/summary_week_2.md).
