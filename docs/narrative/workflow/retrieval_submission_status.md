# Narrative retrieval, RAG and RAGAS: submission status

[Back to Workflow D](workflow_d_evaluation.md)

Snapshot checked: **29 October 2026, 10:48 SGT (UTC+08:00)**.

## Retrieval-only completion (108 claims)

- Dataset: `data/narrative/evaluation/test_claims_v2/cases_combined.csv`.
- 108 claims per configuration; four chunking strategies (C1–C4) and four retrieval backends (bm25, dense, hybrid, r4).
- **16/16 configurations completed**: 1,728/1,728 unique case runs, all with saved status `OK`.
- The selected configurations contain 1,728 `result.json` files and 1,728 accompanying `retrieval.json` files.
- **16/16 configurations scored**: all selected configurations have `evaluation.md` and `evaluation.json` covering 108 cases each. The C3 and both C4 split folders also have their matrix comparison reports and CSV exports.
- These are retrieval-only experiments. They do not establish generation quality, stance classification performance, or RAGAS judge performance.

## Retrieval results across all 16 configurations

Scores below are calculated from the existing `evaluation.json` files in the authoritative configuration folders listed below. For each metric, the displayed value is the arithmetic mean of the support-search and counter-search aggregates: `(support + counter) / 2`. This is not a merged or deduplicated top-k ranking, and it is not the equal-company average.

Each direction's positive-reference metrics cover **71 annotated cases**. The remaining **37 unsupported cases** have no positive gold reference and are excluded from these metrics, not counted as retrieval misses. All 108 cases per configuration were retrieved successfully.

| Chunk | Backend | Source Hit@5 | Source Hit@10 | Known-reference Recall@5 | Known-reference Recall@10 | MRR@5 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | bm25 | 65.49% | 76.76% | 36.62% | 45.07% | 0.1899 | 0.1995 |
| C1 | dense | 36.62% | 45.77% | 9.15% | 13.38% | 0.0440 | 0.0495 |
| C1 | hybrid | 40.85% | 64.08% | 18.31% | 34.51% | 0.1173 | 0.1351 |
| C1 | r4 | 55.63% | 69.01% | 22.54% | 26.06% | 0.1282 | 0.1332 |
| C2 | bm25 | 64.08% | 80.99% | 33.80% | 41.55% | 0.1731 | 0.1834 |
| C2 | dense | 42.25% | 57.75% | 7.75% | 14.08% | 0.0572 | 0.0655 |
| C2 | hybrid | 52.82% | 71.13% | 21.83% | 30.99% | 0.1404 | 0.1507 |
| C2 | r4 | 52.82% | 71.83% | 16.20% | 24.65% | 0.1005 | 0.1119 |
| C3 | bm25 | 64.08% | 79.58% | 32.39% | 42.25% | 0.1792 | 0.1923 |
| C3 | dense | 40.85% | 52.11% | 11.97% | 16.90% | 0.0776 | 0.0839 |
| C3 | hybrid | 50.00% | 69.72% | 26.06% | 36.62% | 0.1887 | 0.2023 |
| C3 | r4 | 58.45% | 71.83% | 23.24% | 28.87% | 0.1353 | 0.1434 |
| C4 | bm25 | 63.38% | 78.87% | 32.39% | 37.32% | 0.1878 | 0.1939 |
| C4 | dense | 44.37% | 58.45% | 8.45% | 11.27% | 0.0595 | 0.0625 |
| C4 | hybrid | 44.37% | 66.90% | 21.13% | 31.69% | 0.1331 | 0.1456 |
| C4 | r4 | 53.52% | 70.42% | 22.54% | 31.69% | 0.1401 | 0.1528 |

### Findings

- **C1/bm25** has the highest Source Hit@5 (65.49%), Recall@5 (36.62%), Recall@10 (45.07%), and MRR@5 (0.1899).
- **C2/bm25** has the highest Source Hit@10 (80.99%).
- **C3/hybrid** has the highest MRR@10 (0.2023), compared with 0.1995 for C1/bm25.
- BM25 has higher source-hit and known-reference recall scores than the other backends within every chunking strategy in this dataset. These observed results do not establish statistical significance or performance on other datasets.

### Metric interpretation

- **Source Hit@K:** at least one of the top K passages comes from any annotated source document. A document hit does not establish passage relevance or support for the claim.
- **Known-reference Recall@K:** the top K passages collectively cover all annotated gold spans for the case. In this scorer it is a binary complete-coverage measure per case, averaged across eligible cases; it is not the fraction of annotated documents retrieved.
- **MRR@K:** reciprocal rank of the first single passage that covers all annotated gold spans, or zero when none occurs within K. Collective coverage across multiple passages can therefore improve Recall@K without improving MRR@K.
- The counter direction is a retrieval query direction, not a label establishing that the returned evidence contradicts the claim. Alternative valid passages not included in the gold annotations remain unjudged.

## Authoritative result locations

All paths below are relative to `C:\Personal\Work\Github\GenAI-group-project`.

| Configurations | Result folder | Completed case runs | Scoring status | Readable score reports |
|---|---|---:|---|---|
| C1 × all four backends | `data/narrative/evaluation/test_claims_v2_result/retrieval_all_v1/configurations/C1` | 432/432 | Scored | `<backend>/evaluation.md` |
| C2 × all four backends | `data/narrative/evaluation/test_claims_v2_result/retrieval_all_v1/configurations/C2` | 432/432 | Scored | `<backend>/evaluation.md` |
| C3 × all four backends | `data/narrative/evaluation/test_claims_v2_result/retrieval_C3_v1` | 432/432 | Scored | `comparison.md` |
| C4 × bm25 and dense | `data/narrative/evaluation/test_claims_v2_result/retrieval_C4_bm25_dense_v1` | 216/216 | Scored | `comparison.md` |
| C4 × hybrid and r4 | `data/narrative/evaluation/test_claims_v2_result/retrieval_C4_hybrid_r4_v1` | 216/216 | Scored | `comparison.md` |

The original `retrieval_all_v1` folder contains interrupted C3/C4 work in addition to completed C1/C2. Use only its C1/C2 configurations for final results. Do not count its C3/C4 entries or the superseded `retrieval_C4_v1` checkpoint alongside the authoritative folders above.

## Scoring completion

No further scoring commands are required for these saved retrieval results. The C4 hybrid/r4 `comparison.json` now records `SCORED_SAVED_OUTPUTS`, two configurations, 108 cases per configuration, and 216 scored case runs.

The command below can regenerate that split's reports from the repository root if needed:

```powershell
python -m scripts.narrative.evaluation.evaluate_rag_comparison `
  --run-dir "data/narrative/evaluation/test_claims_v2_result/retrieval_C4_hybrid_r4_v1"
```

It scores saved outputs without retrieval, generation, or judge calls. This status refresh inspected existing files only; no benchmark or scoring reruns were performed.

The table in this note summarizes all 16 configurations. The underlying scorer outputs remain separate: C1/C2 have eight individual configuration reports; C3 and the two C4 splits have three matrix comparison reports. These cover all 1,728 unique case runs without counting copied checkpoints twice.

## GPT-5.4 RAG performance on the selected 32 cases

Scope: **saved candidate passages -> GPT-5.4 facts, summary and stance -> offline scoring**. This is the standalone RAG experiment. The full `python -m finsight run` orchestrator, post selection, extraction, routing, Auditor and final verdict are outside this evaluation. RAGAS quality judging is reported separately in the following section.

Selection: original CSV data rows **1-20 and 38-49**, inclusive and excluding the header. These are 32 distinct claims: 14 supported, 11 unsupported, six contradicting and one partially supported. The partial case maps to UNSUPPORTED for three-label stance scoring (14 SUPPORTED, 12 UNSUPPORTED, six CONTRADICTING). Retrieval scoring has **21 positive-reference cases**; 11 unsupported cases have no positive gold reference.

**All 16/16 configurations finished generation and offline scoring: 512/512 saved case runs**, with **428 valid assessments** and **84 invalid generations**. Earlier six-case pilots are excluded. Saved invalid outputs count as attempted runs but not valid stance assessments.

| Chunk | Saved / planned runs | Valid assessments | Invalid generations | Configurations scored |
|---|---:|---:|---:|---:|
| C1 | 128/128 | 105 | 23 | 4/4 |
| C2 | 128/128 | 101 | 27 | 4/4 |
| C3 | 128/128 | 112 | 16 | 4/4 |
| C4 | 128/128 | 110 | 18 | 4/4 |

### Stance quality and completion

Accuracy and macro F1 use valid assessments only; completion = valid / 32 attempted cases; all-case accuracy = correct / 32, including invalid outputs in the denominator. False support rate uses valid gold UNSUPPORTED/CONTRADICTING cases. Macro F1 averages all three labels. These are local classification metrics, not RAGAS judge scores.

| Chunk | Backend | Valid / attempted | Accuracy | Macro F1 | Completion | All-case accuracy | False support rate | Recorded cost (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | bm25 | 26/32 | 92.31% | 0.9097 | 81.25% | 75.00% | 0.00% | $1.3746 |
| C1 | dense | 25/32 | 88.00% | 0.8806 | 78.12% | 68.75% | 0.00% | $1.3918 |
| C1 | hybrid | 27/32 | 88.89% | 0.8821 | 84.38% | 75.00% | 0.00% | $1.3477 |
| C1 | r4 | 27/32 | 88.89% | 0.8852 | 84.38% | 75.00% | 0.00% | $1.3710 |
| C2 | bm25 | 28/32 | 92.86% | 0.9203 | 87.50% | 81.25% | 0.00% | $0.7946 |
| C2 | dense | 24/32 | 91.67% | 0.9280 | 75.00% | 68.75% | 0.00% | $0.7690 |
| C2 | hybrid | 25/32 | 92.00% | 0.9122 | 78.12% | 71.88% | 0.00% | $0.7906 |
| C2 | r4 | 24/32 | 91.67% | 0.9246 | 75.00% | 68.75% | 0.00% | $0.7985 |
| C3 | bm25 | 28/32 | 96.43% | 0.9680 | 87.50% | 84.38% | 0.00% | $0.9104 |
| C3 | dense | 31/32 | 83.87% | 0.8520 | 96.88% | 81.25% | 0.00% | $0.9027 |
| C3 | hybrid | 27/32 | 88.89% | 0.8893 | 84.38% | 75.00% | 0.00% | $0.8841 |
| C3 | r4 | 26/32 | 84.62% | 0.8667 | 81.25% | 68.75% | 0.00% | $0.9114 |
| C4 | bm25 | 30/32 | 86.67% | 0.8601 | 93.75% | 81.25% | 0.00% | $0.6713 |
| C4 | dense | 27/32 | 81.48% | 0.8482 | 84.38% | 68.75% | 0.00% | $0.6836 |
| C4 | hybrid | 25/32 | 92.00% | 0.9246 | 78.12% | 71.88% | 0.00% | $0.6425 |
| C4 | r4 | 28/32 | 85.71% | 0.8630 | 87.50% | 75.00% | 0.00% | $0.6414 |

Pooled descriptive totals across all 16 configurations: **381/428 correct valid assessments (89.02%)**, **83.59% completion**, and **381/512 all-case accuracy (74.41%)**. False supports: **0/217 valid negative assessments**. These repeat the same 32 claims across configurations; they are not 512 independent claims.

**Observed leaders:** C3/bm25 has the highest valid-only accuracy (96.43%), macro F1 (0.9680) and all-case accuracy (84.38%). C3/dense has the highest completion (96.88%), with lower valid-only accuracy (83.87%). C4/r4 has the lowest recorded generation cost ($0.6414 for 32 cases). These descriptive differences do not establish statistical significance.

### Retrieval scores for the same 32-case subset

Values average support/counter direction scores over the **21 positive-reference cases**, retaining rankings for invalid generations. They use the saved retrieval supplied to generation; no search was repeated. Compare within this subset: the full 108-case table above has 71 positive-reference cases and a different claim mix.

| Chunk | Backend | Source Hit@5 | Source Hit@10 | Recall@5 | Recall@10 | MRR@5 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | bm25 | 71.43% | 85.71% | 35.71% | 42.86% | 0.2357 | 0.2397 |
| C1 | dense | 52.38% | 57.14% | 14.29% | 14.29% | 0.0774 | 0.0774 |
| C1 | hybrid | 50.00% | 71.43% | 21.43% | 38.10% | 0.1464 | 0.1618 |
| C1 | r4 | 71.43% | 83.33% | 21.43% | 23.81% | 0.1179 | 0.1205 |
| C2 | bm25 | 78.57% | 88.10% | 33.33% | 38.10% | 0.1710 | 0.1790 |
| C2 | dense | 57.14% | 66.67% | 11.90% | 14.29% | 0.0841 | 0.0871 |
| C2 | hybrid | 66.67% | 76.19% | 23.81% | 33.33% | 0.1571 | 0.1700 |
| C2 | r4 | 64.29% | 76.19% | 16.67% | 19.05% | 0.1508 | 0.1532 |
| C3 | bm25 | 69.05% | 88.10% | 33.33% | 38.10% | 0.1881 | 0.1937 |
| C3 | dense | 42.86% | 61.90% | 11.90% | 19.05% | 0.0952 | 0.1056 |
| C3 | hybrid | 57.14% | 73.81% | 23.81% | 33.33% | 0.1548 | 0.1667 |
| C3 | r4 | 69.05% | 80.95% | 23.81% | 26.19% | 0.1488 | 0.1528 |
| C4 | bm25 | 76.19% | 85.71% | 33.33% | 33.33% | 0.1425 | 0.1425 |
| C4 | dense | 66.67% | 69.05% | 14.29% | 16.67% | 0.1060 | 0.1083 |
| C4 | hybrid | 52.38% | 66.67% | 19.05% | 23.81% | 0.1417 | 0.1490 |
| C4 | r4 | 61.90% | 76.19% | 11.90% | 26.19% | 0.0643 | 0.0846 |

### Invalid-generation diagnostics

The 84 invalid outputs fail output validation. Categories count affected case runs and overlap; do not add their counts.

| Validation issue | Affected case runs |
|---|---:|
| Stance/reason/fact consistency rule | 51 |
| Selected evidence exceeds the four-item limit | 22 |
| Unknown evidence ID or source-span pair | 14 |
| Summary schema violation | 5 |

Stance consistency errors include UNSUPPORTED with contradicting facts but an incompatible reason code, PARTIAL_SUPPORT without the required supporting-fact structure, and NO_RELEVANT_EVIDENCE with a non-empty fact selection. Long evidence-ID copying errors also occur but do not explain every invalid output. Failures remain in completion and all-case accuracy.

### Recorded generation usage

- Input: **4,638,486** tokens; output: **219,264** tokens; total: **4,857,750**.
- Recorded generation cost: **$14.885175** for 512 saved outputs. Unknown usage cases: **0**.
- These are saved-report estimates, not an account-wide bill. Earlier pilots, unrelated work and RAGAS judging are excluded.

### RAG result folders and submission files

All 16 subset folders are under `data/narrative/evaluation/test_claims_v2_result/` with this naming pattern:

```text
rag_<CHUNK>_<BACKEND>_gpt-5.4_saved_rows1-20_38-49_v1
```

For each configuration include `comparison.md/json/csv`, `stance_metrics.csv`, `stance_per_case.csv`, `stance_per_label.csv`, `stance_confusion_matrix.csv`, `per_case.csv` and `per_company.csv`. Include `matrix_configuration.json`, configuration-level `configuration.json` and `cases.csv`, and the case-level `result.json`, `retrieval.json`, `retrieval_source.json`, `llm_input.json` and `llm_response.json` for provenance.

## RAGAS evaluation performance

**Scope: the main 32-case C1-C4 cohort only.** Imported blurb/contextual experiments and judge costs remain separate below. Main generation remains complete and scored: **512/512 attempts, 428 valid assessments and 84 invalid generations**.

**Judge:** OpenAI `gpt-4o-mini`; generation `gpt-5.4`; RAGAS `0.4.3`; evaluator protocol `narrative_ragas_stance_v2`. No retrieval or regeneration was performed.

**Final completion: 16/16 judge configurations COMPLETE, 512/512 case rows reported and zero partial configurations.** Reported rows include invalid generations handled as NOT_APPLICABLE; not every metric has 512 scored values.

| Chunk | Complete judge configurations | Partial judge configurations | Reported / planned case rows |
|---|---:|---:|---:|
| C1 | 4/4 | 0 | 128/128 |
| C2 | 4/4 | 0 | 128/128 |
| C3 | 4/4 | 0 | 128/128 |
| C4 | 4/4 | 0 | 128/128 |

| Chunk | Backend | Judge status | Reported / planned | Valid generation rows reported | Metric errors | Judge attempts | Recorded judge cost |
|---|---|---|---:|---:|---:|---:|---:|
| C1 | bm25 | COMPLETE | 32/32 | 26 | 0 | 545 | $0.300640 |
| C1 | dense | COMPLETE | 32/32 | 25 | 0 | 518 | $0.294427 |
| C1 | hybrid | COMPLETE | 32/32 | 27 | 0 | 584 | $0.325585 |
| C1 | r4 | COMPLETE | 32/32 | 27 | 0 | 565 | $0.318864 |
| C2 | bm25 | COMPLETE | 32/32 | 28 | 0 | 583 | $0.220749 |
| C2 | dense | COMPLETE | 32/32 | 24 | 0 | 484 | $0.176566 |
| C2 | hybrid | COMPLETE | 32/32 | 25 | 0 | 508 | $0.187600 |
| C2 | r4 | COMPLETE | 32/32 | 24 | 0 | 492 | $0.182834 |
| C3 | bm25 | COMPLETE | 32/32 | 28 | 0 | 565 | $0.232749 |
| C3 | dense | COMPLETE | 32/32 | 31 | 0 | 617 | $0.257398 |
| C3 | hybrid | COMPLETE | 32/32 | 27 | 0 | 542 | $0.220814 |
| C3 | r4 | COMPLETE | 32/32 | 26 | 0 | 532 | $0.219310 |
| C4 | bm25 | COMPLETE | 32/32 | 30 | 0 | 625 | $0.201939 |
| C4 | dense | COMPLETE | 32/32 | 27 | 0 | 538 | $0.173611 |
| C4 | hybrid | COMPLETE | 32/32 | 25 | 0 | 524 | $0.162284 |
| C4 | r4 | COMPLETE | 32/32 | 28 | 0 | 559 | $0.178156 |

### Final judge scores with coverage

Score columns contain only the mean; separate coverage columns show **scored / all 32 selected cases**. All scored / total columns are grouped at the far right. Answer scored / total applies to faithfulness, citation correctness and relevance, whose counts match; stance metrics retain individual coverage columns. Every configuration is COMPLETE. Invalid/inapplicable/error metrics are excluded from means, not set to zero. Eligible denominators differ by metric and configuration. Compare judge quality alongside RAG completion and all-case accuracy; valid-only quality scores can conceal generation failures.

**Answer, citation and stance quality**

| Configuration | Faithfulness | Citation correctness | Relevance | Stance justification | Stance citation correctness | Unsupported behavior | Answer scored / total | Justification scored / total | Stance citation scored / total | Unsupported behavior scored / total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1/bm25 | 0.9173 | 0.9097 | 0.9167 | 0.6731 | 0.8938 | 1.0000 | 24/32 | 26/32 | 24/32 | 7/32 |
| C1/dense | 0.9082 | 0.7778 | 0.8125 | 0.6000 | 0.6909 | 1.0000 | 24/32 | 25/32 | 22/32 | 8/32 |
| C1/hybrid | 0.9762 | 0.8718 | 0.8846 | 0.6296 | 0.8220 | 1.0000 | 26/32 | 27/32 | 25/32 | 7/32 |
| C1/r4 | 0.9484 | 0.8718 | 0.8654 | 0.6296 | 0.8600 | 1.0000 | 26/32 | 27/32 | 25/32 | 8/32 |
| C2/bm25 | 0.9144 | 0.8580 | 0.9074 | 0.6250 | 0.8654 | 1.0000 | 27/32 | 28/32 | 26/32 | 8/32 |
| C2/dense | 0.9540 | 0.9015 | 0.9318 | 0.7083 | 0.8152 | 1.0000 | 22/32 | 24/32 | 22/32 | 7/32 |
| C2/hybrid | 0.9075 | 1.0000 | 0.8958 | 0.7200 | 0.9174 | 1.0000 | 24/32 | 25/32 | 23/32 | 7/32 |
| C2/r4 | 0.9262 | 0.9394 | 0.9318 | 0.6667 | 0.9167 | 1.0000 | 22/32 | 24/32 | 22/32 | 6/32 |
| C3/bm25 | 0.9594 | 0.8889 | 0.8704 | 0.6786 | 0.8705 | 1.0000 | 27/32 | 28/32 | 26/32 | 8/32 |
| C3/dense | 0.8915 | 0.8161 | 0.8793 | 0.4839 | 0.7397 | 1.0000 | 29/32 | 31/32 | 26/32 | 12/32 |
| C3/hybrid | 0.8995 | 0.8590 | 0.9038 | 0.7037 | 0.8462 | 1.0000 | 26/32 | 27/32 | 26/32 | 9/32 |
| C3/r4 | 0.9339 | 0.9067 | 0.9000 | 0.5000 | 0.8413 | 1.0000 | 25/32 | 26/32 | 23/32 | 8/32 |
| C4/bm25 | 0.8739 | 0.9310 | 0.8793 | 0.6167 | 0.8018 | 1.0000 | 29/32 | 30/32 | 28/32 | 10/32 |
| C4/dense | 0.8912 | 0.9583 | 0.8125 | 0.5185 | 0.7217 | 1.0000 | 24/32 | 27/32 | 20/32 | 9/32 |
| C4/hybrid | 0.9118 | 0.9583 | 0.9167 | 0.7000 | 0.8125 | 1.0000 | 24/32 | 25/32 | 24/32 | 6/32 |
| C4/r4 | 0.8958 | 0.8718 | 0.8077 | 0.5714 | 0.8526 | 1.0000 | 26/32 | 28/32 | 26/32 | 9/32 |

**Context and reference coverage**

| Configuration | Retrieved context precision | Retrieved context recall | Selected context precision | Selected context recall | Summary completeness | Scored / total cases |
|---|---:|---:|---:|---:|---:|---:|
| C1/bm25 | 0.9288 | 0.8947 | 1.0000 | 0.8158 | 0.6063 | 19/32 |
| C1/dense | 0.6885 | 0.7361 | 0.8843 | 0.6389 | 0.3944 | 18/32 |
| C1/hybrid | 0.8495 | 0.8690 | 0.9524 | 0.7857 | 0.4310 | 21/32 |
| C1/r4 | 0.8266 | 0.8625 | 0.9500 | 0.7250 | 0.4560 | 20/32 |
| C2/bm25 | 0.8563 | 0.7937 | 0.9246 | 0.7976 | 0.5262 | 21/32 |
| C2/dense | 0.8306 | 0.7361 | 0.9352 | 0.7593 | 0.4167 | 18/32 |
| C2/hybrid | 0.9020 | 0.7500 | 0.9769 | 0.7778 | 0.4217 | 18/32 |
| C2/r4 | 0.8548 | 0.8111 | 1.0000 | 0.8000 | 0.4233 | 18/32 |
| C3/bm25 | 0.8914 | 0.9000 | 0.9819 | 0.7750 | 0.4885 | 20/32 |
| C3/dense | 0.7682 | 0.7525 | 0.9417 | 0.7450 | 0.3770 | 20/32 |
| C3/hybrid | 0.8925 | 0.8289 | 0.9240 | 0.7719 | 0.4326 | 19/32 |
| C3/r4 | 0.8334 | 0.7746 | 0.8947 | 0.6175 | 0.3147 | 19/32 |
| C4/bm25 | 0.8438 | 0.7913 | 0.9233 | 0.8190 | 0.3619 | 21/32 |
| C4/dense | 0.7874 | 0.6019 | 0.9213 | 0.3889 | 0.2094 | 18/32 |
| C4/hybrid | 0.7934 | 0.6947 | 0.9386 | 0.6053 | 0.3311 | 19/32 |
| C4/r4 | 0.8362 | 0.7933 | 0.8750 | 0.7583 | 0.3475 | 20/32 |

Metric mappings: selected-context precision = `evidence_selection_quality`; selected-context recall = `evidence_completeness`. Retrieved-context scores use candidates before LLM selection. Summary completeness uses factual-correctness recall, not F1. Context precision is rank-aware. Relevance and stance justification normalize 0/1/2 to 0/0.5/1. Unsupported positive-reference metrics are N/A; unsupported behavior uses expected UNSUPPORTED, including partial support. Judge estimates do not replace independent gold review or Auditor verdicts.

### Final metric coverage across all configurations

These counts pool repeated evaluations of the same 32 claims, not independent claims. They describe coverage, not a new quality score.

| Metric | Scored / 512 saved case runs |
|---|---:|
| Faithfulness | 405/512 |
| Citation correctness | 405/512 |
| Relevance | 405/512 |
| Stance justification | 428/512 |
| Stance citation correctness | 388/512 |
| Unsupported behavior | 129/512 |
| Retrieved context precision | 309/512 |
| Retrieved context recall | 309/512 |
| Selected context precision | 309/512 |
| Selected context recall | 309/512 |
| Summary completeness | 309/512 |

### Recorded judge usage and evaluation status

- Judge attempts: **8781**; metric errors: **0**; unknown-usage attempts: **0**.
- Judge tokens: **20,321,925 input / 1,008,732 output**; total **21,330,657**; recorded judge cost: **$3.653528**.
- Main-cohort generation plus completed judging: **$18.538703**. Imported runs, earlier pilots and unrelated work are excluded.
- **No additional main-cohort generation or judge runs are required for this saved evaluation scope.** Preserve current outputs if revising prompts/schema to address the 84 invalid generations; revised runs belong in a new version.
- Full FinSight orchestrator/Auditor performance remains outside this standalone benchmark.

### RAGAS submission files

All 16 completed main-cohort judge folders are under `data/narrative/evaluation/test_claims_v2_result/`:

```text
ragas_<CHUNK>_<BACKEND>_gpt-4o-mini_rows1-20_38-49_v1
```

Submit `ragas_report.md/json`, `ragas_metrics.csv`, `ragas_per_case.csv` and `configuration.json`. Preserve `samples/` for per-case checkpoints/applicability/error reasons and `calls/` for request/usage provenance.

## Blurb and contextual methods: matching report structure

This section follows the C1-C4 report order: completion, stance quality, retrieval scores, RAGAS scores, usage and result locations. It consolidates the existing uploaded artifacts without changing their case snapshots or recreating model outputs.

### Completion and experimental scope

The uploads contain two added methods plus a matching imported C1/BM25 baseline. Each has a disjoint 20-case `_v1` batch and 12-case `_38_49` batch: **32 cases per method, 96 generation attempts total**. These are separate from the main 512-run C1-C4 experiment. Only contextual/hybrid and blurb-contextual/bm25 were supplied; additional backend combinations have not been evaluated.

| Method | Backend | Build | Saved / planned runs | Valid assessments | Invalid generations | Generation errors |
|---|---|---|---:|---:|---:|---:|
| Imported C1/BM25 baseline | bm25 | `full_v1` | 32/32 | 27 | 5 | 0 |
| Contextual hybrid | hybrid | `row512_v1` | 32/32 | 27 | 4 | 1 |
| Blurb + contextual BM25 | bm25 | `blurb_v2_top50` | 32/32 | 27 | 5 | 0 |

**Claim-set difference remains:** the uploaded cohort shares 31 main-benchmark claims and replaces `INF-GOOGL-PAR-03` with `INF-AAPL-SUP-02`. Its labels are 15 supported, 11 unsupported and six contradicting, rather than the main set's 14 supported, 11 unsupported, six contradicting and one partial-support case. Imported filenames are not authoritative row selectors for the current combined CSV. Compare uploaded methods with their matching imported baseline until the missing case is evaluated.

Recorded generation settings are GPT-5.4, top-k 10 per direction, candidate-k 50, output budget 3,000, query expansion disabled and rerank overflow `error`. The imported builds/source-code hashes differ from the main experiment. Original source-machine macOS paths and manifests remain intact.

### Stance quality and completion

The columns and denominators match the C1-C4 stance table: accuracy/F1 use valid outputs; completion = valid / 32 attempted; all-case accuracy = correct / 32. False support rate uses valid gold negative cases. Metrics were recomputed from all 32 saved case records with `stance_record` and `summarize_stances`, not averaged across batches. Saved citation-validation outcomes are retained; missing corpus artifacts prevent a fresh source-grounded citation audit.

| Method | Backend | Valid / attempted | Accuracy | Macro F1 | Completion | All-case accuracy | False support rate | Recorded cost (USD) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Imported C1/BM25 baseline | bm25 | 27/32 | 96.30% | 0.9441 | 84.38% | 81.25% | 0.00% | $1.3723 |
| Contextual hybrid | hybrid | 27/32 | 77.78% | 0.7838 | 84.38% | 65.62% | 0.00% | $0.7242 |
| Blurb + contextual BM25 | bm25 | 27/32 | 85.19% | 0.8510 | 84.38% | 71.88% | 0.00% | $0.7668 |

On this imported cohort the matching C1/BM25 baseline has the highest all-case accuracy (81.25%), followed by blurb/contextual BM25 (71.88%) and contextual hybrid (65.62%). All have 27 valid assessments, but their valid-case sets can differ. These are descriptive results without a significance test.

### Retrieval performance: available metrics and missing scores

**Blurb has judge-based context performance scores, but neither added method has validated offline Source Hit/Recall/MRR scores yet.** These are distinct metric families and must not be substituted for each other in the C1-C4 comparison.

#### Available RAGAS context scores

| Method | Retrieved context precision | Retrieved context recall | Selected context precision | Selected context recall | Scored / total cases |
|---|---:|---:|---:|---:|---:|
| Blurb + contextual BM25 | 0.8730 | 0.8300 | 0.8944 | 0.7250 | 20/32 |
| Contextual hybrid | N/A | N/A | N/A | N/A | No judge output supplied |

The blurb values pool the individual scored results from the uploaded 20-case and 12-case COMPLETE RAGAS reports. All four metrics have 20 scored cases out of 32 total cases; missing/inapplicable cases are excluded from their means. Retrieved-context metrics assess candidates before LLM selection, whereas selected-context metrics assess the chosen evidence. These are GPT-4o-mini judge estimates, not the known-reference span coverage or reciprocal-rank metrics below.

#### Offline source-grounded retrieval scores

| Method | Backend | Source Hit@5 | Source Hit@10 | Recall@5 | Recall@10 | MRR@5 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Imported C1/BM25 baseline | bm25 | N/A | N/A | N/A | N/A | N/A | N/A |
| Contextual hybrid | hybrid | N/A | N/A | N/A | N/A | N/A | N/A |
| Blurb + contextual BM25 | bm25 | N/A | N/A | N/A | N/A | N/A | N/A |

**N/A means no validated retrieval score was supplied, not zero.** The uploads contain rankings and generation captures but no `comparison.json`/`evaluation.json` retrieval-score reports. Matching local contextual indexes/source maps for `contextual/row512_v1` and `contextual/blurb_v2_top50` are absent. Import those exact processed build artifacts, or obtain source-machine score reports for these case snapshots, before filling this table. No approximate text-matching metric has been substituted. The existing 108-claim retrieval benchmark remains unchanged.

### RAGAS evaluation performance

| Method | Backend | Judge model | Judge status | Reported / planned cases | Metric errors |
|---|---|---|---|---:|---:|
| Imported C1/BM25 baseline | bm25 | N/A | No matching imported judge output | 0/32 | N/A |
| Contextual hybrid | hybrid | N/A | No judge output supplied | 0/32 | N/A |
| Blurb + contextual BM25 | bm25 | gpt-4o-mini | COMPLETE (20 + 12 batches) | 32/32 | 0 |

The two blurb judge reports use RAGAS 0.4.3 and protocol `narrative_ragas_stance_v2`. Main-cohort C1 judge reports cannot be reused for the imported baseline because the selected claims and generated outputs differ.

#### Completed judge scores with coverage

Score columns contain only the mean. Separate coverage columns at the far right show **scored / reported cases**, following the C1-C4 convention. Answer scored / total applies jointly to faithfulness, citation correctness and relevance. Combined means pool individual scored values across both batches, weighting by each metric's scored-case count. Invalid/inapplicable cases are excluded from means rather than treated as zero.

**Answer, citation and stance quality**

| Method | Faithfulness | Citation correctness | Relevance | Stance justification | Stance citation correctness | Unsupported behavior | Answer scored / total | Justification scored / total | Stance citation scored / total | Unsupported behavior scored / total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Imported C1/BM25 baseline | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Contextual hybrid | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Blurb + contextual BM25 | 0.9689 | 0.9487 | 0.8654 | 0.5741 | 0.9440 | 1.0000 | 26/32 | 27/32 | 25/32 | 7/32 |

**Context and reference coverage**

| Method | Retrieved context precision | Retrieved context recall | Selected context precision | Selected context recall | Summary completeness | Scored / total cases |
|---|---:|---:|---:|---:|---:|---:|
| Imported C1/BM25 baseline | N/A | N/A | N/A | N/A | N/A | N/A |
| Contextual hybrid | N/A | N/A | N/A | N/A | N/A | N/A |
| Blurb + contextual BM25 | 0.8730 | 0.8300 | 0.8944 | 0.7250 | 0.3930 | 20/32 |

Blurb faithfulness is 0.9689 (26 cases), summary completeness 0.3930 (20 cases), and stance justification 0.5741 (27 cases). High faithfulness does not establish full claim coverage or correct stance. Unsupported behavior is 1.0000 on seven scored cases only. The metric definitions match the main RAGAS section; judge estimates do not replace gold review or Auditor verdicts.

### Recorded generation and judge usage

| Method | Input tokens | Output tokens | Recorded generation cost | Unknown-usage cases |
|---|---:|---:|---:|---:|
| Imported C1/BM25 baseline | 463,515 | 14,234 | $1.372298 | 0 |
| Contextual hybrid | 203,628 | 14,339 | $0.724155 | 1 |
| Blurb + contextual BM25 | 218,496 | 14,706 | $0.766830 | 0 |

- Imported generation totals: **$2.863283** for 96 saved attempts.
- Imported blurb judge: **1,090,494 input / 62,175 output tokens**, **555 attempts**, **$0.200879**, unknown-usage attempts **0**.
- Total recorded imported runtime cost: **$3.064162**, separate from main-cohort totals.
- Contextual hybrid has an `APIConnectionError` for `INF-META-SUP-03` with unknown token usage. The recorded cost omits any unrecorded charge for that request. Context-blurb generation, indexing and embedding-build costs are also excluded.

### Authoritative result locations and submission files

Paths below are relative to `data/narrative/evaluation/test_claims_v2_result/`. Original imported folders remain unchanged.

| Method | Generation folder | Cases | Valid outputs |
|---|---|---:|---:|
| Imported C1/BM25 baseline | `rag_C1_bm25_openai_v1` | 20 | 16 |
| Imported C1/BM25 baseline | `rag_C1_bm25_openai_38_49` | 12 | 11 |
| Contextual hybrid | `rag_contextual_hybrid_openai_v1` | 20 | 17 |
| Contextual hybrid | `rag_contextual_hybrid_openai_38_49` | 12 | 10 |
| Blurb + contextual BM25 | `rag_blurb_contextual_bm25_openai_v1` | 20 | 16 |
| Blurb + contextual BM25 | `rag_blurb_contextual_bm25_openai_38_49` | 12 | 11 |

| Blurb judge folder | Status | Cases | Recorded judge cost |
|---|---|---:|---:|
| `ragas_blurb_contextual_bm25_openai_v1` | COMPLETE | 20/20 | $0.134910 |
| `ragas_blurb_contextual_bm25_openai_38_49` | COMPLETE | 12/12 | $0.065969 |

For review include the source case snapshots, manifests, case-level `result.json`/`retrieval.json`, captured LLM inputs/responses, and uploaded judge reports/checkpoints. Stance metrics in this note were derived from those artifacts; no imported matrix score reports were fabricated.

### Requirements for the exact same 32-claim selection

To run the new methods on precisely the C1-C4 claim set, use the original `cases_combined.csv` with `--case-ranges 1-20 38-49`, GPT-5.4 and the corresponding contextual build. The required `INF-GOOGL-PAR-03` result is absent from all three imported method captures; `INF-AAPL-SUP-02` is outside the main selection. Do not rename/relabel the existing snapshots as if the missing case had been run.

Matching the exact claim set requires the missing contextual build artifacts and a new, separately versioned run/evaluation scope. Existing imported results remain evidence for their original cohort. No new generation, retrieval or judge calls were made during this report-format update.

## Files to include for review

1. The dataset CSV above and this status note.
2. For C1/C2: each backend's `evaluation.md` and `evaluation.json` under the listed configuration directories.
3. For C3 and each C4 split: `comparison.md`, `comparison.json`, `comparison.csv`, `per_case.csv`, and `per_company.csv` (all verified present).
4. For an audit trail: `matrix_configuration.json` from each parent run folder, plus each selected configuration's `configuration.json`, `cases.csv`, `report.json`, and `cases/<CASE_ID>/result.json` and `retrieval.json`.
5. The evaluation code and processed corpus/build provenance if the professor needs to reproduce scoring rather than inspect saved reports. Record the submitted repository commit as well.
6. For the 32-case GPT-5.4 experiment, include its separate RAG reports, stance diagnostics, selected-case manifests and captured generation artifacts listed above. Keep its scope distinct from the 108-case retrieval benchmark.
7. For RAGAS, include the judge reports, metric coverage, evaluator identity and checkpoints listed above; explicitly mark any IN_PROGRESS results as partial.
8. For the imported blurb/contextual experiment, include its separate 20-case and 12-case artifacts and matching imported C1 baseline; identify the cohort mismatch and missing contextual retrieval builds.

The reports contain Source Hit@5/@10, known-reference Recall@5/@10, and MRR@5/@10. Support and counter search directions are reported separately; state any averaging rule when presenting a combined table. Gold-reference retrieval metrics use eligible annotated cases, rather than treating missing gold references as retrieval misses.

## Retrieval-only experiment settings

For the separate generation stage, Workflow D documents [generation from these saved retrieval captures](workflow_d_evaluation.md#generate-from-saved-retrieval-with-gpt-54) using `--retrieval-run-dir`. This preserves saved candidates and rankings while writing GPT-5.4 outputs to separate folders. The 108-claim retrieval-only results retain their original scope; generation and RAGAS results for the 32-case subset are reported in their own sections above.

- Corpus build: `full_v1` under `data/narrative/processed/chunking_strategies`.
- Top-k: 10; candidate-k: 50; rerank overflow: `truncate`.
- Mode: retrieval only; no generation or judge model used for these captures.
- Runner: `scripts/narrative/evaluation/run_rag_comparison.py`.
- Scoring: `scripts/narrative/evaluation/evaluate_rag_pilot.py` for individual C1/C2 configurations; `scripts/narrative/evaluation/evaluate_rag_comparison.py` for the completed C3/C4 matrices.

Saved manifests are the detailed source of experiment settings and hashes. This document is a dated status snapshot, not a live tracker.
