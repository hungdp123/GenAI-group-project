# Contextual row retrieval, 25 Oct 2026

Reference note before trying a different retrieval method. This is not a replacement for the scored C1–C4 table in [workflow_d_evaluation.md](workflow_d_evaluation.md).

## What was built

Row-header contextual indexes for all 16 companies:

`data/narrative/processed/chunking_strategies/contextual/row512_v1/<TICKER>/`

- Each inlined filing table row is its own passage. Column headers stay on the row. The searchable line adds company, fiscal period, segment, and metric.
- BM25 keeps a percent as one token (`28%` does not match page 28 or `$28,786`). Growth wording and segment names add `tag_metric_growth` and `tag_segment_*`.
- A claim that states one increase skips the counter search, and a passage must contain the claimed percent and quarter when the claim names them.
- Semantic vectors are `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, 256-token window), stored under each company's `indexes/qdrant/`. Passages were packed for a 512-token budget, so MiniLM still sees only the start of a long passage.

## Score

84 supported and contradicting claims from `data/narrative/evaluation/test_claims_v1/cases.csv`. The 42 unsupported claims are not in these percentages. Support and counter are separate searches. A recall or MRR hit means one retrieved passage contains the full gold sentence. Source hit means the annotated document is in the list. This is stricter than Workflow D recall when the gold span is split across passages.

| Strategy / method | Cases / positive references | Direction | Source Hit@5 / @10 | Recall@5 / @10 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|
| Contextual / bm25 | 126 / 84 | support | 100.00% / 100.00% | 77.38% / 80.95% | 0.6508 | 0.6555 |
| Contextual / bm25 | 126 / 84 | counter | 100.00% / 100.00% | 80.95% / 84.52% | 0.6815 | 0.6867 |
| Contextual / dense | 126 / 84 | support | 85.71% / 96.43% | 45.24% / 55.95% | 0.3077 | 0.3218 |
| Contextual / dense | 126 / 84 | counter | 82.14% / 91.67% | 42.86% / 54.76% | 0.2788 | 0.2959 |
| Contextual / hybrid | 126 / 84 | support | 94.05% / 100.00% | 66.67% / 79.76% | 0.4970 | 0.5126 |
| Contextual / hybrid | 126 / 84 | counter | 85.71% / 100.00% | 64.29% / 83.33% | 0.4919 | 0.5152 |

Her best published rows, copied here for comparison:

| Strategy / method | Direction | Source Hit@5 / @10 | Recall@5 / @10 | MRR@10 |
|---|---|---|---|---|
| C1 / bm25 | support | 97.62% / 98.81% | 84.52% / 88.10% | 0.7626 |
| C1 / bm25 | counter | 97.62% / 98.81% | 86.90% / 91.67% | 0.7468 |
| C1 / hybrid | support | 78.57% / 100.00% | 63.10% / 90.48% | 0.5147 |
| C1 / hybrid | counter | 79.76% / 98.81% | 63.10% / 92.86% | 0.4728 |

Contextual BM25 ties the best source-hit rate. Recall@10 and MRR@10 stay below C1 BM25. Contextual hybrid recall@10 stays below C1 hybrid.

## Checks that explain the table

Amazon, “AWS grew 28% in Q1 2026,” cutoff 30 Apr 2026. The first contextual passage is the earnings-release row `Q1 2026: $37,587; Y/Y % Change: 28%`. Her original BM25 already had that fact in a longer passage, and its top three hits all stated AWS +28%.

| Case | Her BM25 | Contextual BM25 | Contextual hybrid |
|---|---|---|---|
| AAPL-CTR-04. Gold sentence: “we use some third-party cloud, and we do our own data centers.” | Missed | Found | Right call, sentence not in top 5 |
| MSFT-CTR-01. Gold sentence: “a consistent multi-cloud and on-premises management platform.” | Missed in top 5 | Found at rank 1 | Found |
| CRM-SUP-01. Gold sentence: “deploying autonomous AI agents across business functions.” | Right 10-K, sentence absent | Still absent | Still absent on this index |

Dense search on her existing C1-style CRM index did put the Agentforce sentence first. Hybrid then mixed BM25 back in and the neighboring paragraph took rank 1. The contextual CRM passages are still longer than MiniLM’s 256 tokens, and that sentence sits past the embedded opening.

## Passages sent to the LLM, scored like Workflow D

The two Contextual / bm25 rows are not two indexes. Her table gives every method two rows: `support` is the search for confirming wording, and `counter` is the search for contradicting wording. The same split is used below.

These rows are the reranked hybrid lists, the passages that then go to the LLM. A recall or MRR hit means one of those passages contains the full gold sentence. Source hit means the annotated document is in the list. 84 labeled claims.

| Strategy / method | Cases / positive references | Direction | Source Hit@5 / @10 | Recall@5 / @10 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|
| Contextual / hybrid + rerank | 126 / 84 | support | 96.43% / 100.00% | 76.19% / 77.38% | 0.6617 | 0.6634 |
| Contextual / hybrid + rerank | 126 / 84 | counter | 95.24% / 100.00% | 76.19% / 78.57% | 0.6548 | 0.6582 |
| Contextual / hybrid + rerank, top 10 sent to the LLM | 126 / 84 | merged list | 96.43% / 100.00% | 76.19% / 77.38% | 0.6617 | 0.6634 |

Reranking does not beat her C1 / bm25 recall@10 of 88.10% and 91.67%, or its MRR@10 of 0.7626 and 0.7468.

The last row is the single list of up to 10 passages actually given to Gemini, after the support and counter searches are merged. Sending 10 instead of 5 does not raise recall or MRR. The model then quotes the gold sentence in 39 of 84 labeled claims, 46.43%. Citation checks passed for 106 of all 126 cases. Cost was $0.44. Per-case rows are in `data/narrative/experiments/contextual_llm_top10.json`.

## C1 BM25 plus contextual dense, then rerank

BM25 uses her C1 passages. Dense uses the contextual row passages. The two lists are fused, then the cross-encoder keeps the top 10. Same 84 labeled claims and same hit definition as the tables above.

| Strategy / method | Cases / positive references | Direction | Source Hit@5 / @10 | Recall@5 / @10 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|
| C1 BM25 + contextual dense + rerank | 126 / 84 | support | 92.86% / 98.81% | 71.43% / 77.38% | 0.6179 | 0.6254 |
| C1 BM25 + contextual dense + rerank | 126 / 84 | counter | 92.86% / 100.00% | 72.62% / 76.19% | 0.6403 | 0.6448 |

Recall@10 is 77.38% and 76.19%. Her C1 BM25 is 88.10% and 91.67%. MRR@10 is 0.6254 and 0.6448. Her C1 BM25 is 0.7626 and 0.7468. Mixing her best lexical list with the contextual dense list, then reranking, does not beat C1 BM25.

## Blurb plus contextual rows

A model sentence was prepended only to the 4,314 passages that already appear in the top 50 for the 84 labeled claims. 46 sentences were reused. The other 4,268 were written on 26 Oct, after the Gemini quota reset. Each blurb sits in front of the existing row header. BM25 and the MiniLM vectors for those passages were rebuilt. Other passages keep the row header only.

| Strategy / method | Cases / positive references | Direction | Source Hit@5 / @10 | Recall@5 / @10 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|
| Blurb + contextual / bm25 | 126 / 84 | support | 100.00% / 100.00% | 76.19% / 80.95% | 0.6399 | 0.6459 |
| Blurb + contextual / bm25 | 126 / 84 | counter | 100.00% / 100.00% | 79.76% / 83.33% | 0.6726 | 0.6775 |
| Blurb + contextual / dense | 126 / 84 | support | 80.95% / 90.48% | 45.24% / 51.19% | 0.2869 | 0.2947 |
| Blurb + contextual / dense | 126 / 84 | counter | 79.76% / 88.10% | 38.10% / 50.00% | 0.2480 | 0.2643 |
| Blurb + contextual / hybrid | 126 / 84 | support | 91.67% / 100.00% | 67.86% / 78.57% | 0.4589 | 0.4716 |
| Blurb + contextual / hybrid | 126 / 84 | counter | 88.10% / 100.00% | 65.48% / 84.52% | 0.4464 | 0.4700 |

Recall@10 and MRR@10 stay below her C1 BM25, 88.10% and 91.67%, and 0.7626 and 0.7468. They also stay at or below the contextual rows without a blurb. The full 121,734 blurb calls are not supported by this test.

## C1 BM25 plus blurb contextual dense, then rerank

BM25 uses her C1 passages. Dense uses the blurb-plus-contextual row index: a model sentence in front of the row header on the 4,314 top-50 passages, and the row header alone on the rest. The two lists of 50 are fused, then the cross-encoder keeps the top 10. Same 84 labeled claims and same hit definition as the tables above.

| Strategy / method | Cases / positive references | Direction | Source Hit@5 / @10 | Recall@5 / @10 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|
| C1 BM25 + blurb contextual dense + rerank | 126 / 84 | support | 91.67% / 98.81% | 69.05% / 77.38% | 0.6190 | 0.6304 |
| C1 BM25 + blurb contextual dense + rerank | 126 / 84 | counter | 91.67% / 100.00% | 70.24% / 75.00% | 0.6270 | 0.6333 |

Recall@10 is 77.38% and 75.00%. The same mix without blurbs was 77.38% and 76.19%. Her C1 BM25 is 88.10% and 91.67%. MRR@10 is 0.6304 and 0.6333, against 0.6254 and 0.6448 without blurbs, and 0.7626 and 0.7468 for C1 BM25. Putting the blurb vectors on the dense side does not move this mix toward C1 BM25.

## Support and counter merged

Workflow D prints support and counter on separate rows. The table below folds each pair into one figure: both directions score the same 84 claims, so each rate is `(support hits + counter hits) / 168`. MRR is the mean of the two direction scores. Her C1 rows are merged the same way.

| Strategy / method | Source Hit@5 / @10 | Recall@5 / @10 | MRR@5 | MRR@10 |
|---|---|---|---|---|
| C1 / bm25 | 97.62% / 98.81% | 85.71% / 89.88% | 0.7492 | 0.7547 |
| C1 / dense | 77.38% / 89.88% | 36.31% / 48.21% | 0.2170 | 0.2334 |
| C1 / hybrid | 79.17% / 99.40% | 63.10% / 91.67% | 0.4627 | 0.4938 |
| C1 / r4 | 91.07% / 98.81% | 48.21% / 51.79% | 0.3838 | 0.3888 |
| Contextual / bm25 | 100.00% / 100.00% | 79.17% / 82.74% | 0.6662 | 0.6711 |
| Contextual / dense | 83.93% / 94.05% | 44.05% / 55.36% | 0.2933 | 0.3089 |
| Contextual / hybrid | 89.88% / 100.00% | 65.48% / 81.55% | 0.4945 | 0.5139 |
| Contextual / hybrid + rerank | 95.83% / 100.00% | 76.19% / 77.98% | 0.6583 | 0.6608 |
| C1 BM25 + contextual dense + rerank | 92.86% / 99.40% | 72.02% / 76.79% | 0.6291 | 0.6351 |
| C1 BM25 + blurb contextual dense + rerank | 91.67% / 99.40% | 69.64% / 76.19% | 0.6230 | 0.6319 |
| Blurb + contextual / bm25 | 100.00% / 100.00% | 77.98% / 82.14% | 0.6562 | 0.6617 |
| Blurb + contextual / dense | 80.36% / 89.29% | 41.67% / 50.60% | 0.2674 | 0.2795 |
| Blurb + contextual / hybrid | 89.88% / 100.00% | 66.67% / 81.55% | 0.4526 | 0.4708 |

On this single figure, her C1 BM25 still leads recall@10 (89.88%) and MRR@10 (0.7547). Her C1 hybrid still leads recall@10 among the hybrid rows (91.67%). Contextual BM25 is the closest of the new rows: recall@10 82.74%, MRR@10 0.6711, and it is the only row at 100% source hit at both cutoffs.

