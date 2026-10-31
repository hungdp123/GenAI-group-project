# Claim matching: user prompt → influencer line → facts

FinSight checks **what the influencer said**, not what the user typed. The user's text is only used to
find the influencer's line. Code: `finsight/agents/claim_matching/`.

## The four steps

| Step | What | Where |
|---|---|---|
| 1. Filter | Catalog lines from this influencer, about this company, said on or before the cut-off, in the project window. No ML. | `catalog.lines_for` |
| 2. Rank | Split the prompt and every line into **facts** (measure, direction, one figure, period). Score each line by how many of the user's facts it covers. BM25 breaks ties. | `facts.split_facts`, `matcher.score_lines` |
| 3. Pick | The best video, plus extra lines in it that cover facts the first line misses. With two videos that both state everything → ask which one. In paid mode (`FINSIGHT_MATCHER=llm`, default when not mock) a model picks from the top 8 numbered lines. It can only answer with ids from that list. Anything else falls back to the rules pick. | `matcher.match_claim` |
| 4. Align | Put every fact in one of four buckets (below). | `facts.align` |

When the model's picks span several videos (it often lists every video that repeats a statement), FinSight asks "which video?" only if the rules also see a tie; otherwise it keeps the video the rules rank highest. The matcher is recorded as the `claim_matcher` agent call, so its tokens and cost are in the decision log.

If the catalog has no line that covers the prompt, sentences from the stored transcripts compete as well (`catalog.transcript_lines`). On a machine with the catalog but no transcripts, posts are built from the catalog (`catalog.posts_from_catalog`).

## The four buckets

| Bucket | Example (user → influencer) | What happens |
|---|---|---|
| `both` | "grew 24%" → "Google grew by 24% year-over-year" | Verified. **Counts for the headline verdict.** |
| `conflict` | "grew 40%" → "18.3% growth"; "revenue grew 40%" → "Azure grew 40%"; "rose to 68%" → "fell to 68%" | The influencer's version is verified and counts for the headline. The mismatch is shown ("You said 40%; they said 18.3%."). |
| `quote_only` | (nothing) → "marking its twelfth consecutive quarter of double-digit growth" | Verified, shown as "Also in that line". **Not counted in the headline.** |
| `user_only` | "to $110 billion" → (not in the line) | Never verified as the influencer's claim. Shown as "They did not say $110 billion". |

A share-price view in the prompt ("could be worth $2,000") is set aside: never matched, never verified.

### Sub-claims

* A matched line with **one fact** stays one sub-claim (the stored catalog row, with its type).
* A line with **several facts** is **split per fact**, so each fact gets its own verdict. "Google grew 24%, marking its twelfth straight quarter" becomes `t123-f1` and `t123-f2`. Before this, a fully restated line was one sub-claim, and the true 24% and the unverifiable twelfth quarter got one blended "partly supported".
* A split fact keeps the full line in `parent_span`. Context that only the rest of the line supplied is added in brackets, e.g. `[Alphabet] marking its twelfth…`.
* `expansion_rule` stays empty, so the report never merges split facts back into one verdict.
* `FinalAssessment.headline_sub_claim_ids` lists the sub-claims the overall verdict covers.

### Cut-off

If the line the user means was said **after** their cut-off, the request is refused with the date ("Joseph Carlson said this on 30 Apr 2026, after the cut-off 29 Apr 2026"). FinSight does not substitute an older, weaker line.

## Contracts (schema 0.12, additive; tell Hung)

* `SubClaim.alignment`: `both | conflict | quote_only` (None outside the stored-post flow).
* `FactAlignment`: `{bucket, user_span, quote_span, sub_claim_id, note}`.
* `ClaimMatch`: `{status MATCHED|NONE|ASK, method rules|llm, user_claim, video_id, title, upload_date, quotes, quote_ids, facts, candidates_considered, question}`.
* `FinalAssessment.match`, `FinalAssessment.headline_sub_claim_ids`, `FinSightState.match`.

All fields are optional, so v0.11 payloads still validate.

## Gold labels: three layers, scored separately

Each dev item (`finsight/eval/datasets/claims_dev.jsonl`, built by `scripts/build_dev_gold.py`) carries:

```json
"gold_match": {"status": "MATCHED", "quote_ids": ["IHlc66vNdhY-GOOGL-t123"], "acceptable_quote_ids": []},
"gold_alignment": [
  {"bucket": "both", "quote_span": "Google grew by 24% year-over-year", "quote_id": "IHlc66vNdhY-GOOGL-t123",
   "user_span": "24%", "gold_verdict": "SUPPORTED", "evidence": ["10-Q filed 2026-07-23: +24.2%"]},
  {"bucket": "quote_only", "quote_span": "marking its twelfth consecutive quarter of double-digit growth",
   "quote_id": "IHlc66vNdhY-GOOGL-t123", "gold_verdict": "INSUFFICIENT_EVIDENCE"}
]
```

| Score | Question it answers | Run |
|---|---|---|
| Match accuracy | Right line(s)? Right "which video?" question? Right "not said"? | `finsight eval --matcher` (no agents, free) or the full eval |
| Alignment accuracy | Is each fact in the right bucket? Only facts on lines the system chose are compared. | same |
| Fact verdict accuracy | For each labeled fact the user asked about, is the verdict right? | full eval only: `finsight eval --system finsight` |

A matching error is never counted as a verdict error. `gold_sub_claims` is derived from `gold_alignment`, so the existing Accuracy / Macro-F1 CSV still works. `gold_overall` (the headline) is the overall verdict of the asked facts.

### Labeling rules

1. **Label from the catalog and the filings, never from FinSight's output.** Write the labels before running the matcher on an item.
2. **`quote_span` is the influencer's exact words.** `build_dev_gold.py` fails if a span is not in the catalog line.
3. **Match status:**
   * MATCHED: the line(s) that state what the user means. Put equally good lines in `acceptable_quote_ids`.
   * ASK: the same statement in two or more eligible videos.
   * NONE: never said by this influencer.
4. **Buckets follow the table above.** A different figure on the same measure is `conflict`, not `user_only`. The influencer's own number is what gets checked.
5. **Verdicts are per fact,** from the filings on or before the matched video's date. `user_only` facts never get one.
6. **Fixing a label after a run is allowed only when the catalog or a filing shows the label was wrong.** Write why in `notes` (see dev-085, dev-086).

## Current results (30 Oct 2026, rules matcher, mock mode)

| Set | Items | Match | Alignment | Notes |
|---|---|---|---|---|
| Dev (48 items, 40 with a match label) | 40 | **100%** | 98% (61/62) | Tuned on this set, so optimistic. The miss is dev-034, a negation ("losing subscribers is a misconception"). |
| Blind probe (`docs/drafts/match_probe.jsonl`) | 12 | **50%** | 80% | Lines sampled with a fixed seed, labeled before any run, never tuned on. This is the honest number for the rules matcher. |

The probe failures show where rules are weak:

* **Over-asking.** It asks "which video?" when one video is the clear fit (probe-05, 06, 09).
* **Arbitrary conflict pick.** It pairs a wrong figure with an arbitrary line in another video (probe-03).
* **Missed repeat.** It misses a repeat said in different words (probe-01).

These are the cases the LLM pick is for. To measure it (costs a little):

```bash
finsight mode paid                      # gpt-5.4-nano
FINSIGHT_MATCHER=llm finsight eval --matcher
FINSIGHT_MATCHER=llm finsight eval --matcher --dataset docs/drafts/match_probe.jsonl
```

## Verdict accuracy with the SEC data

The full comparison with the baselines, the current numbers, and the decisions that came out of the disagreements are in [evaluation.md](evaluation.md). On dev in mock mode, FinSight gets **83% of 36 labeled facts** right.

The verdicts were labelled on 30 Oct from `data/sec/processed`, computed by hand from `facts.csv` (not by FinSight), with filings on or before each video, under section 11. They are in `SEC_LABELS` in `scripts/build_dev_gold.py`. `REVIEW:` marks a borderline call for the second labeler.

## What is still open

* **Verdicts.** 43 of the 62 checkable dev facts are labeled: 28 carried from v1 and 15 new (`data/sec` and section 11 rules). The other 19 are segment, run-rate, customer, cost or superlative figures, marked `UNLABELED (narrative)`; label them from Stella's corpus. A second labeler should check every `REVIEW:` note.
* **Held-out set.** `claims_heldout.jsonl` is still the 3 placeholder items. Build it like the probe: sample lines, write prompts and labels blind, freeze, run once.
* **dev-091 (pre-existing).** "Mark Roussin said Apple's stock is up nearly 25% this year": the routing code asks a clarification, but the team's label says refuse. Decide which is right.
* **Negation and other semantic cases** need the LLM pick. The rules only see words and figures.
