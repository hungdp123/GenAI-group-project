# Evaluating FinSight against an LLM and RAG

FinSight's claim is that it is more reliable than asking an LLM, because verdicts come from the filings by calculation and citation. This page says how that is measured.

## The five systems

All five answer the same items, judged as of the same date: the user's cut-off, or else the date of the influencer's video.

| System (`--system`) | Gets | What the comparison shows |
|---|---|---|
| `llm`: Plain LLM | The user's message only | The "just ask ChatGPT" baseline. Expect confident answers and made-up figures. |
| `direct`: LLM + filings | The message + the company's XBRL facts from `data/sec/processed` (latest 6 quarters and 3 years) | Whether handing an LLM the right numbers is enough. |
| `rag`: Basic RAG | The message + the top-5 passages from the narrative corpus | Documents without calculation, cut-off control or matching to the influencer's line. |
| `rag_quote`: RAG + quote | Basic RAG + **the influencer's actual line** (from the gold match) | Isolates FinSight's verification. Matching errors are removed. |
| `finsight` | The full pipeline | |

All baselines get the same refusal, scope **and period** rules in their prompt ("last quarter" = the latest quarter before the as-of date; no period = the latest fiscal year; can't verify = Insufficient Evidence, never ask). The comparison then measures the architecture, not who was told the rules. Without the period rules, the plain LLM asked "which quarter?" on most items (15% correct behaviour on the first paid run, 30 Oct).

## The scores

`comparison.md` in each run folder has the scores. `summary.md` adds behaviour, cost and latency.

| Score | Question | Why it matters |
|---|---|---|
| Headline verdict accuracy | Is the overall verdict right? | It is what the user reads. Gold = the facts the user asked about. |
| Fact verdict accuracy, Macro-F1 | Is each fact's verdict right? | Finer-grained. A missing or refused answer counts as wrong. |
| **Overclaim rate** | Of facts whose gold is "can't confirm", how many got a confident verdict? | **The hallucination measure.** Example: "twelfth straight quarter: Supported" with no data for 12 quarters. |
| **Made-up number rate** | Figures in the answer that are neither in nor calculable from what the system was given | An invented figure. A percent change or ratio of given figures is not counted as made up. |
| After-cut-off dates / evidence | Does the answer use anything published after the as-of date? | Look-ahead bias. |
| Refusal / clarification behaviour | Advice, price talk, missing names | Safety. |
| Cost and latency | Per query | The price of reliability. Report it. |
| By route | Financial vs narrative facts (gold route) | Shows where FinSight's advantage comes from (calculation vs retrieval and verification). |
| Paired test | Exact McNemar on the facts both systems answered | With ~40-100 facts, small gaps are noise. Report counts and the p-value. |

FinSight also has its own matching scores: right line, and facts in the right bucket. See [claim_matching.md](claim_matching.md).

## Running it

```bash
finsight mode paid                                   # gpt-5.4-nano first: cheap, checks the run
finsight eval --system all --split dev               # all five systems, then comparison.md
finsight usage                                       # what it cost
FINSIGHT_LLM_MODEL=gpt-5.4 finsight eval --system all --split dev       # the course model
finsight eval --compare results/<run>                # rebuild comparison.md for a finished run
```

### Financial facts only (no narrative corpus yet)

```bash
finsight eval --system finsight,llm,direct --split dev --route financial
```

* **How it works.** Every gold fact carries a `route` (financial / narrative). It is set when the dev file is built, by the same rule-based router applied to the gold quote, and a labeler can override it with `fact(..., route=...)`.
  * All systems run only on items with at least one labeled fact routed to the financial agent (dev: 20 items, 26 facts; listed in `items_financial.txt` in the run folder).
  * `comparison.md` scores only those facts.
  * An item's headline verdict counts only when every fact the user asked about is financial.
  * The set comes from the gold labels, never from a system's run. A fact FinSight mis-routes, or an item it fails to match, still counts against it.
* **Which systems to include.** Leave out `rag` and `rag_quote` until the narrative corpus is on the machine: without passages they behave like the plain LLM.
* **Narrative later.** The same works with `--route narrative` once the corpus is there.
* **Rescoring a finished run:** `finsight eval --compare results/<run> --route financial`.

The final reported numbers come from the frozen held-out split, once.

**The held-out set** (`claims_heldout.jsonl`, 20 items, built by `scripts/build_heldout_gold.py`): every line comes from `claim_update_v3.jsonl`, no video in it is in the dev set, and FinSight is never adjusted on it. 10 items go to the financial agent (labelled from the XBRL facts), 10 to the narrative agent (labelled from the company's own release; check each source is in the narrative corpus). It includes overclaim tests (a figure from before the stored filings, analyst estimates), a cut-off case, a paraphrase, a misremembered figure and a detail the user added.

1. A second person checks the labels, especially every `REVIEW:` note (`notes` / `rule_note`).
2. Lock it: `finsight eval --freeze` (writes `claims_heldout.lock`; the run refuses a changed file).
3. Run it once, with the course model (`--final` requires it):

```bash
FINSIGHT_LLM_MODEL=gpt-5.4 finsight eval --system finsight,llm,direct --split heldout --final
finsight eval --compare results/<run> --split heldout --route financial   # financial facts only
finsight eval --compare results/<run> --split heldout --route narrative   # narrative facts only
```

Add `rag,rag_quote` to `--system` when the narrative corpus is on the machine. If a held-out result shows a bug, fix it, but report the held-out numbers from before the fix (or build a new held-out set).

What each system needs on the machine:

| System | Needs |
|---|---|
| `direct` | `data/sec/processed` |
| `rag`, `rag_quote` | Stella's narrative corpus (`data/narrative/...`). Without it they get no passages. |
| FinSight | `data/influencers/claims/claim_update_v3.jsonl` |

**Mock mode only tests the harness.** The baselines' canned answers call every sentence "Supported".

## Current FinSight-only results (dev, mock LLM, SEC data, no narrative corpus)

| | |
|---|---|
| Match accuracy | 100% (40 items; tuned here). The blind probe is 50%. |
| Fact verdict accuracy | **83%** (30 of 36 labeled facts) |
| Headline verdict accuracy | 81% (26 items with a full headline label) |
| Overclaim rate | 8% (1 of 13 can't-confirm facts) |

Baseline numbers need a paid run.

## Decisions for the team (30 Oct 2026)

These came out of reading every disagreement between FinSight and the gold labels. **Implemented** means the code and the dev labels already follow the recommendation. Undo it if the team decides otherwise.

### 1. Which period, when the line names none (dev-014, dev-030, dev-051, dev-061)

* **Problem.** The rule book says "no period named = latest fiscal year". Influencers talking right after earnings almost always mean the latest quarter. Strictly, "Meta's revenue is up 33%" on 30 Apr is then Contradicted (FY2025 +22.2%), though Q1 2026 was +33.1%. That is unfair to the influencer and looks wrong to a user.
* **Recommendation.** Keep the fiscal year as the default, but check both the latest quarter and the latest fiscal year when no period is named. If the figure matches either one, use it and say which period matched. A human fact-checker does the same, and it is reproducible.
  * The change lives in the financial agent's period handling (`claim_reader` / `agent.py`, Venes).
  * The dev labels for dev-014 and dev-061 already follow it. They are marked REVIEW.
* **Implemented meanwhile:**
  * The period-cue window is 3 sentences instead of 2 (`clues.CONTEXT_SENTENCES`). This fixed dev-030 and dev-051, where "this quarter" was three lines back. It was tuned on dev, so check it on the held-out set.
  * "Latest earnings call" / "reported earnings" now read as the latest quarter (`claim_reader._LATEST_QUARTER`, knowledge layer section 11).

### 2. Dollar amounts: two auditor bugs, and one rule (dev-006, dev-031)

* **Bugs found and fixed.** The auditor only compared dollar amounts written with a "$", and only one per sentence. So "went from 18 billion dollars to negative 7.6 billion" and "nearly 10 billion" were never checked, and both came back Supported on direction alone. It now:
  * reads amounts without a "$";
  * checks "from X to Y" as earlier and later figures;
  * applies the rounding rule to each.
* **Rule added (implemented, team to confirm).** "Nearly / almost / close to X" also matches up to 10% below X. Netflix's FY2025 free cash flow is $9.46B, and "nearly 10 billion" is a fair description. dev-031 is now labeled Supported. Under plain rounding it would be Contradicted.
* **dev-006 (Amazon free cash flow).** The influencer's figures match Amazon's own release (free cash flow net of proceeds), not the rule book's XBRL formula (−$11.6B vs +$13.5B). The team's v1 label is Partially Supported; FinSight now says Contradicted.
  * **Recommendation:** keep the rule book's formula as the definition. Add one sentence to section 12: "If the figures match the company's own published non-GAAP figure but not the section 11 formula, the label is Partially Supported, and the report names both definitions."
  * The system can only do this once the earnings releases are in the narrative corpus. Until then, expect Contradicted.

### 3. Superlatives and estimate beats are separate facts (dev-005, dev-003)

* **Implemented.** "Their best quarter ever by far" and "which beat expectations" are now their own facts.
* **Why.** Before, they sat in the same clause as "$81.6 billion" or "16%". They pulled a checkable figure down to Partially Supported, or made the classifier type the whole Netflix line OTHER, which sent it to the narrative agent.
* **Labels.** The superlative is "also said" (narrative, unlabeled). The estimate beat is Insufficient (estimates are not stored).
* **Also implemented.** A stored line typed OTHER is re-classified, so a calculable figure goes to the financial agent.

### 4. Relative periods after a historical one (dev-050)

* **Problem.** "The next quarter arrived, Nvidia reported 13.5 billion… up 101%" follows "early 2023". The period reader takes "next quarter" as the latest quarter.
* **Recommendation.** If the line or its context names a period before the corpus window, read "the next / that / following quarter" as outside the window, so the answer is Insufficient Evidence.
* **Status.** Not implemented: one item, and it is in the financial agent's period reader (Venes). Keep dev-050 as the regression test.

### 5. Narrative facts (dev-086 and 19 unlabeled facts)

* **Recommendation.** Run with Stella's corpus. Label the 19 `UNLABELED (narrative)` facts from it, with passage IDs, so retrieval can be scored too.
* **Expectation.** Until then, narrative facts can only come back "can't confirm".

### 6. dev-091 (share-price claim): unchanged

The code asks a clarification question; the team's label says refuse. **Recommendation: refuse.** The line is only about the stock price, and there is nothing to redirect to. Change `routing.py` and keep the label.

## Experiments (dev split only)

One command runs every experiment on the dev claims and writes `results/experiments/experiments.md`:

```powershell
finsight mode paid
python scripts/run_experiments.py                   # all six runs, a few dollars in total
python scripts/run_experiments.py --only matcher_rules,matcher_llm
python scripts/run_experiments.py --reuse main=results\<a finished five-system dev run>
python scripts/run_experiments.py --report-only     # rebuild the report from finished runs
```

| Experiment | What changes | Question |
|---|---|---|
| `main` | all five systems, gpt-5.4 | Does the architecture help? RAG vs RAG + quote. Safety table. |
| `nano` | FinSight, plain LLM, LLM + filings on gpt-5.4-nano | Does FinSight's lead depend on model strength? |
| `retries_off` | `FINSIGHT_MAX_RETRIES=0` | Does the auditor's retry help, or only cost more? |
| `no_period_rules` | `FINSIGHT_BASELINE_PERIOD_RULES=0` (the baselines' old prompt) | How sensitive are the baselines to instructions? |
| `matcher_rules`, `matcher_llm` | `FINSIGHT_MATCHER` | Rule-based vs LLM pick of the influencer's line: accuracy and cost. |
| `agents` | `finsight eval --agents all` | Each agent on its own, with its own small test set. |

**Steps inside FinSight.** Every run also scores FinSight's steps separately, so an error can be traced to where
it starts: claim matching (right line), fact alignment (right bucket), routing (each labelled fact sent to the
agent its gold route names; new 31 Oct, it would have caught the segment bug), the verdict per fact, and the
headline verdict. The report's "Steps inside FinSight" table collects them.

The safety table (refusals, prompt injection, clarifications on dev-060 to dev-097) is also added to every
`comparison.md`. Stella's chunking x search-method study (C1-C4 x BM25 / dense / hybrid / R4) is reported separately.
