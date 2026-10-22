# FinSight dev claims: draft summary

Draft of the dev evaluation set in `claims_dev_drafts.jsonl` (56 items). Built on 15 Oct 2026; items 09, 20 and 21 changed on 16 Oct 2026 to add Palantir, Netflix and Mercado Libre; items 26 to 33 added on 17 Oct 2026, items 34 to 53 on 17 to 18 Oct 2026 (Alex Reffett scrape), items 54 to 56 on 18 Oct 2026 (trailing-12-month ruling and the Meet Kevin finds). Not yet in the repo dataset.

## 1. What this is

A set of test claims, each written the way a user would paste them, with the expected result for each. Every item is built from a real YouTube post by Mark Roussin, Joseph Carlson (main channel or After Hours) or Alex Reffett and checked against real SEC numbers and filing text. It is used to tune FinSight. A separate heldout set, with different claims, is built later and run once at the end.

**What each item records:** what the user types, who the influencer is, what the influencer actually said, the cut-off date the system should use, what the system should do with the request, and the expected verdict for each checkable sub-claim.

**Sources**

- Influencer posts: YouTube scrape of 9 Oct 2026 (window 1 Apr to 30 Sep 2026). Mark Roussin, Joseph Carlson, Joseph Carlson After Hours. Alex Reffett: scrape of 15 Oct 2026 (channel @TickerSymbolYOU, 29 videos with transcripts, same window; the YouTube dates for older videos are month-level, so the real upload date is used).
- Numbers: SEC pull of 11 Oct 2026 (`facts.csv`, `filings.csv`), used for growth, margins and cash flow.
- Filing text: per-company narrative folders (10-K, 10-Q, earnings releases). Used for Amazon, Meta, AMD, Apple, Salesforce, Starbucks, Oracle, Palantir and Mercado Libre (Mercado Libre: earnings releases only; see section 7). Netflix items use the numbers only.
- Rules: Irine's rule book, `docs/knowledge_layer.md` (branch `irine`), plus the team decisions in section 5.

## 2. The items

The full item table (user wording, what the influencer said, expected result for each of the 56 items) is in the shared OneDrive folder as `FinSight_Dev_Claims.docx`. It is generated from `claims_dev_drafts.jsonl`, which is the source of truth; regenerate the document when items change.

Companies used: Google/Alphabet, Amazon, Meta, Microsoft, NVIDIA, AMD, Apple, Starbucks, Salesforce, Mercado Libre, Palantir, Netflix, Tesla.

Categories: `adversarial_advice` (1), `ask_which_post` (1), `comparison` (4), `cutoff_trap` (10), `injection` (1), `misremembered` (3), `missing_name_answered` (3), `missing_name_refused` (1), `no_stored_post` (2), `normal` (23), `out_of_scope` (4), `record` (2), `temporal` (1).

Gold verdicts across all sub-claims: CONTRADICTED 23, INSUFFICIENT_EVIDENCE 15, not yet set 1, PARTIALLY_SUPPORTED 2, SUPPORTED 39.

## 3. Expected results

**Verdict on each sub-claim** (one of four):

| Verdict | Meaning |
| --- | --- |
| `SUPPORTED` | The allowed evidence states the claim, including value and direction. |
| `PARTIALLY_SUPPORTED` | Agrees in part, is forward-looking, or only an earnings release (Tier 2) states it. |
| `CONTRADICTED` | A financial claim disagrees with the filing in value or direction. |
| `INSUFFICIENT_EVIDENCE` | The stored documents cannot decide: not mentioned, outside the window or cut-off, a missing tag, a price, or a ranking against peers. |

**What the system does with the whole request** (`expected_outcome`):

| Outcome | Meaning | Items |
| --- | --- | --- |
| `PROCESS` | Checked normally (including two-influencer comparisons, which are decomposed into one claim per influencer, and record prompts, which check every stored post) | 01, 02, 07, 08, 09, 10, 11, 12, 13, 15, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 50, 51, 54, 55, 56 |
| `PROCESS_AFTER_CLARIFICATION` | System asks one question (name, spelling, date, or confirmation), the user answers, then it is checked | 03, 05, 14, 20, 21, 49 (49 asks which post) |
| `REFUSE_ADVICE` | Whole message refused because it asks for investment advice | 06 |
| `REFUSE_OUT_OF_SCOPE` | Stopped because the request is only a price, an index or macro remark, names two companies with their own claims, or is a company outside scope | 23, 52, 53 |
| `REJECT_NO_INFLUENCER` | No influencer name after one question, so rejected and logged as incomplete | 04 |
| `REJECT_INCOMPLETE` | Still unclear after one question, so rejected and logged as incomplete | 22 |
| `REFUSE_NO_STORED_POST` | The influencer has no stored post about the company, so the request is refused | 33, 44 |

## 4. Fields

Fields marked **new** are not in the current `EvalItem` and need adding (section 6).

| Field | Type | What it holds | Status |
| --- | --- | --- | --- |
| `item_id` | string | Unique id, e.g. `draft-07` | exists |
| `split` | `dev` / `heldout` | Which set the item belongs to. All `dev` here. | exists |
| `category` | string | The kind of test: adversarial_advice, comparison, cutoff_trap, misremembered, missing_name_answered, missing_name_refused, normal, out_of_scope, temporal, injection, no_stored_post, ask_which_post, record | exists; 8 new values (`comparison`, `cutoff_trap`, `misremembered`, `missing_name_answered`, `missing_name_refused`, `no_stored_post`, `ask_which_post`, `record`) |
| `claim` | string | The text the user types, exactly as written. An invalid date stays here. | exists |
| `cutoff_date` | date or null | The cut-off the user gave (a real date only). Empty when none was given. | exists |
| `cutoff_resolved_to` | object or null | The cut-off actually used: `rule` (`user_given`, `user_given_after_clarification`, `latest_filing_date` or `post_date_per_post` for records), `date`, `basis` | **new** |
| `influencer` | string or null | The name the user gave. Empty if none. Two names joined with `;` for a comparison. | **new** |
| `clarification_turns` | list | Each question the system asks and the user's reply: `{system, user}`. Empty if no question. Two entries for a second round. | **new** |
| `influencer_after_clarification` | string or null | The name the user gave after being asked | **new** |
| `expect_refusal` | bool | True for advice or out-of-scope refusals | exists |
| `expect_clarification` | bool | True when the system should ask a question first | exists |
| `expected_outcome` | string | What the system does with the request (section 3) | **new** |
| `posts` | list | The real posts behind the claim. Each has `influencer`, `channel` (`main` or `after_hours`), `video_id`, `url`, `upload_date`, `title`, `quote` (exact words from the transcript). | **new** |
| `user_matches_post` | bool or null | False when the user remembers the post wrongly | **new** |
| `mismatch_note` | string or null | What the user got wrong | **new** |
| `gold_sub_claims` | list | One entry per checkable point (below). Empty for refused and rejected items. | exists |
| `gold_source` | string | `post_text`: gold labels come from the real post, not the user's sentence | **new** |
| `notes` | string | Why the item exists, and anything to watch for | exists |

**Inside each `gold_sub_claims` entry**

| Field | What it holds | Status |
| --- | --- | --- |
| `text` | The claim in plain words | exists |
| `sub_claim_type` | One of Irine's 11 types. Used here: `HISTORICAL_GROWTH` (34), `PROFITABILITY` (13), `OTHER` (16), `CASH_GENERATION` (12), `NET_INCOME_GROWTH` (2), `LIQUIDITY` (1), `LEVERAGE` (1), `COMPARATIVE` (1). | exists; the enum needs the 11 names |
| `gold_verdict` | One of the four verdicts. Empty only where the answer is still pending (item 05, Azure). | exists |
| `evidence` | Where the label comes from: the calculation from `facts.csv` or the filing passage | exists |
| `influencer` | Which influencer a sub-claim belongs to (comparison items) | **new** |
| `rule_note` | Which rule decides the label, when it is not obvious | **new** |
| `post_id` | Which post a sub-claim comes from (record items) | **new** |

## 5. Rules these items follow

**Cut-off date**
- The cut-off is the system's review date, not the post date. A post can be dated after the cut-off.
- If the user gives none, it defaults to the filing date of that company's newest 10-K or 10-Q. News articles are not used. A document filed on the cut-off date is included; one filed after is not.
- "Last quarter" means the latest quarter ending on or before the cut-off. A period after the cut-off, or older than the latest 10-K plus four quarters, is Insufficient Evidence.

**Matching numbers**
- A percentage matches within 0.5 percentage points. "About" or rounded wording does not widen this (20% against an actual 21.8% is Contradicted).
- A dollar amount matches after rounding to the precision the user stated.
- **Periods (team ruling, 18 Oct).** No period named: use the latest fiscal year. "Trailing", "last 12 months" or "TTM": use the sum of the last four quarters. If the fiscal year contradicts a claim that named no period, the financial agent also shows the last-12-months figure as context only; the label does not change. Items 20 and 54.
- Free cash flow is operating cash flow minus capital spending. "Recently" or "quarter" means the latest quarter against the same quarter a year earlier.

**Influencer name**
- A missing name gets one question. If there is still no name, the request is rejected and logged as incomplete.
- A first name only or a misspelling also gets one question.
- Mark Roussin, Joseph Carlson and Alex Reffett are the in-scope names. Items 55 and 56 use Kevin Paffrath (Meet Kevin), who is not in the 13 Oct rule book influencer table, so they are out of scope unless the team adds him. Joseph Hogue is not allowed. Joseph Carlson's two channels are one name.
- A prompt that names two influencers is decomposed into two claims, one per influencer, and is processed only when both names are in scope (team answer, 17 Oct). This differs from section 2 of the 13 Oct rule book, which stops such a prompt; the rule book text needs updating. Items 02, 46, 47 and 48 follow the team answer.

**Misremembered claims**
- The label is based on what the influencer actually said, not on the user's wording.
- The system first tells the user what the post says and asks to confirm. Yes or maybe proceeds. No gets one more question, and if that fails the request is rejected as incomplete.

**Segments**
- A segment figure (AWS, Azure, AMD data center) is checked by the narrative agent only. The financial agent has no segment data and handles only company-wide figures (confirmed by Venes, 17 Oct). The label comes from filing text. Items 03, 05, 18 and 19 involve a segment.

**Record prompts**
- A prompt asking whether an influencer has been consistent or accurate about a company over time does not ask which post. Every stored post with text is checked and each post's own date is its cut-off (a typed date does not replace it). There is no single overall label; the report gives a consistency sentence and an accuracy count. Items 50 and 51.

**Calculations**
- Only the section 11 calculations are used. Growth in operating income, gross profit or EPS is not one of them, so those parts are Insufficient Evidence (items 34 and 41). 

**Other**
- Advice mixed with a claim: refuse the whole message. A price inside an otherwise checkable post is Insufficient Evidence. A request that is only a price is out of scope.
- A level word with no guideline ("very high margins") is Insufficient Evidence.
- At most 8 sub-claims per request. Earnings-release text with no matching filing can support a claim only partially. News never supports a claim.

## 7. Known gaps

- **Empty narrative files.** These are 0 KB in the narrative folders: Amazon FY2025 10-K and Q2 2026 10-Q; Apple FY2025 10-K and the Q2 and Q3 FY26 10-Qs; Starbucks FY2024 and FY2025 10-Ks. Items 19 and 24 carry a caution until they are re-saved. SEC numbers are not affected.
- **Mercado Libre narrative.** In the 2024-2026 folder every 10-Q and the FY2025 10-K are 0 KB; only the earnings releases and the FY2023 and FY2024 10-Ks can be read. Item 21 rests on `facts.csv` and is corroborated by the 2026-08-05 release. The 2021-2023 folder was not sent (outside the corpus window).
- **Missing narrative zips.** Google, NVIDIA, Netflix and Microsoft arrived empty. Item 05's Azure sub-claim waits on Microsoft.
- **Alex Reffett:** 29 of his 46 listed videos have a transcript; the other 17 are not stored, so item 44 (no stored post) could be a missing-transcript case instead. His older videos only have month-level dates; the real upload date is used.
- **Held back.** An AMD data-center comparison (Alex and Roussin, a segment figure; AMD narrative files not checked) is not in the set. Items 42 and 43 (Microsoft) can be dropped as a pair if the team prefers.
- **Operating income, gross profit and EPS growth.** Items 34 and 41 rely on these not being section 11 calculations. If the team adds them, the Insufficient Evidence labels become Supported.
- **Meet Kevin items (55, 56).** Found by a teammate. The net income part rests on facts.csv; the gains figure and both Services margin figures are the teammate's reading and need the Google 10-Q text. Kevin's scrape has 33 transcripts in the window. Decide whether to add him to the influencer table.
- **Earlier error, fixed 18 Oct.** Items 48 and 51 had used two Joseph Hogue videos as Carlson; both were replaced with real Carlson and Roussin posts. All posts in the set are now checked against the scrape's influencer field.
- **Validator.** Items 49 to 51 need new fields or flows (`ask_which_post`, `record`, `post_id`).
- **Heldout set:** not built. It needs different claims, at least 2 items per category, and two labelers per item.
- **Labeling:** gold labels here are from one labeler and have not been double-checked.
