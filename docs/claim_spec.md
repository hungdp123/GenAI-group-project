# FinSight dev claims: draft summary

Draft of the dev evaluation set in `claims_dev_drafts.jsonl` (25 items). Built on 15 Oct 2026. Not yet in the repo dataset.

## 1. What this is

A set of test claims, each written the way a user would paste them, with the expected result for each. Every item is built from a real YouTube post by Mark Roussin or Joseph Carlson (main channel or After Hours) and checked against real SEC numbers and filing text. It is used to tune FinSight. A separate heldout set, with different claims, is built later and run once at the end.

**What each item records:** what the user types, who the influencer is, what the influencer actually said, the cut-off date the system should use, what the system should do with the request, and the expected verdict for each checkable sub-claim.

**Sources**

- Influencer posts: YouTube scrape of 9 Oct 2026 (window 1 Apr to 30 Sep 2026). Mark Roussin, Joseph Carlson, Joseph Carlson After Hours. Alex Reffett is not included (no scrape yet).
- Numbers: SEC pull of 11 Oct 2026 (`facts.csv`, `filings.csv`), used for growth, margins and cash flow.
- Filing text: per-company narrative folders (10-K, 10-Q, earnings releases). Used for Amazon, Meta, AMD, Apple, Salesforce, Starbucks and Oracle.
- Rules: Irine's rule book, `docs/knowledge_layer.md` (branch `irine`), plus the team decisions in section 5.

## 2. The 25 items

Companies used: Google/Alphabet, Amazon, Meta, Microsoft, NVIDIA, AMD, Apple, Starbucks, Salesforce.

| # | Category | What the user types | What the influencer actually said | Expected result |
| --- | --- | --- | --- | --- |
| 01 | normal | Joseph Carlson said Google's revenue grew 24% last quarter, its twelfth straight quarter of double-digit growth. | Joseph Carlson (After Hours), 2026-07-24: “Google grew by 24% year-over-year, marking its twelfth consecutive quarter of double-digit gro…” | Google +24% Supported (actual 24.2%). "Twelfth straight double-digit quarter" is Insufficient Evidence: it needs 12 quarters, outside the corpus window. |
| 02 | comparison | Mark Roussin said Alphabet's revenue grew 20% last quarter, but Joseph Carlson said 22%. Which one is right? Cut-off: 30 Apr 2026 | Mark Roussin, 2026-04-30: “The revenue growth of 20% for the company was the highest rate we have seen since 2022”<br>Joseph Carlson (After Hours), 2026-04-30: “Google grew by 21%. In fact, 22% this most recent quarter.” | Actual growth 21.8%. Carlson's 22% Supported. Roussin's 20% Contradicted (1.8 pp off, outside the 0.5 pp band). Both shown side by side. |
| 03 | missing_name_answered | Amazon's AWS grew 28% in Q1 2026. | Mark Roussin, 2026-04-30: “AWS was a key watch point, and it grew 28% year over year” | System asks who said it. User answers "Mark Roussin". Claim is then processed: AWS +28% Supported (10-Q). |
| 04 | missing_name_refused | Everyone keeps saying Meta is raising its capital spending this year. | None (no name given) | System asks who said it. User dodges ("doesn't matter, just check it"). Request rejected and logged as incomplete. Nothing is analysed. |
| 05 | misremembered | Mark Roussin said Microsoft's revenue grew 40% in the March quarter. | Mark Roussin, 2026-04-30: “Revenues of 82.9 billion, which was an 18.3% year-over-year growth ... Azure Cloud ... we saw…” | System says the post has revenue +18.3% and Azure +40% and asks to confirm. User says yes. Revenue +18.3% Supported. Azure +40% pending (Microsoft filing text missing). |
| 06 | adversarial_advice | Joseph Carlson says AWS grew 28%. Should I buy Amazon? | Joseph Carlson (After Hours), 2026-04-30: “When we look specifically at AWS, and we eliminate all the other revenue lines, it grew by 28%.” | Whole message refused because it asks for investment advice. The AWS claim is not checked. User is told to resend the claim on its own. |
| 07 | temporal | Joseph Carlson said Google's revenue grew 24% in the June 2026 quarter. Cut-off: 1 Apr 2026 | Joseph Carlson (After Hours), 2026-07-24: “Google grew by 24% year-over-year” | Insufficient Evidence. The Q2 10-Q was filed 23 Jul, after the 1 Apr cut-off, so it does not exist for this item. |
| 08 | out_of_scope | Joseph Carlson says Meta's revenue grew 33% in Q1 2026 and that the stock could be worth $2,000 in five years. | Joseph Carlson (After Hours), 2026-04-30: “Meta has dropped 10% even though it grew 33%.”<br>Joseph Carlson (After Hours), 2026-04-13: “I believe in 5 years, Meta could be worth $2,000” | Meta +33% Supported (33.1%). The $2,000 price is Insufficient Evidence (filings are not used to judge a price). |
| 09 | normal | Joseph Carlson said Amazon's custom chip business has a revenue run rate of over $20 billion. | Joseph Carlson (After Hours), 2026-05-13: “The custom chip business revenue run rate is 20 plus billion dollars” | Partially Supported: only the earnings release states it (Tier 2), not a filing. |
| 10 | normal | Joseph Carlson said Nvidia has very high margins. | Joseph Carlson (main), 2026-04-22: “But, what Nvidia also has is high margins, very, very high margins.” | Insufficient Evidence: "very high" is a level word with no guideline. |
| 11 | cutoff_trap | Joseph Carlson said Alphabet's revenue grew 22% to $110 billion last quarter. Cut-off: 30 Apr 2026 | Joseph Carlson (After Hours), 2026-06-02: “Alphabet's revenue growing 22% year-over-year to $110 in Q1” | Supported: Q1 growth 21.8% matches 22%, and $109.9B rounds to $110B. |
| 12 | cutoff_trap | Joseph Carlson said Alphabet's revenue grew 22% to $110 billion last quarter. | Joseph Carlson (After Hours), 2026-06-02: “Alphabet's revenue growing 22% year-over-year to $110 in Q1” | Contradicted: with no cut-off, "last quarter" is Q2 (+24.2%, $119.8B). |
| 13 | cutoff_trap | Joseph Carlson said Meta's revenue grew 33% last quarter. Cut-off: 29 Apr 2026 | Joseph Carlson (After Hours), 2026-04-30: “Meta has dropped 10% even though it grew 33%.” | Partially Supported: only the earnings release exists before the cut-off (10-Q filed the next day). |
| 14 | cutoff_trap | Mark Roussin said AMD's revenue grew about 40% last quarter. Cut-off: 31 Feb 2026 | Mark Roussin, 2026-05-09: “AMD saw top-line revenue growth of about 40%” | System asks for a valid date. User gives 6 May 2026. Contradicted: AMD grew 37.8%, not about 40%. |
| 15 | normal | Mark Roussin said Amazon's free cash flow has recently turned negative. | Mark Roussin, 2026-04-30: “the company's free cash flow has now dipped into the red or negative as they are now spending…” | Supported: Q2 free cash flow is -$8.8B (operating cash flow minus capex). |
| 16 | normal | Mark Roussin said Meta's Q1 2026 revenue, operating profit and operating margin all went up. | Mark Roussin, 2026-04-30: “investing leading to higher revenues, higher operating profits, and bigger operating margins” | Three sub-claims: revenue Supported, operating profit Supported, operating margin Contradicted (40.6% vs 41.5%). |
| 17 | normal | Mark Roussin said Nvidia's revenue was $7.2 billion in fiscal Q1 2024 and $13.5 billion the next quarter, up 101%. | Mark Roussin, 2026-09-20: “Nvidia reported 13.5 billion of total revenue, up 101%” | Insufficient Evidence on both sub-claims: the periods are outside the corpus window. |
| 18 | normal | Mark Roussin said AMD's latest quarter showed over 100% data center growth. | Mark Roussin, 2026-09-01: “AMD's latest quarter produced over 100% data center growth” | Supported: the 10-Q states data center revenue +107%. |
| 19 | normal | Joseph Carlson said AWS grew 24% year over year in Q4 2025 with a $142 billion revenue run rate. | Joseph Carlson (After Hours), 2026-04-13: “In Q4 of 2025, AWS reported 24% year-over-year growth with a $142 billion revenue run rate.” | AWS +24% Partially Supported (release only; unconfirmed because the 10-K file is empty). $142B run rate Insufficient Evidence. |
| 20 | missing_name_answered | Joseph said Meta's revenue grew 33% in Q1 2026. | Joseph Carlson (After Hours), 2026-04-30: “Meta has dropped 10% even though it grew 33%.” | System asks which Joseph. User answers "Joseph Carlson". Meta +33% Supported. |
| 21 | missing_name_answered | Mark Rousin said Amazon's AWS grew 28% in Q1 2026. | Mark Roussin, 2026-04-30: “AWS was a key watch point, and it grew 28% year over year” | System asks "did you mean Mark Roussin?". User says yes. AWS +28% Supported. |
| 22 | misremembered | Mark Roussin said Alphabet's revenue was $20 billion last quarter. | Mark Roussin, 2026-04-30: “Revenues beat at nearly 110 billion dollars. Google Cloud though was impressive ... and revenu…” | System asks to confirm what he actually said. User says no. Asked once what they meant, still unclear. Rejected as incomplete. |
| 23 | out_of_scope | Mark Roussin said Apple's stock is up nearly 25% this year. | Mark Roussin, 2026-09-27: “Shares of Apple are up nearly 25% on the year” | Stopped as out of scope: the whole request is a share price, with no company fact to check. |
| 24 | normal | Joseph Carlson said Starbucks is using AI to cut its reliance on Microsoft and IBM software. | Joseph Carlson (After Hours), 2026-07-09: “Starbucks is using artificial intelligence to reduce its reliance on Microsoft and IBM softwar…” | Insufficient Evidence: nothing in the stored Starbucks documents states it. |
| 25 | normal | Joseph Carlson said Salesforce plans to unveil a new AI platform, codenamed Agent Albert, by the end of this year. | Joseph Carlson (After Hours), 2026-04-20: “By the end of this year, it plans to unveil a new AI platform that automatically studies its u…” | Insufficient Evidence: no Salesforce document mentions it. |

Categories: `adversarial_advice` (1), `comparison` (1), `cutoff_trap` (4), `misremembered` (2), `missing_name_answered` (3), `missing_name_refused` (1), `normal` (10), `out_of_scope` (2), `temporal` (1).

Gold verdicts across all sub-claims: CONTRADICTED 4, INSUFFICIENT_EVIDENCE 9, not yet set 1, PARTIALLY_SUPPORTED 3, SUPPORTED 12.

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
| `PROCESS` | Checked normally | 01, 02, 07, 08, 09, 10, 11, 12, 13, 15, 16, 17, 18, 19, 24, 25 |
| `PROCESS_AFTER_CLARIFICATION` | System asks one question (name, spelling, date, or confirmation), the user answers, then it is checked | 03, 05, 14, 20, 21 |
| `REFUSE_ADVICE` | Whole message refused because it asks for investment advice | 06 |
| `REFUSE_OUT_OF_SCOPE` | Stopped because the request is only a price, a macro remark or a company outside scope | 23 |
| `REJECT_NO_INFLUENCER` | No influencer name after one question, so rejected and logged as incomplete | 04 |
| `REJECT_INCOMPLETE` | Still unclear after one question, so rejected and logged as incomplete | 22 |

## 4. Fields

Fields marked **new** are not in the current `EvalItem` and need adding (section 6).

| Field | Type | What it holds | Status |
| --- | --- | --- | --- |
| `item_id` | string | Unique id, e.g. `draft-07` | exists |
| `split` | `dev` / `heldout` | Which set the item belongs to. All `dev` here. | exists |
| `category` | string | The kind of test: adversarial_advice, comparison, cutoff_trap, misremembered, missing_name_answered, missing_name_refused, normal, out_of_scope, temporal | exists; 5 new values (`comparison`, `cutoff_trap`, `misremembered`, `missing_name_answered`, `missing_name_refused`) |
| `claim` | string | The text the user types, exactly as written. An invalid date stays here. | exists |
| `cutoff_date` | date or null | The cut-off the user gave (a real date only). Empty when none was given. | exists |
| `cutoff_resolved_to` | object or null | The cut-off actually used: `rule` (`user_given`, `user_given_after_clarification` or `latest_filing_date`), `date`, `basis` | **new** |
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
| `sub_claim_type` | One of Irine's 11 types. Used here: `HISTORICAL_GROWTH` (15), `OTHER` (11), `PROFITABILITY` (2), `CASH_GENERATION` (1). | exists; the enum needs the 11 names |
| `gold_verdict` | One of the four verdicts. Empty only where the answer is still pending (item 05, Azure). | exists |
| `evidence` | Where the label comes from: the calculation from `facts.csv` or the filing passage | exists |
| `influencer` | Which influencer a sub-claim belongs to (comparison items) | **new** |
| `rule_note` | Which rule decides the label, when it is not obvious | **new** |

## 5. Rules these items follow

**Cut-off date**
- The cut-off is the system's review date, not the post date. A post can be dated after the cut-off.
- If the user gives none, it defaults to the filing date of that company's newest 10-K or 10-Q. News articles are not used. A document filed on the cut-off date is included; one filed after is not.
- "Last quarter" means the latest quarter ending on or before the cut-off. A period after the cut-off, or older than the latest 10-K plus four quarters, is Insufficient Evidence.

**Matching numbers**
- A percentage matches within 0.5 percentage points. "About" or rounded wording does not widen this (20% against an actual 21.8% is Contradicted).
- A dollar amount matches after rounding to the precision the user stated.
- Free cash flow is operating cash flow minus capital spending. "Recently" or "quarter" means the latest quarter against the same quarter a year earlier.

**Influencer name**
- A missing name gets one question. If there is still no name, the request is rejected and logged as incomplete.
- A first name only or a misspelling also gets one question.
- Names outside Mark Roussin and Joseph Carlson are not used in this file. Joseph Carlson's two channels are one name.

**Misremembered claims**
- The label is based on what the influencer actually said, not on the user's wording.
- The system first tells the user what the post says and asks to confirm. Yes or maybe proceeds. No gets one more question, and if that fails the request is rejected as incomplete.

**Other**
- Advice mixed with a claim: refuse the whole message. A price inside an otherwise checkable post is Insufficient Evidence. A request that is only a price is out of scope.
- A level word with no guideline ("very high margins") is Insufficient Evidence.
- At most 8 sub-claims per request. Earnings-release text with no matching filing can support a claim only partially. News never supports a claim.

## 7. Known gaps

- **Empty narrative files.** These are 0 KB in the narrative folders: Amazon FY2025 10-K and Q2 2026 10-Q; Apple FY2025 10-K and the Q2 and Q3 FY26 10-Qs; Starbucks FY2024 and FY2025 10-Ks. Items 09, 19 and 24 carry a caution until they are re-saved. SEC numbers are not affected.
- **Missing narrative zips.** Google, NVIDIA, Netflix and Microsoft arrived empty. Item 05's Azure sub-claim waits on Microsoft.
- **Alex Reffett:** no scrape yet, so no items.
- **Heldout set:** not built. It needs different claims, at least 2 items per category, and two labelers per item.
- **Labeling:** gold labels here are from one labeler and have not been double-checked.
