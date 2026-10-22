# Claim spec

Short reference. Full rules live in `docs/knowledge_layer.md` (Irine's branch). This file only summarises, flags differences, and lists open decisions.

## 1. Terms

- **Claim**: the user's whole input. **Sub-claim**: one checkable point inside it (as in `schemas.py` v0.4).
- The pasted claim and the influencer post are the material under test, never evidence.

## 2. Taxonomy (details: KL §4)

| Group | Types |
|---|---|
| Financial (calculated, Tier 1 only) | HISTORICAL_GROWTH, NET_INCOME_GROWTH, PROFITABILITY, CASH_GENERATION, LIQUIDITY, LEVERAGE |
| Narrative | DEMAND_ATTRIBUTION, CONTINUATION, BUSINESS_RISK, OTHER (not in table) |
| No lookup | COMPARATIVE with no named measure (always Insufficient Evidence). A named measure across two companies in the table is checked once per company. |

Advice is refused at screening (KL §2) and never becomes a sub-claim.

## 3. Decomposition format (KL §3, §5, §6, §7)

- Keep the user's words; add nothing from model knowledge.
- Only KL §5 expansions allowed (e.g. "financially strong" becomes 4 children). Expanded children store the parent phrase and rule name.
- Max 8 sub-claims; drop order in KL §3; dropped quotes are recorded.
- Pure opinion is left out and listed as "nothing to check". Price or ranking inside a post is kept and labelled Insufficient Evidence.
- Every sub-claim uses the request cut-off (default: newest document date, stated in the report).

## 4. Schema (short)

Per sub-claim: `claim_id, text, source_span, parent_span, expansion_rule, type, cut_off, collector` (KL §7).
Per dev item: see `docs/eval_dataset_guide.md`, plus the fields below.

New dev-item fields for the influencer module (all null until posts are scraped):
`influencer`, `post_text`, `post_date`, `user_matches_post`, `influencer_missing_behavior`, `gold_source` (`user_sentence` or `post`).

## 5. Verification labels (KL §11, §13)

Supported, Partially Supported, Contradicted, Insufficient Evidence. Caps per type are in KL §4 and §11. Overall label rules are in KL §13.

## 6. Differences found

| Item | KL doc | Code / other |
|---|---|---|
| Types | 11, kept separate in code | `SubClaimType` has NET_INCOME_GROWTH, PROFITABILITY, CASH_GENERATION, LIQUIDITY, and LEVERAGE. Older rows may still say PROFITABILITY_CASH_LEVERAGE and are split on load. |
| Sub-claim fields | KL §7 list | `SubClaim` has `sub_claim_id, text, sub_claim_type, company_ticker, is_quantitative` |
| Influencer module | Not mentioned | Not in schema or dataset guide |

## 7. Open decisions

1. **Which type enum is final?** Settled 19 Oct 2026. The separate types are final: historical growth, net income growth, profitability (including gross margin), cash generation, liquidity, and leverage. A debt amount is not leverage. It is quoted by the narrative agent. Leverage stays with the financial agent, which returns no ratio yet.
2. **Why 8 for the cap?** KL gives no rationale. Ask Irine.
3. **If the user's sentence differs from the real post, which is checked?** Settled 16 Oct 2026. A stored row is checked when it shares the measure and any stated figure, including a paraphrase. "Grew 24%" does not attach to a row that says 17%. If no stored row is that close, the transcript is searched. One close spoken sentence is shown and confirmed before it is checked. Several are shown and the user picks one. If nothing is close, the reply says the line is not in the stored videos and stops.
4. **Claim with no influencer named: valid input or Clarification?** Only 3 influencers are chosen, so they may not have covered every company. Undecided.
5. **Cut-off date: user-typed, defaulted, or tied to the post date?** Likely user-typed; how it relates to the post date is undecided.
6. **Price claim inside a post (draft-07): what type?** KL §6 gives the label (Insufficient Evidence) but §4 requires every sub-claim to have a type and none fits a price. OTHER would trigger a document search that §6 says should not happen. Options: add a PRICE type that is never searched, or keep OTHER with a rule that price phrases skip the search. Owner: Irine.
7. **Turnaround phrases (drafts 11, 12): does every sheet measure become its own sub-claim, given the cap of 8?** KL §5 implies yes; confirm. Example: gross margin and inventory would add sub-claims and could pass 8.
8. **For Hung: does the schema need fields for the influencer post?** The influencer module finds a post (Collect posts) and the Orchestrator and Auditor must then work from that post's text and date. `ClaimRequest` in `schemas.py` only has `raw_claim`, `company_ticker` and `cutoff_date`, and the schema rejects unknown fields, so there is nowhere to carry an influencer name, post text or post date. Options: (a) add optional fields to `ClaimRequest` (e.g. `influencer`, `post_text`, `post_date`) and bump the schema version, or (b) put the post text into `raw_claim` and add nothing. The dev items already carry these fields in the test file only (section 4). This depends on decisions 3 to 5 above, so settle those with the group first.
