# Evaluation dataset guide

Files: `finsight/eval/datasets/claims_dev.jsonl` (tune on this) and `claims_heldout.jsonl` (frozen; final run only).
One JSON object per line:

| Field | Required | Meaning |
|---|---|---|
| `item_id` | yes | Unique across both files, e.g. `dev-013`, `ho-007` |
| `split` | yes | `dev` or `heldout`; must match the file |
| `category` | yes | `normal`, `adversarial_advice`, `out_of_scope`, `injection`, `temporal`, `clarification`, `comparison`, `misremembered`, `missing_name_answered` or `missing_name_refused` (see below) |
| `claim` | yes | The pasted text, exactly as a user would write it |
| `cutoff_date` | no | `YYYY-MM-DD`. Omit it to test the default cut-off |
| `expect_refusal` | yes | `true` for advice requests and out-of-scope claims |
| `expect_clarification` | no | `true` when the system should ask a question instead: no company named, no checkable sub-claim, or an empty claim |
| `gold_sub_claims` | if not refused | List of `{text, gold_verdict, sub_claim_type, evidence, influencer, rule_note}`. `gold_verdict` may be left out while the label is still open (it shows as `UNLABELED` in `sub_claims.csv`). `influencer` says whose post the sub-claim comes from; `rule_note` says which verdict rule decided the label |
| `notes` | no | Why the item exists or what makes it tricky |
| `cutoff_resolved_to` | no | What a relative cut-off in the claim ("as of last quarter") resolves to |
| `influencer` | no | The influencer the claim is attributed to |
| `influencer_after_clarification` | no | The influencer once the user has answered the clarification question |
| `clarification_prompt` | no | The question the system should ask |
| `clarification_reply` | no | The user's answer to that question |
| `expected_outcome` | no | What should happen, e.g. `PROCESS_AFTER_CLARIFICATION` |
| `posts` | no | The influencer posts the claim is based on |
| `user_matches_post` | no | `false` when the user's wording differs from what the post said |
| `mismatch_note` | no | How the user's wording differs from the post |
| `gold_source` | no | Where the gold labels came from |

The fields from `cutoff_resolved_to` down are stored and validated but not yet used by the runner.

### Categories added for influencer-post items

- `comparison`: the claim compares the company with peers or with another influencer's view.
- `misremembered`: the user's version of the post is wrong (`user_matches_post: false`, explained in `mismatch_note`).
- `missing_name_answered`: no company or influencer is named, the system asks, and the user answers.
- `missing_name_refused`: no name is given and the user doesn't provide one when asked.

Refusal and clarification items normally have no `gold_sub_claims`. The exception is an item with
`expected_outcome: PROCESS_AFTER_CLARIFICATION`: it is analyzed after the user's reply, so it carries
gold sub-claims for that second run.

## Labeling rules

1. **Label from the corpus, not from FinSight.** Read the Tier 1–2 documents available on or before
   `cutoff_date` and decide each gold verdict *before* running any system. Never copy a system's output
   into the gold labels.
2. **List the evidence.** Put the evidence IDs (or filing + section) that justify each label in
   `evidence`, so a second labeler can check it.
3. **Apply the project's verdict rules**, the same ones FinSight enforces:
   - Tier 3 news alone can never support a sub-claim, and the claim is never evidence.
   - `COMPARATIVE` sub-claims ("biggest winner", "better than peers") → `INSUFFICIENT_EVIDENCE` (peers are out of scope).
   - `DEMAND_ATTRIBUTION` and `CONTINUATION` sub-claims can be at most `PARTIALLY_SUPPORTED`.
   - Evidence published after `cutoff_date` does not exist for that item. Check **filing dates**, not period-end dates.
4. **Atomic sub-claims.** One checkable statement per gold sub-claim. Include sub-claims the claim *implies*:
   "revenue growth proves X" implies "revenue grew".
5. **Two labelers per item.** Two people label independently, then compare. Resolve disagreements
   together and note anything non-obvious in `notes`. Report the agreement rate in the report.

## Coverage targets

- Every category in both splits, with at least 2 items each in heldout.
- Every sub-claim type in at least one dev item and one heldout item.
- At least one item per corpus company.
- Use different claims in dev and heldout; the validator rejects exact duplicates.

## Workflow

```bash
finsight eval --validate                       # after every edit
finsight eval --system all --split dev         # iterate on dev only
finsight eval --freeze                         # ONCE, when the heldout labels are final (commit the .lock file)
finsight eval --system all --split heldout --final
```

After freezing, any edit to `claims_heldout.jsonl` makes `--final` refuse to run. That is intentional:
heldout results are only valid if the labels were fixed before anyone saw them.

## Current state (2026-10-02)

- Dev: 14 items over the placeholder "Company X" corpus, covering all six categories (3 clarification
  items) and every sub-claim type except OTHER. These are **pipeline-test items**: their gold labels were written alongside
  the stub corpus, so they are not an independent accuracy measure.
- Heldout: 3 placeholder items, **not frozen**. Rebuild them from the real corpus (Stella / Venes)
  with two labelers, then freeze.
