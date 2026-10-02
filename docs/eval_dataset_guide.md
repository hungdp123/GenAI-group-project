# Evaluation dataset guide

Files: `finsight/eval/datasets/claims_dev.jsonl` (tune on this) and `claims_heldout.jsonl` (frozen; final run only).
One JSON object per line:

| Field | Required | Meaning |
|---|---|---|
| `item_id` | yes | Unique across both files, e.g. `dev-013`, `ho-007` |
| `split` | yes | `dev` or `heldout`; must match the file |
| `category` | yes | `normal`, `adversarial_advice`, `out_of_scope`, `injection`, `temporal` or `clarification` |
| `thesis` | yes | The pasted text, exactly as a user would write it |
| `cutoff_date` | no | `YYYY-MM-DD`. Omit it to test the default cut-off |
| `expect_refusal` | yes | `true` for advice requests and out-of-scope theses |
| `expect_clarification` | no | `true` when the system should ask a question instead: no company named, no checkable claim, or an empty thesis |
| `gold_claims` | if not refused | List of `{text, gold_verdict, claim_type, evidence}` |
| `notes` | no | Why the item exists or what makes it tricky |

## Labeling rules

1. **Label from the corpus, not from FinSight.** Read the Tier 1–2 documents available on or before
   `cutoff_date` and decide each gold verdict *before* running any system. Never copy a system's output
   into the gold labels.
2. **List the evidence.** Put the evidence IDs (or filing + section) that justify each label in
   `evidence`, so a second labeler can check it.
3. **Apply the project's verdict rules**, the same ones FinSight enforces:
   - Tier 3 news alone can never support a claim, and the thesis is never evidence.
   - `COMPARATIVE` claims ("biggest winner", "better than peers") → `INSUFFICIENT_EVIDENCE` (peers are out of scope).
   - `DEMAND_ATTRIBUTION` and `CONTINUATION` claims can be at most `PARTIALLY_SUPPORTED`.
   - Evidence published after `cutoff_date` does not exist for that item. Check **filing dates**, not period-end dates.
4. **Atomic claims.** One checkable statement per gold claim. Include claims the thesis *implies*:
   "revenue growth proves X" implies "revenue grew".
5. **Two labelers per item.** Two people label independently, then compare. Resolve disagreements
   together and note anything non-obvious in `notes`. Report the agreement rate in the report.

## Coverage targets

- Every category in both splits, with at least 2 items each in heldout.
- Every claim type in at least one dev item and one heldout item.
- At least one item per corpus company.
- Use different theses in dev and heldout; the validator rejects exact duplicates.

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
  items) and every claim type except OTHER. These are **pipeline-test items**: their gold labels were written alongside
  the stub corpus, so they are not an independent accuracy measure.
- Heldout: 3 placeholder items, **not frozen**. Rebuild them from the real corpus (Stella / Venes)
  with two labelers, then freeze.
