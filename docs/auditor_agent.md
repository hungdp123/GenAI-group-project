# Auditor Agent

Owner: Person 3 (Amelia). Code: `finsight/agents/auditor/agent.py` (`AuditorAgent.audit`). Tests: `tests/test_auditor_agent.py`, `tests/test_auditor_checks.py`. Evaluation: `finsight/eval/agent_eval/auditor.py` (`finsight eval --agents auditor`).

## What it does

For one sub-claim, the Auditor takes the evidence the Financial and Narrative agents collected and answers two questions:

1. **Is the evidence usable and correct?** Numbers are checked by calculation, passages by citation.
2. **If so, what verdict does it support?** SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED or INSUFFICIENT_EVIDENCE.

It does not apply the source-tier cap, the cut-off filter or the retry loop. The orchestrator does those (`routing.tier_shortfall`, `routing.split_temporal`, `nodes.retry_or_accept`).

```
audit(sub_claim, evidence) -> AuditResult
```

## Output (`AuditResult`)

| Field | Meaning |
| --- | --- |
| `passed=False` | A specialist's evidence is defective. `failure_reasons` says what; `feedback_for_retry` tells the collector how to fix it. The orchestrator re-gathers once, then marks the sub-claim INSUFFICIENT_EVIDENCE. |
| `passed=True` | The evidence is usable. `verdict` is the **proposed** verdict; the orchestrator applies the knowledge-layer cap afterwards. |
| `numeric_check`, `citation_check` | The check that failed, or the first one if all passed. |
| `evidence_stances` | The stance the Auditor judged for each evidence item. (The branch's `agent.py` returns this; the field must exist in `contracts/schemas.py`, which Hung owns.) |

No evidence at all gives `passed=True` with INSUFFICIENT_EVIDENCE.

## How an audit runs

1. **Judge the stance of every item.** The collector's own label is ignored, with one exception: a collector's CHALLENGING is kept when the Auditor can't confirm it (counter-evidence is never thrown away). A collector's SUPPORTING is never trusted.
2. **Drop unusable items.** Items judged NEUTRAL (not about this sub-claim, or can't tell) are not checked or counted. If items came back and none is usable, the audit fails so the collector gets one retry.
3. **Hygiene checks** on every usable item (below).
4. **Number checks** on items with a `computed_value`; **citation checks** on text items.
5. **Propose a verdict** from the stances that remain.

### Hygiene checks

- `url` and `citation_span` are both present.
- `company` matches the sub-claim's company.
- `source_tier` matches `doc_type` (10-K, 10-Q, xbrl = Tier 1; earnings release, IR remarks = Tier 2; news = Tier 3).
- `publication_date` is the filing date, not the period end.

### Number checks (items with `computed_value`)

- **Metric:** `metric_name` is read literally and must be the concept the sub-claim is about (revenue, margins, free cash flow, gross profit, current ratio, and so on). Growth rates take a `_growth_yoy` suffix. A quarter-on-quarter figure is rejected unless the sub-claim asks for it.
- **Scale:** growth rates and margins are fractions (0.14, not 14). A percent sent where a fraction is expected is rejected.
- **Consistency:** `computed_value` must match the percent in the text, and for growth the two amounts in the text must give the same growth.
- **Stated number:** the number in the sub-claim is compared with the computed one (see the next section). A mismatch makes the verdict CONTRADICTED.

### What the sub-claim states

The Auditor reads this from the wording of the sub-claim; the contract has no comparator field.

| The sub-claim says | The Auditor checks |
| --- | --- |
| "grew 24%", "margin is 34%" | Equal within 0.5 percentage points |
| "$110 billion", "$96.2 billion" | Equal after rounding to the precision stated ($0.5bn and $0.05bn) |
| "above 10%", "less than 10% growth", "over $10 billion" | A bound: the computed figure must be on the right side |
| "turned negative", "turned positive" | The sign of the computed figure |
| "doubled" | 1 + growth between 1.9 and 2.1 (team rule); "tripled" 2.85 to 3.15 and "halved" 0.45 to 0.55 are the Auditor's own extension |
| "fell", "rose", "declined" | The current and prior figures in the evidence text (percents, dollar amounts, or a plain pair such as "2.60 vs 2.98"). With no prior figure the item is sent back once. |

### Text checks (items without `computed_value`)

Rule-based in mock mode so tests are deterministic.

- **SUPPORTING:** the passage states the sub-claim: enough of its key words are present (60%), the numbers it states appear in the passage, and the wording is not hedged ("may", "could" count as hedging, except for BUSINESS_RISK).
- **CHALLENGING:** the passage rules out the claimed driver ("rather than AI infrastructure") or says the opposite direction.
- **NEUTRAL:** anything else.
- **CONTINUATION** ("just getting started") is judged on forward-looking statements (guidance, outlook), with pressure or decline wording as counter-evidence.
- **With a real LLM provider**, `complete()` is also asked whether the passage entails the sub-claim. It can only downgrade a SUPPORTING judgment or add CHALLENGING, never create support on its own. Passage and claim go to the model inside data blocks with a safety preamble, so instructions in them are treated as data.

## Verdict rule

Built from the stances of the usable evidence:

| Evidence | Verdict |
| --- | --- |
| Stated number does not match the computed one | CONTRADICTED |
| Supporting and challenging | PARTIALLY_SUPPORTED |
| Only supporting | SUPPORTED |
| Only challenging | CONTRADICTED |
| Nothing usable | INSUFFICIENT_EVIDENCE |

One extra rule: for computed sub-claim types (growth, profitability, cash, liquidity, leverage), SUPPORTED needs a Tier 1 computed record. If the support is only quoted text, it is PARTIALLY_SUPPORTED.

## Evidence the Auditor expects

From the **Financial agent**: `computed_value` and `metric_name`, and `content` in a fixed shape. Growth: both amounts and the percent ("$4,560m vs $4,000m (+14% YoY)"). Margin: current percent first, then the prior one. Direction claims: the current and prior figures ("Free cash flow $1.7B vs $9.0B"). Negative amounts carry a minus sign. Stance can be NEUTRAL.

From the **Narrative agent**: counter-evidence as well as supporting passages (risk factors, passages that rule out the claimed driver). Set CHALLENGING where a passage goes against the claim and NEUTRAL otherwise. The agent should also report that it searched for counter-evidence and found none, which still needs a field agreed with Stella and Hung.

## Evaluation

`finsight/eval/agent_eval/auditor.py` takes correct evidence and generates broken copies (units x100, calculation error, wrong metric, wrong comparison period, date = period end, wrong company, missing provenance, citation that doesn't support, hedged language, wrong tier, and others). The tests check that each is caught for the right reason, not only that something fails. Gold claims come from `claims_dev.jsonl`.

## Known limits

- **Direction basis is read from text.** There are no `comparison_value` / `comparison_period` fields in `EvidenceItem`, so a direction claim whose evidence text lacks the prior figure fails and retries. A structured basis would be more robust (needs a contract change by Hung).
- **Sign is not matched across wording.** "A $200m loss" in the evidence against "net income of $200m" in the claim is not caught.
- **Leverage metrics:** metric names are not validated until the Financial agent defines them.
- **Segment figures** (AWS, Azure) go through the text path only: the stated number must appear in the passage.
- **Not the Auditor's job:** cut-off filtering, the Tier 2 cap on financial evidence, and rolling up sub-claim verdicts into an overall label (orchestrator).
