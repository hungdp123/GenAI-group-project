# FinSight design decision log

Owner: Person 5 (Hung). The spec left these points open. Each was settled with the simplest workable
option and marked `# DECISION:` in the code. Rows marked **Team** need group sign-off before the final report.

## Orchestration

| # | Decision | Why | Where | Status |
|---|---|---|---|---|
| D1 | Overall verdict: all SUPPORTED → SUPPORTED; all INSUFFICIENT (or no sub-claims) → INSUFFICIENT; more than half of analyzable sub-claims CONTRADICTED → CONTRADICTED; else PARTIALLY_SUPPORTED | Deterministic and explainable; never derived from LLM text | `agents/orchestrator/aggregation.py` | **Team** |
| D2 | Verdict strength order for caps: SUPPORTED > PARTIALLY > INSUFFICIENT > CONTRADICTED | A cap limits how much a sub-claim is *endorsed*; it must never turn a contradiction into something else | `contracts/schemas.py` (`Verdict.support_rank`) | Done |
| D3 | The orchestrator re-applies its own guards after Person 4's TierPolicy: knowledge-layer cap, a positive verdict needs eligible supporting evidence (Tier 3 never counts), counter-evidence rule, and a policy that tries to *raise* a verdict is ignored | Safety holds even if one agent is wrong (defence in depth) | `agents/orchestrator/routing.py`, `nodes.py` | Done |
| D4 | If any collector fails for a sub-claim, that sub-claim becomes INSUFFICIENT_EVIDENCE; partial evidence is not used | Spec: one failed agent marks its sub-claim, not the run. Keeping half the evidence could bias the verdict | `nodes.py: gather_evidence` | Done |
| D5 | The advice gate fails **closed**: if it errors at input or output, the run returns FAILED and nothing is analyzed or released | The advice boundary is a hard requirement; an unchecked answer is worse than no answer | `nodes.py` | **Team** |
| D6 | Price predictions, macro / index claims, and a company named via `company_ticker` that is outside the corpus are REFUSED with a reason. A claim naming no covered company gets a clarification question instead (D19) | Matches the architecture diagram: refuse what is out of scope, ask when information is missing | `routing.py: scope_screen` | **Team** |
| D7 | Overlong (>5000 chars) input is REFUSED, not FAILED; an empty claim asks for clarification | Bad request, not a system failure; empty input is missing info | `nodes.py: intake` | Done |
| D8 | If the output gate flags the answer, all sub-claim results are removed and a refusal reason returned | Partial release could still leak advice-like text | `nodes.py: advice_gate_output` | Done |
| D9 | Cut-off priority: explicit argument → "Cut-off date: …" phrase in the claim → corpus as-of date (flagged as defaulted in the output) | The demo claim carries its own cut-off; a default must be visible | `routing.py: resolve_cutoff` | Done |
| D10 | Sub-claims run in parallel in a thread pool inside the gather and audit nodes, rather than the LangGraph Send API | Keeps the state simple (no reducers); same speed-up | `nodes.py` | Done |
| D11 | The live decision log is held by a thread-safe recorder and copied into the graph state at `finalize` | Parallel agent calls append safely; the state stays plain data | `logging/decision_log.py` | Done |
| D12 | Total latency = wall-clock for the run, not the sum of agent calls | Agents run in parallel, so a sum overstates user-facing latency | `decision_log.py` | Done |
| D13 | Duplicate sub-claim IDs from the decomposer are renumbered, not treated as an error | Robustness; logged as an event | `claim_extraction/extract.py` | Done |
| D14 | NEUTRAL evidence is kept and counts toward `highest_tier_used`, but is listed under neither supporting nor challenging | Schema only has the two lists | `nodes.py: assemble` | Done |
| D15 | Claim text is dropped if it comes back as evidence (`doc_type` claim, or content contained verbatim in the claim) | Spec: the pasted claim is never evidence | `routing.py: is_claim_echo` | Done |
| D17 | End-to-end tests are pinned to the stubs (`STUB_IMPLEMENTATIONS`); only the contract tests run the registered agents | Real agents can merge one at a time without breaking demo expectations | `config.py`, `tests/conftest.py` | Done |
| D18 | Building the agent registry happens inside `execute`'s error handling | A mis-registered agent returns FAILED with the reason instead of crashing the UI | `orchestrator/graph.py` | Done |
| D23 | The user-facing output leads with a plain-language **summary**, then each point's verdict and a short **explanation**, then evidence with readable source names. The summary and explanations are **built by rule from the verdicts and evidence, not by an LLM** (schema v0.4: `FinalAssessment.claim` / `summary`, `SubClaimResult.explanation`) | Readers shouldn't need to decode `[T1 10-Q 2025-07-25]`. Rule-built text can't invent facts or numbers and never contains advice language. The technical view remains (`--technical`) | `orchestrator/explain.py`, `render.py` | Done |
| D16 | BUSINESS_RISK: no cap, but any Tier 1–2 challenging evidence lowers SUPPORTED to PARTIALLY_SUPPORTED | Reading of "counter-evidence only, can block SUPPORTED" | `knowledge_layer.yaml` | **Team** (Irine) |

### Aligning the code with the architecture diagram (2026-10-02)

| # | Decision | Why | Where | Status |
|---|---|---|---|---|
| D19 | **Clarification** node: an empty claim, no covered company named, or no checkable sub-claim (all sub-claims OTHER) stops the run with status NEEDS_CLARIFICATION and a question. The UI resubmits with the answer (`company_ticker=` or an edited claim) | Diagram: "Unclear or missing info → Clarification → asks user". Asking is better than guessing the company or analyzing a vague sub-claim | `nodes.py: clarification`, `routing.py` | **Team** |
| D20 | The advice gate runs before any clarification, so advice requests are refused even when the company is missing | Advice must never be engaged with, even to ask a follow-up question | `graph.py` | Done |
| D21 | **"Source tier: enough?"** is checked before the retry decision: a sub-claim retries if the audit failed **or** it lacks evidence from its required tiers (positive verdict → eligible SUPPORTING evidence; otherwise any required-tier item). Still not enough after the retry → INSUFFICIENT_EVIDENCE | Diagram puts the tier check inside the auditor's decision, with "No" leading to a retry | `routing.tier_shortfall`, `nodes.py: retry_or_accept` | Done |
| D22 | **Re-plan on retry**: each knowledge-layer rule lists `retry_collectors`, added to the sub-claim's plan for its retry (seed: numeric types add narrative; demand / risk add financial; continuation adds none) | Diagram: "Orchestrator re-plans / retrieves more"; data-driven so Irine controls it | `knowledge_layer.yaml`, `routing.replan_entry` | **Team** (Irine) |

## Schema additions (all in `contracts/schemas.py`, version 0.2)

| # | Addition | Why |
|---|---|---|
| S1 | `FinalAssessment.cutoff_was_defaulted` | The output must state when the cut-off was defaulted |
| S2 | `AgentCall.sub_claim_id` | Per-sub-claim cost and latency tracing |
| S3 | `DecisionLog.temporal_violations`, `events`, `status` | Temporal drops must be logged; events give a readable trace |
| S4 | `KnowledgeRule` lives in `schemas.py` | `TierPolicy.apply` needs it, and agents may only import from `contracts` |
| S5 | v0.2: `AssessmentStatus.NEEDS_CLARIFICATION`, `FinalAssessment.clarification_question`, `KnowledgeRule.retry_collectors`, `"clarification"` gate | Clarification and re-plan steps from the architecture diagram; all new fields have defaults, so existing code keeps working |
| S6 | v0.3 (**breaking**): the user's input is the **claim** (`ClaimRequest`, `raw_claim`, `run_finsight(claim)`); its parts are **sub-claims** (`SubClaim`, `SubClaimType`, `SubClaimResult`, `sub_claim_id`, `sub_claim_type`, `sub_claim_results`; IDs `sc1`, `sc2`, …). Formerly "thesis" and "claim" | Team terminology. "Sub-claim" keeps the input and its parts unambiguous in code, logs and the report. Teammates must pull `main` before continuing; old `logs/*.json` cannot be read by `finsight trace` |

## LLM usage and cost

| # | Decision | Why | Where |
|---|---|---|---|
| L1 | Testing defaults to `gpt-5.4-nano`; reported (`--final`) runs must use `gpt-5.4`, the course template's model | Same model family and API behaviour at about 1/12 of the cost while iterating; reported numbers stay on the course model | `config.py`, `eval/run_eval.py` |
| L2 | Prices taken from OpenAI's pricing page on 2026-10-02. A real call to a model missing from `PRICE_TABLE` is refused | Cost numbers in the report must be traceable; an unpriced model would silently report $0 | `config.py`, `llm/client.py` |
| L3 | API keys come from an env var or the gitignored `config.yaml`, never from code | Course template convention; keys never committed | `llm/spend.py` |
| L4 | Per-process spend limit (`FINSIGHT_SPEND_LIMIT_USD`, default $2) and an append-only usage ledger (`finsight usage`) | Protects the shared $150 credit from runaway loops | `llm/spend.py` |
| L5 | Reasoning models (GPT-5 family, o-series) get `max_completion_tokens` and no `temperature` | These parameters are what the API accepts; matches the template | `llm/client.py` |

## Evaluation

| # | Decision | Why | Where | Status |
|---|---|---|---|---|
| E1 | Baselines get the same refusal and scope policy as FinSight, and are asked for atomic sub-claims | Otherwise FinSight "wins" on rules the baselines were never told; the comparison should measure the architecture | `baselines/common.py` | **Team** |
| E2 | Gold sub-claims are matched to predicted sub-claims one-to-one by word overlap; an unmatched gold sub-claim scores as `MISSING` (wrong) | Simple and transparent; under-splitting a claim is a real failure | `eval/metrics.py` | **Team** (Amelia) |
| E3 | Retry rate = sub-claims retried at least once / sub-claims assessed | Per-sub-claim view matches how retries work | `eval/metrics.py` | Done |
| E4 | Heldout split is frozen by a SHA-256 lock file; `--final` refuses to run if the file is unfrozen or changed | The heldout split cannot drift after we have seen results | `eval/run_eval.py` | Done |
| E5 | The dataset validator rejects duplicate IDs, dev→heldout claim leaks, refusal items with gold sub-claims, and items that should have gold sub-claims but don't | Catches labeling mistakes before they become results | `eval/run_eval.py` | Done |
| E6 | Eval items can set `expect_clarification`; task success requires both refusal and clarification behaviour to match. Baselines get the same "CLARIFY: <question>" instruction | Clarification is part of the architecture, so it is scored, and fairly | `eval/`, `baselines/common.py` | Done |
| E7 | **Per-agent evaluation** (`finsight eval --agents`) alongside the end-to-end one. Every agent is scored on its own labelled cases, always the registered implementation. The auditor is scored by **error injection** (correct evidence + 13 automatically broken variants: catch rate and false-alarm rate). Tier-policy cases are generated from the knowledge layer | End-to-end scores can't say *which* agent is wrong, and the orchestrator's safety nets hide agent mistakes (e.g. the tier-policy stub's early cap). Per-agent numbers give each owner a target, and before/after numbers for their PR and the report | `eval/agent_eval/` | Done |

## Known limitations (state these in the report)

1. **Placeholder data.** The corpus is one fictional company ("Company X") and the agents are stubs.
2. **Circular gold labels.** The dev gold labels were written alongside the stub corpus, so FinSight's
   current 100% per-sub-claim agreement shows the pipeline works, not that it is accurate. Real accuracy
   needs real agents, the real corpus, and gold labels written independently of the system
   (see `docs/eval_dataset_guide.md`).
3. **FinSight shows $0 / 0 tokens** in the eval because the stubs make no LLM calls. This becomes
   real once teammates' agents call `finsight.llm.client.complete()`.
4. **Baseline latency vs FinSight latency is not comparable yet**: the baselines make real API calls,
   while FinSight runs local stubs.
