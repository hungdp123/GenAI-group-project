# FinSight design decision log

Owner: Person 5 (Hung). The spec left these points open. Each was settled with the simplest workable
option and marked `# DECISION:` in the code. Rows marked **Team** need group sign-off before the final report.

## Orchestration

| # | Decision | Why | Where | Status |
|---|---|---|---|---|
| D1 | Overall verdict: all SUPPORTED → SUPPORTED; all INSUFFICIENT (or no claims) → INSUFFICIENT; more than half of analyzable claims CONTRADICTED → CONTRADICTED; else PARTIALLY_SUPPORTED | Deterministic and explainable; never derived from LLM text | `agents/orchestrator/aggregation.py` | **Team** |
| D2 | Verdict strength order for caps: SUPPORTED > PARTIALLY > INSUFFICIENT > CONTRADICTED | A cap limits how much a claim is *endorsed*; it must never turn a contradiction into something else | `contracts/schemas.py` (`Verdict.support_rank`) | Done |
| D3 | The orchestrator re-applies its own guards after Person 4's TierPolicy: knowledge-layer cap, a positive verdict needs eligible supporting evidence (Tier 3 never counts), counter-evidence rule, and a policy that tries to *raise* a verdict is ignored | Safety holds even if one agent is wrong (defence in depth) | `agents/orchestrator/routing.py`, `nodes.py` | Done |
| D4 | If any collector fails for a claim, that claim becomes INSUFFICIENT_EVIDENCE; partial evidence is not used | Spec: one failed agent marks its claim, not the run. Keeping half the evidence could bias the verdict | `nodes.py: gather_evidence` | Done |
| D5 | The advice gate fails **closed**: if it errors at input or output, the run returns FAILED and nothing is analyzed or released | The advice boundary is a hard requirement; an unchecked answer is worse than no answer | `nodes.py` | **Team** |
| D6 | All out-of-scope requests (price prediction, macro / index, company not in corpus) are REFUSED with a reason | Simple to score against `expect_refusal`; clear to users | `routing.py: scope_refusal` | **Team** |
| D7 | Empty or overlong (>5000 chars) input is REFUSED, not FAILED | It is a bad request, not a system failure | `nodes.py: intake` | Done |
| D8 | If the output gate flags the answer, all claim results are removed and a refusal reason returned | Partial release could still leak advice-like text | `nodes.py: advice_gate_output` | Done |
| D9 | Cut-off priority: explicit argument → "Cut-off date: …" phrase in the thesis → corpus as-of date (flagged as defaulted in the output) | The demo thesis carries its own cut-off; a default must be visible | `routing.py: resolve_cutoff` | Done |
| D10 | Claims run in parallel in a thread pool inside the gather and audit nodes, rather than the LangGraph Send API | Keeps the state simple (no reducers); same speed-up | `nodes.py` | Done |
| D11 | The live decision log is held by a thread-safe recorder and copied into the graph state at `finalize` | Parallel agent calls append safely; the state stays plain data | `logging/decision_log.py` | Done |
| D12 | Total latency = wall-clock for the run, not the sum of agent calls | Agents run in parallel, so a sum overstates user-facing latency | `decision_log.py` | Done |
| D13 | Duplicate claim IDs from the decomposer are renumbered, not treated as an error | Robustness; logged as an event | `nodes.py: decompose` | Done |
| D14 | NEUTRAL evidence is kept and counts toward `highest_tier_used`, but is listed under neither supporting nor challenging | Schema only has the two lists | `nodes.py: assemble` | Done |
| D15 | Thesis text is dropped if it comes back as evidence (`doc_type` thesis, or content contained verbatim in the thesis) | Spec: the pasted thesis is never evidence | `routing.py: is_thesis_echo` | Done |
| D16 | BUSINESS_RISK: no cap, but any Tier 1–2 challenging evidence lowers SUPPORTED to PARTIALLY_SUPPORTED | Reading of "counter-evidence only, can block SUPPORTED" | `knowledge_layer.yaml` | **Team** (Irine) |

## Schema additions (all in `contracts/schemas.py`, version 0.1)

| # | Addition | Why |
|---|---|---|
| S1 | `FinalAssessment.cutoff_was_defaulted` | The output must state when the cut-off was defaulted |
| S2 | `AgentCall.claim_id` | Per-claim cost and latency tracing |
| S3 | `DecisionLog.temporal_violations`, `events`, `status` | Temporal drops must be logged; events give a readable trace |
| S4 | `KnowledgeRule` lives in `schemas.py` | `TierPolicy.apply` needs it, and agents may only import from `contracts` |

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
| E1 | Baselines get the same refusal and scope policy as FinSight, and are asked for atomic claims | Otherwise FinSight "wins" on rules the baselines were never told; the comparison should measure the architecture | `baselines/common.py` | **Team** |
| E2 | Gold claims are matched to predicted claims one-to-one by word overlap; an unmatched gold claim scores as `MISSING` (wrong) | Simple and transparent; under-splitting a thesis is a real failure | `eval/metrics.py` | **Team** (Amelia) |
| E3 | Retry rate = claims retried at least once / claims assessed | Per-claim view matches how retries work | `eval/metrics.py` | Done |
| E4 | Heldout split is frozen by a SHA-256 lock file; `--final` refuses to run if the file is unfrozen or changed | The heldout split cannot drift after we have seen results | `eval/run_eval.py` | Done |
| E5 | The dataset validator rejects duplicate IDs, dev→heldout thesis leaks, refusal items with gold claims, and items that should have gold claims but don't | Catches labeling mistakes before they become results | `eval/run_eval.py` | Done |

## Known limitations (state these in the report)

1. **Placeholder data.** The corpus is one fictional company ("Company X") and the agents are stubs.
2. **Circular gold labels.** The dev gold labels were written alongside the stub corpus, so FinSight's
   current 100% per-claim agreement shows the pipeline works, not that it is accurate. Real accuracy
   needs real agents, the real corpus, and gold labels written independently of the system
   (see `docs/eval_dataset_guide.md`).
3. **FinSight shows $0 / 0 tokens** in the eval because the stubs make no LLM calls. This becomes
   real once teammates' agents call `finsight.llm.client.complete()`.
4. **Baseline latency vs FinSight latency is not comparable yet**: the baselines make real API calls,
   while FinSight runs local stubs.
