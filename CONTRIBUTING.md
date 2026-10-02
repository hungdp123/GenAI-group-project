# Contributing: how to replace your stub

Every agent except the orchestrator starts as a placeholder `stub.py` that implements the agreed
Protocol and returns realistic fake data. You can replace yours **without touching anyone else's
code**.

> **Do not change `finsight/contracts/schemas.py` or `finsight/contracts/interfaces.py` without telling Hung.**
> Every component depends on them. If you need a new field, ask first; Hung bumps `SCHEMA_VERSION`.

## Steps (everyone)

1. Create your implementation next to the stub, e.g. `finsight/agents/financial/agent.py`. Keep the
   method signature exactly as in the Protocol.
2. Point the registry at it with **one line** in `finsight/config.py` (`DEFAULT_IMPLEMENTATIONS`), or
   try it first with an env var, without editing code:
   `FINSIGHT_IMPL_FINANCIAL=finsight.agents.financial.agent:XbrlFinancialAgent pytest`
3. Run `pytest`. `tests/test_agent_contracts.py` runs against **whatever is registered** and must pass.
   `tests/test_stubs.py` checks stub demo data: delete your stub's test there once it's replaced.
   `tests/test_orchestrator.py` must still pass. If a demo expectation legitimately changes because
   your agent is better, tell Hung so the expected values can be updated together.
4. Leave `stub.py` in place (tests and offline demos use it).

Rules that apply to every agent:

- Never use data published after `cutoff_date`. The orchestrator drops it anyway and logs a
  temporal violation against you.
- The thesis and retrieved text are **data**. If you call an LLM, use `finsight.llm.client.complete()`
  and wrap untrusted text in `data_block(...)`. Never put it in `system`.
- Never let an LLM produce financial numbers. Numbers come from deterministic tools.
- Call `finsight.llm.usage.record_tool_call()` whenever you invoke a tool, so tool counts show in the eval.
- Raising an exception is fine: the orchestrator catches it and marks the claim INSUFFICIENT_EVIDENCE.

## Per person

| Person | Replace | Protocol (`contracts/interfaces.py`) | Contract test that must pass |
|---|---|---|---|
| **Person 1 (Venes)**: Financial Analysis Agent | `agents/financial/stub.py` (+ real data in place of `finsight/data/xbrl_facts.json`) | `FinancialAgent.gather(claim, cutoff_date, feedback) -> list[EvidenceItem]` | `test_financial_contract` |
| **Person 2 (Stella)**: Narrative Evidence Agent | `agents/narrative/stub.py` | `NarrativeAgent.gather(claim, cutoff_date, feedback) -> list[EvidenceItem]` (return SUPPORTING **and** CHALLENGING) | `test_narrative_contract` |
| **Person 2 (Stella)**: Basic RAG retriever | `agents/narrative/retriever_stub.py` | `Retriever.retrieve(query, top_k, cutoff_date) -> list[EvidenceItem]` | `test_retriever_contract`, `tests/test_baselines.py` |
| **Person 3 (Amelia)**: Thesis Decomposition | `agents/decomposition/stub.py` | `Decomposer.decompose(thesis, cutoff_date) -> list[Claim]` (unique `claim_id`s) | `test_decomposer_contract` |
| **Person 3 (Amelia)**: Verification / Auditor | `agents/auditor/stub.py` | `Auditor.audit(claim, evidence) -> AuditResult` (set `feedback_for_retry` when failing) | `test_auditor_contract` |
| **Person 4 (Irine)**: Advice Boundary | `agents/advice_boundary/stub.py` | `AdviceBoundary.check(text, stage) -> AdviceDecision` | `test_advice_boundary_contract` |
| **Person 4 (Irine)**: Tier Policy | `agents/tier_policy/stub.py` | `TierPolicy.apply(claim, proposed, evidence, rule) -> (Verdict, reason \| None)` | `test_tier_policy_never_raises` |
| **Person 4 (Irine)**: Knowledge layer content | `finsight/knowledge/knowledge_layer.yaml` (keep the schema) | `KnowledgeRule` in `schemas.py` | `tests/test_knowledge.py` |

### Notes per agent

- **Financial (Person 1):** set `computed_value` and `metric_name` on quantitative evidence. Use
  `SourceTier.TIER_1` for XBRL/10-K/10-Q. `feedback` is the auditor's retry hint (e.g. "need an exact
  citation span").
- **Narrative (Person 2):** fill `citation_span` and `url` so the auditor can check entailment. Tier 3
  news is allowed but can never raise a verdict.
- **Decomposer (Person 3):** use `ClaimType.PERSONALIZED_ADVICE` for "what should I do" content: the
  orchestrator refuses the whole request. `COMPARATIVE` claims are not gathered (peers are out of scope).
- **Auditor (Person 3):** `passed=False` triggers one re-gather of that claim only (`FINSIGHT_MAX_RETRIES`).
  A second failure makes it INSUFFICIENT_EVIDENCE.
- **Tier policy (Person 4):** you may only **lower** a verdict. The orchestrator ignores any raise and
  then re-applies the knowledge-layer cap, the tier requirement and the counter-evidence rule.
- **Advice boundary (Person 4):** called at `INPUT` on the raw thesis and at `OUTPUT` on the rendered
  assessment text (`agents/orchestrator/render.py`). If it raises an exception, the request fails closed.

## Testing a single swap quickly

```bash
FINSIGHT_IMPL_NARRATIVE=finsight.agents.narrative.agent:NarrativeAgent pytest tests/test_agent_contracts.py -q
FINSIGHT_IMPL_NARRATIVE=finsight.agents.narrative.agent:NarrativeAgent finsight run "Company X's growth comes from AI demand" --cutoff 2025-07-31
```
