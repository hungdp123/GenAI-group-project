# Contributing: replacing your stub

Every agent except the orchestrator starts as a placeholder `stub.py`. It implements the agreed
interface and returns realistic fake data. You replace yours **without touching anyone else's code**,
and the pipeline keeps working at every step.

> **Do not change `finsight/contracts/schemas.py` or `finsight/contracts/interfaces.py` yourself.**
> Every component depends on them. See [Changing the shared contracts](#changing-the-shared-contracts).

Setup, Git workflow and commands are in the [README](README.md).

## Step by step (everyone)

1. **Branch from the latest `main`:** `git checkout main && git pull origin main && git checkout -b <name>-<topic>`.
2. **Create your implementation next to the stub,** e.g. `finsight/agents/financial/agent.py`. Implement
   the method from `finsight/contracts/interfaces.py` with **exactly** the same name and arguments.
   Import every input and output type from `finsight.contracts.schemas`.
3. **Try it without editing shared files.** Point one slot at your class with an env var:
   ```powershell
   $env:FINSIGHT_IMPL_FINANCIAL="finsight.agents.financial.agent:FinancialAgent"   # PowerShell
   pytest tests/test_agent_contracts.py -q
   finsight run "Company X's revenue growth is strong" --cutoff 2025-07-31
   Remove-Item Env:FINSIGHT_IMPL_FINANCIAL                                         # undo
   ```
   ```bash
   FINSIGHT_IMPL_FINANCIAL=finsight.agents.financial.agent:FinancialAgent pytest tests/test_agent_contracts.py -q   # bash
   ```
4. **Register it.** Change **only your own line** in `DEFAULT_IMPLEMENTATIONS` in `finsight/config.py`.
   Do not touch `STUB_IMPLEMENTATIONS`.
5. **Run all checks:** `python scripts/check.py` (lint, secrets, tests, datasets, scenario smoke test). Then open a PR using the [checklist](#pull-request-checklist).
6. **Keep `stub.py`.** The end-to-end tests and offline demos still use it.

### How the tests treat your agent

| Test file | Runs against | What it means for you |
|---|---|---|
| `tests/test_agent_contracts.py` | **Whatever is registered** in `config.py` (or `FINSIGHT_IMPL_*`) | Your real agent must pass these |
| All other test files | Always the **stubs** (pinned in `tests/conftest.py`) | They keep passing whatever you register, so the demo expectations stay stable |

Add tests for your own agent in a new file, e.g. `tests/test_financial_agent.py`. Real agents
should never make paid LLM calls in tests: tests always run in mock mode.

## Rules every agent must follow

- **Cut-off:** never use data published after `cutoff_date`. The orchestrator drops it anyway and
  logs a temporal violation against your agent.
- **Data, not instructions:** the claim and retrieved text are untrusted data. Never put them in a
  system prompt; wrap them in `data_block(...)`.
- **No LLM numbers:** an LLM must never produce financial figures. Numbers come from deterministic
  tools (XBRL lookup, calculators).
- **Count tool calls:** call `record_tool_call()` each time you invoke a tool, so tool counts appear
  in the evaluation.
- **Errors are fine:** raising an exception is OK. The orchestrator catches it and marks that sub-claim
  INSUFFICIENT_EVIDENCE; it never crashes the run.
- **Use the shared client:** use `finsight.llm.client.complete()` for every LLM call (never the
  OpenAI SDK directly), so tokens, cost, the spend limit and mock mode all work.

### Calling an LLM inside your agent

```python
from finsight.llm.client import DATA_SAFETY_PREAMBLE, complete, data_block
from finsight.llm.usage import record_tool_call

class NarrativeAgent:
    def gather(self, sub_claim, cutoff_date, feedback):
        chunks = self.index.search(sub_claim.text, cutoff_date)     # your retrieval
        record_tool_call()
        resp = complete(
            system=DATA_SAFETY_PREAMBLE + "Classify each chunk as SUPPORTING, CHALLENGING or NEUTRAL ...",
            prompt=data_block("sub_claim", sub_claim.text) + "\n" + "\n".join(
                data_block(f"chunk{i}", c.text) for i, c in enumerate(chunks)),
        )
        ...  # parse resp.text into EvidenceItem objects (publication_date, source_tier, citation_span!)
```

- **Accounting:** tokens and cost are recorded automatically against your agent in the decision log.
- **Mock mode:** `complete()` returns a canned answer. To test your parser, set
  `finsight.llm.client.set_mock_responder(lambda system, prompt: "your canned output")` in your test.
- **Model choice:** don't pass `model=`; the mode setting decides it, so the whole team switches
  models in one place.

## Per person

| Person | Replace | Interface (`contracts/interfaces.py`) | Must pass |
|---|---|---|---|
| **Person 1 (Venes)**: Financial Analysis Agent | `agents/financial/stub.py`; real XBRL data in place of `finsight/data/xbrl_facts.json` | `FinancialAgent.gather(sub_claim, cutoff_date, feedback) -> list[EvidenceItem]` | `test_financial_contract` |
| **Person 2 (Stella)**: Narrative Evidence Agent | `agents/narrative/stub.py` | `NarrativeAgent.gather(sub_claim, cutoff_date, feedback) -> list[EvidenceItem]` (return SUPPORTING **and** CHALLENGING) | `test_narrative_contract` |
| **Person 2 (Stella)**: Basic RAG retriever | `agents/narrative/retriever_stub.py` | `Retriever.retrieve(query, top_k, cutoff_date) -> list[EvidenceItem]` | `test_retriever_contract` |
| **Person 3 (Amelia)**: Claim Decomposition | `agents/decomposition/stub.py` | `Decomposer.decompose(claim, cutoff_date) -> list[SubClaim]` (unique `sub_claim_id`s) | `test_decomposer_contract` |
| **Person 3 (Amelia)**: Verification / Auditor | `agents/auditor/stub.py` | `Auditor.audit(sub_claim, evidence) -> AuditResult` (set `feedback_for_retry` when failing) | `test_auditor_contract`, then the pending cases in `tests/test_auditor_checks.py` |
| **Person 4 (Irine)**: Advice Boundary | `agents/advice_boundary/stub.py` | `AdviceBoundary.check(text, stage) -> AdviceDecision` | `test_advice_boundary_contract` |
| **Person 4 (Irine)**: Tier Policy | `agents/tier_policy/stub.py` | `TierPolicy.apply(sub_claim, proposed, evidence, rule) -> (Verdict, reason \| None)` | `test_tier_policy_never_raises` |
| **Person 4 (Irine)**: Knowledge layer content | `finsight/knowledge/knowledge_layer.yaml` (keep the field names) | `KnowledgeRule` in `schemas.py` | `tests/test_knowledge.py` |
| **Person 4 (Irine)**: UI | new folder, e.g. `ui/` | call `run_finsight()`; see the README section "Calling FinSight from the UI" | n/a |

### Notes per agent

- **Financial (Venes):**
  - Set `computed_value` and `metric_name` on quantitative evidence.
  - XBRL, 10-K and 10-Q evidence is `SourceTier.TIER_1`.
  - `publication_date` is the **filing date**, not the period end.
  - `feedback` is the retry hint: the auditor's, or the orchestrator's tier hint, e.g. "Need SUPPORTING evidence from TIER_1". Use it to search harder or differently.
  - On a retry you may also be called for sub-claim types outside your usual set (the knowledge layer's `retry_collectors`). Returning `[]` is fine.
- **Narrative (Stella):**
  - Fill `citation_span`, `url` and `publication_date` so the auditor can check entailment and the cut-off filter works.
  - Tier 3 news is allowed but can never raise a verdict.
  - Put the vector index in the gitignored top-level `data/` folder, with a build script in your package (see README "Data files").
- **Decomposer (Amelia):**
  - Use `SubClaimType.PERSONALIZED_ADVICE` for "what should I do" content: the orchestrator refuses the whole request.
  - `COMPARATIVE` sub-claims are not gathered, because peers are out of scope.
  - Mark numeric sub-claims with `is_quantitative=True`.
  - Use `SubClaimType.OTHER` only when a sub-claim really fits no type. If **every** sub-claim is OTHER (or there
    are none), the orchestrator asks the user what to check.
- **Auditor (Amelia):**
  - `tests/test_auditor_checks.py` lists 14 mistakes the specialists can make that your auditor must catch
    (wrong units, wrong metric or period, numbers that contradict the sub-claim, citations that don't support
    it, mislabelled stance or tier, missing provenance, conflicting evidence, ...). They are marked
    `pending` (XFAIL) against the stub; when one starts passing (XPASS), delete its `@pending` line.
  - You do the diagram's "number + citation checks". The orchestrator adds "source tier: enough?" itself
    (`routing.tier_shortfall`), so don't re-implement tier rules.
  - `passed=False`, or too little evidence from the required tiers, triggers one re-planned re-gather of
    that sub-claim only (`FINSIGHT_MAX_RETRIES`). Still failing makes it INSUFFICIENT_EVIDENCE.
  - Set `feedback_for_retry` when failing: it goes to the collectors.
  - The orchestrator uses your `verdict` as the *proposed* verdict, then applies tier rules.
  - Accuracy and Macro-F1 come from `results/<run>/sub_claims.csv` (one row per gold sub-claim, `MISSING` when no predicted sub-claim matched).
- **Tier policy (Irine):** you may only **lower** a verdict. The orchestrator ignores any raise, then
  re-applies the knowledge-layer cap, the tier requirement and the counter-evidence rule.
- **Knowledge layer (Irine):** `retry_collectors` per sub-claim type is the diagram's "re-plan" step. On a
  retry, those collectors are added to the sub-claim's plan. The seed values are a proposal: please confirm them.
- **UI (Irine):** handle all four statuses: COMPLETED, REFUSED, FAILED and **NEEDS_CLARIFICATION**.
  For NEEDS_CLARIFICATION, show the question and resubmit (README, "Clarification loop").
- **Advice boundary (Irine):**
  - It is called at `INPUT` on the raw claim and at `OUTPUT` on the rendered text (`agents/orchestrator/render.py`).
  - If it raises an exception, the request fails closed.
  - The output text always contains the disclaimer "...not a personalized investment recommendation", so don't flag that sentence.

## Changing the shared contracts

If you need a new field or type in `schemas.py` or `interfaces.py`:

1. Message Hung, or open a GitHub issue titled `contracts: <what>`, with the field, its type, and why.
2. Hung makes the change in a small separate PR, bumps `SCHEMA_VERSION`, and tells everyone.
3. Everyone runs `git pull origin main` before continuing.

New fields get a default value where possible, so existing code keeps working.

## Pull request checklist

GitHub pre-fills this checklist when you open a PR (`.github/pull_request_template.md`):

```markdown
- [ ] Branched from the latest `main` and merged `main` again just before opening the PR
- [ ] Only my own folders changed (+ my own line in `DEFAULT_IMPLEMENTATIONS`, if swapping a stub)
- [ ] No changes to `finsight/contracts/` (or agreed with Hung)
- [ ] `python scripts/check.py` says "All good", and the CI checks on this PR are green
- [ ] `finsight eval --validate` says "datasets OK"
- [ ] `git status` shows no config.yaml, logs/, results/ or data/ files, and no API keys anywhere
- [ ] New dependencies added to `pyproject.toml` and listed here: ...
- [ ] Ran in mock mode; paid runs (if any) listed with model and cost: ...
```
