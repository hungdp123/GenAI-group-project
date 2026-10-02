# FinSight progress report: Person 5 (Hung)

**As of 2 Oct 2026** · Merged into `main` (`94b1107`) · Schema **v0.3** · CI ✅ green

## Summary

The FinSight pipeline runs end to end today. The orchestrator, shared contracts, evaluation
harness and both baselines are complete. Every other agent is a **placeholder stub** that
implements the agreed interface and returns realistic fake data for a fictional "Company X".
So each teammate can replace their own stub independently, test it immediately, and merge it
without breaking anyone else.

| | |
|---|---|
| Automated tests | **91 passing** (mock mode, no API cost) |
| CI pipeline | **GitHub Actions on every push and PR**: lint, secrets check, tests on Ubuntu 3.11 / 3.12 + Windows 3.12; first runs on `Hung` and `main` both ✅ (about 1 min) |
| Per-agent evaluation | `finsight eval --agents all`: each agent scored on its own (7 evaluators; auditor by error injection). Stub baseline: [results](results/2026-10-02_agent_eval_stubs.md) |
| Dev evaluation set | **14 items**, all 6 categories |
| Heldout set | 3 placeholder items, **not frozen yet** |
| OpenAI spend so far | **$0.17 of $150** (one-off validation runs; now in free mock mode by default) |
| Architecture | Code matches the team diagram ([architecture.svg](architecture.svg)) |

## Terminology (changed in schema v0.3)

- **Claim:** what the user pastes in, e.g. *"Company X's revenue growth proves it is the biggest winner
  from AI infrastructure demand, and this growth is just getting started."* (Previously "thesis".)
- **Sub-claim:** one checkable statement the claim is split into (`sc1`, `sc2`, …). Each gets its own
  verdict: SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED or INSUFFICIENT_EVIDENCE. (Previously "claim".)

## What has been built

### 1. Shared contracts (`finsight/contracts/`)
- Pydantic v2 models for everything passed between agents: `ClaimRequest`, `SubClaim`, `EvidenceItem`,
  `AuditResult`, `AdviceDecision`, `SubClaimResult`, `FinalAssessment`, `DecisionLog`, and the graph state.
  All are strict: unknown fields are an error, not silently dropped.
- A `typing.Protocol` for each agent (Decomposer, FinancialAgent, NarrativeAgent, Auditor,
  AdviceBoundary, TierPolicy, Retriever). Every method receives the cut-off date.

### 2. Orchestrator (`finsight/agents/orchestrator/`), following the architecture diagram
| Diagram step | What the code does |
|---|---|
| Initial screening (advice + scope) | Advice requests are refused; price predictions, macro theses and companies outside the corpus are refused with a reason |
| Clarification | An empty claim, no company named, or nothing checkable returns a **question** to the user (status `NEEDS_CLARIFICATION`); the UI resubmits with the answer |
| Orchestrator: decompose + plan | The claim is split into sub-claims; the knowledge layer decides which agents gather evidence and the verdict cap |
| Financial + Narrative agents | Run in parallel per sub-claim; evidence published after the cut-off is dropped and logged |
| Auditor + "source tier enough?" | Number / citation checks, plus a required-tier check |
| Retry (max 1 per sub-claim) | The orchestrator **re-plans** (adds extra collectors) and re-gathers with feedback; still failing → INSUFFICIENT_EVIDENCE for that sub-claim only |
| Overall assessment | Tier policy + caps (Tier 3 news can never raise a verdict), then a deterministic overall-verdict rule |
| Final advice check | An answer with advice-like text is replaced by a refusal |
| Decision log | Every step, agent call, token, cost and retry → `logs/<request_id>.json` (`finsight trace <id>`) |

Safety rules enforced in code:
- The pasted claim is never treated as evidence.
- Text in the input or in documents cannot change verdicts (prompt-injection tests included).
- No LLM ever produces financial numbers.
- One agent failing marks only its sub-claim INSUFFICIENT_EVIDENCE; the run doesn't crash.

### 3. Placeholder stubs for teammates
Each teammate's folder has a `stub.py` with demo data. On the demo claim they produce the 4 expected
sub-claims: revenue growth SUPPORTED (+14% from XBRL), AI-demand attribution and "just getting
started" PARTIALLY_SUPPORTED, "biggest winner" INSUFFICIENT_EVIDENCE. Overall: PARTIALLY_SUPPORTED.

### 4. Knowledge layer (`finsight/knowledge/`)
- A YAML file with the sub-claim type table: which agents to call, the verdict cap, the required
  source tiers, and the extra collectors for a retry.
- A loader that rejects invalid content at startup.

### 5. LLM client and cost control (`finsight/llm/`)
- One `complete()` function for OpenAI (and Anthropic), reading the API key from the local `config.yaml`, as in the course template.
- **Mock mode (free) is the default.** `finsight mode paid` / `finsight mode mock` switches.
- Every paid call is costed from verified OpenAI prices and recorded in a ledger (`finsight usage`).
- A per-run spend cap ($2 default) stops runaway spending.
- Cheap testing model `gpt-5.4-nano`; final reported runs are forced to use `gpt-5.4`.

### 6. Baselines (`finsight/baselines/`)
- **Direct LLM:** the claim plus XBRL numbers in one LLM call.
- **Basic RAG:** top-k retrieval, then one LLM call.
- Both get the same refusal / scope / clarification rules as FinSight, so the comparison is fair.
- Both log tokens, latency and cost in the same format as FinSight.

### 7. Evaluation (`finsight/eval/`)
- `finsight eval --system all --split dev` runs all three systems and writes `summary.md`
  (task success, agent and tool calls, latency p50 / p95 / mean, tokens, cost, retry rate).
- It also writes `sub_claims.csv` (per-sub-claim predicted vs gold verdicts) for Amelia's Accuracy / Macro-F1.
- Dataset validator (`--validate`), and a **heldout freeze** (`--freeze`): once frozen, the final run
  refuses to start if the heldout file was edited.
- First real run (12 dev items, `gpt-5.4`, about $0.07): both baselines worked end to end. It also gave
  two case-study examples: the direct LLM **computed its own growth figure** and **merged two
  sub-claims into one wrong verdict**. FinSight is built to prevent both.

### 8. Repo, docs and team workflow
- **README:** setup for Windows and macOS, every command, mock / paid modes, Git workflow and
  conflict hotspots, how the UI calls FinSight (including the clarification loop), troubleshooting.
- **CONTRIBUTING:** step-by-step stub replacement, an example of calling the LLM from an agent, notes
  per agent, how to request a contract change, and a PR checklist.
- **End-to-end tests always use the stubs.** A teammate's real agent is checked only by the contract
  tests, so one person's merge can't break another person's tests.
- **[decisions.md](decisions.md):** every design decision, its reason, and which need team sign-off.
- **[eval_dataset_guide.md](eval_dataset_guide.md):** how to write and label evaluation items.

### 9. CI pipeline (`.github/`)
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push to any branch and on
every PR into `main`. Results show on the PR and in the repo's **Actions** tab.

| Check | What it does |
|---|---|
| **Lint (ruff)** | Unused imports, undefined names, lines over 120 characters |
| **No secrets committed** | Fails if `config.yaml` / `.env` is committed or an API key appears in any committed file |
| **Tests**: Ubuntu Python 3.11 + 3.12, Windows Python 3.12 | All tests, dataset validation, a demo claim run, and a check that the orchestrator graph doc is up to date |

- **No spending:** CI always runs in **mock mode**. It has no API key and cannot spend OpenAI credit.
- **Code owners** ([`.github/CODEOWNERS`](../.github/CODEOWNERS)): PRs that touch the shared contracts,
  the orchestrator, `config.py` or `.github/` automatically request Hung's review.
- **PR template:** new PRs open with the CONTRIBUTING checklist already filled in.
- **Starts green:** a linter config was added and 13 existing lint issues were fixed.
- **First results:** the first two runs passed: CI #1 on `Hung` (1m 21s) and CI #2 on `main` after the merge (1m 2s).
  This confirms the code works on Python 3.11 and on Windows, not only on the 3.12 used for development.

## What each teammate needs to do next

| Person | Next step | Where |
|---|---|---|
| **Venes**: Financial Analysis | Replace the financial stub with real XBRL tools; replace the demo XBRL fixture | `finsight/agents/financial/`, CONTRIBUTING.md |
| **Stella**: Narrative + retriever | Replace the narrative stub and the Basic RAG retriever; build the vector index into the gitignored `data/` folder | `finsight/agents/narrative/` |
| **Amelia**: Decomposition + Auditor | Replace both stubs; compute Accuracy / Macro-F1 from `sub_claims.csv`; co-own the gold labels | `finsight/agents/decomposition/`, `auditor/` |
| **Irine**: Tier policy, Advice Boundary, knowledge layer, UI | Replace both stubs; confirm the knowledge-layer content (incl. `retry_collectors`); build the UI on `run_finsight()` and handle all 4 statuses | `finsight/agents/tier_policy/`, `advice_boundary/`, `knowledge/`, README "Calling FinSight from the UI" |
| **Everyone** | Pull `main`, run `pytest`, read the README terminology section; replace your stub on your own branch and open a PR, which can only be merged once CI is green | README "Team workflow" and "Continuous integration" |

## Decisions that need team sign-off

From [decisions.md](decisions.md):

- **Overall-verdict rule:** all SUPPORTED → SUPPORTED; majority CONTRADICTED → CONTRADICTED; otherwise PARTIALLY_SUPPORTED.
- **Refuse vs clarify:** which requests are refused and which get a clarification question.
- **Fail closed:** the advice gate fails closed (an error means no answer).
- **BUSINESS_RISK:** how this sub-claim type is treated.
- **Retry collectors:** the seed `retry_collectors` per sub-claim type.
- **Fair baselines:** giving the baselines the same rules as FinSight.
- **Eval matching:** how gold sub-claims are matched to predicted ones.

## Known limitations (to state in the report)

1. **Placeholder data:** all agents except the orchestrator are stubs over one fictional company.
2. **Circular dev labels:** the dev gold labels were written alongside the stub data, so FinSight's
   current perfect score shows the pipeline works, not that it is accurate.
3. **Zero FinSight cost:** FinSight shows $0 / 0 tokens until real agents call the LLM through `complete()`.
4. **Heldout not ready:** the heldout set must be rebuilt from the real corpus by two labelers, then frozen.

## Next milestones

1. ~~**Merge:** merge `Hung` → `main`~~ ✅ done; CI green on `main`. Everyone pulls (schema v0.3 is a breaking rename).
2. **Branch protection** (Hung, GitHub settings): require a PR, Code Owner review and the 5 CI checks on `main`,
   so nothing can be merged with failing checks.
3. **First real agent:** suggested Venes's financial agent, because it's deterministic with no LLM cost.
4. **Real corpus:** Stella and Venes agree the company list and cut-off; update `CORPUS_COMPANIES`.
5. **Heldout labels:** build the real heldout set with two labelers, then run `finsight eval --freeze`.
6. **Final run:** `finsight eval --system all --split heldout --final` with `gpt-5.4`, then write up the results.
