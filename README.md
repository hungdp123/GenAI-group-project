# FinSight

FinSight is a multi-agent system (SMU MITB group project). It stress-tests a **company-level
investment thesis**, often a finfluencer claim, against evidence available as of a cut-off date.
It splits the thesis into claims and gathers both supporting and challenging evidence. Numbers are
checked by calculation and narrative claims by citation. Every verdict is limited by a source-tier rule.
The output is a structured assessment, never personalized investment advice.

| Tier | Sources |
|---|---|
| Tier 1 | SEC filings / XBRL (10-K, 10-Q) |
| Tier 2 | Earnings releases, investor-relations remarks |
| Tier 3 | Frozen news articles (can never raise a verdict) |

The pasted thesis is **never** evidence. Per-claim verdicts: `SUPPORTED`, `PARTIALLY_SUPPORTED`,
`CONTRADICTED`, `INSUFFICIENT_EVIDENCE`.

**Status:** the orchestrator, contracts, baselines and evaluation are built. Every other agent is a
**placeholder stub** returning fake data for a fictional "Company X", so the whole pipeline runs
end to end today. Each teammate replaces their own stub; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Team and ownership

| Person | Owns | Folder(s) |
|---|---|---|
| Person 1, Venes | Financial Analysis Agent + deterministic XBRL tools | `finsight/agents/financial/`, `finsight/data/xbrl_facts.json` |
| Person 2, Stella | Narrative Evidence Agent, vector DB, Basic RAG retriever | `finsight/agents/narrative/` |
| Person 3, Amelia | Thesis Decomposition + Verification/Auditor Agent; Accuracy / Macro-F1 | `finsight/agents/decomposition/`, `finsight/agents/auditor/` |
| Person 4, Irine | Source-tier rules, knowledge-layer content, Advice Boundary Agent, UI | `finsight/agents/tier_policy/`, `finsight/agents/advice_boundary/`, `finsight/knowledge/knowledge_layer.yaml`, UI |
| Person 5, Hung | Orchestrator, shared contracts, repo, LLM client, decision log, baselines, evaluation | everything else |

**Shared files you must not change without telling Hung:** `finsight/contracts/schemas.py`,
`finsight/contracts/interfaces.py`, and `finsight/agents/orchestrator/`. Everyone imports the
contracts, so a silent change breaks every teammate.

## Setup (once per machine)

Requires **Python 3.11+** and Git.

**Windows (PowerShell):**

```powershell
git clone git@github.com:hungdp123/GenAI-group-project.git
cd GenAI-group-project
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -e ".[dev]"
pytest                              # should end with "passed", no failures
```

**macOS / Linux:**

```bash
git clone git@github.com:hungdp123/GenAI-group-project.git
cd GenAI-group-project
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`pip install -e` installs the project in editable mode: your code edits take effect immediately,
and you get the `finsight` command. Re-activate the virtual environment in every new terminal.

## Quick start

```bash
finsight run "A finfluencer I follow said Company X's revenue growth proves it is the biggest winner from AI infrastructure demand, and that this growth is just getting started." --cutoff 2025-07-31
finsight trace <request_id>      # the decision log for the run above (id is printed at the bottom)
```

Expected with the stubs: 4 claims.

- Revenue growth: SUPPORTED (Tier 1, +14% YoY).
- AI demand attribution and "just getting started": PARTIALLY_SUPPORTED (knowledge-layer caps, with challenging evidence shown).
- "Biggest winner": INSUFFICIENT_EVIDENCE (peers are out of scope).
- Overall: PARTIALLY_SUPPORTED.

## Command reference

| Command | What it does |
|---|---|
| `finsight run "<thesis>" [--cutoff YYYY-MM-DD] [--json] [--no-evidence]` | Evaluate a thesis and print the assessment |
| `finsight trace <request_id> [--json]` | Print the decision log of a past run (`logs/<request_id>.json`) |
| `finsight eval --system {direct,rag,finsight,all} --split dev` | Run the evaluation (see below) |
| `finsight eval --validate` | Check both dataset files for mistakes |
| `finsight eval --freeze` | Lock the heldout split (once, when labels are final) |
| `finsight mode [mock\|paid] [--model M]` | Show or switch the LLM mode |
| `finsight usage` | Paid LLM spend recorded on this machine vs the $150 credit |
| `finsight graph` | Regenerate `docs/orchestrator_graph.md` |

If `finsight` is not found, activate the virtual environment, or use `python -m finsight ...`.

## LLM modes: mock (free) vs paid

**Mock mode is the default and costs nothing.** LLM calls return canned responses, so you can
build and test without an API key. Stay in mock mode unless you need a real model.

Paid mode uses the team's OpenAI course credit ($150, shared by everyone):

1. `pip install -e ".[dev,openai]"`
2. Copy `config.example.yaml` to `config.yaml` and paste the key. Get the key from Hung privately.
   `config.yaml` is gitignored: never commit it, and never paste the key in group chat or code.
3. Switch modes. The setting is saved in your `config.yaml`:

```bash
finsight mode                         # show the current mode
finsight mode paid                    # OpenAI, gpt-5.4-nano (cheap testing)
finsight mode paid --model gpt-5.4    # OpenAI, course model (final reported runs only)
finsight mode mock                    # back to free
```

Budget rules:

- Test with `gpt-5.4-nano` (about 12x cheaper than `gpt-5.4`).
- Each process stops at `FINSIGHT_SPEND_LIMIT_USD` (default $2).
- `pytest` never spends money.
- Check the team total on the OpenAI dashboard: https://platform.openai.com/usage

### Environment variables (optional overrides)

PowerShell: `$env:NAME="value"`. bash: `export NAME=value`, or put `NAME=value` before the command.

| Variable | Default | Meaning |
|---|---|---|
| `FINSIGHT_LLM_PROVIDER` | `mock` | `mock`, `openai` or `anthropic`; overrides `llm_provider` in `config.yaml` |
| `FINSIGHT_LLM_MODEL` | per provider | model name; must be in `config.PRICE_TABLE` |
| `FINSIGHT_SPEND_LIMIT_USD` | `2.00` | per-process spend cap for paid calls |
| `FINSIGHT_IMPL_<SLOT>` | from `config.py` | try your agent without editing code, e.g. `FINSIGHT_IMPL_FINANCIAL=finsight.agents.financial.agent:FinancialAgent` |
| `FINSIGHT_MAX_RETRIES` | `1` | evidence re-gathers per claim after a failed audit |
| `FINSIGHT_CORPUS_AS_OF` | `2025-07-31` | default cut-off when none is given |
| `FINSIGHT_LOGS_DIR` | `logs` | where decision logs are written |
| `FINSIGHT_STUB_FAIL_CLAIMS` | empty | stub auditor only: `c1` fails once, `c1:2` fails twice (retry testing) |
| `FINSIGHT_USAGE_LEDGER` | `logs/llm_usage.jsonl` | local record of every paid call |
| `FINSIGHT_SECRETS_FILE` | `config.yaml` | where keys and the LLM mode are stored |
| `FINSIGHT_REASONING_EFFORT` | unset | optional `reasoning_effort` for GPT-5 models |

## Team workflow (Git)

1. **Start from the latest `main`**, on your own branch. Use `<name>-<topic>` names: on Windows,
   `hung/x` cannot coexist with a branch called `Hung`.
   ```bash
   git checkout main
   git pull origin main
   git checkout -b venes-financial-agent
   ```
2. **Work only in your own folders.** The one shared line you edit is your slot in
   `DEFAULT_IMPLEMENTATIONS` in `finsight/config.py`.
3. **Before opening a PR,** bring in the latest `main` and run the checks:
   ```bash
   git pull origin main
   pytest
   finsight eval --validate
   git status                            # config.yaml, logs/, results/, data/ must NOT appear
   ```
4. **Open a PR into `main`** and ask one teammate to review. Use the checklist in
   [CONTRIBUTING.md](CONTRIBUTING.md#pull-request-checklist). Never push directly to `main`.

Files that conflict most often, and how to avoid it:

| File | Rule |
|---|---|
| `finsight/config.py` | Change only your own line in `DEFAULT_IMPLEMENTATIONS`. Ask Hung before changing anything else (corpus companies, prices) |
| `finsight/contracts/*.py` | Hung only. Request changes as described in [CONTRIBUTING.md](CONTRIBUTING.md#changing-the-shared-contracts) |
| `pyproject.toml` | If you add a dependency, add one line to `dependencies` and mention it in your PR |
| `finsight/knowledge/knowledge_layer.yaml` | Irine only |
| `finsight/eval/datasets/*.jsonl` | Coordinate with Hung and Amelia. Never edit the heldout file after it is frozen |

## Data files

- **Small fixtures** (KB-size JSON or YAML that tests need) go inside the package, e.g. `finsight/data/`.
- **Large corpus files** (filings, news dumps, vector DB indexes) go in the top-level `data/`
  folder, which is gitignored. Share them through the team drive, and commit a small script that
  rebuilds them instead of the files themselves.
- Every document needs a **publication date**. The orchestrator drops evidence published after
  the cut-off and logs it as a violation.

## Calling FinSight from the UI (Irine)

```python
from finsight import run_finsight, execute
from finsight.agents.orchestrator.render import render_assessment

assessment = run_finsight(thesis_text, cutoff=None)   # FinalAssessment (Pydantic model)
assessment.status            # COMPLETED | REFUSED | FAILED
assessment.refusal_reason    # show this when REFUSED / FAILED
assessment.overall_verdict
assessment.claim_results     # each: claim, verdict, highest_tier_used, supporting, challenging, rationale, capped_by
assessment.as_of_date, assessment.cutoff_was_defaulted, assessment.boundary_disclaimer   # always display these
render_assessment(assessment)                          # plain-text version
assessment.model_dump_json()                           # JSON for a web front end

run = execute(thesis_text, cutoff=None)                # also returns the decision log
run.assessment, run.decision_log
```

The disclaimer must always be shown. `run_finsight` never raises: problems come back as
`status=FAILED` with a reason. To see the full JSON shape, run `finsight run "<thesis>" --json`.

## Evaluation

```bash
finsight eval --system all --split dev                  # iterate on dev
finsight eval --system all --split heldout --final      # final reported run only
```

Results go to `results/{timestamp}/` (gitignored):

- `summary.md`: comparison table for Direct LLM, Basic RAG and FinSight
- `claims.csv`: per-claim predicted vs gold verdicts, for Accuracy / Macro-F1 (Amelia)
- `records.jsonl`, `raw/`, `logs/`

Two safeguards for the heldout split:

- It must be frozen first (`finsight eval --freeze`), and `--final` refuses to run if the file changed after that.
- In paid mode, final runs must use `gpt-5.4`.

Dataset format and labeling rules are in [docs/eval_dataset_guide.md](docs/eval_dataset_guide.md).

## Architecture

```mermaid
graph TD
  intake --> advice_gate_input
  advice_gate_input -->|flagged| finalize
  advice_gate_input --> scope_check
  scope_check -->|out of scope| finalize
  scope_check --> decompose
  decompose -->|advice claim / failure| finalize
  decompose --> plan
  plan --> gather_evidence
  plan -->|no collectors| apply_tier_policy
  gather_evidence --> audit --> retry_or_accept
  retry_or_accept -->|failed, retries left| gather_evidence
  retry_or_accept --> apply_tier_policy --> assemble --> advice_gate_output --> finalize
```

The generated diagram is in [docs/orchestrator_graph.md](docs/orchestrator_graph.md); regenerate it with `finsight graph`.

Safety rules enforced by the orchestrator:

- Verdicts come only from audit, tier policy and the orchestrator's guards, never from input text.
- Untrusted text goes only inside `data_block()` and is never placed in a system prompt.
- Evidence published after the cut-off is dropped and logged.
- Tier 3 evidence can never raise a verdict, and a tier policy that tries to raise one is ignored.
- If an agent fails, its claim becomes INSUFFICIENT_EVIDENCE instead of crashing the run.

## Documentation

| Document | Read it when |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | You are replacing your stub or opening a PR |
| [docs/decisions.md](docs/decisions.md) | You need the design decisions, the ones awaiting team sign-off, and known limitations |
| [docs/eval_dataset_guide.md](docs/eval_dataset_guide.md) | You are writing or labeling evaluation items |
| [docs/orchestrator_graph.md](docs/orchestrator_graph.md) | You need the exact pipeline graph |
| [docs/results/](docs/results/) | You want past evaluation runs kept for the report |

## Troubleshooting

| Problem | Fix |
|---|---|
| `finsight: command not found` | Activate the virtual environment, or use `python -m finsight` |
| PowerShell won't run `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry |
| `MissingAPIKey` | You are in paid mode without `config.yaml`. Run `finsight mode mock`, or add the key |
| `SpendLimitExceeded` | Intended safety stop. Raise `FINSIGHT_SPEND_LIMIT_USD` only if the spend is deliberate |
| `No price for '<model>'` | Use a model listed in `config.PRICE_TABLE`, or ask Hung to add its price |
| `KnowledgeLayerError` | `knowledge_layer.yaml` has an unknown claim type, verdict or field; the message names it |
| `TypeError: ... does not implement the <slot> protocol` | Your class's method name or arguments differ from `contracts/interfaces.py` |
| `pip install` fails with `OSError ... No such file or directory` and a "Long Path" hint (Windows) | The folder path is too long. Clone into a short path such as `C:\dev\GenAI-group-project` (not deep inside OneDrive), or enable Windows long paths |
| Git warns "LF will be replaced by CRLF" | Harmless on Windows |
