# FinSight

FinSight is a multi-agent system that fact-checks **company-level investment claims**, the kind
finfluencers post, such as "Company X's revenue growth proves it is the biggest winner from AI demand".
It checks the claim against evidence available as of a cut-off date and returns a structured,
cited assessment. It never gives personalized investment advice. (SMU MITB group project.)

## How it works

1. **Gate.** Refuse requests for advice, ask which company if that is unclear, and keep
   untrusted text away from the system prompt.
2. **Match and split.** Find the influencer's own words and split the claim into
   **sub-claims** (`sc1`, `sc2`, …), one checkable statement each. See [docs/claim_matching.md](docs/claim_matching.md).
3. **Gather evidence.**
   - **Financial agent:** checks numbers by calculating from SEC XBRL data.
   - **Narrative agent:** uses RAG over filings, earnings releases and news to find cited passages.
4. **Audit.** An auditor checks that each verdict follows from the evidence, and gathers evidence again if it does not.
5. **Apply the source-tier rule.** A verdict can only be as strong as its sources allow:

   | Tier | Sources |
   |---|---|
   | 1 | SEC filings / XBRL (10-K, 10-Q) |
   | 2 | Earnings releases, investor-relations remarks |
   | 3 | News articles (can never raise a verdict) |

Each sub-claim gets one of these verdicts: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `CONTRADICTED` or
`INSUFFICIENT_EVIDENCE`. The overall verdict combines them.

Ground rules:
- The pasted claim is never used as evidence.
- Evidence published after the cut-off is dropped.
- If an agent fails, its sub-claim becomes `INSUFFICIENT_EVIDENCE` instead of crashing the run.
- Every step is written to a decision log.

The pipeline is a LangGraph orchestrator ([docs/orchestrator_graph.md](docs/orchestrator_graph.md),
[docs/architecture.svg](docs/architecture.svg)).

## Setup

Requires Python 3.11+.

```bash
git clone git@github.com:hungdp123/GenAI-group-project.git
cd GenAI-group-project
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"           # add ,openai,retrieval for real LLM calls and the narrative RAG
pytest
```

The narrative corpus and vector index are large, so they are not in Git. Get them from the team
drive and put them under `data/`.

## Usage

```bash
finsight run "Company X's revenue growth proves it is the biggest winner from AI demand" --cutoff 2025-07-31
finsight trace <request_id>       # decision log for that run
finsight ui --open                # local web page: run a claim, watch each stage, read the citations
```

From Python:

```python
from finsight import run_finsight
assessment = run_finsight(claim_text, cutoff=None)
print(assessment.summary, assessment.overall_verdict)
```

## LLM modes

**Mock mode is the default and free.** LLM calls return canned responses, so the whole pipeline
and every test run with no API key.

To use a real model:
1. Copy `config.example.yaml` to `config.yaml` and add the key. `config.yaml` is gitignored; never commit it.
2. Run `finsight mode paid` (`gpt-5.4-nano`) or `finsight mode paid --model gpt-5.4` (for final runs only).
3. Run `finsight mode mock` to go back to free mode.
4. Run `finsight usage` to see the spend.

Each process stops at `FINSIGHT_SPEND_LIMIT_USD` (default $2).

## Evaluation

```bash
finsight eval --system all --split dev                  # FinSight vs Direct LLM and Basic RAG baselines
finsight eval --system all --split heldout --final      # final reported run
```

Results go to `results/<timestamp>/`, including a `comparison.md`. The datasets are in
`finsight/eval/datasets/`; the heldout split is frozen. See [docs/evaluation.md](docs/evaluation.md).

## Project layout

| Path | What's there |
|---|---|
| `finsight/agents/orchestrator/` | LangGraph pipeline, gates, routing, rendering |
| `finsight/agents/claim_extraction/`, `claim_matching/` | Influencer claim ingestion, matching and sub-claim splitting |
| `finsight/agents/financial/` | XBRL-based financial analysis agent |
| `finsight/agents/narrative/` | Retrieval (BM25 / dense / hybrid) and cited-evidence RAG |
| `finsight/agents/auditor/`, `advice_boundary/`, `tier_policy/` | Verification, advice guardrail, source-tier rules |
| `finsight/contracts/` | Shared Pydantic schemas and agent interfaces |
| `finsight/knowledge/knowledge_layer.yaml` | Domain rules that cap verdicts by sub-claim type |
| `finsight/baselines/`, `finsight/eval/` | Baselines and evaluation harness |
| `finsight/ui/` | Local testing UI |
| `docs/` | Design decisions, specs, results |

## Team

| Member | Area |
|---|---|
| Venes | Financial analysis agent, XBRL tools |
| Stella | Narrative evidence agent, vector DB, RAG retriever |
| Amelia | Claim decomposition, auditor, accuracy metrics |
| Irine | Source-tier rules, knowledge layer, advice boundary, UI |
| Hung | Orchestrator, contracts, LLM client, baselines, evaluation |

## Contributing

Before opening a PR, run `python scripts/check.py`. It runs lint, the tests, a smoke run of every path
and the dev evaluation, all in mock mode. Read [CONTRIBUTING.md](CONTRIBUTING.md) before changing
`finsight/contracts/` or the orchestrator.
