# End-to-end evaluation of the claims routed to the FINANCIAL agent only (narrative agent switched off).
# Run from the repo root in PowerShell:   .\scripts\run_financial_eval.ps1            (mock: free, rules only)
#                                         .\scripts\run_financial_eval.ps1 -Paid      (real model for the LLM fallback)
# Needs: data/sec/processed/facts.csv + filings.csv, data/influencers/... posts, and the corrected
# data/influencers/claims/claim_update_v3.jsonl. Put the two dataset files in finsight/eval/datasets/.
param([switch]$Paid, [string]$Model = "gpt-5.4-nano")

# Narrative agent and retriever -> deterministic stubs: they return no evidence for real companies, so a
# narrative-routed sub-claim ends INSUFFICIENT_EVIDENCE at once instead of calling the unfinished agent.
$env:FINSIGHT_IMPL_NARRATIVE = "finsight.agents.narrative.stub:StubNarrativeAgent"
$env:FINSIGHT_IMPL_RETRIEVER = "finsight.agents.narrative.retriever_stub:StubRetriever"
Remove-Item Env:FINSIGHT_FINANCIAL_PLANNER -ErrorAction SilentlyContinue   # default: rules first, LLM fallback

if ($Paid) { $env:FINSIGHT_LLM_PROVIDER = "openai"; $env:FINSIGHT_LLM_MODEL = $Model }
else       { $env:FINSIGHT_LLM_PROVIDER = "mock";   Remove-Item Env:FINSIGHT_LLM_MODEL -ErrorAction SilentlyContinue }
python -m finsight mode

# 1. Claims routed to the financial agent: verdicts are scored.
python -m finsight eval --system finsight --split dev --dataset finsight/eval/datasets/claims_dev_financial.jsonl --out results/financial
$run = Get-ChildItem results/financial -Directory | Sort-Object Name | Select-Object -Last 1
python scripts/score_sub_claims.py "$($run.FullName)/sub_claims.csv" --out "$($run.FullName)/score.md"
python scripts/usage_report.py "$($run.FullName)" --out "$($run.FullName)/usage.md"   # latency + tokens: total, per item, per agent
python scripts/step_check.py "$($run.FullName)" finsight/eval/datasets/claims_dev_financial.jsonl --out "$($run.FullName)/steps.md"   # where each sub-claim first goes wrong

# 2. Gate items (advice, out of scope, no stored post, missing name): behaviour only, no agent is called.
python -m finsight eval --system finsight --split dev --dataset finsight/eval/datasets/claims_dev_gates.jsonl --out results/gates
$gates = Get-ChildItem results/gates -Directory | Sort-Object Name | Select-Object -Last 1
python scripts/usage_report.py "$($gates.FullName)" --out "$($gates.FullName)/usage.md"

Write-Host "`nFinancial results: $($run.FullName)  (summary.md, score.md, steps.md, usage.md, usage_items.csv, usage_agents.csv, logs/)"
