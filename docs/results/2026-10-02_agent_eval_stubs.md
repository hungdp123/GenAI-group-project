# FinSight per-agent evaluation

LLM mode: mock (mock-model). Each agent is the implementation registered in `finsight/config.py`; stubs score low on purpose where they do no real checking.

| Agent | Owner | Cases passed |
|---|---|---|
| decomposer | Person 3 (Amelia) | 7/9 |
| financial | Person 1 (Venes) | 6/6 |
| narrative | Person 2 (Stella) | 5/5 |
| retriever | Person 2 (Stella) | 5/5 |
| auditor | Person 3 (Amelia) | 3/28 |
| advice_boundary | Person 4 (Irine) | 16/20 |
| tier_policy | Person 4 (Irine) | 75/80 |

## decomposer (Person 3 (Amelia))

Implementation: `finsight.agents.decomposition.stub.StubDecomposer`

| Metric | Value |
|---|---|
| claims evaluated | 7 |
| sub-claim type precision | 82% |
| sub-claim type recall | 100% |
| sub-claim type F1 | 90% |
| sub-claim count exactly right | 71% |
| gold sub-claims matched by text | 100% |
| advice requests flagged (PERSONALIZED_ADVICE) | 100% |
| errors | 0 |

Cases passed: 7/9

- FAIL `dev-006`: missing -, extra ['DEMAND_ATTRIBUTION']
- FAIL `dev-011`: missing -, extra ['DEMAND_ATTRIBUTION', 'PROFITABILITY_CASH_LEVERAGE']

_Gold labels are the dev set's; they were written alongside the stub corpus (see docs/decisions.md, Known limitations)._
## financial (Person 1 (Venes))

Implementation: `finsight.agents.financial.stub.StubFinancialAgent`

| Metric | Value |
|---|---|
| cases | 6 |
| number correct (within 0.005) | 100% |
| right metric | 100% |
| right stance (for / against) | 100% |
| labelled Tier 1 | 100% |
| correctly returns nothing (no data / before filing) | 100% |
| items with URL + citation span | 100% |
| post-cut-off items returned | 0 |
| errors | 0 |

Cases passed: 6/6

_Cases are built on the placeholder Company X corpus, so the stub scores well by construction. Add real-corpus cases to finsight/eval/agent_eval/datasets/financial.jsonl before reporting numbers._
## narrative (Person 2 (Stella))

Implementation: `finsight.agents.narrative.stub.StubNarrativeAgent`

| Metric | Value |
|---|---|
| cases | 5 |
| evidence recall (expected passages found) | 100% |
| evidence precision (returned passages relevant) | 100% |
| right stance (for / against) | 100% |
| finds counter-evidence when it exists | 100% |
| items with URL + citation span | 100% |
| post-cut-off items returned | 0 |
| errors | 0 |

Cases passed: 5/5

_Cases are built on the placeholder Company X corpus, so the stub scores well by construction. Add real-corpus cases to finsight/eval/agent_eval/datasets/narrative.jsonl before reporting numbers._
## retriever (Person 2 (Stella))

Implementation: `finsight.agents.narrative.retriever_stub.StubRetriever`

| Metric | Value |
|---|---|
| queries | 5 |
| recall@3 | 100% |
| MRR (how high the first relevant passage ranks) | 100% |
| post-cut-off items returned | 0 |
| errors | 0 |

Cases passed: 5/5

_Cases are built on the placeholder Company X corpus, so the stub scores well by construction. Add real-corpus cases to finsight/eval/agent_eval/datasets/retriever.jsonl before reporting numbers._
## auditor (Person 3 (Amelia))

Implementation: `finsight.agents.auditor.stub.StubAuditor`

| Metric | Value |
|---|---|
| broken cases | 25 |
| catch rate (broken evidence rejected) | 0% |
| false-alarm rate (correct evidence rejected) | 0% |
|   catches: units / scale (x100) | 0% |
|   catches: calculation error | 0% |
|   catches: stated number contradicts | 0% |
|   catches: wrong metric | 0% |
|   catches: wrong comparison period | 0% |
|   catches: date = period end, not filing date | 0% |
|   catches: wrong company | 0% |
|   catches: missing provenance | 0% |
|   catches: conflicting evidence | 0% |
|   catches: citation doesn't support | 0% |
|   catches: stance mislabelled | 0% |
|   catches: hedged language as proof | 0% |
|   catches: wrong source tier | 0% |
| errors | 0 |

Cases passed: 3/28

- FAIL `units / scale (x100)/growth`: MISSED: endorsed broken evidence
- FAIL `units / scale (x100)/margin`: MISSED: endorsed broken evidence
- FAIL `calculation error/growth`: MISSED: endorsed broken evidence
- FAIL `calculation error/margin`: MISSED: endorsed broken evidence
- FAIL `stated number contradicts/growth`: MISSED: endorsed broken evidence
- FAIL `stated number contradicts/margin`: MISSED: endorsed broken evidence
- FAIL `wrong metric/growth`: MISSED: endorsed broken evidence
- FAIL `wrong metric/margin`: MISSED: endorsed broken evidence
- ... 17 more in `auditor.json`

_Endorse = passed with SUPPORTED or PARTIALLY_SUPPORTED. Both rates matter: misses let wrong evidence through; false alarms waste the retry and hide correct sub-claims._
## advice_boundary (Person 4 (Irine))

Implementation: `finsight.agents.advice_boundary.stub.StubAdviceBoundary`

| Metric | Value |
|---|---|
| cases | 20 |
| input (user's request): recall (advice caught) | 50% |
| input (user's request): precision (flags that were right) | 100% |
| input (user's request): F1 | 67% |
| output (FinSight's answer): recall (advice caught) | 75% |
| output (FinSight's answer): precision (flags that were right) | 100% |
| output (FinSight's answer): F1 | 86% |
| errors | 0 |

Cases passed: 16/20

- FAIL `adv-in-04`: missed advice: Would Company X be a good buy for me right now?
- FAIL `adv-in-05`: missed advice: I'm 25 with $10k. Is Company X a better pick for me than an index fund?
- FAIL `adv-in-06`: missed advice: Tell me whether to sell my Company X shares before earnings.
- FAIL `adv-out-04`: missed advice: Investors in your situation would do well to allocate 10% of their portfolio to Company X.

_Recall matters most at input (an advice request must never be analysed); precision matters for usability (normal claims must not be refused)._
## tier_policy (Person 4 (Irine))

Implementation: `finsight.agents.tier_policy.stub.StubTierPolicy`

| Metric | Value |
|---|---|
| cases (generated from the knowledge layer) | 80 |
| matches the knowledge-layer rules | 94% |
| illegal raises (verdict made stronger) | 0 |
| errors | 0 |

Cases passed: 75/80

- FAIL `DEMAND_ATTRIBUTION/news-only support/SUPPORTED`: expected INSUFFICIENT_EVIDENCE, got PARTIALLY_SUPPORTED
- FAIL `DEMAND_ATTRIBUTION/no evidence/SUPPORTED`: expected INSUFFICIENT_EVIDENCE, got PARTIALLY_SUPPORTED
- FAIL `CONTINUATION/news-only support/SUPPORTED`: expected INSUFFICIENT_EVIDENCE, got PARTIALLY_SUPPORTED
- FAIL `CONTINUATION/no evidence/SUPPORTED`: expected INSUFFICIENT_EVIDENCE, got PARTIALLY_SUPPORTED
- FAIL `BUSINESS_RISK/support + counter-evidence/SUPPORTED`: expected PARTIALLY_SUPPORTED, got SUPPORTED

_The orchestrator re-applies these rules as a safety net, so a mismatch here does not reach users; it shows where the agent itself disagrees with the agreed rules._
