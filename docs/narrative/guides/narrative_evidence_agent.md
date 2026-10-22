# Narrative Evidence Agent (Week 2, Task 6)

Implementation: `finsight.agents.narrative.rag.collect_evidence:NarrativeEvidenceAgent`.
Implements `NarrativeAgent.gather(sub_claim, cutoff_date, feedback)` without changing
shared contracts. Implementation and Task 7 technical verification are complete: 196 regression tests passed,
14 existing auditor checks xfailed, and 48 real-corpus checks passed. See the
[integration report](../results/2026-10-16_narrative_agent_integration.md). Shared-default
rollout and semantic stance acceptance remain pending; no paid calls were made.

## Pipeline

1. Require an explicit ticker, a `datetime.date` cutoff, a nonempty sub-claim ID and
   text (maximum 5,000 characters). Normalize the ticker; never infer it from documents.
2. Load that company's canonical index afresh for each gather. Missing company
   indexes or no temporally eligible passages return an empty list. Invalid inputs
   and corrupt indexes raise errors. Personalized advice sub-claims return no evidence.
3. Build separate observation and contrary-evidence search strings from the claim.
   Feedback contributes at most 40 search tokens; it cannot change backend, paths,
   ticker, cutoff, or tool settings. Claim search terms are bounded to 160 tokens.
4. Retrieve both sets with the requested backend (BM25 by default; dense/hybrid/auto
   available). Explicit paths route both indexes to the requested company, avoiding
   single-index environment overrides. Existing vector checks reject stale vectors.
5. Deduplicate by stable chunk ID, preserving support-search order followed by unseen
   counter-search results. A passage found in both directions is classified once.
6. Require every returned chunk ID and complete text to match eligible canonical
   content. Rebuild EvidenceItems from canonical metadata, including estimated-date
   provenance, page/chart citation spans, source URL and source tier. Missing URLs
   stay missing. Set the actual incoming sub_claim_id and agent producer name.
7. Classify each unique passage relative to the claim. Return original passage text;
   never replace it with generated prose, financial values or a generated citation.

There are exactly two retrieval calls per gather and no internal retry loops. The
orchestrator owns feedback retries (its default is one). A feedback invocation makes
one further pass; the agent keeps no cross-request retry counter. Exceptions propagate
to the orchestrator's existing failure handling rather than masquerading as no evidence.
At most twice top_k passages are returned. A retrieved passage is a candidate, not
automatically relevant evidence; uncertain candidates remain NEUTRAL.

## Documented stance methods

### `exact` (default, no LLM)

Case and whitespace normalized matching of the complete claim against a complete
source sentence/line assigns SUPPORTING. Other passages remain NEUTRAL. It never
infers CHALLENGING from risk keywords or from membership in the counter-search set.
This intentionally has low recall for paraphrases and cannot provide general semantic
support/challenge classification. Matching source statements is not independent factual
verification. Select `llm` or inject a reviewed classifier for semantic stance judgments.

### `llm` (explicit opt-in)

Uses the shared metered `complete()` client and its configured provider/model/spend
guard. One classification call per unique passage, no model retries. The fixed system
prompt defines SUPPORTING, CHALLENGING, and NEUTRAL, requires matching company,
period, metric, units and accounting basis, and disallows numeric computation.
Generic risks alone do not contradict a historical observation. Mixed or ambiguous
passages should be NEUTRAL.

Claims and passages are passed only through escaped `data_block()` user data;
neither changes system instructions. Only strict JSON with `stance` and `quote`
keys is accepted. A polarized stance requires a verbatim source quote of at least
20 characters. Malformed/truncated responses, unknown labels or invalid quotes
abstain to NEUTRAL. Passages over 24,000 characters abstain without a call, rather
than silently truncating source text. Provider failures propagate. The existing mock
provider's baseline prose is not valid classifier JSON and therefore abstains.

The quote guard checks source membership, not semantic entailment. Model judgments
and prompt injection resistance still require acceptance evaluation. This implementation
does not claim validated stance accuracy or a guarantee against all prompt injections.

## Run from the repository root

No LLM call:

```powershell
python -m finsight.agents.narrative.rag.collect_evidence --ticker NVDA --sub-claim-id sc1 --claim-type DEMAND_ATTRIBUTION --claim "NVIDIA's growth was driven by data center demand." --cutoff 2025-07-31 --backend bm25 --output data/narrative/experiments/narrative_agent/NVDA_sc1.json
```

Prints and optionally saves status, the stance method, both search queries and the full
EvidenceItems. `OK` means candidate passages were returned; inspect their stances.
`NO_EVIDENCE` means no candidates were returned. Both searches use the same cutoff.

Semantic classification, using the already configured shared provider/model:

```powershell
python -m finsight.agents.narrative.rag.collect_evidence --ticker NVDA --sub-claim-id sc1 --claim-type DEMAND_ATTRIBUTION --claim "NVIDIA's growth was driven by data center demand." --cutoff 2025-07-31 --backend bm25 --stance-method llm --output data/narrative/experiments/narrative_agent/NVDA_sc1_llm.json
```

The second command can incur API cost when the shared client is configured for a paid
provider. This implementation work did not execute it or change provider settings.

## Python use and dependency injection

```python
from datetime import date
from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent
from finsight.contracts.schemas import SubClaim, SubClaimType

agent = NarrativeEvidenceAgent(backend="bm25", stance_method="exact", top_k=5)
items = agent.gather(
    SubClaim(sub_claim_id="sc1", text="NVIDIA's growth was driven by data center demand.",
             company_ticker="NVDA", sub_claim_type=SubClaimType.DEMAND_ATTRIBUTION),
    date(2025, 7, 31), feedback=None,
)
```

Constructor injection supports `classifier` (a `classify(claim, evidence) -> Stance`
object) and `retriever_factory(ticker) -> Retriever`. Returned evidence is still
checked against the company's canonical index. A custom classifier is trusted code;
the caller owns its stance policy. Retrieval direction does not set stance.

Task 7 can inject the agent through `AgentRegistry.from_config(narrative=agent)`.
After acceptance, its zero-argument implementation path can be selected using:

```powershell
$env:FINSIGHT_IMPL_NARRATIVE = "finsight.agents.narrative.rag.collect_evidence:NarrativeEvidenceAgent"
# Optional semantic stance mode (uses the existing shared LLM configuration):
$env:FINSIGHT_NARRATIVE_STANCE_METHOD = "llm"
```

`FINSIGHT_NARRATIVE_CORPUS_ROOT` and `FINSIGHT_NARRATIVE_BACKEND` are also honored.
Shared application defaults remain unchanged. Set `FINSIGHT_CORPUS_PROFILE=canonical`
to select the 16-company application scope for development review. Switching the agent
alone does not change the default demo scope. Team acceptance is still required before
changing shared defaults; teammate-owned decomposer and auditor defaults remain stubs.
