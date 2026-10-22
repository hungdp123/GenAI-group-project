# Narrative entry points for the orchestrator

The three integration files expose the existing Narrative workflow and shared interfaces. Algorithms remain in their processing, PDF, retrieval, RAG and evaluation modules. Importing these entry points does not run document extraction, experiments, index rebuilding or generation.

| Entry file | Export | Role |
|---|---|---|
| `finsight/agents/narrative/stub.py` | `NarrativeAgent` | Retrieves sources, then uses an LLM for relevance selection, exact facts and a cited summary; no stance/verdict |
| `finsight/agents/narrative/retriever_stub.py` | `NarrativeRetriever` | Real BM25/dense/hybrid retriever, with explicit or unambiguous company scope |
| `finsight/agents/narrative/__init__.py` | Both classes, `run_basic_narrative_rag`, `WORKFLOW_MODULES` | Package imports and the module map for preparation, experiments, inference and validation |

The original `StubNarrativeAgent` and `StubRetriever` remain deterministic demo classes. Compatibility modules under `demo/` preserve the existing `STUB_IMPLEMENTATIONS` import paths. The two Narrative entries in `DEFAULT_IMPLEMENTATIONS` now select the real entry classes. Shared contracts and other owners' registry entries were not changed.

## Input and output contracts

```python
from datetime import date
from finsight.agents.narrative import NarrativeAgent, NarrativeRetriever
from finsight.contracts.schemas import SubClaim, SubClaimType

claim = SubClaim(
    sub_claim_id="meta-dap-june2024",
    text="DAP was 3.27 billion on average for June 2024",
    company_ticker="META",
    sub_claim_type=SubClaimType.OTHER,
)

agent = NarrativeAgent(backend="bm25", stance_method="none", analysis_mode="evidence")
evidence = agent.gather(claim, date(2025, 7, 31), feedback=None)

retriever = NarrativeRetriever(company_ticker="META", backend="bm25")
passages = retriever.retrieve("DAP June 2024", 3, date(2025, 7, 31))
```

`gather` and `retrieve` return `list[EvidenceItem]`, not generated answer dictionaries. Internal stance is an unassessed NEUTRAL placeholder required by the current shared contract. `gather` binds the incoming subclaim ID to every item. Each item preserves indexed content, source tier, document type, company, reporting period, publication-date provenance, URL and citation span. Collection does not classify stance; exported evidence omits the mandatory internal placeholder.

The orchestrator creates `ClaimRequest`, obtains `SubClaim` objects from its decomposer, routes them to collectors and sends evidence to the auditor and tier policy. The collector accepts the orchestrator's optional feedback for its next search; the orchestrator owns retry decisions. Other components can still be stubs, so registering a real collector alone does not establish full-system verdict accuracy.

Stance classification remains removed. The LLM instead prepares relevant evidence, exact facts and a cited summary. Old stance flags remain ignored; model flags now configure this evidence-preparation call.

## Runtime configuration

Run from the project root after restoring current production data:

```powershell
$env:FINSIGHT_CORPUS_PROFILE = "canonical"
$env:FINSIGHT_IMPL_NARRATIVE = "finsight.agents.narrative.stub:NarrativeAgent"
$env:FINSIGHT_IMPL_RETRIEVER = "finsight.agents.narrative.retriever_stub:NarrativeRetriever"
$env:FINSIGHT_NARRATIVE_CORPUS_ROOT = "data/narrative/processed"
$env:FINSIGHT_NARRATIVE_BACKEND = "bm25"
$env:FINSIGHT_NARRATIVE_STANCE_METHOD = "none"
$env:FINSIGHT_NARRATIVE_ANALYSIS = "evidence"
$env:FINSIGHT_NARRATIVE_ANALYSIS_MAX_TOKENS = "1500"
$env:FINSIGHT_LLM_PROVIDER = "openai"
$env:FINSIGHT_LLM_MODEL = "gpt-5.4"
$env:FINSIGHT_NARRATIVE_TOP_K = "5"
$env:FINSIGHT_NARRATIVE_CANDIDATE_K = "50"
```

The registry now defaults to the real classes; the implementation variables above make the selection explicit and override any older shell settings. The corpus profile remains opt-in: `canonical` exposes the 16 company aliases to application scope checks, while the default `demo` profile describes CMPX.

The narrative collector always scopes its searches to `SubClaim.company_ticker`. The separately registered retriever has no ticker argument in the shared `retrieve` contract, so use an explicit constructor ticker or `FINSIGHT_NARRATIVE_TICKER`, or supply a query naming one canonical company. Unscoped queries return no evidence; queries naming multiple companies require explicit scope. No implicit NVDA fallback is used by this registry adapter.

Data: BM25 needs `data/narrative/processed/<TICKER>/indexes/index.json`. Dense/hybrid also need matching Qdrant storage and embeddings. Absent company data yields no evidence; stale or explicitly misconfigured vector/index paths raise errors that the orchestrator handles. Restore the corpus snapshot separately from Git.

## Full workflow map

```python
from finsight.agents.narrative import WORKFLOW_MODULES
print(WORKFLOW_MODULES)
```

The map exposes module names for acquisition/cataloging, text parsing/chunking/BM25, PDF/chart extraction and promotion, vectors, retrieval, collection, Basic RAG answers, the 92-claim batch, provenance validation, integration checks, experimental source/label preparation, C1-C4 validation and retrieval/priority comparison. The runnable commands and saved results are in [summary_week_2.md](summary_week_2.md).

`WORKFLOW_MODULES` is a directory of script locations. Each key describes an operation, and its value identifies the Python module implementing it. Importing this dictionary does not execute those operations or define an automatic execution order.

| Entry | Purpose | When to use it |
|---|---|---|
| `acquire_core_documents` | Downloads core SEC filings, such as annual and quarterly reports | When collecting source documents |
| `acquire_supplemental_documents` | Downloads additional SEC documents and relevant exhibits | When expanding source coverage |
| `catalog` | Creates a readable Markdown inventory from document manifests | To inspect available documents |
| `process_text_and_build_bm25` | Parses documents, creates chunks and builds company BM25 indexes | When preparing or refreshing searchable data |
| `extract_pdf_charts_and_promote` | Extracts PDF text, tables and charts, then merges applicable content into company passages | When processing richer PDF content |
| `build_vectors` | Creates embedding vectors and matching Qdrant indexes | Before using dense or hybrid retrieval |
| `retrieve` | Searches company indexes and returns passages with citation metadata | To inspect retrieval results |
| `collect_evidence` | Builds support/counter searches and returns unassessed source passages with provenance | For evidence collection from a claim |
| `retrieve_basic_rag_evidence` | Retrieves unassessed source evidence with provenance | For the current basic_rag command |
| `generate_basic_rag_answer` | Uses the retained verdict_baseline module to generate a verdict and validated citations | For the historical answer-generation baseline |
| `generate_92_claim_answers` | Runs Basic RAG across the development dataset and aggregates results and usage | For the optional batch experiment |
| `validate_production_provenance` | Checks retrieved evidence against production indexes, including company, cutoff and citation metadata | For production retrieval validation |
| `validate_experiment_provenance` | Checks saved experiment rankings against their indexes and source maps | For experiment provenance validation |
| `verify_integration` | Checks that the Narrative Evidence Agent and retriever work together through the shared contracts | For integration testing |
| `prepare_experiment` | Freezes source inputs, code and configuration into a separate experimental workspace | Before a reproducible comparison |
| `prepare_evidence_labels` | Prepares draft questions and source-based evidence labels, including generated diagnostic cases | During experimental dataset preparation; labels require review |
| `validate_chunking` | Builds/checks chunking candidates for source coverage, provenance, token limits and reference mapping | Before comparing chunking strategies |
| `build_adaptive_chunks` | Builds C4 chunks by selecting candidates that satisfy hard constraints and receive configurable scores | When preparing adaptive chunking |
| `compare_chunking` | Compares C1-C4 using the same questions and BM25 retrieval | To evaluate chunking strategies |
| `compare_retrieval_priority` | Compares retrieval methods and document-priority policies on the experimental representation | To evaluate retrieval and source priority |
| `compare_canonical_92` | Applies canonical chunking rules to shared frozen sources and evaluates the development claims | To compare canonical representation with C4 under shared conditions |

The normal orchestrator runtime uses prepared company indexes, calls `NarrativeAgent.gather()`, and receives `EvidenceItem` objects. It does not loop through this dictionary. The basic_rag command and registry collector prepare cited evidence without stance/verdict. Historical verdict generation is retained in verdict_baseline. Acquisition, preparation and experimental comparisons are invoked separately when needed.

## LLM evidence preparation without stance or verdict

The Narrative Agent retains an LLM step for relevance selection, exact-source
facts and a cited evidence summary. The raw NarrativeEvidenceAgent class still
retrieves candidates without analysis; the registry adapter, basic_rag command
and collect_evidence CLI then invoke the same evidence_summary helper. No stance
label or claim verdict is generated. Detailed commands and current output schema
are in [summary_week_2.md, Section 10](summary_week_2.md#10-current-narrative-rag-output-cited-evidence-without-stance-or-verdict).

```text
SubClaim -> candidate retrieval and canonical provenance checks
         -> one LLM selection/fact/summary call
         -> Python validates selected IDs, exact quotes and summary references
         -> selected original EvidenceItems returned to orchestrator
         -> auditor owns the claim assessment
```

Use FINSIGHT_NARRATIVE_ANALYSIS=evidence (the registry default),
FINSIGHT_NARRATIVE_ANALYSIS_MAX_TOKENS=1500, and configure the shared LLM provider,
model and private key. Old stance/basic_rag analysis modes alias evidence with a
warning; they do not enable stance/verdict decisions. An older shell variable
FINSIGHT_NARRATIVE_ANALYSIS=retrieval still explicitly skips the LLM, so change
it to evidence when enabling preparation. Standalone basic_rag and the
collect_evidence CLI prepare evidence by default; --retrieval-only is the opt-out.

Ollama can be selected with the CLI --ollama-model flag or, for the registry
adapter, FINSIGHT_NARRATIVE_OLLAMA_MODEL. Optional local host/context settings
are FINSIGHT_NARRATIVE_OLLAMA_HOST and FINSIGHT_NARRATIVE_OLLAMA_CONTEXT. This
configures the Narrative helper, not Ollama support for all other shared agents.

The helper never reruns retrieval or rewrites source passages. Every selected
ID must come from the candidates and have an exact-source fact. Every summary
sentence cites selected IDs with facts. Both apparently confirming and
conflicting/qualifying relevant material should be retained. These relevance and
faithfulness instructions are model judgments, not guarantees from Python's
technical checks. All original candidates remain in the report for review.

Invalid/generation-error results retain the original candidate evidence and save
no accepted generated facts/summary. The collector logs the failure rather than
discarding sources. Successful reports record evidence_preparation_status=COMPLETED;
assessment_status=NOT_ASSESSED means the separate stance/verdict assessment has
not occurred, not that the LLM was skipped. Usage and validation are saved.

The shared return contract is still list[EvidenceItem]. Summaries/facts are
available in the saved report and last_analysis; they are not inserted into the
source content or passed to the auditor as a new interface field. The internal
mandatory NEUTRAL stance is an unassessed compatibility placeholder, omitted
from public evidence. The stub auditor/tier policy still rely on stance labels
and need their own assessment handoff update. The merged main workflow now uses
claim extraction/selection instead of the older default stub decomposer. Do not
restore the stale decomposer registry slot: it no longer matches AgentRegistry.
The single registry preserves main's XbrlFinancialAgent and AdviceBoundaryAgent
and selects the Narrative wrapper/retriever only for the Narrative-owned slots.
This implementation does not change shared schemas,
interfaces, the auditor, tier policy, or orchestrator renderer.

Reports remain in data/narrative/experiments/orchestrator_narrative/. New
standalone reports use ticker-model-evidence-time filenames. Historical verdict,
stance and retrieval-only outputs are preserved. The current main-merge repair
passed Ruff and the full local suite: 210 tests passed with 14 expected failures.
The corpus tooling now lives under narrative/corpus/; old pilot/catalog entry
points delegate there. Narrative retrieval uses the new shared company thesaurus.
Search logs record unassessed evidence and safely handle absent/empty indexes.
Narrative evidence preparation and its source
validation/fallback are tested with mocked responses; no paid API calls were
made. Dataset validation, graph freshness and the tracked-file secret guard passed.
The CI mock demo exits successfully but requests an influencer in the new main
workflow, rather than finishing an evidence assessment. Commit/push the repair
for a fresh remote CI matrix result.
