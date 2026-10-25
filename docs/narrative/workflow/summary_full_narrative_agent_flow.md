# Full Narrative Agent workflow: source collection to RAG, evaluation and orchestrator integration

Related documentation: [source collection guide](workflow_a_source_collection.md), [Narrative script inventory](../reference/narrative_script_inventory.md), and [orchestrator connection](workflow_e_orchestrator_integration.md#orchestrator-handoff). Guides, configuration files and retained reports live under `docs/narrative/`; generated data and historical experiment inputs must be restored separately where required.

This guide covers source collection, processing, RAG, evaluation and orchestrator integration for the Narrative Agent. Recorded results retain their original snapshot/dataset scope. File descriptions are available in the [script inventory](../reference/narrative_script_inventory.md).

## 1. Full workflow

This section summarizes each workflow. Open its linked guide for detailed explanations, commands, prerequisites and outputs. Current evaluation status is summarized below; output fields and examples are in [Workflow C](workflow_c_rag.md#rag-output).

Follow **A -> B -> C -> D** for source collection, preparation, RAG execution and evaluation. Workflow E connects Narrative to the orchestrator. Step numbers restart at 1 inside each detailed guide.

Choose the workflow that matches the task:

| Workflow | When to run it | Where to start |
|---|---|---|
| **A. Collect source documents and create manifests** | You need to acquire sources, fill coverage gaps or restore the team's collected corpus. | [Workflow A guide](workflow_a_source_collection.md) |
| **B. Prepare the corpus and indexes** | Sources, parsing, chunking or embedding settings change, or a laptop needs matching artifacts restored. | [Workflow B guide](workflow_b_processing_and_indexes.md) |
| **C. Run RAG for a claim** | You want evidence, extracted facts and a cited summary for an input claim. Reuse existing matching indexes. | [Workflow C guide](workflow_c_rag.md) |
| **D. Evaluate RAG** | You want to compare configurations and measure retrieval/summary quality. | [Workflow D guide](workflow_d_evaluation.md) |
| **E. Connect the orchestrator and Narrative Agent** | You want the orchestrator to call Narrative, receive evidence and pass it to the Auditor, or test that connection. | [Workflow E guide](workflow_e_orchestrator_integration.md) |

Workflows A and B prepare the corpus; reuse matching existing sources/indexes
instead of repeating collection and processing for every claim. Workflow C runs
RAG for an individual claim without the test dataset or comparisons.
Workflow D evaluates recorded RAG inputs and outputs. Workflow E
connects the orchestrator to the Narrative processing in Workflow C; its integration
checks are run separately after configuration or interface changes.

### 1.1 Workflow A — Collect source documents and create manifests

```mermaid
flowchart LR
    Scope["Choose companies, document types and collection date range"] --> SEC["Download core SEC filings and earnings releases"]
    Scope --> Exhibits["Download supplemental SEC exhibits"]
    Scope --> IR["Download official investor-relations materials"]
    Scope --> Manual["Manually download missing official documents when needed"]
    SEC --> Raw["Save original files and create manifest records"]
    Exhibits --> Raw
    IR --> Raw
    Manual --> Metadata["Prepare and review manual source metadata"]
    Metadata --> Import["Validate and import reviewed files"]
    Import --> Raw
    Raw --> Review["Review company, type, dates, URLs, file hashes and duplicates"]
    Review --> Catalog["Generate and inspect the document catalog"]
    Catalog --> Ready["W2-A: collected sources and reviewed manifests ready for Workflow B"]
```

Collect or restore original source files, record their company, dates, URLs and file hashes in manifests, and review the document catalog. These reviewed files and manifests are the input to Workflow B.

[Workflow A details: explanations, scripts and instructions](workflow_a_source_collection.md).

### 1.2 Workflow B — Prepare searchable company evidence

```mermaid
flowchart LR
    Sources["Reviewed source files and manifests from Workflow A"] --> B["W2-B: load manifests and validate source files and date metadata"]
    B --> C["W2-C: parse native text into source blocks"]
    B --> D["W2-D: audit PDF pages when applicable"]
    D --> E["W2-E: extract native PDF text"]
    D --> F["W2-F: layout extraction and optional OCR"]
    F --> G["W2-G: tables and chart candidates"]
    G --> H["W2-H: optional chart vision extraction"]
    E --> I["W2-I: extraction reports and page coverage"]
    F --> I
    G --> I
    H --> I
    C --> J["W2-J: promote content, chunk and build BM25"]
    I --> J
    J --> V{"Dense, hybrid or R4 needed?"}
    V -->|Yes| K["W2-K: build matching embedding vectors"]
    V -->|BM25 only| Ready["Prepared artifacts ready for inference or evaluation"]
    K --> Ready
```

Parse source documents, enrich PDFs where needed, create chunks and build matching search indexes. Reuse existing complete builds; vectors are needed for dense, hybrid and R4 retrieval, while BM25 can run without them.

[Workflow B details: explanations, scripts and instructions](workflow_b_processing_and_indexes.md).

### 1.3 Workflow C — RAG for one claim

```mermaid
flowchart LR
    Index["Existing selected company chunks and indexes"] --> P
    O["W2-O: claim or SubClaim, company and publication cutoff"] --> P["W2-P: support and counter retrieval with the selected backend"]
    P --> Q["W2-Q: deduplicate and retain original source provenance"]
    Q --> R["W2-R: one LLM call selects evidence, extracts facts and writes a cited summary"]
    R --> Check["Python validates output structure, evidence IDs and exact source quotes"]
    Check --> Save["Save report: selected evidence, facts, summary, validation and usage"]
```

Retrieve and deduplicate candidate passages, then use one LLM call to select evidence, extract facts and write a cited summary. Python checks evidence IDs and exact quotes. Narrative does not assign stance labels or a claim verdict.

[Workflow C details: explanations, scripts and instructions](workflow_c_rag.md).

### 1.4 Workflow D — Evaluation and validation

```mermaid
flowchart LR
    Cases["126 claims and reviewed gold evidence"] --> Ready["Step 1: check claims, indexes and source mappings"]
    Builds["Existing C1-C4 full_v1 artifacts"] --> Ready
    Ready --> Run["Step 2: compare retrieval methods without generation"]
    Run --> Save["Save ranked support/counter passages and provenance"]
    Save --> Score["Step 3: validate sources and calculate retrieval metrics"]
    Score --> Select["Review results and select configurations"]
    Select --> RAG["Step 4: run selected configurations with the Workflow C RAG pipeline"]
    RAG --> CheckRAG["Step 5: validate outputs and assess generated evidence quality"]
```

Compare retrieval first using all 126 claims and C1-C4 x BM25/dense/hybrid/R4, without LLM generation. Validate and score the saved rankings, review the results, then run the Workflow C RAG pipeline for selected configurations. Use the offline scorer for technical checks and the separate `evaluate_ragas` script for evidence/summary quality. Both evaluation CLIs now default to local Ollama (`qwen3-vl:4b-instruct`); API use requires explicit opt-in. Workflow D includes one-claim and full-batch commands. No model runs were started for this update.

[Workflow D details: inputs, five steps, commands and current results](workflow_d_evaluation.md).

### 1.5 Workflow E — Orchestrator and Narrative Agent connection

```mermaid
flowchart LR
    Settings["Orchestrator settings and agent input/output formats"] --> Registry["Select the configured NarrativeAgent implementation"]
    Registry --> Call["Orchestrator calls NarrativeAgent.gather"]
    Input["SubClaim: ID, claim text, company, type; cutoff; optional feedback"] --> Call
    Call --> Narrative["Run the configured Narrative evidence flow from Workflow C"]
    Narrative --> Return["W2-T: return selected original EvidenceItems to the orchestrator"]
    Narrative --> Saved["Save facts, cited summary, validation and usage in the Narrative report"]
    Return --> Auditor["Auditor assesses the claim and determines its verdict"]
    Settings -.-> U["W2-U: Test the orchestrator–Narrative Agent connection"]
    U --> Checks["Check routing, input/output fields, company/cutoff handling and errors"]
    Checks --> IntegrationReport["Record integration-check results"]
```

The orchestrator supplies a SubClaim, cutoff and optional feedback to NarrativeAgent.gather(). Narrative reuses Workflow C and returns selected original EvidenceItems; the Auditor assesses the claim and determines its verdict. Integration checks test this connection separately from quality evaluation.

[Workflow E details: explanations, scripts and instructions](workflow_e_orchestrator_integration.md).

## 2. Execution map

Each workflow guide contains its script table, responsibilities and expected outputs:

| Workflow | Script details |
|---|---|
| A. Source collection and manifests | [Collection scripts](workflow_a_source_collection.md#scripts-and-outputs) |
| B. Processing and indexing | [Processing and index builders](workflow_b_processing_and_indexes.md#scripts-and-outputs) |
| C. Per-claim RAG | [Retrieval and evidence-summary entry points](workflow_c_rag.md#scripts-and-outputs) |
| D. Evaluation | [Comparison and validation scripts](workflow_d_evaluation.md#scripts-and-outputs) |
| E. Orchestrator integration | [Agent entry classes and connection checks](workflow_e_orchestrator_integration.md#scripts-and-outputs) |

For descriptions of individual scripts, see the [grouped script inventory](../reference/narrative_script_inventory.md).

### 2.1 Company document coverage

See [Workflow A: company document coverage](workflow_a_source_collection.md#company-document-coverage) for the 16-company source table, collection window and notes on document categories.

## 3. Before running

Run commands from the repository root in PowerShell. Examples use single lines so PowerShell continuation characters are unnecessary.

```powershell
python -m pip install -e ".[dev,retrieval,document-ai]"
```

Restore the team's raw corpus/manifests or its matching processed snapshot under `data/`. A Git clone does not include that ignored folder. See the [source collection guide](workflow_a_source_collection.md) and [acquisition helper inventory](../reference/narrative_script_inventory.md#6-acquisition-and-manual-import-7-files) when acquiring new sources; downloader arguments depend on the source and destination, so acquisition is not an unconditional rerun step.

For local LLM use, install the chosen Ollama model and start Ollama. Check installed names with `ollama list`. The recorded Basic RAG smoke model is `qwen3-vl:4b-instruct`; a vision-capable model is required for chart extraction. For OpenAI, configure `llm_provider`, `llm_model` and `openai_api_key` in local `config.yaml`; copy the format from `config.example.yaml`. Keep credentials local. Chart extraction can explicitly select its provider with `--openai-model` or `--ollama-model`.

If the processed snapshot is current and you only want to run a claim, go directly to [Workflow C's single-claim command](workflow_c_rag.md#standard-command-prepare-a-cited-evidence-summary). Steps 1-3 in the Workflow B guide write processing/index artifacts; run them when building or refreshing the corpus.

## 4. Build searchable company evidence

Parse, enrich, chunk and index sources when building or updating the corpus. Reuse matching artifacts for ordinary requests. See [Workflow B: processing commands and outputs](workflow_b_processing_and_indexes.md) and the [Workflow B: chunk storage and builds](workflow_b_processing_and_indexes.md#chunk-and-index-storage).

### 4.1 Saved chunking and embedding results

Read from the current saved company outputs in the current saved snapshot. This describes the **canonical company corpus used by Basic RAG**, after PDF/chart promotion; it is separate from the C1-C4 comparisons documented in [Workflow D](workflow_d_evaluation.md). No extraction, chunk build or embedding inference was rerun for this summary.

| Item | Saved result |
|---|---|
| Companies / source documents | 16 companies / 1,201 documents |
| Processed source blocks | 10,424, summed from the saved processing reports |
| Canonical chunk settings | Target 800 words / 120-word overlap budget, with existing structure/content handling; not the strict fixed-window experimental C1 arm |
| Generated chunks in `chunks.jsonl` | **23,856** |
| Duplicate chunks excluded from BM25 | **874** |
| Final BM25 passages | **22,982** |
| Saved Qdrant vectors | **22,982**, one per final indexed passage |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector dimension / similarity | **384** dimensions / cosine similarity |
| Embedding input limit | **256 tokens**, confirmed in the cached model configuration |
| BM25/vector metadata consistency | **16/16 companies match**: corpus fingerprints, ordered chunk IDs and passage/vector counts |

Generated chunks and indexed chunks differ because exact duplicate content is excluded during index construction. For example, META has 1,664 generated chunks, 84 excluded duplicates, and 1,580 final BM25 passages/vectors. Actual chunk lengths can be shorter than the configured target.

## 5. Run RAG for a claim

Provide a claim, company and publication cutoff. RAG retrieves source passages, uses an LLM to select relevant evidence and extract facts, and returns a cited summary. It does **not** assign passage stance or a claim verdict. No evaluation dataset is required. See [Workflow C](workflow_c_rag.md) for model and retrieval configuration.

### 5.1 Current command and output

Example using prepared C4 indexes and BM25. This illustrates configuration selection; it does not declare a winning method. Configure the shared LLM provider first, or add the Ollama flags below.

```powershell
python -u -m finsight.agents.narrative.rag.basic_rag --ticker MSFT --claim "Microsoft Azure Arc supports management across multiple clouds and on-premises environments." --cutoff 2024-07-30 --corpus-root data/narrative/processed/chunking_strategies/C4/full_v1 --backend bm25 --top-k 3 --candidate-k 50 --max-tokens 3000
```

| Setting / output | Meaning |
|---|---|
| `--corpus-root` | Selects the saved chunk strategy and matching company indexes; change C4 to C1, C2 or C3 to select another build |
| `--backend` | `bm25`, `dense`, `hybrid` or `r4`; dense/hybrid/R4 require matching vectors and model dependencies |
| `--top-k 3` | Up to three results per support/counter search before deduplication and LLM selection |
| `--candidate-k 50` | Candidate-pool setting used by the retrieval configuration; distinct from the returned top-K |
| Local LLM | Append `--ollama-model qwen3-vl:4b-instruct --ollama-context 20000` to use that installed Ollama model |
| `--max-tokens 3000` | Maximum output tokens for evidence preparation; does not change retrieval depth |
| Report contents | Selected evidence and provenance, exact-source facts, cited summary, citation validation and LLM usage |
| `--retrieval-only` | Explicitly skips LLM preparation; produces retrieval evidence without facts or a generated summary |

For R4, set `--rerank-overflow truncate` only when intentionally permitting cross-encoder input truncation; the default error policy rejects oversized pairs. Complete stored passages remain intact.

Omitting `--output` automatically saves a timestamped report under the renamed folder:

```text
data/narrative/experiments/standalone_rag/<TICKER>-<MODEL>-evidence-<TIMESTAMP>.json
```

The terminal prints the report path. [Workflow C: RAG output](workflow_c_rag.md#rag-output) explains its fields and Python checks. `status=OK` means technical acceptance, not independently verified answer accuracy.

## 6. Evaluate and validate RAG

The current evaluation uses **126 claims across 16 companies**, C1-C4 `full_v1` indexes and four retrieval methods: BM25, dense, hybrid and R4. P0 means no document-priority ranking adjustment. Evaluate retrieval first, then run LLM evidence preparation for selected configurations. See [Workflow D: commands and detailed results](workflow_d_evaluation.md).

### 6.1 Current retrieval evaluation status

Saved run: `data/narrative/evaluation/test_claims_v1_result/retrieval_all126_v1/`. Status: **complete and scored**.

| Measure | Latest saved result |
|---|---|
| Dataset | `data/narrative/evaluation/test_claims_v1/cases.csv`: 42 supported, 42 contradicting and 42 unsupported claims |
| Completed executions | **2,016 / 2,016 (100%)**, all with saved retrieval status `OK` |
| Completed configurations | **16 / 16** complete and scored: C1-C4 with all four methods |
| Remaining configurations | None |
| Source validation | **2,016 / 2,016 passed** across all configurations |
| Retrieval metrics | Source Hit@5/10, known-reference Recall@5/10 and MRR@5/10, reported separately for support and counter searches |
| Metric scope | 84 supported/contradicting claims per configuration; positive-reference metrics are N/A for the 42 unsupported claims |
| LLM usage | **Zero tokens and $0 API cost**: this run uses `--retrieval-only` |
| Summary quality / RAGAS | Not evaluated by this retrieval-only run |

Detailed scores are in [Workflow D's current results](workflow_d_evaluation.md#current-126-claim-retrieval-only-results). The saved matrix is `COMPLETE`, and the full comparison report is scored. Retrieval scores measure recovery of annotated source evidence, not generated-summary accuracy. Alternative valid evidence and the draft annotations still need review.

The retrieval comparison is complete. The command below reproduces or resumes the same configuration from the repository root; use one runner per output folder:

```powershell
$retrievalRun = 'data/narrative/evaluation/test_claims_v1_result/retrieval_all126_v1'
python -u -m scripts.narrative.evaluation.run_rag_comparison --all-cases --strategies C1 C2 C3 C4 --backends bm25 dense hybrid r4 --build-version full_v1 --threads 2 --rerank-overflow truncate --retrieval-only --output-dir $retrievalRun --resume
```

To regenerate the completed comparison report from saved outputs:

```powershell
python -u -m scripts.narrative.evaluation.evaluate_rag_comparison --run-dir $retrievalRun
```

The final reports are `comparison.md`, `comparison.json` and the comparison/per-case/per-company CSV files. Earlier `partial_comparison.*` files retain their historical progress scope. The evaluator reads saved outputs; it does not repeat retrieval or call an LLM.

### 6.2 RAG and integration validation status

| Check | Current status / next step |
|---|---|
| Retrieval provenance | Completed configurations pass checks against their selected indexes and source mappings; see Section 6.1 |
| LLM facts and cited summaries | Saved C1/BM25 and C4/BM25 batches cover all 126 claims, including recorded generation failures. See [Workflow D: selected-run performance](workflow_d_evaluation.md#selected-cases-retrieval-and-ragas-performance) |
| Strict citation validation | Python checks selected IDs, source spans/quotes and output structure during RAG; a technical pass does not establish summary accuracy |
| Summary evaluation | Separate `scripts/narrative/evaluation/evaluate_ragas.py` scores saved facts, summaries, selected evidence and citations, using local Ollama by default. See [Workflow D: full local RAGAS flow](workflow_d_evaluation.md#full-local-flow-generate-and-evaluate-with-ragas). No model evaluation was run for this update. |
| Orchestrator connection | Separate interface/routing checks in [Workflow E](workflow_e_orchestrator_integration.md); the Auditor determines the claim verdict |

### 6.3 Run RAG generation and RAGAS evaluation locally

Run from the repository root in PowerShell, with Ollama running and `qwen3-vl:4b-instruct` installed. Inputs are the current `cases.csv`, parsed documents and matching C4 `full_v1` indexes/vectors. These commands use local Ollama for generation and judging; no OpenAI key is required.

**1. Prepare dependencies and output folders.**

```powershell
python -m pip install -e ".[retrieval,ragas]"
$ragRun = 'data/narrative/evaluation/test_claims_v1_result/rag_C4_r4_ollama_v1'
$judgeRun = 'data/narrative/evaluation/test_claims_v1_result/ragas_C4_r4_ollama_v1'
```

**2. Generate evidence, facts and cited summaries for all 126 claims using C4/R4.**

```powershell
python -u -m scripts.narrative.evaluation.run_rag_comparison --all-cases --strategies C4 --backends r4 --build-version full_v1 --threads 2 --max-tokens 3000 --rerank-overflow truncate --output-dir $ragRun --resume
```

**3. Score saved retrieval results and technical validation, then run RAGAS.**

```powershell
python -u -m scripts.narrative.evaluation.evaluate_rag_comparison --run-dir $ragRun
python -u -m scripts.narrative.evaluation.evaluate_ragas --run-dir $ragRun --all-cases --max-tokens 3000 --output-dir $judgeRun --resume
```

RAGAS reads the saved generation outputs; it does not repeat retrieval or generation. To evaluate an existing RAG run, set `$ragRun` to that folder and run Step 3.

**4. Read the reports.**

```powershell
Get-Content -LiteralPath "$ragRun/comparison.md"
Get-Content -LiteralPath "$judgeRun/ragas_report.md"
```

For a one-claim preview, replace `--all-cases` with `--limit 1` in both generation and judging commands. Use `--resume` only with the same inputs and settings.

See [Workflow D for detailed performance](workflow_d_evaluation.md#selected-cases-retrieval-and-ragas-performance) and the [full local evaluation instructions](workflow_d_evaluation.md#full-local-flow-generate-and-evaluate-with-ragas). The existing C1/BM25 and C4/BM25 RAGAS results used OpenAI models; they are separate from this local C4/R4 run.
