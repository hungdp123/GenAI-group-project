# Narrative tools under scripts/narrative

This folder contains batch builders, evaluation runners and maintenance tools. Reusable Narrative Agent implementation lives in `finsight/agents/narrative/`.

There are **15 implementation/helper modules and 6 package initializers (21 Python files)**. For execution order, use the [full workflow](../workflow/summary_full_narrative_agent_flow.md), [Workflow B](../workflow/workflow_b_processing_and_indexes.md) and [Workflow D](../workflow/workflow_d_evaluation.md).

## Current groups

| Folder | Purpose | Modules excluding initializers |
|---|---|---:|
| `retrieval/` | Build full C1-C4 chunks, BM25 indexes and matching vectors. | 3 |
| `evaluation/` | Run the current CSV comparisons, score saved outputs and evaluate summaries with RAGAS. | 7 |
| `datasets/` | Source-span mapping and shared label/file helpers. | 2 |
| `experiments/` | Snapshot preparation and shared identity/JSON helpers. | 1 |
| `maintenance/` | Optional source recovery and Microsoft presentation conversion. | 2 |

## Index builders

| File | Responsibility |
|---|---|
| [retrieval/build_full_chunking_indexes.py](../../../scripts/narrative/retrieval/build_full_chunking_indexes.py) | Main full-corpus C1-C4 builder. Creates strategy-specific chunks, source maps, BM25 indexes, vectors and reports. Delegates C4 to its existing builder. |
| [retrieval/build_full_c4_indexes.py](../../../scripts/narrative/retrieval/build_full_c4_indexes.py) | Full-corpus adaptive C4 builder; also supplies helpers imported by the C1-C4 builder. |
| [retrieval/c4_storage.py](../../../scripts/narrative/retrieval/c4_storage.py) | Shared chunk/source-map exports, output-path validation and build-resume identity checks. Required by both builders. |

Build commands and artifact locations are in [Workflow B](../workflow/workflow_b_processing_and_indexes.md#chunk-and-index-storage). These builders do not calculate retrieval-quality scores.

## Current evaluation

| File | Responsibility |
|---|---|
| [evaluation/run_rag_comparison.py](../../../scripts/narrative/evaluation/run_rag_comparison.py) | Main current CSV batch runner. Supports retrieval-only comparisons or full RAG evidence preparation across selected strategies/backends. |
| [evaluation/run_rag_pilot.py](../../../scripts/narrative/evaluation/run_rag_pilot.py) | Shared per-configuration execution and trace capture used by the batch runner; also offers a small standalone pilot. Keep despite the pilot filename. |
| [evaluation/evaluate_rag_comparison.py](../../../scripts/narrative/evaluation/evaluate_rag_comparison.py) | Scores saved matrix outputs, source/citation validity, retrieval metrics and recorded usage without new retrieval or model calls. |
| [evaluation/evaluate_rag_pilot.py](../../../scripts/narrative/evaluation/evaluate_rag_pilot.py) | Shared CSV loading, source-span scoring and report helpers used by the comparison and RAGAS runners. Keep despite the pilot filename. |
| [evaluation/evaluate_ragas.py](../../../scripts/narrative/evaluation/evaluate_ragas.py) | Prepares saved RAG outputs for judging and reports grounding, evidence/summary quality, coverage and judge usage separately from generation. |
| [evaluation/model_options.py](../../../scripts/narrative/evaluation/model_options.py) | Shared local-model defaults and explicit shared-provider CLI selection. |
| [evaluation/resume.py](../../../scripts/narrative/evaluation/resume.py) | Validates checkpoint compatibility and archives explicitly retried generation attempts. |

The current dataset is `data/narrative/evaluation/test_claims_v1/cases.csv`, with 126 claims across 16 companies. Run instructions and saved performance are in [Workflow D](../workflow/workflow_d_evaluation.md).

Generation and RAGAS CLIs default to local Ollama. API mode requires explicit selection; see Workflow D. A retrieval-only comparison does not generate facts/summaries or run RAGAS.

## Shared mapping and preparation helpers

| File | Responsibility |
|---|---|
| [datasets/narrative_reference_mapping.py](../../../scripts/narrative/datasets/narrative_reference_mapping.py) | Checks whether retrieved source intervals cover annotated evidence spans. Current retrieval scoring imports `covers`. |
| [datasets/prepare_narrative_labels.py](../../../scripts/narrative/datasets/prepare_narrative_labels.py) | Retained label-draft utility and shared JSONL helper imported by source mapping. Not required as a separate command for the current CSV. |
| [experiments/prepare_narrative_experiment.py](../../../scripts/narrative/experiments/prepare_narrative_experiment.py) | Retained snapshot utility and identity/JSON helpers imported by the label module. Not required as a separate command for the current CSV. |

Keep all three: current scoring imports source mapping, which imports the label helper, which imports experiment helpers. The retained label/snapshot CLIs concern their existing experiment inputs; they do not create or replace the current 126-claim CSV automatically.

## Optional source maintenance

| File | Responsibility |
|---|---|
| [maintenance/recover_narrative_sources.py](../../../scripts/narrative/maintenance/recover_narrative_sources.py) | Stages missing public sources and checks exact-hash source URLs; applying reviewed results can update manifests. |
| [maintenance/convert_msft_ir_presentations_to_pdf.py](../../../scripts/narrative/maintenance/convert_msft_ir_presentations_to_pdf.py) | Exports Microsoft IR PPTX decks to PDF and updates manifest paths. Requires Windows PowerPoint and pywin32. |

These are corpus-maintenance tools, not per-claim inference steps. The Microsoft conversion script runs its work directly; do not assume `--help` prevents writes. See [Workflow A](../workflow/workflow_a_source_collection.md) for source collection and manifest review.

Package `__init__.py` files keep these folders importable. Maintenance, builders and current evaluation are separate responsibilities; not running a helper directly does not make it unused.
