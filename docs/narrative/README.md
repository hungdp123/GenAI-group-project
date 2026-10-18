# Narrative pipeline documentation

Start with [Week 2 summary](workflow/summary_week_2.md) for the complete source-to-answer workflow, commands, experiment results and files to commit.

| Document | Use |
|---|---|
| [Script inventory](reference/narrative_script_inventory.md) | Purpose, inputs/outputs and current path of every Narrative Python module |
| [Narrative tools](reference/narrative_scripts.md) | All 19 tools under `scripts/narrative/`, their groups and command paths |
| [GitHub handoff](workflow/narrative_pipeline_handoff.md) | Files to push, external data requirements and explicit staging commands |
| [Module migration](reference/narrative_module_migration.md) | Old-to-new Python module paths after source regrouping |
| [Week 1 recap](workflow/recap_week_1.md) | Corpus acquisition and source processing |
| [Week 2 recap](workflow/recap_week_2.md) | Detailed execution status and evidence links |
| [Experimental design](experiments/narrative_rag_experimental_design.md) | Chunking, retrieval/priority and Basic RAG methodology and acceptance gates |
| [Narrative Evidence Agent](guides/narrative_evidence_agent.md) | Claim-level evidence gathering and application integration |
| [PDF/chart processing](guides/pdf_chart_processing.md) | Rich PDF extraction and promotion |
| [Corpus download guide](corpus/narrative_corpus_download_guide.md) | Document availability and acquisition details |
| [Retrieval status](corpus/narrative_corpus_retrieval_status.md) | Document eligibility and source/provenance issues |
| [Chunking/retrieval results](results/2026-10-18_chunking_retrieval_summary.md) | C1-C4 and R1-R4 x P0-P2 development results |
| [Basic RAG consistency results](results/2026-10-18_basic_narrative_rag_consistency.md) | Numerical verdict regression and local smoke outcomes |

## Folder contents

```text
docs/narrative/
    README.md
    workflow/                           weekly summaries, recaps and GitHub handoff
    guides/                             agent, Basic RAG and PDF/chart guides
    corpus/                             document coverage, downloads and eligibility
    reference/                          code inventories, schemas and migration map
    experiments/                        design and earlier META pilot notes
    evaluation/                         query JSONL, evidence, review and lock files
    config/                             C4 and document-priority configuration JSON
    results/                            Narrative result summaries and inventories
    examples/                           chart examples
    archive/                            earlier Narrative documents
```

## Documentation path change

Narrative documents and their supporting inputs are grouped under **`docs/narrative/`**. Use `workflow/summary_week_2.md` for the runbook, `evaluation/` for questions/references, and `config/` for experiment settings. Current code references, default config paths and documented commands use these grouped locations. For example:

```powershell
python -u -m finsight.agents.narrative.evaluation.run_retrieval_comparison --queries docs/narrative/evaluation/retrieval_queries_v2.jsonl --top-k 10 --candidate-k 50 --output-dir data/narrative/experiments/retrieval_comparison/handoff_v1
```

The JSON/JSONL/CSV inputs, lock records, configuration files and non-Markdown result files were moved without changing their bytes. Historical path strings inside those records retain their original provenance. Existing frozen snapshots under `data/` remain unchanged. The label-preparation runner supports all three query layouts: `docs/`, `docs/narrative/` and the current `docs/narrative/evaluation/`.

Shared project architecture, orchestrator documentation and general evaluation guides remain in the parent `docs/` folder. Corpus data and indexes remain under `data/narrative/` and are shared separately from GitHub.

Regrouping verification: **42 focused tests passed**, covering grouped command entry points, Basic RAG regressions, approved-query hash preservation and all combinations of historical/current frozen-query layouts. All active documentation links and current code references to Narrative inputs/configs resolve. All 10 relocated non-Markdown inputs were preserved byte-for-byte. Archived historical documents retain pre-existing unavailable references; use current guides for operational links. No processing/index builds, model requests, experiment reruns or Git staging/commit/push were performed for this documentation regrouping.
