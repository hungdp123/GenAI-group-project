# Narrative tools under scripts/narrative

Updated: 19 October 2026. There are **19 Narrative tools** under `scripts/narrative/`: 18 moved tools and the new 92-claim canonical comparison runner. Shared `scripts/check.py` remains in place.

## Groups

| Folder | Purpose | Tools |
|---|---|---:|
| `experiments/` | Freeze reproducible all-company experiment foundations and historical baselines. | 2 |
| `datasets/` | Prepare source-grounded draft questions/labels and remap evidence using source spans. | 2 |
| `chunking/` | Validate C1-C4, build/inspect adaptive plans and compare chunking retrieval quality. | 4 |
| `retrieval/` | Compare R1-R4 with source-priority policies and canonical chunking on shared development labels. | 2 |
| `diagnostics/` | Audit corpus coverage, preview parsing and summarize document availability. | 3 |
| `maintenance/` | Convert Microsoft decks and recover/merge source documents; run only when required. | 4 |
| `pilots/` | Reproduce the earlier META-only chunking diagnostic, not the all-company benchmark. | 2 |

## Complete file and command map

The old filename column refers to its former location directly under `scripts/`. Run current command modules from the repository root with `python -m <new module>`. `datasets/narrative_reference_mapping.py` is an imported helper, not a standalone workflow command.

| New file | Old filename | New Python module |
|---|---|---|
| [experiments/prepare_narrative_experiment.py](../../../scripts/narrative/experiments/prepare_narrative_experiment.py) | `prepare_narrative_experiment.py` | `scripts.narrative.experiments.prepare_narrative_experiment` |
| [experiments/freeze_chunking_baseline.py](../../../scripts/narrative/experiments/freeze_chunking_baseline.py) | `freeze_chunking_baseline.py` | `scripts.narrative.experiments.freeze_chunking_baseline` |
| [datasets/prepare_narrative_labels.py](../../../scripts/narrative/datasets/prepare_narrative_labels.py) | `prepare_narrative_labels.py` | `scripts.narrative.datasets.prepare_narrative_labels` |
| [datasets/narrative_reference_mapping.py](../../../scripts/narrative/datasets/narrative_reference_mapping.py) | `narrative_reference_mapping.py` | `scripts.narrative.datasets.narrative_reference_mapping` |
| [chunking/validate_narrative_chunking.py](../../../scripts/narrative/chunking/validate_narrative_chunking.py) | `validate_narrative_chunking.py` | `scripts.narrative.chunking.validate_narrative_chunking` |
| [chunking/build_adaptive_narrative_chunks.py](../../../scripts/narrative/chunking/build_adaptive_narrative_chunks.py) | `build_adaptive_narrative_chunks.py` | `scripts.narrative.chunking.build_adaptive_narrative_chunks` |
| [chunking/inspect_adaptive_chunking.py](../../../scripts/narrative/chunking/inspect_adaptive_chunking.py) | `inspect_adaptive_chunking.py` | `scripts.narrative.chunking.inspect_adaptive_chunking` |
| [chunking/compare_narrative_chunking.py](../../../scripts/narrative/chunking/compare_narrative_chunking.py) | `compare_narrative_chunking.py` | `scripts.narrative.chunking.compare_narrative_chunking` |
| [retrieval/compare_narrative_retrieval_priority.py](../../../scripts/narrative/retrieval/compare_narrative_retrieval_priority.py) | `compare_narrative_retrieval_priority.py` | `scripts.narrative.retrieval.compare_narrative_retrieval_priority` |
| [retrieval/compare_canonical_development.py](../../../scripts/narrative/retrieval/compare_canonical_development.py) | New tool | `scripts.narrative.retrieval.compare_canonical_development` |
| [diagnostics/audit_week1_corpus.py](../../../scripts/narrative/diagnostics/audit_week1_corpus.py) | `audit_week1_corpus.py` | `scripts.narrative.diagnostics.audit_week1_corpus` |
| [diagnostics/preview_narrative_parser.py](../../../scripts/narrative/diagnostics/preview_narrative_parser.py) | `preview_narrative_parser.py` | `scripts.narrative.diagnostics.preview_narrative_parser` |
| [diagnostics/summarize_narrative_document_coverage.py](../../../scripts/narrative/diagnostics/summarize_narrative_document_coverage.py) | `summarize_narrative_document_coverage.py` | `scripts.narrative.diagnostics.summarize_narrative_document_coverage` |
| [maintenance/convert_msft_ir_presentations_to_pdf.py](../../../scripts/narrative/maintenance/convert_msft_ir_presentations_to_pdf.py) | `convert_msft_ir_presentations_to_pdf.py` | `scripts.narrative.maintenance.convert_msft_ir_presentations_to_pdf` |
| [maintenance/recover_narrative_sources.py](../../../scripts/narrative/maintenance/recover_narrative_sources.py) | `recover_narrative_sources.py` | `scripts.narrative.maintenance.recover_narrative_sources` |
| [maintenance/merge_narrative_coverage_recovery.py](../../../scripts/narrative/maintenance/merge_narrative_coverage_recovery.py) | `merge_narrative_coverage_recovery.py` | `scripts.narrative.maintenance.merge_narrative_coverage_recovery` |
| [maintenance/finalize_narrative_coverage_recovery.py](../../../scripts/narrative/maintenance/finalize_narrative_coverage_recovery.py) | `finalize_narrative_coverage_recovery.py` | `scripts.narrative.maintenance.finalize_narrative_coverage_recovery` |
| [pilots/build_meta_chunking_pilots.py](../../../scripts/narrative/pilots/build_meta_chunking_pilots.py) | `build_meta_chunking_pilots.py` | `scripts.narrative.pilots.build_meta_chunking_pilots` |
| [pilots/compare_meta_chunking.py](../../../scripts/narrative/pilots/compare_meta_chunking.py) | `compare_meta_chunking.py` | `scripts.narrative.pilots.compare_meta_chunking` |

## Main experiment sequence

Use a new run ID for a new experiment foundation. Existing immutable versions should be inspected with their documented verification commands.

```powershell
$narrativeRunId = "handoff_all16_v1"
python -u -m scripts.narrative.experiments.prepare_narrative_experiment --all --run-id $narrativeRunId
python -u -m scripts.narrative.datasets.prepare_narrative_labels --run-id $narrativeRunId --dataset-id evidence_labels_v3
python -u -m scripts.narrative.chunking.validate_narrative_chunking --run-id $narrativeRunId --dataset-id evidence_labels_v3 --validation-id technical_v1
python -u -m scripts.narrative.chunking.build_adaptive_narrative_chunks --run-id $narrativeRunId --dataset-id evidence_labels_v3 --validation-id c4_v1 --config docs/narrative/config/narrative_c4_config_v1.json
python -u -m scripts.narrative.chunking.compare_narrative_chunking --run-id $narrativeRunId --dataset-id evidence_labels_v3 --comparison-id bm25_dev_v1 --allow-draft
python -u -m scripts.narrative.retrieval.compare_narrative_retrieval_priority --run-id $narrativeRunId --experiment-id c4_dev_v1 --allow-draft --resume
```

Draft-label runs remain diagnostics, not accepted final benchmarks. See the [Week 2 summary](../workflow/summary_week_2.md) for results, prerequisites, review gates and current limitations.

## Relationship to the reusable implementation

### Compare canonical chunking against C4 on all 92 development claims

For the existing frozen foundation and labels, run:

```powershell
python -u -m scripts.narrative.retrieval.compare_canonical_development --run-id 20261018_all16_v1 --experiment-id canonical92_v1 --allow-draft --overflow-policy truncate --resume
```

This evaluates 92 claims x four retrieval methods with P0, using canonical 800/120 chunking rules on the same frozen parsed sources as C4. Outputs go to `data/narrative/experiments/narrative_rag/20261018_all16_v1/retrieval_canonical/canonical92_v1/`. The explicit overflow policy permits and records model truncation; full source passages remain stored. Without it, oversized embedding inputs/reranker pairs are rejected. This does not reuse the historical 16-claim parser outputs. See the [implementation report](../results/2026-10-19_canonical92_runner.md).

### Shared responsibilities

These tools coordinate datasets, snapshots, audits and reports. Reusable algorithms and application interfaces remain in `finsight/agents/narrative/`. Commit both the relevant tool packages and the implementation, including their `__init__.py` files and helper dependencies.

Run from the repository root using `python -m scripts.narrative.<group>.<name>`. Old flat `scripts.<name>` commands have been replaced; there is no second copy of each tool at its old path. The snapshot preparer now includes nested Python tools when freezing code. Existing frozen data/code snapshots, labels and saved result hashes were not rewritten. Use a new experiment version when code-identity checks reject changed source.

Some specialist tools have no argument parser: META pilots, coverage finalization/merge, availability reporting and Microsoft conversion run their work when invoked; do not assume `--help` prevents it. Microsoft conversion requires Windows PowerPoint and `pywin32`. Maintenance tools can write manifests/files; they are not routine inference steps.

The historical module-path report covers `finsight/agents/narrative/`; this page covers the separate tools move. Historical result reports and archives may contain originally executed paths; use this page and current runbooks for new commands.

## Verification after the move

- All 18 moved Narrative tools are present under the seven groups above; the 92-claim canonical runner was subsequently added as tool 19. Only shared `check.py` and the package initializer remain as Python files directly under `scripts/`.
- Imports between tools and tests use the grouped module paths. Source paths used to copy/hash experiment code were updated; the snapshot preparer includes nested Python tools.
- The source-recovery tool's repository-root calculation was adjusted for its deeper directory.
- **38 focused tests passed**, covering tool/module command entry points, nested snapshot source selection, historical/current frozen-query layouts, query hash preservation and Basic RAG numerical regressions.
- Active documentation links resolve; focused test lint and `git diff --check` passed.

No source acquisition, document processing, corpus/index rebuilding, model request, experiment rerun, Git staging, commit or push was performed for this regrouping.
