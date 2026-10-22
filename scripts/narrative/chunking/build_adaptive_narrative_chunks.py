"""Execute isolated C4 plan selection, independent validation and span mapping."""
from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from finsight.agents.narrative.corpus.processing.adaptive_chunking import AdaptiveConfig, choose_plan, context_spans
from finsight.agents.narrative.corpus.processing.chunk import NarrativeChunk
from finsight.agents.narrative.corpus.processing.corpus import file_sha256
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock
from scripts.narrative.datasets.narrative_reference_mapping import covers, map_references
from scripts.narrative.experiments.prepare_narrative_experiment import identity, write_json
from scripts.narrative.datasets.prepare_narrative_labels import jsonl, verify as verify_labels, write_jsonl
from scripts.narrative.chunking.validate_narrative_chunking import sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261018_all16_v1")
    parser.add_argument("--dataset-id", default="evidence_labels_v3")
    parser.add_argument("--validation-id", default="c4_v1")
    parser.add_argument("--fixed-validation-id", default="technical_v1")
    parser.add_argument("--config", type=Path, default=Path("docs/narrative/config/narrative_c4_config_v1.json"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) for value in (args.run_id, args.dataset_id, args.validation_id, args.fixed_validation_id)):
        parser.error("Invalid version ID")
    root = Path.cwd().resolve()
    run = root / "data/narrative/experiments/narrative_rag" / args.run_id
    output = run / "validation" / args.validation_id
    fixed = run / "validation" / args.fixed_validation_id
    dataset = run / "labels" / args.dataset_id
    if args.verify:
        manifest = json.loads((output / "validation_manifest.json").read_text(encoding="utf-8"))
        failures = [f["path"] for f in manifest["files"] if not (output / f["path"]).is_file() or file_sha256(output / f["path"]) != f["sha256"]]
        print(json.dumps({"status": "verified" if not failures else "failed", "files_checked": len(manifest["files"]), "failures": failures}, indent=2))
        raise SystemExit(bool(failures))
    if output.exists():
        parser.error("Version exists; use --verify or a new --validation-id")
    started = time.monotonic()
    foundation = json.loads((run / "foundation_manifest.json").read_text(encoding="utf-8"))
    fixed_config = json.loads((fixed / "configuration.json").read_text(encoding="utf-8"))
    fixed_report = json.loads((fixed / "technical_validation.json").read_text(encoding="utf-8"))
    if fixed_config["foundation_build_identity"] != foundation["build_identity"] or fixed_config["labels_manifest_sha256"] != file_sha256(dataset / "dataset_manifest.json"):
        raise RuntimeError("Fixed/C4 source or label builds differ")
    if verify_labels(dataset, run)["failures"]:
        raise RuntimeError("Draft label integrity failed")
    fixed_manifest = json.loads((fixed / "validation_manifest.json").read_text(encoding="utf-8"))
    if any(file_sha256(fixed / f["path"]) != f["sha256"] for f in fixed_manifest["files"]):
        raise RuntimeError("Preserved fixed-plan artifacts changed")
    limit = fixed_config["effective_token_limit"]
    config = AdaptiveConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    config.validate(limit)
    output.mkdir(parents=True)
    shutil.copytree(fixed / "tokenizer", output / "tokenizer")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output / "tokenizer", local_files_only=True, use_fast=True)
    code_paths = [Path(__file__).resolve(), root / "finsight/agents/narrative/corpus/processing/adaptive_chunking.py", root / "scripts/narrative/datasets/narrative_reference_mapping.py"]
    configuration = {"adaptive": asdict(config), "foundation_build_identity": foundation["build_identity"],
        "labels_manifest_sha256": file_sha256(dataset / "dataset_manifest.json"),
        "fixed_configuration_sha256": file_sha256(fixed / "configuration.json"),
        "fixed_report_sha256": file_sha256(fixed / "technical_validation.json"),
        "tokenizer_revision": fixed_config["cached_revision"], "effective_token_limit": limit,
        "embedding_input": "literal source context plus literal body; 2 special tokens; no truncation or synthetic financial facts",
        "code_hashes": {p.relative_to(root).as_posix(): file_sha256(p) for p in code_paths},
        "selection_inputs": "source blocks and configuration only; no questions, labels, rankings or heldout performance",
        "coherence": "lexical adjacent-unit cosine proxy; not semantic entailment; no model inference",
        "weights_status": "initial equal weights, not tuned or claimed optimal",
        "candidate_scoring": "only complete plans passing content preservation, traceability and model-token hard rules"}
    write_json(output / "configuration.json", configuration)
    all_questions = {s: jsonl(dataset / s / "questions.jsonl") for s in ("dev", "heldout")}
    all_evidence = {s: jsonl(dataset / s / "evidence_labels.jsonl") for s in ("dev", "heldout")}
    summaries = []
    for ticker in foundation["companies"]:
        inputs = run / "parsed" / ticker
        source_manifest = json.loads((inputs / "input_manifest.json").read_text(encoding="utf-8"))
        if file_sha256(inputs / "documents.jsonl") != source_manifest["blocks_sha256"]:
            raise RuntimeError("Frozen parsed input changed")
        sources = {s["document_id"]: s for s in source_manifest["sources"]}
        for s in sources.values():
            if file_sha256(Path(s["local_path"])) != s["sha256"]:
                raise RuntimeError("Frozen raw source hash mismatch")
        rows = jsonl(inputs / "documents.jsonl")
        destination = output / "arms/C4" / ticker
        destination.mkdir(parents=True)
        generated, source_maps, metrics, overflow, boundary_flags, failures, provenance_errors = [], [], [], [], [], [], []
        selections, selection_policies, candidate_counts = Counter(), Counter(), Counter()
        source_words = covered_words = source_chars = covered_chars = repeated_chars = 0
        with gzip.open(destination / "candidate_scores.jsonl.gz", "wt", encoding="utf-8") as trace:
            for line, row in enumerate(rows, 1):
                block = ProcessedBlock(**row)
                source = sources[block.document_id]
                source_words += len(block.text.split())
                source_chars += len(block.text)
                pieces, decision = choose_plan(block, tokenizer, limit, config)
                candidate_counts.update("hard_valid" if c["hard_valid"] else "hard_rejected" for c in decision["candidates"])
                trace.write(json.dumps({"document_id": block.document_id, "block_line_1based": line, **decision}, ensure_ascii=False) + "\n")
                if pieces is None:
                    failures.append({"document_id": block.document_id, "block_line_1based": line, "reason": "no_hard_valid_plan", "source_retained": str(inputs / "documents.jsonl")})
                    continue
                selections[decision["selected_candidate"]] += 1
                selected_record = next(c for c in decision["candidates"] if c["candidate_id"] == decision["selected_candidate"])
                selection_policies[selected_record["boundary_policy"]] += 1
                # Independent literal reconstruction checks selected plans, not just
                # the selector's self-reported hard-rule booleans.
                body = "".join(block.text[p["body_start"]:p["body_end"]] for p in pieces)
                if body != block.text or pieces[0]["body_start"] != 0 or pieces[-1]["body_end"] != len(block.text) or any(a["body_end"] != b["body_start"] for a, b in zip(pieces, pieces[1:])):
                    raise RuntimeError("C4 independent source reconstruction failed")
                covered_words += len(block.text.split())
                covered_chars += len(block.text)
                required = context_spans(block)
                for ordinal, p in enumerate(pieces, 1):
                    prefix = "\n\n".join(block.text[a:z] for a, z in p["context_spans"])
                    expected_text = (prefix + "\n\n" if prefix else "") + block.text[p["body_start"]:p["body_end"]]
                    if expected_text != p["text"]:
                        raise RuntimeError("C4 source-context traceability failed")
                    chunk_id = f"{block.document_id}:c4:block{line:06d}:{ordinal:04d}:{sha(p['text'])[:12]}"
                    chunk = NarrativeChunk(**{**row, "chunk_id": chunk_id, "text": p["text"], "text_sha256": sha(p["text"])})
                    for key in ("company_ticker", "document_type", "publication_date", "source_url"):
                        if getattr(chunk, key) != source[key] or chunk.document_id != source["document_id"]:
                            provenance_errors.append({"chunk_id": chunk_id, "field": key})
                    intervals = p["context_spans"] + [[p["body_start"], p["body_end"]]]
                    repeated_chars += sum(z-a for a, z in p["context_spans"])
                    generated.append(chunk.to_json())
                    source_maps.append({"chunk_id": chunk_id, "document_id": block.document_id, "source_sha256": source["sha256"],
                        "parsed_file": (inputs / "documents.jsonl").relative_to(run).as_posix(), "parsed_file_sha256": source_manifest["blocks_sha256"],
                        "block_line_1based": line, "block_text_sha256": sha(block.text),
                        "source_char_start": p["body_start"], "source_char_end": p["body_end"], "source_char_spans": intervals,
                        "context_char_spans": p["context_spans"], "body_char_span": [p["body_start"], p["body_end"]],
                        "normalization": "literal body and literal repeated source context joined with blank lines",
                        "page": block.page, "section": block.section, "citation_span": block.citation_span, "chart_id": block.chart_id})
                    missing = [[a, z] for a, z in required if not covers(block.text, a, z, intervals)]
                    if missing:
                        boundary_flags.append({"chunk_id": chunk_id, "issue": "source_context_requires_external_bundle", "source_context_spans": missing, "block_line_1based": line})
                    for end in [p["body_start"], p["body_end"]]:
                        if 0 < end < len(block.text) and not (block.text[end-1:end].isspace() or block.text[end:end+1].isspace()):
                            boundary_flags.append({"chunk_id": chunk_id, "issue": "midword_character_fallback", "source_char": end})
                if line % 1000 == 0:
                    print(f"{ticker}: selected {line}/{len(rows)} block plans", flush=True)
        # Batch recount actual final text independently of cached selector counts.
        for start in range(0, len(generated), 256):
            batch = generated[start:start+256]
            encoded = tokenizer([c["text"] for c in batch], add_special_tokens=True, truncation=False, padding=False, verbose=False)
            for c, tokens in zip(batch, encoded["input_ids"]):
                item = {"chunk_id": c["chunk_id"], "words": len(c["text"].split()), "tokens_including_special": len(tokens),
                        "effective_limit": limit, "token_limit_pass": len(tokens) <= limit, "content_type": c["content_type"]}
                metrics.append(item)
                if len(tokens) > limit:
                    overflow.append(item)
        if len({c["chunk_id"] for c in generated}) != len(generated):
            raise RuntimeError("C4 duplicate chunk IDs")
        write_jsonl(destination / "chunks.jsonl", generated)
        write_jsonl(destination / "source_map.jsonl", source_maps)
        write_jsonl(destination / "chunk_metrics.jsonl", metrics)
        write_jsonl(destination / "token_overflow.jsonl", overflow)
        write_jsonl(destination / "boundary_flags.jsonl", boundary_flags)
        write_jsonl(destination / "quarantine.jsonl", failures)
        mapping = map_references(rows=rows, sources=sources, source_maps=source_maps, all_questions=all_questions,
            all_evidence=all_evidence, output=output, arm="C4", ticker=ticker)
        summary = {"ticker": ticker, "arm": "C4", "documents": len(sources), "source_blocks": len(rows), "chunks": len(generated),
            "source_word_occurrences": source_words, "covered_source_word_occurrences": covered_words,
            "source_characters": source_chars, "covered_source_characters": covered_chars, "repeated_context_characters": repeated_chars,
            "overflow_chunks": len(overflow), "max_tokens": max((m["tokens_including_special"] for m in metrics), default=0),
            "provenance_errors": len(provenance_errors), "quarantined_blocks": len(failures), "boundary_flags": len(boundary_flags),
            "context_bundle_required_chunks": sum(b["issue"] == "source_context_requires_external_bundle" for b in boundary_flags),
            "candidate_counts": dict(candidate_counts), "selected_configurations": dict(selections), "selected_boundary_policies": dict(selection_policies),
            "evidence_mapping": dict(mapping), "hard_rules_pass": covered_chars == source_chars and not overflow and not provenance_errors and not failures}
        write_json(destination / "validation.json", summary)
        summaries.append(summary)
        print(f"{ticker}: C4 {len(generated)} chunks; {covered_words}/{source_words} source words; {len(overflow)} oversized; {mapping['ambiguity_queue']} unresolved spans", flush=True)
    totals = {k: sum(s[k] for s in summaries) for k in ("documents", "source_blocks", "chunks", "source_word_occurrences", "covered_source_word_occurrences", "source_characters", "covered_source_characters", "repeated_context_characters", "overflow_chunks", "provenance_errors", "quarantined_blocks", "boundary_flags", "context_bundle_required_chunks")}
    totals["max_tokens"] = max(s["max_tokens"] for s in summaries)
    totals["hard_rules_pass"] = all(s["hard_rules_pass"] for s in summaries)
    totals["evidence_mapping"] = {k: sum(s["evidence_mapping"].get(k, 0) for s in summaries) for k in ("contained_in_one_or_more_chunks", "requires_chunk_bundle", "unresolved", "ambiguity_queue", "question_bundles", "complete_question_bundles")}
    totals["candidate_counts"] = {k: sum(s["candidate_counts"].get(k, 0) for s in summaries) for k in ("hard_valid", "hard_rejected")}
    c4_report = {"status": "technical_checks_passed" if totals["hard_rules_pass"] and not totals["evidence_mapping"]["ambiguity_queue"] else "technical_failures", "C4": totals,
        "configuration": configuration, "companies": summaries, "elapsed_seconds": round(time.monotonic()-started, 2),
        "api_requests": 0, "winner_selected": False, "semantic_context_approved": False, "evaluation_gate_pass": False}
    write_json(output / "technical_validation.json", c4_report)
    for p in code_paths:
        target = output / "code" / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    inventory = [{"path": p.relative_to(output).as_posix(), "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "validation_manifest.json", {"files": inventory})
    prefix = output.relative_to(root).as_posix()
    combined = {**fixed_report, "status": "executed_C1_C4_acceptance_pending", "C4": {"status": c4_report["status"], "validated": True, **totals},
        "validation_path": prefix, "fixed_validation_path": fixed_report["validation_path"],
        "tokenizer": {**fixed_report["tokenizer"], "C4": {"status": c4_report["status"], "configuration_sha256": file_sha256(output / "configuration.json")}},
        "company_results": fixed_report["company_results"] + summaries,
        "elapsed_seconds": fixed_report["elapsed_seconds"] + c4_report["elapsed_seconds"],
        "C4_elapsed_seconds": c4_report["elapsed_seconds"],
        "arms": {**fixed_report["arms"], "C4": totals}, "C4_report": prefix + "/technical_validation.json",
        "evaluation_gate_pass": False, "api_requests": 0,
        "limitations": ["All four arms have technical results; only C4 is token-constrained by construction.",
            "C4 soft scores use initial weights and lexical/context proxies, not measured retrieval superiority.",
            "Frozen extraction/INTC structure issues and semantic label/event/context review remain unresolved."]}
    write_json(output / "step_03_validation_mapping.json", combined)
    write_json(run / "reports/step_03_validation_mapping.json", combined)
    table = "\n".join(f"| {arm} | {r['chunks']:,} | {r['covered_source_word_occurrences']:,}/{r['source_word_occurrences']:,} | {r['overflow_chunks']:,} | {r['provenance_errors']} | {r['evidence_mapping'].get('ambiguity_queue', 0)} |" for arm, r in combined["arms"].items())
    md = f"""# Step 3 update - C4 adaptive implementation and all-arm technical results

Status: **C4 implemented and executed for all 16 companies.** C4 technical status:
**{c4_report['status']}**. All-arm execution is complete; overall evaluation acceptance
remains pending. No strategy has been selected as the retrieval-quality winner.

## Measured results

| Arm | Chunks | Source words covered | Over 256 tokens | Provenance errors | Unresolved mappings |
|---|---:|---:|---:|---:|---:|
{table}

C4 independently reconstructs every selected source body exactly, including whitespace,
and recounts all final embedding inputs with the same pinned local tokenizer and
special tokens as C1-C3. Maximum C4 tokens: **{totals['max_tokens']}**. Quarantined
blocks: {totals['quarantined_blocks']}. Reference spans can map one-to-many; split
spans retain source-position bundles rather than nearest-text guesses. C4 mapping
summary: {json.dumps(totals['evidence_mapping'])}.

## Configurable selection mechanism

Module: `finsight/agents/narrative/corpus/processing/adaptive_chunking.py`; runner:
`scripts/narrative/chunking/build_adaptive_narrative_chunks.py`. Configuration:
`docs/narrative/config/narrative_c4_config_v1.json`, copied into this run's configuration record.

Nine complete plans per source block compare targets 128/192/256 tokens with
structure, sentence and token boundary policies. Plans preserve source blocks;
there is no cross-document/page/section combination. Repeated titles/headers/units
are literal spans from the same source block and count toward the input limit.
Unfit sentences/rows fall back to word/token boundaries; a character fallback can
preserve unusually large tokens. No content is discarded to meet the token limit.

**Hard rules:** full source-character coverage, traceable literal body/context and
actual untruncated token count <=256. Failed plans receive no soft scores. A block
with no passing plan is quarantined explicitly, never silently omitted. Candidate
counts: {json.dumps(totals['candidate_counts'])}.

**Soft criteria**, normalized to [0,1], with initial configurable weights of 0.25:

| Criterion | Recorded implementation |
|---|---|
| Structural integrity | Mean internal-boundary reward: paragraph/row 1.0, sentence 0.8, word 0.25, character 0.0; values configurable. |
| Context completeness | Fraction of applicable literal context spans retained; financial-field regex presence when no explicit headers exist. A proxy, not validated semantic completeness. |
| Semantic coherence | Mean adjacent-unit lexical cosine inside chunk bodies; stop words excluded; single-unit neutral score 0.5. No embedding or LLM. |
| Length efficiency | Mean actual token utilization times (1 - repeated-context character fraction). |

Weighted score is the weighted mean of those criteria. Highest score wins among
hard-valid plans; ties use fewer chunks, larger token target, then candidate ID.
Every candidate's criteria, features, hard checks and score, plus the selected
configuration and reason, are recorded. Equal weights/rewards are explicit starting
choices, not tuned/optimal weights. Selection reads no questions, evidence labels,
retrieval rankings or heldout performance. Identical plans reuse deterministic scores.

## Remaining context and evaluation gates

Chunks requiring source-context bundles: {totals['context_bundle_required_chunks']};
boundary/context flags: {totals['boundary_flags']}. Oversized headers may not fit
the repeated-context budget; their original text is preserved and missing context
is explicitly flagged. Lexical coherence and header retention do not prove that a
financial fact has the correct period, basis, unit or consolidated/segment scope.
Do not assume every chart has a GAAP distinction; only source-applicable fields
are considered. Original PDF correctness is outside these text-preservation checks.

The 11 INTC single-block filing issues remain in the frozen source build. C4 does
not rewrite that snapshot. Draft-label and event-family review remain pending.
C1-C3 still have measured token overflow. Therefore the overall experimental gate
has not passed, and no production indexes are promoted. Selecting a final strategy
still requires the same reviewed questions/labels and measured retrieval quality.
This run performs no retrieval experiment, generation evaluation, model inference
or API calls. Heldout mappings are technical only; no heldout claim text/labels/
per-item outcomes are exposed in this report.

## Outputs - full paths from the project root

- `{prefix}/configuration.json`: weights, candidate options, source/code/tokenizer hashes.
- `{prefix}/technical_validation.json`: C4 aggregate and company-level results.
- `{prefix}/step_03_validation_mapping.json`: combined C1-C4 status and metrics.
- `{prefix}/arms/C4/<TICKER>/chunks.jsonl`: selected adaptive chunks.
- `{prefix}/arms/C4/<TICKER>/source_map.jsonl`: exact body and repeated-context intervals.
- `{prefix}/arms/C4/<TICKER>/candidate_scores.jsonl.gz`: all block candidate scores and selection reasons (gzip JSONL).
- `{prefix}/arms/C4/<TICKER>/chunk_metrics.jsonl`: independently recounted input lengths.
- `{prefix}/arms/C4/<TICKER>/boundary_flags.jsonl`: context/boundary review queue.
- `{prefix}/arms/C4/<TICKER>/token_overflow.jsonl`: token failures, if any.
- `{prefix}/arms/C4/<TICKER>/quarantine.jsonl`: blocks with no hard-valid plan, if any.
- `{prefix}/mappings/dev/C4/<TICKER>/`: evidence mappings, bundles and ambiguity queues.
- `{prefix}/mappings/heldout/`: reviewer-only technical mappings, not tuning inputs.
- `{prefix}/validation_manifest.json`: file hashes.

Fixed C1-C3 artifacts and their original report remain preserved under
`{fixed.relative_to(root).as_posix()}/`. Canonical data, frozen inputs, labels,
production BM25/Qdrant and prior experiment scores remain unchanged.

## Commands

Read-only inspection:

```powershell
Get-Content -Raw {prefix}/technical_validation.json
python -m scripts.narrative.chunking.build_adaptive_narrative_chunks --run-id {args.run_id} --dataset-id {args.dataset_id} --validation-id {args.validation_id} --verify
Get-Content {prefix}/arms/C4/META/chunk_metrics.jsonl -TotalCount 1
```

Create a new isolated C4 version after changing configuration:

```powershell
python -u -m scripts.narrative.chunking.build_adaptive_narrative_chunks --run-id {args.run_id} --dataset-id {args.dataset_id} --validation-id {args.validation_id}_next --config docs/narrative/config/narrative_c4_config_v1.json
```

Elapsed: {c4_report['elapsed_seconds']} seconds. Artifact bytes before reports:
{sum(f['bytes'] for f in inventory):,}. API/model inference requests: **0**.
"""
    (output / "step_03_validation_mapping.md").write_text(md, encoding="utf-8")
    (run / "reports/step_03_validation_mapping.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": c4_report["status"], "C4": totals, "elapsed_seconds": c4_report["elapsed_seconds"], "report": str(output / "step_03_validation_mapping.md")}, indent=2))


if __name__ == "__main__":
    main()
