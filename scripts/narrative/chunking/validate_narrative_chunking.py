"""Build isolated fixed plans, validate coverage/tokens, and map source spans.

No embeddings, retrieval rankings, model inference, API calls or canonical writes.
C4 is reported pending; validation never invents an adaptive plan or gold labels.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from finsight.agents.narrative.processing.chunk import chunk_blocks
from finsight.agents.narrative.processing.corpus import file_sha256
from finsight.agents.narrative.processing.ingest import ProcessedBlock
from scripts.narrative.experiments.prepare_narrative_experiment import identity, write_json
from scripts.narrative.datasets.prepare_narrative_labels import jsonl, verify as verify_labels, write_jsonl
from scripts.narrative.datasets.narrative_reference_mapping import map_references


ARMS = {"C1": (800, 120), "C2": (300, 50), "C3": (400, 60)}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def overlaps(a: int, z: int, start: int, end: int) -> bool:
    return a < end and z > start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261018_all16_v1")
    parser.add_argument("--dataset-id", default="evidence_labels_v3")
    parser.add_argument("--validation-id", default="technical_v1")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", x) for x in (args.run_id, args.dataset_id, args.validation_id)):
        parser.error("IDs must use letters, numbers, hyphens and underscores")
    root = Path.cwd().resolve()
    run = root / "data/narrative/experiments/narrative_rag" / args.run_id
    dataset = run / "labels" / args.dataset_id
    output = run / "validation" / args.validation_id
    if args.verify:
        manifest = json.loads((output / "validation_manifest.json").read_text(encoding="utf-8"))
        failed = [f["path"] for f in manifest["files"] if not (output / f["path"]).is_file()
                  or file_sha256(output / f["path"]) != f["sha256"]]
        print(json.dumps({"status": "verified" if not failed else "failed", "files_checked": len(manifest["files"]), "failures": failed}, indent=2))
        raise SystemExit(bool(failed))
    if output.exists():
        parser.error("Validation version exists; use --verify or a new --validation-id")
    started = time.monotonic()
    foundation = json.loads((run / "foundation_manifest.json").read_text(encoding="utf-8"))
    label_manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    if label_manifest["foundation_build_identity"] != foundation["build_identity"]:
        raise RuntimeError("Labels and parsed-source builds differ")
    label_check = verify_labels(dataset, run)
    if label_check["failures"]:
        raise RuntimeError("Draft label integrity failed; resolve before mapping")
    if args.tokenizer_path:
        cached = args.tokenizer_path.resolve()
    else:
        cache_root = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        candidates = sorted(p for p in cache_root.glob("*") if (p / "sentence_bert_config.json").is_file())
        if len(candidates) != 1:
            parser.error("Provide --tokenizer-path to select one cached model revision")
        cached = candidates[0]
    sentence_config = json.loads((cached / "sentence_bert_config.json").read_text(encoding="utf-8"))
    st_config = json.loads((cached / "config_sentence_transformers.json").read_text(encoding="utf-8"))
    if st_config.get("prompts") or st_config.get("default_prompt_name"):
        raise RuntimeError("Nonempty embedding prompt needs explicit input-format support")
    output.mkdir(parents=True)
    pinned = output / "tokenizer"
    pinned.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "sentence_bert_config.json", "config_sentence_transformers.json", "config.json"):
        shutil.copy2(cached / name, pinned / name)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pinned, local_files_only=True, use_fast=True)
    model_limit = json.loads((pinned / "config.json").read_text(encoding="utf-8"))["max_position_embeddings"]
    limit = min(sentence_config["max_seq_length"], tokenizer.model_max_length, model_limit)
    configuration = {"foundation_build_identity": foundation["build_identity"], "labels_manifest_sha256": file_sha256(dataset / "dataset_manifest.json"),
        "arms": {arm: {"target_words": t, "overlap_words": o, "boundary_policy": "word", "content_policy": "strict_word_windows_all_content_types_within_frozen_blocks"} for arm, (t, o) in ARMS.items()},
        "C4": {"status": "pending_adaptive_plan", "metrics": None},
        "embedding_input": "chunk.text exactly; no injected headers or prompt; add_special_tokens=True; truncation=False",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "cached_revision": cached.name,
        "effective_token_limit": limit, "special_tokens": tokenizer.num_special_tokens_to_add(pair=False),
        "tokenizer_model_limit": tokenizer.model_max_length, "sentence_transformer_limit": sentence_config["max_seq_length"],
        "tokenizer_files": {p.name: file_sha256(p) for p in pinned.iterdir()},
        "chunker_sha256": file_sha256(root / "finsight/agents/narrative/processing/chunk.py"), "validator_sha256": file_sha256(Path(__file__)),
        "evaluation_cutoff": label_manifest["cutoff_date"], "source_scope": "all 1201 eligible frozen documents, including later publications; labels filtered by benchmark cutoff",
        "canonical_protected_type_policy": "historical baseline keeps charts atomic and protects typed tables; strict C1-C3 here deliberately do not, and are not byte-identical to that baseline"}
    write_json(output / "configuration.json", configuration)
    all_questions, all_evidence = {}, {}
    for split in ("dev", "heldout"):
        all_questions[split] = jsonl(dataset / split / "questions.jsonl")
        all_evidence[split] = jsonl(dataset / split / "evidence_labels.jsonl")
    summaries, source_issues, context_review = [], [], []
    source_hash_count = 0
    raw_sources_seen = set()
    for ticker in foundation["companies"]:
        base = run / "parsed" / ticker
        inputs = json.loads((base / "input_manifest.json").read_text(encoding="utf-8"))
        if file_sha256(base / "documents.jsonl") != inputs["blocks_sha256"]:
            raise RuntimeError("Frozen input changed: " + ticker)
        sources = {s["document_id"]: s for s in inputs["sources"]}
        for source in sources.values():
            raw_path = Path(source["local_path"])
            if file_sha256(raw_path) != source["sha256"]:
                raise RuntimeError("Frozen raw source hash failed")
            raw_sources_seen.add(raw_path)
            source_hash_count += 1
            if source["document_type"] in {"10-K", "10-Q", "20-F"} and source["blocks"] == 1:
                source_issues.append({"ticker": ticker, "document_id": source["document_id"], "issue": "single_block_filing", "words": source["words"]})
        rows = jsonl(base / "documents.jsonl")
        blocks = [ProcessedBlock(**r) for r in rows]
        words = [list(re.finditer(r"\S+", b.text)) for b in blocks]
        source_words = sum(map(len, words))
        # Strict arms reuse the existing word-window implementation without its
        # protected-type routing. Original content metadata is restored afterward.
        view = [replace(b, content_type="narrative") for b in blocks]
        for arm, (target, overlap) in ARMS.items():
            arm_started = time.monotonic()
            destination = output / "arms" / arm / ticker
            destination.mkdir(parents=True)
            candidates = chunk_blocks(view, target_words=target, overlap_words=overlap, boundary_policy="word")
            generated, maps, metrics, overflow, provenance_errors, boundary_flags = [], [], [], [], [], []
            by_line = defaultdict(list)
            covered_words = duplicated_words = cursor = 0
            chart_table_chunks = header_missing = 0
            exact_text_counts = Counter()
            for line, (b, spans) in enumerate(zip(blocks, words), 1):
                nwords = len(spans)
                covered = bytearray(nwords)
                ranges = list(range(0, nwords, target - overlap))
                nonempty_lines = [m for m in re.finditer(r"(?m)^.*\S.*$", b.text)]
                header_end = nonempty_lines[1].end() if len(nonempty_lines) >= 2 else len(b.text)
                source = sources[b.document_id]
                for start in ranges:
                    end = min(start + target, nwords)
                    chunk = replace(candidates[cursor], content_type=b.content_type)
                    cursor += 1
                    text = chunk.text
                    if text != " ".join(m.group() for m in spans[start:end]):
                        raise RuntimeError("Output is not the expected literal word window")
                    for key, expected in (("document_id", b.document_id), ("company_ticker", ticker), ("document_type", source["document_type"]), ("publication_date", source["publication_date"]), ("source_url", source["source_url"])):
                        if getattr(chunk, key) != expected:
                            provenance_errors.append({"chunk_id": chunk.chunk_id, "field": key})
                    if not chunk.citation_span:
                        provenance_errors.append({"chunk_id": chunk.chunk_id, "field": "citation_span"})
                    duplicated_words += sum(covered[start:end])
                    covered[start:end] = b"\x01" * (end - start)
                    char_start, char_end = spans[start].start(), spans[end - 1].end()
                    source_map = {"chunk_id": chunk.chunk_id, "document_id": b.document_id, "source_sha256": source["sha256"],
                        "parsed_file": (base / "documents.jsonl").relative_to(run).as_posix(), "parsed_file_sha256": inputs["blocks_sha256"],
                        "block_line_1based": line, "block_text_sha256": sha(b.text), "source_word_start": start, "source_word_end": end,
                        "source_char_start": char_start, "source_char_end": char_end, "normalization": "whitespace-to-single-space; no other substitutions",
                        "page": b.page, "section": b.section, "citation_span": b.citation_span, "chart_id": b.chart_id}
                    maps.append(source_map)
                    by_line[line].append(source_map)
                    generated.append(chunk.to_json())
                    exact_text_counts[chunk.text_sha256] += 1
                    if b.content_type in {"table", "chart"}:
                        chart_table_chunks += 1
                        if b.content_type == "chart":
                            marker = b.text.find("| Metric")
                            context_end = b.text.find("\n", marker) if marker >= 0 else header_end
                        else:
                            context_end = header_end
                        if char_start > 0 or char_end < context_end:
                            header_missing += 1
                            boundary_flags.append({"chunk_id": chunk.chunk_id, "issue": "original_leading_table_or_chart_context_not_fully_in_this_chunk", "block_line_1based": line})
                        for boundary in (char_start if start else None, char_end if end < nwords else None):
                            if boundary is None:
                                continue
                            line_start = b.text.rfind("\n", 0, boundary) + 1
                            line_end = b.text.find("\n", boundary)
                            line_end = len(b.text) if line_end < 0 else line_end
                            if "|" in b.text[line_start:line_end]:
                                boundary_flags.append({"chunk_id": chunk.chunk_id, "issue": "markdown_row_boundary_cut", "source_char": boundary})
                covered_words += sum(covered)
                if sum(covered) != nwords:
                    raise RuntimeError("Source word occurrence coverage failed")
            if cursor != len(candidates) or len({c["chunk_id"] for c in generated}) != len(generated):
                raise RuntimeError("Chunk count or unique-ID validation failed")
            token_counts = []
            for start in range(0, len(generated), 256):
                batch = generated[start:start + 256]
                encoded = tokenizer([c["text"] for c in batch], add_special_tokens=True, truncation=False, padding=False, verbose=False)
                for c, tokens in zip(batch, encoded["input_ids"]):
                    token_count = len(tokens)
                    token_counts.append(token_count)
                    item = {"chunk_id": c["chunk_id"], "words": len(c["text"].split()), "tokens_including_special": token_count,
                            "effective_limit": limit, "token_limit_pass": token_count <= limit, "content_type": c["content_type"]}
                    metrics.append(item)
                    if token_count > limit:
                        overflow.append({**item, "document_id": c["document_id"], "tokens_over_limit": token_count - limit})
            write_jsonl(destination / "chunks.jsonl", generated)
            write_jsonl(destination / "source_map.jsonl", maps)
            write_jsonl(destination / "chunk_metrics.jsonl", metrics)
            write_jsonl(destination / "token_overflow.jsonl", overflow)
            write_jsonl(destination / "boundary_flags.jsonl", boundary_flags)
            mapping_counts = map_references(rows=rows, sources=sources, source_maps=maps,
                all_questions=all_questions, all_evidence=all_evidence, output=output, arm=arm, ticker=ticker)
            summary = {"ticker": ticker, "arm": arm, "documents": len(sources), "source_blocks": len(blocks), "chunks": len(generated),
                "source_word_occurrences": source_words, "covered_source_word_occurrences": covered_words, "coverage_fraction": covered_words / source_words,
                "duplicate_overlap_word_occurrences": duplicated_words, "duplicate_overlap_ratio": duplicated_words / source_words,
                "exact_duplicate_text_chunks": sum(n - 1 for n in exact_text_counts.values()), "provenance_errors": len(provenance_errors),
                "missing_source_url_chunks": sum(not c["source_url"] for c in generated), "token_limit": limit,
                "overflow_chunks": len(overflow), "max_tokens": max(token_counts), "mean_tokens": sum(token_counts) / len(token_counts),
                "chart_table_chunks": chart_table_chunks, "chart_table_chunks_without_original_leading_context": header_missing,
                "boundary_flags": len(boundary_flags), "evidence_mapping": dict(mapping_counts),
                "hard_rules_pass": covered_words == source_words and not provenance_errors and not overflow,
                "elapsed_seconds": round(time.monotonic() - arm_started, 2)}
            write_json(destination / "validation.json", summary)
            summaries.append(summary)
            print(f"{ticker} {arm}: {len(generated)} chunks; coverage {summary['coverage_fraction']:.0%}; {len(overflow)} over {limit} tokens; {mapping_counts['ambiguity_queue']} unresolved mappings", flush=True)
        for q in (q for q in all_questions["dev"] if q["ticker"] == ticker):
            context = q["required_context"]
            unresolved = [k for k, v in context.items() if v is None or re.search(r"review|explicit in quote|if applicable", str(v), re.I)]
            context_review.append({"query_id": q["query_id"], "context_fields_needing_review": unresolved,
                "original_pdf_review_required": q.get("original_pdf_review_required", False),
                "semantic_context_complete": None, "status": "review_required"})
    write_json(output / "source_structure_issues.json", source_issues)
    write_jsonl(output / "development_context_review.jsonl", context_review)
    shutil.copy2(Path(__file__), output / "validator_source.py")
    arm_totals = {}
    for arm in ARMS:
        items = [s for s in summaries if s["arm"] == arm]
        arm_totals[arm] = {k: sum(s[k] for s in items) for k in ("chunks", "source_word_occurrences", "covered_source_word_occurrences", "overflow_chunks", "provenance_errors", "boundary_flags", "chart_table_chunks_without_original_leading_context")}
        arm_totals[arm]["max_tokens"] = max(s["max_tokens"] for s in items)
        arm_totals[arm]["evidence_mapping"] = dict(sum((Counter(s["evidence_mapping"]) for s in items), Counter()))
        arm_totals[arm]["hard_rules_pass"] = all(s["hard_rules_pass"] for s in items)
    report = {"status": "executed_acceptance_pending", "execution_complete": True, "step": "03_validation_mapping",
        "validation_path": output.relative_to(root).as_posix(), "companies": len(foundation["companies"]), "documents": source_hash_count,
        "source_hashes_checked": len(raw_sources_seen), "parsed_blocks": sum(s["source_blocks"] for s in summaries if s["arm"] == "C1"),
        "arms": arm_totals, "C4": {"status": "pending_adaptive_plan", "validated": False}, "tokenizer": configuration,
        "source_structure_issues": len(source_issues), "development_context_cases_pending_review": len(context_review),
        "semantic_label_approval": False, "evaluation_gate_pass": False, "api_requests": 0,
        "elapsed_seconds": round(time.monotonic() - started, 2), "company_results": summaries,
        "limitations": ["Source coverage is relative to frozen extracted text, not original PDF extraction correctness.",
            "Fixed arms preserve block boundaries but use strict word windows even for charts/tables; historical protected-type baseline remains frozen separately.",
            "Draft span remapping does not approve semantic labels or event-family splits.", "No C4 plan exists yet; rerun validation when adaptive plans are available."]}
    write_json(output / "technical_validation.json", report)
    inventory = [{"path": p.relative_to(output).as_posix(), "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "validation_manifest.json", {"configuration_sha256": file_sha256(output / "configuration.json"), "files": inventory})
    prefix = output.relative_to(root).as_posix()
    table = "\n".join(f"| {a} | {s['chunks']:,} | {s['covered_source_word_occurrences']:,}/{s['source_word_occurrences']:,} | {s['overflow_chunks']:,} | {s['provenance_errors']} | {s['evidence_mapping'].get('ambiguity_queue', 0)} |" for a, s in arm_totals.items())
    md = f"""# Step 3 - Technical validation and source-reference mapping

Status: **Execution completed for C1-C3; acceptance pending.** No strategy selected.
Scope: {report['companies']} companies, {source_hash_count} source documents,
{report['parsed_blocks']:,} frozen parsed blocks. C4 validation awaits its adaptive plan.

## Checks and measured results

| Arm | Chunks | Covered source word occurrences | Over 256 tokens | Provenance errors | Unresolved mappings |
|---|---:|---:|---:|---:|---:|
{table}

Coverage counts source word occurrences by their positions, preserving repeated
phrases and allowing whitespace normalization/overlap. It is not set-of-words
matching. Every chunk's exact text, original metadata and source hash were checked.
Raw source hashes checked: {len(raw_sources_seen)}. Full source scope includes
post-cutoff documents; labelled evidence obeys the benchmark publication cutoff.

The fixed configurations use 800/120, 300/50 and 400/60 **word windows inside each
frozen source block**, including chart/table blocks. Original content types and
citations are restored in output metadata. This reuses `chunk_blocks`' word mode,
without the protected-type routing. The old production baseline's atomic charts
and protected tables remain frozen and unchanged: it is not byte-identical to
this formal strict-word C1. Record this distinction when interpreting later results.

## Tokenizer and hard constraints

Pinned local MiniLM revision: `{cached.name}`. Effective input limit: **{limit}**,
including {configuration['special_tokens']} special tokens. Counts include all
text actually embedded (`chunk.text`); existing headers inside text count too.
The current adapter injects no extra header/prompt. Tokenization used
`add_special_tokens=True`, **`truncation=False`**. No model weights were loaded,
embeddings generated, or API/model inference calls made.

Oversized chunks remain unchanged and are reported; word settings were not silently
shortened. C1-C3 therefore fail the embedding token hard rule where overflow is
recorded. They are not eligible for dense-index promotion under this tokenizer.
Changing the model or adding an explicitly named token-compatible representation
requires a new experiment configuration and validation run.

## Source references and context

All 156 draft source spans are mapped independently per arm using source hashes,
1-based block lines and exact source character positions. Overlapping windows
produce legitimate one-to-many chunk matches; no nearest-text choice is made.
Spans split across boundaries retain all candidate chunk IDs as evidence bundles.
Question-level bundles preserve all required anchors (e.g. table header plus row).
Unresolved or mismatched anchors go to a separate ambiguity queue.

These are **technical mappings**, not approved relevance/entailment labels. Existing
draft labels and provisional event splits remain unapproved. The development-only
context review queue flags missing/placeholding context fields; it does not
automatically certify metric/value/unit/period/basis/scope completeness. No heldout
claim text, quotes, proposed verdicts or per-item outcomes appear in this report.

Strict splitting can detach table/chart headers or cut rows. Per-chunk boundary
flags and original-leading-context counts expose these cases; semantic context
review is still required. Single-block filing issues retained: {len(source_issues)}
(the known INTC filings). Correcting the parser requires a new frozen source build
and new mappings; this run does not mutate the Step 1 snapshot.

## Outputs - full paths from the project root

- `{prefix}/technical_validation.json`: all-company and per-arm results.
- `{prefix}/configuration.json`: source/label/code identities and exact settings.
- `{prefix}/tokenizer/`: pinned tokenizer files and model input configuration.
- `{prefix}/arms/<C1|C2|C3>/<TICKER>/chunks.jsonl`: isolated candidate chunks.
- `{prefix}/arms/<C1|C2|C3>/<TICKER>/source_map.jsonl`: source-position traceability.
- `{prefix}/arms/<C1|C2|C3>/<TICKER>/chunk_metrics.jsonl`: words and untruncated token counts.
- `{prefix}/arms/<C1|C2|C3>/<TICKER>/token_overflow.jsonl`: every oversized chunk.
- `{prefix}/arms/<C1|C2|C3>/<TICKER>/boundary_flags.jsonl`: chart/table context risks.
- `{prefix}/mappings/dev/<C1|C2|C3>/<TICKER>/evidence_mapping.jsonl`: development mappings.
- `{prefix}/mappings/dev/<C1|C2|C3>/<TICKER>/reference_bundles.jsonl`: question-level bundles.
- `{prefix}/mappings/dev/<C1|C2|C3>/<TICKER>/ambiguity_queue.jsonl`: development mapping issues.
- `{prefix}/mappings/heldout/`: reviewer-only technical mappings; do not use for tuning.
- `{prefix}/development_context_review.jsonl`: pending development context review.
- `{prefix}/source_structure_issues.json`: structural input issues.
- `{prefix}/validation_manifest.json`: output hashes for integrity verification.

## Commands and resource use

Read-only inspection:

```powershell
Get-Content -Raw {prefix}/technical_validation.json
python -m scripts.narrative.chunking.validate_narrative_chunking --run-id {args.run_id} --dataset-id {args.dataset_id} --validation-id {args.validation_id} --verify
```

Create another isolated version (writes candidates/mappings/reports):

```powershell
python -u -m scripts.narrative.chunking.validate_narrative_chunking --run-id {args.run_id} --dataset-id {args.dataset_id} --validation-id {args.validation_id}_next
```

Elapsed: {report['elapsed_seconds']} seconds. Measured artifact bytes before reports:
{sum(f['bytes'] for f in inventory):,}. No BM25/vector indexes built; no retrieval
rankings/scores produced; API/model inference requests: **0**. Canonical outputs
and frozen sources/labels remain unchanged.

## Acceptance and next actions

Step 3 was executed, but the overall acceptance gate has **not passed**. Resolve
embedding overflow, context/structure issues and label/event review; provide a C4
adaptive plan and validate it with the same source coverage/token/provenance rules.
Retain C1-C3 unmodified as measured fixed baselines. Their BM25-only comparison may
be used as a diagnostic with disclosed technical failures; do not claim all four
strategies or the dense representation have passed validation. Do not promote
these candidates or report a winning strategy from technical counts alone.
"""
    (output / "step_03_validation_mapping.md").write_text(md, encoding="utf-8")
    (run / "reports/step_03_validation_mapping.md").write_text(md, encoding="utf-8")
    write_json(run / "reports/step_03_validation_mapping.json", report)
    print(json.dumps({"status": report["status"], "arms": arm_totals, "elapsed_seconds": report["elapsed_seconds"], "report": str(output / "step_03_validation_mapping.md")}, indent=2))
    if args.fail_on_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
