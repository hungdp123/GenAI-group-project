"""Freeze all-company baselines and common parsed inputs; never build canonical indexes.

Run from the repository root with ``python -m scripts.narrative.experiments.prepare_narrative_experiment
--all --run-id <unique-id>``. Use the same run ID with --resume after interruption,
or --verify for a read-only integrity check. No model or API calls are made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from finsight.agents.narrative.processing.corpus import CANONICAL_MANIFESTS, file_sha256, load_documents
from finsight.agents.narrative.processing.ingest import ProcessedBlock, _base, parse_document
from finsight.agents.narrative.processing.process_corpus import discover_tickers


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def identity(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_code_paths(root: Path) -> set[Path]:
    """Include nested Narrative tools and shared code in reproducible snapshots."""
    return set(root.glob("finsight/**/*.py")) | set(root.glob("scripts/**/*.py"))


def freeze(path: Path, root: Path, output: Path) -> dict:
    relative = path.relative_to(root)
    target = output / "snapshot" / relative
    before = file_sha256(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != before:
            raise RuntimeError(f"Snapshot/source mismatch; use a new run ID: {relative}")
    else:
        shutil.copy2(path, target)
    if file_sha256(path) != before or file_sha256(target) != before:
        raise RuntimeError(f"Source changed while copying: {relative}")
    return {"source_path": relative.as_posix(), "snapshot_path": target.relative_to(output).as_posix(),
            "sha256": before, "bytes": target.stat().st_size}


def verify(output: Path, *, canonical: bool = False) -> dict:
    manifest = json.loads((output / "foundation_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"] + manifest["generated_files"]:
        path = output / item["snapshot_path"]
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            failures.append(item["snapshot_path"])
        if canonical and "source_path" in item:
            source = Path.cwd() / item["source_path"]
            if not source.is_file() or file_sha256(source) != item["sha256"]:
                failures.append("canonical changed: " + item["source_path"])
    return {"status": "verified" if not failures else "failed", "files_checked":
            len(manifest["files"]) + len(manifest["generated_files"]), "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all", action="store_true")
    scope.add_argument("--ticker", action="append")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("Use letters, numbers, hyphens and underscores for run ID")
    root = Path.cwd().resolve()
    output = root / "data/narrative/experiments/narrative_rag" / args.run_id
    if args.verify:
        result = verify(output)
        print(json.dumps(result, indent=2))
        raise SystemExit(bool(result["failures"]))
    if not (args.all or args.ticker):
        parser.error("Select --all or --ticker")
    if output.exists() and not args.resume:
        parser.error("Run exists; use --verify, --resume, or a new run ID")
    if (output / "foundation_manifest.json").exists():
        parser.error("Completed foundation is immutable; use --verify or a new run ID")
    started = time.monotonic()
    tickers = discover_tickers(root) if args.all else sorted(set(t.upper() for t in args.ticker))
    if any(not re.fullmatch(r"[A-Z0-9]+", t) for t in tickers):
        parser.error("Invalid ticker")
    manifests = [root / p for p in CANONICAL_MANIFESTS if (root / p).exists()]
    excluded = {row["document_id"] for path in manifests
                for row in json.loads(path.read_text(encoding="utf-8"))["documents"]
                if row.get("retrieval_eligible") is False}
    documents = {t: [d for d in load_documents(t, root=root) if d.document_id not in excluded]
                 for t in tickers}
    if any(not docs for docs in documents.values()):
        parser.error("Every selected company must have eligible documents")
    selected = set(manifests)
    selected.update(d.local_path for docs in documents.values() for d in docs)
    selected.update(source_code_paths(root))
    selected.update(p for name in ("pyproject.toml", "uv.lock", "requirements.txt")
                    if (p := root / name).is_file())
    selected.update(root.glob("docs/narrative/evaluation/retrieval_queries_v2*"))
    for ticker in tickers:
        base = root / "data/narrative/processed" / ticker
        for name in ("documents.jsonl", "chunks.jsonl", "processing_report.json", "pdf_enrichment.json"):
            if (base / name).is_file():
                selected.add(base / name)
        for folder in ("indexes", "enrichments"):
            selected.update(p for p in (base / folder).rglob("*") if p.is_file())
    for folder in ("retrieval_comparison/ready_16_v2", "basic_rag", "narrative_agent/task7"):
        selected.update(p for p in (root / "data/narrative/experiments" / folder).rglob("*") if p.is_file())
    needed = sum(p.stat().st_size for p in selected)
    if shutil.disk_usage(root).free < needed * 1.2:
        raise RuntimeError("Insufficient free disk space for source/baseline snapshot")
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"run_id": args.run_id, "companies": tickers, "scope": "all eligible documents",
        "canonical_writes": False, "llm_calls": 0,
        "parsed_inputs": "current HTML parser; source-matched saved PDF blocks including promoted charts",
        "phase_1": {"backend": "bm25", "priority_policy": "P0", "arms_not_yet_built": {
            "C1": {"target_words": 800, "overlap_words": 120, "boundary_policy": "word"},
            "C2": {"target_words": 300, "overlap_words": 50, "boundary_policy": "word"},
            "C3": {"target_words": 400, "overlap_words": 60, "boundary_policy": "word"},
            "C4": {"strategy": "adaptive", "status": "not implemented"}}},
        "evaluation_cutoff": "2025-07-31 (existing v2 queries; filter at retrieval, not source acquisition)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "audited_embedding_token_limit": 256, "token_validation": "pending roadmap Step 3"}
    config_path = output / "configuration.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != configuration:
        raise RuntimeError("Resume configuration differs; choose a new run ID")
    write_json(config_path, configuration)
    print(f"Freezing {len(selected)} files ({needed / 1024**3:.2f} GiB) for {len(tickers)} companies", flush=True)
    files = []
    for number, path in enumerate(sorted(selected), 1):
        files.append(freeze(path, root, output))
        if number % 200 == 0:
            print(f"Frozen {number}/{len(selected)} files", flush=True)
    versions = {}
    for package in ("beautifulsoup4", "pypdf", "sentence-transformers", "transformers", "qdrant-client", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    write_json(output / "environment.json", {"python": sys.version, "platform": platform.platform(), "packages": versions})
    snapshot = output / "snapshot"
    summaries = []
    for ticker, docs in documents.items():
        destination = output / "parsed" / ticker
        checkpoint = destination / "input_manifest.json"
        if args.resume and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if file_sha256(destination / "documents.jsonl") != saved["blocks_sha256"]:
                raise RuntimeError(f"Changed checkpoint: {ticker}")
            summaries.append(saved)
            continue
        base = snapshot / "data/narrative/processed" / ticker
        stats = {r["document_id"]: r for r in json.loads((base / "processing_report.json").read_text(encoding="utf-8"))["documents"]}
        pdf_ids = {d.document_id for d in docs if d.local_path.suffix.lower() == ".pdf"}
        cached = defaultdict(list)
        with (base / "documents.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["document_id"] in pdf_ids:
                    cached[row["document_id"]].append(ProcessedBlock(**row))
        destination.mkdir(parents=True, exist_ok=True)
        rows, warnings, counts = [], [], {"html_reparsed": 0, "pdf_blocks_reused": 0}
        print(f"{ticker}: preparing {len(docs)} documents ({len(pdf_ids)} saved PDFs)", flush=True)
        with (destination / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for number, doc in enumerate(docs, 1):
                frozen = snapshot / doc.local_path.relative_to(root)
                if file_sha256(frozen) != doc.sha256:
                    raise RuntimeError(f"Manifest/source hash mismatch: {doc.document_id}")
                source = replace(doc, local_path=frozen, original_local_path=doc.original_local_path or doc.local_path)
                if doc.document_id in pdf_ids:
                    previous = stats.get(doc.document_id, {})
                    blocks = cached[doc.document_id]
                    if previous.get("sha256") != doc.sha256 or len(blocks) != previous.get("processed_blocks") or not blocks:
                        raise RuntimeError(f"PDF has no complete source-matched saved extraction: {doc.document_id}")
                    metadata = {k: v for k, v in _base(source).items() if k not in {"extraction_method", "source_format"}}
                    blocks = [replace(block, **metadata) for block in blocks]
                    doc_warnings = [w for w in json.loads((base / "processing_report.json").read_text(encoding="utf-8")).get("warnings", [])
                                    if w.startswith(doc.document_id + ":")]
                    counts["pdf_blocks_reused"] += 1
                else:
                    blocks, doc_warnings = parse_document(source)
                    counts["html_reparsed"] += 1
                if not blocks:
                    raise RuntimeError(f"No parsed content: {doc.document_id}")
                for block in blocks:
                    if block.document_id != doc.document_id or block.company_ticker != ticker:
                        raise RuntimeError("Block provenance mismatch")
                    handle.write(json.dumps(block.to_json(), ensure_ascii=False) + "\n")
                warnings.extend(doc_warnings)
                rows.append({**source.to_json(), "blocks": len(blocks), "words": sum(len(b.text.split()) for b in blocks),
                             "input_method": "saved_pdf_blocks" if doc.document_id in pdf_ids else "current_html_parser"})
                if number % 20 == 0:
                    print(f"{ticker}: {number}/{len(docs)} prepared", flush=True)
        summary = {"ticker": ticker, "documents": len(docs), **counts,
                   "blocks": sum(r["blocks"] for r in rows), "words": sum(r["words"] for r in rows),
                   "blocks_sha256": file_sha256(destination / "documents.jsonl"), "sources": rows, "warnings": warnings}
        write_json(checkpoint, summary)
        summaries.append(summary)
    generated = []
    for path in [config_path, output / "environment.json", *sorted((output / "parsed").rglob("*"))]:
        if path.is_file():
            generated.append({"snapshot_path": path.relative_to(output).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "files": files, "generated_files": generated,
                "source_identity": identity([f for f in files if f["source_path"].startswith("data/narrative/raw/")]),
                "build_identity": identity({"files": files, "generated": generated}), "companies": tickers}
    # Verify preserved canonical files before marking the foundation complete.
    changed = [f["source_path"] for f in files if file_sha256(root / f["source_path"]) != f["sha256"]]
    if changed:
        raise RuntimeError(f"Canonical inputs changed during the run: {changed}; use a new run ID")
    write_json(output / "foundation_manifest.json", manifest)
    report = {"status": "complete", "step": "01_foundation", "companies": [{k: v for k, v in s.items() if k not in {"sources", "warnings"}}
                for s in summaries], "eligible_documents": sum(s["documents"] for s in summaries),
              "excluded_records": sorted(excluded), "blocks": sum(s["blocks"] for s in summaries),
              "warning_occurrences": sum(len(s["warnings"]) for s in summaries),
              "elapsed_seconds": round(time.monotonic() - started, 2), "snapshot_bytes": sum(f["bytes"] for f in files),
              "generated_bytes": sum(f["bytes"] for f in generated), "api_requests": 0,
              "source_identity": manifest["source_identity"], "build_identity": manifest["build_identity"],
              "canonical_files_unchanged": True, "integrity": verify(output),
              "next_step": "Step 2: expand and freeze reviewed source-grounded development/heldout labels"}
    write_json(output / "reports/step_01_foundation.json", report)
    table = "\n".join(f"| {s['ticker']} | {s['documents']} | {s['html_reparsed']} | {s['pdf_blocks_reused']} | {s['blocks']} |" for s in summaries)
    command = f"python -m scripts.narrative.experiments.prepare_narrative_experiment --all --run-id {args.run_id}"
    markdown = f"""# Step 1 - All-company experimental foundation

Status: **{report['status']}**. Created {manifest['created_at']}.

## Scope and results

{len(tickers)} companies; {report['eligible_documents']} eligible documents; {report['blocks']} parsed blocks.
{len(excluded)} ineligible records remain excluded; no canonical files were changed.

| Company | Documents | HTML reparsed | Saved PDFs reused | Blocks |
|---|---:|---:|---:|---:|
{table}

## Work completed and reused modules

Reused `corpus.load_documents`, `file_sha256`, `ingest.parse_document`, `ProcessedBlock`
and `process_corpus.discover_tickers`. Current HTML parsing now supplies a common
all-company source representation. PDF inputs reuse source-matched saved blocks,
including promoted chart/table content; PDF extraction was not rerun.
The legacy META pilots remain diagnostic and are not the formal C1-C3 fixed arms.

## Inputs, configuration and identities

- `../configuration.json`: future C1-C4 definitions, BM25/P0, cutoff policy.
- `../foundation_manifest.json`: source/build identities and hashes for every frozen/generated input.
- `../snapshot/`: eligible raw sources, canonical manifests, saved blocks/chunks/indexes/enrichments,
  approved v2 questions/references, existing Task 4/5/7 reports and Python source code.
- `../parsed/<TICKER>/documents.jsonl`: frozen common parsed blocks for future arms.
- `../parsed/<TICKER>/input_manifest.json`: source hashes, provenance, methods and warnings.
- `../environment.json`: installed package versions and Python environment.

Source identity: `{manifest['source_identity']}`.
Build identity: `{manifest['build_identity']}`.
No API keys or `config.yaml` were copied. Corpus includes eligible dates beyond the
development benchmark cutoff; subsequent retrieval must apply each question's cutoff.

## Reproduce and inspect

Create a **new** run (writes only its experiment folder):

```powershell
{command}_new
```

Read-only snapshot integrity check for this run:

```powershell
python -m scripts.narrative.experiments.prepare_narrative_experiment --run-id {args.run_id} --verify
Get-Content -Raw data/narrative/experiments/narrative_rag/{args.run_id}/reports/step_01_foundation.json
```

Interrupted runs support `--resume` with the original scope and run ID. Completed
foundations cannot be overwritten. Future experiment runners must explicitly read
these isolated inputs and keep their caches/indexes here; canonical runners are not
redirected automatically by this step.

## Checks, costs and limitations

Verified source hashes against manifests, PDF extraction hashes/block counts,
block company/document provenance, copied-file hashes and unchanged canonical files.
Snapshot integrity: {report['integrity']['status']} ({report['integrity']['files_checked']} files).
Elapsed: {report['elapsed_seconds']} seconds. Snapshot: {report['snapshot_bytes'] / 1024**3:.3f} GiB;
generated inputs: {report['generated_bytes'] / 1024**2:.1f} MiB. Model/API requests: **0**.
Warnings retained: {report['warning_occurrences']} occurrences; see per-company input manifests.

These checks establish reproducible input preparation, not source extraction accuracy,
complete page coverage, chunk token validity, retrieval quality or a winning strategy.
Parser outputs require Step 3 content/context validation. The audited embedding limit
is 256 tokens; no chunk truncation or embedding was performed here. Model artifacts
are not vendored; model/tokenizer revision pinning is required before token validation.
Historical scores/indexes are preserved as historical outputs, not regenerated against
the new parsed inputs. No new C1-C4 chunks, production indexes or rankings were built.

## Decision and next action

Step 1 is complete for all selected companies. Steps 2-7 remain pending.
Next: prepare representative source-span evidence labels across companies and document
families, with development/heldout grouping, before selecting any strategy.
"""
    (output / "reports/step_01_foundation.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "eligible_documents", "blocks", "elapsed_seconds", "integrity")}, indent=2))
    print(f"Report: {output / 'reports/step_01_foundation.md'}", flush=True)


if __name__ == "__main__":
    main()
