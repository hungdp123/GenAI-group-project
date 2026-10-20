"""Preserve saved company outputs and parser previews for chunking comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="META")
    parser.add_argument("--verify", action="store_true", help="Check an existing snapshot without rewriting it")
    args = parser.parse_args()
    ticker = args.ticker.upper()
    if not ticker.isalnum():
        parser.error("Ticker must contain letters and digits only")
    root = Path.cwd().resolve()
    output = root / "data/narrative/experiments/chunking" / ticker / "baseline_800_120"
    manifest_path = output / "baseline_manifest.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        failures = [item["snapshot_path"] for item in manifest["files"]
                    if not (output / item["snapshot_path"]).is_file()
                    or digest(output / item["snapshot_path"]) != item["sha256"]]
        print(json.dumps({"status": "verified" if not failures else "failed",
                          "files_checked": len(manifest["files"]), "failures": failures,
                          "baseline": str(output)}, indent=2))
        if failures:
            raise SystemExit(1)
        return
    if output.exists():
        parser.error(f"Baseline already exists; use --verify: {output}")
    source = root / "data/narrative/processed" / ticker
    metadata = json.loads((source / "indexes/index_metadata.json").read_text(encoding="utf-8"))
    if metadata["chunking"] != {"target_words": 800, "overlap_words": 120}:
        parser.error("Saved company index is not configured for 800/120")
    selected = [source / name for name in (
        "documents.jsonl", "chunks.jsonl", "processing_report.json", "pdf_enrichment.json")]
    selected += [p for folder in ("indexes", "enrichments") for p in (source / folder).rglob("*") if p.is_file()]
    selected += list((root / "data/narrative/raw").glob("manifest*.json"))
    selected += [root / "docs/narrative/evaluation/retrieval_queries_v2.jsonl"]
    preview_root = root / "data/narrative/experiments/parser_preview"
    previews = []
    for report in sorted(preview_root.glob(f"{ticker}-*/report.json")):
        info = json.loads(report.read_text(encoding="utf-8"))
        if info["chunking"] == metadata["chunking"]:
            previews.append(info["document_id"])
            selected += [p for p in report.parent.rglob("*") if p.is_file()]
    files = []
    output.mkdir(parents=True)
    for path in sorted(set(selected)):
        relative = path.relative_to(root)
        destination = output / relative
        before = digest(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if digest(path) != before or digest(destination) != before:
            raise RuntimeError(f"File changed while snapshotting: {relative}")
        files.append({"source_path": relative.as_posix(), "snapshot_path": relative.as_posix(),
                      "sha256": before, "bytes": destination.stat().st_size})
    manifest = {
        "status": "frozen", "created_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker, "chunking": metadata["chunking"],
        "company_chunk_count": metadata["chunk_count"],
        "corpus_fingerprint": metadata["corpus_fingerprint"], "parser_previews": previews,
        "notes": [
            "Company snapshot preserves the saved corpus, BM25 and Qdrant outputs without rebuilding.",
            "Company outputs predate the SEC paragraph parser change; they are a historical reference.",
            "Frozen parser previews use the improved parser at 800/120; use those same blocks for a controlled chunk-size comparison.",
            "Raw source PDFs/HTML are referenced by captured manifests and are not duplicated.",
            "Existing experiment scores are not rerun or replaced by this snapshot.",
        ], "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("status", "ticker", "chunking", "company_chunk_count", "parser_previews")}, indent=2))
    print(f"Baseline: {output}")


if __name__ == "__main__":
    main()
