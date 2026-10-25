"""Preview or copy legacy full C4 builds into processed/chunking_strategies.

No parsing, embeddings, model loading or API calls. The original build is retained.
Stop any builders/RAG processes using the source before applying the migration.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil

from scripts.narrative.retrieval.c4_storage import (
    BUILDER_PATH, DEFAULT_OUTPUT, sha256, validate_output, write_chunk_artifacts,
)


def migrate(workspace: Path, source: Path, destination: Path, *, apply: bool = False) -> dict:
    workspace, source = workspace.resolve(), source.resolve()
    destination = validate_output(workspace, destination)
    legacy = (workspace / "data/narrative/runtime").resolve()
    if (not source.is_relative_to(workspace) or not source.is_relative_to(legacy)
            or source == legacy or not source.is_dir()):
        raise ValueError("Source must be an existing versioned build inside data/narrative/runtime")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; choose a new build version: {destination}")
    config = source / "configuration.json"
    if not config.is_file():
        raise ValueError("Source is not a full-corpus C4 build")
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings.get("strategy") != "C4" or settings.get("publication_cutoff") is not None:
        raise ValueError("Source is not a full-corpus C4 build")
    # Exclude only ephemeral Qdrant locks. Preserve all index/database/checkpoint files.
    files = [p for p in source.rglob("*") if p.is_file() and p.name != ".lock"]
    if any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("Symlinked source files/directories are not supported")
    companies = [p for p in sorted(source.iterdir()) if p.is_dir() and re.fullmatch(r"[A-Z0-9.-]+", p.name)]
    if not companies:
        raise ValueError("No company outputs found")
    for company in companies:
        index_path = company / "indexes/index.json"
        if not index_path.is_file():
            continue  # An unfinished chunk build will restart on --resume.
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("chunking", {}).get("strategy") != "adaptive":
            raise ValueError(f"{company.name}: not an adaptive index")
        for report_path in (company / "chunking_report.json", company / "build_report.json"):
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report["index_sha256"] != sha256(index_path):
                    raise ValueError(f"{company.name}: index/checkpoint mismatch")
        metadata = company / "indexes/qdrant/qdrant_metadata.json"
        if metadata.exists():
            vector = json.loads(metadata.read_text(encoding="utf-8"))
            if (vector.get("corpus_fingerprint") != payload.get("corpus_fingerprint")
                    or vector.get("chunk_ids") != [c["chunk_id"] for c in payload["chunks"]]
                    or vector.get("vector_count") != len(payload["chunks"])):
                raise ValueError(f"{company.name}: BM25/vector metadata mismatch")
    plan = {"status": "PREVIEW", "source": str(source), "destination": str(destination),
            "companies": [p.name for p in companies], "files": len(files),
            "bytes_to_copy": sum(p.stat().st_size for p in files), "original_retained": True}
    if not apply:
        return plan
    before = {p.relative_to(source).as_posix(): sha256(p) for p in files}
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".lock"))
    for relative, digest in before.items():
        if sha256(source / relative) != digest or sha256(destination / relative) != digest:
            raise ValueError(f"Copy/source changed: {relative}; stop active builds and use a fresh destination")
    for company in companies:
        target = destination / company.name
        index_path = target / "indexes/index.json"
        # Only export completed chunk checkpoints; leave partial outputs for --resume.
        if index_path.exists() and (target / "chunking_report.json").exists():
            write_chunk_artifacts(target, json.loads(index_path.read_text(encoding="utf-8")))
    for strategy in ("C1", "C2", "C3", "C4"):
        (destination.parent.parent / strategy).mkdir(parents=True, exist_ok=True)
    manifest = {
        **plan, "status": "COPIED_AND_VERIFIED", "migrated_at": datetime.now(timezone.utc).isoformat(),
        "source_configuration_sha256": sha256(destination / "configuration.json"),
        "compatible_builder_sha256": sha256(workspace / BUILDER_PATH),
        "source_files_sha256": before,
        "note": "Storage-only migration; original configuration, chunk IDs and index fingerprints retained.",
    }
    (destination / "layout_migration.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in manifest.items() if k != "source_files_sha256"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/narrative/runtime/c4_full_v1"))
    parser.add_argument("--destination", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true", help="Copy and verify; default only previews")
    args = parser.parse_args()
    try:
        report = migrate(Path.cwd(), args.source, args.destination, apply=args.apply)
    except (OSError, ValueError, KeyError, EOFError) as exc:
        parser.exit(2, f"Migration failed: {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
