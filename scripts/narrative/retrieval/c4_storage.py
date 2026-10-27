"""Storage-only helpers shared by the C4 builder and offline layout migration."""

from __future__ import annotations

import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
from uuid import uuid4


BUILDER_PATH = "scripts/narrative/retrieval/build_full_c4_indexes.py"
DEFAULT_OUTPUT = Path("data/narrative/processed/chunking_strategies/C4/full_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output(workspace: Path, output: Path) -> Path:
    output = output.resolve()
    base = (workspace / "data/narrative/processed/chunking_strategies/C4").resolve()
    if not output.is_relative_to(workspace.resolve()) or output.parent != base or not output.name:
        raise ValueError("Use data/narrative/processed/chunking_strategies/C4/<build-version>")
    return output


def write_chunk_artifacts(company: Path, payload: dict) -> dict:
    """Export exact indexed records; verify maps without changing index identities."""
    chunks_path = company / "chunks.jsonl"
    if not chunks_path.exists():
        temporary = company / f".chunks-{uuid4().hex}.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in payload["chunks"]:
                handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        temporary.replace(chunks_path)
    with chunks_path.open(encoding="utf-8") as handle:
        actual = (json.loads(line) for line in handle if line.strip())
        if any(a != b for a, b in zip_longest(actual, payload["chunks"])):
            raise ValueError(f"Chunk records differ from the BM25 index: {chunks_path}")

    map_path = company / "source_map.jsonl"
    compressed = company / "source_map.jsonl.gz"
    if not map_path.exists():
        if not compressed.is_file():
            raise FileNotFoundError(f"Source mapping is missing for {company.name}")
        temporary = company / f".source-map-{uuid4().hex}.tmp"
        with gzip.open(compressed, "rt", encoding="utf-8") as source, \
                temporary.open("w", encoding="utf-8", newline="\n") as target:
            for line in source:
                target.write(line)
        temporary.replace(map_path)
    with map_path.open(encoding="utf-8") as handle:
        mapping = (json.loads(line) for line in handle if line.strip())
        for item, chunk in zip_longest(mapping, payload["chunks"]):
            if item is None or chunk is None or item.get("chunk_id") != chunk["chunk_id"]:
                raise ValueError(f"Source-map order/IDs differ from the BM25 index: {map_path}")
            if item.get("document_id") != chunk["document_id"]:
                raise ValueError(f"Source-map provenance differs: {map_path}")
    report = {
        "layout_version": 2, "chunks": len(payload["chunks"]),
        "chunks_sha256": sha256(chunks_path), "source_map_sha256": sha256(map_path),
        "index_sha256": sha256(company / "indexes/index.json"),
        "corpus_fingerprint": payload["corpus_fingerprint"],
    }
    (company / "layout_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def resume_settings(output: Path, current: dict, workspace: Path) -> dict:
    """Permit only an explicitly recorded storage-only migration of the old runner."""
    config_path = output / "configuration.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    if saved == current:
        return saved
    migration_path = output / "layout_migration.json"
    if not migration_path.is_file():
        raise ValueError("Build configuration/code differs; migrate legacy storage or use a new build version")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    if (migration.get("source_configuration_sha256") != sha256(config_path)
            or migration.get("compatible_builder_sha256") != sha256(workspace / BUILDER_PATH)):
        raise ValueError("Migration compatibility record does not match this configuration/builder")

    def algorithm_settings(settings):
        result = {k: v for k, v in settings.items() if k != "code_sha256"}
        result["code_sha256"] = {
            k.replace("\\", "/"): v for k, v in settings["code_sha256"].items()
            if k.replace("\\", "/") != BUILDER_PATH
        }
        return result

    if algorithm_settings(saved) != algorithm_settings(current):
        raise ValueError("Chunking/indexing/model inputs changed; migration cannot bypass a rebuild")
    # Keep the original processing identity and record the new layout separately.
    return saved
