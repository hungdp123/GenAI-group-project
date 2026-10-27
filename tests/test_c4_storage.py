"""Current builders preserve source records and enforce saved build identities."""

from copy import deepcopy
import gzip
import json

import pytest

from scripts.narrative.retrieval.c4_storage import (
    BUILDER_PATH, resume_settings, sha256, validate_output, write_chunk_artifacts,
)


@pytest.fixture
def legacy_build(tmp_path):
    source = tmp_path / "data/narrative/runtime/c4_full_v1"
    destination = tmp_path / "data/narrative/processed/chunking_strategies/C4/full_v1"
    builder = tmp_path / BUILDER_PATH
    builder.parent.mkdir(parents=True)
    builder.write_text("current storage-only runner", encoding="utf-8")
    settings = {"strategy": "C4", "publication_cutoff": None, "config": {"token_targets": [256]},
                "code_sha256": {BUILDER_PATH: "legacy-runner", "adaptive_chunking.py": "same-algorithm"}}
    company = source / "PLTR"
    index = company / "indexes/index.json"
    index.parent.mkdir(parents=True)
    (source / "configuration.json").write_text(json.dumps(settings), encoding="utf-8")
    payload = {"schema_version": "1.0", "corpus_fingerprint": "fingerprint",
               "chunking": {"strategy": "adaptive"},
               "chunks": [{"chunk_id": "original-id", "document_id": "source-doc", "text": "73 deals"}]}
    index.write_text(json.dumps(payload), encoding="utf-8")
    report = {"index_sha256": sha256(index), "inputs": {"settings_sha256": "original-processing-identity"}}
    for name in ("chunking_report.json", "build_report.json"):
        (company / name).write_text(json.dumps(report), encoding="utf-8")
    with gzip.open(company / "source_map.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"chunk_id": "original-id", "document_id": "source-doc"}) + "\n")
    vectors = index.parent / "qdrant"
    vectors.mkdir()
    (vectors / "qdrant_metadata.json").write_text(json.dumps({
        "corpus_fingerprint": "fingerprint", "chunk_ids": ["original-id"], "vector_count": 1,
    }), encoding="utf-8")
    (vectors / "storage").mkdir()
    (vectors / "storage/data.bin").write_bytes(b"existing vector bytes")
    return tmp_path, source, destination, settings, report


def test_chunk_export_preserves_index_vectors_and_source_maps(legacy_build):
    _, source, _, _, original_report = legacy_build
    company = source / "PLTR"
    index = company / "indexes/index.json"
    vector_bytes = (company / "indexes/qdrant/storage/data.bin").read_bytes()
    original_index_hash = sha256(index)
    payload = json.loads(index.read_text())
    report = write_chunk_artifacts(company, payload)
    assert json.loads((company / "chunks.jsonl").read_text()) == payload["chunks"][0]
    assert json.loads((company / "source_map.jsonl").read_text())["document_id"] == "source-doc"
    assert report["index_sha256"] == original_index_hash == sha256(index)
    assert (company / "indexes/qdrant/storage/data.bin").read_bytes() == vector_bytes
    assert json.loads((company / "build_report.json").read_text()) == original_report
    assert write_chunk_artifacts(company, payload) == report
    changed = deepcopy(payload)
    changed["chunks"][0]["text"] = "different source text"
    with pytest.raises(ValueError, match="Chunk records differ"):
        write_chunk_artifacts(company, changed)


def test_recorded_layout_compatibility_cannot_bypass_changed_chunker(legacy_build):
    workspace, source, _, saved, _ = legacy_build
    current = deepcopy(saved)
    current["code_sha256"][BUILDER_PATH] = sha256(workspace / BUILDER_PATH)
    assert resume_settings(source, saved, workspace) == saved
    with pytest.raises(ValueError, match="Build configuration/code differs"):
        resume_settings(source, current, workspace)
    # Existing compatibility records are still checked by current builders.
    record = {"source_configuration_sha256": sha256(source / "configuration.json"),
              "compatible_builder_sha256": sha256(workspace / BUILDER_PATH)}
    (source / "layout_migration.json").write_text(json.dumps(record), encoding="utf-8")
    assert resume_settings(source, current, workspace) == saved
    current["code_sha256"]["adaptive_chunking.py"] = "changed-algorithm"
    with pytest.raises(ValueError, match="Chunking/indexing/model inputs changed"):
        resume_settings(source, current, workspace)


def test_new_output_cannot_overwrite_canonical_company_folder(tmp_path):
    with pytest.raises(ValueError, match="chunking_strategies/C4"):
        validate_output(tmp_path, tmp_path / "data/narrative/processed/PLTR")
