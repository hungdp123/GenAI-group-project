"""Layout changes preserve indexed evidence and do not bypass algorithm identities."""

from copy import deepcopy
import gzip
import json

import pytest

from scripts.narrative.maintenance.migrate_c4_storage import migrate
from scripts.narrative.retrieval.c4_storage import BUILDER_PATH, resume_settings, sha256, validate_output


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


def test_preview_has_no_writes_and_copy_preserves_evidence_and_checkpoints(legacy_build):
    workspace, source, destination, settings, report = legacy_build
    assert migrate(workspace, source, destination)["status"] == "PREVIEW"
    assert not destination.exists()
    assert migrate(workspace, source, destination, apply=True)["status"] == "COPIED_AND_VERIFIED"
    assert source.is_dir()  # Original retained for rollback.
    assert json.loads((destination / "configuration.json").read_text()) == settings
    assert json.loads((destination / "PLTR/build_report.json").read_text()) == report
    assert sha256(source / "PLTR/indexes/index.json") == sha256(destination / "PLTR/indexes/index.json")
    assert sha256(source / "PLTR/indexes/qdrant/storage/data.bin") == sha256(
        destination / "PLTR/indexes/qdrant/storage/data.bin")
    assert json.loads((destination / "PLTR/chunks.jsonl").read_text())["chunk_id"] == "original-id"
    assert json.loads((destination / "PLTR/source_map.jsonl").read_text())["document_id"] == "source-doc"
    with pytest.raises(FileExistsError):
        migrate(workspace, source, destination, apply=True)


def test_layout_migration_accepts_runner_change_but_rejects_changed_chunker(legacy_build):
    workspace, source, destination, saved, _ = legacy_build
    migrate(workspace, source, destination, apply=True)
    current = deepcopy(saved)
    current["code_sha256"][BUILDER_PATH] = sha256(workspace / BUILDER_PATH)
    assert resume_settings(destination, current, workspace) == saved
    current["code_sha256"]["adaptive_chunking.py"] = "changed-algorithm"
    with pytest.raises(ValueError, match="Chunking/indexing/model inputs changed"):
        resume_settings(destination, current, workspace)


def test_new_output_cannot_overwrite_canonical_company_folder(tmp_path):
    with pytest.raises(ValueError, match="chunking_strategies/C4"):
        validate_output(tmp_path, tmp_path / "data/narrative/processed/PLTR")
