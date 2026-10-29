"""Local Qdrant lock recovery and cleanup; no model downloads or API requests."""
from datetime import date
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from finsight.agents.narrative.retrieval import qdrant_index as vectors


def busy(storage):
    return RuntimeError(f"Storage folder {storage} is already accessed by another instance of Qdrant client. "
                        "If you require concurrent access, use Qdrant server instead.")


def test_waits_only_for_this_stores_native_lock(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    attempts, sleeps = [], []
    expected = object()
    def client(**kwargs):
        attempts.append(kwargs)
        if len(attempts) < 3:
            raise busy(storage)
        return expected
    monkeypatch.setattr(vectors.time, "sleep", sleeps.append)
    assert vectors._open_local_client(client, storage, 60) is expected
    assert len(attempts) == 3 and sleeps == [.2, .2]


@pytest.mark.parametrize("message", [
    "Invalid collection",
    "Storage folder unrelated is already accessed by another instance of Qdrant client.",
])
def test_other_runtime_errors_are_not_retried(tmp_path, monkeypatch, message):
    def client(**kwargs):
        raise RuntimeError(message)
    monkeypatch.setattr(vectors.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    with pytest.raises(RuntimeError, match=message):
        vectors._open_local_client(client, tmp_path / "storage", 60)


def test_lock_wait_is_bounded_and_does_not_remove_lock(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    lock = storage / ".lock"
    lock.write_text("owned by another client", encoding="utf-8")
    def client(**kwargs):
        raise busy(storage)
    with pytest.raises(RuntimeError, match="Timed out after 0s"):
        vectors._open_local_client(client, storage, 0)
    assert lock.read_text(encoding="utf-8") == "owned by another client"


@pytest.mark.parametrize("failure", ["collection", "count", "dimension"])
def test_validation_failures_release_the_client(tmp_path, monkeypatch, failure):
    payload = {"corpus_fingerprint": "test", "chunks": [{"chunk_id": "a"}]}
    metadata = {"schema_version": "1.0", "corpus_fingerprint": "test", "embedding_model": "fake",
                "dimension": 2, "chunk_ids": ["a"], "collection_name": "narrative_chunks"}
    (tmp_path / "qdrant_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    closed = []
    class Client:
        def __init__(self, **kwargs):
            pass
        def get_collection(self, name):
            if failure == "collection":
                raise RuntimeError("Broken collection")
            return SimpleNamespace(points_count=0 if failure == "count" else 1,
                                   config=SimpleNamespace(params=SimpleNamespace(
                                       vectors=SimpleNamespace(size=3 if failure == "dimension" else 2))))
        def close(self):
            closed.append(True)
    monkeypatch.setattr(vectors, "_qdrant_modules", lambda: (Client, None))
    with pytest.raises((RuntimeError, ValueError)):
        vectors.load_qdrant_index(tmp_path, payload)
    assert closed == [True]


def test_real_qdrant_process_lock_releases_then_search_preserves_results(tmp_path):
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    class Embeddings:
        model_name = "offline-test-vectors"
        def encode_documents(self, texts):
            return [[1., 0.], [0., 1.]]
    chunks = [{"chunk_id": name, "text": name, "company_ticker": "AAPL", "publication_date": "2024-01-01",
               "document_id": name, "document_type": "earnings_release", "source_tier": "TIER_2",
               "source_url": "https://example.test/" + name, "citation_span": "paragraph"}
              for name in ["alpha", "beta"]]
    payload = {"corpus_fingerprint": "unchanged", "chunks": chunks}
    vectors.build_qdrant_index(payload, tmp_path, Embeddings())
    metadata = (tmp_path / "qdrant_metadata.json").read_bytes()
    held, release = tmp_path / "held", tmp_path / "release"
    script = tmp_path / "hold_store.py"
    script.write_text(
        "import sys,time\nfrom pathlib import Path\nfrom qdrant_client import QdrantClient\n"
        "client=QdrantClient(path=sys.argv[1])\n"
        "try:\n Path(sys.argv[2]).write_text('held')\n deadline=time.monotonic()+15\n"
        " while not Path(sys.argv[3]).exists() and time.monotonic()<deadline: time.sleep(.02)\n"
        "finally: client.close()\n", encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(script), str(tmp_path / "storage"), str(held), str(release)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    timer = None
    try:
        deadline = time.monotonic() + 10
        while not held.exists() and time.monotonic() < deadline and child.poll() is None:
            time.sleep(.02)
        assert held.exists(), "Child did not acquire the test storage"
        with pytest.raises(RuntimeError, match="already accessed"):
            QdrantClient(path=str(tmp_path / "storage"))
        timer = threading.Timer(.3, lambda: release.write_text("release", encoding="utf-8"))
        timer.start()
        loaded = vectors.load_qdrant_index(tmp_path, payload, lock_timeout=5)
        try:
            hits = vectors.search_qdrant(loaded, payload, [1., 0.], top_k=2,
                                         cutoff_date=date(2024, 1, 2), company_ticker="AAPL")
            assert [h["chunk_id"] for h in hits] == ["alpha", "beta"]
        finally:
            loaded.client.close()
        assert (tmp_path / "qdrant_metadata.json").read_bytes() == metadata
    finally:
        if timer:
            timer.cancel()
        release.write_text("release", encoding="utf-8")
        try:
            child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()
