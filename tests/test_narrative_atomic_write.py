"""Checkpoint publication and Windows lock recovery, without provider calls."""
import json
from pathlib import Path

import pytest

from scripts.narrative.evaluation import evaluate_rag_pilot as scoring


def locked(code=5):
    error = PermissionError("Simulated Windows file lock")
    error.winerror = code
    return error


@pytest.mark.parametrize("code", [5, 32, 33])
def test_checkpoint_retries_transient_lock_without_truncating_old_report(tmp_path, monkeypatch, code):
    target = tmp_path / "report.json"
    target.write_text('{"old": true}\n', encoding="utf-8")
    legacy = tmp_path / "report.json.tmp"
    legacy.write_text("preserved interrupted payload", encoding="utf-8")
    replace = Path.replace
    attempts, sleeps = [], []

    def temporarily_locked(source, destination):
        attempts.append(source)
        assert json.loads(source.read_text(encoding="utf-8")) == {"new": "complete"}
        assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
        if len(attempts) < 3:
            raise locked(code)
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr(scoring.time, "sleep", sleeps.append)
    scoring.write_json(target, {"new": "complete"})
    assert len(attempts) == 3 and sleeps == [.1, .2]
    assert len(set(attempts)) == 1 and attempts[0] != legacy
    assert not attempts[0].exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": "complete"}
    assert legacy.read_text(encoding="utf-8") == "preserved interrupted payload"


def test_persistent_lock_preserves_last_good_report_and_new_payload(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    target.write_text('{"saved_cases": 3}\n', encoding="utf-8")
    attempts = []
    def always_locked(source, destination):
        attempts.append(source)
        raise locked()
    monkeypatch.setattr(Path, "replace", always_locked)
    monkeypatch.setattr(scoring.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        scoring.write_json(target, {"saved_cases": 4})
    assert len(attempts) == 8
    assert json.loads(target.read_text(encoding="utf-8")) == {"saved_cases": 3}
    assert json.loads(attempts[0].read_text(encoding="utf-8")) == {"saved_cases": 4}


def test_distinct_writes_use_distinct_staging_paths(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    replace = Path.replace
    staged = []
    def publish(source, destination):
        staged.append(source)
        assert source.parent == target.parent and source != target
        return replace(source, destination)
    monkeypatch.setattr(Path, "replace", publish)
    scoring.write_json(target, {"case": 1})
    scoring.write_json(target, {"case": 2})
    assert len(set(staged)) == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"case": 2}


def test_other_permission_errors_are_not_retried(tmp_path, monkeypatch):
    def deny(source, destination):
        raise PermissionError("Permanent permission error without a Windows lock code")
    monkeypatch.setattr(Path, "replace", deny)
    monkeypatch.setattr(scoring.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    with pytest.raises(PermissionError):
        scoring.write_json(tmp_path / "report.json", {"case": 1})
