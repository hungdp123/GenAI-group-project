"""One saved question, so a later date can finish the claim that named a bad cut-off."""

from __future__ import annotations

import json
from pathlib import Path

from finsight.agents.orchestrator.routing import lone_date, replace_cutoff


def remember_cutoff(logs_dir: Path, prompt: str) -> None:
    path = _path(logs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "cutoff", "prompt": prompt}), encoding="utf-8")


def claim_with_saved_cutoff(logs_dir: Path, text: str) -> str | None:
    """A reply that is only a date continues the saved claim."""
    if lone_date(text) is None:
        return None
    path = _path(logs_dir)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    prompt = str(record.get("prompt") or "")
    if record.get("kind") != "cutoff" or not prompt:
        return None
    path.unlink(missing_ok=True)
    return replace_cutoff(prompt, text.strip())


def _path(logs_dir: Path) -> Path:
    return Path(logs_dir) / "pending_check.json"
