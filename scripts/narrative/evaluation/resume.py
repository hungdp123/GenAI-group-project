"""Validate append-only case extensions and preserve previous report scope."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


# Audited predecessors: this update changes capture/resume bookkeeping, not RAG behavior.
PREVIOUS_MATRIX_SHA = "f20cd4bbef075e62a71a1c5cc6b82515f694a0ee346bcbeb30197240bce57a7b"
PREVIOUS_CAPTURE_SHA = "e47906cac9a74e9eda189c6e61faf91acfed4cdb874afa96cb3c727b7ae8083a"

# Exact before/after pairs for the standalone report-folder rename only.
# No general exemption for future retrieval, prompt or model-code changes.
OUTPUT_PATH_MIGRATIONS = {
    ("basic_rag.py", "81547867f47fe5c08063c8b675b64c5dbe3b5f8b3fcb09cfe8efab0aa627217f",
     "6a6d9c5dfd0e640f42430f6d505c6a5e12e9e9359770a238db6e584f55e31094"),
    ("verdict_baseline.py", "fb3189a6391e70965d00fa9bfc9f25561fe25bd968a1ab3cf0ac80010d3f84e7",
     "2fb90c5b246ce132ae166ec68f7c822013ec0db0660ce07925c238475d97e0b9"),
}


# Exact default-input-path edits only; saved reports retain their original provenance.
DATASET_PATH_MIGRATIONS = {
    ("run_rag_comparison.py", "82881c8518e07013c5f06eecd55df46ea9b355f43d2cc231f9344f4e62f725d3",
     "3f4436d2d8d21d4f9e8d2e4b4c6704e86d6943130a4d44809eda6c6385c08b0b"),
    ("run_rag_pilot.py", "72dc3d5ce0cdb99d35bb92db97687c81d8046dfc1919de08ec632d09e20d519e",
     "4792cbf0cb876a941c5a478d3bde801262fbeadd92770a6c880b531bf8bb85de"),
}


def _dataset_path(value: str) -> str:
    value = value.replace("\\", "/")
    old = "data/narrative/evaluation/narrative_claims_v1/cases.csv"
    if value == old or value.endswith("/" + old):
        return value[:-len(old)] + "data/narrative/evaluation/test_claims_v1/cases.csv"
    return value


def _settings_match(previous: dict, current: dict, ignored: set[str]) -> bool:
    old = {k: v for k, v in previous.items() if k not in ignored}
    new = {k: v for k, v in current.items() if k not in ignored}
    if old.get("dataset") != new.get("dataset"):
        digest = old.get("dataset_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or digest != new.get("dataset_sha256")
                or not isinstance(old.get("dataset"), str)
                or not isinstance(new.get("dataset"), str)
                or _dataset_path(old["dataset"]) != _dataset_path(new["dataset"])):
            return False
        old["dataset"] = new["dataset"]
    return old == new


def _prefix(previous: list, current: list) -> bool:
    return bool(previous) and len(previous) <= len(current) and previous == current[:len(previous)]


def matrix_compatible(previous: dict, current: dict) -> bool:
    ignored = {"selected_case_ids", "case_count_per_configuration", "runner_sha256"}
    return (
        _settings_match(previous, current, ignored)
        and _prefix(previous["selected_case_ids"], current["selected_case_ids"])
        and previous["case_count_per_configuration"] == len(previous["selected_case_ids"])
        and (previous["runner_sha256"] in {current["runner_sha256"], PREVIOUS_MATRIX_SHA}
             or ("run_rag_comparison.py", previous["runner_sha256"], current["runner_sha256"])
             in DATASET_PATH_MIGRATIONS)
    )


def capture_compatible(previous: dict, current: dict, saved_cases: list, selected: list) -> bool:
    from scripts.narrative.evaluation.evaluate_rag_pilot import case_identity

    ignored = {"created_utc", "scope", "case_ids", "selected_cases_sha256", "code_hashes"}
    if not _settings_match(previous, current, ignored):
        return False
    if (not _prefix(saved_cases, selected)
            or previous["case_ids"] != [c["id"] for c in saved_cases]
            or previous["selected_cases_sha256"] != case_identity(saved_cases)):
        return False
    old_code, new_code = previous["code_hashes"], current["code_hashes"]
    if old_code.keys() != new_code.keys():
        return False
    return all(old_code[k] == new_code[k] or (
        Path(k).name == "run_rag_pilot.py" and old_code[k] == PREVIOUS_CAPTURE_SHA
    ) or (Path(k).name, old_code[k], new_code[k]) in (
        OUTPUT_PATH_MIGRATIONS | DATASET_PATH_MIGRATIONS
    ) for k in old_code)


def archive_state(output: Path, previous: dict) -> None:
    """Keep reports/configuration from the previous scope; never move case artifacts."""
    identity = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()[:16]
    destination = output / "history" / identity
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("configuration.json", "cases.csv", "report.json", "evaluation.json", "evaluation.md",
                 "matrix_configuration.json", "matrix_report.json", "comparison.json", "comparison.md",
                 "comparison.csv", "per_company.csv", "per_case.csv"):
        source = output / name
        target = destination / name
        if source.is_file() and not target.exists():
            shutil.copyfile(source, target)
