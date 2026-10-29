"""Validate append-only case extensions and preserve previous report scope."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4


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

# Exact runner-only recovery changes: explicit failed-attempt archiving/retry and
# stopping at the spend guard. Retrieval, prompts and generation settings are unchanged.
RECOVERY_CODE_MIGRATIONS = {
    ("run_rag_comparison.py", "3f4436d2d8d21d4f9e8d2e4b4c6704e86d6943130a4d44809eda6c6385c08b0b",
     "f5897f0eb6cbd60efa264f0b7996d99f8993e475047c22c1797ba83dbf11e021"),
    ("run_rag_pilot.py", "4792cbf0cb876a941c5a478d3bde801262fbeadd92770a6c880b531bf8bb85de",
     "34d91cfa254c69000e4ae2be188a2ca94c83d4881ac347cfb0c382063000f815"),
}

# CLI defaults only. Effective model/provider settings must still match the run;
# historical API generation now requires --shared-llm explicitly.
LOCAL_DEFAULT_CODE_MIGRATIONS = {
    ("run_rag_comparison.py", "f5897f0eb6cbd60efa264f0b7996d99f8993e475047c22c1797ba83dbf11e021",
     "dd91b756b076d57a9b5ad820abc205cf6e5e5746e8f10b111e9901949445fe9a"),
}

# The optional saved-retrieval branch leaves fresh-retrieval behavior unchanged.
# Settings still have to match: an existing live-retrieval run cannot switch modes.
SAVED_RETRIEVAL_CODE_MIGRATIONS = {
    ("run_rag_comparison.py", "3eefe3fe512ae8fd14615dbe0cc59ea111d005e2ef7c0bb67b851023df2636ac",
     "8a24c536501e6813ac56b28da1a580ed3a63281107316ed55864c4780bfaea5f"),
    ("run_rag_pilot.py", "d614fc8d5d8c51362e2f82cf5042e81a7aef74db44fc3b31f9b6dfdcacb9d344",
     "3b228895823792251b1260150f9d225d6662b6b8b273552282d25dc728a58631"),
}

# Optional row-range selection; default prefix selection and generation are unchanged.
# Existing snapshot-prefix and settings checks still govern any scope extension.
CASE_RANGE_CODE_MIGRATIONS = {
    ("run_rag_comparison.py", "8a24c536501e6813ac56b28da1a580ed3a63281107316ed55864c4780bfaea5f",
     "0d0ef60e8f9dcc5c912c2f5aef4f5b7bd20846afbf86c315df52b84455400792"),
    ("run_rag_pilot.py", "3b228895823792251b1260150f9d225d6662b6b8b273552282d25dc728a58631",
     "6061f91902c8a22e70e44847b4f640d4ad58b9e8360c77ff6e0be53a4b181f8c"),
}

# Exact LF/CRLF byte variants of reviewed source versions. Git uses LF on Linux
# and commonly CRLF on Windows. Keep raw hashes in captures; do not normalize
# arbitrary code or accept future unreviewed changes just because it is Python.
LINE_ENDING_EQUIVALENTS = {
    ("run_rag_comparison.py", "20706cd4176ff7703c8208fe0749ad36cae871442b65fd7531bf3fbc494d2652",
     "0d0ef60e8f9dcc5c912c2f5aef4f5b7bd20846afbf86c315df52b84455400792"),
    ("run_rag_pilot.py", "e74c9162ccac49475595566cb18192ca3abdeacd85747a814fda38168a490184",
     "6061f91902c8a22e70e44847b4f640d4ad58b9e8360c77ff6e0be53a4b181f8c"),
    ("basic_rag.py", "98c4f75a1f6a64e4e6715f21a25cff69fc24486558b504da9177c4adccf6ada9",
     "315a9abd2c2ebc5b708035d43aab3e71e699f4e6b25d909f6ab63f3b3a004b00"),
    ("basic_rag.py", "9dfc6886495b464420e852b27ed220088bf7c396d055d503e2cf7d0dbd2ef542",
     "7c38fc3751b37bf6a85d010ca09f436794e664396339a3540526b04e909ad93b"),
    ("evidence_summary.py", "c524491427f960a2351416c5e89095542e16098b5eecdceb9fab6bf821c71a10",
     "9c9191df335b68353d676966cc4515713984413273c9aa37f588c773e2e89f76"),
    ("evidence_summary.py", "a54d469b76e0a2a97a38647456c094b44606c03b354d7a900d24359fffec3767",
     "9e71cc27dc0af5cdde713ba351a9d263f8d4ce74cf93a52983e9256175e3fd5c"),
}

# Ruff line wrapping only. CLI help, prompts, output strings and executable ASTs
# are unchanged; allow saved runs to resume across these exact formatting edits.
FORMATTING_CODE_MIGRATIONS = {
    ("basic_rag.py", "98c4f75a1f6a64e4e6715f21a25cff69fc24486558b504da9177c4adccf6ada9",
     "9dfc6886495b464420e852b27ed220088bf7c396d055d503e2cf7d0dbd2ef542"),
    ("evidence_summary.py", "c524491427f960a2351416c5e89095542e16098b5eecdceb9fab6bf821c71a10",
     "a54d469b76e0a2a97a38647456c094b44606c03b354d7a900d24359fffec3767"),
}

# The source-span validator fix admits blank-only source segments omitted by the
# existing generation protocol. Previously accepted samples and judge prompts are unchanged.
EVALUATOR_INPUT_MIGRATIONS = {
    ("1c9676ac356c40e0c169f5de257aee954a56571ae7b35cdd161abcea677c19b0",
     "ef6e65e087ba0df9820ca5d6738eda7dde096681bc0349762f286cef46a7b3cc"),
    # Local-default CLI selection only; explicit shared/local settings are unchanged.
    ("ef6e65e087ba0df9820ca5d6738eda7dde096681bc0349762f286cef46a7b3cc",
     "b1055cbc5505a60219d6ff397f1aabc739e3c557eb45c3af3aebb7be5acf8229"),
}


def evaluator_compatible(previous: dict, current: dict) -> bool:
    if previous == current:
        return True
    reachable = {previous.get("code_sha256")}
    for _ in EVALUATOR_INPUT_MIGRATIONS:
        reachable.update(new for old, new in EVALUATOR_INPUT_MIGRATIONS if old in reachable)
    return (
        current.get("code_sha256") in reachable
        and {k: v for k, v in previous.items() if k != "code_sha256"}
        == {k: v for k, v in current.items() if k != "code_sha256"}
    )


def _migrated(name: str, previous: str, current: str) -> bool:
    reachable = {previous}
    migrations = (OUTPUT_PATH_MIGRATIONS | DATASET_PATH_MIGRATIONS | RECOVERY_CODE_MIGRATIONS
                  | LOCAL_DEFAULT_CODE_MIGRATIONS | SAVED_RETRIEVAL_CODE_MIGRATIONS
                  | CASE_RANGE_CODE_MIGRATIONS | FORMATTING_CODE_MIGRATIONS
                  | LINE_ENDING_EQUIVALENTS
                  | {(module, crlf, lf) for module, lf, crlf in LINE_ENDING_EQUIVALENTS})
    for _ in migrations:
        reachable.update(new for module, old, new in
                         migrations
                         if module == name and old in reachable)
    return current in reachable


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
             or _migrated("run_rag_comparison.py", previous["runner_sha256"], current["runner_sha256"]))
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
    ) or _migrated(Path(k).name, old_code[k], new_code[k]) for k in old_code)


def archive_generation_error(folder: Path) -> Path:
    """Copy failed attempt artifacts before clearing only the active attempt files."""
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    if result.get("status") != "GENERATION_ERROR":
        raise ValueError("Only an explicit GENERATION_ERROR can be retried")
    destination = folder / "attempts" / uuid4().hex
    destination.mkdir(parents=True)
    names = ("result.json", "retrieval.json", "llm_input.json", "llm_response.json")
    for name in (*names, "preflight.json"):
        source = folder / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    # All copies succeeded first. An interrupted clear retains the archived originals;
    # the runner refuses orphaned LLM captures rather than issuing an unnoticed duplicate.
    for name in names:
        (folder / name).unlink(missing_ok=True)
    return destination


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
