"""Scope/signature guards for metadata-only same-folder extensions."""

from copy import deepcopy

import pytest

from scripts.narrative.chunking import build_c2_enriched as builder
from scripts.narrative.chunking.extend_c2_metadata import EXTENSION_KEY, extension_plan


def configs():
    saved = {
        "mode": "metadata", "cases_sha256": "unchanged-dataset", "embed": True,
        "selected_case_ids": ["case-a", "case-c"],
        "sources": {"AMZN": {"index": "original-index", "source_map": "original-map"}},
    }
    requested = deepcopy(saved)
    requested["selected_case_ids"] = ["case-a", "case-b", "case-c"]
    requested["sources"]["CRM"] = {"index": "new-index", "source_map": "new-map"}
    return saved, requested


def test_extension_preserves_old_company_signature_and_accepts_ordered_superset():
    saved, requested = configs()
    original, signatures = extension_plan(saved, requested)
    assert original == saved
    assert signatures == {"AMZN": builder.digest(saved)}
    assert requested["sources"]["AMZN"] == saved["sources"]["AMZN"]


def test_resume_keeps_distinct_old_and_new_company_signatures():
    old, current = configs()
    expected = {"AMZN": builder.digest(old), "CRM": builder.digest(current)}
    saved = {**current, EXTENSION_KEY: {
        "configuration_sha256": builder.digest(current),
        "company_configuration_sha256": expected,
    }}
    _, signatures = extension_plan(saved, deepcopy(current))
    assert signatures == expected


@pytest.mark.parametrize("change", ["dataset", "source", "removed_case", "embed", "blurb"])
def test_extension_refuses_changes_that_could_invalidate_existing_outputs(change):
    saved, requested = configs()
    if change == "dataset":
        requested["cases_sha256"] = "different-dataset"
    elif change == "source":
        requested["sources"]["AMZN"]["index"] = "changed-source-text"
    elif change == "removed_case":
        requested["selected_case_ids"] = ["case-b", "case-c"]
    elif change == "embed":
        requested["embed"] = False
    else:
        saved["mode"] = requested["mode"] = "blurb"
    with pytest.raises(ValueError):
        extension_plan(saved, requested)


def test_extension_refuses_a_tampered_signature_mapping():
    saved, requested = configs()
    saved[EXTENSION_KEY] = {"configuration_sha256": "incorrect", "company_configuration_sha256": {}}
    with pytest.raises(ValueError, match="signatures"):
        extension_plan(saved, requested)
