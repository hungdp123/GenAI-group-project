"""Merge recovered source documents into the canonical narrative manifests.

The script verifies every copied file by SHA-256 before writing manifests and
removing the now-redundant per-company coverage_recovery files. Intake JSON
metadata is retained as the recovery audit trail.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import date
from pathlib import Path


ROOT = Path.cwd()
RAW = ROOT / "data/narrative/raw"
RECOVERY = ROOT / "data/narrative/intake/coverage_recovery/manifest_document_recovery.json"
EVENTS = ROOT / "data/narrative/intake/event_ir/manifest_event_reports.json"
OFFICIAL_MANIFEST = RAW / "manifest_official_ir.json"
EVENT_MANIFEST = RAW / "manifest_event_reports.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


recovery = read(RECOVERY)
event_payload = read(EVENTS)
official = read(OFFICIAL_MANIFEST)
event_main = read(EVENT_MANIFEST) if EVENT_MANIFEST.exists() else {
    "schema_version": 1,
    "source_policy": "Event reports collected from official company IR sites and public filing-provider copies; each document records its source tier.",
    "from_date": "2024-01-01",
    "cutoff_date": "2026-10-04",
    "generated_on": date.today().isoformat(),
    "documents": [],
}

# Index canonical content before adding anything. Duplicate source hashes are
# not re-added under a second document ID.
official_by_id = {x["document_id"]: x for x in official["documents"]}
official_hashes = {x.get("sha256", "").lower() for x in official["documents"]}
event_by_id = {x["document_id"]: x for x in event_main["documents"]}
event_hashes = {x.get("sha256", "").lower() for x in event_main["documents"]}
moved = []
counts = Counter()

for item in recovery["documents"]:
    ticker = item["company_ticker"].upper()
    source = ROOT / item["local_path"]
    if not source.is_file() or sha256(source) != item["sha256"].lower():
        raise RuntimeError(f"Recovery file missing/hash mismatch: {source}")
    kind = item["document_type"]
    if kind == "8-K":
        target_dir = RAW / ticker / "event_ir"
        target_manifest = event_by_id
        target_hashes = event_hashes
    else:
        target_dir = RAW / ticker / "official_ir"
        target_manifest = official_by_id
        target_hashes = official_hashes
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if item["document_id"] not in target_manifest and item["sha256"].lower() not in target_hashes:
        if target.exists() and sha256(target) != item["sha256"].lower():
            raise RuntimeError(f"Destination collision: {target}")
        if not target.exists():
            shutil.copy2(source, target)
            moved.append((source, target, item["sha256"].lower()))
        if sha256(target) != item["sha256"].lower():
            raise RuntimeError(f"Copied file hash mismatch: {target}")

        row = {
            "document_id": item["document_id"],
            "company_ticker": ticker,
            "company_name": item.get("company_name") or ticker,
            "document_type": kind,
            "period": item.get("period"),
            "publication_date": item.get("publication_date"),
            "date_precision": "day" if item.get("publication_date") else None,
            "publication_date_basis": item.get("publication_date_basis"),
            "retrieval_eligible": bool(item.get("publication_date")),
            "source_tier": item["source_tier"],
            "source_origin": item.get("publisher") or item.get("source_origin"),
            "publisher": item.get("publisher"),
            "source_url": item.get("source_url"),
            "source_page": item.get("source_page"),
            "local_path": target.relative_to(ROOT).as_posix(),
            "sha256": item["sha256"].lower(),
            "availability_status": "downloaded",
            "processing_status": "registered_not_processed",
            "label": item.get("label"),
            "content_role": item.get("content_role"),
        }
        if row["publication_date"] is None:
            row["retrieval_eligibility_reason"] = "Publication date not confirmed; retained in the canonical availability manifest."
        if kind == "8-K":
            original = item.get("original_local_path")
            if original:
                old_original = ROOT / original
                if not old_original.is_file() or sha256(old_original) != item.get("source_sha256", "").lower():
                    raise RuntimeError(f"Original provider file missing/hash mismatch: {old_original}")
                new_original = target_dir / old_original.name
                if new_original.exists() and sha256(new_original) != item.get("source_sha256", "").lower():
                    raise RuntimeError(f"Original provider destination collision: {new_original}")
                if not new_original.exists():
                    shutil.copy2(old_original, new_original)
                    moved.append((old_original, new_original, item["source_sha256"].lower()))
                row["original_local_path"] = new_original.relative_to(ROOT).as_posix()
                row["source_sha256"] = item.get("source_sha256")
            for key in ("form", "accession_number", "cik", "is_amendment", "source_ref", "primary_body_status", "event_date", "alternate_source_url"):
                if key in item:
                    row[key] = item[key]
        else:
            row["page_count_or_slides"] = item.get("pages_or_slides")
        target_manifest[row["document_id"]] = row
        target_hashes.add(item["sha256"].lower())
        counts[kind] += 1

# Reconcile the older 415-report intake into the same canonical event manifest.
for item in event_payload["documents"]:
    row = dict(item)
    row["availability_status"] = "downloaded"
    row["processing_status"] = "registered_not_processed"
    if row["document_id"] in event_by_id:
        existing = event_by_id[row["document_id"]]
        if existing.get("sha256") != row.get("sha256"):
            raise RuntimeError(f"Event document ID has conflicting hashes: {row['document_id']}")
        continue
    if row.get("sha256", "").lower() in event_hashes:
        raise RuntimeError(f"Unresolved duplicate event source hash: {row['document_id']}")
    for key in ("local_path", "original_local_path"):
        if row.get(key):
            p = ROOT / row[key]
            expected = row.get("sha256") if key == "local_path" else row.get("source_sha256")
            if not p.is_file() or sha256(p) != str(expected).lower():
                raise RuntimeError(f"Existing event file missing/hash mismatch: {p}")
    event_by_id[row["document_id"]] = row
    event_hashes.add(row.get("sha256", "").lower())

event_main["documents"] = sorted(event_by_id.values(), key=lambda r: (r.get("company_ticker", ""), r.get("publication_date", ""), r["document_id"]))
event_main["generated_on"] = date.today().isoformat()
event_main["document_count"] = len(event_main["documents"])
event_main["availability_scope"] = "Canonical event-report availability; includes main report bodies and locally preserved source files where available."
official["documents"] = sorted(official_by_id.values(), key=lambda r: (r.get("company_ticker", ""), r.get("publication_date") or "9999-99-99", r["document_id"]))
official["generated_on"] = date.today().isoformat()

# Update the retained recovery metadata to point at canonical locations and
# clearly record which main manifest owns each entry.
for item in recovery["documents"]:
    ticker = item["company_ticker"].upper()
    if item["document_type"] == "8-K":
        item["local_path"] = f"data/narrative/raw/{ticker}/event_ir/{Path(item['local_path']).name}"
        if item.get("original_local_path"):
            item["original_local_path"] = f"data/narrative/raw/{ticker}/event_ir/{Path(item['original_local_path']).name}"
        item["canonical_manifest"] = "data/narrative/raw/manifest_event_reports.json"
    else:
        item["local_path"] = f"data/narrative/raw/{ticker}/official_ir/{Path(item['local_path']).name}"
        item["canonical_manifest"] = "data/narrative/raw/manifest_official_ir.json"
    item["processing_status"] = "registered_not_processed"
    item["availability_status"] = "downloaded"
recovery["scope"] = "Retained audit record; document paths point to the canonical raw folders and records are registered in the main manifests."
recovery["canonical_registration_status"] = "merged_into_main_manifests"
recovery["documents"] = recovery["documents"]

event_payload["canonical_registration_status"] = "merged_into_main_manifest"
event_payload["canonical_manifest"] = "data/narrative/raw/manifest_event_reports.json"
event_payload["documents"] = [dict(r, processing_status="registered_not_processed") for r in event_payload["documents"]]

# Write main manifests only after all source files have passed hash checks.
official["documents"] = list(official_by_id.values())
write(OFFICIAL_MANIFEST, official)
write(EVENT_MANIFEST, event_main)
write(RECOVERY, recovery)
write(EVENTS, event_payload)

# Verify canonical references before deleting redundant recovery copies.
for manifest, rows in [(OFFICIAL_MANIFEST, official["documents"]), (EVENT_MANIFEST, event_main["documents"])]:
    for row in rows:
        if row.get("local_path"):
            path = ROOT / row["local_path"]
            if not path.is_file() or sha256(path) != row["sha256"].lower():
                raise RuntimeError(f"Canonical verification failed: {row['document_id']} -> {path}")

for source, target, expected in moved:
    if source != target and source.exists():
        if sha256(source) != expected or sha256(target) != expected:
            raise RuntimeError(f"Refusing to remove unverified duplicate: {source}")
        source.unlink()

# Remove only emptied per-company recovery directories; leave all intake
# metadata, candidate records, inventories, and comparison files intact.
for ticker_dir in RAW.iterdir():
    recovery_dir = ticker_dir / "coverage_recovery"
    if recovery_dir.is_dir():
        leftovers = list(recovery_dir.rglob("*"))
        if any(p.is_file() for p in leftovers):
            raise RuntimeError(f"Unmerged files remain in {recovery_dir}")
        recovery_dir.rmdir()

print(json.dumps({
    "added_presentations": counts["earnings_presentation"] + counts["investor_presentation"],
    "added_event_reports": counts["8-K"],
    "canonical_presentations": sum("presentation" in r.get("document_type", "") for r in official["documents"]),
    "canonical_event_reports": len(event_main["documents"]),
    "recovery_metadata_retained": True,
    "moved_files_hash_verified": len(moved),
}, indent=2))
