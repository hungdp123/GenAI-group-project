"""Validate and import manually downloaded official IR documents into a corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from finsight.agents.narrative.corpus.acquisition.download_corpus import _safe_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip files whose SHA-256 already exists instead of failing the import.",
    )
    args = parser.parse_args()

    metadata: dict[str, Any] = json.loads(args.metadata.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    from_date = date.fromisoformat(manifest["from_date"])
    cutoff_date = date.fromisoformat(manifest["cutoff_date"])
    inbox = args.metadata.parent
    corpus_root = args.manifest.parent

    known_hashes = {document["sha256"] for document in manifest.get("documents", [])}
    for excluded_path in args.exclude_manifest:
        excluded = json.loads(excluded_path.read_text(encoding="utf-8"))
        known_hashes.update(document["sha256"] for document in excluded.get("documents", []))

    imported: list[dict[str, Any]] = []
    duplicate_count = 0
    for record in metadata["documents"]:
        source = inbox / record["filename"]
        if not source.is_file():
            raise FileNotFoundError(f"Manual file not found: {source}")
        payload = source.read_bytes()
        if source.suffix.lower() == ".pdf" and not payload.startswith(b"%PDF"):
            raise ValueError(f"File does not have a PDF signature: {source}")
        digest = hashlib.sha256(payload).hexdigest()
        expected_hash = record.get("sha256")
        if expected_hash and expected_hash.lower() != digest:
            raise ValueError(f"SHA-256 mismatch for {source}")
        if digest in known_hashes:
            if args.skip_duplicates:
                duplicate_count += 1
                continue
            raise ValueError(f"Duplicate hash already exists in a corpus manifest: {source}")

        publication_date = date.fromisoformat(record["publication_date"])
        if not from_date <= publication_date <= cutoff_date:
            raise ValueError(f"Publication date outside corpus window: {source}")

        ticker = record["company_ticker"].upper()
        document_type = record["document_type"]
        canonical_folder = {
            "manifest_core_sec.json": "core_sec",
            "manifest_supplemental_sec.json": "supplemental_sec",
            "manifest_official_ir.json": "official_ir",
        }.get(args.manifest.name)
        if canonical_folder and corpus_root.name == "raw":
            target_dir = corpus_root / ticker / canonical_folder
        else:
            target_dir = corpus_root.parent / ticker / corpus_root.name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / _safe_name(
            f"{publication_date.isoformat()}_{document_type}_{source.name}"
        )
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Target exists with different content: {target}")
        shutil.copy2(source, target)

        imported.append({
            "document_id": f"{ticker}-IR-{digest[:16]}",
            "company_ticker": ticker,
            "document_type": document_type,
            "publication_date": publication_date.isoformat(),
            "date_precision": "day",
            "retrieval_eligible": True,
            "source_tier": 2,
            "source_origin": "Official company investor-relations website; manual browser download",
            "source_page": record["source_page"],
            "source_url": record["source_url"],
            "local_path": target.as_posix(),
            "sha256": digest,
            "link_text": record["link_text"],
        })
        known_hashes.add(digest)

    manifest["documents"].extend(imported)
    manifest["documents"].sort(key=lambda item: (
        item["company_ticker"], item.get("publication_date") or "", item["document_type"], item["document_id"]
    ))
    manifest["review_required"] = [
        document["document_id"] for document in manifest["documents"]
        if not document.get("retrieval_eligible", True)
    ]
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Imported {len(imported)} documents into {args.manifest}; "
        f"skipped {duplicate_count} existing hash(es)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
