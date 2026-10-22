"""Canonical narrative-corpus loading and validation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable


CANONICAL_MANIFESTS = (
    Path("data/narrative/raw/manifest_core_sec.json"),
    Path("data/narrative/raw/manifest_supplemental_sec.json"),
    Path("data/narrative/raw/manifest_official_ir.json"),
    Path("data/narrative/raw/manifest_third_party_transcripts.json"),
    Path("data/narrative/raw/manifest_event_reports.json"),
)


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    company_ticker: str
    company_name: str
    document_type: str
    publication_date: date
    source_tier: str
    source_url: str
    local_path: Path
    sha256: str
    period: str | None = None
    retrieval_eligible: bool = True
    original_local_path: Path | None = None
    publication_date_is_estimated: bool = False
    publication_date_precision: str = "day"
    publication_date_basis: str | None = None

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["publication_date"] = self.publication_date.isoformat()
        value["local_path"] = self.local_path.as_posix()
        value["original_local_path"] = (
            self.original_local_path.as_posix() if self.original_local_path else None
        )
        return value


def _source_tier(value: object) -> str:
    text = str(value).upper()
    if text in {"1", "TIER_1"}:
        return "TIER_1"
    if text in {"2", "TIER_2"}:
        return "TIER_2"
    if text in {"3", "TIER_3"}:
        return "TIER_3"
    raise ValueError(f"Unsupported source tier: {value!r}")


def load_documents(
    ticker: str,
    manifests: Iterable[Path] = CANONICAL_MANIFESTS,
    root: Path | None = None,
) -> list[CorpusDocument]:
    """Load retrieval-eligible documents for one ticker from canonical manifests."""
    workspace = (root or Path.cwd()).resolve()
    ticker = ticker.upper()
    documents: list[CorpusDocument] = []
    for manifest in manifests:
        manifest_path = manifest if manifest.is_absolute() else workspace / manifest
        if manifest_path.name == "manifest_third_party_transcripts.json" and not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("documents", []):
            if str(item.get("company_ticker", "")).upper() != ticker:
                continue
            exact_date = item.get("publication_date")
            estimated_date = item.get("estimated_publication_date")
            if item.get("retrieval_eligible") is False and not estimated_date:
                continue
            effective_date = exact_date or estimated_date
            if not effective_date:
                continue
            date_is_estimated = not bool(exact_date) and bool(estimated_date)
            relative_path = Path(item["local_path"])
            local_path = relative_path if relative_path.is_absolute() else workspace / relative_path
            original_relative_path = item.get("original_local_path")
            original_local_path = None
            if original_relative_path:
                original_relative_path = Path(original_relative_path)
                original_local_path = (
                    original_relative_path
                    if original_relative_path.is_absolute()
                    else workspace / original_relative_path
                )
            documents.append(
                CorpusDocument(
                    document_id=item["document_id"],
                    company_ticker=ticker,
                    company_name=item.get("company_name") or ticker,
                    document_type=item["document_type"],
                    publication_date=date.fromisoformat(effective_date),
                    source_tier=_source_tier(item["source_tier"]),
                    source_url=item["source_url"],
                    local_path=local_path.resolve(),
                    sha256=item["sha256"].lower(),
                    period=item.get("period"),
                    retrieval_eligible=True,
                    original_local_path=original_local_path.resolve() if original_local_path else None,
                    publication_date_is_estimated=date_is_estimated,
                    publication_date_precision=(
                        "estimated_day" if date_is_estimated else item.get("date_precision") or "day"
                    ),
                    publication_date_basis=(
                        item.get("estimated_publication_date_basis")
                        if date_is_estimated
                        else item.get("publication_date_basis")
                    ),
                )
            )
    canonical_name = next(
        (document.company_name for document in documents if document.company_name != ticker),
        ticker,
    )
    normalized = [
        replace(document, company_name=canonical_name)
        if document.company_name == ticker
        else document
        for document in documents
    ]
    return sorted(normalized, key=lambda d: (d.publication_date, d.document_id))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_document(document: CorpusDocument) -> None:
    if not document.local_path.is_file():
        raise FileNotFoundError(f"Missing corpus file: {document.local_path}")
    actual = file_sha256(document.local_path)
    if actual != document.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {document.document_id}: expected {document.sha256}, got {actual}"
        )


def corpus_fingerprint(documents: Iterable[CorpusDocument]) -> str:
    rows = sorted(
        json.dumps(
            {
                "document_id": doc.document_id,
                "sha256": doc.sha256,
                "publication_date": doc.publication_date.isoformat(),
                "publication_date_is_estimated": doc.publication_date_is_estimated,
                "publication_date_precision": doc.publication_date_precision,
                "source_tier": doc.source_tier,
                "source_url": doc.source_url,
            },
            sort_keys=True,
        )
        for doc in documents
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
