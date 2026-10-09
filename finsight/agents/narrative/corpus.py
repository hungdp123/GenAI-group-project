"""Canonical narrative-corpus loading and validation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Iterable


CANONICAL_MANIFESTS = (
    Path("data/narrative/raw/from_2021_final/manifest.json"),
    Path("data/narrative/raw/supplemental_from_2021_official_final/manifest.json"),
    Path("data/narrative/raw/ir_from_2021_official_final/manifest.json"),
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

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["publication_date"] = self.publication_date.isoformat()
        value["local_path"] = self.local_path.as_posix()
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
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("documents", []):
            if str(item.get("company_ticker", "")).upper() != ticker:
                continue
            if item.get("retrieval_eligible") is False:
                continue
            relative_path = Path(item["local_path"])
            local_path = relative_path if relative_path.is_absolute() else workspace / relative_path
            documents.append(
                CorpusDocument(
                    document_id=item["document_id"],
                    company_ticker=ticker,
                    company_name=item.get("company_name") or ticker,
                    document_type=item["document_type"],
                    publication_date=date.fromisoformat(item["publication_date"]),
                    source_tier=_source_tier(item["source_tier"]),
                    source_url=item["source_url"],
                    local_path=local_path.resolve(),
                    sha256=item["sha256"].lower(),
                    period=item.get("period"),
                    retrieval_eligible=True,
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
    rows = sorted(f"{doc.document_id}:{doc.sha256}" for doc in documents)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
