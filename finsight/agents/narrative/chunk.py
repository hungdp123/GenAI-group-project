"""Section-aware, deterministic narrative-document chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from finsight.agents.narrative.ingest import ProcessedBlock


_SLUG = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class NarrativeChunk:
    chunk_id: str
    document_id: str
    company_ticker: str
    company_name: str
    document_type: str
    publication_date: str
    source_tier: str
    source_url: str
    local_path: str
    period: str | None
    section: str
    page: int | None
    citation_span: str
    text: str
    text_sha256: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _section_slug(section: str) -> str:
    return _SLUG.sub("-", section.lower()).strip("-")[:48] or "section"


def chunk_blocks(
    blocks: Iterable[ProcessedBlock], target_words: int = 800, overlap_words: int = 120
) -> list[NarrativeChunk]:
    if target_words < 1 or overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("Require target_words > overlap_words >= 0")
    chunks: list[NarrativeChunk] = []
    ordinals: dict[tuple[str, str], int] = {}
    step = target_words - overlap_words
    for block in blocks:
        words = block.text.split()
        for start in range(0, len(words), step):
            window = words[start : start + target_words]
            if not window:
                continue
            text = " ".join(window)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            key = (block.document_id, block.section)
            ordinal = ordinals.get(key, 0) + 1
            ordinals[key] = ordinal
            chunk_id = (
                f"{block.document_id}:{_section_slug(block.section)}:"
                f"{ordinal:03d}:{text_hash[:12]}"
            )
            chunks.append(
                NarrativeChunk(
                    chunk_id=chunk_id,
                    document_id=block.document_id,
                    company_ticker=block.company_ticker,
                    company_name=block.company_name,
                    document_type=block.document_type,
                    publication_date=block.publication_date,
                    source_tier=block.source_tier,
                    source_url=block.source_url,
                    local_path=block.local_path,
                    period=block.period,
                    section=block.section,
                    page=block.page,
                    citation_span=block.citation_span,
                    text=text,
                    text_sha256=text_hash,
                )
            )
            if start + target_words >= len(words):
                break
    return chunks
