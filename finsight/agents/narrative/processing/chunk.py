"""Section-aware, deterministic narrative-document chunking."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from finsight.agents.narrative.processing.ingest import ProcessedBlock


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
    source_format: str = "unknown"
    extraction_method: str = "unknown"
    content_type: str = "narrative"
    chart_id: str | None = None
    validation_status: str = "native_text"
    source_ref: str | None = None
    original_local_path: str | None = None
    publication_date_is_estimated: bool = False
    publication_date_precision: str = "day"
    publication_date_basis: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _section_slug(section: str) -> str:
    return _SLUG.sub("-", section.lower()).strip("-")[:48] or "section"


def chunk_blocks(
    blocks: Iterable[ProcessedBlock], target_words: int = 800, overlap_words: int = 120,
    *, boundary_policy: str = "word",
) -> list[NarrativeChunk]:
    if target_words < 1 or overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("Require target_words > overlap_words >= 0")
    if boundary_policy not in {"word", "paragraph"}:
        raise ValueError("boundary_policy must be word or paragraph")
    chunks: list[NarrativeChunk] = []
    ordinals: dict[tuple[str, str], int] = {}
    step = target_words - overlap_words
    for block in blocks:
        words = block.text.split()
        if block.content_type == "chart":
            windows = [block.text]
        elif block.content_type == "table":
            windows = _table_windows(block.text, target_words)
        elif boundary_policy == "paragraph":
            windows = _paragraph_windows(block.text, target_words, overlap_words)
        else:
            windows = [
                " ".join(words[start : start + target_words])
                for start in range(0, len(words), step)
            ]
        for text in windows:
            if not text:
                continue
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
                    source_format=block.source_format,
                    extraction_method=block.extraction_method,
                    content_type=block.content_type,
                    chart_id=block.chart_id,
                    validation_status=block.validation_status,
                    source_ref=block.source_ref,
                    original_local_path=block.original_local_path,
                    publication_date_is_estimated=block.publication_date_is_estimated,
                    publication_date_precision=block.publication_date_precision,
                    publication_date_basis=block.publication_date_basis,
                )
            )
    return chunks


def _paragraph_windows(text: str, target_words: int, overlap_words: int) -> list[str]:
    """Pack complete paragraphs; split oversized paragraphs at sentences first.

    Overlap is a maximum budget of complete units, so the actual overlap can be
    smaller. Only a sentence exceeding the target needs a hard word split.
    Source blocks (and therefore documents/sections/pages) are never combined.
    """
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph.split()) <= target_words:
            units.append(paragraph)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            words = sentence.split()
            units.extend(" ".join(words[start:start + target_words])
                         for start in range(0, len(words), target_words))
    windows: list[str] = []
    current: list[str] = []
    count = 0
    for unit in units:
        size = len(unit.split())
        if current and count + size > target_words:
            windows.append("\n\n".join(current))
            suffix: list[str] = []
            suffix_words = 0
            for previous in reversed(current):
                previous_words = len(previous.split())
                if suffix_words + previous_words > overlap_words:
                    break
                suffix.insert(0, previous)
                suffix_words += previous_words
            while suffix and suffix_words + size > target_words:
                suffix_words -= len(suffix.pop(0).split())
            current, count = suffix, suffix_words
        current.append(unit)
        count += size
    if current:
        windows.append("\n\n".join(current))
    return windows


def _table_windows(text: str, target_words: int) -> list[str]:
    """Split oversized Markdown tables by rows while repeating their header."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(text.split()) <= target_words or len(lines) <= 2:
        return ["\n".join(lines)]
    header = lines[:2]
    windows: list[str] = []
    current = list(header)
    current_words = len(" ".join(current).split())
    for row in lines[2:]:
        row_words = len(row.split())
        if len(current) > 2 and current_words + row_words > target_words:
            windows.append("\n".join(current))
            current = list(header)
            current_words = len(" ".join(current).split())
        current.append(row)
        current_words += row_words
    if len(current) > 2:
        windows.append("\n".join(current))
    return windows
