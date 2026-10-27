"""Contextual passages for Workflow B, searched by Workflow C.

The canonical chunk text stays the source quote. A short prefix is stored only
in ``search_text``, which BM25 and the vector builder read. The prefix names the
company, document type, period, section, and content type so a segment row is
not the same passage as the company total.

Passages are packed to a 512-token budget, the window of ``BAAI/bge-base-en-v1.5``.
Table headers are repeated on every row window. This does not replace the
canonical ``data/narrative/processed/<TICKER>`` build.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable


CONTEXTUAL_STRATEGY = "contextual"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
TOKEN_LIMIT = 512
_GENERIC_SECTIONS = {"", "Document text", "Front matter"}
_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_SENTENCE = re.compile(r"(?<=[.!?])\s+| • |\n+")
_FIGURE = re.compile(
    r"\$\d[\d,]*(?:\.\d+)?(?:\s*(?:billion|million))?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:%|percent|billion|million)",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """Conservative stand-in for the bge wordpiece count.

    Real bge tokenization is used when that tokenizer is available. This
    estimate stays above a plain word count so a fitted passage still fits
    the 512-token window.
    """

    words = re.findall(r"\S+", text)
    return max(1, math.ceil(len(words) * 1.3) + 2)


_DASH = re.compile(r"\s+(?:—|--)\s+")
_VALUE = r"(?:N/A|\$?\(?-?[\d,]+(?:\.\d+)?\)?%?)"
_VALUE_WORD = re.compile(rf"^{_VALUE}$", re.IGNORECASE)
_CELL = re.compile(rf"([^:;]{{1,48}}?):\s*({_VALUE})", re.IGNORECASE)
_HEADING = re.compile(r"([A-Z][^:]{2,70}):\s*$")
_COLUMN_PERIOD = re.compile(
    r"((?:Three|Six|Nine|Twelve) Months Ended [A-Z][a-z]+ \d{1,2}|Q[1-4]\s*20\d{2}|FY\s*20\d{2})",
    re.IGNORECASE,
)
_SEGMENTS = (
    "North America", "International", "United States", "Rest of world", "Google Cloud",
    "Google Services", "YouTube ads", "Family of Apps", "Reality Labs", "Data Center",
    "Consolidated", "Worldwide", "Commercial", "Government", "AWS", "Azure", "U.S.",
)


def context_prefix(chunk: dict[str, Any], *, segment: str | None = None, metric: str | None = None) -> str:
    """One line that keeps company, fiscal period, segment, and metric apart."""

    period = chunk.get("period") or chunk.get("publication_date") or "undated"
    section = (chunk.get("section") or "").strip()
    pieces = [
        str(chunk.get("company_name") or "").strip(),
        f"({chunk.get('company_ticker') or ''})",
        str(chunk.get("document_type") or "document"),
        f"fiscal period {period}",
    ]
    if segment:
        pieces.append(f"segment {segment}")
    elif section not in _GENERIC_SECTIONS:
        pieces.append(f"section {section}")
    if metric:
        pieces.append(f"metric {metric}")
    if chunk.get("page"):
        pieces.append(f"page {chunk['page']}")
    return " ".join(piece for piece in pieces if piece).strip() + "."


def labeled_rows(text: str) -> list[dict[str, Any]]:
    """Split inlined filing tables into one row, with that row's column headers.

    A row is source text such as ``AWS — 2025: 17; 2026: 28``. The metric heading
    that introduces the row, such as ``Year-over-year Percentage Growth``, is kept
    with it. Prose between tables is not returned here.
    """

    rows: list[dict[str, Any]] = []
    cursor = 0
    current_heading: str | None = None
    for dash in _DASH.finditer(text):
        if dash.start() < cursor:
            continue
        label = _label_before(text, dash.start())
        cells, end = _cells_after(text, dash.end())
        if not label or len(cells) < 2:
            continue
        label_start = dash.start() - len(label)
        gap = text[cursor:label_start]
        found = _trailing_heading(gap)
        if found:
            current_heading = found
        elif len(gap.split()) > 12:
            current_heading = None
        segment, metric = _segment_and_metric(label, current_heading)
        rows.append({
            "text": text[label_start:end].strip(),
            "segment": segment,
            "metric": metric,
            "heading": current_heading,
            "start": label_start,
            "end": end,
        })
        cursor = end
    return rows


def _label_before(text: str, dash_start: int) -> str:
    words = text[max(0, dash_start - 70):dash_start].split()
    kept: list[str] = []
    for word in reversed(words):
        if word in {";", "—", "--"} or ":" in word or _VALUE_WORD.match(word):
            break
        kept.append(word.strip(","))
        if len(kept) == 6:
            break
    return " ".join(word for word in reversed(kept) if word)


def _cells_after(text: str, dash_end: int) -> tuple[list[str], int]:
    cells: list[str] = []
    pos = dash_end
    while True:
        match = _CELL.match(text, pos)
        if match is None:
            break
        cells.append(f"{match.group(1).strip()}: {match.group(2)}")
        pos = match.end()
        if text[pos:pos + 1] == ";":
            pos += 1
            continue
        break
    return cells, pos


def _trailing_heading(gap: str) -> str | None:
    match = _HEADING.search(" ".join(gap.split()))
    if match is None:
        return None
    heading = match.group(1).strip(" ,")
    if len(heading.split()) > 12:
        return None
    return heading


def _segment_and_metric(label: str, heading: str | None) -> tuple[str, str]:
    for segment in _SEGMENTS:
        if label.lower() == segment.lower():
            return segment, heading or label
        prefix = segment + " "
        if label.lower().startswith(prefix.lower()):
            metric = label[len(prefix):].strip(" :-")
            return segment, metric or heading or label
    return label, heading or label


def _row_records(
    chunk: dict[str, Any],
    source: str,
    rows: list[dict[str, Any]],
    token_limit: int,
    counter: Callable[[str], int],
) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        body = row["text"]
        heading = row.get("heading")
        if heading and heading.lower() not in body.lower():
            body = f"{heading}: {body}"
        metric = row["metric"]
        column_period = _column_period(body)
        if column_period and column_period.lower() not in metric.lower() and column_period.lower() not in body.lower():
            metric = f"{metric}, {column_period}"
        prefix = context_prefix(chunk, segment=row["segment"], metric=metric)
        records.append(_make_record(chunk, body, prefix, len(records) + 1, token_limit, counter))
    return records


def _column_period(text: str) -> str | None:
    match = _COLUMN_PERIOD.search(text)
    return match.group(1) if match else None


def _prose_gaps(source: str, rows: list[dict[str, Any]]) -> str:
    pieces = []
    cursor = 0
    for row in rows:
        pieces.append(source[cursor:row["start"]])
        cursor = row["end"]
    pieces.append(source[cursor:])
    kept = []
    for piece in pieces:
        words = piece.split()
        if len(words) >= 8:
            kept.append(" ".join(words))
    return "\n".join(kept)


def _prose_records(
    chunk: dict[str, Any],
    prose: str,
    token_limit: int,
    counter: Callable[[str], int],
) -> list[dict[str, Any]]:
    if not prose.strip():
        return []
    prefix = context_prefix(chunk)
    header, units = _units(prose, "narrative")
    records = []
    for ordinal, window in enumerate(_pack(units, prefix, header, token_limit, counter), start=1):
        body = f"{header}\n{window}".strip() if header else window
        records.append(_make_record(chunk, body, prefix, 1000 + ordinal, token_limit, counter))
    return records


def _make_record(
    chunk: dict[str, Any],
    body: str,
    prefix: str,
    ordinal: int,
    token_limit: int,
    counter: Callable[[str], int],
) -> dict[str, Any]:
    search_text = f"{prefix}\n{body}"
    if counter(search_text) > token_limit:
        body, search_text = _shrink_to_limit(prefix, body, token_limit, counter)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    parent = str(chunk.get("chunk_id") or "chunk")
    record = dict(chunk)
    record.update(
        chunk_id=f"{parent}:c{ordinal:02d}:{digest[:12]}",
        text=body,
        text_sha256=digest,
        search_text=search_text,
        parent_chunk_id=parent,
    )
    return record


def split_claim_parts(text: str) -> list[str]:
    """Split a claim where each side states its own figure.

    "sales and marketing" stays one part. "grew 39% and R&D grew 43%" becomes
    two searches. A claim with no figure stays whole.
    """

    parts = [text.strip()]
    if not parts[0]:
        return []
    changed = True
    while changed:
        changed = False
        nxt: list[str] = []
        for part in parts:
            pieces = _split_part(part)
            changed = changed or len(pieces) > 1
            nxt.extend(pieces)
        parts = nxt
    return [part for part in parts if part]


def figure_phrases(text: str) -> list[str]:
    phrases = []
    for match in _FIGURE.finditer(text):
        phrase = re.sub(r"\s+", " ", match.group(0).lower().replace(",", "").replace("$", "")).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def passage_states_figures(claim: str, passage: str) -> bool:
    """True when every figure in the claim is written in the source passage."""

    haystack = passage.lower().replace(",", "").replace("$", "")
    haystack = re.sub(r"\s+", " ", haystack)
    return all(phrase in haystack for phrase in figure_phrases(claim))


def single_growth_claim(text: str) -> bool:
    """A claim that states one increase, such as ``AWS grew 28%``.

    The counter search looks for declines and risks, which pulls page numbers
    and other segments into the list. One growth figure does not need it.
    """

    percents = re.findall(r"\d+(?:\.\d+)?%", text)
    if len(percents) != 1:
        return False
    if re.search(r"\b(?:grew|grown|increased|growth)\b", text, re.IGNORECASE) is None:
        return False
    if re.search(r"\b(?:declin\w*|decreas\w*|down|fell|not|no|only)\b", text, re.IGNORECASE):
        return False
    return True


def passage_matches_claim(claim: str, passage: str) -> bool:
    """Require a stated percent and a named quarter to occur in the passage.

    ``28%`` does not match a page number or ``$28,786``. ``Q1 2026`` does not
    match a prior-quarter row that only contains ``Q4 2025``.
    """

    percents = [item.lower() for item in re.findall(r"\d+(?:\.\d+)?%", claim)]
    quarters = [re.sub(r"\s+", "", item.lower()) for item in re.findall(r"Q[1-4]\s*20\d{2}", claim, re.IGNORECASE)]
    if not percents and not quarters:
        return True
    haystack = passage.lower()
    compact = re.sub(r"\s+", "", haystack)
    if percents and not any(percent in haystack or percent in compact for percent in percents):
        return False
    if quarters and not any(quarter in compact for quarter in quarters):
        return False
    return True


def followup_query(claim: str, passages: Iterable[str], document_types: Iterable[str]) -> str | None:
    """One extra search when the figure is missing from the first passages.

    The extra terms point at the other document family: the earnings release
    when the first hits are filings, and the filing when they are releases.
    """

    pending = [part for part in split_claim_parts(claim) if not any(passage_states_figures(part, text) for text in passages)]
    pending = [part for part in pending if figure_phrases(part)]
    if not pending:
        return None
    types = {item for item in document_types if item}
    if types and types <= {"10-K", "10-Q", "8-K"}:
        hint = "earnings release"
    elif "earnings_release" in types and not ({"10-K", "10-Q"} & types):
        hint = "10-K 10-Q"
    else:
        hint = "earnings release 10-Q"
    return f"{pending[0]} {hint}"


def contextualize_chunk(
    chunk: dict[str, Any],
    *,
    token_limit: int = TOKEN_LIMIT,
    count_tokens: Callable[[str], int] | None = None,
) -> list[dict[str, Any]]:
    """Return token-fitted copies. ``text`` is source text. ``search_text`` adds the prefix."""

    counter = count_tokens or estimate_tokens
    source = str(chunk.get("text") or "").strip()
    if not source:
        return []
    rows = labeled_rows(source)
    if rows:
        return _row_records(chunk, source, rows, token_limit, counter) + _prose_records(
            chunk, _prose_gaps(source, rows), token_limit, counter
        )
    prefix = context_prefix(chunk)
    header, units = _units(source, str(chunk.get("content_type") or ""))
    windows = _pack(units, prefix, header, token_limit, counter)
    records = []
    for ordinal, window in enumerate(windows, start=1):
        body = f"{header}\n{window}".strip() if header else window
        search_text = f"{prefix}\n{body}"
        if counter(search_text) > token_limit:
            body, search_text = _shrink_to_limit(prefix, body, token_limit, counter)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        parent = str(chunk.get("chunk_id") or "chunk")
        record = dict(chunk)
        record.update(
            chunk_id=f"{parent}:c{ordinal:02d}:{digest[:12]}",
            text=body,
            text_sha256=digest,
            search_text=search_text,
            parent_chunk_id=parent,
        )
        records.append(record)
    return records


def write_contextual_index(
    chunks: Iterable[dict[str, Any]],
    output_dir: Path,
    *,
    token_limit: int = TOKEN_LIMIT,
    count_tokens: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Write ``chunks.jsonl`` and a BM25 ``indexes/index.json`` for one company."""

    import json

    from finsight.agents.narrative.corpus.processing.index import INDEX_SCHEMA_VERSION

    counter = count_tokens or estimate_tokens
    records: list[dict[str, Any]] = []
    parents = 0
    for chunk in chunks:
        parents += 1
        records.extend(contextualize_chunk(chunk, token_limit=token_limit, count_tokens=counter))
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    fingerprint = hashlib.sha256("\n".join(record["chunk_id"] for record in records).encode("utf-8")).hexdigest()
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "corpus_fingerprint": fingerprint,
        "chunking": {
            "strategy": CONTEXTUAL_STRATEGY,
            "target_words": token_limit,
            "overlap_words": 0,
            "token_limit": token_limit,
            "embedding_model": EMBEDDING_MODEL,
            "token_counter": "estimate" if count_tokens is None else "tokenizer",
        },
        "chunk_count": len(records),
        "duplicates_excluded": 0,
        "parent_chunk_count": parents,
        "chunks": records,
    }
    index_path = output_dir / "indexes" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (output_dir / "indexes" / "index_metadata.json").write_text(
        json.dumps({key: value for key, value in payload.items() if key != "chunks"}, indent=2),
        encoding="utf-8",
    )
    return payload


def _split_part(text: str) -> list[str]:
    for separator in (" with ", " and ", "; "):
        if separator not in text:
            continue
        left, right = text.split(separator, 1)
        if figure_phrases(left) and figure_phrases(right):
            return [left.strip(), right.strip()]
    return [text.strip()]


def _markdown_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(_SEPARATOR_CELL.fullmatch(cell) for cell in cells)


def _units(text: str, content_type: str) -> tuple[str, list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if content_type == "table" or (len(lines) > 2 and lines[0].startswith("|")):
        if len(lines) >= 2 and lines[0].startswith("|") and _markdown_separator(lines[1]):
            return "\n".join(lines[:2]), lines[2:]
        return "", lines
    units = [unit.strip() for unit in _SENTENCE.split(text) if unit.strip()]
    return "", units or [text.strip()]


def _pack(
    units: list[str],
    prefix: str,
    header: str,
    limit: int,
    counter: Callable[[str], int],
) -> list[str]:
    budget = max(limit - counter(prefix) - (counter(header) if header else 0) - 1, 8)
    windows: list[list[str]] = []
    current: list[str] = []
    cost = 0
    for unit in _fit_unit(units, budget, counter):
        unit_cost = counter(unit)
        if current and cost + unit_cost > budget:
            windows.append(current)
            current = []
            cost = 0
        current.append(unit)
        cost += unit_cost
    if current:
        windows.append(current)
    return [" ".join(window) for window in windows]


def _fit_unit(units: list[str], budget: int, counter: Callable[[str], int]) -> list[str]:
    fitted: list[str] = []
    for unit in units:
        if counter(unit) <= budget:
            fitted.append(unit)
            continue
        words = unit.split()
        start = 0
        while start < len(words):
            end = min(len(words), start + budget)
            while end > start + 1 and counter(" ".join(words[start:end])) > budget:
                end -= 1
            fitted.append(" ".join(words[start:end]))
            start = end
    return fitted


def _shrink_to_limit(
    prefix: str,
    body: str,
    limit: int,
    counter: Callable[[str], int],
) -> tuple[str, str]:
    words = body.split()
    end = len(words)
    while end > 1 and counter(f"{prefix}\n{' '.join(words[:end])}") > limit:
        end -= 1
    shrunk = " ".join(words[:end])
    return shrunk, f"{prefix}\n{shrunk}"
