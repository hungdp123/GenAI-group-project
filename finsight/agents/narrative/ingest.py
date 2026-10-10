"""Deterministic HTML/PDF extraction for the narrative corpus."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag
from pypdf import PdfReader

from finsight.agents.narrative.corpus import CorpusDocument


_ITEM_HEADING = re.compile(r"^Item\s+\d+[A-Z]?\.?\s+\S", re.IGNORECASE)
_WHITESPACE = re.compile(r"[ \t\u00a0]+")
_HIDDEN_STYLE = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)
_FURNITURE_NAME = re.compile(
    r"(?:^|[-_\s])(?:nav(?:igation)?|menu|menubar|breadcrumb|toolbar|pagination|"
    r"masthead|header|footer|page[-_ ]?(?:header|footer)|site[-_ ]?(?:header|footer)|search[-_ ]?form)"
    r"(?:$|[-_\s])",
    re.IGNORECASE,
)
_NUMBER_LIKE = re.compile(r"^[\s($€£¥+\-]*\d[\d,]*(?:\.\d+)?%?[)\s]*$")
_PERIOD_HEADER_HINT = re.compile(
    r"\b(?:20\d{2}|fy\s*\d{2,4}|q[1-4]|year ended|months ended|quarter ended|"
    r"three months|six months|twelve months|q/q|y/y|actual|estimate)\b",
    re.IGNORECASE,
)
_UNIT_ONLY = {"%", "bp", "bps", "million", "millions", "billion", "billions"}
_CURRENCY_ONLY = {"$", "€", "£", "¥"}
_ROW_HEADER_HINT = re.compile(r"^(?:metric|description|line item|particulars|item|account)$", re.IGNORECASE)
_UNIT_HEADER_HINT = re.compile(r"\b(?:in thousands|in millions|in billions|except per share)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProcessedBlock:
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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _clean_lines(text: str) -> list[str]:
    # Some filing/PDF font maps expose smart apostrophes as U+FFFD. Repair the
    # common in-word case and retain other replacements as visible separators.
    text = re.sub(r"(?<=\w)\ufffd(?=\w)", "'", text).replace("\ufffd", " — ")
    lines: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        line = _WHITESPACE.sub(" ", raw).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return lines


def _base(document: CorpusDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "company_ticker": document.company_ticker,
        "company_name": document.company_name,
        "document_type": document.document_type,
        "publication_date": document.publication_date.isoformat(),
        "source_tier": document.source_tier,
        "source_url": document.source_url,
        "local_path": document.local_path.as_posix(),
        "period": document.period,
    }


def _remove_hidden_and_furniture(soup: BeautifulSoup) -> None:
    """Remove non-content HTML, hidden nodes, and common navigation/page chrome."""
    for tag in soup.find_all(
        ["script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form", "button"]
    ):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        if tag.parent is None:
            continue
        role = str(tag.get("role", "")).lower()
        aria_hidden = str(tag.get("aria-hidden", "")).lower() == "true"
        style = str(tag.get("style", ""))
        if tag.has_attr("hidden") or aria_hidden or role in {"navigation", "banner", "contentinfo"}:
            tag.decompose()
            continue
        if _HIDDEN_STYLE.search(style):
            tag.decompose()
            continue

        labels = [str(tag.get("id", "")), str(tag.get("aria-label", ""))]
        classes = tag.get("class", [])
        labels.extend(str(value) for value in classes)
        if any(_FURNITURE_NAME.search(label.strip()) for label in labels if label.strip()):
            tag.decompose()


def _table_rows(table: Tag) -> list[tuple[list[str], list[bool], bool]]:
    """Read direct table rows as cell text, header flags, and thead membership."""
    rows: list[tuple[list[str], list[bool], bool]] = []
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        cells_and_values = [
            (cell, re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).strip()) for cell in cells
        ]
        cells_and_values = [(cell, value) for cell, value in cells_and_values if value]
        cells = [cell for cell, _ in cells_and_values]
        values = [value for _, value in cells_and_values]
        if not any(values):
            continue
        headers = [cell.name == "th" for cell in cells]
        rows.append((values, headers, row.find_parent("thead") is not None))
    return rows


def _table_text(table: Tag) -> str:
    """Serialize a table with row labels and column labels instead of a number stream."""
    rows = _table_rows(table)
    if not rows:
        return ""

    thead_candidates = [
        i for i, (values, _, in_head) in enumerate(rows) if in_head and len(values) > 1
    ]
    header_index: int | None = next(
        (
            i for i in thead_candidates
            if any(_PERIOD_HEADER_HINT.search(value) for value in rows[i][0])
        ),
        thead_candidates[-1] if thead_candidates else None,
    )
    if header_index is None and len(rows[0][0]) > 1 and all(rows[0][1]):
        header_index = 0
    if header_index is None:
        header_index = next(
            (
                index
                for index, (values, _, _) in enumerate(rows[:5])
                if len(values) > 1
                and any(_PERIOD_HEADER_HINT.search(value) for value in values)
                and (
                    all(not _NUMBER_LIKE.match(value) for value in values)
                    or all(re.fullmatch(r"(?:FY\s*)?20\d{2}", value, re.IGNORECASE) for value in values)
                )
            ),
            None,
        )
    headers = rows[header_index][0] if header_index is not None else []
    header_offset = (
        1
        if headers and (
            not headers[0]
            or _ROW_HEADER_HINT.match(headers[0])
            or _UNIT_HEADER_HINT.search(headers[0])
        )
        else 0
    )
    caption_tag = table.find("caption", recursive=False)
    caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""
    output = [f"Table: {caption}"] if caption else []

    for index, (values, header_flags, _) in enumerate(rows):
        if index == header_index:
            continue
        values = _merge_unit_cells(values)
        if len(values) == 1:
            output.append(values[0])
            continue

        has_row_label = header_flags[0] or header_index is not None or not _NUMBER_LIKE.match(values[0])
        row_label = values[0] if has_row_label else ""
        start = 1 if has_row_label else 0
        pairs: list[str] = []
        for pair_index, cell_index in enumerate(range(start, len(values))):
            header_cell = pair_index + header_offset
            if header_index is not None and header_cell < len(headers) and headers[header_cell]:
                label = headers[header_cell]
            elif cell_index == 0 and not row_label:
                label = headers[0] if headers else "Column 1"
            else:
                label = f"Column {cell_index + 1}"
            pairs.append(f"{label}: {values[cell_index]}")
        if row_label:
            output.append(f"{row_label} — " + "; ".join(pairs))
        else:
            output.append("; ".join(pairs))

    return "\n".join(output)


def _replace_tables_with_readable_text(soup: BeautifulSoup) -> None:
    """Replace tables in place so their position relative to surrounding text is kept."""
    for table in reversed(soup.find_all("table")):
        text = _table_text(table)
        if text:
            table.replace_with(NavigableString(f"\n{text}\n"))
        else:
            table.decompose()


def _remove_table_of_contents(soup: BeautifulSoup) -> None:
    """Drop hyperlink-heavy SEC contents tables to avoid treating their rows as sections."""
    item_link = re.compile(r"^\s*Item\s+\d+[A-Z]?\.?\s+", re.IGNORECASE)
    for table in list(soup.find_all("table")):
        text = table.get_text(" ", strip=True)
        anchors = table.find_all("a")
        linked_items = sum(bool(item_link.match(anchor.get_text(" ", strip=True))) for anchor in anchors)
        item_count = len(re.findall(r"\bItem\s+\d+[A-Z]?\.?", text, re.IGNORECASE))
        part_count = len(re.findall(r"\bP\s*art\s+(?:I{1,3}|IV|[1-4])\b", text, re.IGNORECASE))
        if linked_items >= 3 or (len(anchors) >= 10 and item_count >= 8 and part_count >= 2):
            table.decompose()


def _merge_unit_cells(values: list[str]) -> list[str]:
    """Join detached currency/unit cells to the adjacent amount."""
    merged: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value in _CURRENCY_ONLY and index + 1 < len(values) and re.search(r"\d", values[index + 1]):
            merged.append(value + values[index + 1])
            index += 2
            continue
        if value.lower() in _UNIT_ONLY and merged and re.search(r"\d", merged[-1]):
            merged[-1] += value
        else:
            merged.append(value)
        index += 1
    return merged


def parse_html(document: CorpusDocument) -> list[ProcessedBlock]:
    markup = document.local_path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(markup, "html.parser")
    _remove_hidden_and_furniture(soup)
    for hidden in soup.select("ix\\:hidden"):
        hidden.decompose()
    _remove_table_of_contents(soup)
    _replace_tables_with_readable_text(soup)

    lines = _clean_lines(soup.get_text("\n"))
    # SEC filings normally contain a table of contents with bare "Item 1." links.
    # Requiring title text avoids treating those links as full sections.
    starts = [index for index, line in enumerate(lines) if _ITEM_HEADING.match(line)]
    if not starts:
        text = "\n".join(lines)
        return [
            ProcessedBlock(
                **_base(document), section="Document text", page=None,
                citation_span="HTML document text", text=text,
            )
        ] if text else []

    blocks: list[ProcessedBlock] = []
    if starts[0] > 0:
        front = "\n".join(lines[: starts[0]])
        if front:
            blocks.append(
                ProcessedBlock(
                    **_base(document), section="Front matter", page=None,
                    citation_span="HTML front matter", text=front,
                )
            )
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        section_lines = lines[start:end]
        if len(section_lines) < 2:
            continue
        heading = section_lines[0]
        blocks.append(
            ProcessedBlock(
                **_base(document), section=heading, page=None,
                citation_span=heading, text="\n".join(section_lines),
            )
        )
    return blocks


def parse_pdf(document: CorpusDocument) -> tuple[list[ProcessedBlock], list[str]]:
    reader = PdfReader(str(document.local_path))
    blocks: list[ProcessedBlock] = []
    warnings: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        lines = _clean_lines(page.extract_text() or "")
        if not lines:
            warnings.append(f"{document.document_id}: PDF page {page_number} has no extractable text")
            continue
        title = lines[0][:160]
        blocks.append(
            ProcessedBlock(
                **_base(document), section=title, page=page_number,
                citation_span=f"PDF page {page_number}", text="\n".join(lines),
            )
        )
    return blocks, warnings


def parse_document(document: CorpusDocument) -> tuple[list[ProcessedBlock], list[str]]:
    suffix = document.local_path.suffix.lower()
    if suffix in {".htm", ".html"}:
        return parse_html(document), []
    if suffix == ".pdf":
        return parse_pdf(document)
    raise ValueError(f"Unsupported narrative document format: {document.local_path}")
