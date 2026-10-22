"""Page-cited PDF enrichment blocks and persistent canonical promotion metadata."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from finsight.agents.narrative.corpus.processing.corpus import CorpusDocument, corpus_fingerprint, file_sha256
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock, _base

ENRICHMENT_VERSION = "2.1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _block(document: CorpusDocument, page: int, section: str, text: str, *,
           kind: str = "narrative", chart_id: str | None = None,
           method: str = "docling_layout", source_ref: str | None = None) -> ProcessedBlock:
    base = _base(document)
    base["extraction_method"] = method
    return ProcessedBlock(
        **base, section=section, page=page, text=text.strip(), content_type=kind,
        chart_id=chart_id, validation_status="model_extracted",
        source_ref=source_ref or f"{document.local_path.name}#page={page}",
        citation_span=f"PDF page {page}" + (f", {kind} {chart_id or section}" if kind != "narrative" else ""),
    )


def docling_blocks(document: CorpusDocument, parsed: dict[str, Any], records: list[Any],
                   pages: set[int], *, ocr_pages: set[int] | None = None
                   ) -> tuple[list[ProcessedBlock], list[dict[str, Any]]]:
    """Build page replacements directly from Docling provenance, without Markdown-order guesses.

    A page's native block is replaced only when this produces nonempty blocks. Text inside
    successful chart/table regions is excluded from the narrative to avoid duplicate passages.
    Failed charts and charts without numeric values keep their Docling text and
    an explicit coverage limitation.
    """
    from docling_core.types.doc import DoclingDocument
    from finsight.agents.narrative.corpus.pdf.docling_charts import (
        _bbox_from_provenance, _intersection_ratio, _page_size, render_chart_markdown,
    )

    rich = DoclingDocument.model_validate(parsed)
    output: list[ProcessedBlock] = []
    exclusions: dict[int, list[Any]] = defaultdict(list)
    limitations: list[dict[str, Any]] = []
    ocr_pages = ocr_pages or set()
    for record in records:
        page = record.source.page_number
        if page not in pages:
            continue
        if record.status != "success" or record.extraction is None:
            limitations.append({"document_id": document.document_id, "page_number": page,
                                "scope": "chart",
                                "chart_id": record.source.chart_id,
                                "reason": record.error or "Chart extraction was not requested"})
            continue
        text = render_chart_markdown(record)
        if not record.extraction.data_points:
            limitations.append({"document_id": document.document_id, "page_number": page, "scope": "chart",
                                "chart_id": record.source.chart_id,
                                "reason": "No numeric chart values returned; "
                                          "descriptive fields and Docling text retained"})
        # Keep structured information not present in the human-readable chart table.
        detail = record.extraction.model_dump(mode="json")
        text += "\nAxis labels: " + ", ".join(detail.get("axis_labels") or ["unknown"])
        text += "\nUnit scale: " + (detail.get("overall_unit") or "unknown")
        text += "\nPeriod order: " + ", ".join(detail.get("reporting_periods") or [])
        text += "\nLegend/series: " + json.dumps(detail.get("legend", []), ensure_ascii=False)
        for point in detail.get("data_points", []):
            if point.get("ambiguity"):
                text += f"\nUncertainty ({point['metric']}, {point['reporting_period']}): {point['ambiguity']}"
        for ambiguity in detail.get("ambiguities", []):
            text += "\nUncertainty: " + ambiguity
        chart_id = record.source.chart_id
        output.append(_block(document, page, f"Page {page} chart {chart_id}", text,
                             kind="chart", chart_id=chart_id, method=f"docling+{record.provider}:{record.model}",
                             source_ref=record.source.source_ref))
        if record.extraction.data_points:
            exclusions[page].append(record.source.bbox)

    for index, table in enumerate(rich.tables):
        if not table.prov:
            limitations.append({"document_id": document.document_id, "scope": "table",
                                "reason": f"Table {index} has no page provenance"})
            continue
        page = table.prov[0].page_no
        if page not in pages:
            continue
        bbox = table.prov[0].bbox
        from finsight.agents.narrative.corpus.pdf.chart_models import BoundingBox
        box = BoundingBox(left=bbox.l, top=bbox.t, right=bbox.r, bottom=bbox.b,
                          coord_origin=str(bbox.coord_origin.value))
        _, height = _page_size(parsed, page)
        if any(_intersection_ratio(box, region, height) >= 0.6 for region in exclusions[page]):
            continue
        text = table.export_to_markdown(doc=rich).strip()
        if not text:
            continue
        coordinates = f"{page}:{bbox.l:.1f}:{bbox.t:.1f}:{bbox.r:.1f}:{bbox.b:.1f}"
        identity = hashlib.sha256(coordinates.encode()).hexdigest()[:10]
        table_id = f"p{page:03d}-table-{identity}"
        method = "docling_ocr_table" if page in ocr_pages else "docling_table"
        output.append(_block(document, page, table_id, text, kind="table", method=method,
                             source_ref=f"{document.local_path.name}#page={page}&table={table_id}"))
        exclusions[page].append(box)

    narrative: dict[int, list[str]] = defaultdict(list)
    for item in parsed.get("texts", []):
        prov = item.get("prov") or []
        if not prov:
            continue
        page = int(prov[0].get("page_no", 0))
        if page not in pages:
            continue
        text = str(item.get("text") or item.get("orig") or "").strip()
        if not text:
            continue
        try:
            bbox = _bbox_from_provenance(prov[0])
            _, height = _page_size(parsed, page)
            if any(_intersection_ratio(bbox, region, height) >= 0.6 for region in exclusions[page]):
                continue
        except (KeyError, TypeError, ValueError):
            pass
        if not narrative[page] or narrative[page][-1] != text:
            narrative[page].append(text)
    for page, lines in sorted(narrative.items()):
        method = "docling_ocr" if page in ocr_pages else "docling_layout"
        output.append(_block(document, page, f"PDF page {page}", "\n".join(lines), method=method))
    covered = {block.page for block in output}
    for page in sorted(pages - covered):
        limitations.append({"document_id": document.document_id, "page_number": page,
                            "scope": "page",
                            "reason": "Docling produced no usable text, table, or chart for this page"})
    # Deduplicate only within a source page so later publication dates remain available.
    seen: set[tuple[int | None, str]] = set()
    unique: list[ProcessedBlock] = []
    for block in sorted(output, key=lambda value: (value.page or 0, value.content_type, value.section)):
        key = (block.page, re.sub(r"\s+", " ", block.text).strip())
        if key not in seen:
            seen.add(key)
            unique.append(block)
    return unique, limitations


def load_promoted_enrichment(workspace: Path, ticker: str, documents: list[CorpusDocument]
                             ) -> tuple[dict[str, list[ProcessedBlock]], dict[str, Any]]:
    root = workspace / "data/narrative/processed" / ticker
    manifest_path = root / "pdf_enrichment.json"
    if not manifest_path.is_file():
        return {}, {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = {doc.document_id: doc for doc in documents}
    output: dict[str, list[ProcessedBlock]] = {}
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for document_id, item in sorted(manifest.get("documents", {}).items()):
        doc = current.get(document_id)
        if doc is None:
            continue  # Quarantined or out-of-scope sources cannot re-enter through promotion.
        if doc.sha256 != item["source_sha256"]:
            raise ValueError(f"PDF enrichment source changed for {document_id}; re-extract it")
        artifact = (root / item["blocks_path"]).resolve()
        if not artifact.is_relative_to((root / "enrichments").resolve()):
            raise ValueError("Enrichment artifact must remain in the company's enrichments directory")
        if file_sha256(artifact) != item["blocks_sha256"]:
            raise ValueError(f"PDF enrichment artifact hash mismatch for {document_id}")
        blocks = [ProcessedBlock(**json.loads(line))
                  for line in artifact.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(block.document_id != document_id or not block.page for block in blocks):
            raise ValueError(f"Invalid page provenance in PDF enrichment for {document_id}")
        base = _base(doc)
        # Source metadata always comes from the current eligible manifest, including estimated dates.
        output[document_id] = [replace(block, **{key: value for key, value in base.items()
                                                if key not in {"extraction_method", "source_format"}})
                               for block in blocks]
        rows.append({"document_id": document_id, "source_sha256": doc.sha256,
                     "blocks_sha256": item["blocks_sha256"], "version": item.get("version"),
                     "coverage_sha256": hashlib.sha256(json.dumps(item.get("coverage_limitations", []),
                                                                   sort_keys=True).encode()).hexdigest()})
        coverage.extend({**limitation, "company_ticker": ticker, "publication_date": doc.publication_date.isoformat(),
                         "source_url": doc.source_url, "source_tier": doc.source_tier}
                        for limitation in item.get("coverage_limitations", []))
    if not rows:
        return {}, {}
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return output, {"version": ENRICHMENT_VERSION, "fingerprint": fingerprint,
                    "documents": rows, "coverage_limitations": coverage}


def effective_corpus_fingerprint(documents: list[CorpusDocument], enrichment: dict[str, Any]) -> str:
    source = corpus_fingerprint(documents)
    if not enrichment:
        return source
    return hashlib.sha256(f"{source}\n{enrichment['fingerprint']}".encode()).hexdigest()
