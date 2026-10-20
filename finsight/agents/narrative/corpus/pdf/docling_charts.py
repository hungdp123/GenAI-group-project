"""Add traceable financial-chart extraction to Docling Markdown/JSON exports.

The module is intentionally additive: Docling remains responsible for parsing text,
tables, structure, and provenance. This layer detects picture regions that look like
financial charts, crops them from the source PDF, calls a configurable vision provider,
validates the structured result, and selectively replaces matching Markdown image
placeholders.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import re
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from finsight.agents.narrative.corpus.pdf.chart_models import (
    BoundingBox,
    ChartRecord,
    ChartSource,
    DetectionEvidence,
    ValidationResult,
    VisionChartExtraction,
    VisionDataPoint,
)
from finsight.agents.narrative.corpus.pdf.chart_vision import (
    ChartVisionProvider,
    OllamaChartVisionProvider,
    OpenAIChartVisionProvider,
    VisionRequest,
)

LOGGER = logging.getLogger("finsight.docling_charts")
PIPELINE_VERSION = "1.0"
IMAGE_MARKER = "<!-- image -->"

VALUE_RE = re.compile(r"(?:[$€£]|-?\d)[\d,.]*(?:\.\d+)?%?|\(\$?[\d,.]+\)")
PERIOD_RE = re.compile(r"\b(?:Q[1-4][’']?\d{2,4}|FY[’']?\d{2,4}|20\d{2})\b", re.IGNORECASE)
NORMALIZED_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
FINANCIAL_TERMS = {
    "revenue",
    "income",
    "loss",
    "margin",
    "earnings",
    "eps",
    "ebitda",
    "cash flow",
    "operating",
    "gross",
    "gaap",
    "non-gaap",
    "om%",
    "gm%",
}
NEGATIVE_IMAGE_LABELS = {"logo", "photograph", "photo", "decorative", "icon", "signature"}
POSITIVE_IMAGE_LABELS = {"chart", "graph", "plot", "diagram", "bar_chart", "line_chart"}


@dataclass(frozen=True)
class ChartCandidate:
    chart_id: str
    page_number: int
    bbox: BoundingBox
    picture_ref: str | None
    picture_index: int | None
    placeholder_index: int | None
    confidence: float
    reasons: list[str]
    classification_labels: list[str]
    nearby_text: list[str]
    fallback_region: bool = False
    anchor_text: str | None = None

    def detection_evidence(self) -> DetectionEvidence:
        return DetectionEvidence(
            accepted=True,
            confidence=self.confidence,
            reasons=self.reasons,
            classification_labels=self.classification_labels,
            nearby_text=self.nearby_text,
            placeholder_index=self.placeholder_index,
            fallback_region=self.fallback_region,
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _first_provenance(item: dict[str, Any]) -> dict[str, Any] | None:
    provenance = item.get("prov") or []
    return provenance[0] if provenance else None


def _bbox_from_provenance(provenance: dict[str, Any]) -> BoundingBox:
    bbox = provenance["bbox"]
    return BoundingBox(
        left=float(bbox["l"]),
        top=float(bbox["t"]),
        right=float(bbox["r"]),
        bottom=float(bbox["b"]),
        coord_origin=str(bbox.get("coord_origin", "BOTTOMLEFT")).upper(),
    )


def _page_size(document: dict[str, Any], page_number: int) -> tuple[float, float]:
    page = document.get("pages", {}).get(str(page_number), {})
    size = page.get("size", {})
    return float(size.get("width", 0)), float(size.get("height", 0))


def _normalized_edges(bbox: BoundingBox, page_height: float) -> tuple[float, float, float, float]:
    if bbox.coord_origin == "BOTTOMLEFT":
        return bbox.left, page_height - bbox.top, bbox.right, page_height - bbox.bottom
    return bbox.left, bbox.top, bbox.right, bbox.bottom


def _intersection_ratio(inner: BoundingBox, outer: BoundingBox, page_height: float) -> float:
    il, it, ir, ib = _normalized_edges(inner, page_height)
    ol, ot, or_, ob = _normalized_edges(outer, page_height)
    intersection = max(0.0, min(ir, or_) - max(il, ol)) * max(0.0, min(ib, ob) - max(it, ot))
    inner_area = max(0.0, ir - il) * max(0.0, ib - it)
    return intersection / inner_area if inner_area else 0.0


def _iou(first: BoundingBox, second: BoundingBox, page_height: float) -> float:
    fl, ft, fr, fb = _normalized_edges(first, page_height)
    sl, st, sr, sb = _normalized_edges(second, page_height)
    intersection = max(0.0, min(fr, sr) - max(fl, sl)) * max(0.0, min(fb, sb) - max(ft, st))
    first_area = max(0.0, fr - fl) * max(0.0, fb - ft)
    second_area = max(0.0, sr - sl) * max(0.0, sb - st)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _annotation_labels(value: Any) -> list[str]:
    labels: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in {"label", "class_name", "predicted_class", "category"} and isinstance(nested, str):
                labels.append(nested.lower())
            else:
                labels.extend(_annotation_labels(nested))
    elif isinstance(value, list):
        for nested in value:
            labels.extend(_annotation_labels(nested))
    return list(dict.fromkeys(labels))


def _picture_placeholder_ordinals(document: dict[str, Any]) -> dict[str, int]:
    """Map picture references to Markdown image ordinals using Docling body order."""

    order: list[str] = []
    visited_groups: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        reference = node.get("$ref")
        if isinstance(reference, str):
            if reference.startswith("#/pictures/"):
                order.append(reference)
                return
            if reference.startswith("#/groups/") and reference not in visited_groups:
                visited_groups.add(reference)
                try:
                    group_index = int(reference.rsplit("/", 1)[1])
                    walk(document.get("groups", [])[group_index].get("children", []))
                except (IndexError, TypeError, ValueError):
                    LOGGER.warning("Could not resolve Docling group reference %s", reference)
                return
        walk(node.get("children", []))

    walk(document.get("body", {}).get("children", []))
    if not order:
        order = [
            str(picture.get("self_ref") or f"#/pictures/{index}")
            for index, picture in enumerate(document.get("pictures", []))
        ]
    return {reference: index for index, reference in enumerate(dict.fromkeys(order))}


def _texts_in_region(document: dict[str, Any], page_number: int, region: BoundingBox) -> list[str]:
    _, page_height = _page_size(document, page_number)
    results: list[str] = []
    for item in document.get("texts", []):
        provenance = _first_provenance(item)
        if not provenance or int(provenance.get("page_no", -1)) != page_number:
            continue
        try:
            text_box = _bbox_from_provenance(provenance)
        except (KeyError, TypeError, ValueError):
            continue
        if _intersection_ratio(text_box, region, page_height) >= 0.45:
            text = str(item.get("text") or item.get("orig") or "").strip()
            if text:
                results.append(text)
    return results


def _page_section_anchor(document: dict[str, Any], page_number: int) -> str | None:
    for item in document.get("texts", []):
        provenance = _first_provenance(item)
        if provenance and int(provenance.get("page_no", -1)) == page_number:
            if item.get("label") in {"section_header", "title"} and item.get("text"):
                return str(item["text"]).strip()
    return None


def _financial_text_features(texts: Iterable[str]) -> tuple[int, int, int]:
    joined = " ".join(texts)
    lowered = joined.lower().replace("–", "-").replace("—", "-")
    values = len(VALUE_RE.findall(joined))
    periods = len(PERIOD_RE.findall(joined.replace("’", "'")))
    terms = sum(1 for term in FINANCIAL_TERMS if term in lowered)
    return values, periods, terms


def detect_chart_candidates(
    document: dict[str, Any],
    *,
    page_filter: set[int] | None = None,
    fallback_page_filter: set[int] | None = None,
    include_fallback_regions: bool = True,
) -> list[ChartCandidate]:
    """Detect financial charts using picture labels, geometry, and financial OCR evidence."""

    candidates: list[ChartCandidate] = []
    covered_by_page: dict[int, list[BoundingBox]] = {}
    placeholder_ordinals = _picture_placeholder_ordinals(document)
    for picture_index, picture in enumerate(document.get("pictures", [])):
        provenance = _first_provenance(picture)
        if not provenance:
            continue
        page_number = int(provenance.get("page_no", 0))
        if page_number < 1 or (page_filter is not None and page_number not in page_filter):
            continue
        bbox = _bbox_from_provenance(provenance)
        nearby_text = _texts_in_region(document, page_number, bbox)
        labels = _annotation_labels(picture.get("annotations", []))
        values, periods, terms = _financial_text_features(nearby_text)
        page_width, page_height = _page_size(document, page_number)
        area_fraction = 0.0
        if page_width and page_height:
            area_fraction = abs((bbox.right - bbox.left) * (bbox.top - bbox.bottom)) / (page_width * page_height)

        positive_label = any(any(token in label for token in POSITIVE_IMAGE_LABELS) for label in labels)
        negative_label = any(any(token in label for token in NEGATIVE_IMAGE_LABELS) for label in labels)
        strong_financial_pattern = values >= 3 and periods >= 2 and terms >= 1
        accepted = positive_label or (strong_financial_pattern and area_fraction >= 0.035)
        if negative_label and not strong_financial_pattern:
            accepted = False
        if not accepted:
            LOGGER.debug(
                "Rejected picture %s on page %s (labels=%s values=%s periods=%s terms=%s area=%.3f)",
                picture_index,
                page_number,
                labels,
                values,
                periods,
                terms,
                area_fraction,
            )
            continue

        confidence = 0.55
        reasons: list[str] = []
        if positive_label:
            confidence += 0.25
            reasons.append("Docling picture classification indicates a chart/graph")
        if strong_financial_pattern:
            confidence += 0.25
            reasons.append(f"region contains {values} values, {periods} periods, and {terms} financial labels")
        if area_fraction >= 0.12:
            confidence += 0.05
            reasons.append(f"region covers {area_fraction:.1%} of the page")
        if negative_label:
            confidence -= 0.20
            reasons.append("image classification includes a decorative/photo label; financial evidence overrode it")
        chart_id = f"p{page_number:03d}-picture-{picture_index:03d}"
        picture_ref = picture.get("self_ref") or f"#/pictures/{picture_index}"
        candidates.append(
            ChartCandidate(
                chart_id=chart_id,
                page_number=page_number,
                bbox=bbox,
                picture_ref=picture_ref,
                picture_index=picture_index,
                placeholder_index=placeholder_ordinals.get(picture_ref),
                confidence=min(confidence, 0.99),
                reasons=reasons,
                classification_labels=labels,
                nearby_text=nearby_text,
                anchor_text=_page_section_anchor(document, page_number),
            )
        )
        covered_by_page.setdefault(page_number, []).append(bbox)

    if include_fallback_regions:
        fallback_filter = fallback_page_filter if fallback_page_filter is not None else page_filter
        candidates.extend(_detect_fallback_regions(document, covered_by_page, fallback_filter))

    deduplicated: list[ChartCandidate] = []
    for candidate in sorted(candidates, key=lambda c: (c.page_number, c.picture_index is None, c.chart_id)):
        _, page_height = _page_size(document, candidate.page_number)
        duplicate = next(
            (
                existing
                for existing in deduplicated
                if existing.page_number == candidate.page_number
                and _iou(existing.bbox, candidate.bbox, page_height) >= 0.80
            ),
            None,
        )
        if duplicate is None:
            deduplicated.append(candidate)
        else:
            LOGGER.info("Dropped duplicate chart candidate %s in favor of %s", candidate.chart_id, duplicate.chart_id)
    return deduplicated


def _detect_fallback_regions(
    document: dict[str, Any],
    covered_by_page: dict[int, list[BoundingBox]],
    page_filter: set[int] | None,
) -> list[ChartCandidate]:
    """Find chart-like OCR clusters on pages where Docling emitted no picture region."""

    page_items: dict[int, list[tuple[dict[str, Any], BoundingBox]]] = {}
    for item in document.get("texts", []):
        provenance = _first_provenance(item)
        if not provenance:
            continue
        page_number = int(provenance.get("page_no", 0))
        if page_number < 1 or (page_filter is not None and page_number not in page_filter):
            continue
        try:
            bbox = _bbox_from_provenance(provenance)
        except (KeyError, TypeError, ValueError):
            continue
        page_items.setdefault(page_number, []).append((item, bbox))

    fallbacks: list[ChartCandidate] = []
    for page_number, items in page_items.items():
        if covered_by_page.get(page_number):
            continue
        texts = [str(item.get("text") or "").strip() for item, _ in items]
        # Long prose containing financial numbers is not a chart region.
        if sum(len(text.split()) for text in texts) > 300 or any(len(text.split()) > 70 for text in texts):
            continue
        values, periods, terms = _financial_text_features(texts)
        if values < 6 or periods < 3 or terms < 2:
            continue
        # Avoid synthesizing a chart region over a page dominated by a Docling table.
        page_has_table = any(
            int((_first_provenance(table) or {}).get("page_no", -1)) == page_number
            for table in document.get("tables", [])
        )
        if page_has_table:
            continue
        left = min(bbox.left for _, bbox in items)
        right = max(bbox.right for _, bbox in items)
        top = max(bbox.top for _, bbox in items)
        bottom = min(bbox.bottom for _, bbox in items)
        region = BoundingBox(left=left, top=top, right=right, bottom=bottom, coord_origin=items[0][1].coord_origin)
        fallbacks.append(
            ChartCandidate(
                chart_id=f"p{page_number:03d}-fallback-000",
                page_number=page_number,
                bbox=region,
                picture_ref=None,
                picture_index=None,
                placeholder_index=None,
                confidence=0.62,
                reasons=[
                    "no Docling picture region was present",
                    f"page OCR contains {values} values, {periods} periods, and {terms} financial labels",
                ],
                classification_labels=[],
                nearby_text=texts,
                fallback_region=True,
                anchor_text=_page_section_anchor(document, page_number),
            )
        )
    return fallbacks


def crop_chart_region(
    pdf_path: Path,
    candidate: ChartCandidate,
    output_path: Path,
    *,
    dpi: int = 240,
    padding_points: float = 4.0,
) -> str:
    """Render and crop a Docling PDF bounding box, returning the PNG SHA-256."""

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the chart pipeline dependencies with: pip install -e '.[document-ai]'") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[candidate.page_number - 1]
        page_width = float(page.get_width())
        page_height = float(page.get_height())
        scale = dpi / 72.0
        rendered = page.render(scale=scale).to_pil()
        left, top, right, bottom = _normalized_edges(candidate.bbox, page_height)
        crop_box = (
            max(0, round((left - padding_points) * scale)),
            max(0, round((top - padding_points) * scale)),
            min(round(page_width * scale), round((right + padding_points) * scale)),
            min(round(page_height * scale), round((bottom + padding_points) * scale)),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError(f"Invalid crop box {crop_box} for {candidate.chart_id}")
        rendered.crop(crop_box).save(output_path, format="PNG", optimize=True)
    finally:
        pdf.close()
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def _source_ref(document_name: str, page_number: int, chart_id: str, bbox: BoundingBox) -> str:
    coords = f"{bbox.left:.2f},{bbox.top:.2f},{bbox.right:.2f},{bbox.bottom:.2f}"
    return f"{document_name}#page={page_number}&chart={chart_id}&bbox={coords}"


def validate_chart_extraction(
    extraction: VisionChartExtraction,
    source: ChartSource,
    nearby_text: list[str],
) -> ValidationResult:
    """Apply deterministic completeness, duplication, and OCR cross-checks."""

    issues: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    matched = 0
    normalized_context = " ".join(nearby_text).replace("’", "'").replace("â€™", "'")

    if not extraction.data_points:
        issues.append("No chart data points were extracted")
    if not extraction.reporting_periods:
        issues.append("No reporting periods were extracted")
    if extraction.overall_confidence < 0.85:
        warnings.append(f"Overall model confidence is {extraction.overall_confidence:.2f}")
    warnings.extend(f"Model ambiguity: {item}" for item in extraction.ambiguities)

    for index, point in enumerate(extraction.data_points):
        point.source_ref = source.source_ref
        key = (point.metric.lower(), point.series_name.lower(), point.reporting_period.lower(), point.accounting_basis)
        if key in seen:
            issues.append(f"Duplicate data point at index {index}: {key}")
        seen.add(key)
        if point.reporting_period not in extraction.reporting_periods:
            warnings.append(f"Point period {point.reporting_period!r} is absent from reporting_periods")
        if point.value_text is None:
            warnings.append(f"Unreadable value for {point.metric} / {point.reporting_period}")
        elif point.value_text in normalized_context or point.value_text.replace("’", "'") in normalized_context:
            matched += 1
        else:
            warnings.append(
                f"Value {point.value_text!r} for {point.metric} / {point.reporting_period} was not found in Docling OCR"
            )
        if point.normalized_value is not None and not NORMALIZED_NUMBER_RE.fullmatch(point.normalized_value):
            issues.append(f"Invalid normalized_value {point.normalized_value!r} at index {index}")
        if point.confidence < 0.85:
            warnings.append(f"Low confidence ({point.confidence:.2f}) for {point.metric} / {point.reporting_period}")
        if point.ambiguity:
            warnings.append(f"Point ambiguity for {point.metric} / {point.reporting_period}: {point.ambiguity}")

    return ValidationResult(
        valid=not issues,
        requires_manual_validation=bool(issues or warnings),
        issues=issues,
        warnings=list(dict.fromkeys(warnings)),
        points_checked=len(extraction.data_points),
        points_matched_in_docling_text=matched,
    )


def _failure_record(
    source: ChartSource,
    candidate: ChartCandidate,
    provider_name: str,
    model: str | None,
    status: str,
    error: str,
) -> ChartRecord:
    return ChartRecord(
        status=status,
        source=source,
        detection=candidate.detection_evidence(),
        extraction=None,
        validation=ValidationResult(
            valid=False,
            requires_manual_validation=True,
            issues=[error],
            warnings=[],
        ),
        provider=provider_name,
        model=model,
        error=error,
    )


def extract_positioned_table(document: dict[str, Any], candidate: ChartCandidate) -> VisionChartExtraction | None:
    """Read a complete rectangular table only when cell/header geometry is unambiguous."""
    _, height = _page_size(document, candidate.page_number)
    left, top, right, bottom = _normalized_edges(candidate.bbox, height)
    if (right - left) / max(bottom - top, 1) < 3.5:
        return None
    spans = []
    for item in document.get("texts", []):
        prov = _first_provenance(item)
        if not prov or int(prov.get("page_no", 0)) != candidate.page_number:
            continue
        bbox = _bbox_from_provenance(prov)
        if _intersection_ratio(bbox, candidate.bbox, height) < 0.45:
            continue
        left, top, right, bottom = _normalized_edges(bbox, height)
        spans.append((
            str(item.get("text", "")).strip(),
            (left + right) / 2,
            (top + bottom) / 2,
            left,
            right,
            bottom - top,
        ))
    number = re.compile(r"(?:\(?[+\-]?\d[\d,]*(?:\.\d+)?%?\)?|[-\u2013\u2014])")
    cells = [s for s in spans if number.fullmatch(s[0])]
    if len(cells) < 40:
        return None
    tolerance = median(s[5] for s in cells) * 0.55
    rows: list[list[tuple]] = []
    for cell in sorted(cells, key=lambda s: (s[2], s[1])):
        if not rows or abs(cell[2] - rows[-1][0][2]) > tolerance:
            rows.append([])
        rows[-1].append(cell)
    for row in rows:
        row.sort(key=lambda s: s[1])
    count = len(rows[0])
    if len(rows) < 4 or count < 4 or any(len(row) != count for row in rows):
        return None
    xs = [median(row[i][1] for row in rows) for i in range(count)]
    spacing = min(b - a for a, b in zip(xs, xs[1:]))
    if spacing <= 0 or any(abs(row[i][1] - xs[i]) > spacing * 0.15 for row in rows for i in range(count)):
        return None
    period_re = re.compile(r"(?:[1-4]Q\d{2,4}|Q[1-4]['\u2019]?\s*\d{2,4})", re.I)
    periods = [s for s in spans if period_re.fullmatch(s[0]) and s[2] < rows[0][0][2]]
    if len(periods) != count:
        return None
    periods.sort(key=lambda s: s[1])
    if any(abs(p[1] - x) > spacing * 0.2 for p, x in zip(periods, xs)):
        return None
    headers = [
        s
        for s in spans
        if s[3] > rows[0][0][3] and s[2] < min(p[2] for p in periods) and re.fullmatch(r"[A-Za-z][A-Za-z ]*", s[0])
    ]
    if len(headers) < 2 or len(headers) > count:
        return None
    labels = []
    for row in rows:
        label = [
            s for s in spans if s[4] < row[0][3] and abs(s[2] - row[0][2]) <= tolerance and not number.fullmatch(s[0])
        ]
        if not label:
            return None
        labels.append(" ".join(s[0] for s in sorted(label, key=lambda s: s[1])))
    unit = next((s[0] for s in spans if re.fullmatch(r"\(US\$\s*M{1,2}\)", s[0])), None)
    series = [min(headers, key=lambda s: abs(s[1] - x))[0] for x in xs]
    points = []
    for metric, row in zip(labels, rows):
        for index, cell in enumerate(row):
            value = cell[0]
            normalized = value.replace(",", "").replace("%", "")
            if normalized.startswith("(") and normalized.endswith(")"):
                normalized = "-" + normalized[1:-1]
            points.append(
                VisionDataPoint(
                    metric=metric,
                    series_name=series[index],
                    accounting_basis="unknown",
                    reporting_period=periods[index][0],
                    value_text=value,
                    normalized_value=normalized if NORMALIZED_NUMBER_RE.fullmatch(normalized) else None,
                    unit=unit,
                    confidence=0.95,
                    ambiguity=None,
                    source_ref="",
                )
            )
    return VisionChartExtraction(
        title=candidate.anchor_text,
        chart_type="table",
        financial_metrics=labels,
        reporting_periods=list(dict.fromkeys(p[0] for p in periods)),
        overall_unit=unit,
        legend=[],
        data_points=points,
        factual_trend_summary="Table cells and headers extracted from saved text positions.",
        overall_confidence=0.95,
        ambiguities=["Position-based text extraction; financial accuracy comparison remains deferred."],
    )


def extract_positioned_grouped_percent_chart(
    document: dict[str, Any], candidate: ChartCandidate
) -> VisionChartExtraction | None:
    """Read labelled grouped bars with complete period legends and category labels.

    Requires every panel/category to have exactly one displayed percentage per
    legend period. Never estimates values from bar heights or fills missing cells.
    """
    _, height = _page_size(document, candidate.page_number)
    spans = []
    for item in document.get("texts", []):
        prov = _first_provenance(item)
        if not prov or int(prov.get("page_no", 0)) != candidate.page_number:
            continue
        bbox = _bbox_from_provenance(prov)
        if _intersection_ratio(bbox, candidate.bbox, height) < 0.8:
            continue
        left, top, right, _bottom = _normalized_edges(bbox, height)
        spans.append((str(item.get("text", "")).strip(), (left + right) / 2, (top + _bottom) / 2, left, right))
    period_re = re.compile(r"Q[1-4]['\u2019]\d{2,4}")
    legends = [s for s in spans if len(period_re.findall(s[0])) >= 2 and not period_re.sub("", s[0]).strip()]
    # Separate panels need their own complete legends; single-panel or stacked
    # charts continue through vision extraction.
    if len(legends) < 2:
        return None
    legends.sort(key=lambda s: s[1])
    bounds = [float("-inf")] + [(a[1] + b[1]) / 2 for a, b in zip(legends, legends[1:])] + [float("inf")]
    points, metrics, all_periods = [], [], []
    for i, legend in enumerate(legends):
        panel = [s for s in spans if bounds[i] < s[1] <= bounds[i + 1]]
        periods = period_re.findall(legend[0])
        values = sorted(
            (s for s in panel if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?%", s[0]) and s[2] > legend[2]), key=lambda s: s[1]
        )
        if len(values) < 3 * len(periods):
            return None
        gaps = [b[1] - a[1] for a, b in zip(values, values[1:])]
        threshold = median(gaps) * 1.5
        groups: list[list[tuple]] = [[]]
        for value in values:
            if groups[-1] and value[1] - groups[-1][-1][1] > threshold:
                groups.append([])
            groups[-1].append(value)
        if any(len(g) != len(periods) for g in groups):
            return None
        centers = [sum(s[1] for s in g) / len(g) for g in groups]
        labels = [s for s in panel if s[2] > max(v[2] for v in values) and re.fullmatch(r"[A-Za-z ]+", s[0])]
        names: dict[int, str] = {}
        for label in labels:
            covered = [n for n, x in enumerate(centers) if label[3] - 8 <= x <= label[4] + 8]
            if len(covered) == 1:
                names[covered[0]] = label[0]
            elif len(covered) > 1 and len(label[0].split()) == len(covered):
                names.update(zip(covered, label[0].split()))
        if len(names) != len(groups):
            return None
        titles = sorted((s for s in panel if s[2] < legend[2] and not period_re.search(s[0])), key=lambda s: s[2])
        if not titles:
            return None
        metric = " ".join(s[0] for s in titles)
        metrics.append(metric)
        all_periods.extend(periods)
        for n, group in enumerate(groups):
            for period, value in zip(periods, group):
                points.append(
                    VisionDataPoint(
                        metric=metric,
                        series_name=names[n],
                        accounting_basis="unknown",
                        reporting_period=period,
                        value_text=value[0],
                        normalized_value=value[0][:-1],
                        unit="%",
                        confidence=0.9,
                        ambiguity=None,
                        source_ref="",
                    )
                )
    return VisionChartExtraction(
        title=" / ".join(metrics),
        chart_type="grouped_bar",
        axis_labels=[],
        financial_metrics=metrics,
        reporting_periods=list(dict.fromkeys(all_periods)),
        overall_unit="%",
        legend=[],
        data_points=points,
        factual_trend_summary="Displayed percentage labels extracted by panel, category and legend order.",
        overall_confidence=0.9,
        ambiguities=[
            "Period association follows left-to-right grouped bars and legend order; "
            "accuracy comparison remains deferred."
        ],
    )


def extract_chart(
    provider: ChartVisionProvider | None,
    candidate: ChartCandidate,
    source: ChartSource,
    document: dict[str, Any] | None = None,
) -> ChartRecord:
    """Extract one chart; failures become explicit records and never remove source content."""

    if provider is None:
        return _failure_record(source, candidate, "none", None, "skipped", "Vision extraction was not requested")
    try:
        positioned = (
            (
                extract_positioned_table(document, candidate)
                or extract_positioned_grouped_percent_chart(document, candidate)
            )
            if document is not None
            else None
        )
        if positioned is not None:
            return ChartRecord(
                status="success",
                source=source,
                detection=candidate.detection_evidence(),
                extraction=positioned,
                validation=validate_chart_extraction(positioned, source, candidate.nearby_text),
                provider="docling_text_positions",
                model=None,
            )
        response = provider.extract(
            VisionRequest(
                image_path=Path(source.crop_path),
                document_name=source.document_name,
                page_number=source.page_number,
                chart_id=source.chart_id,
                nearby_text=candidate.nearby_text,
            )
        )
        validation = validate_chart_extraction(response.extraction, source, candidate.nearby_text)
        return ChartRecord(
            status="success",
            source=source,
            detection=candidate.detection_evidence(),
            extraction=response.extraction,
            validation=validation,
            provider=provider.name,
            model=provider.model,
            response_id=response.response_id,
            usage=response.usage,
        )
    except Exception as exc:  # provider/network/schema errors must be recorded per chart
        LOGGER.exception("Chart extraction failed for %s", candidate.chart_id)
        return _failure_record(source, candidate, provider.name, provider.model, "failed", str(exc))


def render_chart_markdown(record: ChartRecord) -> str:
    extraction = record.extraction
    if record.status != "success" or extraction is None:
        raise ValueError("Only successful chart records can be rendered")
    title = extraction.title or f"Financial chart on page {record.source.page_number}"
    lines = [
        f"<!-- finsight-chart:{record.source.chart_id}:start -->",
        f"### Chart: {title}",
        "",
        (
            f"_Source: `{record.source.document_name}`, page {record.source.page_number}, "
            f"chart `{record.source.chart_id}`. Type: {extraction.chart_type}. "
            f"Source reference: `{record.source.source_ref}`._"
        ),
        "",
        "| Metric | Series | Basis | Period | Displayed value | Unit | Confidence |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for point in extraction.data_points:

        def clean(value: str | None) -> str:
            return (value or "null").replace("|", "\\|").replace("\n", " ")

        lines.append(
            f"| {clean(point.metric)} | {clean(point.series_name)} | {point.accounting_basis} | "
            f"{clean(point.reporting_period)} | {clean(point.value_text)} | {clean(point.unit)} | "
            f"{point.confidence:.2f} |"
        )
    lines.extend(["", f"**Trend:** {extraction.factual_trend_summary}"])
    if record.validation.requires_manual_validation:
        validation_notes = "; ".join(record.validation.warnings or record.validation.issues)
        lines.extend(["", "> **Validation required:** " + validation_notes])
    lines.extend([f"<!-- finsight-chart:{record.source.chart_id}:end -->", ""])
    return "\n".join(lines)


def merge_chart_markdown(markdown: str, records: list[ChartRecord]) -> tuple[str, list[str]]:
    """Replace only chart-related placeholders; failed/decorative images remain unchanged."""

    warnings: list[str] = []
    merged = markdown
    original_matches = list(re.finditer(re.escape(IMAGE_MARKER), markdown))
    replacements: list[tuple[int, int, str]] = []
    fallback_records: list[ChartRecord] = []
    for record in records:
        if record.status != "success" or record.extraction is None:
            continue
        marker_start = f"<!-- finsight-chart:{record.source.chart_id}:start -->"
        marker_end = f"<!-- finsight-chart:{record.source.chart_id}:end -->"
        rendered = render_chart_markdown(record)
        if marker_start in merged and marker_end in merged:
            pattern = re.compile(re.escape(marker_start) + r".*?" + re.escape(marker_end), re.DOTALL)
            merged = pattern.sub(rendered.rstrip(), merged, count=1)
            continue
        placeholder_index = record.detection.placeholder_index
        if placeholder_index is None:
            fallback_records.append(record)
        elif placeholder_index >= len(original_matches):
            warnings.append(
                f"No Markdown image placeholder #{placeholder_index} exists for "
                f"{record.source.chart_id}; used page anchor"
            )
            fallback_records.append(record)
        else:
            match = original_matches[placeholder_index]
            replacements.append((match.start(), match.end(), rendered.rstrip()))

    for start, end, replacement in sorted(replacements, reverse=True):
        merged = merged[:start] + replacement + merged[end:]

    for record in fallback_records:
        rendered = render_chart_markdown(record)
        anchor = next((text for text in record.detection.nearby_text if text.strip()), None)
        section_anchor = None
        for candidate in record.detection.nearby_text:
            heading = f"## {candidate.strip()}"
            if heading in merged:
                section_anchor = heading
                break
        needle = section_anchor or anchor
        if needle and needle in merged:
            position = merged.find(needle) + len(needle)
            merged = merged[:position] + "\n\n" + rendered + merged[position:]
        else:
            warnings.append(f"Could not locate page anchor for {record.source.chart_id}; appended chart to Markdown")
            merged = merged.rstrip() + "\n\n" + rendered
    return merged, warnings


def merge_chart_json(
    document: dict[str, Any],
    records: list[ChartRecord],
    *,
    source_pdf: Path,
    model: str | None,
    dpi: int,
) -> dict[str, Any]:
    """Preserve Docling JSON while adding a namespaced chart layer and picture links."""

    merged = copy.deepcopy(document)
    serialized = [record.model_dump(mode="json") for record in records]
    merged["finsight_chart_pipeline"] = {
        "version": PIPELINE_VERSION,
        "source_pdf": str(source_pdf.resolve()),
        "vision_model": model,
        "crop_dpi": dpi,
        "chart_extractions": serialized,
    }
    pictures = merged.get("pictures", [])
    for record in records:
        index = record.source.picture_index
        if index is not None and 0 <= index < len(pictures):
            pictures[index]["finsight_chart_ref"] = record.source.chart_id
            pictures[index]["finsight_chart_status"] = record.status
    return merged


def _record_cache_matches(record: ChartRecord, source: ChartSource, provider: ChartVisionProvider | None) -> bool:
    expected_provider = provider.name if provider else "none"
    expected_model = provider.model if provider else None
    return (
        record.source.crop_sha256 == source.crop_sha256
        and record.provider == expected_provider
        and record.model == expected_model
        and record.status == "success"
    )


def process_charts(
    pdf_path: Path,
    document: dict[str, Any],
    markdown: str,
    output_dir: Path,
    *,
    provider: ChartVisionProvider | None,
    dpi: int = 240,
    page_filter: set[int] | None = None,
    fallback_page_filter: set[int] | None = None,
    max_workers: int = 2,
    force: bool = False,
    reuse_any_success: bool = False,
) -> tuple[list[ChartRecord], str, dict[str, Any], dict[str, Any]]:
    """Run detection, cropping, extraction, validation, caching, and merging."""

    candidates = detect_chart_candidates(
        document,
        page_filter=page_filter,
        fallback_page_filter=fallback_page_filter,
        include_fallback_regions=True,
    )
    LOGGER.info("Detected %d financial chart candidates", len(candidates))
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[ChartCandidate, ChartSource, Path]] = []
    for candidate in candidates:
        crop_path = chart_dir / f"{pdf_path.stem}.{candidate.chart_id}.png"
        crop_hash = crop_chart_region(pdf_path, candidate, crop_path, dpi=dpi)
        source_ref = _source_ref(pdf_path.name, candidate.page_number, candidate.chart_id, candidate.bbox)
        source = ChartSource(
            document_name=pdf_path.name,
            document_path=str(pdf_path.resolve()),
            page_number=candidate.page_number,
            chart_id=candidate.chart_id,
            picture_ref=candidate.picture_ref,
            picture_index=candidate.picture_index,
            bbox=candidate.bbox,
            crop_path=str(crop_path.resolve()),
            crop_sha256=crop_hash,
            source_ref=source_ref,
        )
        prepared.append((candidate, source, chart_dir / f"{candidate.chart_id}.json"))

    records_by_id: dict[str, ChartRecord] = {}
    pending: list[tuple[ChartCandidate, ChartSource, Path]] = []
    for candidate, source, cache_path in prepared:
        if cache_path.exists() and not force:
            try:
                cached = ChartRecord.model_validate(_read_json(cache_path))
                if _record_cache_matches(cached, source, provider) or (
                    reuse_any_success
                    and cached.status == "success"
                    and cached.extraction is not None
                    and cached.source.crop_sha256 == source.crop_sha256
                ):
                    cached.source = source
                    if cached.extraction:
                        for point in cached.extraction.data_points:
                            point.source_ref = source.source_ref
                    records_by_id[candidate.chart_id] = cached
                    LOGGER.info("Reused cached extraction for %s", candidate.chart_id)
                    continue
            except Exception:
                LOGGER.warning("Ignoring invalid chart cache %s", cache_path, exc_info=True)
        pending.append((candidate, source, cache_path))

    def run_one(item: tuple[ChartCandidate, ChartSource, Path]) -> tuple[str, ChartRecord, Path]:
        candidate, source, cache_path = item
        LOGGER.info("Extracting %s (page %s)", candidate.chart_id, candidate.page_number)
        record = extract_chart(provider, candidate, source, document=document)
        LOGGER.info(
            "Finished %s: %s (%s data points)",
            candidate.chart_id,
            record.status,
            len(record.extraction.data_points) if record.extraction else 0,
        )
        return candidate.chart_id, record, cache_path

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(run_one, item): item[0].chart_id for item in pending}
            for future in as_completed(futures):
                chart_id, record, cache_path = future.result()
                records_by_id[chart_id] = record
                _write_json(cache_path, record.model_dump(mode="json"))

    records = [records_by_id[candidate.chart_id] for candidate in candidates]
    merged_markdown, merge_warnings = merge_chart_markdown(markdown, records)
    model = provider.model if provider else None
    merged_json = merge_chart_json(document, records, source_pdf=pdf_path, model=model, dpi=dpi)
    comparison = build_comparison(markdown, merged_markdown, records, merge_warnings)
    return records, merged_markdown, merged_json, comparison


def build_comparison(
    original_markdown: str,
    enhanced_markdown: str,
    records: list[ChartRecord],
    merge_warnings: list[str],
) -> dict[str, Any]:
    successes = [record for record in records if record.status == "success" and record.extraction]
    return {
        "original_image_placeholders": original_markdown.count(IMAGE_MARKER),
        "enhanced_image_placeholders": enhanced_markdown.count(IMAGE_MARKER),
        "detected_chart_regions": len(records),
        "successfully_extracted_charts": len(successes),
        "failed_or_skipped_charts": len(records) - len(successes),
        "recovered_data_points": sum(len(record.extraction.data_points) for record in successes if record.extraction),
        "pages_with_recovered_charts": sorted({record.source.page_number for record in successes}),
        "preserved_unrelated_or_failed_images": enhanced_markdown.count(IMAGE_MARKER),
        "merge_warnings": merge_warnings,
    }


def write_validation_reports(output_dir: Path, records: list[ChartRecord]) -> None:
    report = {
        "procedure": [
            "Open each crop_path and the cited source PDF page side by side.",
            "Verify title, chart type, legend, units, accounting basis, periods, and every value_text.",
            "Confirm negative signs/parentheses, percentages, currency scale, and decimal precision.",
            "Resolve every warning or null value before approving the document for claim verification.",
        ],
        "charts": [record.model_dump(mode="json") for record in records],
    }
    _write_json(output_dir / "chart_validation_report.json", report)
    csv_path = output_dir / "chart_validation_values.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "approved",
                "document",
                "page",
                "chart_id",
                "metric",
                "series",
                "basis",
                "period",
                "displayed_value",
                "normalized_value",
                "unit",
                "confidence",
                "source_ref",
                "ambiguity",
            ],
        )
        writer.writeheader()
        for record in records:
            if not record.extraction:
                continue
            for point in record.extraction.data_points:
                writer.writerow(
                    {
                        "approved": "",
                        "document": record.source.document_name,
                        "page": record.source.page_number,
                        "chart_id": record.source.chart_id,
                        "metric": point.metric,
                        "series": point.series_name,
                        "basis": point.accounting_basis,
                        "period": point.reporting_period,
                        "displayed_value": point.value_text,
                        "normalized_value": point.normalized_value,
                        "unit": point.unit,
                        "confidence": point.confidence,
                        "source_ref": point.source_ref,
                        "ambiguity": point.ambiguity,
                    }
                )


def parse_pdf_with_docling(
    pdf_path: Path,
    output_dir: Path,
    *,
    first_page: int = 1,
    last_page: int | None = None,
    timeout_seconds: float = 900,
) -> tuple[Path, Path]:
    """Preserve the existing OCR/table Docling parse and export Markdown plus JSON."""

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the chart pipeline dependencies with: pip install -e '.[document-ai]'") from exc

    options = PdfPipelineOptions()
    options.do_ocr = True
    options.do_table_structure = True
    options.do_chart_extraction = False
    options.document_timeout = timeout_seconds
    options.ocr_options = RapidOcrOptions(mode=OcrMode.FULL_PAGE)
    if hasattr(options, "do_picture_classification"):
        options.do_picture_classification = True

    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    page_range = (first_page, last_page) if last_page is not None else None
    kwargs = {"page_range": page_range} if page_range else {}
    result = converter.convert(pdf_path, **kwargs)
    if result.errors:
        LOGGER.warning("Docling completed with %d errors: %s", len(result.errors), result.errors)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"{pdf_path.stem}.md"
    json_path = output_dir / f"{pdf_path.stem}.json"
    markdown_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
    _write_json(json_path, result.document.export_to_dict())
    return markdown_path, json_path


def parse_page_spec(value: str | None) -> set[int] | None:
    if not value:
        return None
    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending page range: {part}")
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    if any(page < 1 for page in pages):
        raise ValueError("Page numbers must be positive")
    return pages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Original PDF used for source-accurate chart crops")
    parser.add_argument("output_dir", type=Path, help="Directory for enhanced outputs and chart crops")
    parser.add_argument("--docling-json", type=Path, help="Reuse an existing Docling JSON export")
    parser.add_argument("--markdown", type=Path, help="Reuse the matching existing Docling Markdown export")
    parser.add_argument("--chart-pages", help="Optional pages/ranges, for example 7-10 or 7,9-10")
    parser.add_argument("--provider", choices=("openai", "ollama", "none"), default="openai")
    parser.add_argument("--vision-model", default=os.getenv("FINSIGHT_CHART_VISION_MODEL"))
    parser.add_argument("--ollama-url", default=os.getenv("FINSIGHT_OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="Ignore successful per-chart caches")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    if bool(args.docling_json) != bool(args.markdown):
        raise SystemExit("--docling-json and --markdown must be supplied together")
    if not args.pdf.exists():
        raise SystemExit(f"PDF does not exist: {args.pdf}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.docling_json:
        json_path, markdown_path = args.docling_json, args.markdown
    else:
        markdown_path, json_path = parse_pdf_with_docling(args.pdf, args.output_dir / "docling")
    document = _read_json(json_path)
    markdown = markdown_path.read_text(encoding="utf-8")

    provider: ChartVisionProvider | None
    if args.provider == "openai":
        provider = OpenAIChartVisionProvider(args.vision_model or "gpt-5.4")
    elif args.provider == "ollama":
        provider = OllamaChartVisionProvider(args.vision_model or "qwen3-vl:4b", base_url=args.ollama_url)
    else:
        provider = None
    records, enhanced_markdown, enhanced_json, comparison = process_charts(
        args.pdf,
        document,
        markdown,
        args.output_dir,
        provider=provider,
        dpi=args.dpi,
        page_filter=parse_page_spec(args.chart_pages),
        max_workers=args.workers,
        force=args.force,
    )
    stem = args.pdf.stem
    enhanced_md_path = args.output_dir / f"{stem}.enhanced.md"
    enhanced_json_path = args.output_dir / f"{stem}.enhanced.json"
    enhanced_md_path.write_text(enhanced_markdown, encoding="utf-8")
    _write_json(enhanced_json_path, enhanced_json)
    _write_json(args.output_dir / "chart_output_comparison.json", comparison)
    write_validation_reports(args.output_dir, records)
    LOGGER.info("Enhanced Markdown: %s", enhanced_md_path)
    LOGGER.info("Enhanced JSON: %s", enhanced_json_path)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
