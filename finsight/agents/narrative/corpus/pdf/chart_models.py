"""Data contracts for traceable financial-chart extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


AccountingBasis = Literal["GAAP", "non-GAAP", "not_applicable", "unknown"]
ExtractionStatus = Literal["success", "failed", "skipped"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundingBox(StrictModel):
    left: float
    top: float
    right: float
    bottom: float
    coord_origin: Literal["BOTTOMLEFT", "TOPLEFT"] = "BOTTOMLEFT"


class ChartSource(StrictModel):
    document_name: str
    document_path: str
    page_number: int = Field(ge=1)
    chart_id: str
    picture_ref: str | None
    picture_index: int | None
    bbox: BoundingBox
    crop_path: str
    crop_sha256: str
    source_ref: str


class DetectionEvidence(StrictModel):
    accepted: bool
    confidence: float = Field(ge=0, le=1)
    reasons: list[str]
    classification_labels: list[str]
    nearby_text: list[str]
    placeholder_index: int | None
    fallback_region: bool = False


class LegendItem(StrictModel):
    label: str
    series_name: str
    visual_marker: str | None
    meaning: str | None


class VisionDataPoint(StrictModel):
    """A value as read by the model; source_ref is filled by the pipeline."""

    metric: str
    series_name: str
    accounting_basis: AccountingBasis
    reporting_period: str
    value_text: str | None
    normalized_value: str | None
    unit: str | None
    confidence: float = Field(ge=0, le=1)
    ambiguity: str | None
    source_ref: str


class VisionChartExtraction(StrictModel):
    title: str | None
    chart_type: str
    axis_labels: list[str] = Field(default_factory=list)
    financial_metrics: list[str]
    reporting_periods: list[str]
    overall_unit: str | None
    legend: list[LegendItem]
    data_points: list[VisionDataPoint]
    factual_trend_summary: str
    overall_confidence: float = Field(ge=0, le=1)
    ambiguities: list[str]


class ValidationResult(StrictModel):
    valid: bool
    requires_manual_validation: bool
    issues: list[str]
    warnings: list[str]
    points_checked: int = 0
    points_matched_in_docling_text: int = 0


class UsageSummary(StrictModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ChartRecord(StrictModel):
    status: ExtractionStatus
    source: ChartSource
    detection: DetectionEvidence
    extraction: VisionChartExtraction | None
    validation: ValidationResult
    provider: str
    model: str | None
    response_id: str | None = None
    usage: UsageSummary | None = None
    error: str | None = None

