"""Configurable vision-LLM adapters for financial chart extraction."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol
from pydantic import Field

from finsight.agents.narrative.corpus.pdf.chart_models import (
    AccountingBasis,
    LegendItem,
    StrictModel,
    UsageSummary,
    VisionChartExtraction,
    VisionDataPoint,
)


SYSTEM_INSTRUCTIONS = """You extract auditable data from financial charts.
Read the supplied chart image carefully. Return only facts visibly supported by the image.
Do not provide a chain of thought or separate reasoning. Return the requested structured result directly.
All confidence fields must be between 0 and 1. Never express confidence as a percentage such as 100.
Use null when there is no ambiguity. Keep each ambiguity to one short sentence (at most 160 characters).
Keep the trend summary to at most two short sentences (360 characters). Do not add explanatory reasoning.
Preserve value_text exactly as displayed, including currency symbols, commas, percent signs,
parentheses, minus signs, and decimal precision. normalized_value is a plain signed decimal
string without currency symbols, commas, or percent signs; use null when normalization is not
reliable. Use null for unreadable values and explain the problem in ambiguity. Never infer a
missing value from bar height. Distinguish GAAP and non-GAAP exactly; use unknown if the image
does not establish the basis. Include every visible reporting period and metric/series
relationship. Capture visible x/y axis labels and unit scale when legible; use an empty list if
axis labels are absent or unreadable. List each metric, period, axis label and legend item
only once. financial_metrics contains distinct metric names, not one entry per data point.
If the chart exceeds the schema's list limits, disclose that limitation in ambiguities
instead of silently omitting values. Extract only visibly associated series/period values;
never generate a Cartesian product of all series and all periods. For a nonquantitative
timeline, process diagram, or logo graphic, return no numeric data points (an empty series
list in the compact schema). Describe visible events briefly in the factual summary instead;
years labeling events are not measured values. Keep the trend summary concise and factual.
source_ref must be the empty string;
the pipeline fills it with trusted source metadata after extraction."""


@dataclass(frozen=True)
class VisionRequest:
    image_path: Path
    document_name: str
    page_number: int
    chart_id: str
    nearby_text: list[str]
    series_limit: int | None = None
    periods_per_series_limit: int | None = None


@dataclass(frozen=True)
class VisionResponse:
    extraction: VisionChartExtraction
    response_id: str | None = None
    usage: UsageSummary | None = None


class ChartVisionProvider(Protocol):
    name: str
    model: str | None

    def extract(self, request: VisionRequest) -> VisionResponse: ...


class _CompactValue(StrictModel):
    reporting_period: str = Field(max_length=160)
    value_text: str | None = Field(max_length=160)
    normalized_value: str | None = Field(max_length=64)
    confidence: float = Field(ge=0, le=1)
    ambiguity: str | None = Field(max_length=160)


class _CompactSeries(StrictModel):
    metric: str = Field(max_length=160)
    series_name: str = Field(max_length=160)
    accounting_basis: AccountingBasis
    unit: str | None = Field(max_length=64)
    values: list[_CompactValue] = Field(max_length=128)


class _CompactChartExtraction(StrictModel):
    title: str | None = Field(max_length=256)
    chart_type: str = Field(max_length=64)
    axis_labels: list[Annotated[str, Field(max_length=160)]] = Field(default_factory=list, max_length=16)
    financial_metrics: list[Annotated[str, Field(max_length=160)]] = Field(max_length=32)
    reporting_periods: list[Annotated[str, Field(max_length=160)]] = Field(max_length=64)
    overall_unit: str | None = Field(max_length=64)
    legend: list[LegendItem] = Field(max_length=32)
    # Dense tables can have one series per metric and geography; 8 metrics x
    # 5 geographies already exceeds 32. Keep all visible combinations possible.
    series: list[_CompactSeries] = Field(max_length=128)
    factual_trend_summary: str = Field(max_length=360)
    overall_confidence: float = Field(ge=0, le=1)
    ambiguities: list[Annotated[str, Field(max_length=160)]] = Field(max_length=6)

    def expand(self) -> VisionChartExtraction:
        points = [
            VisionDataPoint(
                metric=series.metric,
                series_name=series.series_name,
                accounting_basis=series.accounting_basis,
                reporting_period=value.reporting_period,
                value_text=value.value_text,
                normalized_value=value.normalized_value,
                unit=series.unit,
                confidence=value.confidence,
                ambiguity=value.ambiguity,
                source_ref="",
            )
            for series in self.series
            for value in series.values
        ]
        return VisionChartExtraction(
            title=self.title,
            chart_type=self.chart_type,
            axis_labels=self.axis_labels,
            financial_metrics=self.financial_metrics,
            reporting_periods=self.reporting_periods,
            overall_unit=self.overall_unit,
            legend=self.legend,
            data_points=points,
            factual_trend_summary=self.factual_trend_summary,
            overall_confidence=self.overall_confidence,
            ambiguities=self.ambiguities,
        )


class _LeanSeries(StrictModel):
    metric: str = Field(max_length=160)
    series_name: str = Field(max_length=160)
    accounting_basis: AccountingBasis
    unit: str | None = Field(max_length=64)
    reporting_periods: list[str] = Field(max_length=64)
    values: list[str | None] = Field(max_length=64)
    confidence: float = Field(ge=0, le=1)


class _LeanChartExtraction(StrictModel):
    title: str | None = Field(max_length=256)
    chart_type: str = Field(max_length=64)
    axis_labels: list[str] = Field(max_length=16)
    overall_unit: str | None = Field(max_length=64)
    series: list[_LeanSeries] = Field(max_length=128)
    legend: list[LegendItem] = Field(default_factory=list, max_length=32)
    factual_trend_summary: str = Field(default="", max_length=360)
    ambiguities: list[Annotated[str, Field(max_length=160)]] = Field(default_factory=list, max_length=6)

    def expand(self) -> VisionChartExtraction:
        points = []
        for series in self.series:
            if len(series.reporting_periods) != len(series.values):
                raise ValueError(f"Period/value count differs for {series.metric} / {series.series_name}")
            for period, value in zip(series.reporting_periods, series.values):
                normalized = re.sub(r"[$,%\s]", "", value or "")
                if normalized.startswith("(") and normalized.endswith(")"):
                    normalized = "-" + normalized[1:-1]
                points.append(
                    VisionDataPoint(
                        metric=series.metric,
                        series_name=series.series_name,
                        accounting_basis=series.accounting_basis,
                        reporting_period=period,
                        value_text=value,
                        normalized_value=normalized if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized) else None,
                        unit=series.unit,
                        confidence=series.confidence,
                        ambiguity="Unreadable value" if value is None else None,
                        source_ref="",
                    )
                )
        return VisionChartExtraction(
            title=self.title,
            chart_type=self.chart_type,
            axis_labels=self.axis_labels,
            financial_metrics=list(dict.fromkeys(s.metric for s in self.series)),
            reporting_periods=list(dict.fromkeys(p.reporting_period for p in points)),
            overall_unit=self.overall_unit,
            legend=self.legend,
            data_points=points,
            factual_trend_summary=self.factual_trend_summary,
            overall_confidence=min((s.confidence for s in self.series), default=0),
            ambiguities=self.ambiguities,
        )


class _MatrixColumn(StrictModel):
    series_name: str
    reporting_period: str
    accounting_basis: AccountingBasis
    unit: str | None


class _MatrixRow(StrictModel):
    metric: str
    unit: str | None
    confidence: float = Field(ge=0, le=1)
    values: list[str | None] = Field(max_length=128)


class _MatrixChartExtraction(StrictModel):
    """Lossless table layout: one row/column header, then displayed cell strings.

    Avoid repeating metric, geography, period, confidence and normalization keys
    for every cell in a dense financial table. Stored output still uses the same
    public VisionChartExtraction contract as all other charts.
    """

    title: str | None
    chart_type: str
    axis_labels: list[str] = Field(default_factory=list)
    overall_unit: str | None
    columns: list[_MatrixColumn] = Field(min_length=1, max_length=128)
    rows: list[_MatrixRow] = Field(max_length=128)
    legend: list[LegendItem] = Field(max_length=32)
    factual_trend_summary: str = Field(default="", max_length=360)
    overall_confidence: float = Field(ge=0, le=1)
    ambiguities: list[Annotated[str, Field(max_length=160)]] = Field(default_factory=list, max_length=6)

    def expand(self, nearby_text: list[str] | None = None) -> VisionChartExtraction:
        points = []
        # Some small models return one geography header but both period cells.
        # Restore repeated subheaders only when the source text explicitly repeats
        # every period once per group and every row has exactly the matching size.
        if nearby_text:
            period_headers = [
                text.strip()
                for text in nearby_text
                if re.fullmatch(r"(?:[1-4]Q\d{2,4}|Q[1-4]['’]?\s*\d{2,4})", text.strip())
            ]
            periods = list(dict.fromkeys(period_headers))
            if (
                len(periods) >= 2
                and self.rows
                and all(len(row.values) == len(self.columns) * len(periods) for row in self.rows)
                and all(column.reporting_period == periods[0] for column in self.columns)
                and all(period_headers.count(period) == len(self.columns) for period in periods)
            ):
                self.columns = [
                    column.model_copy(update={"reporting_period": period})
                    for column in self.columns
                    for period in periods
                ]
                self.ambiguities.append("Repeated period subheaders restored from source text; cell strings unchanged.")
            if not re.search(r"\b(?:non[ -]?)?gaap\b", " ".join(nearby_text), re.I):
                for column in self.columns:
                    column.accounting_basis = "unknown"
        for row in self.rows:
            if len(row.values) != len(self.columns):
                raise ValueError(
                    f"Table row {row.metric!r} has {len(row.values)} cells; expected {len(self.columns)} columns"
                )
            for column, value in zip(self.columns, row.values, strict=True):
                normalized = None
                if value is not None:
                    candidate = value.strip().replace(",", "").replace("−", "-")
                    candidate = candidate.strip("$€£% ")
                    if candidate.startswith("(") and candidate.endswith(")"):
                        candidate = "-" + candidate[1:-1]
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
                        normalized = candidate
                points.append(
                    VisionDataPoint(
                        metric=row.metric,
                        series_name=column.series_name,
                        accounting_basis=column.accounting_basis,
                        reporting_period=column.reporting_period,
                        value_text=value,
                        normalized_value=normalized,
                        unit=row.unit or column.unit or self.overall_unit,
                        confidence=row.confidence,
                        ambiguity="Unreadable table cell" if value is None else None,
                        source_ref="",
                    )
                )
        return VisionChartExtraction(
            title=self.title,
            chart_type=self.chart_type,
            axis_labels=self.axis_labels,
            financial_metrics=list(dict.fromkeys(row.metric for row in self.rows)),
            reporting_periods=list(dict.fromkeys(column.reporting_period for column in self.columns)),
            overall_unit=self.overall_unit,
            legend=self.legend,
            data_points=points,
            factual_trend_summary=self.factual_trend_summary,
            overall_confidence=self.overall_confidence,
            ambiguities=self.ambiguities,
        )


class OpenAIChartVisionProvider:
    """OpenAI Responses API adapter using image input and Structured Outputs."""

    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_output_tokens: int = 6000,
        timeout_seconds: float = 180,
        max_retries: int = 0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("Install the chart pipeline dependencies with: pip install -e '.[document-ai]'") from exc
        from finsight.llm.spend import api_key as load_api_key

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(
            api_key=api_key or load_api_key("openai"),
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._fatal_error: str | None = None
        self._fatal_error_lock = threading.Lock()
        self.usage_tracker = None

    def extract(self, request: VisionRequest) -> VisionResponse:
        with self._fatal_error_lock:
            if self._fatal_error:
                raise RuntimeError(f"Vision provider unavailable: {self._fatal_error}")
        mime_type = mimetypes.guess_type(request.image_path.name)[0] or "image/png"
        encoded = base64.b64encode(request.image_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        context = "\n".join(request.nearby_text)
        prompt = (
            f"Extract chart data from {request.document_name}, page {request.page_number}, "
            f"chart {request.chart_id}. Docling OCR context follows and may contain errors; "
            f"the image is authoritative.\n\nOCR context:\n{context[:12000]}"
        )
        try:
            response = self.client.responses.parse(
                model=self.model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url, "detail": "high"},
                        ],
                    }
                ],
                text_format=VisionChartExtraction,
                max_output_tokens=self.max_output_tokens,
                store=False,
            )
        except BaseException as exc:
            if self.usage_tracker is not None:
                self.usage_tracker.record(request, error=exc)
            error_code = getattr(exc, "code", None)
            error_text = str(exc)
            if error_code in {"insufficient_quota", "credit_balance_exhausted"} or "no credits remaining" in error_text:
                with self._fatal_error_lock:
                    self._fatal_error = error_text
            raise
        # Record returned usage before interpreting extraction output. A refusal
        # or missing parsed result can still have consumed billed tokens.
        if self.usage_tracker is not None:
            self.usage_tracker.record(request, response=response)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Vision model returned no parsed chart extraction")
        usage_obj = getattr(response, "usage", None)
        usage = None
        if usage_obj is not None:
            usage = UsageSummary(
                input_tokens=getattr(usage_obj, "input_tokens", None),
                output_tokens=getattr(usage_obj, "output_tokens", None),
                total_tokens=getattr(usage_obj, "total_tokens", None),
            )
        return VisionResponse(extraction=parsed, response_id=getattr(response, "id", None), usage=usage)


class OllamaChartVisionProvider:
    """Local Ollama adapter using vision input and a Pydantic JSON schema."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3-vl:4b",
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 900,
        keep_alive: str = "10m",
        context_size: int = 8192,
        max_output_tokens: int = 4000,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive
        self.context_size = context_size
        self.max_output_tokens = max_output_tokens
        self._fatal_error: str | None = None

    def extract(self, request: VisionRequest) -> VisionResponse:
        if self._fatal_error:
            raise RuntimeError(self._fatal_error)
        from PIL import Image

        # Keep the original high-resolution crop untouched. Bound the inference
        # image's pixel count so rendered 5,000+px slides do not consume the
        # entire context window before any JSON can be generated.
        with Image.open(request.image_path) as crop:
            original_size = crop.size
            max_pixels = 2_000_000
            scale = min(1.0, (max_pixels / (crop.width * crop.height)) ** 0.5)
            if scale < 1:
                inference = crop.resize(
                    (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.Resampling.LANCZOS
                )
                buffer = io.BytesIO()
                inference.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
                input_size = inference.size
            else:
                image_bytes = request.image_path.read_bytes()
                input_size = original_size
        encoded = base64.b64encode(image_bytes).decode("ascii")
        dense_table = (
            original_size[0] / original_size[1] >= 3.5
            and sum(bool(re.fullmatch(r"[\d,().%+\-]+", text.strip())) for text in request.nearby_text) >= 40
        )
        extraction_schema = _MatrixChartExtraction if dense_table else _LeanChartExtraction
        context = "\n".join(request.nearby_text)
        prompt = (
            f"Extract chart data from {request.document_name}, page {request.page_number}, "
            f"chart {request.chart_id}. Docling OCR context follows and may contain errors; "
            f"the image is authoritative. Use only context nearest the chart.\n\n"
            f"OCR context:\n{context[:3000]}\n\n"
            "Return an object matching the supplied JSON schema. "
            "Use compact single-line JSON without indentation or newlines. "
            "Do not show reasoning; output only the JSON object. /no_think"
        )
        if dense_table:
            prompt += (
                "\nThis region is a dense financial table. Use the matrix schema: columns contain "
                "each visible geography/series and period pair in left-to-right order; rows contain "
                "each visible metric with its displayed cell strings in that exact column order. "
                "Include every row, column and total, preserving signs, commas and precision. "
                "Every row must have exactly one value per column. Use null for unreadable cells. "
                "Do not calculate missing totals or duplicate headers per cell. "
                "Row confidence applies to the row's cells. Output only compact JSON. /no_think"
            )
        else:
            prompt += (
                "\nFor each series, reporting_periods and values are parallel arrays in the same order. "
                "Write each displayed value once, with one series per visible metric/group. "
                "Do not repeat a series or add commentary inside cell strings."
            )
        schema = extraction_schema.model_json_schema()
        # Numeric extraction is the task. Small local models can spend thousands
        # of tokens repeating prose after all cells are complete. Leave summaries
        # empty instead of requesting free-form commentary in the Ollama grammar.
        for field in ("factual_trend_summary", "ambiguities"):
            schema["properties"].pop(field, None)
            if field in schema.get("required", []):
                schema["required"].remove(field)
        if not dense_table and request.series_limit is not None:
            schema["properties"]["series"]["maxItems"] = min(128, request.series_limit)
        if not dense_table and request.periods_per_series_limit is not None:
            for field in ("values", "reporting_periods"):
                schema["$defs"]["_LeanSeries"]["properties"][field]["maxItems"] = min(
                    64, request.periods_per_series_limit
                )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt, "images": [encoded]},
            ],
            # Structured chart extraction needs a final JSON body. Some Qwen variants
            # otherwise spend the generation budget in a separate thinking field.
            "think": False,
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "num_predict": self.max_output_tokens, "num_ctx": self.context_size},
            "keep_alive": self.keep_alive,
        }
        http_request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve`.") from exc

        # Keep the complete model response next to the source crop. A 1,000-char
        # error excerpt cannot distinguish truncated JSON from schema violations.
        diagnostic_path = request.image_path.with_suffix(".ollama_response.json")
        diagnostic_path.write_text(
            json.dumps(
                {
                    "model": self.model,
                    "context_size": self.context_size,
                    "max_output_tokens": self.max_output_tokens,
                    "schema_variant": "matrix" if dense_table else "lean_series",
                    "original_image_size": original_size,
                    "input_image_size": input_size,
                    "response": response_data,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        content = ((response_data.get("message") or {}).get("content") or "").strip()
        if not content:
            message = response_data.get("message") or {}
            details = response_data.get("error") or (
                f"empty response (done_reason={message.get('done_reason') or response_data.get('done_reason')!r}, "
                f"eval_count={response_data.get('eval_count')!r}, thinking_present={bool(message.get('thinking'))})"
            )
            if message.get("thinking") and response_data.get("done_reason") == "length":
                self._fatal_error = (
                    f"Ollama model {self.model} exhausted its output budget in thinking without returning JSON. "
                    "Use a non-thinking vision model, such as qwen3-vl:4b-instruct-q4_K_M."
                )
            raise RuntimeError(f"Ollama returned no chart JSON: {details}")
        try:
            parsed = extraction_schema.model_validate_json(content)
            extraction = parsed.expand(request.nearby_text) if dense_table else parsed.expand()
        except Exception as exc:
            reason = str(exc).splitlines()[:12]
            raise RuntimeError(
                f"Ollama returned invalid chart JSON "
                f"(done_reason={response_data.get('done_reason')!r}, "
                f"eval_count={response_data.get('eval_count')!r}, "
                f"context={self.context_size}, max_output={self.max_output_tokens}). "
                f"Parser: {'; '.join(reason)}. Full response: {diagnostic_path}"
            ) from exc

        input_tokens = response_data.get("prompt_eval_count")
        output_tokens = response_data.get("eval_count")
        usage = UsageSummary(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(
                input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
            ),
        )
        created_at = response_data.get("created_at")
        response_id = f"ollama:{created_at}" if created_at else None
        return VisionResponse(extraction=extraction, response_id=response_id, usage=usage)
