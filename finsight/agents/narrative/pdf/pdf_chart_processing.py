"""Resumable PDF audit, Docling/OCR, chart extraction, and canonical promotion.

Examples::
    python -m finsight.agents.narrative.pdf.pdf_chart_processing --all --audit-only
    python -m finsight.agents.narrative.pdf.pdf_chart_processing --all --ollama-model qwen3-vl:4b-instruct-q4_K_M
    python -m finsight.agents.narrative.pdf.pdf_chart_processing --all --promote-only --rebuild-vectors

Experimental artifacts remain below each company's experiments directory. Visual
evaluation is separate; unreadable fields and unsupported regions remain explicit.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pypdf import PdfReader

from finsight.agents.narrative.processing.corpus import (
    CANONICAL_MANIFESTS, CorpusDocument, file_sha256, load_documents, validate_document,
)
from finsight.agents.narrative.pdf.pdf_enrichment import ENRICHMENT_VERSION, docling_blocks, write_json

PIPELINE_VERSION = ENRICHMENT_VERSION


@dataclass(frozen=True)
class PageAudit:
    page_number: int
    classification: str
    text_characters: int
    image_count: int
    table_like: bool
    chart_like: bool
    route: str
    reason: str


def _image_count(page: Any) -> int:
    """Count resources without decoding every image into a bitmap."""
    visited = set()

    def count(resources: Any) -> int:
        if not resources:
            return 0
        objects = resources.get_object().get("/XObject", {})
        objects = objects.get_object() if hasattr(objects, "get_object") else objects
        total = 0
        for ref in objects.values():
            obj = ref.get_object()
            if id(obj) in visited:
                continue
            visited.add(id(obj))
            if obj.get("/Subtype") == "/Image":
                total += 1
            elif obj.get("/Subtype") == "/Form":
                total += count(obj.get("/Resources"))
        return total

    return count(page.get("/Resources"))


def _chart_like(text: str) -> bool:
    periods = len(re.findall(r"\b(?:q[1-4](?:['’]?\s*\d{2,4})?|fy\s*\d{2,4}|20\d{2})\b", text, re.I))
    financial = re.search(r"\b(revenue|margin|income|cash flow|earnings|eps|gaap|growth)\b", text, re.I)
    return periods >= 2 and len(re.findall(r"\d[\d,.]*%?", text)) >= 4 and bool(financial) and len(text.split()) < 350


def _table_like(text: str) -> bool:
    rows = [line for line in text.splitlines() if line.strip()]
    numeric_rows = sum(len(re.findall(r"\d[\d,.]*%?", line)) >= 2 and len(line.split()) < 18 for line in rows)
    return (numeric_rows >= 4 and numeric_rows / max(1, len(rows)) >= 0.35
            and bool(re.search(r"\b(revenue|income|cash|million|billion|20\d{2})\b", text, re.I)))


def audit_pdf(path: Path, *, docling_available: bool = True, document_type: str | None = None) -> list[PageAudit]:
    audits = []
    presentation = any(word in (document_type or "").lower() for word in ("presentation", "slide"))
    with path.open("rb") as source:
        for number, page in enumerate(PdfReader(source).pages, 1):
            text = page.extract_text() or ""
            images, table = _image_count(page), _table_like(text)
            chart = bool((images or presentation) and _chart_like(text))
            if not text.strip():
                label, route, reason = "image_only", "docling_ocr", "No native text layer"
            elif sum(c.isalnum() for c in text) < 24 and images:
                label, route, reason = "requiring_ocr", "docling_ocr", "Image-bearing page has very little native text"
            elif chart:
                label, route, reason = ("chart_heavy", "docling_chart",
                                        "Visual page contains financial periods and values")
            elif table:
                label, route, reason = "table_heavy", "docling_table", "Short numeric rows require table structure"
            else:
                label, route, reason = "digitally_readable", "native_pdf_text", "Use native PDF text"
            if route.startswith("docling") and not docling_available:
                route = "unsupported_" + route
            audits.append(PageAudit(number, label, len(text), images, table, chart, route, reason))
    return audits


def parse_page_spec(value: str | None) -> set[int] | None:
    if not value:
        return None
    pages = set()
    for part in value.split(","):
        first, last = map(int, part.split("-", 1)) if "-" in part else (int(part), int(part))
        if first < 1 or last < first:
            raise ValueError(f"Invalid page range: {part}")
        pages.update(range(first, last + 1))
    return pages


def _experiment_root(workspace: Path, ticker: str, experiment: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment) or not re.fullmatch(r"[A-Z0-9.-]+", ticker.upper()):
        raise ValueError("Ticker/experiment contains unsupported path characters")
    return workspace / "data/narrative/processed" / ticker.upper() / "experiments" / experiment


def _ranges(pages: set[int]) -> list[tuple[int, int]]:
    output = []
    for page in sorted(pages):
        if output and output[-1][1] == page - 1:
            output[-1] = (output[-1][0], page)
        else:
            output.append((page, page))
    return output


class DoclingRunner:
    """Reuse layout/OCR models; process only contiguous routed page ranges."""

    def __init__(self) -> None:
        self.converters = {}

    def convert(self, document: CorpusDocument, output: Path, first: int, last: int,
                *, ocr: bool) -> tuple[Path, Path]:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if ocr not in self.converters:
            options = PdfPipelineOptions()
            options.do_ocr, options.do_table_structure, options.do_chart_extraction = ocr, True, False
            options.document_timeout = 600
            if ocr:
                options.ocr_options = RapidOcrOptions(mode=OcrMode.FULL_PAGE)
            if hasattr(options, "do_picture_classification"):
                options.do_picture_classification = True
            self.converters[ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        messages: list[str] = []

        class WarningCapture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if record.levelno >= logging.WARNING:
                    messages.append(record.getMessage())

        capture = WarningCapture(level=logging.WARNING)
        loggers = (logging.getLogger(), logging.getLogger("MatchingPostProcessor"))
        for logger in loggers:
            logger.addHandler(capture)
        try:
            result = self.converters[ocr].convert(document.local_path, page_range=(first, last))
        finally:
            for logger in loggers:
                logger.removeHandler(capture)
        if str(getattr(result.status, "value", result.status)) not in {"success", "partial_success"}:
            raise RuntimeError(f"Docling status {result.status}: {result.errors}")
        output.mkdir(parents=True, exist_ok=True)
        md, js = output / "document.md", output / "document.json"
        md.write_text(result.document.export_to_markdown(), encoding="utf-8")
        write_json(js, result.document.export_to_dict())
        write_json(output / "metadata.json", {"source_sha256": document.sha256, "pages": [first, last],
                   "ocr": ocr, "errors": [str(error) for error in result.errors],
                   "warnings": list(dict.fromkeys(messages))})
        return md, js


def _seed_trial_cache(document: CorpusDocument, destination: Path, workspace: Path) -> None:
    for path in (workspace / "tmp/pdfs/docling_chart_ollama_test/charts").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        source = record.get("source", {})
        if record.get("status") != "success" or source.get("document_name") != document.local_path.name:
            continue
        original = Path(source.get("document_path", ""))
        if not original.is_file() or file_sha256(original) != document.sha256:
            continue
        target = destination / "charts" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)


def _full_cached_export(document: CorpusDocument, destination: Path, workspace: Path,
                        required: set[int]) -> tuple[Path, Path] | None:
    for directory in (destination / "docling", workspace / "tmp/pdfs/docling_trial/output_full"):
        md, js = directory / f"{document.local_path.stem}.md", directory / f"{document.local_path.stem}.json"
        if not md.is_file() or not js.is_file():
            continue
        parsed = json.loads(js.read_text(encoding="utf-8"))
        if parsed.get("origin", {}).get("filename") != document.local_path.name:
            continue
        available = {int(page) for page in parsed.get("pages", {})}
        previous = destination / "audit.json"
        history = json.loads(previous.read_text(encoding="utf-8")) if previous.is_file() else {}
        trial = directory.name == "output_full" and document.document_id == "INTC-IR-2b85d5c638b905e1"
        referenced = any((workspace / item["json"]).resolve() == js.resolve()
                         for item in history.get("docling_exports", []))
        if history.get("docling_json"):
            referenced |= (destination.parent / history["docling_json"]).resolve() == js.resolve()
        current = history.get("sha256") == document.sha256 and referenced
        if required <= available and (trial or current):
            return md, js
    return None


def _checkpoint(root: Path, report: dict[str, Any], *, final: bool = False) -> None:
    docs = report["documents"]
    report["documents_completed"] = len(docs)
    report["pages_audited"] = sum(len(doc.get("pages", [])) for doc in docs)
    report["chart_count"] = sum(len(doc.get("chart_records", [])) for doc in docs)
    report["successful_charts"] = sum(item["status"] == "success" for doc in docs
                                      for item in doc.get("chart_records", []))
    report["documents_with_errors"] = [{"document_id": doc["document_id"], "warnings": doc.get("warnings", []),
                                        "chart_failures": doc.get("chart_failures", [])}
                                       for doc in docs if doc.get("warnings") or doc.get("chart_failures")]
    report["coverage_limitations"] = [item for doc in docs for item in doc.get("coverage_limitations", [])]
    report["unsupported_pages"] = [item for item in report["coverage_limitations"] if item.get("scope") == "page"]
    if final:
        report["status"] = "partial" if report["documents_with_errors"] else "complete"
    write_json(root / ("audit_report.json" if report.get("audit_only") else "experiment_report.json"), report)


def process_pdf_experiment(workspace: Path, ticker: str, *, experiment: str = "pdf_chart_processing",
                           manifests: Iterable[Path] = CANONICAL_MANIFESTS, provider: Any | None = None,
                           pages: set[int] | None = None, chart_workers: int = 1, audit_only: bool = False,
                           document_ids: set[str] | None = None, runner: DoclingRunner | None = None,
                           force: bool = False) -> dict[str, Any]:
    from finsight.agents.narrative.processing.process_corpus import _write_jsonl
    from finsight.agents.narrative.pdf.docling_charts import process_charts

    workspace, ticker = workspace.resolve(), ticker.upper()
    documents = [doc for doc in load_documents(ticker, manifests=manifests, root=workspace)
                 if doc.local_path.suffix.lower() == ".pdf" and (not document_ids or doc.document_id in document_ids)]
    root, runner = _experiment_root(workspace, ticker, experiment), runner or DoclingRunner()
    cached_audits: dict[str, dict[str, Any]] = {}
    audit_report_path = root / "audit_report.json"
    if not audit_only and not force and audit_report_path.is_file():
        try:
            cached_report = json.loads(audit_report_path.read_text(encoding="utf-8"))
            if (cached_report.get("audit_only") is True
                    and cached_report.get("status") == "complete"
                    and cached_report.get("pipeline_version") == PIPELINE_VERSION
                    and cached_report.get("configuration", {}).get("pages") is None):
                cached_audits = {row["document_id"]: row for row in cached_report.get("documents", [])}
        except (OSError, ValueError, KeyError, TypeError):
            # An unusable cache must not prevent a fresh source audit.
            cached_audits = {}
    report = {"status": "running", "experiment": experiment, "pipeline_version": PIPELINE_VERSION,
              "company_ticker": ticker, "created_at": datetime.now(timezone.utc).isoformat(),
              "documents_selected": len(documents), "audit_only": audit_only, "documents": [],
              "configuration": {"provider": getattr(provider, "name", None),
                                "model": getattr(provider, "model", None), "chart_workers": chart_workers,
                                "max_output_tokens": getattr(provider, "max_output_tokens", None),
                                "context_size": getattr(provider, "context_size", None),
                                "timeout_seconds": getattr(provider, "timeout_seconds", None),
                                "pages": sorted(pages) if pages is not None else None,
                                "document_ids": sorted(document_ids) if document_ids is not None else None}}
    _checkpoint(root, report)
    for number, document in enumerate(documents, 1):
        print(f"[{ticker} {number}/{len(documents)}] Auditing {document.local_path.name}", flush=True)
        validate_document(document)
        cached_audit = cached_audits.get(document.document_id, {})
        if (cached_audit.get("sha256") == document.sha256
                and cached_audit.get("document_type") == document.document_type):
            try:
                audit = [PageAudit(**page) for page in cached_audit["pages"]]
                print("  Reusing source-matched full-document page audit", flush=True)
            except (KeyError, TypeError, ValueError):
                audit = audit_pdf(document.local_path, document_type=document.document_type)
        else:
            audit = audit_pdf(document.local_path, document_type=document.document_type)
        audit = [page for page in audit if pages is None or page.page_number in pages]
        selected = {page.page_number for page in audit if page.route.startswith("docling")}
        destination = root / document.document_id
        summary = {**document.to_json(), "pipeline_version": PIPELINE_VERSION,
                   "pages": [asdict(page) for page in audit], "chart_records": [], "warnings": [],
                   "coverage_limitations": [], "blocks_path": "blocks.jsonl"}
        print("  Routes: " + str(dict(Counter(page.route for page in audit))), flush=True)
        if audit_only:
            write_json(destination / "page_audit.json", summary)
            report["documents"].append(summary)
            _checkpoint(root, report)
            continue
        # Resume complete documents as well as individual Docling/chart ranges.
        # Metadata is refreshed from the current manifest, while matching source
        # bytes and page selection allow the extracted content to be retained.
        previous_path = destination / "audit.json"
        if not force and previous_path.is_file():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            previous_blocks = destination / previous.get("blocks_path", "blocks.jsonl")
            if (previous.get("sha256") == document.sha256
                    and previous.get("pipeline_version") == PIPELINE_VERSION
                    and previous.get("pages") == summary["pages"]
                    and "blocks_created" in previous
                    and previous_blocks.is_file()
                    and not previous.get("warnings")
                    and not previous.get("chart_failures")
                    and all(item.get("status") == "success"
                            for item in previous.get("chart_records", []))):
                previous.update(document.to_json())
                previous["reused_completed_extraction"] = True
                write_json(previous_path, previous)
                report["documents"].append(previous)
                _checkpoint(root, report)
                print("  Reusing completed source-matched PDF extraction", flush=True)
                continue
        blocks = []
        try:
            exports = []
            full = None if force else _full_cached_export(document, destination, workspace, selected)
            if selected and full:
                print("  Reusing complete Docling export", flush=True)
                exports.append((full[0], full[1], selected, destination / "charts"))
            elif selected:
                ocr_pages = {page.page_number for page in audit if page.route == "docling_ocr"}
                ranges = sorted(_ranges(selected - ocr_pages) + _ranges(ocr_pages))
                for first, last in ranges:
                    group = set(range(first, last + 1))
                    ocr = bool(group & ocr_pages)
                    directory = destination / "ranges" / f"p{first:03d}-{last:03d}"
                    md, js, meta = directory / "document.md", directory / "document.json", directory / "metadata.json"
                    expected = {"source_sha256": document.sha256, "pages": [first, last], "ocr": ocr}
                    stored = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
                    stale = any(stored.get(k) != v for k, v in expected.items())
                    if force or not md.is_file() or not js.is_file() or stale:
                        print(f"  Docling pages {first}-{last} (OCR={ocr})", flush=True)
                        md, js = runner.convert(document, directory, first, last, ocr=ocr)
                    exports.append((md, js, group, directory / "enhanced"))
            for md, js, group, chart_output in exports:
                parsed = json.loads(js.read_text(encoding="utf-8"))
                chart_pages = {page.page_number for page in audit if page.page_number in group
                               and page.route in {"docling_chart", "docling_ocr"}}
                fallback_pages = {page.page_number for page in audit if page.page_number in chart_pages
                                  and ("presentation" in document.document_type or page.route == "docling_ocr")}
                records = []
                if chart_pages:
                    _seed_trial_cache(document, chart_output, workspace)
                    records, enhanced_md, enhanced_json, comparison = process_charts(
                        document.local_path, parsed, md.read_text(encoding="utf-8"), chart_output,
                        provider=provider, page_filter=chart_pages, fallback_page_filter=fallback_pages,
                        max_workers=max(1, chart_workers), force=force, reuse_any_success=True)
                    (chart_output / "document.enhanced.md").write_text(enhanced_md, encoding="utf-8")
                    write_json(chart_output / "document.enhanced.json", enhanced_json)
                    write_json(chart_output / "chart_output_comparison.json", comparison)
                ocr_pages = {page.page_number for page in audit if page.route == "docling_ocr"}
                extracted, limits = docling_blocks(document, parsed, records, group, ocr_pages=ocr_pages)
                blocks.extend(extracted)
                summary["chart_records"].extend(record.model_dump(mode="json") for record in records)
                summary["coverage_limitations"].extend(limits)
                summary.setdefault("docling_exports", []).append({"json": js.relative_to(workspace).as_posix(),
                                                                  "markdown": md.relative_to(workspace).as_posix(),
                                                                  "pages": sorted(group)})
                summary["chart_failures"] = [{"chart_id": item["source"]["chart_id"], "error": item.get("error")}
                                               for item in summary["chart_records"] if item["status"] != "success"]
                _write_jsonl(destination / "blocks.jsonl", (block.to_json() for block in blocks))
                write_json(destination / "audit.json", summary)
        except KeyboardInterrupt:
            summary["warnings"].append("Interrupted; completed chart caches are retained")
            report["documents"].append(summary)
            report["status"] = "interrupted"
            write_json(destination / "audit.json", summary)
            _checkpoint(root, report)
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print("  Extraction error: " + message, flush=True)
            summary["warnings"].append(message)
            summary["coverage_limitations"].extend({"document_id": document.document_id,
                                                    "page_number": page, "scope": "page", "reason": message}
                                                   for page in sorted(selected - {block.page for block in blocks}))
        summary["blocks_created"] = len(blocks)
        summary["pages_enriched"] = sorted({block.page for block in blocks})
        summary["content_types"] = dict(Counter(block.content_type for block in blocks))
        _write_jsonl(destination / "blocks.jsonl", (block.to_json() for block in blocks))
        write_json(destination / "audit.json", summary)
        report["documents"].append(summary)
        _checkpoint(root, report)
    _checkpoint(root, report, final=True)
    print(f"{ticker}: {report['status']}; {report['pages_audited']} pages; "
          f"{report['successful_charts']}/{report['chart_count']} charts", flush=True)
    return report


def promote_chart_chunks(workspace: Path, ticker: str, *, experiment: str,
                         approved_document_ids: set[str] | None = None) -> dict[str, Any]:
    """Persist source-hashed page replacements and rebuild canonical BM25."""
    from finsight.agents.narrative.processing.process_corpus import build_ticker_corpus

    workspace, ticker = workspace.resolve(), ticker.upper()
    root = _experiment_root(workspace, ticker, experiment)
    canonical = workspace / "data/narrative/processed" / ticker
    manifest_path = canonical / "pdf_enrichment.json"
    previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists()
                else {"version": ENRICHMENT_VERSION, "documents": {}})
    eligible = {doc.document_id: doc for doc in load_documents(ticker, root=workspace)}
    promoted = []
    previous_artifacts: dict[Path, bytes | None] = {}
    for audit_path in sorted(root.glob("*/audit.json")):
        summary = json.loads(audit_path.read_text(encoding="utf-8"))
        identity = summary["document_id"]
        if approved_document_ids is not None and identity not in approved_document_ids:
            continue
        doc = eligible.get(identity)
        if doc is None or summary.get("sha256") != doc.sha256:
            raise ValueError(f"Experiment source is no longer eligible/current: {identity}")
        validate_document(doc)
        artifact = audit_path.parent / summary.get("blocks_path", "blocks.jsonl")
        if not artifact.is_file() or (
            not artifact.read_text(encoding="utf-8").strip() and not summary.get("coverage_limitations")
        ):
            continue
        destination = canonical / "enrichments" / f"{identity}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous_artifacts[destination] = destination.read_bytes() if destination.exists() else None
        shutil.copy2(artifact, destination)
        coverage = list(summary.get("coverage_limitations", []))
        for export in summary.get("docling_exports", []):
            metadata_path = (workspace / export["json"]).parent / "metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                for message in dict.fromkeys(metadata.get("errors", []) + metadata.get("warnings", [])):
                    coverage.append({"document_id": identity, "scope": "range",
                                     "pages": export["pages"], "reason": message})
        manifest["documents"][identity] = {"source_sha256": doc.sha256,
            "blocks_path": destination.relative_to(canonical).as_posix(), "blocks_sha256": file_sha256(destination),
            "version": ENRICHMENT_VERSION, "experiment": experiment,
            "coverage_limitations": coverage}
        promoted.append(identity)
    if not promoted:
        return {"status": "no_new_blocks", "documents": []}
    manifest["version"] = ENRICHMENT_VERSION
    write_json(manifest_path, manifest)
    try:
        report = build_ticker_corpus(workspace, ticker, replace_existing=True, reuse_saved_blocks=True)
    except Exception:
        for artifact, previous in previous_artifacts.items():
            if previous is None:
                artifact.unlink(missing_ok=True)
            else:
                artifact.write_bytes(previous)
        if previous_manifest is None:
            manifest_path.unlink(missing_ok=True)
        else:
            manifest_path.write_bytes(previous_manifest)
        raise
    return {"status": "promoted", "documents": promoted, "indexed_chunks": report["indexed_chunks"],
            "corpus_fingerprint": report["corpus_fingerprint"], "qdrant_rebuild_required": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ticker", action="append")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--experiment", default="pdf_chart_processing")
    parser.add_argument("--pages")
    parser.add_argument("--document-id", action="append")
    for flag in ("audit-only", "promote-only", "promote", "rebuild-vectors", "force", "status"):
        parser.add_argument("--" + flag, action="store_true")
    parser.add_argument("--usage", action="store_true",
                        help="Show locally tracked OpenAI usage across runs for the selected companies; no API calls.")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--ollama-model")
    models.add_argument("--openai-model")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--ollama-context", type=int, default=8192)
    parser.add_argument("--chart-workers", type=int, default=1)
    parser.add_argument("--chart-timeout", type=float, default=300)
    parser.add_argument("--chart-max-output-tokens", type=int)
    args = parser.parse_args()
    if args.audit_only and (args.promote or args.promote_only or args.rebuild_vectors):
        parser.error("--audit-only cannot promote or rebuild vectors")
    if args.chart_workers < 1:
        parser.error("--chart-workers must be positive")
    if args.chart_max_output_tokens is not None and args.chart_max_output_tokens < 1:
        parser.error("--chart-max-output-tokens must be positive")
    logger = logging.getLogger("finsight.docling_charts")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    from finsight.agents.narrative.processing.process_corpus import discover_tickers
    workspace = args.workspace.resolve()
    tickers = discover_tickers(workspace) if args.all else sorted({ticker.upper() for ticker in args.ticker})
    if args.usage:
        from finsight.agents.narrative.pdf.chart_usage import read_usage
        print(json.dumps(read_usage(workspace, args.experiment, set(tickers)), indent=2), flush=True)
        return
    if args.status:
        status = {}
        for ticker in tickers:
            path = _experiment_root(workspace, ticker, args.experiment) / "experiment_report.json"
            if not path.is_file():
                status[ticker] = {"status": "not_run"}
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            status[ticker] = {key: report.get(key) for key in
                             ("status", "documents_completed", "documents_selected", "pages_audited",
                              "successful_charts", "chart_count")}
            status[ticker]["updated_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        print(json.dumps(status, indent=2), flush=True)
        return
    provider = None
    usage_tracker = None
    if args.ollama_model:
        from finsight.agents.narrative.pdf.chart_vision import OllamaChartVisionProvider
        provider = OllamaChartVisionProvider(args.ollama_model, base_url=args.base_url,
                                             context_size=args.ollama_context, timeout_seconds=args.chart_timeout,
                                             max_output_tokens=args.chart_max_output_tokens or 4000)
    elif args.openai_model:
        from finsight.agents.narrative.pdf.chart_vision import OpenAIChartVisionProvider
        provider = OpenAIChartVisionProvider(args.openai_model, timeout_seconds=args.chart_timeout,
                                             max_output_tokens=args.chart_max_output_tokens or 6000)
        from finsight.agents.narrative.pdf.chart_usage import OpenAIChartUsage
        usage_tracker = OpenAIChartUsage(workspace, args.experiment, args.openai_model)
        provider.usage_tracker = usage_tracker
        print(f"OpenAI usage report: {usage_tracker.directory / 'summary.json'}", flush=True)
    summaries, promoted_tickers, runner = {}, [], DoclingRunner()
    run_status = "interrupted_or_failed"
    try:
        for ticker in tickers:
            if usage_tracker is not None:
                usage_tracker.ticker = ticker
            summary = {}
            if not args.promote_only:
                report = process_pdf_experiment(workspace, ticker, experiment=args.experiment, provider=provider,
                    pages=parse_page_spec(args.pages), chart_workers=args.chart_workers, audit_only=args.audit_only,
                    document_ids=set(args.document_id) if args.document_id else None, runner=runner, force=args.force)
                summary = {key: report[key] for key in
                           ("status", "documents_selected", "pages_audited", "successful_charts", "chart_count")}
            if args.promote or args.promote_only:
                print(f"[{ticker}] Promoting PDF blocks and rebuilding BM25", flush=True)
                summary["promotion"] = promote_chart_chunks(workspace, ticker, experiment=args.experiment,
                    approved_document_ids=set(args.document_id) if args.document_id else None)
                print(f"[{ticker}] Promotion: {summary['promotion']['status']}", flush=True)
                if summary["promotion"]["status"] == "promoted":
                    promoted_tickers.append(ticker)
            summaries[ticker] = summary
        if args.rebuild_vectors and promoted_tickers:
            from finsight.agents.narrative.retrieval.build_vector_databases import build_vector_databases
            print("Synchronizing Qdrant indexes: " + ", ".join(promoted_tickers), flush=True)
            summaries["vectors"] = build_vector_databases(workspace, promoted_tickers)
        run_status = "partial" if any(row.get("status") == "partial" for row in summaries.values()) else "finished"
        print(json.dumps(summaries, indent=2), flush=True)
    finally:
        if usage_tracker is not None:
            usage = usage_tracker.finish(run_status)
            qualifier = "" if usage["cost_estimate_complete"] else " (partial; some request costs unknown)"
            print(f"OpenAI usage (this run): {usage['api_requests']} requests; "
                  f"{usage['input_tokens']} input tokens ({usage['cached_input_tokens']} cached); "
                  f"{usage['output_tokens']} output tokens; "
                  f"estimated cost USD ${usage['estimated_cost_usd']:.6f}{qualifier}", flush=True)
            print(f"Usage saved: {usage_tracker.directory / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
