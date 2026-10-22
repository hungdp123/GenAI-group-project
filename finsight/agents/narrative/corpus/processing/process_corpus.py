"""Build processed narrative corpora and BM25 indexes for one or more tickers.

Examples:
    python -m finsight.agents.narrative.corpus.processing.process_corpus --all
    python -m finsight.agents.narrative.corpus.processing.process_corpus --ticker AAPL --ticker MSFT
    python -m finsight.agents.narrative.corpus.processing.process_corpus --ticker NVDA --rebuild
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from finsight.agents.narrative.corpus.processing.chunk import chunk_blocks
from finsight.agents.narrative.corpus.processing.corpus import (
    CANONICAL_MANIFESTS,
    CorpusDocument,
    corpus_fingerprint,
    load_documents,
    validate_document,
)
from finsight.agents.narrative.corpus.processing.index import build_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock, _base, parse_document
from finsight.agents.narrative.corpus.pdf.pdf_enrichment import effective_corpus_fingerprint, load_promoted_enrichment


TARGET_WORDS = 800
OVERLAP_WORDS = 120


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def discover_tickers(
    workspace: Path,
    manifests: Iterable[Path] = CANONICAL_MANIFESTS,
) -> list[str]:
    """Return tickers having at least one retrieval-eligible manifest record."""
    tickers: set[str] = set()
    for manifest in manifests:
        manifest_path = manifest if manifest.is_absolute() else workspace / manifest
        if manifest_path.name == "manifest_third_party_transcripts.json" and not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("documents", []):
            ticker = str(item.get("company_ticker", "")).upper()
            if ticker and item.get("retrieval_eligible") is not False:
                tickers.add(ticker)
    return sorted(tickers)


def _saved_document_blocks(base: Path, documents: list[CorpusDocument],
                           enriched: dict[str, list[ProcessedBlock]]
                           ) -> dict[str, tuple[list[ProcessedBlock], list[str]]]:
    """Reuse source-matched saved extraction when promoting new page replacements.

    Previously enriched pages must still be replaced in the new build. Otherwise
    parse the original source again so removed enrichment cannot leak into output.
    Normal rebuilds continue to extract sources afresh.
    """
    report_path, blocks_path = base / "processing_report.json", base / "documents.jsonl"
    if not report_path.is_file() or not blocks_path.is_file():
        return {}
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    stats = {item["document_id"]: item for item in previous.get("documents", [])}
    reusable = {}
    for document in documents:
        item = stats.get(document.document_id, {})
        replacement_pages = {block.page for block in enriched.get(document.document_id, [])}
        if (item.get("sha256") == document.sha256
                and set(item.get("enriched_pages", [])) <= replacement_pages):
            reusable[document.document_id] = document
    grouped: dict[str, list[ProcessedBlock]] = {identity: [] for identity in reusable}
    for line in blocks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        identity = row["document_id"]
        if identity in grouped:
            grouped[identity].append(ProcessedBlock(**row))
    cached = {}
    for identity, blocks in grouped.items():
        if len(blocks) != stats[identity].get("processed_blocks"):
            continue
        metadata = {key: value for key, value in _base(reusable[identity]).items()
                    if key not in {"extraction_method", "source_format"}}
        cached[identity] = ([replace(block, **metadata) for block in blocks],
                            [warning for warning in previous.get("warnings", [])
                             if warning.startswith(identity + ":")])
    return cached


def _process_documents(documents: list[CorpusDocument], enriched: dict[str, list[ProcessedBlock]] | None = None,
                       cached: dict[str, tuple[list[ProcessedBlock], list[str]]] | None = None
                       ) -> tuple[list[ProcessedBlock], list[str], list[dict[str, Any]]]:
    blocks: list[ProcessedBlock] = []
    warnings: list[str] = []
    document_stats: list[dict[str, Any]] = []
    for number, document in enumerate(documents, 1):
        validate_document(document)
        saved = (cached or {}).get(document.document_id)
        print(f"[{document.company_ticker} {number}/{len(documents)}] "
              f"{'Reusing source-matched blocks' if saved is not None else 'Extracting'}: "
              f"{document.local_path.name}", flush=True)
        parsed, parse_warnings = saved if saved is not None else parse_document(document)
        replacements = (enriched or {}).get(document.document_id, [])
        replaced_pages = {block.page for block in replacements}
        if replacements:
            parsed = [block for block in parsed if block.page not in replaced_pages] + replacements
            parse_warnings = [warning for warning in parse_warnings
                              if not any(f"PDF page {page} has no extractable text" in warning
                                         for page in replaced_pages)]
        blocks.extend(parsed)
        warnings.extend(parse_warnings)
        document_stats.append(
            {
                "document_id": document.document_id,
                "document_type": document.document_type,
                "publication_date": document.publication_date.isoformat(),
                "local_path": document.local_path.as_posix(),
                "source_url": document.source_url,
                "sha256": document.sha256,
                "processed_blocks": len(parsed),
                "extracted_words": sum(len(block.text.split()) for block in parsed),
                "enriched_pages": sorted(page for page in replaced_pages if page is not None),
            }
        )
    return blocks, warnings, document_stats


def build_ticker_corpus(
    workspace: Path,
    ticker: str,
    *,
    output_name: str | None = None,
    manifests: Iterable[Path] = CANONICAL_MANIFESTS,
    replace_existing: bool = False,
    reuse_saved_blocks: bool = False,
) -> dict[str, Any]:
    """Build one ticker using staged files and rollback, preserving experiments and enrichments."""
    workspace = workspace.resolve()
    ticker = ticker.upper()
    output_name = output_name or ticker
    if output_name != ticker:
        raise ValueError(
            f"Non-canonical output name {output_name!r}; use the company ticker {ticker!r}"
        )
    documents = load_documents(ticker, manifests=manifests, root=workspace)
    if not documents:
        raise ValueError(f"No retrieval-eligible documents found for {ticker}")

    processed_root = workspace / "data/narrative/processed"
    processed_target = processed_root / ticker
    if processed_target.exists() and not replace_existing:
        raise FileExistsError(f"Output already exists for {ticker}: {processed_target}")

    build_id = uuid4().hex
    processed_build = processed_root / f".{ticker}.{build_id}.building"
    if not processed_build.resolve().is_relative_to(processed_root.resolve()):
        raise ValueError("Generated build directory must stay within the processed corpus root")
    index_build = processed_build / "indexes"
    try:
        enriched, enrichment_metadata = load_promoted_enrichment(workspace, ticker, documents)
        cached = _saved_document_blocks(processed_target, documents, enriched) if reuse_saved_blocks else None
        blocks, warnings, document_stats = _process_documents(documents, enriched, cached)
        chunks = chunk_blocks(blocks, TARGET_WORDS, OVERLAP_WORDS)
        _write_jsonl(processed_build / "documents.jsonl", (block.to_json() for block in blocks))
        _write_jsonl(processed_build / "chunks.jsonl", (chunk.to_json() for chunk in chunks))

        fingerprint = effective_corpus_fingerprint(documents, enrichment_metadata)
        index_payload = build_index(
            chunks,
            index_build / "index.json",
            corpus_fingerprint=fingerprint,
            target_words=TARGET_WORDS,
            overlap_words=OVERLAP_WORDS,
        )
        index_payload["source_corpus_fingerprint"] = corpus_fingerprint(documents)
        index_payload["pdf_processing"] = enrichment_metadata
        index_payload["coverage_limitations"] = enrichment_metadata.get("coverage_limitations", [])
        (index_build / "index.json").write_text(
            json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        index_metadata = {
            key: index_payload[key]
            for key in (
                "schema_version",
                "corpus_fingerprint",
                "chunking",
                "chunk_count",
                "duplicates_excluded",
            )
        }
        index_metadata["company_ticker"] = ticker
        index_metadata["document_ids"] = sorted(document.document_id for document in documents)
        index_metadata["source_corpus_fingerprint"] = corpus_fingerprint(documents)
        index_metadata["pdf_processing"] = enrichment_metadata
        (index_build / "index_metadata.json").write_text(
            json.dumps(index_metadata, indent=2), encoding="utf-8"
        )

        report = {
            "status": "complete",
            "scope": f"{ticker} full eligible corpus",
            "company_ticker": ticker,
            "corpus_fingerprint": fingerprint,
            "source_corpus_fingerprint": corpus_fingerprint(documents),
            "pdf_processing": enrichment_metadata,
            "coverage_limitations": enrichment_metadata.get("coverage_limitations", []),
            "documents_selected": len(documents),
            "processed_blocks": len(blocks),
            "chunks_created": len(chunks),
            "indexed_chunks": index_payload["chunk_count"],
            "duplicates_excluded": index_payload["duplicates_excluded"],
            "warnings": warnings,
            "documents": document_stats,
        }
        (processed_build / "processing_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        processed_target.mkdir(parents=True, exist_ok=True)
        backup_target = processed_build / ".previous"
        generated = ("documents.jsonl", "chunks.jsonl", "indexes/index_metadata.json",
                     "indexes/index.json", "processing_report.json")
        published: list[str] = []
        try:
            for name in generated:
                target = processed_target / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    previous = backup_target / name
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, previous)
                (processed_build / name).replace(target)
                published.append(name)
        except Exception:
            for name in reversed(published):
                previous = backup_target / name
                if previous.exists():
                    previous.replace(processed_target / name)
                else:
                    (processed_target / name).unlink(missing_ok=True)
            raise
        shutil.rmtree(processed_build)
        return report
    except Exception:
        shutil.rmtree(processed_build, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Process every eligible ticker.")
    selection.add_argument(
        "--ticker",
        action="append",
        help="Ticker to process; repeat this option to process multiple tickers.",
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Atomically replace existing generated outputs for the selected ticker(s).",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    tickers = discover_tickers(workspace) if args.all else sorted(set(args.ticker))
    summary: dict[str, Any] = {"processed": {}, "skipped": {}, "failed": {}}
    for ticker in tickers:
        processed_path = workspace / "data/narrative/processed" / ticker
        if processed_path.exists() and not args.rebuild:
            summary["skipped"][ticker] = "output already exists"
            continue
        try:
            report = build_ticker_corpus(workspace, ticker, replace_existing=args.rebuild)
            summary["processed"][ticker] = {
                "documents": report["documents_selected"],
                "blocks": report["processed_blocks"],
                "chunks": report["chunks_created"],
                "indexed_chunks": report["indexed_chunks"],
                "warnings": len(report["warnings"]),
            }
        except Exception as exc:  # Continue so one bad ticker does not block the batch.
            summary["failed"][ticker] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
