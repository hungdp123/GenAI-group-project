"""Build and query the two-document NVDA narrative vertical slice.

Run with:
    python -m finsight.agents.narrative.corpus.maintenance.pilot
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from finsight.agents.narrative.corpus.processing.chunk import chunk_blocks
from finsight.agents.narrative.corpus.processing.corpus import (
    corpus_fingerprint,
    load_documents,
    validate_document,
)
from finsight.agents.narrative.corpus.processing.index import build_index, load_index, search_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock, parse_document


PILOT_DOCUMENT_IDS = {
    "NVDA-0001045810-25-000023-10-K",
    "NVDA-IR-0387401647e4fe23",
}
DEFAULT_QUERY = "NVIDIA data center growth accelerated computing Blackwell inference customer demand"
DEFAULT_CUTOFF = date(2025, 7, 31)
TARGET_WORDS = 800
OVERLAP_WORDS = 120


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_pilot(workspace: Path, *, all_documents: bool = False) -> dict[str, Any]:
    all_nvda = load_documents("NVDA", root=workspace)
    selected = (
        all_nvda if all_documents else [document for document in all_nvda if document.document_id in PILOT_DOCUMENT_IDS]
    )
    selected_ids = {document.document_id for document in selected}
    missing_ids = set() if all_documents else PILOT_DOCUMENT_IDS - selected_ids
    if missing_ids:
        raise ValueError(f"Pilot document IDs missing from canonical manifests: {sorted(missing_ids)}")

    blocks: list[ProcessedBlock] = []
    warnings: list[str] = []
    document_stats: list[dict[str, Any]] = []
    for document in selected:
        validate_document(document)
        parsed, parse_warnings = parse_document(document)
        blocks.extend(parsed)
        warnings.extend(parse_warnings)
        document_stats.append(
            {
                "document_id": document.document_id,
                "document_type": document.document_type,
                "publication_date": document.publication_date.isoformat(),
                "local_path": document.local_path.relative_to(workspace).as_posix(),
                "source_url": document.source_url,
                "sha256": document.sha256,
                "processed_blocks": len(parsed),
                "extracted_words": sum(len(block.text.split()) for block in parsed),
            }
        )

    chunks = chunk_blocks(blocks, TARGET_WORDS, OVERLAP_WORDS)
    processed_dir = (
        workspace / "data/narrative/processed/NVDA"
        if all_documents
        else workspace / "data/narrative/experiments/NVDA_pilot"
    )
    index_dir = processed_dir / "indexes"
    _write_jsonl(processed_dir / "documents.jsonl", (block.to_json() for block in blocks))
    _write_jsonl(processed_dir / "chunks.jsonl", (chunk.to_json() for chunk in chunks))

    fingerprint = corpus_fingerprint(selected)
    index_payload = build_index(
        chunks,
        index_dir / "index.json",
        corpus_fingerprint=fingerprint,
        target_words=TARGET_WORDS,
        overlap_words=OVERLAP_WORDS,
    )
    index_metadata = {
        key: index_payload[key]
        for key in ("schema_version", "corpus_fingerprint", "chunking", "chunk_count", "duplicates_excluded")
    }
    index_metadata["company_ticker"] = "NVDA"
    index_metadata["document_ids"] = sorted(selected_ids)
    (index_dir / "index_metadata.json").write_text(json.dumps(index_metadata, indent=2), encoding="utf-8")

    report = {
        "status": "complete",
        "scope": f"NVDA {'full canonical corpus' if all_documents else 'two-document vertical slice'}",
        "corpus_fingerprint": fingerprint,
        "documents_selected": len(selected),
        "processed_blocks": len(blocks),
        "chunks_created": len(chunks),
        "indexed_chunks": index_payload["chunk_count"],
        "duplicates_excluded": index_payload["duplicates_excluded"],
        "warnings": warnings,
        "documents": document_stats,
    }
    (processed_dir / "processing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def query_pilot(
    workspace: Path, query: str, cutoff: date, top_k: int, *, all_documents: bool = False
) -> list[dict[str, Any]]:
    processed_dir = (
        workspace / "data/narrative/processed/NVDA"
        if all_documents
        else workspace / "data/narrative/experiments/NVDA_pilot"
    )
    index_path = processed_dir / "indexes/index.json"
    results = search_index(load_index(index_path), query, top_k, cutoff, "NVDA")
    output = {
        "query": query,
        "company_ticker": "NVDA",
        "cutoff_date": cutoff.isoformat(),
        "top_k": top_k,
        "results": results,
    }
    output_path = processed_dir / "query_results.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--cutoff", type=date.fromisoformat, default=DEFAULT_CUTOFF)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--all-documents",
        action="store_true",
        help=(
            "Process all eligible NVDA records into the canonical processed/NVDA corpus; "
            "the default pilot writes to experiments/NVDA_pilot."
        ),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    report = build_pilot(workspace, all_documents=args.all_documents)
    results = query_pilot(workspace, args.query, args.cutoff, args.top_k, all_documents=args.all_documents)
    print(json.dumps({"report": report, "query_results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
