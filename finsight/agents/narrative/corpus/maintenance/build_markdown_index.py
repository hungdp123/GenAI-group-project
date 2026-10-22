"""Build combined BM25 + Qdrant indexes from enhanced Markdown and an existing corpus.

The enhanced Markdown inherits company, publication, source-tier, URL, document-type,
and period metadata from the original PDF's canonical manifest record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from finsight.agents.narrative.corpus.processing.chunk import NarrativeChunk, chunk_blocks
from finsight.agents.narrative.corpus.processing.corpus import file_sha256, load_documents, validate_document
from finsight.agents.narrative.corpus.processing.index import build_index, load_index
from finsight.agents.narrative.corpus.processing.ingest import parse_document
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformersBackend,
)
from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index


TARGET_WORDS = 500
OVERLAP_WORDS = 75


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _upgrade_existing_chunk(row: dict[str, Any]) -> NarrativeChunk:
    value = dict(row)
    suffix = Path(str(value.get("local_path", ""))).suffix.lower()
    source_format = "html" if suffix in {".htm", ".html"} else "pdf" if suffix == ".pdf" else "unknown"
    value.setdefault("source_format", source_format)
    value.setdefault(
        "extraction_method",
        "native_html" if source_format == "html" else "native_pdf_text" if source_format == "pdf" else "unknown",
    )
    value.setdefault("content_type", "narrative")
    value.setdefault("chart_id", None)
    value.setdefault("validation_status", "native_text")
    value.setdefault("source_ref", None)
    value.setdefault("original_local_path", value.get("local_path"))
    return NarrativeChunk(**value)


def build_markdown_index(
    *,
    workspace: Path,
    markdown_path: Path,
    ticker: str,
    document_id: str,
    output_dir: Path,
    base_index_path: Path | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    markdown_path = markdown_path.resolve()
    output_dir = output_dir.resolve()
    if (output_dir / "index.json").exists():
        raise FileExistsError(f"Index already exists: {output_dir / 'index.json'}")

    source_documents = load_documents(ticker, root=workspace)
    source = next((document for document in source_documents if document.document_id == document_id), None)
    if source is None:
        raise ValueError(f"Document {document_id!r} was not found in canonical manifests for {ticker}")

    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    copied_markdown = source_dir / markdown_path.name
    shutil.copy2(markdown_path, copied_markdown)
    derived = replace(
        source,
        local_path=copied_markdown,
        sha256=file_sha256(copied_markdown),
        original_local_path=source.local_path,
    )
    validate_document(derived)
    blocks, warnings = parse_document(derived)
    markdown_chunks = chunk_blocks(blocks, TARGET_WORDS, OVERLAP_WORDS)

    existing_chunks: list[NarrativeChunk] = []
    base_fingerprint = "none"
    if base_index_path is not None:
        base = load_index(base_index_path)
        base_fingerprint = base["corpus_fingerprint"]
        existing_chunks = [_upgrade_existing_chunk(row) for row in base["chunks"]]

    combined_chunks = existing_chunks + markdown_chunks
    combined_fingerprint = hashlib.sha256(
        f"{base_fingerprint}:{derived.document_id}:{derived.sha256}".encode("utf-8")
    ).hexdigest()
    index_payload = build_index(
        combined_chunks,
        output_dir / "index.json",
        corpus_fingerprint=combined_fingerprint,
        target_words=TARGET_WORDS,
        overlap_words=OVERLAP_WORDS,
    )
    embeddings = SentenceTransformersBackend(embedding_model)
    vector_metadata = build_qdrant_index(index_payload, output_dir / "qdrant", embeddings)

    _write_jsonl(output_dir / "markdown_blocks.jsonl", (block.to_json() for block in blocks))
    _write_jsonl(output_dir / "markdown_chunks.jsonl", (chunk.to_json() for chunk in markdown_chunks))
    report = {
        "status": "complete",
        "company_ticker": ticker.upper(),
        "document_id": document_id,
        "source_pdf": source.local_path.as_posix(),
        "enhanced_markdown": copied_markdown.as_posix(),
        "source_url": source.source_url,
        "document_type": source.document_type,
        "publication_date": source.publication_date.isoformat(),
        "source_tier": source.source_tier,
        "embedding_model": embedding_model,
        "markdown_blocks": len(blocks),
        "markdown_chunks": len(markdown_chunks),
        "chart_chunks": sum(chunk.content_type == "chart" for chunk in markdown_chunks),
        "table_chunks": sum(chunk.content_type == "table" for chunk in markdown_chunks),
        "base_chunks": len(existing_chunks),
        "combined_index_chunks": index_payload["chunk_count"],
        "vector_store": "qdrant_local",
        "qdrant_vectors": vector_metadata["vector_count"],
        "warnings": warnings,
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--base-index", type=Path)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    report = build_markdown_index(
        workspace=args.workspace,
        markdown_path=args.markdown,
        ticker=args.ticker,
        document_id=args.document_id,
        output_dir=args.output,
        base_index_path=args.base_index,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

