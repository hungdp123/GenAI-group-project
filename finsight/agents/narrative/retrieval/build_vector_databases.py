"""Build and validate one persistent Qdrant vector database per eligible ticker.

Examples:
    python -m finsight.agents.narrative.retrieval.build_vector_databases --all
    python -m finsight.agents.narrative.retrieval.build_vector_databases --ticker NVDA --rebuild
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from finsight.agents.narrative.corpus.processing.corpus import corpus_fingerprint, load_documents
from finsight.agents.narrative.corpus.pdf.pdf_enrichment import effective_corpus_fingerprint, load_promoted_enrichment
from finsight.agents.narrative.corpus.processing.index import load_index
from finsight.agents.narrative.corpus.processing.process_corpus import discover_tickers
from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index, load_qdrant_index
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformersBackend,
)


def _load_company_index(workspace: Path, ticker: str) -> dict[str, Any]:
    index_dir = workspace / "data/narrative/processed" / ticker / "indexes"
    index_payload = load_index(index_dir / "index.json")
    index_metadata = json.loads((index_dir / "index_metadata.json").read_text(encoding="utf-8"))
    documents = load_documents(ticker, root=workspace)
    if not documents:
        raise ValueError(f"No retrieval-eligible manifest documents found for {ticker}")

    _, enrichment = load_promoted_enrichment(workspace, ticker, documents)
    expected_fingerprint = effective_corpus_fingerprint(documents, enrichment)
    if index_payload.get("source_corpus_fingerprint", corpus_fingerprint(documents)) != corpus_fingerprint(documents):
        raise ValueError(f"Source manifest fingerprint is stale for {ticker}")
    if index_payload.get("corpus_fingerprint") != expected_fingerprint:
        raise ValueError(f"BM25 corpus fingerprint is stale for {ticker}; rebuild processed outputs first")
    if index_metadata.get("corpus_fingerprint") != expected_fingerprint:
        raise ValueError(f"BM25 metadata fingerprint is stale for {ticker}")
    if index_metadata.get("schema_version") != index_payload.get("schema_version"):
        raise ValueError(f"BM25 schema metadata is inconsistent for {ticker}")
    if str(index_metadata.get("company_ticker", "")).upper() != ticker:
        raise ValueError(f"BM25 index metadata belongs to another company, not {ticker}")

    expected_document_ids = sorted(document.document_id for document in documents)
    if sorted(index_metadata.get("document_ids", [])) != expected_document_ids:
        raise ValueError(f"BM25 document IDs do not match the current manifest for {ticker}")
    chunks = index_payload.get("chunks", [])
    if index_payload.get("chunk_count") != len(chunks):
        raise ValueError(f"BM25 chunk count metadata is inconsistent for {ticker}")
    if any(str(chunk.get("company_ticker", "")).upper() != ticker for chunk in chunks):
        raise ValueError(f"BM25 index contains chunks from another company: {ticker}")
    return index_payload


def _validate_qdrant(
    output_dir: Path,
    index_payload: dict[str, Any],
    *,
    model_name: str,
    dimension: int,
) -> dict[str, Any]:
    index = load_qdrant_index(
        output_dir,
        index_payload,
        expected_model_name=model_name,
        expected_dimension=dimension,
    )
    try:
        if index.dimension != dimension:
            raise ValueError(
                f"Stored vector dimension {index.dimension} does not match configured dimension {dimension}"
            )
        return json.loads((output_dir / "qdrant_metadata.json").read_text(encoding="utf-8"))
    finally:
        index.client.close()


def _build_company(
    workspace: Path,
    ticker: str,
    backend: SentenceTransformersBackend,
    dimension: int,
    *,
    rebuild: bool,
) -> dict[str, Any]:
    index_payload = _load_company_index(workspace, ticker)
    indexes_dir = workspace / "data/narrative/processed" / ticker / "indexes"
    target = indexes_dir / "qdrant"

    if target.exists() and not rebuild:
        try:
            metadata = _validate_qdrant(
                target,
                index_payload,
                model_name=backend.model_name,
                dimension=dimension,
            )
            return {
                "status": "current",
                "documents": len(load_documents(ticker, root=workspace)),
                "chunks": len(index_payload["chunks"]),
                "vectors": metadata["vector_count"],
                "dimension": metadata["dimension"],
                "corpus_fingerprint": metadata["corpus_fingerprint"],
            }
        except (OSError, ValueError, KeyError, RuntimeError):
            # Incomplete or stale databases are replaced after a staged rebuild succeeds.
            pass

    build_id = uuid4().hex
    staging = indexes_dir / f".qdrant.{build_id}.building"
    backup = indexes_dir / f".qdrant.{build_id}.backup"
    processed_root = (workspace / "data/narrative/processed").resolve()
    if not indexes_dir.resolve().is_relative_to(processed_root):
        raise ValueError("Vector index directory must remain within the processed corpus root")
    if staging.resolve().parent != indexes_dir.resolve() or backup.resolve().parent != indexes_dir.resolve():
        raise ValueError("Vector staging and backup directories must remain within the company indexes directory")
    indexes_dir.mkdir(parents=True, exist_ok=True)
    try:
        metadata = build_qdrant_index(index_payload, staging, backend)
        if metadata["embedding_model"] != backend.model_name:
            raise ValueError(f"Builder stored an unexpected embedding model for {ticker}")
        if int(metadata["dimension"]) != dimension:
            raise ValueError(
                f"Model produced {metadata['dimension']} dimensions for {ticker}; expected {dimension}"
            )
        if int(metadata["vector_count"]) != len(index_payload["chunks"]):
            raise ValueError(f"Qdrant vector count does not match BM25 chunk count for {ticker}")
        _validate_qdrant(
            staging,
            index_payload,
            model_name=backend.model_name,
            dimension=dimension,
        )

        if target.exists():
            target.rename(backup)
        try:
            staging.rename(target)
        except Exception:
            if backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        return {
            "status": "built",
            "documents": len(load_documents(ticker, root=workspace)),
            "chunks": len(index_payload["chunks"]),
            "vectors": metadata["vector_count"],
            "dimension": metadata["dimension"],
            "corpus_fingerprint": metadata["corpus_fingerprint"],
        }
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def build_vector_databases(
    workspace: Path,
    tickers: list[str],
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build canonical Qdrant indexes while reusing one embedding model instance."""
    backend = SentenceTransformersBackend(embedding_model)
    if backend.model_name != embedding_model:
        raise ValueError(
            f"Loaded model {backend.model_name!r} does not match configured model {embedding_model!r}"
        )
    dimension_reader = getattr(backend.model, "get_embedding_dimension", None)
    if dimension_reader is None:
        dimension_reader = backend.model.get_sentence_embedding_dimension
    model_dimension = dimension_reader()
    if model_dimension is None or int(model_dimension) < 1:
        raise ValueError(f"Embedding model has no valid dimension: {embedding_model}")
    dimension = int(model_dimension)

    summary: dict[str, Any] = {
        "embedding_model": backend.model_name,
        "dimension": dimension,
        "distance": "cosine",
        "built": {},
        "current": {},
        "failed": {},
    }
    workspace = workspace.resolve()
    for ticker in sorted({ticker.upper() for ticker in tickers}):
        try:
            result = _build_company(
                workspace,
                ticker,
                backend,
                dimension,
                rebuild=rebuild,
            )
            bucket = "built" if result["status"] == "built" else "current"
            summary[bucket][ticker] = result
        except Exception as exc:  # Continue through the batch and report every failed company.
            summary["failed"][ticker] = f"{type(exc).__name__}: {exc}"
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Build for every eligible ticker.")
    selection.add_argument("--ticker", action="append", help="Ticker to build; repeat to select several.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild otherwise-current vector indexes.")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    tickers = discover_tickers(workspace) if args.all else sorted(set(args.ticker))
    summary = build_vector_databases(
        workspace,
        tickers,
        embedding_model=args.embedding_model,
        rebuild=args.rebuild,
    )
    print(json.dumps(summary, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
