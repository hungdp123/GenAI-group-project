"""Build one company's contextual BM25 index for Workflow B.

The canonical company index is left in place. Workflow C selects this build
with ``--corpus-root``.

Example, from the repository root:

    python -m scripts.narrative.chunking.build_contextual_index --ticker AMZN
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finsight.agents.narrative.corpus.processing.contextual import (
    EMBEDDING_MODEL,
    TOKEN_LIMIT,
    write_contextual_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/narrative/processed"))
    parser.add_argument("--build-version", default="bge512_v1")
    parser.add_argument("--token-limit", type=int, default=TOKEN_LIMIT)
    parser.add_argument(
        "--embed",
        action="store_true",
        help=f"Also build Qdrant vectors with {EMBEDDING_MODEL}. Requires the retrieval extra.",
    )
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    source = args.processed_root / ticker / "chunks.jsonl"
    if not source.is_file():
        parser.exit(2, f"Canonical chunks are missing: {source}\n")
    chunks = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = args.processed_root / "chunking_strategies" / "contextual" / args.build_version / ticker
    payload = write_contextual_index(chunks, output, token_limit=args.token_limit)
    print(
        json.dumps(
            {
                "ticker": ticker,
                "output": str(output),
                "parent_chunks": payload["parent_chunk_count"],
                "contextual_chunks": payload["chunk_count"],
                "embedding_model": EMBEDDING_MODEL,
                "token_limit": args.token_limit,
                "vectors": "pending" if args.embed else "not_built",
            },
            indent=2,
        )
    )
    if not args.embed:
        return
    from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index
    from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend

    metadata = build_qdrant_index(
        payload,
        output / "indexes" / "qdrant",
        SentenceTransformersBackend(EMBEDDING_MODEL),
    )
    print(json.dumps({"vectors": metadata.get("vector_count"), "model": metadata.get("embedding_model")}, indent=2))


if __name__ == "__main__":
    main()
