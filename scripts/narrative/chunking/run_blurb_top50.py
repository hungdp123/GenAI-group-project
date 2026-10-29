"""Blurb the contextual passages retrieved for the v2 claims.

Collect the top 50 BM25 and dense passages for the supported and contradicting
claims, prepend an LLM sentence in front of the existing header, and rebuild
those vectors. A sentence already written for the same passage is reused.
Score the result with run_rag_comparison, not this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from finsight.agents.narrative.corpus.processing.chunk_context import (
    BLURB_SYSTEM,
    prepend_blurb,
    write_blurb,
)
from finsight.agents.narrative.corpus.processing.index import load_index
from finsight.agents.narrative.rag.collect_evidence import build_queries
from finsight.agents.narrative.retrieval.qdrant_index import QDRANT_COLLECTION, _qdrant_modules
from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend
from finsight.contracts.schemas import SubClaim, SubClaimType
from finsight.llm.client import complete

SOURCE = Path("data/narrative/processed/chunking_strategies/contextual/row512_v1")
DEST = Path("data/narrative/processed/chunking_strategies/contextual/blurb_v2_top50")
CASES = Path("data/narrative/evaluation/test_claims_v2/cases.csv")
PROGRESS = Path("data/narrative/experiments/context_blurbs/progress.jsonl")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_done() -> dict[str, str]:
    done = {}
    if PROGRESS.is_file():
        for line in PROGRESS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["chunk_id"]] = row["context_blurb"]
    return done


def ask(prompt: str) -> str:
    delay = 2.0
    last_error: Exception | None = None
    for _ in range(4):
        try:
            return complete(system=BLURB_SYSTEM, prompt=prompt, max_tokens=80).text
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "429" in message and "per_day" in message:
                raise
            time.sleep(delay)
            delay *= 2
    raise last_error or RuntimeError("blurb request failed")


def labeled_cases() -> list[dict]:
    cases = list(csv.DictReader(CASES.open(encoding="utf-8-sig")))
    id_key = next(key for key in cases[0] if "id" in key)
    for case in cases:
        case["id"] = case[id_key]
    return [case for case in cases if case["type"] != "unsupported"]


def collect(embedder) -> dict[str, set[str]]:
    chosen: dict[str, set[str]] = {}
    cache = {}
    for case in labeled_cases():
        ticker = case["company"]
        claim = SubClaim(
            sub_claim_id=case["id"], text=case["claims"], company_ticker=ticker, sub_claim_type=SubClaimType.OTHER
        )
        cutoff = date.fromisoformat(case["cut_off"])
        support, counter = build_queries(claim)
        for kind in ("bm25", "dense"):
            key = (kind, ticker)
            if key not in cache:
                cache[key] = CorpusRetriever(
                    company_ticker=ticker, corpus_root=SOURCE, backend=kind, candidate_k=50,
                    embedding_backend=embedder if kind == "dense" else None,
                )
            retriever = cache[key]
            for query in (support, counter):
                for item in retriever.retrieve(query, 50, cutoff):
                    chosen.setdefault(ticker, set()).add(item.evidence_id)
    return chosen


def point(models, position: int, chunk: dict, vector: list[float]):
    return models.PointStruct(
        id=position,
        vector=vector,
        payload={
            "position": position,
            "chunk_id": chunk["chunk_id"],
            "company_ticker": chunk["company_ticker"],
            "publication_date": chunk["publication_date"],
            "publication_date_is_estimated": chunk.get("publication_date_is_estimated", False),
            "publication_date_precision": chunk.get("publication_date_precision", "day"),
            "publication_date_basis": chunk.get("publication_date_basis"),
            "document_id": chunk["document_id"],
            "document_type": chunk["document_type"],
            "source_tier": chunk["source_tier"],
            "source_url": chunk["source_url"],
            "citation_span": chunk["citation_span"],
            "page": chunk.get("page"),
            "chart_id": chunk.get("chart_id"),
            "source_ref": chunk.get("source_ref"),
            "content_type": chunk.get("content_type", "narrative"),
            "extraction_method": chunk.get("extraction_method"),
            "validation_status": chunk.get("validation_status"),
        },
    )


def main() -> None:
    embedder = SentenceTransformersBackend(MODEL)
    chosen = collect(embedder)
    total = sum(len(ids) for ids in chosen.values())
    done = load_done()
    missing = [chunk_id for ids in chosen.values() for chunk_id in ids if chunk_id not in done]
    print(f"top50 passages {total} reused blurbs {total - len(missing)} to write {len(missing)}", flush=True)
    by_id = {}
    for ticker, ids in chosen.items():
        payload = load_index(SOURCE / ticker / "indexes" / "index.json")
        for chunk in payload["chunks"]:
            if chunk["chunk_id"] in ids and chunk["chunk_id"] not in done:
                by_id[chunk["chunk_id"]] = chunk
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(write_blurb, by_id[chunk_id], ask): chunk_id for chunk_id in missing if chunk_id in by_id}
            finished = 0
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    blurb = future.result()
                except Exception as exc:
                    print(f"blurb failed {chunk_id}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                done[chunk_id] = blurb
                handle.write(json.dumps({"chunk_id": chunk_id, "context_blurb": blurb}, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
                if finished % 100 == 0:
                    print(f"blurbs {finished}/{len(futures)}", flush=True)
    QdrantClient, models = _qdrant_modules()
    tickers = sorted(path.name for path in SOURCE.iterdir() if (path / "indexes" / "index.json").is_file())
    for ticker in tickers:
        ids = chosen.get(ticker, set())
        payload = load_index(SOURCE / ticker / "indexes" / "index.json")
        changed = []
        for position, chunk in enumerate(payload["chunks"]):
            blurb = done.get(chunk["chunk_id"])
            if blurb and chunk["chunk_id"] in ids:
                payload["chunks"][position] = prepend_blurb(chunk, blurb)
                changed.append(position)
        dest = DEST / ticker / "indexes"
        dest.mkdir(parents=True, exist_ok=True)
        index_path = dest / "index.json"
        index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        qdrant_dest = dest / "qdrant"
        if not (qdrant_dest / "qdrant_metadata.json").is_file():
            shutil.copytree(SOURCE / ticker / "indexes" / "qdrant", qdrant_dest)
        if changed:
            vectors = embedder.encode_documents([
                payload["chunks"][position].get("search_text") or payload["chunks"][position]["text"]
                for position in changed
            ])
            client = QdrantClient(path=str(qdrant_dest / "storage"))
            try:
                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=[point(models, position, payload["chunks"][position], list(map(float, vectors[i])))
                            for i, position in enumerate(changed)],
                    wait=True,
                )
            finally:
                client.close()
        write_runner_files(ticker, payload, index_path)
        print(f"updated {ticker} {len(changed)}", flush=True)
    print("BUILD_DONE", flush=True)
    print("Score with: python -m scripts.narrative.evaluation.run_rag_comparison "
          "--all-cases --cases data/narrative/evaluation/test_claims_v2/cases.csv "
          "--strategies contextual --backends bm25 dense hybrid r4 --build-version blurb_v2_top50 "
          "--retrieval-only --rerank-overflow truncate "
          "--output-dir data/narrative/evaluation/test_claims_v2_result/retrieval_blurb_v2 --resume")


def write_runner_files(ticker: str, payload: dict, index_path: Path) -> None:
    """Write the build report and source map the retrieval runner checks."""
    source_map = SOURCE / ticker / "source_map.jsonl"
    if not source_map.is_file():
        raise FileNotFoundError(f"Missing source map for {ticker}")
    shutil.copy2(source_map, DEST / ticker / "source_map.jsonl")
    parsed = Path("data/narrative/processed") / ticker / "documents.jsonl"
    report = {
        "status": "COMPLETE",
        "strategy": "contextual",
        "ticker": ticker,
        "inputs": {"parsed_sha256": file_sha(parsed)},
        "corpus_fingerprint": payload.get("corpus_fingerprint"),
        "index_sha256": file_sha(index_path),
    }
    (DEST / ticker / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
