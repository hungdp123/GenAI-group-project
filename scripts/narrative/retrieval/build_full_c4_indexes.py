"""Build isolated, full-corpus C4 runtime indexes without benchmark date filtering.

Reuses processed sources, adaptive plan selection and the existing BM25/Qdrant
implementations. No evaluation labels or generation LLM/API requests are used.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

from finsight.agents.narrative.corpus.processing.adaptive_chunking import AdaptiveConfig, choose_plan, count_tokens
from finsight.agents.narrative.corpus.processing.chunk import NarrativeChunk
from finsight.agents.narrative.corpus.processing.corpus import file_sha256
from finsight.agents.narrative.corpus.processing.index import build_index, load_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock
from finsight.agents.narrative.retrieval.build_vector_databases import _load_company_index, _validate_qdrant
from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    DEFAULT_EMBEDDING_MODEL, SentenceTransformersBackend,
)
from scripts.narrative.retrieval.c4_storage import (
    DEFAULT_OUTPUT, resume_settings, validate_output, write_chunk_artifacts,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_company(workspace, output, ticker, backend, config, settings, resume):
    started = time.monotonic()
    source_index = _load_company_index(workspace, ticker)
    parsed = workspace / "data/narrative/processed" / ticker / "documents.jsonl"
    inputs = {
        "parsed_sha256": file_sha256(parsed),
        "canonical_fingerprint": source_index["corpus_fingerprint"],
        "settings_sha256": identity(settings),
    }
    company = output / ticker
    checkpoint = company / "build_report.json"
    index_path = company / "indexes/index.json"
    vector_dir = index_path.parent / "qdrant"
    if checkpoint.exists():
        report = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not resume or report["inputs"] != inputs:
            raise ValueError(f"{ticker}: existing build/config differs; use --resume or a new output directory")
        if file_sha256(index_path) != report["index_sha256"]:
            raise ValueError(f"{ticker}: saved BM25 index changed")
        payload = load_index(index_path)
        write_chunk_artifacts(company, payload)
        _validate_qdrant(vector_dir, payload, model_name=backend.model_name,
                         dimension=backend.model.get_sentence_embedding_dimension())
        print(f"{ticker}: verified complete checkpoint", flush=True)
        return report

    chunks_checkpoint = company / "chunking_report.json"
    if chunks_checkpoint.exists():
        report = json.loads(chunks_checkpoint.read_text(encoding="utf-8"))
        if not resume or report["inputs"] != inputs or file_sha256(index_path) != report["index_sha256"]:
            raise ValueError(f"{ticker}: changed chunk checkpoint; use a new output directory")
        payload = load_index(index_path)
        print(f"{ticker}: reusing validated C4 chunks", flush=True)
    else:
        if company.exists():
            if not resume:
                raise ValueError(f"{ticker}: unfinished output exists; use --resume")
        company.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in parsed.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_docs = {c["document_id"] for c in source_index["chunks"]}
        if {r["document_id"] for r in rows} != expected_docs:
            raise ValueError(f"{ticker}: parsed and indexed source document sets differ")
        chunks, selections = [], Counter()
        source_chars = covered_chars = source_words = max_tokens = 0
        limit = settings["effective_token_limit"]
        tokenizer = backend.model.tokenizer
        print(f"{ticker}: C4 processing {len(expected_docs)} documents / {len(rows)} source blocks", flush=True)
        with gzip.open(company / "candidate_scores.jsonl.gz", "wt", encoding="utf-8") as trace, \
                (company / "source_map.jsonl").open("w", encoding="utf-8", newline="\n") as maps:
            for line, row in enumerate(rows, 1):
                block = ProcessedBlock(**row)
                if block.company_ticker != ticker:
                    raise ValueError(f"{ticker}: source belongs to another company")
                pieces, decision = choose_plan(block, tokenizer, limit, config)
                trace.write(json.dumps({"document_id": block.document_id, "block_line_1based": line,
                                        **decision}, ensure_ascii=False) + "\n")
                if pieces is None:
                    raise ValueError(f"{ticker}: no hard-valid C4 plan for source block {line}")
                cursor = 0
                for ordinal, piece in enumerate(pieces, 1):
                    a, z = piece["body_start"], piece["body_end"]
                    if a != cursor or z < a:
                        raise ValueError(f"{ticker}: noncontiguous source coverage at block {line}")
                    prefix = "\n\n".join(block.text[x:y] for x, y in piece["context_spans"])
                    text = (prefix + "\n\n" if prefix else "") + block.text[a:z]
                    tokens = count_tokens(tokenizer, text)
                    if text != piece["text"] or tokens > limit:
                        raise ValueError(f"{ticker}: C4 literal-text/token check failed at block {line}")
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    chunk_id = f"{block.document_id}:c4:block{line:06d}:{ordinal:04d}:{digest[:12]}"
                    chunk = NarrativeChunk(**{**row, "chunk_id": chunk_id, "text": text, "text_sha256": digest})
                    chunks.append(chunk)
                    maps.write(json.dumps({"chunk_id": chunk_id, "document_id": block.document_id,
                                           "block_line_1based": line, "body_char_span": [a, z],
                                           "context_char_spans": piece["context_spans"],
                                           "parsed_sha256": inputs["parsed_sha256"]}) + "\n")
                    cursor = z
                    covered_chars += z - a
                    max_tokens = max(max_tokens, tokens)
                if cursor != len(block.text):
                    raise ValueError(f"{ticker}: incomplete source coverage at block {line}")
                source_chars += len(block.text)
                source_words += len(block.text.split())
                selections[decision["selected_candidate"]] += 1
                if line % 250 == 0:
                    print(f"{ticker}: {line}/{len(rows)} blocks; {len(chunks)} chunks", flush=True)
        if source_chars != covered_chars or {c.document_id for c in chunks} != expected_docs:
            raise ValueError(f"{ticker}: full-corpus coverage validation failed")
        if len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError(f"{ticker}: duplicate chunk identifiers")
        if file_sha256(parsed) != inputs["parsed_sha256"]:
            raise ValueError(f"{ticker}: input changed during generation")
        fingerprint = identity({"ticker": ticker, **inputs})
        # Keep provenance for identical text at different publication dates.
        # Removing it globally could hide an eligible earlier source at query time.
        payload = build_index(chunks, index_path, corpus_fingerprint=fingerprint,
                              target_words=0, overlap_words=0, deduplicate_text=False)
        payload["chunking"] = {"strategy": "adaptive", "arm": "C4", **asdict(config),
                               "effective_token_limit": limit}
        payload["source_corpus_fingerprint"] = source_index.get(
            "source_corpus_fingerprint", source_index["corpus_fingerprint"])
        payload["corpus_scope"] = {"publication_cutoff": None, "policy": "all current eligible processed documents"}
        write_json(index_path, payload)
        dates = [c.publication_date for c in chunks]
        report = {
            "ticker": ticker, "inputs": inputs, "status": "CHUNKS_VALIDATED", "documents": len(expected_docs),
            "source_blocks": len(rows), "chunks": len(chunks), "source_words": source_words,
            "source_characters": source_chars, "covered_source_characters": covered_chars,
            "content_preservation_passed": source_chars == covered_chars, "maximum_embedding_tokens": max_tokens,
            "publication_cutoff": None, "earliest_publication_date": min(dates), "latest_publication_date": max(dates),
            "documents_after_2025_07_31": len({c.document_id for c in chunks if c.publication_date > "2025-07-31"}),
            "selected_configurations": dict(selections), "corpus_fingerprint": fingerprint,
            "index_sha256": file_sha256(index_path), "exact_text_duplicates_retained_for_provenance": True,
            "llm_api_requests": 0,
        }
        write_json(chunks_checkpoint, report)

    write_chunk_artifacts(company, payload)
    if not (vector_dir / "qdrant_metadata.json").exists():
        if vector_dir.exists():
            raise RuntimeError(
                f"{ticker}: incomplete Qdrant directory retained at {vector_dir}; use a new build version")
        print(f"{ticker}: embedding {len(payload['chunks'])} C4 chunks locally", flush=True)
        build_qdrant_index(payload, vector_dir, backend)
    metadata = _validate_qdrant(vector_dir, payload, model_name=backend.model_name,
                                dimension=backend.model.get_sentence_embedding_dimension())
    report = {**report, "status": "COMPLETE", "vector_count": metadata["vector_count"],
              "elapsed_seconds": round(time.monotonic() - started, 2)}
    write_json(checkpoint, report)
    print(f"{ticker}: complete — {report['documents']} documents, {report['chunks']} chunks/vectors", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--ticker", type=str.upper)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=Path("docs/narrative/config/narrative_c4_config_v1.json"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    workspace = Path.cwd().resolve()
    try:
        output = validate_output(workspace, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    tickers = ([args.ticker] if args.ticker else
               sorted(p.parent.name for p in (workspace / "data/narrative/processed").glob("*/documents.jsonl")))
    if not tickers or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", t) for t in tickers):
        parser.error("No valid company inputs")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from huggingface_hub import snapshot_download

    torch.set_num_threads(args.threads)
    snapshot = (Path(args.embedding_model).resolve() if Path(args.embedding_model).is_dir() else
                Path(snapshot_download(args.embedding_model, local_files_only=True)))
    backend = SentenceTransformersBackend(args.embedding_model)
    tokenizer = backend.model.tokenizer
    limit = min(backend.model.max_seq_length, tokenizer.model_max_length)
    config = AdaptiveConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    config.validate(limit)
    code = [Path(__file__), workspace / "finsight/agents/narrative/corpus/processing/adaptive_chunking.py",
            workspace / "finsight/agents/narrative/corpus/processing/index.py",
            workspace / "finsight/agents/narrative/retrieval/qdrant_index.py"]
    settings = {
        "strategy": "C4", "config": asdict(config), "effective_token_limit": limit,
        "embedding_model": args.embedding_model, "embedding_revision": snapshot.name,
        "model_files": {p.name: file_sha256(p) for p in snapshot.iterdir()
                        if p.is_file() and p.suffix in {".json", ".txt", ".safetensors", ".bin"}},
        "code_sha256": {str(p.relative_to(workspace)): file_sha256(p) for p in code},
        "publication_cutoff": None, "input_policy": "current eligible processed source blocks; no evaluation labels",
    }
    config_path = output / "configuration.json"
    if config_path.exists():
        if not args.resume:
            parser.error("Existing build requires --resume or a new output version")
        try:
            settings = resume_settings(output, settings, workspace)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        write_json(config_path, settings)
    reports = []
    for ticker in tickers:
        reports.append(build_company(workspace, output, ticker, backend, config, settings, args.resume))
    report = {
        "status": "COMPLETE", "built_at": datetime.now(timezone.utc).isoformat(),
        "companies": len(reports), "documents": sum(r["documents"] for r in reports),
        "chunks": sum(r["chunks"] for r in reports), "vectors": sum(r["vector_count"] for r in reports),
        "publication_cutoff": None, "llm_api_requests": 0, "company_reports": reports,
        "note": "Technical build validation only; historical retrieval scores do not apply to this build.",
    }
    write_json(output / ("build_report.json" if args.all else f"build_report_{args.ticker}.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "company_reports"}, indent=2))


if __name__ == "__main__":
    main()
