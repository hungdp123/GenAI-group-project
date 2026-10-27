"""Build full-corpus C1/C2/C3/C4 chunks, BM25 and vectors in separate strategy folders.

No benchmark cutoff or evaluation labels. C4 delegates to its existing builder;
fixed arms reuse the existing strict word-window chunker and BM25/Qdrant builders.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from finsight.agents.narrative.corpus.processing.chunk import chunk_blocks
from finsight.agents.narrative.corpus.processing.corpus import file_sha256
from finsight.agents.narrative.corpus.processing.index import build_index, load_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock
from finsight.agents.narrative.retrieval.build_vector_databases import _load_company_index, _validate_qdrant
from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    DEFAULT_EMBEDDING_MODEL, SentenceTransformersBackend,
)
from scripts.narrative.retrieval.build_full_c4_indexes import identity, write_json
from scripts.narrative.retrieval.c4_storage import write_chunk_artifacts


FIXED_ARMS = {"C1": (800, 120), "C2": (300, 50), "C3": (400, 60)}
STRATEGIES = (*FIXED_ARMS, "C4")
OUTPUT_ROOT = Path("data/narrative/processed/chunking_strategies")


def fixed_chunks(rows: list[dict], arm: str):
    """Reuse strict windows, independently checking literal words and full coverage."""
    target, overlap = FIXED_ARMS[arm]
    blocks = [ProcessedBlock(**row) for row in rows]
    candidates = chunk_blocks([replace(b, content_type="narrative") for b in blocks],
                              target_words=target, overlap_words=overlap, boundary_policy="word")
    chunks, maps = [], []
    cursor = words_total = covered_total = 0
    for line, block in enumerate(blocks, 1):
        words = list(re.finditer(r"\S+", block.text))
        covered_end = 0
        for start in range(0, len(words), target - overlap):
            end = min(start + target, len(words))
            chunk = replace(candidates[cursor], content_type=block.content_type)
            cursor += 1
            if start > covered_end or chunk.text != " ".join(w.group() for w in words[start:end]):
                raise ValueError(f"{arm}: fixed-window coverage/literal-text validation failed at block {line}")
            if not chunk.citation_span or not chunk.local_path:
                raise ValueError(f"{arm}: missing source provenance at block {line}")
            covered_end = max(covered_end, end)
            chunks.append(chunk)
            maps.append({"chunk_id": chunk.chunk_id, "document_id": chunk.document_id,
                         "block_line_1based": line, "source_word_start": start, "source_word_end": end,
                         "body_char_span": [words[start].start(), words[end - 1].end()],
                         "context_char_spans": [], "normalization": "whitespace-to-single-space"})
        if covered_end != len(words):
            raise ValueError(f"{arm}: incomplete source coverage at block {line}")
        words_total += len(words)
        covered_total += covered_end
    if cursor != len(candidates) or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError(f"{arm}: chunk count/identifier validation failed")
    return chunks, maps, words_total, covered_total


def build_fixed_company(workspace, output, ticker, backend, settings, resume):
    started = time.monotonic()
    arm = settings["strategy"]
    source = _load_company_index(workspace, ticker)
    parsed = workspace / "data/narrative/processed" / ticker / "documents.jsonl"
    inputs = {"parsed_sha256": file_sha256(parsed), "canonical_fingerprint": source["corpus_fingerprint"],
              "settings_sha256": identity(settings)}
    company = output / ticker
    index_path = company / "indexes/index.json"
    vector_dir = index_path.parent / "qdrant"
    complete, chunk_report = company / "build_report.json", company / "chunking_report.json"
    checkpoint = complete if complete.exists() else chunk_report
    if checkpoint.exists():
        report = json.loads(checkpoint.read_text(encoding="utf-8"))
        if not resume or report["inputs"] != inputs or file_sha256(index_path) != report["index_sha256"]:
            raise ValueError(f"{arm}/{ticker}: changed checkpoint; use a new build version")
        payload = load_index(index_path)
        print(f"{arm}/{ticker}: reusing validated chunks", flush=True)
    else:
        if company.exists() and not resume:
            raise ValueError(f"{arm}/{ticker}: unfinished output exists; use --resume")
        company.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in parsed.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_docs = {c["document_id"] for c in source["chunks"]}
        if ({r["document_id"] for r in rows} != expected_docs
                or any(r["company_ticker"] != ticker for r in rows)):
            raise ValueError(f"{arm}/{ticker}: parsed/indexed source document sets differ")
        print(f"{arm}/{ticker}: chunking all {len(expected_docs)} eligible documents", flush=True)
        chunks, maps, source_words, covered_words = fixed_chunks(rows, arm)
        if {c.document_id for c in chunks} != expected_docs or file_sha256(parsed) != inputs["parsed_sha256"]:
            raise ValueError(f"{arm}/{ticker}: missing document coverage or changed source text")
        for item in maps:
            item["parsed_sha256"] = inputs["parsed_sha256"]
        with (company / "source_map.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in maps:
                handle.write(json.dumps(item) + "\n")
        counts = []
        overflow = []
        for start in range(0, len(chunks), 256):
            batch = chunks[start:start + 256]
            ids = backend.model.tokenizer([c.text for c in batch], add_special_tokens=True,
                                          truncation=False, padding=False, verbose=False)["input_ids"]
            for chunk, tokens in zip(batch, ids, strict=True):
                counts.append(len(tokens))
                if len(tokens) > settings["effective_token_limit"]:
                    overflow.append({"chunk_id": chunk.chunk_id, "tokens_including_special": len(tokens),
                                     "effective_limit": settings["effective_token_limit"]})
        with (company / "token_overflow.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in overflow:
                handle.write(json.dumps(item) + "\n")
        target, overlap = FIXED_ARMS[arm]
        payload = build_index(chunks, index_path, corpus_fingerprint=identity({"ticker": ticker, **inputs}),
                              target_words=target, overlap_words=overlap, deduplicate_text=False)
        payload["chunking"].update(strategy="fixed", arm=arm, boundary_policy="word",
                                   content_policy="strict_word_windows_all_content_types_within_source_blocks")
        payload["corpus_scope"] = {"publication_cutoff": None, "policy": "all current eligible processed documents"}
        payload["embedding_input_policy"] = {"overflow_policy": settings["embedding_overflow"],
                                             "effective_token_limit": settings["effective_token_limit"],
                                             "oversized_chunks": len(overflow), "source_text_truncated": False}
        write_json(index_path, payload)
        report = {
            "status": "CHUNKS_VALIDATED", "strategy": arm, "ticker": ticker, "inputs": inputs,
            "documents": len(expected_docs), "chunks": len(chunks), "source_blocks": len(rows),
            "source_words": source_words, "covered_source_words": covered_words,
            "content_preservation_passed": source_words == covered_words,
            "content_preservation_basis": "source word occurrences; whitespace normalization",
            "maximum_embedding_tokens": max(counts, default=0), "overflow_chunks": len(overflow),
            "embedding_overflow": settings["embedding_overflow"], "publication_cutoff": None,
            "latest_publication_date": max(c.publication_date for c in chunks),
            "corpus_fingerprint": payload["corpus_fingerprint"], "index_sha256": file_sha256(index_path),
            "exact_text_duplicates_retained_for_provenance": True, "llm_api_requests": 0,
        }
        write_json(chunk_report, report)
    write_chunk_artifacts(company, payload)
    if report["overflow_chunks"] and settings["embedding_overflow"] == "error":
        raise ValueError(
            f"{arm}/{ticker}: {report['overflow_chunks']} chunks exceed the embedding token limit. "
            "To keep fixed chunks and permit embedding truncation, use --embedding-overflow truncate "
            "with a new --build-version. BM25 text and source maps remain complete.")
    if not (vector_dir / "qdrant_metadata.json").exists():
        if vector_dir.exists():
            raise ValueError(f"{arm}/{ticker}: incomplete vector storage retained; use a new build version")
        metadata = build_qdrant_index(payload, vector_dir, backend)
        metadata["embedding_input_policy"] = payload["embedding_input_policy"]
        write_json(vector_dir / "qdrant_metadata.json", metadata)
    metadata = _validate_qdrant(vector_dir, payload, model_name=backend.model_name,
                                dimension=backend.model.get_sentence_embedding_dimension())
    if metadata.get("embedding_input_policy") != payload["embedding_input_policy"]:
        raise ValueError(f"{arm}/{ticker}: vector embedding policy differs from selected BM25 configuration")
    report = {**report, "status": "COMPLETE", "vector_count": metadata["vector_count"],
              "elapsed_seconds": round(time.monotonic() - started, 2)}
    write_json(complete, report)
    print(f"{arm}/{ticker}: complete; {report['documents']} documents / {report['chunks']} chunks", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="All companies with current processed source blocks")
    selection.add_argument("--ticker", type=str.upper)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--build-version", default="full_v1")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--embedding-overflow", choices=("error", "truncate"), default="error",
                        help="Explicit policy for fixed C1-C3 inputs; C4 remains token-valid")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--config", type=Path, default=Path("docs/narrative/config/narrative_c4_config_v1.json"))
    args = parser.parse_args()
    if args.threads < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.build_version):
        parser.error("Use positive --threads and a build version containing letters, numbers, underscores or hyphens")
    workspace = Path.cwd().resolve()
    root = (workspace / OUTPUT_ROOT).resolve()
    if not root.is_relative_to(workspace):
        parser.error("Strategy output root resolves outside the workspace")
    tickers = [args.ticker] if args.ticker else sorted(
        p.parent.name for p in (workspace / "data/narrative/processed").glob("*/documents.jsonl"))
    if not tickers or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", t) for t in tickers):
        parser.error("No valid company inputs")
    arms = list(dict.fromkeys(args.strategies))
    backend = None
    company_reports = {}
    for arm in arms:
        output = root / arm / args.build_version
        if arm == "C4":
            command = [sys.executable, "-u", "-m", "scripts.narrative.retrieval.build_full_c4_indexes",
                       *( ["--all"] if args.all else ["--ticker", args.ticker]),
                       "--output-dir", str(output), "--threads", str(args.threads),
                       "--embedding-model", args.embedding_model, "--config", str(args.config)]
            if args.resume:
                command.append("--resume")
            subprocess.run(command, check=True)
        else:
            if backend is None:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                import torch
                from huggingface_hub import snapshot_download

                torch.set_num_threads(args.threads)
                snapshot = (Path(args.embedding_model).resolve() if Path(args.embedding_model).is_dir() else
                            Path(snapshot_download(args.embedding_model, local_files_only=True)))
                backend = SentenceTransformersBackend(args.embedding_model)
            code = [Path(__file__), workspace / "finsight/agents/narrative/corpus/processing/chunk.py",
                    workspace / "finsight/agents/narrative/corpus/processing/index.py",
                    workspace / "finsight/agents/narrative/retrieval/qdrant_index.py"]
            target, overlap = FIXED_ARMS[arm]
            settings = {
                "strategy": arm, "config": {"target_words": target, "overlap_words": overlap,
                                             "boundary_policy": "word", "protect_tables_charts": False},
                "embedding_model": args.embedding_model, "embedding_revision": snapshot.name,
                "effective_token_limit": min(backend.model.max_seq_length, backend.model.tokenizer.model_max_length),
                "embedding_overflow": args.embedding_overflow, "publication_cutoff": None,
                "model_files": {p.name: file_sha256(p) for p in snapshot.iterdir()
                                if p.is_file() and p.suffix in {".json", ".txt", ".safetensors", ".bin"}},
                "code_sha256": {str(p.relative_to(workspace)): file_sha256(p) for p in code},
            }
            config_path = output / "configuration.json"
            if config_path.exists():
                if not args.resume or json.loads(config_path.read_text(encoding="utf-8")) != settings:
                    raise ValueError(f"{arm}: configuration/code differs; use --resume or a new build version")
            else:
                write_json(config_path, settings)
            for ticker in tickers:
                build_fixed_company(workspace, output, ticker, backend, settings, args.resume)
        company_reports[arm] = [json.loads((output / t / "build_report.json").read_text(encoding="utf-8"))
                                for t in tickers]
        write_json(output / ("build_report.json" if args.all else f"build_report_{args.ticker}.json"), {
            "status": "COMPLETE", "strategy": arm, "companies": len(tickers), "publication_cutoff": None,
            "documents": sum(r["documents"] for r in company_reports[arm]),
            "chunks": sum(r["chunks"] for r in company_reports[arm]),
            "vectors": sum(r["vector_count"] for r in company_reports[arm]),
            "company_reports": company_reports[arm], "llm_api_requests": 0,
        })
    for i, ticker in enumerate(tickers):
        identities = {(company_reports[a][i]["inputs"]["parsed_sha256"],
                       company_reports[a][i]["inputs"]["canonical_fingerprint"]) for a in arms}
        if len(identities) != 1:
            raise ValueError(f"{ticker}: source builds differ between strategies; do not compare these builds")
    report = {"status": "COMPLETE", "built_at": datetime.now(timezone.utc).isoformat(),
              "strategies": arms, "companies": len(tickers), "publication_cutoff": None,
              "documents_per_strategy": {a: sum(r["documents"] for r in company_reports[a]) for a in arms},
              "llm_api_requests": 0, "note": "Index build only; no retrieval quality evaluation was run."}
    suffix = "" if args.all else f"_{args.ticker}"
    write_json(root / f"{args.build_version}{suffix}_build_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
