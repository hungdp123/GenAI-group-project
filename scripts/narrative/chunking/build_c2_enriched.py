"""Build C2 metadata/blurb variants without changing original C2 passages.

Run from the repository root. Both modes build matching vectors by default,
using the source build's embedding model. ``--no-embed`` builds BM25 only.

Metadata (no generation calls)::

    python -m scripts.narrative.chunking.build_c2_enriched --mode metadata \
        --cases data/narrative/evaluation/test_claims_v2/cases_combined.csv \
        --case-ranges 1-20 38-49 --resume

Preview the blurb shortlist without generation calls::

    python -m scripts.narrative.chunking.build_c2_enriched --mode blurb \
        --case-ranges 1-20 38-49 --blurb-model gpt-5.4 --prepare-only --resume

Generate blurbs and build the enriched indexes (uses the configured provider)::

    python -m scripts.narrative.chunking.build_c2_enriched --mode blurb \
        --case-ranges 1-20 38-49 --blurb-model gpt-5.4 --resume

Blurb mode enriches the union of top-k BM25/dense results for BOTH query
directions and ALL selected claim types. It does not pass gold labels, gold
answers, gold snippets or distractors to retrieval or blurb generation. This
is query-selected partial enrichment, not blurb generation over every chunk.

Outputs are separate builds; the source build is never overwritten. Successful
API responses are checkpointed. An interrupted/failed API attempt requires
explicit ``--retry-incomplete`` before it can be issued again. Run only one
builder per output root. Incomplete staging directories are left for inspection.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from uuid import uuid4

from finsight.agents.narrative.corpus.processing.chunk_context import (
    BLURB_SYSTEM,
    blurb_prompt,
    prepend_blurb,
    write_blurb,
)
from finsight.agents.narrative.corpus.processing.contextual import context_prefix
from finsight.agents.narrative.corpus.processing.index import load_index


PROTOCOL = "c2_preserved_enrichment_v1"
BASE = Path("data/narrative/processed/chunking_strategies/C2")
CASES = Path("data/narrative/evaluation/test_claims_v2/cases_combined.csv")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_cases(path: Path, ranges: list[str] | None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any({"id", "company", "claims", "cut_off"} - row.keys() for row in rows):
        raise ValueError("Cases require id, company, claims and cut_off columns")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Case IDs must be unique")
    if ranges:
        wanted: set[int] = set()
        for value in ranges:
            if not re.fullmatch(r"\d+(?:-\d+)?", value):
                raise ValueError(f"Invalid case range: {value}")
            bounds = [int(n) for n in value.split("-")]
            start, end = bounds[0], bounds[-1]
            if not 1 <= start <= end <= len(rows):
                raise ValueError(f"Out-of-bounds case range: {value}")
            part = set(range(start, end + 1))
            if wanted & part:
                raise ValueError(f"Overlapping case range: {value}")
            wanted.update(part)
        rows = [row for number, row in enumerate(rows, 1) if number in wanted]
    # Only these fields can affect selection. Gold annotations are not consumed.
    selected = [{k: row[k] for k in ("id", "company", "claims", "cut_off")} for row in rows]
    for row in selected:
        row["company"] = row["company"].upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", row["company"]):
            raise ValueError(f"Invalid company ticker: {row['company']}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["cut_off"]):
            raise ValueError(f"Missing/invalid cutoff for {row['id']}")
        date.fromisoformat(row["cut_off"])
    return selected


def load_source(workspace: Path, root: Path, ticker: str, mode: str) -> dict:
    company = root / ticker
    index = company / "indexes/index.json"
    report = read_json(company / "build_report.json")
    parsed = workspace / "data/narrative/processed" / ticker / "documents.jsonl"
    parsed_sha = file_sha(parsed)
    if (report.get("status") != "COMPLETE"
            or report.get("index_sha256") != file_sha(index)
            or report.get("inputs", {}).get("parsed_sha256") != parsed_sha):
        raise ValueError(f"Incomplete/stale source build for {ticker}")
    payload = load_index(index)
    chunking = payload.get("chunking", {})
    if chunking.get("target_words") != 300 or chunking.get("overlap_words") != 50:
        raise ValueError(f"{ticker}: expected original C2 300/50 boundaries")
    if mode == "metadata" and payload.get("enrichment"):
        raise ValueError("Metadata mode requires the original unenriched C2 build")
    if mode == "blurb" and payload.get("enrichment", {}).get("mode") != "metadata":
        raise ValueError("Blurb mode requires this script's C2 metadata build first")
    chunks = payload["chunks"]
    ids = [chunk["chunk_id"] for chunk in chunks]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{ticker}: empty/duplicate source chunk IDs")
    map_path = company / "source_map.jsonl"
    maps = [json.loads(line) for line in map_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row["chunk_id"] for row in maps] != ids:
        raise ValueError(f"{ticker}: source-map order/IDs differ from index")
    for chunk, mapping in zip(chunks, maps, strict=True):
        if (chunk["company_ticker"] != ticker or mapping["document_id"] != chunk["document_id"]
                or mapping.get("parsed_sha256") != parsed_sha):
            raise ValueError(f"{ticker}: source-map provenance differs from source build")
    vector_metadata = company / "indexes/qdrant/qdrant_metadata.json"
    if vector_metadata.is_file():
        vectors = read_json(vector_metadata)
        if vectors["corpus_fingerprint"] != payload["corpus_fingerprint"] or vectors["chunk_ids"] != ids:
            raise ValueError(f"{ticker}: source vector metadata is stale")
        model = vectors["embedding_model"]
    else:
        model = report.get("embedding_model")
        if not model:
            raise ValueError(f"{ticker}: source build does not record its embedding model")
    return {
        "company": company, "payload": payload, "parsed_sha": parsed_sha,
        "model": model,
        "hashes": {"index": file_sha(index), "source_map": file_sha(map_path), "parsed": parsed_sha},
    }


def shortlist(args, cases: list[dict], backend, output: Path, signature: str) -> dict[str, list[str]]:
    path = output / "selected_passages.json"
    if path.exists():
        saved = read_json(path)
        if saved["configuration_sha256"] != signature:
            raise ValueError("Changed shortlist configuration; use a new output root")
        return saved["selected_chunk_ids"]
    from finsight.agents.narrative.rag.collect_evidence import build_queries
    from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
    from finsight.contracts.schemas import SubClaim, SubClaimType

    chosen: dict[str, set[str]] = {}
    for ticker in sorted({row["company"] for row in cases}):
        retrievers = {
            kind: CorpusRetriever(
                company_ticker=ticker, corpus_root=args.source_root, backend=kind,
                candidate_k=args.candidate_k,
                embedding_backend=backend if kind == "dense" else None,
            )
            for kind in args.selection_backends
        }
        for case in (row for row in cases if row["company"] == ticker):
            claim = SubClaim(
                sub_claim_id=case["id"], company_ticker=ticker,
                text=case["claims"], sub_claim_type=SubClaimType.OTHER,
            )
            for query in build_queries(claim):
                for retriever in retrievers.values():
                    hits = retriever.retrieve(query, args.candidate_k, date.fromisoformat(case["cut_off"]))
                    chosen.setdefault(ticker, set()).update(hit.evidence_id for hit in hits)
    result = {ticker: sorted(ids) for ticker, ids in chosen.items()}
    write_json(path, {"configuration_sha256": signature, "selected_chunk_ids": result})
    return result


def cache_entries(path: Path) -> tuple[dict[str, str], set[str]]:
    done, pending = {}, set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["key"]
            if row["status"] == "DONE":
                done[key] = row["blurb"]
                pending.discard(key)
            else:
                pending.add(key)
    return done, pending


def blurb_key(chunk: dict, configuration: dict) -> str:
    return digest({
        "provider": configuration["provider"], "model": configuration["blurb_model"],
        "system": BLURB_SYSTEM, "prompt": blurb_prompt(chunk),
        "max_tokens": configuration["blurb_max_tokens"],
    })


def generate_blurb(chunk: dict, args, configuration: dict, cache: Path,
                   done: dict[str, str], pending: set[str]) -> str:
    key = blurb_key(chunk, configuration)
    if key in done:
        return done[key]
    if key in pending and not args.retry_incomplete:
        raise ValueError(
            f"Unresolved prior API attempt for {chunk['chunk_id']}. Review {cache}; "
            "use --retry-incomplete only if you intend another possibly billed request."
        )
    from finsight.llm.client import complete

    def append(row):
        with cache.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "chunk_id": chunk["chunk_id"], **row}) + "\n")
            handle.flush()

    append({"status": "STARTED"})
    usage = {}

    def ask(prompt):
        response = complete(
            prompt=prompt, system=BLURB_SYSTEM, model=configuration["blurb_model"],
            max_tokens=args.blurb_max_tokens,
        )
        usage.update({name: getattr(response, name) for name in (
            "provider", "model", "input_tokens", "output_tokens", "cost_usd", "latency_ms",
        )})
        if response.finish_reason in {"length", "max_tokens"}:
            raise ValueError("Incomplete blurb response; review output-token budget before retrying")
        return response.text

    try:
        blurb = write_blurb(chunk, ask)
    except BaseException as exc:
        append({"status": "FAILED", "error_type": type(exc).__name__, "usage": usage})
        pending.add(key)
        raise
    append({"status": "DONE", "blurb": blurb, "usage": usage})
    done[key] = blurb
    pending.discard(key)
    return blurb


def already_built(target: Path, source: dict, signature: str, embed: bool) -> bool:
    if not target.exists():
        return False
    report = read_json(target / "build_report.json")
    index = target / "indexes/index.json"
    if (report.get("status") != "COMPLETE" or report.get("configuration_sha256") != signature
            or report.get("index_sha256") != file_sha(index)
            or file_sha(target / "source_map.jsonl") != source["hashes"]["source_map"]
            or report.get("chunks_sha256") != file_sha(target / "chunks.jsonl")):
        raise ValueError(f"Changed/incomplete output at {target}; use a new output root")
    if embed:
        payload = load_index(index)
        metadata = read_json(target / "indexes/qdrant/qdrant_metadata.json")
        if (metadata["corpus_fingerprint"] != payload["corpus_fingerprint"]
                or metadata["embedding_model"] != source["model"]
                or metadata["chunk_ids"] != [chunk["chunk_id"] for chunk in payload["chunks"]]
                or not (target / "indexes/qdrant/storage").is_dir()):
            raise ValueError(f"Changed/incomplete vector output at {target}")
    return True


def publish(args, ticker: str, source: dict, chunks: list[dict], backend,
            signature: str, selected: list[str], configuration: dict) -> None:
    # These are the only per-chunk fields enrichment is permitted to change/add.
    allowed = {"search_text", "context_blurb"}
    original = source["payload"]["chunks"]
    if len(chunks) != len(original):
        raise ValueError("Enrichment must preserve the source chunk count")
    for before, after in zip(original, chunks, strict=True):
        if ({k: v for k, v in before.items() if k not in allowed}
                != {k: v for k, v in after.items() if k not in allowed}):
            raise ValueError("Enrichment changed source text, identity, order or provenance")
    payload = {**source["payload"], "chunks": chunks}
    payload["source_corpus_fingerprint"] = source["payload"]["corpus_fingerprint"]
    payload["corpus_fingerprint"] = digest({
        "source_index_sha256": source["hashes"]["index"], "configuration": signature,
        "search_text": [(c["chunk_id"], c.get("search_text") or c["text"]) for c in chunks],
    })
    payload["enrichment"] = {
        "protocol": PROTOCOL, "mode": args.mode, "chunk_boundaries_preserved": True,
        "selected_blurb_chunks": len(selected) if args.mode == "blurb" else 0,
        "blurb_provider": configuration.get("provider"),
        "blurb_model": configuration.get("blurb_model"),
    }
    limit, overflow = None, []
    if args.embed:
        limit = int(backend.model.max_seq_length)
        expected_limit = source["payload"].get("embedding_input_policy", {}).get("effective_token_limit")
        if expected_limit is not None and limit != expected_limit:
            raise ValueError("Embedding token limit changed from the source build")
        for start in range(0, len(chunks), 128):
            batch = chunks[start:start + 128]
            encoded = backend.model.tokenizer(
                [c.get("search_text") or c["text"] for c in batch],
                add_special_tokens=True, truncation=False, padding=False, verbose=False,
            )["input_ids"]
            for chunk, tokens in zip(batch, encoded, strict=True):
                if len(tokens) > limit:
                    overflow.append({"chunk_id": chunk["chunk_id"], "tokens": len(tokens), "limit": limit})
    payload["embedding_input_policy"] = {
        "overflow_policy": "truncate", "effective_token_limit": limit,
        "oversized_chunks": len(overflow) if args.embed else None,
        "source_text_truncated": False, "search_text_embedded": True if args.embed else False,
    }
    stage = args.output_root / f".staging-{ticker}-{uuid4().hex}"
    stage.mkdir()
    index = stage / "indexes/index.json"
    write_json(index, payload)
    write_json(stage / "indexes/index_metadata.json", {k: v for k, v in payload.items() if k != "chunks"})
    with (stage / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    shutil.copy2(source["company"] / "source_map.jsonl", stage / "source_map.jsonl")
    with (stage / "token_overflow.jsonl").open("w", encoding="utf-8") as handle:
        for row in overflow:
            handle.write(json.dumps(row) + "\n")
    if args.embed:
        from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index

        build_qdrant_index(payload, stage / "indexes/qdrant", backend)
    write_json(stage / "build_report.json", {
        "status": "COMPLETE", "strategy": "C2", "ticker": ticker,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_sha256": signature,
        "inputs": {"parsed_sha256": source["parsed_sha"], **source["hashes"]},
        "chunks": len(chunks), "chunk_boundaries_preserved": True,
        "corpus_fingerprint": payload["corpus_fingerprint"], "index_sha256": file_sha(index),
        "chunks_sha256": file_sha(stage / "chunks.jsonl"),
        "source_map_sha256": file_sha(stage / "source_map.jsonl"),
        "embedding_model": source["model"], "vectors_built": args.embed,
        "selected_blurb_chunks": len(selected),
    })
    stage.rename(args.output_root / ticker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("metadata", "blurb"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--case-ranges", nargs="+", metavar="START-END")
    parser.add_argument("--selection-backends", nargs="+", choices=("bm25", "dense"), default=["bm25", "dense"])
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--blurb-model", help="Defaults to the configured shared model")
    parser.add_argument("--blurb-max-tokens", type=int, default=3000)
    parser.add_argument("--embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true", help="Write/inspect the shortlist; no blurb calls")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true", help="Explicitly retry unresolved API attempts")
    return parser


def run(args) -> None:
    workspace = Path.cwd().resolve()
    args.source_root = (args.source_root or BASE / ("full_v1" if args.mode == "metadata" else "contextual_metadata_v1")).resolve()
    args.output_root = (args.output_root or BASE / ("contextual_metadata_v1" if args.mode == "metadata" else "contextual_blurb_v1")).resolve()
    if (args.source_root.is_relative_to(args.output_root)
            or args.output_root.is_relative_to(args.source_root)
            or not args.output_root.is_relative_to(workspace)):
        raise ValueError("Use a separate output root inside the workspace; source/output must not overlap")
    if args.candidate_k < 1 or args.blurb_max_tokens < 1:
        raise ValueError("Candidate count and token budget must be positive")
    cases = selected_cases(args.cases, args.case_ranges)
    tickers = sorted({row["company"] for row in cases})
    sources = {ticker: load_source(workspace, args.source_root, ticker, args.mode) for ticker in tickers}
    models = {source["model"] for source in sources.values()}
    if len(models) != 1:
        raise ValueError("Selected source companies must use the same embedding model")
    configuration = {
        "protocol": PROTOCOL, "mode": args.mode, "source_root": str(args.source_root),
        "cases_sha256": file_sha(args.cases), "selected_case_ids": [row["id"] for row in cases],
        "sources": {ticker: source["hashes"] for ticker, source in sources.items()},
        "embedding_model": next(iter(models)), "embed": args.embed,
        "candidate_k": args.candidate_k, "selection_backends": args.selection_backends,
        "builder_sha256": file_sha(Path(__file__)),
        "context_helper_sha256": file_sha(Path(context_prefix.__code__.co_filename)),
        "blurb_helper_sha256": file_sha(Path(write_blurb.__code__.co_filename)),
    }
    if args.mode == "blurb":
        from finsight.config import get_settings

        settings = get_settings()
        configuration.update(provider=settings.llm_provider, blurb_model=args.blurb_model or settings.llm_model,
                             blurb_max_tokens=args.blurb_max_tokens)
    signature = digest(configuration)
    manifest = args.output_root / "enrichment_configuration.json"
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.resume or not manifest.is_file() or read_json(manifest) != configuration:
            raise ValueError("Existing/changed build: use --resume with identical settings or a new output root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        write_json(manifest, configuration)
    backend = None
    if args.embed or (args.mode == "blurb" and "dense" in args.selection_backends):
        from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend

        if not (args.prepare_only and args.mode == "metadata"):
            backend = SentenceTransformersBackend(next(iter(models)))
    chosen = shortlist(args, cases, backend, args.output_root, signature) if args.mode == "blurb" else {}
    cache = args.output_root / "blurb_cache.jsonl"
    done, pending = cache_entries(cache)
    planned = 0
    for ticker, source in sources.items():
        if already_built(args.output_root / ticker, source, signature, args.embed):
            continue
        selected = set(chosen.get(ticker, []))
        ids = {c["chunk_id"] for c in source["payload"]["chunks"]}
        if not selected <= ids:
            raise ValueError(f"{ticker}: shortlist contains unknown chunk IDs")
        if args.mode == "blurb":
            planned += sum(blurb_key(c, configuration) not in done
                           for c in source["payload"]["chunks"] if c["chunk_id"] in selected)
    write_json(args.output_root / "build_plan.json", {
        "configuration_sha256": signature, "companies": tickers, "cases": len(cases),
        "selected_blurb_passages": sum(len(ids) for ids in chosen.values()),
        "new_blurb_requests_needed": planned,
        "note": "No gold labels are used. Top-k selection is query-dependent partial enrichment.",
    })
    print(json.dumps({"companies": len(tickers), "cases": len(cases), "new_blurb_requests_needed": planned,
                      "plan": str(args.output_root / "build_plan.json")}, indent=2), flush=True)
    if args.prepare_only:
        return
    for ticker, source in sources.items():
        target = args.output_root / ticker
        if already_built(target, source, signature, args.embed):
            print(f"Reusing completed {ticker}", flush=True)
            continue
        selected = set(chosen.get(ticker, []))
        updated = []
        for chunk in source["payload"]["chunks"]:
            if args.mode == "metadata":
                record = {**chunk, "search_text": f"{context_prefix(chunk)}\n{chunk.get('search_text') or chunk['text']}"}
            elif chunk["chunk_id"] in selected:
                record = prepend_blurb(chunk, generate_blurb(chunk, args, configuration, cache, done, pending))
            else:
                record = dict(chunk)
            updated.append(record)
        publish(args, ticker, source, updated, backend, signature, sorted(selected), configuration)
        print(f"Built {ticker}: {len(updated)} preserved chunks, {len(selected)} blurbs", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
