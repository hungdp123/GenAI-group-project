"""Enrich every C2 metadata chunk with a claim-independent context blurb.

No cases file, claim queries, retrieval shortlist or gold annotations are read.
Run --prepare-only to inspect counts without model calls or embedding loads.
Actual builds checkpoint each response and preserve C2 source text, IDs, order,
source maps and embedding model through the existing enrichment helpers.

Use a separate output root from query-selected builds. Completed responses from
those builds may be reused with --reuse-cache when their exact prompt/model key
matches. Stop other builders using those caches before starting this build.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from scripts.narrative.chunking import build_c2_enriched as shared


PROTOCOL = "c2_corpus_blurbs_v1"


def company_names(root: Path, requested: list[str] | None) -> list[str]:
    names = requested if requested else [
        path.name for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    names = sorted(set(name.upper() for name in names))
    if not names:
        raise ValueError("No source companies found")
    for name in names:
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", name):
            raise ValueError(f"Invalid company directory: {name}")
        if not (root / name / "build_report.json").is_file():
            raise ValueError(f"Missing source company build report: {name}")
    return names


def reusable_blurbs(paths: list[Path]) -> dict[str, dict]:
    """Read only completed responses; retain provenance without double-counting cost."""
    reusable: dict[str, dict] = {}
    for path in paths:
        rows: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["key"]] = row
        for key, row in rows.items():
            if row["status"] != "DONE":
                continue
            blurb = row.get("blurb")
            if not isinstance(blurb, str) or not blurb.strip():
                raise ValueError(f"Invalid completed blurb in {path}")
            if key in reusable and reusable[key]["blurb"] != blurb:
                raise ValueError(f"Conflicting completed blurbs for the same prompt key in {path}")
            reusable[key] = {"blurb": blurb, "reused_from": str(path.resolve())}
    return reusable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=shared.BASE / "contextual_metadata_v1")
    parser.add_argument("--output-root", type=Path, default=shared.BASE / "contextual_blurb_corpus_v1")
    parser.add_argument("--companies", nargs="+", help="Optional explicit corpus scope; default: every source company")
    parser.add_argument("--blurb-model", help="Defaults to the configured shared model")
    parser.add_argument("--blurb-max-tokens", type=int, default=3000)
    parser.add_argument("--embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true", help="Save a plan; no generation or embedding calls")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true", help="Explicitly retry possibly billed API attempts")
    parser.add_argument("--reuse-cache", nargs="+", type=Path, default=[], help="Read-only completed blurb caches")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser


def run(args) -> None:
    workspace = Path.cwd().resolve()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.mode = "blurb"  # Shared publication contract; scope is recorded in this build's manifest.
    if (args.source_root.is_relative_to(args.output_root)
            or args.output_root.is_relative_to(args.source_root)
            or not args.output_root.is_relative_to(workspace)):
        raise ValueError("Use a separate output root inside the workspace; source/output must not overlap")
    if args.blurb_max_tokens < 1 or args.progress_every < 1:
        raise ValueError("Token budget and progress interval must be positive")
    tickers = company_names(args.source_root, args.companies)
    sources = {ticker: shared.load_source(workspace, args.source_root, ticker, "blurb") for ticker in tickers}
    models = {source["model"] for source in sources.values()}
    if len(models) != 1:
        raise ValueError("Selected source companies must use the same embedding model")
    from finsight.config import get_settings

    settings = get_settings()
    configuration = {
        "protocol": PROTOCOL, "mode": "blurb", "scope": "all_chunks",
        "claim_independent": True, "source_root": str(args.source_root), "companies": tickers,
        "sources": {ticker: source["hashes"] for ticker, source in sources.items()},
        "embedding_model": next(iter(models)), "embed": args.embed,
        "provider": settings.llm_provider, "blurb_model": args.blurb_model or settings.llm_model,
        "blurb_max_tokens": args.blurb_max_tokens,
        "builder_sha256": shared.file_sha(Path(__file__)),
        "shared_builder_sha256": shared.file_sha(Path(shared.__file__)),
        "context_helper_sha256": shared.file_sha(Path(shared.context_prefix.__code__.co_filename)),
        "blurb_helper_sha256": shared.file_sha(Path(shared.write_blurb.__code__.co_filename)),
    }
    signature = shared.digest(configuration)
    manifest = args.output_root / "enrichment_configuration.json"
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.resume or not manifest.is_file() or shared.read_json(manifest) != configuration:
            raise ValueError("Existing/changed build: use --resume with identical settings or a new output root")
    cache = args.output_root / "blurb_cache.jsonl"
    done, pending = shared.cache_entries(cache)
    if any(path.resolve() == cache for path in args.reuse_cache):
        raise ValueError("--reuse-cache must reference a different build's cache")
    reusable = reusable_blurbs(args.reuse_cache)
    completed_companies = {
        ticker for ticker, source in sources.items()
        if shared.already_built(args.output_root / ticker, source, signature, args.embed)
    }
    total = sum(len(source["payload"]["chunks"]) for source in sources.values())
    saved = reused = new = unresolved = 0
    for source in sources.values():
        for chunk in source["payload"]["chunks"]:
            key = shared.blurb_key(chunk, configuration)
            if key in done:
                saved += 1
            elif key in pending:
                unresolved += 1
            elif key in reusable:
                reused += 1
            else:
                new += 1
    plan = {
        "configuration_sha256": signature, "scope": "all_chunks", "claim_independent": True,
        "companies": tickers, "total_chunks": total, "saved_blurbs": saved,
        "reusable_blurbs": reused, "unresolved_prior_attempts": unresolved,
        "new_blurb_requests_needed": new,
        "completed_company_indexes": sorted(completed_companies),
        "note": "All source chunks are enriched. No claims, test rows or retrieval shortlist are used.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        shared.write_json(manifest, configuration)
    shared.write_json(args.output_root / "build_plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.prepare_only:
        return
    if pending and not args.retry_incomplete:
        raise ValueError(f"Unresolved prior API attempt; review {cache} before using --retry-incomplete")
    backend = None
    if args.embed:
        from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend

        backend = SentenceTransformersBackend(next(iter(models)))
    processed = 0
    for ticker, source in sources.items():
        chunks = source["payload"]["chunks"]
        if ticker in completed_companies:
            processed += len(chunks)
            print(f"Reusing completed {ticker}: {processed}/{total} corpus chunks", flush=True)
            continue
        updated = []
        for number, chunk in enumerate(chunks, 1):
            key = shared.blurb_key(chunk, configuration)
            if key not in done and key not in pending and key in reusable:
                imported = {"key": key, "chunk_id": chunk["chunk_id"], "status": "DONE",
                            **reusable[key], "usage": {}}
                with cache.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(imported, ensure_ascii=False) + "\n")
                done[key] = imported["blurb"]
            blurb = shared.generate_blurb(chunk, args, configuration, cache, done, pending)
            updated.append(shared.prepend_blurb(chunk, blurb))
            processed += 1
            if number % args.progress_every == 0 or number == len(chunks):
                print(f"{ticker}: {number}/{len(chunks)} blurbs; corpus {processed}/{total}", flush=True)
        shared.publish(args, ticker, source, updated, backend, signature,
                       [chunk["chunk_id"] for chunk in chunks], configuration)
        print(f"Built {ticker}: {len(chunks)} preserved chunks, all with blurbs", flush=True)
    shared.write_json(args.output_root / "corpus_build_report.json", {
        "status": "COMPLETE", "configuration_sha256": signature,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "claim_independent": True, "scope": "all_chunks", "companies": tickers, "chunks": total,
    })


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
