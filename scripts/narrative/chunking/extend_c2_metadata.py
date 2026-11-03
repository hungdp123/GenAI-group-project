"""Extend a C2 metadata build to additional cases/companies in the same folder.

Existing completed company indexes are validated and reused byte-for-byte.
Only missing companies are built, using the existing C2 metadata builder's
unchanged prefix/embedding implementation. No blurb generation is performed.

Preview (read-only)::

    python -m scripts.narrative.chunking.extend_c2_metadata \
        --output-root data/narrative/processed/chunking_strategies/C2/contextual_metadata_v1 \
        --cases data/narrative/evaluation/test_claims_v2/cases_combined.csv \
        --prepare-only --resume

Omit --prepare-only to build missing companies. No --case-ranges means all CSV
rows. After an extension, use this command for future metadata resumes in the
same folder. The original builder and existing blurb caches are not modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from scripts.narrative.chunking import build_c2_enriched as builder


EXTENSION_KEY = "metadata_extension"


def extension_plan(saved: dict, requested: dict) -> tuple[dict, dict]:
    """Permit only a scope superset with identical dataset, settings and sources."""
    previous = {k: v for k, v in saved.items() if k != EXTENSION_KEY}
    if previous.get("mode") != "metadata" or requested.get("mode") != "metadata":
        raise ValueError("Only metadata builds support same-folder scope extension")
    fixed_previous = {k: v for k, v in previous.items() if k not in {"selected_case_ids", "sources"}}
    fixed_requested = {k: v for k, v in requested.items() if k not in {"selected_case_ids", "sources"}}
    if fixed_previous != fixed_requested:
        raise ValueError("Dataset, code, model or build settings changed; use a new output root")
    old_ids, new_ids = previous["selected_case_ids"], requested["selected_case_ids"]
    if len(set(new_ids)) != len(new_ids) or [i for i in new_ids if i in set(old_ids)] != old_ids:
        raise ValueError("The new selection must include all previous cases in CSV order")
    for ticker, source in previous["sources"].items():
        if requested["sources"].get(ticker) != source:
            raise ValueError(f"Changed or removed source build for {ticker}; use a new output root")
    previous_signature = builder.digest(previous)
    extension = saved.get(EXTENSION_KEY)
    if extension:
        if (extension.get("configuration_sha256") != previous_signature
                or set(extension.get("company_configuration_sha256", {})) != set(previous["sources"])):
            raise ValueError("Metadata extension signatures do not match the current manifest")
        signatures = dict(extension["company_configuration_sha256"])
    else:
        signatures = {ticker: previous_signature for ticker in previous["sources"]}
    return previous, signatures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=builder.BASE / "contextual_metadata_v1")
    parser.add_argument("--source-root", type=Path, help="Must match the saved source root")
    parser.add_argument("--cases", type=Path, default=builder.CASES)
    parser.add_argument("--case-ranges", nargs="+", metavar="START-END")
    parser.add_argument("--embed", action=argparse.BooleanOptionalAction, default=None,
                        help="Must match the saved build setting; otherwise inherited")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and print the plan without writes/builds")
    parser.add_argument("--resume", action="store_true", help="Required when building missing companies")
    return parser


def run(args) -> None:
    workspace = Path.cwd().resolve()
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(workspace):
        raise ValueError("The existing output root must be inside the workspace")
    if not args.prepare_only and not args.resume:
        raise ValueError("Use --resume to extend an existing metadata build")
    manifest = args.output_root / "enrichment_configuration.json"
    saved = builder.read_json(manifest)
    previous = {k: v for k, v in saved.items() if k != EXTENSION_KEY}
    if previous.get("mode") != "metadata":
        raise ValueError("This command cannot extend a query-dependent blurb build")
    if previous.get("builder_sha256") != builder.file_sha(Path(builder.__file__)):
        raise ValueError("The original builder changed since this build; use a new output root")
    for key, function in (("context_helper_sha256", builder.context_prefix),
                          ("blurb_helper_sha256", builder.write_blurb)):
        if previous.get(key) != builder.file_sha(Path(function.__code__.co_filename)):
            raise ValueError("An enrichment helper changed; use a new output root")
    source_root = Path(previous["source_root"]).resolve()
    if args.source_root is not None and args.source_root.resolve() != source_root:
        raise ValueError("Source root differs from the saved metadata build")
    if source_root.is_relative_to(args.output_root) or args.output_root.is_relative_to(source_root):
        raise ValueError("Source and output roots must not overlap")
    if args.embed is not None and args.embed != previous["embed"]:
        raise ValueError("Keep the original vector-build setting when extending the scope")
    args.embed = previous["embed"]
    args.mode = "metadata"
    cases = builder.selected_cases(args.cases, args.case_ranges)
    tickers = sorted({case["company"] for case in cases})
    sources = {ticker: builder.load_source(workspace, source_root, ticker, "metadata") for ticker in tickers}
    if {source["model"] for source in sources.values()} != {previous["embedding_model"]}:
        raise ValueError("The source embedding model changed")
    requested = {
        **previous, "cases_sha256": builder.file_sha(args.cases),
        "selected_case_ids": [case["id"] for case in cases],
        "sources": {ticker: source["hashes"] for ticker, source in sources.items()},
    }
    previous, old_signatures = extension_plan(saved, requested)
    signature = builder.digest(requested)
    signatures, reuse, missing = {}, [], []
    for ticker, source in sources.items():
        target = args.output_root / ticker
        if target.exists():
            if ticker not in old_signatures:
                raise ValueError(f"Untracked company output at {target}; inspect it before extending")
            accepted = old_signatures[ticker]
            builder.already_built(target, source, accepted, args.embed)
            signatures[ticker] = accepted
            reuse.append(ticker)
        else:
            signatures[ticker] = signature
            missing.append(ticker)
    plan = {
        "mode": "metadata", "cases": len(cases), "companies": len(tickers),
        "reuse_completed_companies": reuse, "build_missing_companies": missing,
        "new_blurb_requests_needed": 0,
        "note": "Existing company files/signatures are preserved; only missing companies are built.",
    }
    import json

    print(json.dumps(plan, indent=2), flush=True)
    if args.prepare_only:
        return
    history = list(saved.get(EXTENSION_KEY, {}).get("history", []))
    if requested != previous:
        history.append({"at_utc": datetime.now(timezone.utc).isoformat(), "previous_configuration": previous})
    extended = {
        **requested,
        EXTENSION_KEY: {
            "schema_version": 1, "configuration_sha256": signature,
            "company_configuration_sha256": signatures,
            "history": history,
            "extender_sha256": builder.file_sha(Path(__file__)),
        },
    }
    # One atomic manifest write records the expanded scope and all accepted
    # old/new company signatures together, including unfinished new companies.
    builder.write_json(manifest, extended)
    builder.write_json(args.output_root / "build_plan.json", plan)
    backend = None
    if missing and args.embed:
        from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend

        backend = SentenceTransformersBackend(previous["embedding_model"])
    for ticker in missing:
        source = sources[ticker]
        chunks = [
            {**chunk, "search_text": f"{builder.context_prefix(chunk)}\n{chunk.get('search_text') or chunk['text']}"}
            for chunk in source["payload"]["chunks"]
        ]
        builder.publish(args, ticker, source, chunks, backend, signatures[ticker], [], requested)
        print(f"Built missing company {ticker}: {len(chunks)} preserved C2 chunks", flush=True)
    print(f"Metadata build covers {len(cases)} cases across {len(tickers)} companies.", flush=True)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
