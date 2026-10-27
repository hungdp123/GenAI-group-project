"""Run a C1-C4 retrieval/RAG matrix; generation uses local Ollama by default."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path

from scripts.narrative.evaluation.evaluate_rag_pilot import load_cases, read_json, write_json
from scripts.narrative.evaluation.model_options import add_model_options, configure_cli_model, describe_cli_model
from scripts.narrative.evaluation.resume import archive_state, matrix_compatible
from scripts.narrative.evaluation.run_rag_pilot import run as run_configuration, sha


def run(args: argparse.Namespace) -> dict:
    cases = load_cases(args.cases)
    count = len(cases) if args.all_cases else (args.limit if args.limit is not None else 2)
    if not cases or not 1 <= count <= len(cases) or args.threads < 1:
        raise ValueError("Require available cases, a positive limit within the CSV, and threads >= 1")
    if len(set(args.strategies)) != len(args.strategies) or len(set(args.backends)) != len(args.backends):
        raise ValueError("Strategies and backends must be unique")
    pairs = list(itertools.product(args.strategies, args.backends))
    retrieval_only = getattr(args, "retrieval_only", False)
    retry_errors = getattr(args, "retry_generation_errors", False)
    if retry_errors and (not args.resume or retrieval_only):
        raise ValueError("--retry-generation-errors requires --resume and a full RAG run")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output exists; use --resume with identical settings or a new output folder")
    configuration = {
        "dataset": str(args.cases.resolve()), "dataset_sha256": sha(args.cases),
        "selected_case_ids": [c["id"] for c in cases[:count]],
        "case_count_per_configuration": count, "strategies": args.strategies, "backends": args.backends,
        "configurations": [{"strategy": arm, "backend": backend,
                            "folder": f"configurations/{arm}/{backend}"} for arm, backend in pairs],
        "processed_root": str(args.processed_root.resolve()), "build_version": args.build_version,
        "top_k": args.top_k, "candidate_k": args.candidate_k, "threads": args.threads,
        "max_tokens": args.max_tokens, "ollama_model": args.ollama_model,
        "ollama_host": args.ollama_host if args.ollama_model else None,
        "ollama_context": args.ollama_context if args.ollama_model else None,
        "rerank_overflow": args.rerank_overflow, "source_priority": "P0",
        "retrieval_only": retrieval_only,
        "runner_sha256": sha(Path(__file__)),
    }
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(
            "Matrix output is locked; check whether another runner or interrupted attempt owns it"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        manifest = output / "matrix_configuration.json"
        previous = None
        if manifest.exists():
            previous = read_json(manifest)
            if not matrix_compatible(previous, configuration):
                raise ValueError("Matrix settings/dataset/code changed; select a new output version")
        models = {}
        jobs = []
        for arm, backend in pairs:
            jobs.append(argparse.Namespace(
                cases=args.cases, limit=count,
                corpus_root=args.processed_root / arm / args.build_version,
                backend=backend, top_k=args.top_k, candidate_k=args.candidate_k,
                max_tokens=args.max_tokens, ollama_model=args.ollama_model,
                ollama_host=args.ollama_host, ollama_context=args.ollama_context,
                rerank_overflow=args.rerank_overflow,
                output_dir=output / "configurations" / arm / backend,
                resume=True, prepare_only=True, model_cache=models,
                retrieval_only=retrieval_only,
                retry_generation_errors=retry_errors,
            ))
        # Validate every requested dataset/build before the first model call.
        for job in jobs:
            print(f"Preflight {job.corpus_root.parent.name}/{job.backend}: {count} cases", flush=True)
            run_configuration(job)
        if previous != configuration:
            if previous is not None:
                archive_state(output, previous)
            write_json(manifest, configuration)
        report = {"status": "PREFLIGHT_PASSED", "case_count_per_configuration": count,
                  "configuration_count": len(jobs), "planned_case_runs": count * len(jobs),
                  "reports": [], "ragas": "NOT_RUN"}
        report["retrieval_only"] = retrieval_only
        # A previous completed report must not describe the newly extended scope as finished.
        write_json(output / "matrix_report.json", report)
        if args.preflight_only:
            print("Preflight passed; no retrieval or model calls made. Use --resume to execute this matrix.")
            return report
        if any(backend != "bm25" for backend in args.backends):
            import torch

            torch.set_num_threads(args.threads)
        for job in jobs:
            job.prepare_only = False
            part = run_configuration(job)
            report["reports"].append({"strategy": job.corpus_root.parent.name, "backend": job.backend,
                                      "status": part["status"], "case_count": part["case_count"],
                                      "folder": str(job.output_dir.relative_to(output))})
            report["status"] = "IN_PROGRESS"
            write_json(output / "matrix_report.json", report)
        report["status"] = ("COMPLETE" if all(r["status"] == "COMPLETE" for r in report["reports"])
                            else "COMPLETE_WITH_ERRORS")
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "matrix_report.json", report)
        return report
    finally:
        lock.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path,
                        default=Path("data/narrative/evaluation/test_claims_v1/cases.csv"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--limit", type=int, help="First N CSV claims; defaults to two for a preview")
    group.add_argument("--all-cases", action="store_true", help="Explicitly select every CSV row")
    parser.add_argument("--strategies", nargs="+", choices=("C1", "C2", "C3", "C4", "contextual"),
                        default=["C1", "C2", "C3", "C4"])
    parser.add_argument("--backends", nargs="+", choices=("bm25", "dense", "hybrid", "r4"),
                        default=["bm25", "dense", "hybrid", "r4"])
    parser.add_argument("--processed-root", type=Path,
                        default=Path("data/narrative/processed/chunking_strategies"))
    parser.add_argument("--build-version", default="full_v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1500)
    add_model_options(parser)
    parser.add_argument("--rerank-overflow", choices=("error", "truncate"), default="error")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Save and score retrieval rankings without any LLM generation calls")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse saved results; allow increasing the case limit with identical inputs/settings")
    parser.add_argument("--retry-generation-errors", action="store_true",
                        help="With --resume, archive/retry GENERATION_ERROR cases; keep OK and INVALID_GENERATION")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_cli_model(args)
    print(describe_cli_model(args), flush=True)
    try:
        report = run(args)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(2, f"Comparison stopped: {type(exc).__name__}: {exc}\n")
    print(f"{report['status']}: {report['configuration_count']} configurations, "
          f"{report['planned_case_runs']} planned case runs")
    print("Score saved outputs with: python -m scripts.narrative.evaluation.evaluate_rag_comparison "
          f'--run-dir "{args.output_dir}"')


if __name__ == "__main__":
    main()
