"""Build C2 metadata + blurb indexes with Ollama, without a shared/paid LLM.

Default scope: union of BM25/dense candidate passages for every selected claim
type in --cases. With --all-chunks, no cases file or query shortlist is read.
Use a separate output root for each model/scope/settings combination.

--prepare-only writes the plan without contacting Ollama. Query-selected dense
shortlisting still needs the source embedding model; all-chunk previews do not.
Completed responses and actual Ollama token counts are saved in blurb_cache.jsonl.
Only one builder should write to an output root. No automatic model downloads,
retry of unfinished requests, or fallback to a paid API are performed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from scripts.narrative.chunking import build_c2_enriched as shared


PROTOCOL = "c2_ollama_blurbs_v1"


def normalize_host(host: str) -> str:
    url = urllib.parse.urlsplit(host)
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path.rstrip("/") not in {"", "/api"}):
        raise ValueError("Ollama host must be an http(s) server root or /api URL, without credentials")
    return urllib.parse.urlunsplit((url.scheme, url.netloc, "", "", "")).rstrip("/")


def complete_ollama(prompt: str, configuration: dict, timeout: float) -> dict:
    """Return text and usage from a plain-text, non-streaming Ollama chat response."""
    body = {
        "model": configuration["blurb_model"],
        "messages": [{"role": "system", "content": shared.BLURB_SYSTEM},
                     {"role": "user", "content": prompt}],
        "stream": False, "think": False,
        "options": {"num_predict": configuration["blurb_max_tokens"],
                    "num_ctx": configuration["ollama_context"], "temperature": 0},
    }
    request = urllib.request.Request(
        configuration["ollama_host"] + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama request failed; check that the server is running and the model is installed"
        ) from exc
    if result.get("error"):
        raise RuntimeError(f"Ollama returned an error: {result['error']}")
    message = result.get("message") or {}
    reported = all(isinstance(result.get(key), int) and result[key] >= 0
                   for key in ("prompt_eval_count", "eval_count"))
    return {
        "text": message.get("content", ""), "done": result.get("done") is True,
        "thinking_present": bool(message.get("thinking")),
        "usage": {
            "provider": "ollama", "model": configuration["blurb_model"],
            "input_tokens": int(result.get("prompt_eval_count") or 0),
            "output_tokens": int(result.get("eval_count") or 0),
            "token_usage_reported": reported, "cost_usd": 0.0,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "finish_reason": result.get("done_reason"),
        },
    }


def blurb_key(chunk: dict, configuration: dict) -> str:
    return shared.digest({
        "provider": "ollama", "model": configuration["blurb_model"],
        "host": configuration["ollama_host"], "context": configuration["ollama_context"],
        "max_tokens": configuration["blurb_max_tokens"], "temperature": 0, "think": False,
        "system": shared.BLURB_SYSTEM, "prompt": shared.blurb_prompt(chunk),
    })


def generate_blurb(chunk: dict, args, configuration: dict, cache: Path,
                   done: dict[str, str], pending: set[str]) -> str:
    key = blurb_key(chunk, configuration)
    if key in done:
        return done[key]
    if key in pending and not args.retry_incomplete:
        raise ValueError(f"Unresolved Ollama attempt for {chunk['chunk_id']}; review {cache} before --retry-incomplete")

    def append(row):
        with cache.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "chunk_id": chunk["chunk_id"],
                                     "at_utc": datetime.now(timezone.utc).isoformat(), **row},
                                    ensure_ascii=False) + "\n")

    usage = {}
    append({"status": "STARTED"})

    def ask(prompt):
        response = complete_ollama(prompt, configuration, args.ollama_timeout)
        usage.update(response["usage"])
        if not response["done"] or usage["finish_reason"] in {"length", "max_tokens"}:
            raise ValueError("Incomplete Ollama response; inspect the context and output-token limits")
        text = response["text"]
        if not isinstance(text, str) or not text.strip():
            detail = " (thinking output only)" if response["thinking_present"] else ""
            raise ValueError(f"Ollama returned no final blurb{detail}; use an installed non-thinking/instruct model")
        return text

    try:
        blurb = shared.write_blurb(chunk, ask)
    except BaseException as exc:
        append({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "usage": usage})
        pending.add(key)
        raise
    append({"status": "DONE", "blurb": blurb, "usage": usage})
    done[key] = blurb
    pending.discard(key)
    return blurb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=shared.BASE / "contextual_metadata_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help=f"Default in query-selected mode: {shared.CASES}")
    parser.add_argument("--case-ranges", nargs="+", metavar="START-END")
    parser.add_argument("--all-chunks", action="store_true", help="Enrich every source chunk, without claims")
    parser.add_argument("--selection-backends", nargs="+", choices=("bm25", "dense"), default=["bm25", "dense"])
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--ollama-model", required=True, help="Exact model name from ollama list")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-context", type=int, default=8192)
    parser.add_argument("--ollama-timeout", type=float, default=900)
    parser.add_argument("--blurb-max-tokens", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--embed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="Explicitly retry unfinished/failed local requests")
    return parser


def run(args) -> None:
    workspace = Path.cwd().resolve()
    args.source_root, args.output_root = args.source_root.resolve(), args.output_root.resolve()
    args.mode = "blurb"
    host = normalize_host(args.ollama_host)
    if (args.source_root.is_relative_to(args.output_root)
            or args.output_root.is_relative_to(args.source_root)
            or not args.output_root.is_relative_to(workspace)):
        raise ValueError("Use a separate output root inside the workspace; source/output must not overlap")
    if min(args.candidate_k, args.ollama_context, args.blurb_max_tokens, args.progress_every) < 1:
        raise ValueError("Candidate count, context, token budget and progress interval must be positive")
    if args.ollama_timeout <= 0 or not args.ollama_model.strip():
        raise ValueError("Provide a model name and a positive request timeout")
    if args.all_chunks:
        if args.cases is not None or args.case_ranges:
            raise ValueError("--all-chunks does not accept --cases or --case-ranges")
        cases = []
        tickers = sorted(path.name for path in args.source_root.iterdir()
                         if path.is_dir() and re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", path.name))
    else:
        args.cases = args.cases or shared.CASES
        cases = shared.selected_cases(args.cases, args.case_ranges)
        tickers = sorted({case["company"] for case in cases})
    if not tickers:
        raise ValueError("No source companies selected")
    sources = {ticker: shared.load_source(workspace, args.source_root, ticker, "blurb") for ticker in tickers}
    models = {source["model"] for source in sources.values()}
    if len(models) != 1:
        raise ValueError("Selected source companies must use the same embedding model")
    configuration = {
        "protocol": PROTOCOL, "mode": "blurb", "provider": "ollama", "blurb_model": args.ollama_model,
        "ollama_host": host, "ollama_context": args.ollama_context, "blurb_max_tokens": args.blurb_max_tokens,
        "scope": "all_chunks" if args.all_chunks else "query_selected", "claim_independent": args.all_chunks,
        "source_root": str(args.source_root), "sources": {t: s["hashes"] for t, s in sources.items()},
        "embedding_model": next(iter(models)), "embed": args.embed,
        "builder_sha256": shared.file_sha(Path(__file__)),
        "shared_builder_sha256": shared.file_sha(Path(shared.__file__)),
        "blurb_helper_sha256": shared.file_sha(Path(shared.write_blurb.__code__.co_filename)),
    }
    if not args.all_chunks:
        configuration.update(cases_sha256=shared.file_sha(args.cases), selected_case_ids=[c["id"] for c in cases],
                             candidate_k=args.candidate_k, selection_backends=args.selection_backends)
    signature = shared.digest(configuration)
    manifest = args.output_root / "enrichment_configuration.json"
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.resume or not manifest.is_file() or shared.read_json(manifest) != configuration:
            raise ValueError("Existing/changed build: use --resume with identical settings or a new output root")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        shared.write_json(manifest, configuration)
    cache = args.output_root / "blurb_cache.jsonl"
    done, pending = shared.cache_entries(cache)
    if pending and not args.prepare_only and not args.retry_incomplete:
        raise ValueError(f"Unresolved Ollama attempts; review {cache} before --retry-incomplete")
    backend = None
    needs_selection_vectors = not args.all_chunks and "dense" in args.selection_backends
    if needs_selection_vectors or (args.embed and not args.prepare_only):
        from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend

        backend = SentenceTransformersBackend(next(iter(models)))
    if args.all_chunks:
        chosen = {ticker: [c["chunk_id"] for c in source["payload"]["chunks"]]
                  for ticker, source in sources.items()}
    else:
        chosen = shared.shortlist(args, cases, backend, args.output_root, signature)
    completed = set()
    new_keys = set()
    for ticker, source in sources.items():
        ids = {c["chunk_id"] for c in source["payload"]["chunks"]}
        selected = set(chosen.get(ticker, []))
        if not selected <= ids:
            raise ValueError(f"{ticker}: shortlist contains unknown chunk IDs")
        if shared.already_built(args.output_root / ticker, source, signature, args.embed):
            completed.add(ticker)
            continue
        new_keys.update(blurb_key(c, configuration) for c in source["payload"]["chunks"]
                        if c["chunk_id"] in selected and blurb_key(c, configuration) not in done)
    total = sum(len(ids) for ids in chosen.values())
    plan = {
        "configuration_sha256": signature, "provider": "ollama", "model": args.ollama_model,
        "scope": configuration["scope"], "companies": tickers, "cases": len(cases),
        "selected_blurb_passages": total, "new_blurb_requests_needed": len(new_keys),
        "unresolved_prior_attempts": len(pending), "completed_company_indexes": sorted(completed),
        "note": "Ollama only; no shared LLM or paid-API fallback. Local inference API cost is recorded as zero.",
    }
    shared.write_json(args.output_root / "build_plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.prepare_only:
        return
    processed = 0
    for ticker, source in sources.items():
        selected = set(chosen.get(ticker, []))
        if ticker in completed:
            processed += len(selected)
            print(f"Reusing completed {ticker}", flush=True)
            continue
        updated, company_done = [], 0
        for chunk in source["payload"]["chunks"]:
            if chunk["chunk_id"] in selected:
                blurb = generate_blurb(chunk, args, configuration, cache, done, pending)
                updated.append(shared.prepend_blurb(chunk, blurb))
                company_done += 1
                processed += 1
                if company_done % args.progress_every == 0 or company_done == len(selected):
                    print(f"{ticker}: {company_done}/{len(selected)} blurbs; total {processed}/{total}", flush=True)
            else:
                updated.append(dict(chunk))
        shared.publish(args, ticker, source, updated, backend, signature, sorted(selected), configuration)
        print(f"Built {ticker}: {len(updated)} preserved chunks, {len(selected)} blurbs", flush=True)
    shared.write_json(args.output_root / "ollama_build_report.json", {
        "status": "COMPLETE", "configuration_sha256": signature,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "provider": "ollama",
        "model": args.ollama_model, "scope": configuration["scope"],
        "companies": tickers, "selected_blurb_passages": total,
    })


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
