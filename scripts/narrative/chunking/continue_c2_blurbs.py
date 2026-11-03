"""Continue an existing C2 blurb build with disjoint company workers.

Stop the original build_c2_enriched process first; it does not use these locks.
Run two instances of this script with different --companies lists and the same
--output-root. Existing successful responses and COMPLETE indexes are reused.
The parent manifest, source chunks, prompt, model and output paths are unchanged.

Company ownership locks prevent duplicate workers. A short cache lock serializes
JSONL appends, never model calls. Each event is also checkpointed under
.workers/<TICKER>/blurb_cache.jsonl for recovery after an interrupted cache write.
Worker logs are mirrors: count usage from the root cache, not both copies.
Spend limits are per process. Failed requests need explicit --retry-incomplete.
No generation or embedding is performed with --prepare-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from scripts.narrative.chunking import build_c2_enriched as shared


def windows_sharing_error(exc: PermissionError) -> bool:
    # Windows can report a busy O_EXCL lock as EACCES instead of EEXIST.
    return os.name == "nt" and (
        getattr(exc, "winerror", None) in {5, 32, 33} or exc.errno == errno.EACCES
    )


def release_lock(path: Path) -> None:
    deadline = time.monotonic() + 10
    while True:
        try:
            path.unlink()
            return
        except PermissionError as exc:
            if not windows_sharing_error(exc) or time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


@contextmanager
def file_lock(path: Path, timeout: float = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    absent_denials = 0
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError) as exc:
            if isinstance(exc, PermissionError):
                if not windows_sharing_error(exc):
                    raise
                # The holder may remove the lock between os.open and exists().
                # Allow that race once; repeated denial with no lock is an actual
                # permission problem, not contention that should be hidden.
                absent_denials = 0 if path.exists() else absent_denials + 1
                if absent_denials > 1:
                    raise
            else:
                absent_denials = 0
            if time.monotonic() >= deadline:
                raise ValueError(f"Locked: {path}. Check the owning process; do not delete an active lock.") from exc
            time.sleep(0.05)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        release_lock(path)


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_event(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()


def snapshot(rows: list[dict]) -> tuple[dict[str, str], set[str]]:
    done, pending = {}, set()
    for row in rows:
        if row["status"] == "DONE":
            done[row["key"]] = row["blurb"]
            pending.discard(row["key"])
        else:
            pending.add(row["key"])
    return done, pending


def sync_worker_events(root: Path, ticker: str) -> tuple[dict[str, str], set[str]]:
    cache = root / "blurb_cache.jsonl"
    with file_lock(root / ".workers/cache.lock", timeout=10):
        rows = read_events(cache)
        known = {row.get("event_id") for row in rows if row.get("event_id")}
        latest = {row["key"]: row for row in rows}
        for row in read_events(root / ".workers" / ticker / "blurb_cache.jsonl"):
            if row["event_id"] in known:
                continue
            current = latest.get(row["key"], {})
            if current.get("status") == "DONE":
                if row["status"] == "DONE" and row["blurb"] != current["blurb"]:
                    raise ValueError("Conflicting successful worker/root blurbs; inspect the caches")
                # An old STARTED/FAILED event must never invalidate a saved success.
                if row["status"] != "DONE":
                    continue
            append_event(cache, row)
            known.add(row["event_id"])
            latest[row["key"]] = row
            rows.append(row)
        return snapshot(rows)


def generate_blurb(chunk: dict, args, config: dict, done: dict, pending: set) -> str:
    key = shared.blurb_key(chunk, config)
    if key in done:
        return done[key]
    if key in pending and not args.retry_incomplete:
        raise ValueError(f"Unresolved attempt for {chunk['chunk_id']}; inspect before --retry-incomplete")
    from finsight.llm.client import complete

    def append(row):
        event = {"key": key, "chunk_id": chunk["chunk_id"], "event_id": uuid4().hex,
                 "at_utc": datetime.now(timezone.utc).isoformat(), "worker_pid": os.getpid(), **row}
        # Save locally first: a successful response can be recovered if the shared append fails.
        append_event(args.output_root / ".workers" / chunk["company_ticker"] / "blurb_cache.jsonl", event)
        with file_lock(args.output_root / ".workers/cache.lock", timeout=10):
            append_event(args.output_root / "blurb_cache.jsonl", event)

    usage = {}
    append({"status": "STARTED"})

    def ask(prompt):
        response = complete(prompt=prompt, system=shared.BLURB_SYSTEM, model=config["blurb_model"],
                            max_tokens=config["blurb_max_tokens"])
        usage.update({name: getattr(response, name) for name in (
            "provider", "model", "input_tokens", "output_tokens", "cost_usd", "latency_ms",
        )})
        if response.finish_reason in {"length", "max_tokens"}:
            raise ValueError("Incomplete blurb response; inspect the token budget before retrying")
        return response.text

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
    parser.add_argument("--output-root", type=Path, default=shared.BASE / "contextual_blurb_all_v1")
    parser.add_argument("--cases", type=Path, default=shared.CASES, help="Must match the parent build's dataset hash")
    parser.add_argument("--companies", nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for local embedding")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-incomplete", action="store_true")
    return parser


def run(args) -> dict:
    workspace = Path.cwd().resolve()
    args.output_root = args.output_root.resolve()
    if not args.output_root.is_relative_to(workspace) or args.output_root == workspace:
        raise ValueError("Output root must be an existing build inside the workspace")
    if args.threads < 1 or args.progress_every < 1:
        raise ValueError("Thread count and progress interval must be positive")
    if not args.prepare_only and not args.resume:
        raise ValueError("Use --resume to continue an existing build")
    config = shared.read_json(args.output_root / "enrichment_configuration.json")
    if config.get("protocol") != shared.PROTOCOL or config.get("mode") != "blurb":
        raise ValueError("Expected a query-selected build_c2_enriched blurb manifest")
    for field, path in (
        ("builder_sha256", Path(shared.__file__)),
        ("context_helper_sha256", Path(shared.context_prefix.__code__.co_filename)),
        ("blurb_helper_sha256", Path(shared.write_blurb.__code__.co_filename)),
        ("cases_sha256", args.cases),
    ):
        if config.get(field) != shared.file_sha(path):
            raise ValueError(f"Parent build changed ({field}); do not mix source/configuration revisions")
    from finsight.config import get_settings

    if get_settings().llm_provider != config["provider"]:
        raise ValueError("Shared provider differs from the parent build")
    tickers = [ticker.upper() for ticker in args.companies]
    if len(tickers) != len(set(tickers)) or any(ticker not in config["sources"] for ticker in tickers):
        raise ValueError("Companies must be unique members of the parent build")
    signature = shared.digest(config)
    selection = shared.read_json(args.output_root / "selected_passages.json")
    if selection["configuration_sha256"] != signature:
        raise ValueError("Shortlist differs from the parent configuration")
    args.mode, args.embed = "blurb", config["embed"]
    sources = {ticker: shared.load_source(workspace, Path(config["source_root"]), ticker, "blurb")
               for ticker in tickers}
    for ticker, source in sources.items():
        if source["hashes"] != config["sources"][ticker] or source["model"] != config["embedding_model"]:
            raise ValueError(f"Source metadata/index changed for {ticker}")
        ids = {chunk["chunk_id"] for chunk in source["payload"]["chunks"]}
        if not set(selection["selected_chunk_ids"].get(ticker, [])) <= ids:
            raise ValueError(f"Unknown selected chunks for {ticker}")
    with file_lock(args.output_root / ".workers/cache.lock", timeout=10):
        done, pending = snapshot(read_events(args.output_root / "blurb_cache.jsonl"))
    plan = []
    for ticker, source in sources.items():
        selected = set(selection["selected_chunk_ids"].get(ticker, []))
        keys = {shared.blurb_key(chunk, config) for chunk in source["payload"]["chunks"]
                if chunk["chunk_id"] in selected}
        complete = shared.already_built(args.output_root / ticker, source, signature, args.embed)
        plan.append({"company": ticker, "selected_passages": len(selected), "index_complete": complete,
                     "new_prompt_keys": len(keys - done.keys()) if not complete else 0,
                     "unresolved_prompt_keys": len(keys & pending)})
    print(json.dumps({"provider": config["provider"], "model": config["blurb_model"],
                      "companies": plan, "output_root": str(args.output_root)}, indent=2), flush=True)
    if args.prepare_only:
        return {"status": "PREPARED", "companies": plan}
    backend = None
    built = []
    for ticker, source in sources.items():
        with file_lock(args.output_root / ".workers/locks" / f"{ticker}.lock"):
            done, pending = sync_worker_events(args.output_root, ticker)
            if shared.already_built(args.output_root / ticker, source, signature, args.embed):
                print(f"Reusing completed {ticker}", flush=True)
                continue
            selected = set(selection["selected_chunk_ids"].get(ticker, []))
            keys = {shared.blurb_key(chunk, config) for chunk in source["payload"]["chunks"]
                    if chunk["chunk_id"] in selected}
            if keys & pending and not args.retry_incomplete:
                raise ValueError(f"Unresolved request for {ticker}; inspect before --retry-incomplete")
            shared.write_json(args.output_root / ".workers" / ticker / "worker.json", {
                "configuration_sha256": signature, "worker_sha256": shared.file_sha(Path(__file__)),
                "pid": os.getpid(), "company": ticker,
            })
            chunks, number = [], 0
            for chunk in source["payload"]["chunks"]:
                if chunk["chunk_id"] in selected:
                    blurb = generate_blurb(chunk, args, config, done, pending)
                    chunks.append(shared.prepend_blurb(chunk, blurb))
                    number += 1
                    if number % args.progress_every == 0 or number == len(selected):
                        print(f"{ticker}: {number}/{len(selected)} blurbs ready", flush=True)
                else:
                    chunks.append(dict(chunk))
            if args.embed and backend is None:
                import torch
                from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend

                torch.set_num_threads(args.threads)
                backend = SentenceTransformersBackend(config["embedding_model"])
            shared.publish(args, ticker, source, chunks, backend, signature, sorted(selected), config)
            built.append(ticker)
            print(f"Built {ticker}: {len(chunks)} preserved chunks, {len(selected)} blurbs", flush=True)
    return {"status": "COMPLETE", "built_companies": built, "output_root": str(args.output_root)}


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
