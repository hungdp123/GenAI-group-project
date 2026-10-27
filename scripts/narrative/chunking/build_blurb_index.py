"""Add LLM context blurbs to short contextual table rows and rebuild the BM25 index.

Resumes from ``data/narrative/experiments/context_blurbs/progress.jsonl``.
Long prose passages keep the row-header search text already stored.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from finsight.agents.narrative.corpus.processing.chunk_context import BLURB_SYSTEM, prepend_blurb, write_blurb
from finsight.agents.narrative.corpus.processing.index import INDEX_SCHEMA_VERSION
from finsight.llm.client import complete

ROOT = Path("data/narrative/processed/chunking_strategies/contextual")
SOURCE = ROOT / "row512_v1"
DEST = ROOT / "blurb_v1"
PROGRESS = Path("data/narrative/experiments/context_blurbs/progress.jsonl")


def _needs_blurb(chunk: dict) -> bool:
    text = chunk.get("text") or ""
    return len(text) <= 400 and (" — " in text or " -- " in text)


def _load_done() -> dict[str, str]:
    done = {}
    if PROGRESS.is_file():
        for line in PROGRESS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["chunk_id"]] = row["context_blurb"]
    return done


def _ask(prompt: str) -> str:
    return complete(system=BLURB_SYSTEM, prompt=prompt, max_tokens=80).text


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-company", type=int, default=150)
    args = parser.parse_args()
    done = _load_done()
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    pending: list[dict] = []
    for ticker_dir in sorted(p for p in SOURCE.iterdir() if (p / "chunks.jsonl").is_file()):
        ranked = []
        for line in (ticker_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            if not _needs_blurb(chunk) or chunk["chunk_id"] in done:
                continue
            text = chunk["text"]
            rank = (0 if "%" in text or "Y/Y" in text else 1, len(text))
            ranked.append((rank, chunk))
        ranked.sort(key=lambda item: item[0])
        pending.extend(chunk for _, chunk in ranked[: args.per_company])
    print(f"pending {len(pending)} already {len(done)}", flush=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(write_blurb, chunk, _ask): chunk for chunk in pending}
            finished = 0
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    blurb = future.result()
                except Exception as exc:
                    print(f"stopped after {finished}: {type(exc).__name__}: {exc}", flush=True)
                    break
                done[chunk["chunk_id"]] = blurb
                handle.write(json.dumps({"chunk_id": chunk["chunk_id"], "context_blurb": blurb}, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
                if finished % 100 == 0:
                    print(f"blurbs {finished}/{len(pending)}", flush=True)
    for ticker_dir in sorted(p for p in SOURCE.iterdir() if (p / "chunks.jsonl").is_file()):
        records = []
        for line in (ticker_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            blurb = done.get(chunk["chunk_id"])
            records.append(prepend_blurb(chunk, blurb) if blurb else chunk)
        out = DEST / ticker_dir.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "chunks.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
        )
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "corpus_fingerprint": "blurb_v1",
            "chunking": {"strategy": "contextual-blurb", "token_limit": 512},
            "chunk_count": len(records),
            "duplicates_excluded": 0,
            "chunks": records,
        }
        index = out / "indexes"
        index.mkdir(parents=True, exist_ok=True)
        (index / "index.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"indexed {ticker_dir.name} {len(records)}", flush=True)
    print("BLURBS_DONE", flush=True)


if __name__ == "__main__":
    main()
