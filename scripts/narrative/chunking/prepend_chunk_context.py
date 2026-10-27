"""Prepend an LLM context sentence to contextual row passages.

Example, from the repository root. ``--limit`` keeps the first matching rows
so a full company is not sent to the model by accident:

    python -m scripts.narrative.chunking.prepend_chunk_context --ticker AMZN --limit 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finsight.agents.narrative.corpus.processing.chunk_context import BLURB_SYSTEM, prepend_blurb, write_blurb
from finsight.llm.client import complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--build-version", default="row512_v1")
    parser.add_argument("--processed-root", type=Path, default=Path("data/narrative/processed"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    source = (
        args.processed_root / "chunking_strategies" / "contextual" / args.build_version / ticker / "chunks.jsonl"
    )
    if not source.is_file():
        parser.exit(2, f"Contextual chunks are missing: {source}\n")
    chosen = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        if chunk.get("content_type") == "table" or " — " in chunk.get("text", "") or "Y/Y" in chunk.get("text", ""):
            chosen.append(chunk)
        if len(chosen) == args.limit:
            break
    if len(chosen) < args.limit:
        for line in source.read_text(encoding="utf-8").splitlines():
            if len(chosen) == args.limit:
                break
            if not line.strip():
                continue
            chunk = json.loads(line)
            if chunk not in chosen:
                chosen.append(chunk)

    def ask(prompt: str) -> str:
        response = complete(system=BLURB_SYSTEM, prompt=prompt, max_tokens=120)
        return response.text

    records = [prepend_blurb(chunk, write_blurb(chunk, ask)) for chunk in chosen]
    output = args.output or Path("data/narrative/experiments/context_blurbs") / f"{ticker}_sample.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(json.dumps({"ticker": ticker, "chunks": len(records), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
