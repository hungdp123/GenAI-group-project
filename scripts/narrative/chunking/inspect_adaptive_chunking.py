"""Read one C4 block's recorded candidate scores and selection without model calls."""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261018_all16_v1")
    parser.add_argument("--validation-id", default="c4_v1")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--block-line", type=int, default=1)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", x) for x in (args.run_id, args.validation_id, args.ticker)) or args.block_line < 1:
        parser.error("Use valid IDs and a positive 1-based block line")
    path = Path.cwd() / "data/narrative/experiments/narrative_rag" / args.run_id / "validation" / args.validation_id / "arms/C4" / args.ticker.upper() / "candidate_scores.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value["block_line_1based"] == args.block_line:
                print(json.dumps(value, indent=2, ensure_ascii=False))
                return
    parser.error("Requested block line was not recorded")


if __name__ == "__main__":
    main()
