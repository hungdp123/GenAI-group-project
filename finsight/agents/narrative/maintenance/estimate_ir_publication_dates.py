"""Populate provisional publication-date estimates on quarter-labelled IR rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finsight.agents.narrative.maintenance.publication_dates import (
    build_quarter_period_lookup,
    estimate_publication_date,
)


def update(ir_path: Path, core_path: Path) -> tuple[int, int]:
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    core = json.loads(core_path.read_text(encoding="utf-8"))
    periods = build_quarter_period_lookup(core)
    updated = 0
    for doc in ir.get("documents", []):
        if doc.get("publication_date"):
            estimate, basis = None, None
        else:
            estimate, basis = estimate_publication_date(
                doc.get("company_ticker", ""),
                doc.get("period") or doc.get("link_text", ""),
                doc.get("local_path", ""),
                periods,
            )
        doc["estimated_publication_date"] = estimate
        doc["estimated_publication_date_basis"] = basis
        if estimate:
            updated += 1
            doc["retrieval_eligible"] = True
            doc["publication_date_is_estimated"] = True
        elif doc.get("publication_date"):
            doc["publication_date_is_estimated"] = False
    ir_path.write_text(json.dumps(ir, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return updated, len(ir.get("documents", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-manifest", type=Path, default=Path("data/narrative/raw/manifest_official_ir.json"))
    parser.add_argument("--core-manifest", type=Path, default=Path("data/narrative/raw/manifest_core_sec.json"))
    args = parser.parse_args()
    updated, total = update(args.ir_manifest, args.core_manifest)
    print(f"Added provisional estimates to {updated} of {total} IR records; estimated rows enabled for retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
