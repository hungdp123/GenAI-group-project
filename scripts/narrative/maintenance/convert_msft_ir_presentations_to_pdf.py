"""Export official Microsoft IR PPTX decks to PDF and update their manifest paths.

Run from the repository root in an environment with Microsoft PowerPoint installed:
    python scripts/narrative/maintenance/convert_msft_ir_presentations_to_pdf.py
The original PPTX files are retained and recorded in each manifest entry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pythoncom
import win32com.client
from pypdf import PdfReader


ROOT = Path.cwd()
MANIFEST = ROOT / "data/narrative/raw/manifest_official_ir.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [
        row for row in manifest["documents"]
        if row.get("company_ticker") == "MSFT"
        and row.get("document_type") == "earnings_presentation"
        and Path(row["local_path"]).suffix.lower() == ".pptx"
    ]
    if len(records) != 11:
        raise SystemExit(f"Expected 11 Microsoft PPTX earnings decks; found {len(records)}")

    pythoncom.CoInitialize()
    powerpoint = win32com.client.DispatchEx("PowerPoint.Application")
    # PowerPoint rejects Visible=False; keep its window minimized during export.
    powerpoint.Visible = 1
    powerpoint.WindowState = 2
    powerpoint.DisplayAlerts = 0
    converted: list[tuple[dict, Path, Path]] = []
    try:
        for index, row in enumerate(records, start=1):
            source = ROOT / row["local_path"]
            if not source.is_file() or sha256(source) != row["sha256"].lower():
                raise SystemExit(f"PPTX missing or hash mismatch: {row['document_id']}")
            destination = source.with_suffix(".pdf")
            print(f"[{index}/{len(records)}] Exporting {source.name}", flush=True)
            presentation = powerpoint.Presentations.Open(str(source.resolve()), -1, 0, 0)
            try:
                presentation.SaveAs(str(destination.resolve()), 32, 0)
            finally:
                presentation.Close()
            reader = PdfReader(str(destination))
            if not reader.pages:
                raise SystemExit(f"PowerPoint produced an empty PDF: {destination}")
            converted.append((row, source, destination))
            print(f"    {len(reader.pages)} PDF pages", flush=True)
    finally:
        powerpoint.Quit()
        pythoncom.CoUninitialize()

    for row, source, destination in converted:
        row["original_local_path"] = source.relative_to(ROOT).as_posix()
        row["local_path"] = destination.relative_to(ROOT).as_posix()
        row["sha256"] = sha256(destination)
        row["source_conversion"] = "PowerPoint SaveAs PDF; source PPTX retained"
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated manifest paths and hashes for {len(converted)} converted decks.")


if __name__ == "__main__":
    main()
