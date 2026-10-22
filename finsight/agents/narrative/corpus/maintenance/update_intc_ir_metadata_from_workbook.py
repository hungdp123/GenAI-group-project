"""Apply Intel IR period/date annotations from the local update-year workbook.

Rows are matched to existing official-IR manifest records by exact file stem.
Only workbook rows with a populated publication_date are date-verified here;
blank dates leave the document ineligible and retain any provisional estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path
from xml.sax.saxutils import escape
from xml.etree import ElementTree as ET


NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
EXCEL_EPOCH = date(1899, 12, 30)


def read_rows(workbook_path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(workbook_path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.findall(".//m:t", NS))
            for item in shared_root.findall("m:si", NS)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", NS):
        cells: dict[str, str] = {}
        for cell in row.findall("m:c", NS):
            ref = cell.attrib["r"]
            column = "".join(char for char in ref if char.isalpha())
            if cell.attrib.get("t") == "inlineStr":
                cells[column] = "".join(
                    node.text or "" for node in cell.findall("m:is/m:t", NS)
                )
                continue
            value_node = cell.find("m:v", NS)
            if value_node is None or value_node.text is None:
                cells[column] = ""
                continue
            value = value_node.text
            if cell.attrib.get("t") == "s":
                value = shared[int(value)]
            elif cell.attrib.get("s") == "1":
                value = (EXCEL_EPOCH + timedelta(days=float(value))).isoformat()
            cells[column] = value
        if cells:
            rows.append(cells)

    if not rows or [rows[0].get(col) for col in ("A", "B", "C", "D")] != [
        "file_name", "year", "quarter", "publication_date"
    ]:
        raise ValueError("Workbook must have file_name, year, quarter, publication_date columns in A:D.")
    return [
        {
            "file_name": row.get("A", "").strip(),
            "year": row.get("B", "").strip(),
            "quarter": row.get("C", "").strip().upper(),
            "publication_date": row.get("D", "").strip(),
        }
        for row in rows[1:]
        if row.get("A", "").strip()
    ]


def apply(
    workbook_path: Path,
    manifest_path: Path,
    workbook_output_path: Path | None = None,
    manifest_only: bool = False,
) -> tuple[int, int]:
    rows = read_rows(workbook_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = manifest.get("documents", [])
    def workbook_stem(path: str) -> str:
        stem = Path(path).name
        if stem.lower().endswith(".pdf"):
            stem = stem[:-4]
        return re.sub(r"^(?:\d{4}-\d{2}-\d{2}|\d{4})_", "date-unknown_", stem)

    by_stem = {
        workbook_stem(doc["local_path"]): doc
        for doc in documents
        if doc.get("company_ticker") == "INTC"
    }

    matches: list[tuple[dict[str, str], dict[str, object], Path, Path]] = []
    seen: set[str] = set()
    for row in rows:
        stem = workbook_stem(row["file_name"])
        if stem in seen:
            raise ValueError(f"Duplicate workbook file_name: {stem}")
        seen.add(stem)
        doc = by_stem.get(stem)
        if doc is None:
            raise ValueError(f"No exact INTC manifest match for workbook row: {stem}")
        source = Path(str(doc["local_path"]))
        if not source.exists():
            raise FileNotFoundError(source)
        source_stem = workbook_stem(source.as_posix())
        if source_stem != stem:
            raise ValueError(f"Workbook row does not normalize to current filename: {source}")
        if row["publication_date"]:
            date.fromisoformat(row["publication_date"])
            prefix = row["publication_date"]
        else:
            prefix = row["year"]
        is_pdf = source.read_bytes().startswith(b"%PDF")
        extension = ".pdf" if is_pdf else ""
        new_name = prefix + source_stem[len("date-unknown"): ] + extension
        target = source.with_name(new_name)
        if target.exists() and target.resolve() != source.resolve():
            raise FileExistsError(f"Rename would overwrite existing file: {target}")
        matches.append((row, doc, source, target))

    # Validate all targets before any filesystem or manifest mutation.
    targets = [target.resolve() for _, _, _, target in matches]
    if len(targets) != len(set(targets)):
        raise ValueError("Multiple workbook rows resolve to the same target path.")

    updated_dates = 0
    for row, doc, source, target in matches:
        digest_before = hashlib.sha256(source.read_bytes()).hexdigest()
        if source != target:
            source.rename(target)
            doc["local_path"] = target.as_posix()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest_before:
            raise RuntimeError(f"Content hash changed during rename: {target}")

        doc["period"] = f"FY{row['year']} {row['quarter']}"
        if row["publication_date"]:
            doc["publication_date"] = row["publication_date"]
            doc["date_precision"] = "day"
            doc["retrieval_eligible"] = True
            doc["estimated_publication_date"] = None
            doc["estimated_publication_date_basis"] = None
            doc["publication_date_basis"] = (
                "Manually reviewed from document content and recorded in update-year.xlsx."
            )
            updated_dates += 1

    manifest["review_required"] = [
        doc["document_id"] for doc in documents if not doc.get("retrieval_eligible")
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Make the workbook's file_name column point to the actual renamed files.
    # Replace only A2:A(last data row); all other cells/styles/package parts stay intact.
    if manifest_only:
        return len(matches), updated_dates
    workbook_target_path = workbook_output_path or workbook_path
    if workbook_output_path and workbook_output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing workbook output: {workbook_output_path}")
    xlsx_entry = "xl/worksheets/sheet1.xml"
    with zipfile.ZipFile(workbook_path, "r") as source_zip:
        sheet_xml = source_zip.read(xlsx_entry).decode("utf-8")
        updated_sheet = sheet_xml
        for row_number, (_, _, _, target) in enumerate(matches, start=2):
            cell_ref = f"A{row_number}"
            pattern = re.compile(rf'(<c\b(?=[^>]*\br="{cell_ref}")[^>]*>).*?</c>', re.DOTALL)
            inline_cell = f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(target.name)}</t></is></c>'
            updated_sheet, replacements = pattern.subn(lambda _match: inline_cell, updated_sheet)
            if replacements != 1:
                raise ValueError(f"Expected exactly one workbook cell {cell_ref}, found {replacements}.")

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="update-year-", suffix=".xlsx", dir=workbook_target_path.parent, delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
            with zipfile.ZipFile(temp_path, "w") as target_zip:
                for item in source_zip.infolist():
                    payload = (
                        updated_sheet.encode("utf-8")
                        if item.filename == xlsx_entry
                        else source_zip.read(item.filename)
                    )
                    target_zip.writestr(item, payload)
            with zipfile.ZipFile(temp_path, "r") as check_zip:
                if check_zip.testzip() is not None:
                    raise RuntimeError("Rewritten workbook failed ZIP integrity validation.")
                check_sheet = check_zip.read(xlsx_entry).decode("utf-8")
                for row_number, (_, _, _, target) in enumerate(matches, start=2):
                    if escape(target.name) not in check_sheet:
                        raise RuntimeError(f"Updated filename missing from workbook cell A{row_number}.")
            temp_path.replace(workbook_target_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    return len(matches), updated_dates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook", type=Path,
        default=Path("data/narrative/raw/INTC/official_ir/update-year.xlsx"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/narrative/raw/manifest_official_ir.json"),
    )
    parser.add_argument(
        "--output-workbook", type=Path,
        help="Write a corrected copy here instead of replacing the source workbook.",
    )
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="Apply workbook period/date metadata to the manifest without editing the workbook.",
    )
    args = parser.parse_args()
    matched, dated = apply(
        args.workbook, args.manifest, args.output_workbook, args.manifest_only
    )
    destination = "manifest only" if args.manifest_only else (args.output_workbook or args.workbook)
    print(
        f"Updated {matched} INTC reporting periods and {dated} verified dates; "
        f"workbook filename cells written to {destination}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
