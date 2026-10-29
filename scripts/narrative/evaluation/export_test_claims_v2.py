"""Write the retrieval CSV for test_claims_v2.

The workbook has two sheets. This reads Cases only, renames the claim column,
and copies cut_off from column M. Excel stores that cell as a day count, so it
is written as YYYY-MM-DD. The partially supported row is left out: the retrieval
runner scores supported, contradicting, and unsupported cases. Run from the
repository root:

    python -m scripts.narrative.evaluation.export_test_claims_v2
"""

from __future__ import annotations

import csv
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

from scripts.narrative.evaluation.evaluate_rag_pilot import source_anchors


WORKBOOK = Path("data/narrative/evaluation/test_claims_v2/cases_combined.xlsx")
OUTPUT = Path("data/narrative/evaluation/test_claims_v2/cases.csv")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
EXCEL_EPOCH = datetime(1899, 12, 30)
COLUMNS = (
    "id", "claims", "video", "type", "company", "source_doc", "gold_answer", "gold_snippet",
    "distractor_snippet", "distractor_strategies", "expected_behavior", "rationale", "cut_off",
)
RENAMES = {
    "influencer claims": "claims",
    "video containing the influencer claim": "video",
}
SKIPPED_TYPE = "partially supported"


def _column_row(ref: str) -> tuple[int, int]:
    match = re.match(r"([A-Z]+)(\d+)", ref)
    if match is None:
        raise ValueError(f"Unrecognised cell {ref}")
    column = 0
    for character in match.group(1):
        column = column * 26 + ord(character) - 64
    return column, int(match.group(2))


def _sheet_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as workbook:
        strings: list[str] = []
        shared = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        for item in shared.findall(f"{NS}si"):
            strings.append("".join(text.text or "" for text in item.iter(f"{NS}t")))
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    grid: dict[int, dict[int, str]] = {}
    for cell in sheet.iter(f"{NS}c"):
        ref = cell.attrib.get("r")
        if not ref:
            continue
        column, row = _column_row(ref)
        value = cell.find(f"{NS}v")
        if value is None or value.text is None:
            text = ""
        elif cell.attrib.get("t") == "s":
            text = strings[int(value.text)]
        else:
            text = value.text
        grid.setdefault(row, {})[column] = text
    header = {grid[1][column]: column for column in grid[1]}
    rows = []
    for number in sorted(row for row in grid if row != 1):
        record = {name: grid[number].get(column, "") for name, column in header.items()}
        rows.append({RENAMES.get(name, name): value for name, value in record.items()})
    return rows


def cutoff_date(value: str) -> str:
    """Column M is an Excel day count. A cell already stored as a date is kept."""
    text = (value or "").strip()
    match = ISO_DATE.fullmatch(text)
    if match:
        return match.group(0)
    try:
        serial = float(text)
    except ValueError as exc:
        raise ValueError(f"Unrecognised cutoff {value!r}") from exc
    if not serial.is_integer():
        raise ValueError(f"Unrecognised cutoff {value!r}")
    return (EXCEL_EPOCH + timedelta(days=int(serial))).date().isoformat()


def retrieval_rows(records: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    kept: list[dict[str, str]] = []
    skipped: list[str] = []
    for record in records:
        if record.get("type") == SKIPPED_TYPE:
            skipped.append(record.get("id", ""))
            continue
        if not str(record.get("cut_off", "")).strip():
            raise ValueError(f"Missing evidence cutoff for {record.get('id')}")
        record["cut_off"] = cutoff_date(record["cut_off"])
        if record["type"] != "unsupported" and not source_anchors(record.get("source_doc", "")):
            raise ValueError(f"Missing source anchor for {record.get('id')}")
        missing = [name for name in COLUMNS if name not in record]
        if missing:
            raise ValueError(f"{record.get('id')} is missing {', '.join(missing)}")
        kept.append({name: record[name] for name in COLUMNS})
    if len({row["id"] for row in kept}) != len(kept):
        raise ValueError("CSV case IDs must be unique")
    return kept, skipped


def write_cases(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows, skipped = retrieval_rows(_sheet_rows(WORKBOOK))
    write_cases(rows, OUTPUT)
    print(f"Wrote {len(rows)} cases to {OUTPUT}")
    if skipped:
        print("Left out: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
