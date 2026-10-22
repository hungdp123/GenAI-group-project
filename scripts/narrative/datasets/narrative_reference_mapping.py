"""Source-position reference mapping shared by fixed and adaptive validation."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from scripts.narrative.datasets.prepare_narrative_labels import write_jsonl


def source_intervals(source_map: dict) -> list[list[int]]:
    return source_map.get("source_char_spans", [[source_map["source_char_start"], source_map["source_char_end"]]])


def covers(text: str, start: int, end: int, intervals: list[list[int]]) -> bool:
    """Cover non-whitespace source characters; gaps cannot be hidden by a bbox."""
    cursor = start
    for a, z in sorted(intervals):
        a, z = max(a, start), min(z, end)
        if z <= cursor:
            continue
        if a > cursor and text[cursor:a].strip():
            return False
        cursor = max(cursor, z)
    return not text[cursor:end].strip()


def map_references(*, rows: list[dict], sources: dict, source_maps: list[dict],
                   all_questions: dict, all_evidence: dict, output: Path,
                   arm: str, ticker: str) -> Counter:
    by_line = defaultdict(list)
    for m in source_maps:
        by_line[m["block_line_1based"]].append(m)
    counts = Counter()
    for split in ("dev", "heldout"):
        mapped, ambiguous, bundles = [], [], []
        for e in (e for e in all_evidence[split] if e["document_id"] in sources):
            line = e["block_line_1based"]
            b = rows[line - 1]
            a, z = e["char_start"], e["char_end"]
            correct = (b["document_id"] == e["document_id"] and b["text"][a:z] == e["quote"]
                       and hashlib.sha256(b["text"].encode()).hexdigest() == e["block_text_sha256"]
                       and sources[e["document_id"]]["sha256"] == e["source_sha256"])
            if not correct:
                ambiguous.append({"evidence_label_id": e["evidence_label_id"], "reason": "anchor/source mismatch; review required"})
                continue
            intersecting = [m for m in by_line[line] if any(x < z and y > a for x, y in source_intervals(m))]
            contained = [m["chunk_id"] for m in intersecting if covers(b["text"], a, z, source_intervals(m))]
            complete = covers(b["text"], a, z, [r for m in intersecting for r in source_intervals(m)])
            status = "contained_in_one_or_more_chunks" if contained else "requires_chunk_bundle" if complete else "unresolved"
            counts[status] += 1
            mapped.append({"evidence_label_id": e["evidence_label_id"], "query_id": e["query_id"],
                "source_sha256": e["source_sha256"], "block_line_1based": line,
                "source_char_start": a, "source_char_end": z, "status": status,
                "fully_containing_chunk_ids": contained, "overlapping_chunk_ids": [m["chunk_id"] for m in intersecting],
                "coverage_complete": complete, "ambiguity": False,
                "note": "Exact source positions; all valid aliases retained; no nearest-text selection."})
            if not complete:
                ambiguous.append({"evidence_label_id": e["evidence_label_id"], "reason": "incomplete source-span coverage"})
        by_id = {m["evidence_label_id"]: m for m in mapped}
        for q in (q for q in all_questions[split] if q["ticker"] == ticker):
            members = [by_id[e] for e in q["evidence_label_ids"] if e in by_id]
            single = set(members[0]["fully_containing_chunk_ids"]) if members else set()
            for m in members[1:]:
                single.intersection_update(m["fully_containing_chunk_ids"])
            bundles.append({"query_id": q["query_id"], "required_evidence_label_ids": q["evidence_label_ids"],
                "all_spans_mapped": bool(members) and len(members) == len(q["evidence_label_ids"]) and all(m["coverage_complete"] for m in members),
                "chunk_ids_covering_entire_bundle_individually": sorted(single),
                "candidate_chunk_ids": sorted({c for m in members for c in m["overlapping_chunk_ids"]}),
                "multiple_chunks_required": not bool(single), "semantic_label_status": "unreviewed_draft"})
        folder = output / "mappings" / split / arm / ticker
        write_jsonl(folder / "evidence_mapping.jsonl", mapped)
        write_jsonl(folder / "reference_bundles.jsonl", bundles)
        write_jsonl(folder / "ambiguity_queue.jsonl", ambiguous)
        counts["ambiguity_queue"] += len(ambiguous)
        counts["question_bundles"] += len(bundles)
        counts["complete_question_bundles"] += sum(m["all_spans_mapped"] for m in bundles)
    return counts
