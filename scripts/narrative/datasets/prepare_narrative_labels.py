"""Prepare source-span label drafts without rankings, models, or canonical writes.

New claims and semantic labels require review. Heldout contents are written only
to their split files, never to progress reports or the development review document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from finsight.agents.narrative.corpus.processing.corpus import file_sha256
from scripts.narrative.experiments.prepare_narrative_experiment import identity, write_json


METRIC = re.compile(r"\b(revenues?|net sales|gross margin|operating (?:income|margin|loss)|net (?:income|loss)|earnings per share|cash flow|subscriptions?|subscribers?|daily active|segment|capital expenditures)\b", re.I)
VALUE = re.compile(r"(?:\$\s*\d[\d,.]*|\b\d[\d,.]*\s*(?:%|percent|billion|million|thousand)\b|\b\d[\d,.]*%)", re.I)
PERIOD = re.compile(r"\b(?:20\d{2}|Q[1-4]|fiscal|quarter|year ended|months ended)\b", re.I)
BASIS = re.compile(r"\b(?:non[ -]?GAAP|GAAP|IFRS|constant currency|adjusted)\b", re.I)
PRESENTATIONS = {"earnings_presentation", "investor_presentation", "investor_day_presentation", "special_presentation"}
FAMILIES = {"10-K": "filing", "10-Q": "filing", "20-F": "filing", "8-K": "event_filing", "6-K": "event_filing", "earnings_release": "release", "earnings_call_transcript": "transcript", **{t: "presentation" for t in PRESENTATIONS}}


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frozen_query_path(snapshot: Path, filename: str) -> Path:
    """Prefer grouped inputs while supporting both unchanged older layouts."""
    candidates = [snapshot / folder / filename for folder in (
        "docs/narrative/evaluation", "docs/narrative", "docs",
    )]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Frozen query input missing: " + ", ".join(map(str, candidates)))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def fact_spans(block: dict) -> list[tuple[int, int]]:
    text = block["text"]
    boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\"\u201c])|\n+", text)] + [len(text)]
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        part = text[start:end]
        left, right = len(part) - len(part.lstrip()), len(part.rstrip())
        quote = part[left:right]
        if (14 <= len(quote.split()) <= 100 and METRIC.search(quote) and VALUE.search(quote)
                and not re.search(r"forward.looking|safe harbor|copyright|operator|consensus estimates", quote, re.I)):
            result.append((start + left, start + right))
    return result


def anchor(block: dict, source: dict, start: int, end: int, relation: str) -> dict:
    quote = block["text"][start:end]
    return {"document_id": block["document_id"], "source_sha256": source["sha256"],
            "parsed_file": block["_parsed_file"], "parsed_file_sha256": block["_parsed_sha256"],
            "block_line_1based": block["_line"], "block_text_sha256": hashlib.sha256(block["text"].encode()).hexdigest(),
            "char_start": start, "char_end": end, "offset_convention": "Python Unicode code points; end exclusive",
            "quote": quote, "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
            "section": block["section"], "page": block["page"], "citation_span": block["citation_span"],
            "source_url": block["source_url"], "source_ref": block.get("source_ref"),
            "chart_id": block.get("chart_id"), "content_type": block["content_type"],
            "document_type": block["document_type"], "publication_date": block["publication_date"],
            "evidence_relation_proposed": relation, "review_status": "pending"}


def visual_fact(block: dict) -> tuple[str, list[tuple[int, int]], dict] | None:
    """Extract a displayed chart row or markdown table cell, not a model judgment."""
    lines = []
    for m in re.finditer(r"(?m)^\|.*\|\s*$", block["text"]):
        cells = [c.strip() for c in m.group().strip().strip("|").split("|")]
        lines.append((m.start(), m.end(), cells))
    if len(lines) < 3:
        return None
    header = lines[0]
    chart = header[2][:4] == ["Metric", "Series", "Basis", "Period"]
    data = [row for row in lines[2:] if len(row[2]) == len(header[2]) and METRIC.search(row[2][0])
            and any(re.search(r"\d", c) for c in row[2][1:])]
    if not data:
        return None
    first_metric = data[0][2][0]
    row = [r for r in data if r[2][0] == first_metric][-1]
    if chart and len(row[2]) >= 6:
        metric, series, basis, period, value, unit = row[2][:6]
        claim = f"{block['company_ticker']} reported {metric} of {value} for {period} (series: {series}; basis: {basis}; units: {unit})."
        context = {"metric": metric, "value": value, "units": unit, "reporting_period": period, "accounting_basis": basis, "scope_or_series": series}
    else:
        # Table cells can be structurally aligned while units/title remain elsewhere.
        metric, period, value = row[2][0], header[2][-1], row[2][-1]
        if not period or not re.search(r"\d", value):
            return None
        claim = f"{block['company_ticker']} reported {metric} of {value} for the column labelled {period}."
        context = {"metric": metric, "value": value, "units": None, "reporting_period": period,
                   "accounting_basis": None, "scope_or_series": metric}
    return claim, [(header[0], header[1]), (row[0], row[1])], context


def verify(dataset: Path, foundation: Path) -> dict:
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    failures, counts, cached = [], Counter(), {}
    source_metadata = {s["document_id"]: s for path in (foundation / "parsed").glob("*/input_manifest.json")
                       for s in json.loads(path.read_text(encoding="utf-8"))["sources"]}
    for item in manifest["files"]:
        path = dataset / item["path"]
        if not path.is_file() or file_sha256(path) != item["sha256"]:
            failures.append("file hash mismatch: " + item["path"])
    groups, qids, seen_quotes = {"dev": set(), "heldout": set()}, set(), {}
    for split in ("dev", "heldout"):
        questions = jsonl(dataset / split / "questions.jsonl")
        evidence = jsonl(dataset / split / "evidence_labels.jsonl")
        by_id = {e["evidence_label_id"]: e for e in evidence}
        for q in questions:
            counts[split + "_claims"] += 1
            if q["query_id"] in qids:
                failures.append("duplicate query ID")
            qids.add(q["query_id"])
            groups[split].update(q["event_group_ids"])
            if any(e not in by_id for e in q["evidence_label_ids"]):
                failures.append("missing evidence label")
            if not q["evidence_label_ids"]:
                failures.append("question has no source evidence spans")
        company_by_question = {q["query_id"]: q["ticker"] for q in questions}
        for e in evidence:
            counts[split + "_evidence_spans"] += 1
            source = foundation / e["parsed_file"]
            if source not in cached:
                cached[source] = (file_sha256(source), jsonl(source))
            digest, blocks = cached[source]
            b = blocks[e["block_line_1based"] - 1]
            if digest != e["parsed_file_sha256"] or e["block_text_sha256"] not in {
                    hashlib.sha256(b["text"].encode()).hexdigest(), identity(b["text"])}:
                failures.append("parsed source mismatch")
            if source_metadata.get(e["document_id"], {}).get("sha256") != e["source_sha256"]:
                failures.append("original source hash mismatch")
            if b["document_id"] != e["document_id"] or b["text"][e["char_start"]:e["char_end"]] != e["quote"]:
                failures.append("source offset mismatch")
            if b["company_ticker"] != company_by_question.get(e["query_id"]):
                failures.append("company mismatch")
            if b["publication_date"] != e["publication_date"]:
                failures.append("publication date mismatch")
            if e["publication_date"] > manifest["cutoff_date"]:
                failures.append("post-cutoff evidence")
            quote_key = normalize(e["quote"])
            if quote_key in seen_quotes and seen_quotes[quote_key] != split:
                failures.append("exact evidence quote crosses splits")
            seen_quotes[quote_key] = split
    if groups["dev"] & groups["heldout"]:
        failures.append("event group crosses splits")
    return {"status": "verified" if not failures else "failed", **counts,
            "failures": sorted(set(failures)), "semantic_labels_validated": False,
            "event_group_review_complete": False}


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261018_all16_v1")
    parser.add_argument("--dataset-id", default="evidence_labels_v3")
    parser.add_argument("--cutoff", type=date.fromisoformat, default=date(2025, 7, 31))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", v) for v in (args.run_id, args.dataset_id)):
        parser.error("IDs must contain letters, numbers, hyphens and underscores")
    root = Path.cwd().resolve()
    foundation = root / "data/narrative/experiments/narrative_rag" / args.run_id
    dataset = foundation / "labels" / args.dataset_id
    if args.verify:
        result = verify(dataset, foundation)
        print(json.dumps(result, indent=2))
        raise SystemExit(bool(result["failures"]))
    if dataset.exists():
        parser.error("Dataset exists; use --verify or a new dataset ID")
    frozen = json.loads((foundation / "foundation_manifest.json").read_text(encoding="utf-8"))
    sources, blocks, by_doc, all_sources = {}, [], defaultdict(list), {}
    for ticker in frozen["companies"]:
        folder = foundation / "parsed" / ticker
        info = json.loads((folder / "input_manifest.json").read_text(encoding="utf-8"))
        path = folder / "documents.jsonl"
        if file_sha256(path) != info["blocks_sha256"]:
            raise RuntimeError("Frozen parsed input changed")
        for source in info["sources"]:
            all_sources[source["document_id"]] = source
            if source["publication_date"] <= args.cutoff.isoformat():
                sources[source["document_id"]] = source
        for line, b in enumerate(jsonl(path), 1):
            if b["document_id"] not in sources:
                continue
            b.update({"_line": line, "_parsed_file": path.relative_to(foundation).as_posix(), "_parsed_sha256": info["blocks_sha256"]})
            blocks.append(b)
            by_doc[b["document_id"]].append(b)
    snapshot = foundation / "snapshot"
    legacy = jsonl(frozen_query_path(snapshot, "retrieval_queries_v2.jsonl"))
    sidecar = jsonl(frozen_query_path(snapshot, "retrieval_queries_v2_evidence.jsonl"))
    old_evidence, sidecar_mismatches = [], []
    for old in legacy:
        saved_chunks = {r["chunk_id"]: r for r in jsonl(snapshot / "data/narrative/processed" / old["ticker"] / "chunks.jsonl")}
        for direction in ("support", "counter"):
            for chunk_id in old[f"expected_{direction}_chunk_ids"]:
                if chunk_id not in saved_chunks:
                    raise RuntimeError("Approved legacy chunk missing from frozen baseline: " + chunk_id)
                old_evidence.append({**saved_chunks[chunk_id], "query_id": old["query_id"], "direction": direction})
        if not any(e["query_id"] == old["query_id"] for e in sidecar):
            sidecar_mismatches.append(old["query_id"])
    legacy_doc_ids = {e["document_id"] for e in old_evidence}
    # Both META pilot events have already been examined and must remain development-only.
    legacy_doc_ids.update({"META-0001326801-24-000069-10-Q", "META-IR-e45c0feb1cd3f306"})
    parent = {d: d for d in sources}

    def find(d: str) -> str:
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    def union(a: str, b: str) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # Conservative date clusters join releases, calls, decks and filings around an
    # event. Period/hash aliases catch repeated sources. Human event review remains
    # mandatory: publication dates alone cannot prove reporting-event equivalence.
    aliases = {}
    for ticker in frozen["companies"]:
        docs = sorted((s for s in sources.values() if s["company_ticker"] == ticker
                       and FAMILIES.get(s["document_type"]) in {"filing", "release", "presentation", "transcript"}),
                      key=lambda s: (s["publication_date"], s["document_id"]))
        for a, b in zip(docs, docs[1:]):
            if (date.fromisoformat(b["publication_date"]) - date.fromisoformat(a["publication_date"])).days <= 35:
                union(a["document_id"], b["document_id"])
        for s in docs:
            for alias in ((ticker, "sha256", s["sha256"]), (ticker, "period", s.get("period"))):
                if not alias[-1]:
                    continue
                if alias in aliases:
                    union(s["document_id"], aliases[alias])
                aliases[alias] = s["document_id"]
    group = {d: "event-" + identity(find(d))[:16] for d in sources}
    forced_dev = {group[d] for d in legacy_doc_ids if d in group}
    holdout_groups = set()
    for ticker in frozen["companies"]:
        candidates = sorted({group[d] for d, s in sources.items() if s["company_ticker"] == ticker
                             and group[d] not in forced_dev and FAMILIES.get(s["document_type"]) in {"filing", "release", "presentation", "transcript"}})
        # Reserve one entire unseen event cluster per company where available.
        if candidates:
            holdout_groups.add(candidates[0])
    questions, evidence, issues = [], [], []

    def add(q: dict, anchors: list[dict]) -> None:
        q["evidence_label_ids"] = []
        q["event_group_ids"] = sorted({group[e["document_id"]] for e in anchors})
        q["split"] = "heldout" if q["event_group_ids"] and all(g in holdout_groups for g in q["event_group_ids"]) else "dev"
        q["review_status"] = "pending"
        q["evaluation_eligible"] = False
        q["label_completeness"] = "seed evidence only; additional relevant evidence is unjudged"
        for i, e in enumerate(anchors, 1):
            e.update({"evidence_label_id": q["query_id"] + f":e{i:02d}", "query_id": q["query_id"], "split": q["split"], "event_group_id": group[e["document_id"]]})
            q["evidence_label_ids"].append(e["evidence_label_id"])
            evidence.append(e)
        questions.append(q)

    # Retain original questions verbatim and preserve the old references as lineage,
    # while anchoring an exact matching sentence in the new parsed representation.
    for old in legacy:
        anchors = []
        for ref in (e for e in old_evidence if e["query_id"] == old["query_id"]):
            text = ref["text"]
            candidates = []
            phrases = [m.group().strip() for m in re.finditer(r"[^\n.!?]+(?:[.!?](?=\s|$)|$)", text)]
            words = text.split()
            phrases.extend(" ".join(words[i:i + 30]) for i in range(0, len(words) - 20, 20))
            for quote in phrases:
                if not 60 <= len(quote) <= 800:
                    continue
                expression = re.compile(r"\s+".join(re.escape(w) for w in quote.split()))
                matches = [(b, m.start(), m.end()) for b in by_doc[ref["document_id"]] for m in expression.finditer(b["text"])]
                if len(matches) == 1:
                    terms = set(normalize(old[ref["direction"] + "_query"]).split())
                    candidates.append((len(terms & set(normalize(quote).split())), len(quote), quote, matches[0]))
            if not candidates:
                issues.append({"query_id": old["query_id"], "reason": "legacy reference has no unique exact sentence anchor", "document_id": ref["document_id"]})
                continue
            _, _, quote, (b, start, end) = max(candidates, key=lambda x: (x[0], -x[1]))
            a = anchor(b, sources[ref["document_id"]], start, end,
                       "SUPPORTING" if ref["direction"] == "support" else "BACKGROUND_RISK")
            a.update({"legacy_chunk_id": ref["chunk_id"], "legacy_direction": ref["direction"],
                      "mapping_status": "unique literal quote allowing whitespace normalization; semantic sufficiency requires review"})
            anchors.append(a)
        add({**old, "legacy_review_status": old.get("review_status"), "origin": "approved_v2_development",
             "proposed_verdict": "UNREVIEWED", "cutoff_date": args.cutoff.isoformat(),
             "question": "Assess this claim: " + old["claim"], "task_tags": ["legacy_narrative", "qualification_or_background_risk"],
             "required_context": {"metric": "review required", "value": "if applicable", "units": "if applicable", "reporting_period": "review required", "accounting_basis": "if applicable", "scope_or_series": "review required"}}, anchors)
    used_quotes = set()
    slots = [("annual_fact", "filing", False), ("quarterly_counter", "filing", True),
             ("release_fact", "release", False), ("release_basis", "release", False),
             ("presentation_visual", "presentation", False), ("transcript_fact", "transcript", False),
             ("transcript_counter", "transcript", True)]
    for company_index, ticker in enumerate(frozen["companies"]):
        company_blocks = [b for b in blocks if b["company_ticker"] == ticker]
        for slot_index, (slot, family, mutate) in enumerate(slots):
            # Rotate heldout slot preferences so neither contradictions nor visuals
            # are systematically confined to heldout across all companies.
            desired_split = "heldout" if (slot_index + company_index) % 3 == 1 else "dev"
            pool = [b for b in company_blocks if FAMILIES.get(b["document_type"]) == family]
            missing_family = not pool
            if family == "presentation":
                pool += [b for b in company_blocks if FAMILIES.get(b["document_type"]) == "filing"]
            if missing_family:
                pool = [b for b in company_blocks if FAMILIES.get(b["document_type"]) == "filing"]
            facts = []
            for b in pool:
                is_heldout = group[b["document_id"]] in holdout_groups
                if family == "presentation" and (visual := visual_fact(b)):
                    claim, spans, context = visual
                    clean_chart = not re.search(r"Validation required|ambiguity|not found in Docling", b["text"], re.I)
                    facts.append((int(is_heldout == (desired_split == "heldout")), 3 + int(clean_chart), b, claim, spans, context, "displayed_chart" if b["content_type"] == "chart" else "markdown_table"))
                if b["content_type"] != "narrative":
                    continue
                for start, end in fact_spans(b):
                    quote = b["text"][start:end]
                    if normalize(quote) in used_quotes:
                        continue
                    context = {"metric": METRIC.search(quote).group(), "value": VALUE.search(quote).group(),
                               "units": "explicit in quote; review scale", "reporting_period": " ".join(dict.fromkeys(m.group() for m in PERIOD.finditer(quote))) or None,
                               "accounting_basis": BASIS.search(quote).group() if BASIS.search(quote) else None,
                               "scope_or_series": "review consolidated/segment scope"}
                    preference = int(bool(PERIOD.search(quote))) + int(slot == "release_basis" and bool(BASIS.search(quote)))
                    if slot == "annual_fact":
                        preference += int(b["document_type"] in ("10-K", "20-F"))
                    if slot == "quarterly_counter":
                        preference += int(b["document_type"] == "10-Q")
                    facts.append((int(is_heldout == (desired_split == "heldout")), preference, b, quote, [(start, end)], context, "narrative"))
            if not facts:
                issues.append({"ticker": ticker, "slot": slot, "reason": "no suitable financial fact; do not fabricate evidence"})
                continue
            # Transparent draft sampling, not relevance evaluation: prefer desired
            # split, then explicit context, then stable source order. No rankings read.
            facts.sort(key=lambda f: (int(FAMILIES.get(f[2]["document_type"]) != family), -f[0], -f[1], identity([f[2]["document_id"], f[3]])))
            _, _, b, claim, spans, context, form = facts[0]
            missing_family = FAMILIES.get(b["document_type"]) != family
            original_claim = claim
            transformations = None
            if mutate:
                match = VALUE.search(claim)
                if not match:
                    raise RuntimeError("Counter draft requires an explicit numeric fact")
                number = re.search(r"\d[\d,]*(?:\.\d+)?", match.group())
                position = match.start() + number.start()
                original_value = number.group()
                wrong_value = format(float(original_value.replace(",", "")) * 2 + 1, ".6f").rstrip("0").rstrip(".")
                claim = claim[:position] + wrong_value + claim[position + len(original_value):]
                transformations = {"type": "synthetic_numeric_perturbation", "original": original_value, "replacement": wrong_value,
                                   "caution": "Proposed contradiction requires same metric, basis, period, scope and unit review; not a natural user claim"}
            relations = ["HEADER_CONTEXT"] * (len(spans) - 1) + ["CHALLENGING" if mutate else "SUPPORTING"]
            anchors = [anchor(b, sources[b["document_id"]], a, z, relation) for (a, z), relation in zip(spans, relations)]
            tags = [slot, form, "synthetic_counterexample" if mutate else "source_derived_fact"]
            if BASIS.search(original_claim):
                tags.append("accounting_basis")
            if re.search(r"segment|consolidated|total revenue|net sales|company.wide", original_claim, re.I):
                tags.append("scope_distinction")
            if PERIOD.search(original_claim):
                tags.append("period_distinction")
            add({"query_id": f"{ticker.lower()}_{slot}", "ticker": ticker, "cutoff_date": args.cutoff.isoformat(),
                 "question": f"Does {ticker}'s reported information support this claim: {claim}", "claim": claim,
                 "support_query": f"{ticker} {context['metric']} {context.get('reporting_period') or ''}",
                 "counter_query": f"{ticker} {context['metric']} reported actual results {context.get('reporting_period') or ''}",
                 "proposed_verdict": "CONTRADICTED" if mutate else "SUPPORTED", "origin": "deterministic_source_draft",
                 "synthetic_transformation": transformations, "task_tags": tags,
                 "requested_family": family, "actual_family": FAMILIES.get(b["document_type"]),
                 "family_fallback": missing_family, "required_context": context,
                 "context_review_required": True, "original_pdf_review_required": form != "narrative",
                 "extraction_warning_present": bool(re.search(r"Validation required|ambiguity", b["text"], re.I))}, anchors)
            used_quotes.add(normalize(original_claim))
            if missing_family:
                issues.append({"ticker": ticker, "slot": slot, "reason": "requested family unavailable or has no qualifying financial fact before cutoff; filing fallback explicitly marked"})
    # Evidence quote aliases discovered in the selected labels cannot straddle splits.
    # Move the entire heldout event cluster to development rather than leaving leakage.
    quote_groups = defaultdict(set)
    for e in evidence:
        quote_groups[normalize(e["quote"])].add(e["event_group_id"])
    moved = True
    while moved:
        moved = False
        for aliases in quote_groups.values():
            if aliases & holdout_groups and aliases - holdout_groups:
                holdout_groups -= aliases
                moved = True
    for q in questions:
        q["split"] = "heldout" if q["event_group_ids"] and all(g in holdout_groups for g in q["event_group_ids"]) else "dev"
        for e in evidence:
            if e["query_id"] == q["query_id"]:
                e["split"] = q["split"]
    dataset.mkdir(parents=True)
    for split in ("dev", "heldout"):
        write_jsonl(dataset / split / "questions.jsonl", [q for q in questions if q["split"] == split])
        write_jsonl(dataset / split / "evidence_labels.jsonl", [e for e in evidence if e["split"] == split])
        write_jsonl(dataset / split / "review_decisions.template.jsonl", [{"query_id": q["query_id"],
            "decision": "pending", "reviewer": None, "independent_adjudicator": None,
            "approved_verdict": None, "approved_evidence_relations": {}, "context_complete": None,
            "source_document_checked": None, "event_family_split_checked": None, "notes": ""}
            for q in questions if q["split"] == split])
    write_json(dataset / "event_groups.json", {"policy": "35-day connected publication clusters + exact source hashes + period aliases; provisional until human review",
        "groups": [{"group_id": g, "split": "heldout" if g in holdout_groups else "dev", "documents": sorted(d for d in group if group[d] == g),
                    "review_status": "pending"} for g in sorted(set(group.values()))]})
    # Technical/mapping issues contain development-only details. Heldout progress is aggregate.
    write_json(dataset / "development_issues.json", [i for i in issues if not i.get("query_id") or next((q["split"] for q in questions if q["query_id"] == i["query_id"]), "dev") == "dev"])
    dev_md = ["# Development label review", "", "Drafts only. Review source documents and context before approval.", ""]
    for q in (q for q in questions if q["split"] == "dev"):
        dev_md += [f"## {q['query_id']}", "", q["question"], "", f"Proposed verdict: {q['proposed_verdict']} (unreviewed).", ""]
        for e in (e for e in evidence if e["query_id"] == q["query_id"]):
            dev_md += [f"- `{e['evidence_label_id']}`: {e['evidence_relation_proposed']}; `{e['document_id']}`; {e['citation_span']}", "", "> " + e["quote"].replace("\n", "\n> "), ""]
        dev_md += ["Required context: `" + json.dumps(q["required_context"], ensure_ascii=False) + "`", ""]
    (dataset / "development_review.md").write_text("\n".join(dev_md), encoding="utf-8")
    readme = """# Evidence-label drafts: review and release gates

These files are not approved benchmark ground truth. Existing approved v2 questions
are preserved as development material; their previous chunk-level approval does not
approve the new spans or verdicts. Seed labels are incomplete relevance judgments.

## Files

- `dev/`: development questions, evidence spans and review template.
- `heldout/`: provisional heldout questions, evidence spans and review template.
  Only a designated reviewer should inspect these; do not use them for tuning.
- `development_review.md`: readable development questions and quoted evidence.
- `event_groups.json`: provisional event grouping for split review.
- `development_issues.json`: explicit source-family/mapping gaps.
- `dataset_manifest.json`: file hashes, scope, identities and evaluation gate.

Span offsets address the exact frozen parsed block (1-based JSONL line, 0-based
Unicode offsets with exclusive end). Multiple anchors form a required evidence
bundle, e.g. table header plus data row; future mapping must not treat those as
independent complete answers. Source PDF/hash/page references are preserved.

## Review every question before release

1. Rewrite source-derived questions where necessary into natural verification
   questions without changing metric, scope, reporting period or cutoff. Some
   drafts quote sources closely and can favor lexical retrieval.
2. Verify the quote in the original source, including signs, decimals, units,
   reporting period, GAAP/non-GAAP basis and consolidated/segment scope where
   applicable. A PDF-derived chart/table needs original-PDF verification; matching
   extracted text is not extraction accuracy. Resolve any extraction warnings.
3. Approve or correct the verdict and each evidence relation separately:
   SUPPORTING, CHALLENGING (direct contradiction), MATERIAL_QUALIFICATION,
   BACKGROUND_RISK, HEADER_CONTEXT or NOT_RELEVANT. Generic risks must not count as
   direct contradictions. Synthetic numeric perturbations are diagnostics, not
   representative naturally occurring false claims. Insufficient-evidence and
   mixed/partially-supported natural claims still need curated additions.
4. Add all required context anchors when units, period, speaker or scope lie
   outside the selected quote. Do not equate publication date with reporting period.
5. Review event groups across filings/releases/decks/transcripts and source aliases.
   The date/period heuristic is conservative but cannot certify event identity.
   Both prior META pilot events and all old v2 reference events are development.
6. Copy the split's review template outside this immutable dataset, e.g. into
   `reviews/<DATASET_ID>/`, and fill in actual reviewer/adjudicator identities;
   retain rejections and ambiguous mappings. Do not invent independent review.
7. Create a new reviewed dataset version, preserve this draft, validate all anchors
   and split boundaries, and then freeze a heldout lock. The current generator
   intentionally does not create an approved heldout lock or evaluation-ready flag.

Pool candidate passages after development retrieval, judge previously unjudged
results, and version expanded labels. Treat unjudged passages as unknown rather
than automatically irrelevant; Precision@5 requires adequately judged pools.
Keep heldout content/outcomes out of progress reports. A proper heldout evaluation
must use an untouched reviewer-managed release, not this unreviewed proposal.
"""
    (dataset / "README.md").write_text(readme, encoding="utf-8")
    generator_copy = dataset / "generator/prepare_narrative_labels.py"
    generator_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), generator_copy)
    selected_files = [p for p in dataset.rglob("*") if p.is_file()]
    manifest = {"status": "draft_prepared_review_pending", "evaluation_ready": False,
        "dataset_id": args.dataset_id, "foundation_build_identity": frozen["build_identity"],
        "cutoff_date": args.cutoff.isoformat(), "companies": frozen["companies"],
        "claim_count": len(questions), "legacy_claims_preserved": len(legacy),
        "source_span_count": len(evidence), "reviewed_new_labels": 0, "heldout_locked": False,
        "generator_sha256": file_sha256(Path(__file__)),
        "legacy_reference_recovery": {"source": "approved query expected IDs resolved against frozen canonical chunks", "sidecar_query_mismatches": sidecar_mismatches},
        "draft_selection": "no rankings read; deterministic explicit financial facts; split/context preference then stable source hash",
        "split_sampling": "one unseen event cluster reserved per company; heldout slot preference rotates by (slot_index + company_index) modulo 3",
        "files": [{"path": p.relative_to(dataset).as_posix(), "sha256": file_sha256(p)} for p in sorted(selected_files)]}
    write_json(dataset / "dataset_manifest.json", manifest)
    write_json(foundation / "labels/active_dataset.json", {
        "dataset_id": args.dataset_id, "status": manifest["status"], "evaluation_ready": False,
        "manifest_sha256": file_sha256(dataset / "dataset_manifest.json")})
    integrity = verify(dataset, foundation)
    coverage = []
    for ticker in frozen["companies"]:
        rows = [q for q in questions if q["ticker"] == ticker]
        coverage.append({"ticker": ticker, "claims": len(rows), "dev": sum(q["split"] == "dev" for q in rows),
                         "heldout": sum(q["split"] == "heldout" for q in rows),
                         "new_label_families": dict(Counter(q.get("actual_family") for q in rows if q["origin"] != "approved_v2_development")),
                         "family_fallbacks": sum(q.get("family_fallback", False) for q in rows)})
    # No heldout claim text, proposed verdicts, quotes or per-item outcomes in reports.
    dev_questions = [q for q in questions if q["split"] == "dev"]
    report = {"status": "draft_prepared_review_pending", "step": "02_evaluation_labels",
        "dataset_path": dataset.relative_to(root).as_posix(), "claim_count": len(questions),
        "company_count": len(coverage), "coverage_by_company": coverage,
        "split_counts": dict(Counter(q["split"] for q in questions)), "source_span_count": len(evidence),
        "event_group_count": len(set(group.values())), "issue_count": len(issues),
        "legacy_sidecar_mismatches_recovered": len(sidecar_mismatches),
        "legacy_reference_spans": sum("legacy_chunk_id" in e for e in evidence),
        "questions_without_source_spans": sum(not q["evidence_label_ids"] for q in questions),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "dataset_bytes": sum(p.stat().st_size for p in dataset.rglob("*") if p.is_file()),
        "technical_integrity": integrity, "evaluation_ready": False, "independent_review_complete": False,
        "heldout_locked": False, "api_requests": 0,
        "development_tags": dict(Counter(t for q in dev_questions for t in q["task_tags"])),
        "next_action": "Review question wording, verdicts, evidence/context and event groups; curate missing natural claim types; release a new reviewed version before evaluation"}
    write_json(dataset / "reports/step_02_evaluation_labels.json", report)
    write_json(foundation / "reports/step_02_evaluation_labels.json", report)
    table = "\n".join(f"| {r['ticker']} | {r['claims']} | {r['dev']} | {r['heldout']} |" for r in coverage)
    md = f"""# Step 2 - All-company evidence-label preparation

Status: **Draft prepared; review and final freeze pending.** Evaluation-ready: **No**.

## Scope and results

Prepared {len(questions)} claims across {len(coverage)} companies, with {len(evidence)}
exact source-span anchors. Preserved all {len(legacy)} original v2 development claims.
Split counts: {json.dumps(report['split_counts'])}. No heldout claims, evidence text,
proposed verdicts or per-item outcomes are included in this progress report.

| Company | Claims | Development | Provisional heldout |
|---|---:|---:|---:|
{table}

## Implementation and inputs

`scripts/narrative/datasets/prepare_narrative_labels.py` reuses frozen Step 1 inputs, the approved v2
questions/reference lineage and existing SHA helpers. It does not call retrievers,
models or APIs. New questions come from explicit financial statements, chart rows
or table cells. Controlled numeric changes add proposed direct-counter examples;
these remain unreviewed synthetic diagnostics, not natural false-claim gold labels.
Legacy contextual risks remain distinct from proposed direct contradictions.
The reference sidecar has query-ID mismatches for {len(sidecar_mismatches)} original
questions. Approved expected chunk IDs were resolved against frozen canonical
chunks instead, preserving that lineage. Legacy source spans recovered:
{report['legacy_reference_spans']}; questions without any source spans:
{report['questions_without_source_spans']}.

Source build: `{frozen['build_identity']}`. Cutoff: {args.cutoff.isoformat()}.
Configuration and generator hashes are in the dataset manifest. No production
outputs, indexes, old labels or results were modified. Run output:
`{dataset.relative_to(root).as_posix()}/` (relative to the project root).

## Outputs and review workflow

All paths below start at the project root (`GenAI-group-project/`).

- `{dataset.relative_to(root).as_posix()}/dev/questions.jsonl`: development claims/questions.
- `{dataset.relative_to(root).as_posix()}/dev/evidence_labels.jsonl`: development evidence quotes and source locations.
- `{dataset.relative_to(root).as_posix()}/development_review.md`: readable development questions and evidence.
- `{dataset.relative_to(root).as_posix()}/dev/review_decisions.template.jsonl`: development review template; copy outside the immutable dataset before editing.
- `{dataset.relative_to(root).as_posix()}/heldout/questions.jsonl`: provisional heldout claims; reserved for the designated reviewer, not tuning.
- `{dataset.relative_to(root).as_posix()}/heldout/evidence_labels.jsonl`: provisional heldout evidence labels.
- `{dataset.relative_to(root).as_posix()}/heldout/review_decisions.template.jsonl`: designated reviewer's heldout review template.
- `{dataset.relative_to(root).as_posix()}/event_groups.json`: candidate event groups and source aliases; review required.
- `{dataset.relative_to(root).as_posix()}/development_issues.json`: mapping/family gaps, without heldout item details.
- `{dataset.relative_to(root).as_posix()}/dataset_manifest.json`: file hashes and explicit evaluation gate.
- `{dataset.relative_to(root).as_posix()}/README.md`: annotation instructions and release criteria.

Sources unavailable before the cutoff are not invented or substituted silently:
AAPL, ORCL and SBUX have no presentation records in the eligible frozen pre-cutoff
input. Presentation slots fall back to filings and are explicitly marked. These
are dataset coverage gaps, not claims that presentations do not exist externally.
NFLX's pre-cutoff presentation did not yield a qualifying financial fact under
the declared selection rules; its slot also has an explicit filing fallback.

## Checks and limitations

Technical integrity: **{integrity['status']}**. Checked exact block/quote offsets,
frozen parsed-file hashes, company-local source preparation, publication cutoff,
unique question IDs, reference links and no event-group/exact-quote overlap between
splits. See the JSON report for aggregate counts/errors. Technical checks do not
validate semantic verdicts or prove that original-PDF extraction is correct.

Event grouping combines 35-day publication clusters, source hashes and period
aliases. Entire old-v2/META-pilot event clusters stay development-only. The grouping
is provisional: event/duplicate equivalence requires reviewer confirmation before
a heldout lock. Exact selected-quote aliases crossing splits move the whole event
cluster to development. Broader paraphrase aliases still require review.

No independent new label review has occurred. Source-derived wording can favor
BM25; rewrite/curate natural questions before finalization. Source units, basis,
period, scope and table/chart context are explicit review requirements, not
automatically certified complete. Only seed evidence is labelled; unjudged
passages are unknown. Genuine insufficient-evidence/partial-support claims and
adequately judged relevance pools are still required. INTC's 11 single-block
filings remain a known Step 3 parser/structure issue.

## Commands

Read-only integrity check and development inspection (no model calls):

```powershell
python -m scripts.narrative.datasets.prepare_narrative_labels --run-id {args.run_id} --dataset-id {args.dataset_id} --verify
$labels = "{dataset.relative_to(root).as_posix()}"
Get-Content -Raw "$labels/dataset_manifest.json"
Get-Content "$labels/dev/questions.jsonl" -TotalCount 1
```

Create a new isolated draft version (writes new dataset/report):

```powershell
python -m scripts.narrative.datasets.prepare_narrative_labels --run-id {args.run_id} --dataset-id {args.dataset_id}_next
```

Preparation elapsed: {report['elapsed_seconds']} seconds; dataset files:
{report['dataset_bytes']} bytes at report generation. API/model usage: **0**.
No embedding/indexing overhead was incurred. Labels remain
drafts; no retrieval or end-to-end quality scores were generated in this step.

## Acceptance gate and next action

Draft preparation is complete; **Step 2 is not fully accepted**. Review/correct the
claims, exact evidence, required context, verdict labels and cross-format event
groups, then create a new reviewed release and lock its heldout split. Preserve
this draft and its hashes. Proceed to evaluation only after those gates pass;
Step 3 technical validation may be developed against this draft without selecting
a winner or reporting benchmark quality.
"""
    (dataset / "reports/step_02_evaluation_labels.md").write_text(md, encoding="utf-8")
    (foundation / "reports/step_02_evaluation_labels.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": report["status"], "claims": len(questions), "companies": len(coverage),
                      "split_counts": report["split_counts"], "evidence_spans": len(evidence), "integrity": integrity}, indent=2))
    if integrity["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
