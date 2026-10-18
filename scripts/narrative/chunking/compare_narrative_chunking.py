"""Compare frozen C1-C4 chunks with existing BM25 on development claims only.

Draft labels require --allow-draft and produce diagnostics, never a selected
winner. Reserved claims are not evaluated. Canonical indexes remain untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from finsight.agents.narrative.processing.chunk import NarrativeChunk
from finsight.agents.narrative.processing.corpus import file_sha256
from finsight.agents.narrative.processing.index import build_index, search_index
from scripts.narrative.datasets.narrative_reference_mapping import covers, source_intervals
from scripts.narrative.experiments.prepare_narrative_experiment import identity, write_json
from scripts.narrative.datasets.prepare_narrative_labels import jsonl, write_jsonl


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def evidence_bundles(question: dict, evidence: list[dict]) -> list[dict]:
    selected = [e for e in evidence if e["query_id"] == question["query_id"]]
    context = [e for e in selected if e["evidence_relation_proposed"] == "HEADER_CONTEXT"]
    groups = {}
    for e in selected:
        if e["evidence_relation_proposed"] in {"HEADER_CONTEXT", "NOT_RELEVANT"}:
            continue
        key = (e["source_sha256"], e["document_id"], e["block_line_1based"], e["char_start"], e["char_end"])
        if key not in groups:
            groups[key] = {"spans": [e] + context, "roles": set()}
        groups[key]["roles"].add(e["evidence_relation_proposed"])
    if not groups:
        raise ValueError("Question has no fact reference bundle")
    return list(groups.values())


def evaluate(question: dict, bundles: list[dict], ranking: list[dict], aliases: dict, maps: dict,
             rows: list[dict]) -> dict:
    """Score explicit source-span bundles; context-only hits never count as facts."""
    def complete(bundle: dict, results: list[dict]) -> bool:
        for e in bundle["spans"]:
            ranges = []
            for hit in results:
                for alias in aliases[hit["chunk_id"]]:
                    m = maps[alias["chunk_id"]]
                    if (m["document_id"] == e["document_id"] and m["source_sha256"] == e["source_sha256"]
                            and m["block_line_1based"] == e["block_line_1based"]):
                        ranges.extend(source_intervals(m))
            if not covers(rows[e["block_line_1based"] - 1]["text"], e["char_start"], e["char_end"], ranges):
                return False
        return True

    values = {"query_id": question["query_id"], "ticker": question["ticker"],
              "origin": question["origin"], "reference_bundle_count": len(bundles)}
    for k in (5, 10):
        done = [complete(b, ranking[:k]) for b in bundles]
        values[f"recall@{k}"] = sum(done) / len(done)
        values[f"all_bundles_found@{k}"] = all(done)
        for role, name in (("SUPPORTING", "support"), ("CHALLENGING", "direct_counter")):
            matching = [done[i] for i, b in enumerate(bundles) if role in b["roles"]]
            values[f"{name}_recall@{k}"] = mean(matching)
    first_relevant = next((rank for rank, hit in enumerate(ranking[:10], 1)
                           if any(complete(b, [hit]) for b in bundles)), None)
    # Standard MRR requires a single relevant result. A split reference can instead
    # complete across several results; report that separately rather than relabel it.
    values["mrr@10"] = 1 / first_relevant if first_relevant else 0.0
    completion = next((k for k in range(1, min(10, len(ranking)) + 1)
                       if all(complete(b, ranking[:k]) for b in bundles)), None)
    values["bundle_completion_rr@10"] = 1 / completion if completion else 0.0
    values["first_single_result_reference_rank"] = first_relevant
    values["all_bundle_completion_rank"] = completion
    return values


def summarize(per_query: list[dict]) -> dict:
    fields = ("recall@5", "recall@10", "mrr@10", "bundle_completion_rr@10", "support_recall@5",
              "support_recall@10", "direct_counter_recall@5", "direct_counter_recall@10")
    result = {name: mean([r[name] for r in per_query if r[name] is not None]) for name in fields}
    result["claims"] = len(per_query)
    result["support_claims"] = sum(r["support_recall@10"] is not None for r in per_query)
    result["direct_counter_claims"] = sum(r["direct_counter_recall@10"] is not None for r in per_query)
    latencies = sorted(r["search_ms"] for r in per_query)
    result["search_median_ms"] = statistics.median(latencies)
    result["search_p95_ms"] = latencies[min(len(latencies)-1, int(0.95 * len(latencies)))]
    return result


def bootstrap(per_query: list[dict], samples: int = 2000) -> dict:
    """Paired company-cluster bootstrap; exploratory dev estimates, not final claims."""
    rng = random.Random(20261018)
    tickers = sorted({r["ticker"] for r in per_query})
    by_arm = {a: {r["query_id"]: r for r in per_query if r["arm"] == a} for a in ("C1", "C2", "C3", "C4")}
    groups = {t: [r["query_id"] for r in per_query if r["arm"] == "C1" and r["ticker"] == t] for t in tickers}
    draws = [[rng.choice(tickers) for _ in tickers] for _ in range(samples)]
    results = []
    for arm in ("C2", "C3", "C4"):
        for metric in ("recall@5", "recall@10", "mrr@10"):
            group_sums = {t: sum(by_arm[arm][q][metric] - by_arm["C1"][q][metric] for q in groups[t]) for t in tickers}
            differences = sorted(sum(group_sums[t] for t in draw) / sum(len(groups[t]) for t in draw) for draw in draws)
            point = mean([by_arm[arm][q][metric] - by_arm["C1"][q][metric] for q in by_arm["C1"]])
            results.append({"comparison": arm + " minus C1", "metric": metric, "mean_difference": point,
                            "interval_95": [differences[int(samples*0.025)], differences[min(samples-1, int(samples*0.975))]]})
    return {"method": "paired company-cluster bootstrap", "company_clusters": len(tickers), "samples": samples,
            "seed": 20261018, "scope": "conditional on this sparse draft development set; exploratory only", "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261018_all16_v1")
    parser.add_argument("--dataset-id", default="evidence_labels_v3")
    parser.add_argument("--comparison-id", default="bm25_dev_v1")
    parser.add_argument("--fixed-validation-id", default="technical_v1")
    parser.add_argument("--adaptive-validation-id", default="c4_v1")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    import re
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", x) for x in (args.run_id, args.dataset_id, args.comparison_id, args.fixed_validation_id, args.adaptive_validation_id)):
        parser.error("Invalid version ID")
    root = Path.cwd().resolve()
    run = root / "data/narrative/experiments/narrative_rag" / args.run_id
    output = run / "comparisons" / args.comparison_id
    dataset = run / "labels" / args.dataset_id
    if args.verify:
        m = json.loads((output / "comparison_manifest.json").read_text(encoding="utf-8"))
        failures = [f["path"] for f in m["files"] if not (output / f["path"]).is_file() or file_sha256(output / f["path"]) != f["sha256"]]
        print(json.dumps({"status": "verified" if not failures else "failed", "files_checked": len(m["files"]), "failures": failures}, indent=2))
        raise SystemExit(bool(failures))
    if output.exists():
        parser.error("Comparison exists; use --verify or a new --comparison-id")
    started = time.monotonic()
    labels = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    foundation = json.loads((run / "foundation_manifest.json").read_text(encoding="utf-8"))
    if labels["foundation_build_identity"] != foundation["build_identity"]:
        raise RuntimeError("Source/label build mismatch")
    if not labels["evaluation_ready"] and not args.allow_draft:
        parser.error("Labels are unreviewed; --allow-draft is required for development diagnostics")
    # Hash the reserved files for integrity, but do not parse or evaluate their data.
    for f in labels["files"]:
        if file_sha256(dataset / f["path"]) != f["sha256"]:
            raise RuntimeError("Frozen label file changed: " + f["path"])
    questions = jsonl(dataset / "dev/questions.jsonl")
    evidence = jsonl(dataset / "dev/evidence_labels.jsonl")
    if any(q["split"] != "dev" for q in questions) or any(e["split"] != "dev" for e in evidence):
        raise RuntimeError("Development split contains a non-development record")
    if len({q["query_id"] for q in questions}) != len(questions):
        raise RuntimeError("Duplicate question ID")
    sources_root = {a: run / "validation" / (args.adaptive_validation_id if a == "C4" else args.fixed_validation_id) for a in ("C1", "C2", "C3", "C4")}
    for folder in set(sources_root.values()):
        conf = json.loads((folder / "configuration.json").read_text(encoding="utf-8"))
        if conf["foundation_build_identity"] != foundation["build_identity"] or conf["labels_manifest_sha256"] != file_sha256(dataset / "dataset_manifest.json"):
            raise RuntimeError("Chunk-plan source or label identity differs")
        vm = json.loads((folder / "validation_manifest.json").read_text(encoding="utf-8"))
        if any(file_sha256(folder / f["path"]) != f["sha256"] for f in vm["files"]):
            raise RuntimeError("Frozen chunk-plan output changed")
    output.mkdir(parents=True)
    configuration = {"scope": "development diagnostic" if not labels["evaluation_ready"] else "reviewed development comparison",
        "arms": list(sources_root), "source_build_identity": foundation["build_identity"], "dataset_id": args.dataset_id,
        "dataset_manifest_sha256": file_sha256(dataset / "dataset_manifest.json"), "questions_sha256": file_sha256(dataset / "dev/questions.jsonl"),
        "evidence_sha256": file_sha256(dataset / "dev/evidence_labels.jsonl"), "query_field": "claim (unchanged; no gold support/counter query supplied)",
        "backend": "existing index.search_index BM25", "priority": "P0", "top_k": [5, 10],
        "BM25": {"k1": 1.5, "b": 0.75, "query_term_duplicates": "removed", "ranking": "existing citation-location diversification; deferred same-location hits fill remaining slots"},
        "deduplication": "existing build_index exact-text dedup after company/date/eligibility filters; all literal source aliases retained and made citable",
        "metrics": "macro provisional seed fact-bundle Recall@5/10; context-only hits do not count; MRR@10 needs one individually complete bundle",
        "unjudged_policy": "unknown, not irrelevant; Precision/nDCG and semantic context/citation accuracy not asserted",
        "heldout_evaluated": False, "labels_reviewed": labels["evaluation_ready"], "winner_selection": "deferred",
        "timing": "one measured search per claim/arm; index building/loading excluded from query latency; no hardware significance claims",
        "runner_sha256": file_sha256(Path(__file__)), "BM25_module_sha256": file_sha256(root / "finsight/agents/narrative/processing/index.py")}
    write_json(output / "configuration.json", configuration)
    all_results, index_reports, citation_errors, input_identities = [], [], [], []
    pool = {}
    for ticker in foundation["companies"]:
        qrows = sorted((q for q in questions if q["ticker"] == ticker), key=lambda q: q["query_id"])
        parsed = run / "parsed" / ticker
        inputs = json.loads((parsed / "input_manifest.json").read_text(encoding="utf-8"))
        if file_sha256(parsed / "documents.jsonl") != inputs["blocks_sha256"]:
            raise RuntimeError("Frozen source text changed")
        rows = jsonl(parsed / "documents.jsonl")
        src = {s["document_id"]: s for s in inputs["sources"]}
        for arm in sources_root:
            prepare_started = time.monotonic()
            folder = sources_root[arm] / "arms" / arm / ticker
            chunks = jsonl(folder / "chunks.jsonl")
            maps = {m["chunk_id"]: m for m in jsonl(folder / "source_map.jsonl")}
            token_metrics = {m["chunk_id"]: m for m in jsonl(folder / "chunk_metrics.jsonl")}
            input_identities.append({"ticker": ticker, "arm": arm, "chunks_sha256": file_sha256(folder / "chunks.jsonl"), "source_map_sha256": file_sha256(folder / "source_map.jsonl")})
            for cutoff in sorted({q["cutoff_date"] for q in qrows}):
                allowed = {d for d, s in src.items() if s["retrieval_eligible"] and s["publication_date"] <= cutoff}
                eligible = [c for c in chunks if c["document_id"] in allowed and c["publication_date"] <= cutoff and c["company_ticker"] == ticker]
                if {c["document_id"] for c in eligible} != allowed:
                    raise RuntimeError("Arm lacks an eligible source document")
                by_hash = defaultdict(list)
                for c in eligible:
                    if maps[c["chunk_id"]]["source_sha256"] != src[c["document_id"]]["sha256"]:
                        raise RuntimeError("Chunk/source lineage mismatch")
                    by_hash[c["text_sha256"]].append(c)
                dest = output / "indexes" / arm / ticker / cutoff
                index_started = time.monotonic()
                settings = {"C1": (800, 120), "C2": (300, 50), "C3": (400, 60), "C4": (0, 0)}[arm]
                fingerprint = identity({"source_build": foundation["build_identity"], "arm": arm, "cutoff": cutoff,
                                        "chunks_sha256": file_sha256(folder / "chunks.jsonl")})
                payload = build_index((NarrativeChunk(**c) for c in eligible), dest / "index.json", corpus_fingerprint=fingerprint,
                                      target_words=settings[0], overlap_words=settings[1])
                if arm == "C4":
                    c4conf = json.loads((sources_root[arm] / "configuration.json").read_text(encoding="utf-8"))
                    payload["chunking"] = {"strategy": "adaptive", **c4conf["adaptive"], "effective_token_limit": 256}
                    write_json(dest / "index.json", payload)
                aliases = {c["chunk_id"]: by_hash[c["text_sha256"]] for c in payload["chunks"]}
                alias_rows = [{"representative_chunk_id": rep, "aliases": [{k: c[k] for k in ("chunk_id", "document_id", "publication_date", "source_url", "section", "page", "citation_span")} for c in variants]}
                              for rep, variants in aliases.items()]
                write_jsonl(dest / "source_aliases.jsonl", alias_rows)
                index_ms = (time.monotonic() - index_started) * 1000
                word_lengths = [len(c["text"].split()) for c in payload["chunks"]]
                source_nonwhite = sum(len("".join(r["text"].split())) for r in rows if r["document_id"] in allowed)
                represented_nonwhite = sum(len("".join(rows[maps[c["chunk_id"]]["block_line_1based"]-1]["text"][a:z].split()))
                                          for c in eligible for a, z in source_intervals(maps[c["chunk_id"]]))
                index_reports.append({"ticker": ticker, "arm": arm, "cutoff": cutoff, "eligible_documents": len(allowed),
                    "eligible_source_ids_sha256": identity(sorted(allowed)), "candidate_chunks": len(eligible), "indexed_chunks": payload["chunk_count"],
                    "exact_text_duplicates": payload["duplicates_excluded"], "index_build_ms": index_ms,
                    "prepare_and_build_ms": (time.monotonic()-prepare_started)*1000, "index_bytes": (dest / "index.json").stat().st_size,
                    "alias_bytes": (dest / "source_aliases.jsonl").stat().st_size, "indexed_word_sum": sum(word_lengths),
                    "indexed_mean_words": statistics.mean(word_lengths), "indexed_median_words": statistics.median(word_lengths),
                    "source_nonwhitespace_characters": source_nonwhite, "represented_source_character_occurrences": represented_nonwhite,
                    "repeated_source_character_fraction": represented_nonwhite / source_nonwhite - 1,
                    "over_limit_indexed_chunks": sum(not token_metrics[c["chunk_id"]]["token_limit_pass"] for c in payload["chunks"])})
                rankings = []
                for q in (q for q in qrows if q["cutoff_date"] == cutoff):
                    bundles = evidence_bundles(q, evidence)
                    t0 = time.monotonic()
                    ranking = search_index(payload, q["claim"], 10, date.fromisoformat(cutoff), ticker)
                    search_ms = (time.monotonic()-t0)*1000
                    values = evaluate(q, bundles, ranking, aliases, maps, rows)
                    values.update(arm=arm, search_ms=search_ms, cutoff=cutoff, query_sha256=identity(q["claim"]))
                    all_results.append(values)
                    ranked_hits = []
                    for rank, hit in enumerate(ranking, 1):
                        available = aliases[hit["chunk_id"]]
                        for alias in available:
                            if alias["document_id"] not in allowed or alias["publication_date"] > cutoff or alias["company_ticker"] != ticker or not alias["citation_span"]:
                                citation_errors.append({"query_id": q["query_id"], "arm": arm, "reason": "source/company/date/citation metadata mismatch"})
                        ranked_hits.append({**hit, "rank": rank, "source_alias_chunk_ids": [a["chunk_id"] for a in available]})
                        key = (q["query_id"], hit["text_sha256"])
                        candidate = pool.setdefault(key, {"query_id": q["query_id"], "ticker": ticker, "pool_id": identity(key), "text": hit["text"],
                            "judgment_status": "UNJUDGED", "ranks_by_arm": {}, "sources_by_arm": {}})
                        candidate["ranks_by_arm"][arm] = rank
                        candidate["sources_by_arm"][arm] = [{"chunk_id": a["chunk_id"], "document_id": a["document_id"], "publication_date": a["publication_date"],
                            "source_url": a["source_url"], "citation_span": a["citation_span"], "page": a["page"], "source_map": maps[a["chunk_id"]]} for a in available]
                    rankings.append({"query_id": q["query_id"], "ticker": ticker, "query": q["claim"], "cutoff_date": cutoff,
                        "arm": arm, "corpus_fingerprint": fingerprint, "search_ms": search_ms, "ranking": ranked_hits})
                write_jsonl(output / "rankings/dev" / arm / ticker / f"{cutoff}.jsonl", rankings)
            print(f"{ticker} {arm}: compared {len(qrows)} development claims", flush=True)
    # Denominators, source identities and paired questions must agree before aggregation.
    if len(all_results) != len(questions) * 4:
        raise RuntimeError("Incomplete paired run")
    for ticker in foundation["companies"]:
        for cutoff in {q["cutoff_date"] for q in questions if q["ticker"] == ticker}:
            if len({r["eligible_source_ids_sha256"] for r in index_reports if r["ticker"] == ticker and r["cutoff"] == cutoff}) != 1:
                raise RuntimeError("Eligible corpus differs across arms")
    aggregates = {a: summarize([r for r in all_results if r["arm"] == a]) for a in sources_root}
    for arm, summary in aggregates.items():
        ir = [r for r in index_reports if r["arm"] == arm]
        summary.update(indexed_chunks=sum(r["indexed_chunks"] for r in ir), exact_text_duplicates=sum(r["exact_text_duplicates"] for r in ir),
            index_bytes=sum(r["index_bytes"] for r in ir), alias_bytes=sum(r["alias_bytes"] for r in ir), index_build_seconds=sum(r["index_build_ms"] for r in ir)/1000,
            mean_indexed_words=sum(r["indexed_word_sum"] for r in ir)/sum(r["indexed_chunks"] for r in ir),
            repeated_source_character_fraction=sum(r["represented_source_character_occurrences"] for r in ir)/sum(r["source_nonwhitespace_characters"] for r in ir)-1,
            over_limit_indexed_chunks=sum(r["over_limit_indexed_chunks"] for r in ir))
    write_jsonl(output / "per_query.jsonl", all_results)
    with (output / "per_query.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_results[0]))
        writer.writeheader(); writer.writerows(all_results)
    write_json(output / "indexing_report.json", index_reports)
    write_json(output / "input_identities.json", input_identities)
    write_jsonl(output / "development_candidate_pool.jsonl", list(pool.values()))
    intervals = bootstrap(all_results)
    write_json(output / "paired_bootstrap.json", intervals)
    report = {"status": "development_diagnostic_complete_formal_acceptance_pending", "configuration": configuration,
        "companies": len(foundation["companies"]), "development_claims": len(questions), "heldout_claims_evaluated": 0,
        "retrieval_cases": len(all_results), "aggregates": aggregates, "source_metadata_errors": len(citation_errors),
        "candidate_pool_size": len(pool), "labels_reviewed": labels["evaluation_ready"], "winner_selected": False,
        "api_requests": 0, "api_cost_usd": 0, "local_compute_cost_usd": None,
        "elapsed_seconds": round(time.monotonic()-started, 2),
        "strata": {origin: {a: summarize([r for r in all_results if r["arm"] == a and r["origin"] == origin]) for a in sources_root}
                   for origin in sorted({q["origin"] for q in questions})},
        "semantic_context_completeness": None, "semantic_citation_accuracy": None,
        "unmeasured": ["Independently reviewed relevance/contradiction labels", "Original-source semantic context/citation correctness", "Final verdict/answer quality"]}
    write_json(output / "comparison.json", report)
    write_jsonl(output / "source_metadata_errors.jsonl", citation_errors)
    for path in (Path(__file__).resolve(), root / "finsight/agents/narrative/processing/index.py", root / "scripts/narrative/datasets/narrative_reference_mapping.py"):
        destination = output / "code" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    inventory = [{"path": p.relative_to(output).as_posix(), "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "comparison_manifest.json", {"files": inventory})
    prefix = output.relative_to(root).as_posix()
    table = "\n".join(f"| {a} | {s['recall@5']:.2%} | {s['recall@10']:.2%} | {s['mrr@10']:.4f} | {s['mean_indexed_words']:.1f} | {s['search_median_ms']:.1f} | {s['index_bytes']/1024**2:.1f} |" for a,s in aggregates.items())
    md = f"""# Step 4 - C1-C4 development comparison (provisional labels)

Status: **Development diagnostic complete; formal acceptance/selection pending.**
Compared {len(questions)} unchanged development claims across all 16 companies,
four arms and {len(all_results)} BM25 retrieval cases. Reserved claims evaluated: **0**.
Labels remain drafts. These are seed-reference diagnostics, not validated benchmark
scores or evidence of a winning production strategy.

## Results

| Arm | Seed bundle Recall@5 | Seed bundle Recall@10 | MRR@10 | Mean indexed words | Median search ms | Index MiB |
|---|---:|---:|---:|---:|---:|---:|
{table}

Recall is macro-averaged per claim over unique labelled fact bundles. A bundle
includes its required header/context spans; retrieving a header alone earns no
fact recall. Source-position unions support evidence split across retrieved chunks.
MRR@10 counts the first individually complete fact bundle in one result. Cumulative
multi-result bundle completion is reported separately as `bundle_completion_rr@10`.
Background risk is distinct from proposed direct contradiction; support/direct-
counter subgroup scores and their denominators are in `comparison.json`.

## Controlled inputs and reuse

Same frozen source build `{foundation['build_identity']}`, claims, per-claim cutoff,
eligibility, P0, BM25 k1=1.5 / b=0.75 and top K=5/10 across all arms. Existing
`index.build_index` and `index.search_index` are reused, including citation-location
diversification/deferred hits; this is not pure unconstrained BM25 sorting. The
query is the unchanged claim, without gold support/counter queries or decompositions.
Source IDs are checked identical per company/cutoff before aggregating scores.

Company/date/eligibility filtering precedes exact-text deduplication. Exact-text
groups occupy one ranking slot, with all eligible literal source aliases retained
and made citable. Source-span matching checks those explicit aliases; exact text
is not asserted to establish semantic event equivalence. Semantic release/exhibit
duplicate groups still require review. No chunk IDs or labels are rewritten.

No reserved questions/labels were used for retrieval or quality evaluation; file
hashes were checked for integrity only. Production indexes/canonical outputs and
all previous baselines/results remain unchanged. No embeddings, cross-encoder,
LLM, generation or API calls were used.

## Technical and methodological limits

C4 passed the recorded token hard rules. Fixed C1-C3 retain their token overflows;
BM25 can search them, but they remain incompatible with the current embedding
limit without an explicitly declared representation/model change. Source coverage
and structure/context flags remain those measured in Step 3; this run does not
turn proxy context scores into semantic validation.

New drafts are source-derived and some contain synthetic numeric perturbations;
lexical overlap can favor BM25. Reporting-period/scope context and relevance labels
are not independently adjudicated. Original approved claims and new source drafts
are reported separately in `strata`. Sparse seeds miss other relevant evidence:
unjudged results remain **UNJUDGED**, never assumed irrelevant. Precision/nDCG,
semantic context completeness and semantic citation accuracy are not asserted.
Source/company/date/citation metadata errors: {len(citation_errors)}; this is not
an entailment/citation-support accuracy score. Original PDF accuracy and the INTC
single-block issues remain unresolved.

Exploratory paired differences versus C1 use 2,000 company-cluster bootstrap draws
(16 clusters; seed 20261018), preserving within-company dependence. Intervals are
conditional on these draft development claims and do not repair annotation bias
or establish heldout generalization. One timed search per case: index building
and loading excluded from query latency, measured separately. Single-process arm
order/cache effects are not controlled sufficiently for hardware performance claims.

## Outputs - full paths from project root

- `{prefix}/comparison.json`: aggregate metrics, denominators and development strata.
- `{prefix}/per_query.csv` and `{prefix}/per_query.jsonl`: 368 claim/arm rows.
- `{prefix}/rankings/dev/<C1|C2|C3|C4>/<TICKER>/<CUTOFF>.jsonl`: ranked passages and source aliases.
- `{prefix}/indexes/<C1|C2|C3|C4>/<TICKER>/<CUTOFF>/index.json`: isolated BM25 indexes.
- `{prefix}/indexes/<C1|C2|C3|C4>/<TICKER>/<CUTOFF>/source_aliases.jsonl`: exact-text alias ledger.
- `{prefix}/indexing_report.json`: build time, storage, lengths and repeated-source-character counts.
- `{prefix}/development_candidate_pool.jsonl`: {len(pool)} pooled passages awaiting relevance review.
- `{prefix}/paired_bootstrap.json`: exploratory paired intervals.
- `{prefix}/configuration.json` and `{prefix}/input_identities.json`: source/query/code hashes and exact settings.
- `{prefix}/comparison_manifest.json`: output integrity inventory.

## Inspect and reproduce

Read-only:

```powershell
Get-Content -Raw {prefix}/comparison.json
Import-Csv {prefix}/per_query.csv | Select-Object -First 4
python -m scripts.narrative.chunking.compare_narrative_chunking --run-id {args.run_id} --dataset-id {args.dataset_id} --comparison-id {args.comparison_id} --verify
```

New isolated development version (draft gate explicit; no reserved evaluation):

```powershell
python -u -m scripts.narrative.chunking.compare_narrative_chunking --run-id {args.run_id} --dataset-id {args.dataset_id} --comparison-id {args.comparison_id}_next --allow-draft
```

Elapsed: {report['elapsed_seconds']} seconds. Artifact bytes before reports:
{sum(f['bytes'] for f in inventory):,}. API requests/cost: **0**. Local compute cost
in dollars was not measured.

## Decision and next action

**No final strategy selected or promoted.** Review/correct the development
questions/context/labels and pooled candidate judgments, resolve structure and
token compatibility decisions, then version the reviewed labels and rerun the
same four arms. Final reserved evaluation requires an approved, frozen release
and predeclared selection criteria. Do not treat these provisional results as
completion of the formal Phase 1 acceptance gate.
"""
    (output / "step_04_chunking_comparison.md").write_text(md, encoding="utf-8")
    (run / "reports/step_04_chunking_comparison.md").write_text(md, encoding="utf-8")
    write_json(run / "reports/step_04_chunking_comparison.json", report)
    print(json.dumps({"status": report["status"], "cases": len(all_results), "aggregates": aggregates, "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
