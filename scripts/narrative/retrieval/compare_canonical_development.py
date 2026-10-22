"""Run canonical 800/120 rules on the same 92 development claims as C4.

Builds isolated passages on the SAME frozen parsed sources as C4, not the old
canonical parser snapshot. Reuses P0 retrieval and source-bundle scoring. Long
embedding/reranker inputs require an explicit --overflow-policy truncate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re

from finsight.agents.narrative.corpus.processing.chunk import chunk_blocks
from finsight.agents.narrative.corpus.processing.corpus import file_sha256
from finsight.agents.narrative.corpus.processing.index import build_index
from finsight.agents.narrative.corpus.processing.ingest import ProcessedBlock
from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index, load_qdrant_index
from finsight.agents.narrative.retrieval.retrieval_experiments import (
    SentenceTransformersBackend,
    SentenceTransformersReranker,
    _diverse,
)
from finsight.agents.narrative.retrieval.source_priority import assign_priority
from scripts.narrative.chunking.compare_narrative_chunking import evaluate, evidence_bundles, summarize
from scripts.narrative.datasets.narrative_reference_mapping import covers
from scripts.narrative.datasets.prepare_narrative_labels import jsonl, write_jsonl
from scripts.narrative.experiments.prepare_narrative_experiment import identity, write_json
from scripts.narrative.retrieval.compare_narrative_retrieval_priority import Features, Methods, model_snapshot


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_block_chunks(block: ProcessedBlock, line: int, source_sha: str):
    """Apply the real canonical chunker and map every output to literal source spans."""
    chunks = chunk_blocks([block], 800, 120)
    words = list(re.finditer(r"\S+", block.text))
    lines = [(m.group().strip(), m.start(), m.end()) for m in re.finditer(r"[^\n]+", block.text) if m.group().strip()]
    table_cursor = 2
    mapped = []
    for ordinal, original in enumerate(chunks):
        chunk = replace(original, chunk_id=f"{original.chunk_id}:frozen-block-{line}")
        if block.content_type == "chart":
            spans = [[0, len(block.text)]]
        elif block.content_type == "table":
            if len(words) <= 800 or len(lines) <= 2:
                spans = [[0, len(block.text)]]
            else:
                selected = chunk.text.splitlines()
                body = selected[2:]
                expected = [row[0] for row in lines[table_cursor : table_cursor + len(body)]]
                if selected[:2] != [row[0] for row in lines[:2]] or body != expected:
                    raise ValueError(f"Canonical table/source mapping mismatch at block {line}")
                spans = [[a, z] for _, a, z in lines[:2] + lines[table_cursor : table_cursor + len(body)]]
                table_cursor += len(body)
        else:
            start = ordinal * (800 - 120)
            selected = words[start : start + 800]
            spans = [[selected[0].start(), selected[-1].end()]]
        represented = " ".join(block.text[a:z] for a, z in spans).split()
        if represented != chunk.text.split():
            raise ValueError(f"Canonical chunk text/source mapping mismatch at block {line}")
        mapped.append(
            (
                chunk,
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": block.document_id,
                    "source_sha256": source_sha,
                    "block_line_1based": line,
                    "source_char_start": min(a for a, _ in spans),
                    "source_char_end": max(z for _, z in spans),
                    "source_char_spans": spans,
                },
            )
        )
    if block.text.strip() and not covers(
        block.text, 0, len(block.text), [s for _, m in mapped for s in m["source_char_spans"]]
    ):
        raise ValueError(f"Canonical chunks lose source content at block {line}")
    return mapped


class RecordedEmbedding:
    """Use the existing model, recording every input that needs native truncation."""

    def __init__(self, base, policy):
        self.base, self.policy = base, policy
        self.model_name = base.model_name
        self.documents, self.over_limit, self.maximum = 0, 0, 0

    def counts(self, texts):
        ids = self.base.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)["input_ids"]
        return [len(tokens) for tokens in ids]

    def encode_documents(self, texts):
        counts = self.counts(texts)
        limit = int(self.base.model.max_seq_length)
        over = sum(n > limit for n in counts)
        self.documents += len(counts)
        self.over_limit += over
        self.maximum = max([self.maximum] + counts)
        if over and self.policy == "error":
            raise ValueError(
                f"{over} passages exceed {limit} embedding tokens. "
                "Use --overflow-policy truncate to explicitly record native truncation."
            )
        return self.base.encode_documents(texts)

    def encode_query(self, text):
        if self.counts([text])[0] > int(self.base.model.max_seq_length):
            raise ValueError("Claim exceeds embedding query limit; query truncation is prohibited")
        return self.base.encode_query(text)


class RecordedMethods(Methods):
    """Same R4 scoring/diversity, with explicit native pair-truncation accounting."""

    def __init__(self, *args, overflow_policy, **kwargs):
        super().__init__(*args, **kwargs)
        self.overflow_policy = overflow_policy
        self.over_limit_pairs = 0

    def rerank(self, query, candidates, cache=True):
        missing = [c for c in candidates if not cache or (query, c["chunk_id"]) not in self.score_cache]
        values = {}
        if missing:
            model = self.reranker.model
            limit = int(getattr(model, "max_seq_length", None) or model.tokenizer.model_max_length)
            ids = model.tokenizer(
                [query] * len(missing), [c["text"] for c in missing], truncation=False, add_special_tokens=True
            )["input_ids"]
            counts = [len(tokens) for tokens in ids]
            self.max_pair_tokens = max([self.max_pair_tokens] + counts)
            over = sum(n > limit for n in counts)
            self.over_limit_pairs += over
            if over and self.overflow_policy == "error":
                raise ValueError(f"{over} reranker pairs exceed {limit} tokens; explicit truncation policy required")
            scores = self.reranker.score(query, [c["text"] for c in missing])
            if len(scores) != len(missing):
                raise ValueError("Reranker returned the wrong score count")
            values = {c["chunk_id"]: float(s) for c, s in zip(missing, scores)}
            self.pair_calls += len(missing)
            if cache:
                self.score_cache.update({(query, cid): score for cid, score in values.items()})
        ranked = [
            {**c, "retrieval_score": self.score_cache[(query, c["chunk_id"])] if cache else values[c["chunk_id"]]}
            for c in candidates
        ]
        ranked.sort(key=lambda c: (-c["retrieval_score"], c["chunk_id"]))
        return _diverse(ranked, len(ranked))


def prepare_company(run, company, ticker, questions, evidence, source_build):
    parsed = run / "parsed" / ticker
    inputs = read(parsed / "input_manifest.json")
    if file_sha256(parsed / "documents.jsonl") != inputs["blocks_sha256"]:
        raise ValueError(f"Frozen parsed input changed for {ticker}")
    rows = jsonl(parsed / "documents.jsonl")
    cutoffs = {q["cutoff_date"] for q in questions}
    if len(cutoffs) != 1:
        raise ValueError("This comparison requires one shared cutoff per company")
    cutoff = next(iter(cutoffs))
    sources = {s["document_id"]: s for s in inputs["sources"]}
    allowed = {
        key for key, s in sources.items() if s.get("retrieval_eligible", True) and s["publication_date"] <= cutoff
    }
    for e in evidence:
        if e["document_id"] not in allowed:
            raise ValueError("Reference source fails eligibility/cutoff")
        row = rows[e["block_line_1based"] - 1]
        if (
            row["document_id"] != e["document_id"]
            or row["text"][e["char_start"] : e["char_end"]] != e["quote"]
            or sources[e["document_id"]]["sha256"] != e["source_sha256"]
        ):
            raise ValueError("Evidence/source anchor mismatch")
        if hashlib.sha256(row["text"].encode()).hexdigest() != e["block_text_sha256"]:
            raise ValueError("Evidence block hash mismatch")
    chunks, maps = [], {}
    for line, row in enumerate(rows, 1):
        if row["document_id"] not in allowed:
            continue
        if row["company_ticker"] != ticker:
            raise ValueError("Frozen source company mismatch")
        for chunk, mapping in canonical_block_chunks(
            ProcessedBlock(**row), line, sources[row["document_id"]]["sha256"]
        ):
            chunks.append(chunk)
            maps[chunk.chunk_id] = mapping
    if {c.document_id for c in chunks} != allowed:
        raise ValueError("Canonical arm does not represent all eligible source documents")
    groups = defaultdict(list)
    for chunk in chunks:
        groups[chunk.text_sha256].append(chunk.to_json())
    fingerprint = identity(
        {
            "source_build": source_build,
            "ticker": ticker,
            "cutoff": cutoff,
            "chunker": file_sha256(Path("finsight/agents/narrative/corpus/processing/chunk.py")),
        }
    )
    payload = build_index(
        chunks, company / "index.json", corpus_fingerprint=fingerprint, target_words=800, overlap_words=120
    )
    aliases = {c["chunk_id"]: groups[c["text_sha256"]] for c in payload["chunks"]}
    write_jsonl(company / "source_map.jsonl", list(maps.values()))
    write_json(company / "source_aliases.json", aliases)
    return payload, aliases, maps, rows, cutoff, allowed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261018_all16_v1")
    parser.add_argument("--dataset-id", default="evidence_labels_v3")
    parser.add_argument("--experiment-id", default="canonical92_v1")
    parser.add_argument("--reference-experiment", default="c4_dev_v1")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--overflow-policy", choices=("error", "truncate"), default="error")
    parser.add_argument("--methods", nargs="+", choices=("R1", "R2", "R3", "R4"), default=["R1", "R2", "R3", "R4"])
    parser.add_argument("--ticker", action="append", help="Optional subset for diagnostics; omit for all 92 claims")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for value in [args.run_id, args.dataset_id, args.experiment_id, args.reference_experiment, *(args.ticker or [])]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            parser.error("IDs must use letters, numbers, hyphens or underscores")
    if args.threads < 1 or len(set(args.methods)) != len(args.methods):
        parser.error("Use positive threads and unique methods")
    root = Path.cwd().resolve()
    run = root / "data/narrative/experiments/narrative_rag" / args.run_id
    dataset = run / "labels" / args.dataset_id
    manifest, foundation = read(dataset / "dataset_manifest.json"), read(run / "foundation_manifest.json")
    if not manifest["evaluation_ready"] and not args.allow_draft:
        parser.error("Unreviewed development labels require --allow-draft")
    questions = jsonl(dataset / "dev/questions.jsonl")
    evidence = jsonl(dataset / "dev/evidence_labels.jsonl")
    if len(questions) != 92 or any(q["split"] != "dev" for q in questions):
        parser.error("Expected the recorded 92-claim development split; heldout is not evaluated")
    reference = read(run / "retrieval_priority" / args.reference_experiment / "comparison.json")
    config = reference["configuration"]
    checks = {
        "source_build_identity": foundation["build_identity"],
        "dataset_manifest_sha256": file_sha256(dataset / "dataset_manifest.json"),
        "questions_sha256": file_sha256(dataset / "dev/questions.jsonl"),
        "evidence_sha256": file_sha256(dataset / "dev/evidence_labels.jsonl"),
    }
    if manifest["foundation_build_identity"] != foundation["build_identity"] or any(
        config[k] != v for k, v in checks.items()
    ):
        raise ValueError("C4 reference and canonical run do not share source/query/label identities")
    tickers = sorted(set(args.ticker or [q["ticker"] for q in questions]))
    if not set(tickers) <= {q["ticker"] for q in questions}:
        parser.error("Unknown company selected")
    selected = [q for q in questions if q["ticker"] in tickers]
    output = run / "retrieval_canonical" / args.experiment_id
    if output.exists() and not args.resume:
        parser.error("Experiment exists; use --resume or a new --experiment-id")
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        **checks,
        "tickers": tickers,
        "methods": args.methods,
        "policy": "P0",
        "query_field": "claim",
        "chunking": "canonical chunk_blocks 800/120, on C4 frozen parsed sources",
        "overflow_policy": args.overflow_policy,
        "threads": args.threads,
        "reference_experiment": args.reference_experiment,
        "retrieval_config": config["policies"],
        "parsed_input_hashes": {
            t: {name: file_sha256(run / "parsed" / t / name) for name in ("documents.jsonl", "input_manifest.json")}
            for t in tickers
        },
        "code_hashes": {
            p: file_sha256(root / p)
            for p in (
                "scripts/narrative/retrieval/compare_canonical_development.py",
                "scripts/narrative/retrieval/compare_narrative_retrieval_priority.py",
                "scripts/narrative/chunking/compare_narrative_chunking.py",
                "finsight/agents/narrative/corpus/processing/chunk.py",
                "finsight/agents/narrative/corpus/processing/index.py",
                "finsight/agents/narrative/retrieval/qdrant_index.py",
                "finsight/agents/narrative/retrieval/retrieval_experiments.py",
            )
        },
    }
    if (output / "configuration.json").exists() and read(output / "configuration.json") != settings:
        raise ValueError("Resume configuration/code differs; use a new experiment ID")
    write_json(output / "configuration.json", settings)
    embedding = reranker = None
    if any(m != "R1" for m in args.methods):
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch

        torch.set_num_threads(args.threads)
        embed_path = model_snapshot("sentence-transformers/all-MiniLM-L6-v2")
        rerank_path = model_snapshot("cross-encoder/ms-marco-MiniLM-L-6-v2")
        for prefix, path in (("embedding", embed_path), ("reranker", rerank_path)):
            if (
                path.name != config[prefix + "_revision"]
                or file_sha256(path / "model.safetensors") != config[prefix + "_weight_sha256"]
            ):
                raise ValueError("Model revision/weight differs from the C4 reference")
        print("Loading pinned local models; no API calls", flush=True)
        base = SentenceTransformersBackend(str(embed_path))
        base.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embedding = RecordedEmbedding(base, args.overflow_policy)
        if "R4" in args.methods:
            reranker = SentenceTransformersReranker(str(rerank_path))
    all_metrics, all_indexes = [], []
    for number, ticker in enumerate(tickers, 1):
        company = output / "companies" / ticker
        checkpoint = company / "checkpoint.json"
        if checkpoint.exists():
            saved = read(checkpoint)
            if any(file_sha256(company / f["path"]) != f["sha256"] for f in saved["files"]):
                raise ValueError(f"Changed checkpoint: {ticker}")
            all_metrics.extend(jsonl(company / "per_query.jsonl"))
            all_indexes.append(read(company / "indexing_report.json"))
            print(f"[{number}/{len(tickers)}] {ticker}: reusing checkpoint", flush=True)
            continue
        print(f"[{number}/{len(tickers)}] {ticker}: canonical 800/120, 92-claim dataset", flush=True)
        company.mkdir(parents=True, exist_ok=True)
        qrows = [q for q in selected if q["ticker"] == ticker]
        erows = [e for e in evidence if e["query_id"] in {q["query_id"] for q in qrows}]
        payload, aliases, maps, rows, cutoff, allowed = prepare_company(
            run, company, ticker, qrows, erows, foundation["build_identity"]
        )
        c4_folder = run / "comparisons/bm25_dev_v1/indexes/C4" / ticker / cutoff
        c4_aliases = jsonl(c4_folder / "source_aliases.jsonl")
        if {a["document_id"] for group in c4_aliases for a in group["aliases"]} != allowed:
            raise ValueError("Canonical and C4 eligible source identities differ")
        priorities = {
            c["chunk_id"]: assign_priority(c, aliases[c["chunk_id"]], config["policies"]) for c in payload["chunks"]
        }
        vector = None
        over_before = embedding.over_limit if embedding else 0
        input_audit = None
        if embedding:
            counts = embedding.counts([c["text"] for c in payload["chunks"]])
            limit = int(embedding.base.model.max_seq_length)
            input_audit = {
                "limit": limit,
                "passages": len(counts),
                "over_limit": sum(n > limit for n in counts),
                "maximum_tokens": max(counts, default=0),
            }
            if input_audit["over_limit"] and args.overflow_policy == "error":
                raise ValueError(
                    "Canonical passages exceed the embedding limit; use explicit --overflow-policy truncate"
                )
            vectors = company / "qdrant"
            if (vectors / "qdrant_metadata.json").exists():
                vector = load_qdrant_index(
                    vectors, payload, expected_model_name=embedding.model_name, expected_dimension=384
                )
            else:
                build_qdrant_index(payload, vectors, embedding)
                vector = load_qdrant_index(
                    vectors, payload, expected_model_name=embedding.model_name, expected_dimension=384
                )
        methods = RecordedMethods(
            Features(payload, vector, embedding, ticker, cutoff),
            reranker,
            priorities,
            config["policies"],
            overflow_policy=args.overflow_policy,
        )
        metrics, rankings = [], []
        try:
            for q in qrows:
                for method in args.methods:
                    ranking, trace = methods.search(q["claim"], method, "P0")
                    values = evaluate(q, evidence_bundles(q, erows), ranking, aliases, maps, rows)
                    values.update(method=method, policy="P0", search_ms=trace["elapsed_ms"])
                    metrics.append(values)
                    rankings.append(
                        {
                            "query_id": q["query_id"],
                            "query": q["claim"],
                            "cutoff": cutoff,
                            "method": method,
                            "ranking": ranking,
                            "trace": trace,
                        }
                    )
        finally:
            if vector:
                vector.client.close()
        index_report = {
            "ticker": ticker,
            "eligible_documents": len(allowed),
            "indexed_chunks": len(payload["chunks"]),
            "eligible_source_ids_sha256": identity(sorted(allowed)),
            "embedding_input_audit": input_audit,
            "embedding_over_limit_documents_encoded_this_attempt": embedding.over_limit - over_before
            if embedding
            else None,
            "reranker_over_limit_pairs": methods.over_limit_pairs,
            "maximum_reranker_pair_tokens": methods.max_pair_tokens,
            "overflow_policy": args.overflow_policy,
        }
        write_jsonl(company / "per_query.jsonl", metrics)
        write_jsonl(company / "rankings.jsonl", rankings)
        write_json(company / "indexing_report.json", index_report)
        names = [
            "per_query.jsonl",
            "rankings.jsonl",
            "indexing_report.json",
            "index.json",
            "source_map.jsonl",
            "source_aliases.json",
        ]
        write_json(checkpoint, {"files": [{"path": n, "sha256": file_sha256(company / n)} for n in names]})
        all_metrics.extend(metrics)
        all_indexes.append(index_report)
    if len(all_metrics) != len(selected) * len(args.methods):
        raise ValueError("Incomplete claim/method run")
    summaries = {method: summarize([r for r in all_metrics if r["method"] == method]) for method in args.methods}
    c4_rows = jsonl(run / "retrieval_priority" / args.reference_experiment / "per_query.jsonl")
    selected_ids = {q["query_id"] for q in selected}
    c4_summaries = {
        method: summarize(
            [r for r in c4_rows if r["query_id"] in selected_ids and r["method"] == method and r["policy"] == "P0"]
        )
        for method in args.methods
    }
    report = {
        "status": "development_diagnostic_complete",
        "development_claims": len(selected),
        "cases": len(all_metrics),
        "companies": tickers,
        "heldout_evaluated": 0,
        "winner_selected": False,
        "configuration": settings,
        "summaries": summaries,
        "c4_reference_summaries": c4_summaries,
        "indexing": all_indexes,
        "api_requests": 0,
        "api_cost_usd": 0,
        "limitations": [
            "Draft source-grounded labels, not final accuracy labels.",
            "Canonical rules rebuilt on the same frozen parsed source as C4; "
            "not the historical canonical index/parser.",
            "Native truncation, if explicitly enabled, is a representation trade-off; "
            "full passage scoring can credit text outside the embedded/reranked prefix.",
            "Cached method timings are diagnostics, not independent cold deployment latency.",
        ],
    }
    write_jsonl(output / "per_query.jsonl", all_metrics)
    with (output / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0]))
        writer.writeheader()
        writer.writerows(all_metrics)
    write_json(output / "comparison.json", report)
    table = "\n".join(
        f"| {m} | {s['recall@5']:.2%} | {s['recall@10']:.2%} | {s['mrr@10']:.4f} | {c4_summaries[m]['recall@10']:.2%} |"
        for m, s in summaries.items()
    )
    (output / "report.md").write_text(
        f"# Canonical rules on the 92-claim development dataset\n\n"
        f"Completed {len(selected)} claims / {len(all_metrics)} cases. P0; source/claim/label identities match C4.\n\n"
        "| Method | Canonical Recall@5 | Canonical Recall@10 | Canonical MRR@10 | C4 reference Recall@10 |\n"
        "|---|---:|---:|---:|---:|\n"
        + table
        + "\n\nOverflow policy: `"
        + args.overflow_policy
        + "`.\n\n"
        + "\n".join("- " + s for s in report["limitations"])
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "claims": len(selected),
                "cases": len(all_metrics),
                "output": str(output),
                "summaries": summaries,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
