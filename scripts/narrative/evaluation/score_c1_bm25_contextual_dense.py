"""Fuse C1 BM25 with contextual dense and score like the comparison runner.

BM25 reads the C1 passages. Dense reads the contextual row vectors. Reciprocal
rank fusion keeps the top 10, including the top two BM25 hits. No model calls.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from finsight.agents.narrative.corpus.processing.index import load_index, search_index
from finsight.agents.narrative.rag.collect_evidence import build_queries
from finsight.agents.narrative.retrieval.qdrant_index import load_qdrant_index, reciprocal_rank_fusion, search_qdrant
from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend
from finsight.contracts.schemas import SubClaim, SubClaimType
from scripts.narrative.evaluation.evaluate_rag_pilot import load_reference, score_direction

CASES = Path("data/narrative/evaluation/test_claims_v2/cases.csv")
C1 = Path("data/narrative/processed/chunking_strategies/C1/full_v1")
DENSE = Path("data/narrative/processed/chunking_strategies/contextual/row512_v1")
OUTPUT = Path("data/narrative/evaluation/test_claims_v2_result/retrieval_C1_bm25_contextual_dense/comparison.md")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WORKSPACE = Path(".")


def source_map(folder: Path) -> dict[str, dict]:
    rows = {}
    for line in (folder / "source_map.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["chunk_id"]] = row
    return rows


def main() -> None:
    cases = list(csv.DictReader(CASES.open(encoding="utf-8-sig")))
    labeled = [case for case in cases if case["type"] != "unsupported"]
    tickers = sorted({case["company"] for case in labeled})
    print(f"cases {len(labeled)} tickers {len(tickers)}", flush=True)
    embedder = SentenceTransformersBackend(MODEL)
    c1_index = {ticker: load_index(C1 / ticker / "indexes" / "index.json") for ticker in tickers}
    dense_index = {}
    dense_vectors = {}
    maps: dict[str, dict] = {}
    for ticker in tickers:
        payload = load_index(DENSE / ticker / "indexes" / "index.json")
        dense_index[ticker] = payload
        dense_vectors[ticker] = load_qdrant_index(
            DENSE / ticker / "indexes" / "qdrant", payload, expected_model_name=MODEL,
        )
        maps.update(source_map(C1 / ticker))
        maps.update(source_map(DENSE / ticker))
        print(f"loaded {ticker}", flush=True)

    totals = {direction: [] for direction in ("support", "counter")}
    reference_cache: dict = {}
    try:
        for number, case in enumerate(labeled, 1):
            ticker = case["company"]
            claim = SubClaim(
                sub_claim_id=case["id"], text=case["claims"], company_ticker=ticker,
                sub_claim_type=SubClaimType.OTHER,
            )
            support, counter = build_queries(claim)
            cutoff = date.fromisoformat(case["cut_off"])
            reference, _, _ = load_reference(case, WORKSPACE, C1, reference_cache)
            for direction, query in (("support", support), ("counter", counter)):
                bm25 = search_index(c1_index[ticker], query, 50, cutoff, ticker)
                dense = search_qdrant(
                    dense_vectors[ticker], dense_index[ticker], embedder.encode_query(query),
                    top_k=50, cutoff_date=cutoff, company_ticker=ticker,
                )
                fused = reciprocal_rank_fusion(bm25, dense, top_k=10)
                totals[direction].append(score_direction(fused, reference, maps))
            if number % 12 == 0 or number == len(labeled):
                print(f"scored {number}/{len(labeled)}", flush=True)
    finally:
        for vectors in dense_vectors.values():
            vectors.client.close()

    lines = [
        "# C1 BM25 fused with contextual dense",
        "",
        "BM25 uses the C1 passages. Dense uses the contextual row vectors. "
        "Fusion keeps the top 10. Same 107 v2 claims and same span scoring as the other retrieval tables.",
        "",
        "| Strategy | Retrieval | Direction | Source Hit@5 | Source Hit@10 | Recall@5 | Recall@10 | MRR@5 | MRR@10 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    print("TABLE")
    for direction in ("support", "counter"):
        rows = totals[direction]
        count = len(rows)

        def rate(name: str) -> str:
            return f"{100 * sum(row[name] for row in rows) / count:.2f}%"

        def mean(name: str) -> str:
            return f"{sum(row[name] for row in rows) / count:.4f}"

        print(
            f"{direction}\t{rate('source_hit_at_5')} / {rate('source_hit_at_10')}\t"
            f"{rate('known_reference_recall_at_5')} / {rate('known_reference_recall_at_10')}\t"
            f"{mean('mrr_at_5')}\t{mean('mrr_at_10')}",
            flush=True,
        )
        lines.append(
            "| C1 BM25 + contextual dense | hybrid | "
            f"{direction} | {rate('source_hit_at_5')} | {rate('source_hit_at_10')} | "
            f"{rate('known_reference_recall_at_5')} | {rate('known_reference_recall_at_10')} | "
            f"{mean('mrr_at_5')} | {mean('mrr_at_10')} |"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
