"""Score one contextual company index against the canonical BM25 index.

This is a Workflow D style check for the narrative claims of a single company.
It reports whether each stated figure appears in the top source passages.
It does not call an LLM and does not assign a verdict.

    python -m scripts.narrative.evaluation.score_contextual_claims --ticker AMZN
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from finsight.agents.narrative.corpus.processing.contextual import (
    figure_phrases,
    followup_query,
    passage_states_figures,
    split_claim_parts,
)
from finsight.agents.narrative.corpus.processing.index import load_index, search_index


AMZN_CLAIMS = (
    {
        "id": "03",
        "claim": "Amazon's AWS grew 28% in Q1 2026.",
        "cutoff": "2026-04-30",
    },
    {
        "id": "19",
        "claim": "AWS grew 24% year over year in Q4 2025 with a $142 billion revenue run rate.",
        "cutoff": "2026-04-13",
    },
)


def score_index(payload: dict, claim: str, cutoff: str, top_k: int) -> dict:
    found: dict[str, list[str]] = {phrase: [] for phrase in figure_phrases(claim)}
    hits = []
    for part in split_claim_parts(claim):
        matches = search_index(payload, part, top_k, date.fromisoformat(cutoff), payload_ticker(payload))
        hits.extend(matches)
    passages = [hit["text"] for hit in hits]
    types = [hit["document_type"] for hit in hits]
    follow = followup_query(claim, passages, types)
    if follow:
        hits.extend(search_index(payload, follow, top_k, date.fromisoformat(cutoff), payload_ticker(payload)))
        passages = [hit["text"] for hit in hits]
        types = [hit["document_type"] for hit in hits]
    for phrase in found:
        found[phrase] = sorted({hit["document_type"] for hit in hits if phrase in _flat(hit["text"])})
    return {
        "figures_found": {phrase: bool(docs) for phrase, docs in found.items()},
        "figure_documents": found,
        "top_document_types": types[:top_k],
        "followup": follow,
        "stated": passage_states_figures(claim, "\n".join(passages)),
    }


def payload_ticker(payload: dict) -> str:
    return str(payload["chunks"][0]["company_ticker"]).upper()


def _flat(text: str) -> str:
    return " ".join(text.lower().replace(",", "").replace("$", "").split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AMZN")
    parser.add_argument("--processed-root", type=Path, default=Path("data/narrative/processed"))
    parser.add_argument("--build-version", default="bge512_v1")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()
    if ticker != "AMZN":
        parser.exit(2, "The bundled narrative claims are the Amazon items. Build another company before scoring it.\n")
    canonical = load_index(args.processed_root / ticker / "indexes" / "index.json")
    contextual_path = (
        args.processed_root / "chunking_strategies" / "contextual" / args.build_version / ticker / "indexes" / "index.json"
    )
    contextual = load_index(contextual_path)
    rows = []
    for item in AMZN_CLAIMS:
        rows.append(
            {
                "id": item["id"],
                "claim": item["claim"],
                "canonical": score_index(canonical, item["claim"], item["cutoff"], args.top_k),
                "contextual": score_index(contextual, item["claim"], item["cutoff"], args.top_k),
            }
        )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
