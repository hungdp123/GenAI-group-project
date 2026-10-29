"""Retrieve evidence, then use an LLM for cited facts, summary and claim stance.

The basic_rag command name is retained for compatibility. This is now a
evidence-preparation and stance entry point. Historical verdict
experiments explicitly use rag.verdict_baseline.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any

from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent, build_queries
from finsight.agents.narrative.rag.evidence_summary import prepare_narrative_evidence
from finsight.agents.narrative.retrieval.retriever import BACKENDS
from finsight.contracts.schemas import SubClaim, SubClaimType


def run_basic_narrative_rag(
    ticker: str,
    claim: str,
    cutoff: date,
    *,
    backend: str = "bm25",
    top_k: int = 3,
    candidate_k: int = 50,
    corpus_root: Path | str | None = None,
    max_tokens: int = 1500,
    ollama_model: str | None = None,
    ollama_host: str = "http://localhost:11434",
    ollama_context: int = 8192,
    experiment_dir: Path | str | None = None,
    stance_method: str | None = None,
    retrieval_only: bool = False,
    reranker_model: str | None = None,
    rerank_overflow: str | None = None,
    part_search: bool = False,
) -> dict[str, Any]:
    """Return original passages, exact facts, cited summary and a reasoned claim stance.

    EvidenceItem internally requires stance=NEUTRAL; public JSON omits that
    compatibility placeholder. The claim-level stance is separate from passage labels.
    """
    ticker = ticker.strip().upper()
    if not 1 <= top_k <= 25 or candidate_k < top_k:
        raise ValueError("Require 1 <= top_k <= 25 and candidate_k >= top_k")
    if stance_method not in {None, "none", "exact", "llm"}:
        raise ValueError("stance_method must be none, exact or llm (legacy retrieval option)")
    if stance_method in {"exact", "llm"}:
        logging.getLogger(__name__).warning(
            "Legacy passage stance options are ignored; the LLM produces a separate claim stance"
        )
    sub_claim = SubClaim(
        sub_claim_id="basic-rag-claim", text=claim, company_ticker=ticker, sub_claim_type=SubClaimType.OTHER
    )
    experiment = Path(experiment_dir).resolve() if experiment_dir is not None else None
    if experiment is not None and not (experiment / "companies" / ticker / "index.json").is_file():
        raise FileNotFoundError(f"Experiment index is missing for {ticker}: {experiment}")

    collector = NarrativeEvidenceAgent(
        corpus_root=corpus_root,
        backend=backend,
        top_k=top_k,
        candidate_k=candidate_k,
        stance_method="none",
        reranker_model=reranker_model,
        rerank_overflow=rerank_overflow,
        part_search=part_search,
        index_template=str(experiment / "companies/{ticker}/index.json") if experiment is not None else None,
        qdrant_template=str(experiment / "companies/{ticker}/qdrant") if experiment is not None else None,
    )
    started = time.perf_counter()
    items = collector.gather(sub_claim, cutoff)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    candidates = items
    if retrieval_only:
        preparation = {
            "status": "OK" if items else "NO_EVIDENCE",
            "assessment_status": "NOT_ASSESSED",
            "stance": None,
            "evidence_preparation_status": "SKIPPED_RETRIEVAL_ONLY",
            "facts": [],
            "summary": [],
            "selection_applied": False,
            "candidate_evidence": [i.model_dump(mode="json", exclude={"stance"}) for i in items],
            "llm": {
                "provider": "none",
                "model": None,
                "api_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
            "citation_validation": {"passed": None, "status": "NOT_RUN", "errors": []},
            "consistency_check": {"status": "NOT_APPLICABLE", "reason": "No stance or claim verdict is generated."},
        }
    else:
        items, preparation = prepare_narrative_evidence(
            sub_claim,
            cutoff,
            items,
            max_tokens=max_tokens,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            ollama_context=ollama_context,
        )
    return {
        **preparation,
        "pipeline": "narrative_evidence_rag" if not retrieval_only else "narrative_evidence_retrieval",
        "ticker": ticker,
        "claim": claim,
        "cutoff_date": cutoff.isoformat(),
        "backend": backend,
        "reranking": (collector.last_search or {}).get("reranking"),
        "index": (collector.last_search or {}).get("index"),
        "retrieval": {
            "count": len(candidates),
            "elapsed_ms": elapsed_ms,
            "queries": dict(zip(("support", "counter"), build_queries(sub_claim))),
            "evidence_ids": [item.evidence_id for item in candidates],
        },
        "selection": {"returned_count": len(items), "returned_evidence_ids": [i.evidence_id for i in items]},
        "evidence": [item.model_dump(mode="json", exclude={"stance"}) for item in items],
        "contract_note": "Internal EvidenceItem.stance=NEUTRAL is an unassessed compatibility "
        "placeholder; public evidence omits stance.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--cutoff", required=True, type=date.fromisoformat)
    parser.add_argument("--backend", type=str.lower, choices=BACKENDS, default="bm25")
    parser.add_argument("--top-k", type=int, default=3, help="Per support/counter search")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--reranker-model")
    parser.add_argument("--rerank-overflow", choices=("error", "truncate"))
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stance-method", choices=("none", "exact", "llm"),
        help="Deprecated passage option; claim stance runs unless retrieval-only",
    )
    parser.add_argument("--retrieval-only", action="store_true", help="Explicitly skip LLM evidence preparation")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="After BM25 and semantic retrieval, rerank the fused list before the top passages go to the LLM",
    )
    parser.add_argument(
        "--part-search",
        action="store_true",
        help="Search each figured part of the claim, then once more if the figure is missing",
    )
    parser.add_argument("--max-tokens", type=int, default=1500, help="Output limit for the evidence preparation call")
    parser.add_argument("--ollama-model", help="Local model for evidence selection, facts, summary and claim stance")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--ollama-context", type=int, default=8192)
    args = parser.parse_args()
    if args.rerank:
        args.backend = "r4"
        if args.rerank_overflow is None:
            args.rerank_overflow = "truncate"
    try:
        result = run_basic_narrative_rag(
            args.ticker,
            args.claim,
            args.cutoff,
            backend=args.backend,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            corpus_root=args.corpus_root,
            max_tokens=args.max_tokens,
            ollama_model=args.ollama_model,
            ollama_host=args.ollama_host,
            ollama_context=args.ollama_context,
            experiment_dir=args.experiment_dir,
            stance_method=args.stance_method,
            retrieval_only=args.retrieval_only,
            reranker_model=args.reranker_model,
            rerank_overflow=args.rerank_overflow,
            part_search=args.part_search,
        )
    except Exception as exc:
        parser.exit(2, f"Narrative retrieval failed: {type(exc).__name__}: {exc}\n")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", result["llm"].get("model") or "none")
    output = args.output or Path("data/narrative/experiments/standalone_rag") / (
        f"{result['ticker']}-{model_slug}-evidence-{stamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    print(f"Saved Narrative evidence report: {output}", file=sys.stderr)
    if result["status"] not in {"OK", "NO_EVIDENCE"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
