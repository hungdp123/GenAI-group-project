"""Run canonical or isolated 92-claim Narrative Evidence Agent integration checks.

Uses exact stance classification, cached embeddings and no LLM API calls.
Use --experiment-dir and --queries for the isolated development experiment.
Without arguments, retains the legacy 16-claim production-index batch.
"""


def legacy_main() -> None:
    import json
    from pathlib import Path
    from datetime import date, datetime, timezone
    from collections import Counter
    from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent
    from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
    from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend
    from finsight.contracts.schemas import SubClaim, SubClaimType

    root = Path("data/narrative/processed")
    out = Path("data/narrative/experiments/narrative_agent/task7")
    out.mkdir(parents=True, exist_ok=True)
    queries = [
        json.loads(line)
        for line in Path("docs/narrative/evaluation/retrieval_queries_v2.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks = []
    embedding = None
    for backend in ["bm25", "dense", "hybrid"]:
        if backend != "bm25" and embedding is None:
            embedding = SentenceTransformersBackend("sentence-transformers/all-MiniLM-L6-v2")
        for q in queries:
            ticker = q["ticker"]
            print(f"[{backend}] {ticker}", flush=True)

            def factory(t):
                p = root / t / "indexes/index.json"
                return CorpusRetriever(
                    index_path=p,
                    company_ticker=t,
                    qdrant_dir=p.parent / "qdrant",
                    backend=backend,
                    embedding_backend=embedding,
                )

            agent = NarrativeEvidenceAgent(corpus_root=root, backend=backend, retriever_factory=factory)
            claim = SubClaim(
                sub_claim_id="task7-" + q["query_id"],
                text=q["claim"],
                company_ticker=ticker,
                sub_claim_type=SubClaimType.DEMAND_ATTRIBUTION,
            )
            items = agent.gather(claim, date.fromisoformat(q["cutoff_date"]))
            index = json.loads((root / ticker / "indexes/index.json").read_text(encoding="utf-8"))
            byid = {c["chunk_id"]: c for c in index["chunks"]}
            errors = []
            if not items:
                errors.append("No candidates returned for recorded development claim")
            if len({e.evidence_id for e in items}) != len(items):
                errors.append("Duplicate evidence IDs")
            for e in items:
                c = byid.get(e.evidence_id)
                if (
                    not c
                    or c["company_ticker"] != ticker
                    or e.content != c["text"]
                    or e.sub_claim_id != claim.sub_claim_id
                    or e.publication_date > date.fromisoformat(q["cutoff_date"])
                ):
                    errors.append(e.evidence_id)
            name = f"{ticker}_{backend}.json"
            (out / name).write_text(
                json.dumps([e.model_dump(mode="json") for e in items], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            checks.append(
                {
                    "ticker": ticker,
                    "backend": backend,
                    "evidence_count": len(items),
                    "errors": errors,
                    "corpus_fingerprint": index["corpus_fingerprint"],
                    "stances": dict(Counter(e.stance.value for e in items)),
                    "missing_urls": sum(not e.url for e in items),
                    "output": (out / name).as_posix(),
                }
            )
    report = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(not c["errors"] for c in checks) else "partial",
        "check_count": len(checks),
        "error_count": sum(len(c["errors"]) for c in checks),
        "scope": "16 company claims x 3 backends; exact stance; no LLM requests or semantic accuracy judgments",
        "checks": checks,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))



def main() -> None:
    """Use isolated experiment indexes when requested; retain the legacy default."""
    import argparse
    import hashlib
    import json
    import os
    from collections import Counter
    from datetime import date, datetime, timezone
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Check Narrative Evidence Agent integration; no LLM calls.")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backends", nargs="+", choices=["bm25", "dense", "hybrid"],
                        default=["bm25", "dense", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.experiment_dir is None:
        if args.queries or args.output_dir:
            parser.error("Custom queries/output require --experiment-dir")
        legacy_main()
        return
    if args.queries is None:
        parser.error("--queries is required with --experiment-dir")
    if not 1 <= args.top_k <= 50 or args.candidate_k < args.top_k or args.threads < 1:
        parser.error("Require 1 <= top-k <= 50, candidate-k >= top-k and threads >= 1")
    if len(set(args.backends)) != len(args.backends):
        parser.error("Backends must be unique")
    experiment = args.experiment_dir.resolve()
    config = json.loads((experiment / "configuration.json").read_text(encoding="utf-8"))
    query_bytes = args.queries.read_bytes()
    if hashlib.sha256(query_bytes).hexdigest() != config["questions_sha256"]:
        parser.error("Questions hash differs from the experiment's recorded dataset")
    queries = [json.loads(line) for line in query_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(queries) != 92 or any(q.get("split") != "dev" for q in queries):
        parser.error("Expected the recorded 92 development claims")
    if len({q["query_id"] for q in queries}) != 92:
        parser.error("Claim IDs must be unique")
    if {q["ticker"] for q in queries} != set(config["tickers"]):
        parser.error("Company scope does not match the complete 92-claim experiment")
    out = (args.output_dir or experiment / "agent_integration").resolve()
    # Refuse to overwrite an existing batch or write over experiment inputs.
    if out == experiment or out in experiment.parents or (experiment / "companies") in [out, *out.parents]:
        parser.error("Output must be separate from experiment inputs and company indexes")
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; use a new --output-dir")
    from finsight.agents.narrative.processing.index import load_index
    from finsight.agents.narrative.rag.collect_evidence import NarrativeEvidenceAgent, build_queries
    from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
    from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend
    from finsight.contracts.schemas import SubClaim, SubClaimType

    def index_path(ticker):
        return experiment / "companies" / ticker / "index.json"

    indexes = {ticker: load_index(index_path(ticker)) for ticker in config["tickers"]}
    by_ids = {ticker: {c["chunk_id"]: c for c in index["chunks"]} for ticker, index in indexes.items()}
    embedding = None
    if any(backend != "bm25" for backend in args.backends):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        torch.set_num_threads(args.threads)
        embedding = SentenceTransformersBackend("sentence-transformers/all-MiniLM-L6-v2")
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    report = {
        "status": "running", "experiment_dir": str(experiment),
        "query_set": str(args.queries.resolve()), "query_set_sha256": config["questions_sha256"],
        "claim_count": len(queries), "expected_checks": len(queries) * len(args.backends),
        "backends": args.backends, "top_k": args.top_k, "candidate_k": args.candidate_k,
        "stance_method": "exact", "api_requests": 0,
        "scope": "Production Narrative Evidence Agent on isolated canonical92 experiment indexes",
        "limitations": [
            "Integration consistency, not semantic stance or generated-verdict accuracy.",
            "Agent generates two search queries; Step 4 used the original claim query.",
            "Production hybrid implementation is used; this does not reproduce Step 4 rankings.",
            "No cross-encoder backend is exposed by the production agent.",
            "Dense passage vectors retain the originating experiment's truncation limitations.",
        ],
        "checks": checks,
    }
    for backend in args.backends:
        for q in queries:
            ticker = q["ticker"]
            cutoff = date.fromisoformat(q["cutoff_date"])
            print(f"[{len(checks)+1}/{report['expected_checks']}] {ticker} {backend} {q['query_id']}", flush=True)

            def factory(t):
                return CorpusRetriever(index_path=index_path(t), company_ticker=t,
                                       qdrant_dir=experiment / "companies" / t / "qdrant",
                                       backend=backend, candidate_k=args.candidate_k,
                                       embedding_backend=embedding)

            claim = SubClaim(sub_claim_id="task7-" + q["query_id"], text=q["claim"],
                             company_ticker=ticker, sub_claim_type=SubClaimType.OTHER)
            agent = NarrativeEvidenceAgent(backend=backend, top_k=args.top_k,
                                           candidate_k=args.candidate_k, stance_method="exact",
                                           retriever_factory=factory, index_path_factory=index_path)
            items, errors = [], []
            try:
                items = agent.gather(claim, cutoff)
                if not items:
                    errors.append("No evidence returned")
                if len({e.evidence_id for e in items}) != len(items):
                    errors.append("Duplicate evidence IDs")
                for e in items:
                    c = by_ids[ticker].get(e.evidence_id)
                    if c is None:
                        errors.append(f"Unknown evidence ID: {e.evidence_id}")
                        continue
                    expected = {"content": c["text"], "company": c["company_name"],
                                "publication_date": c["publication_date"], "doc_type": c["document_type"],
                                "source_tier": c["source_tier"], "url": c.get("source_url"),
                                "citation_span": c.get("citation_span"), "sub_claim_id": claim.sub_claim_id}
                    actual = e.model_dump(mode="json")
                    for field, value in expected.items():
                        if actual[field] != value:
                            errors.append(f"{e.evidence_id}: {field} mismatch")
                    if c["company_ticker"] != ticker or e.publication_date > cutoff:
                        errors.append(f"{e.evidence_id}: company/date leakage")
                    if not e.citation_span:
                        errors.append(f"{e.evidence_id}: missing citation span")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
            name = f"{ticker}_{backend}_{hashlib.sha256(q['query_id'].encode()).hexdigest()}.json"
            payload = {"query_id": q["query_id"], "claim": q["claim"], "ticker": ticker,
                       "backend": backend, "cutoff_date": cutoff.isoformat(),
                       "queries": build_queries(claim), "errors": errors,
                       "evidence": [e.model_dump(mode="json") for e in items]}
            (out / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            checks.append({"query_id": q["query_id"], "ticker": ticker, "backend": backend,
                           "evidence_count": len(items), "errors": errors,
                           "corpus_fingerprint": indexes[ticker]["corpus_fingerprint"],
                           "stances": dict(Counter(e.stance.value for e in items)),
                           "missing_urls": sum(not e.url for e in items), "output": name})
            report.update(check_count=len(checks), error_count=sum(len(c["errors"]) for c in checks))
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report.update(status="complete" if report["error_count"] == 0 else "partial",
                  completed_at=datetime.now(timezone.utc).isoformat())
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
