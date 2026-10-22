"""Validate retrieval provenance on a recorded frozen query set.

This runs actual BM25/dense/hybrid retrieval, records complete evidence, and checks
source mapping and temporal/company isolation. It does not judge financial accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from finsight.agents.narrative.corpus.processing.index import load_index
from finsight.agents.narrative.corpus.pdf.pdf_enrichment import write_json
from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend
from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
from finsight.agents.narrative.evaluation.run_retrieval_comparison import _load_queries


def refresh(workspace: Path, queries_path: Path, *, top_k: int = 10, candidate_k: int = 50) -> dict:
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("Require candidate_k >= top_k >= 1")
    workspace = workspace.resolve()
    queries, query_hash = _load_queries(queries_path)
    root = workspace / "data/narrative/experiments/basic_rag"
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "post-PDF canonical corpus; production cited-passage retriever",
              "query_set": str(queries_path.resolve()), "query_set_sha256": query_hash,
              "top_k": top_k, "candidate_k": candidate_k, "checks": [], "corpus_fingerprints": {},
              "limitations": ["Source mapping is checked; relevance and financial accuracy are not judged.",
                              "The frozen development labels are sparse and have no independent review.",
                              "Production hybrid retains lexical candidates and differs from experimental pure RRF."]}
    embedding = SentenceTransformersBackend()
    report["embedding_model"] = embedding.model_name
    for query in queries:
        ticker = query["ticker"].upper()
        company_root = workspace / "data/narrative/processed" / ticker
        index_path = company_root / "indexes/index.json"
        index = load_index(index_path)
        canonical = {chunk["chunk_id"]: chunk for chunk in index["chunks"]}
        report["corpus_fingerprints"][ticker] = index["corpus_fingerprint"]
        cutoff = date.fromisoformat(query["cutoff_date"])
        for backend in ("bm25", "dense", "hybrid"):
            retriever = CorpusRetriever(index_path=index_path, company_ticker=ticker,
                                        qdrant_dir=company_root / "indexes/qdrant", backend=backend,
                                        candidate_k=candidate_k, embedding_backend=embedding)
            for direction in ("support", "counter"):
                print(f"[{ticker}/{backend}/{direction}] {query['query_id']}", flush=True)
                text = query[f"{direction}_query"]
                evidence = retriever.retrieve(text, top_k, cutoff)
                errors = []
                types: Counter = Counter()
                missing_urls = 0
                for item in evidence:
                    chunk = canonical.get(item.evidence_id)
                    if chunk is None:
                        errors.append(f"Unknown evidence ID: {item.evidence_id}")
                        continue
                    types[chunk.get("content_type", "narrative")] += 1
                    missing_urls += not bool(item.url)
                    expected = {"content": chunk["text"], "company": chunk["company_name"],
                                "publication_date": chunk["publication_date"], "url": chunk.get("source_url"),
                                "citation_span": chunk.get("citation_span"), "source_tier": chunk["source_tier"],
                                "doc_type": chunk["document_type"],
                                "publication_date_is_estimated": chunk.get("publication_date_is_estimated", False)}
                    actual = item.model_dump(mode="json")
                    for field, value in expected.items():
                        if actual[field] != value:
                            errors.append(f"{item.evidence_id}: {field} differs from the canonical chunk")
                    if chunk["company_ticker"].upper() != ticker or item.publication_date > cutoff:
                        errors.append(f"Company/cutoff leak: {item.evidence_id}")
                    if not chunk.get("local_path") or not Path(chunk["local_path"]).is_file():
                        errors.append(f"Source file unavailable: {item.evidence_id}")
                    if not item.citation_span:
                        errors.append(f"Citation span unavailable: {item.evidence_id}")
                ids = [item.evidence_id for item in evidence]
                if len(ids) != len(set(ids)):
                    errors.append("Duplicate evidence IDs")
                if not evidence:
                    errors.append("No evidence returned for this recorded development query")
                name = hashlib.sha256(f"{query['query_id']}:{direction}:{backend}".encode()).hexdigest()[:16]
                output = root / "post_pdf_review" / f"{ticker}_{backend}_{direction}_{name}.json"
                payload = {"status": "OK" if evidence else "NO_EVIDENCE", "company_ticker": ticker,
                           "query_id": query["query_id"], "direction": direction, "query": text,
                           "cutoff_date": cutoff.isoformat(), "backend_used": retriever.last_backend,
                           "top_k": top_k, "candidate_k": candidate_k,
                           "corpus_fingerprint": index["corpus_fingerprint"],
                           "evidence": [item.model_dump(mode="json") for item in evidence],
                           "errors": errors, "missing_urls": missing_urls,
                           "content_types": dict(types), "coverage_limitations": retriever.coverage_limitations(cutoff)}
                write_json(output, payload)
                report["checks"].append({key: payload[key] for key in
                    ("company_ticker", "query_id", "direction", "query", "cutoff_date", "backend_used",
                     "corpus_fingerprint", "errors", "missing_urls", "content_types")})
                report["checks"][-1].update(ticker=ticker, backend=backend, evidence_ids=ids,
                                            output_path=output.relative_to(workspace).as_posix())
                write_json(root / "task5_smoke.json", report)
    report["check_count"] = len(report["checks"])
    report["error_count"] = sum(len(check["errors"]) for check in report["checks"])
    report["missing_url_occurrences"] = sum(check["missing_urls"] for check in report["checks"])
    report["status"] = "complete" if not report["error_count"] else "failed"
    write_json(root / "task5_smoke.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    args = parser.parse_args()
    report = refresh(args.workspace, args.queries, top_k=args.top_k, candidate_k=args.candidate_k)
    print(json.dumps({key: report[key] for key in
                      ("status", "check_count", "error_count", "missing_url_occurrences")}, indent=2))
    if report["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
