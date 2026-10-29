"""Read-only validation of captured retrieval inputs for generation-only runs."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from scripts.narrative.evaluation.evaluate_rag_pilot import case_identity, load_cases, read_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_configuration(run_dir: Path, strategy: str, backend: str) -> Path:
    """Resolve one requested configuration from an existing matrix capture."""
    run_dir = run_dir.resolve()
    manifest = read_json(run_dir / "matrix_configuration.json")
    if manifest.get("retrieval_only") is not True:
        raise ValueError("--retrieval-run-dir must point to a retrieval-only matrix")
    matches = [e for e in manifest["configurations"]
               if e["strategy"] == strategy and e["backend"] == backend]
    if len(matches) != 1:
        raise ValueError(f"Saved retrieval matrix does not contain exactly one {strategy}/{backend}")
    folder = (run_dir / matches[0]["folder"]).resolve()
    if not folder.is_relative_to(run_dir):
        raise ValueError("Saved retrieval configuration must remain inside its matrix folder")
    return folder


def validate_configuration(folder: Path, args, selected: list[dict], workspace: Path) -> dict:
    """Require matching retrieval settings and the exact reviewed dataset snapshot."""
    config = read_json(folder / "configuration.json")
    expected = {"dataset_sha256": digest(args.cases), "backend": args.backend,
                "top_k_per_direction": args.top_k, "candidate_k": args.candidate_k,
                "rerank_overflow": args.rerank_overflow, "source_priority": "P0",
                "retrieval_only": True}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Saved retrieval setting differs: {key} in {folder}")
    if args.backend == "r4" and config.get("reranker_model") != os.getenv(
            "FINSIGHT_NARRATIVE_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        raise ValueError("Saved retrieval reranker_model differs from the requested settings")
    source_corpus = Path(config["corpus_root"])
    if not source_corpus.is_absolute():
        source_corpus = workspace / source_corpus
    if source_corpus.resolve() != args.corpus_root.resolve():
        raise ValueError("Saved retrieval corpus/chunk strategy/build differs")
    # The saved rank order is authoritative; no reranking runs in this mode.
    snapshot = load_cases(folder / "cases.csv")
    if (case_identity(snapshot) != config.get("selected_cases_sha256")
            or config.get("case_ids") != [c["id"] for c in snapshot]
            or len({c["id"] for c in snapshot}) != len(snapshot)):
        raise ValueError("Saved retrieval case snapshot differs from its recorded identity")
    by_id = {c["id"]: c for c in snapshot}
    if any(by_id.get(c["id"]) != c for c in selected):
        raise ValueError("Selected cases are missing or different in the saved retrieval snapshot")
    return {"protocol": "saved_retrieval_v1", "configuration_dir": str(folder.resolve()),
            "configuration_sha256": digest(folder / "configuration.json"),
            "cases_sha256": digest(folder / "cases.csv")}


def read_payload(folder: Path, case_id: str, expected: dict | None = None) -> tuple[dict, dict, dict]:
    """Read a case and pin its source bytes; later reads must match that pin."""
    case_dir = folder / "cases" / case_id
    hashes = {name: digest(case_dir / name) for name in
              ("result.json", "retrieval.json", "preflight.json")}
    if expected is not None and hashes != expected["sha256"]:
        raise ValueError(f"Saved retrieval inputs changed after preflight for {case_id}")
    result, trace = read_json(case_dir / "result.json"), read_json(case_dir / "retrieval.json")
    # Catch concurrent changes while parsing as well as between preflight and execution.
    if hashes != {name: digest(case_dir / name) for name in hashes}:
        raise ValueError(f"Saved retrieval inputs changed while reading {case_id}")
    origin = {"case_dir": str(case_dir.resolve()), "sha256": hashes}
    return result, trace, origin


def validate_case(folder: Path, case: dict, preflight: dict, chunks: list[dict], args) -> dict:
    """Validate rankings plus the exact filtered candidates actually supplied to Narrative."""
    from finsight.contracts.schemas import EvidenceItem, Stance
    from finsight.agents.narrative.retrieval.retriever import normalize_backend

    result, trace, origin = read_payload(folder, case["id"])
    if read_json(folder / "cases" / case["id"] / "preflight.json") != preflight:
        raise ValueError(f"Saved retrieval build/source identity differs for {case['id']}")
    request = {"ticker": case["company"], "claim": case["claims"], "cutoff": case["cut_off"],
               "backend": normalize_backend(args.backend), "top_k": args.top_k, "candidate_k": args.candidate_k}
    if any(trace.get(k) != v for k, v in request.items()):
        raise ValueError(f"Saved retrieval request differs for {case['id']}")
    if trace.get("index", {}).get("corpus_fingerprint") != preflight["corpus_fingerprint"]:
        raise ValueError(f"Saved retrieval fingerprint differs for {case['id']}")
    if (result.get("pipeline") != "narrative_evidence_retrieval"
            or result.get("status") not in {"OK", "NO_EVIDENCE"}
            or result.get("ticker") != case["company"] or result.get("claim") != case["claims"]
            or result.get("cutoff_date") != case["cut_off"] or result.get("backend") != args.backend
            or result.get("index") != trace.get("index")
            or result.get("reranking") != trace.get("reranking")):
        raise ValueError(f"Saved result is not a matching successful retrieval capture: {case['id']}")
    indexed = {c["chunk_id"]: c for c in chunks}
    ranked_ids = []
    for direction in ("support", "counter"):
        hits = trace[direction]
        if ([h["rank"] for h in hits] != list(range(1, len(hits) + 1))
                or len({h["chunk_id"] for h in hits}) != len(hits)):
            raise ValueError(f"Invalid saved retrieval ranking for {case['id']}/{direction}")
        for hit in hits:
            chunk = indexed.get(hit["chunk_id"])
            if (chunk is None or hit["text"] != chunk["text"]
                    or chunk["company_ticker"] != case["company"]
                    or chunk["publication_date"] > case["cut_off"]):
                raise ValueError(f"Invalid saved retrieval passage for {case['id']}")
            for field, source_field in (("doc_type", "document_type"), ("publication_date", "publication_date"),
                                        ("url", "source_url"), ("citation_span", "citation_span")):
                if hit.get(field) != chunk.get(source_field):
                    raise ValueError(f"Saved passage provenance differs for {case['id']}/{field}")
            if hit["chunk_id"] not in ranked_ids:
                ranked_ids.append(hit["chunk_id"])
    candidates = result["candidate_evidence"]
    ids = [e["evidence_id"] for e in candidates]
    if (len(set(ids)) != len(ids) or any(i not in ranked_ids for i in ids)
            or ids != [i for i in ranked_ids if i in ids]):
        raise ValueError(f"Saved candidate order/IDs differ from retrieval for {case['id']}")
    for candidate in candidates:
        c = indexed[candidate["evidence_id"]]
        canonical = EvidenceItem(
            evidence_id=c["chunk_id"], sub_claim_id=trace["sub_claim_id"], stance=Stance.NEUTRAL,
            content=c["text"], source_tier=c["source_tier"], doc_type=c["document_type"],
            company=c["company_name"], period=c.get("period"), publication_date=c["publication_date"],
            publication_date_is_estimated=c.get("publication_date_is_estimated", False),
            publication_date_precision=c.get("publication_date_precision", "day"),
            publication_date_basis=c.get("publication_date_basis"), url=c.get("source_url"),
            citation_span=c.get("citation_span"), produced_by="narrative_retrieval_unassessed",
        ).model_dump(mode="json", exclude={"stance"})
        if candidate != canonical:
            raise ValueError(f"Saved candidate content/provenance differs for {case['id']}")
    retrieval = result["retrieval"]
    if (result.get("evidence") != candidates or result.get("selection_applied") is not False
            or result.get("selection") != {"returned_count": len(ids), "returned_evidence_ids": ids}
            or retrieval.get("evidence_ids") != ids or retrieval.get("count") != len(ids)
            or retrieval.get("queries") != {"support": trace["support_query"], "counter": trace["counter_query"]}
            or (result["status"] == "OK") != bool(ids)):
        raise ValueError(f"Saved retrieval candidate snapshot is inconsistent for {case['id']}")
    return origin


def evidence_items(result: dict) -> list:
    """Restore only validated saved candidates; never expose gold fields to generation."""
    from finsight.contracts.schemas import EvidenceItem, Stance

    return [EvidenceItem.model_validate({**item, "stance": Stance.NEUTRAL})
            for item in result["candidate_evidence"]]
