"""Configurable experimental document priority, separate from source tier.

P1 deliberately cannot certify semantic sufficiency from keywords. Until an
independently validated sufficiency component exists, Medium expansion is
mandatory. P2 is a small, period-guarded relevance adjustment.
"""

from __future__ import annotations

import re
from typing import Any


def assign_priority(chunk: dict, aliases: list[dict], config: dict) -> dict:
    assignments = []
    for alias in aliases:
        dtype = alias["document_type"]
        if dtype in config["high_types"]:
            priority, reason = "HIGH", "classified underlying document type"
        elif dtype in config["medium_types"]:
            priority, reason = "MEDIUM", "classified underlying document type"
        else:
            priority, reason = config["unknown_priority"], "unclassified content; searchable; content review required"
        assignments.append(
            {
                "chunk_id": alias["chunk_id"],
                "document_id": alias["document_id"],
                "document_type": dtype,
                "priority": priority,
                "reason": reason,
            }
        )
    high = [a for a in assignments if a["priority"] == "HIGH"]
    return {
        "priority": "HIGH" if high else "MEDIUM",
        "source_tier": chunk["source_tier"],
        "aliases": assignments,
        "basis": "highest verified underlying-type priority among exact-text aliases",
        "needs_type_review": any("review required" in a["reason"] for a in assignments),
    }


def period_anchors(query: str) -> list[str]:
    """Use only the actual query; never inspect labels or required gold context."""
    pattern = (
        r"\b(?:20\d{2}|19\d{2}|q[1-4]|january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b"
    )
    return sorted(set(re.findall(pattern, query.lower())))


def period_match(query: str, hit: dict) -> float:
    anchors = period_anchors(query)
    if not anchors:
        return 0.0
    text = (hit["text"] + " " + str(hit.get("period") or "")).lower()
    return sum(bool(re.search(r"\b" + re.escape(a) + r"\b", text)) for a in anchors) / len(anchors)


def sufficiency_diagnostic(query: str, candidates: list[dict]) -> dict:
    text = "\n".join(c["text"] for c in candidates)
    years = [a for a in period_anchors(query) if a.isdigit()]
    return {
        "status": "UNKNOWN",
        "expand_medium": True,
        "reason": (
            "No validated semantic sufficiency checker; "
            "keyword or context presence cannot establish evidence sufficiency"
        ),
        "candidate_count": len(candidates),
        "query_period_anchors": period_anchors(query),
        "period_tokens_present": all(y in text for y in years) if years else None,
        "numeric_value_present": bool(re.search(r"\d", text)),
        "unit_marker_present": bool(re.search(r"\b(?:million|billion|percent|shares|people|users)\b|[$%]", text, re.I)),
        "metric_scope_entailment": "not established",
        "unresolved_counter_evidence": "not established",
    }


def apply_bounded_priority(
    query: str, ranking: list[dict], priorities: dict, config: dict
) -> tuple[list[dict], list[dict]]:
    """Promote High across near-ties only; never cross a better period match.

    A stable adjacent-swap algorithm enforces both maximum promotion and score
    gap. All raw model/BM25 scores remain available in the trace.
    """
    result = list(ranking)
    old = {h["chunk_id"]: i for i, h in enumerate(ranking)}
    scores = [float(h["retrieval_score"]) for h in ranking]
    span = max(scores, default=0) - min(scores, default=0)
    norm = {h["chunk_id"]: (float(h["retrieval_score"]) - min(scores)) / span if span else 0.0 for h in ranking}
    moves = {}
    for original in ranking:
        cid = original["chunk_id"]
        if priorities[cid]["priority"] != "HIGH":
            continue
        pos = next(i for i, h in enumerate(result) if h["chunk_id"] == cid)
        while pos > 0 and old[cid] - pos < config["p2_max_rank_promotion"]:
            previous = result[pos - 1]
            if priorities[previous["chunk_id"]]["priority"] == "HIGH":
                break
            gap = norm[previous["chunk_id"]] - norm[cid]
            if gap > config["p2_normalized_score_gap"] + 1e-12 or period_match(query, original) < period_match(
                query, previous
            ):
                break
            result[pos - 1], result[pos] = result[pos], result[pos - 1]
            moves[cid] = "High priority within normalized relevance gap and no worse explicit query-period match"
            pos -= 1
    trace = []
    for new, hit in enumerate(result, 1):
        cid = hit["chunk_id"]
        trace.append(
            {
                "chunk_id": cid,
                "document_type": hit["document_type"],
                "priority": priorities[cid]["priority"],
                "raw_relevance_score": hit["retrieval_score"],
                "normalized_relevance_score": norm[cid],
                "period_match": period_match(query, hit),
                "base_rank": old[cid] + 1,
                "final_rank": new,
                "rank_change": old[cid] + 1 - new,
                "reason": moves.get(cid, "relevance order retained or displacement by a bounded near-tie promotion"),
            }
        )
    return result, trace


def unchanged_trace(query: str, ranking: list[dict], priorities: dict, policy: str) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": h["chunk_id"],
            "document_type": h["document_type"],
            "priority": priorities[h["chunk_id"]]["priority"],
            "raw_relevance_score": h["retrieval_score"],
            "period_match": period_match(query, h),
            "base_rank": i,
            "final_rank": i,
            "rank_change": 0,
            "reason": "P0 relevance baseline"
            if policy == "P0"
            else "P1 expanded all priorities; final order follows original-claim relevance",
        }
        for i, h in enumerate(ranking, 1)
    ]
