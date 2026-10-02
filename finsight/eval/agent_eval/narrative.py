"""Narrative Evidence Agent + Basic RAG retriever evaluation (Person 2, Stella)."""

from __future__ import annotations

from finsight.config import AgentRegistry
from finsight.contracts.schemas import Stance, SubClaim
from finsight.eval.agent_eval.common import (
    AgentReport,
    CaseResult,
    as_date,
    corpus_note,
    implementation_name,
    load_jsonl,
    ratio,
)

TOP_K = 3


def evaluate_narrative(registry: AgentRegistry) -> AgentReport:
    agent = registry.narrative
    report = AgentReport("narrative", "Person 2 (Stella)", implementation_name(agent))
    n = {k: 0 for k in ("expected", "found", "returned", "relevant_returned", "stance_ok", "stance_checked",
                        "counter_cases", "counter_found", "items", "cited", "leaks", "errors", "cases")}

    for case in load_jsonl("narrative.jsonl"):
        n["cases"] += 1
        sub_claim = SubClaim.model_validate(case["sub_claim"])
        cutoff, expect = as_date(case["cutoff"]), case["expect"]
        try:
            items = agent.gather(sub_claim, cutoff, None)
        except Exception as exc:  # noqa: BLE001
            n["errors"] += 1
            report.cases.append(CaseResult(case["case_id"], False, f"raised {type(exc).__name__}: {exc}"))
            continue
        gold = {eid: Stance.SUPPORTING for eid in expect["supporting"]}
        gold |= {eid: Stance.CHALLENGING for eid in expect["challenging"]}
        returned = {i.evidence_id: i for i in items}
        leaks = [i.evidence_id for i in items if i.publication_date > cutoff]
        n["leaks"] += len(leaks)
        n["items"] += len(items)
        n["cited"] += sum(bool(i.url and i.citation_span) for i in items)

        found = [eid for eid in gold if eid in returned]
        unexpected = [eid for eid in returned if eid not in gold and eid not in expect.get("also_ok", [])]
        n["expected"] += len(gold)
        n["found"] += len(found)
        n["returned"] += len([eid for eid in returned if eid not in expect.get("also_ok", [])])
        n["relevant_returned"] += len(found)
        wrong_stance = [eid for eid in found if returned[eid].stance is not gold[eid]]
        n["stance_checked"] += len(found)
        n["stance_ok"] += len(found) - len(wrong_stance)
        if expect["challenging"]:
            n["counter_cases"] += 1
            n["counter_found"] += any(i.stance is Stance.CHALLENGING for i in items)

        problems = []
        if missing := [eid for eid in gold if eid not in returned]:
            problems.append(f"missed {missing}")
        if unexpected:
            problems.append(f"irrelevant {unexpected}")
        if wrong_stance:
            problems.append(f"wrong stance on {wrong_stance}")
        if leaks:
            problems.append(f"post-cut-off evidence {leaks}")
        report.cases.append(CaseResult(case["case_id"], not problems, "; ".join(problems) or case.get("note", "")))

    report.metrics = {
        "cases": n["cases"],
        "evidence recall (expected passages found)": ratio(n["found"], n["expected"]),
        "evidence precision (returned passages relevant)": ratio(n["relevant_returned"], n["returned"]),
        "right stance (for / against)": ratio(n["stance_ok"], n["stance_checked"]),
        "finds counter-evidence when it exists": ratio(n["counter_found"], n["counter_cases"]),
        "items with URL + citation span": ratio(n["cited"], n["items"]),
        "post-cut-off items returned": n["leaks"],
        "errors": n["errors"],
    }
    report.notes.append(corpus_note("narrative.jsonl"))
    return report


def evaluate_retriever(registry: AgentRegistry) -> AgentReport:
    agent = registry.retriever
    report = AgentReport("retriever", "Person 2 (Stella)", implementation_name(agent))
    recall_sum = rr_sum = scored = leaks = errors = cases = 0

    for case in load_jsonl("retriever.jsonl"):
        cases += 1
        cutoff = as_date(case["cutoff"])
        try:
            items = agent.retrieve(case["query"], TOP_K, cutoff)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            report.cases.append(CaseResult(case["case_id"], False, f"raised {type(exc).__name__}: {exc}"))
            continue
        ids = [i.evidence_id for i in items]
        case_leaks = [i.evidence_id for i in items if i.publication_date > cutoff]
        leaks += len(case_leaks)
        problems = [f"post-cut-off evidence {case_leaks}"] if case_leaks else []
        if len(items) > TOP_K:
            problems.append(f"returned {len(items)} > top_k {TOP_K}")
        relevant = case["relevant"]
        if relevant:
            scored += 1
            hits = [eid for eid in relevant if eid in ids[:TOP_K]]
            recall_sum += len(hits) / len(relevant)
            rank = next((pos for pos, eid in enumerate(ids, start=1) if eid in relevant), None)
            rr_sum += 1 / rank if rank else 0
            if not hits:
                problems.append(f"none of {relevant} in top {TOP_K}: {ids}")
        report.cases.append(CaseResult(case["case_id"], not problems,
                                       "; ".join(problems) or f"top {TOP_K}: {ids}"))

    report.metrics = {
        "queries": cases,
        f"recall@{TOP_K}": ratio(recall_sum, scored),
        "MRR (how high the first relevant passage ranks)": ratio(rr_sum, scored),
        "post-cut-off items returned": leaks,
        "errors": errors,
    }
    report.notes.append(corpus_note("retriever.jsonl"))
    return report
