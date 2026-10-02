"""Financial Analysis Agent evaluation (Person 1, Venes): right numbers, metric, stance, tier, dates."""

from __future__ import annotations

from finsight.config import AgentRegistry
from finsight.contracts.schemas import SourceTier, Stance, SubClaim
from finsight.eval.agent_eval.common import (
    AgentReport,
    CaseResult,
    as_date,
    corpus_note,
    implementation_name,
    load_jsonl,
    ratio,
)

TOLERANCE = 0.005  # absolute, on fractions (0.14 = 14%)


def evaluate(registry: AgentRegistry) -> AgentReport:
    agent = registry.financial
    report = AgentReport("financial", "Person 1 (Venes)", implementation_name(agent))
    n = {k: 0 for k in ("value_cases", "value_ok", "metric_ok", "stance_ok", "tier_ok", "empty_cases",
                        "empty_ok", "items", "cited", "leaks", "errors")}

    for case in load_jsonl("financial.jsonl"):
        sub_claim = SubClaim.model_validate(case["sub_claim"])
        cutoff, expect = as_date(case["cutoff"]), case["expect"]
        try:
            items = agent.gather(sub_claim, cutoff, None)
        except Exception as exc:  # noqa: BLE001
            n["errors"] += 1
            report.cases.append(CaseResult(case["case_id"], False, f"raised {type(exc).__name__}: {exc}"))
            continue
        leaks = [i.evidence_id for i in items if i.publication_date > cutoff]
        n["leaks"] += len(leaks)
        n["items"] += len(items)
        n["cited"] += sum(bool(i.url and i.citation_span) for i in items)
        problems = [f"post-cut-off evidence {leaks}"] if leaks else []

        if expect.get("empty"):
            n["empty_cases"] += 1
            if items:
                problems.append(f"expected no evidence, got {[i.evidence_id for i in items]}")
            else:
                n["empty_ok"] += 1
        else:
            n["value_cases"] += 1
            numeric = [i for i in items if i.computed_value is not None]
            best = next((i for i in numeric if i.metric_name == expect["metric_name"]), numeric[0] if numeric else None)
            if best is None:
                problems.append("no computed value returned")
            else:
                checks = {
                    "value_ok": abs(best.computed_value - expect["computed_value"]) <= TOLERANCE,
                    "metric_ok": best.metric_name == expect["metric_name"],
                    "stance_ok": best.stance is Stance(expect["stance"]),
                    "tier_ok": best.source_tier is SourceTier(expect["tier"]),
                }
                for key, ok in checks.items():
                    n[key] += ok
                if not checks["value_ok"]:
                    problems.append(f"value {best.computed_value} != {expect['computed_value']}")
                if not checks["metric_ok"]:
                    problems.append(f"metric {best.metric_name} != {expect['metric_name']}")
                if not checks["stance_ok"]:
                    problems.append(f"stance {best.stance.value} != {expect['stance']}")
                if not checks["tier_ok"]:
                    problems.append(f"tier {best.source_tier.value} != {expect['tier']}")
        report.cases.append(CaseResult(case["case_id"], not problems, "; ".join(problems) or case.get("note", "")))

    report.metrics = {
        "cases": n["value_cases"] + n["empty_cases"] + n["errors"],
        "number correct (within 0.005)": ratio(n["value_ok"], n["value_cases"]),
        "right metric": ratio(n["metric_ok"], n["value_cases"]),
        "right stance (for / against)": ratio(n["stance_ok"], n["value_cases"]),
        "labelled Tier 1": ratio(n["tier_ok"], n["value_cases"]),
        "correctly returns nothing (no data / before filing)": ratio(n["empty_ok"], n["empty_cases"]),
        "items with URL + citation span": ratio(n["cited"], n["items"]),
        "post-cut-off items returned": n["leaks"],
        "errors": n["errors"],
    }
    report.notes.append(corpus_note("financial.jsonl"))
    return report
