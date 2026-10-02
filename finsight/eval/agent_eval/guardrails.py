"""Advice Boundary + Tier Policy evaluation (Person 4, Irine)."""

from __future__ import annotations

from datetime import date

from finsight.agents.orchestrator import routing
from finsight.config import AgentRegistry
from finsight.contracts.schemas import (
    AdviceStage,
    EvidenceItem,
    SourceTier,
    Stance,
    SubClaim,
    Verdict,
)
from finsight.eval.agent_eval.common import AgentReport, CaseResult, implementation_name, load_jsonl, prf, ratio
from finsight.knowledge.loader import default_knowledge_layer


def evaluate_advice_boundary(registry: AgentRegistry) -> AgentReport:
    agent = registry.advice_boundary
    report = AgentReport("advice_boundary", "Person 4 (Irine)", implementation_name(agent))
    counts = {stage: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for stage in AdviceStage}
    errors = 0
    for case in load_jsonl("advice_boundary.jsonl"):
        stage, expected = AdviceStage(case["stage"]), case["flag"]
        try:
            flagged = agent.check(case["text"], stage).flagged
        except Exception as exc:  # noqa: BLE001
            errors += 1
            report.cases.append(CaseResult(case["case_id"], False, f"raised {type(exc).__name__}: {exc}"))
            continue
        key = ("tp" if flagged else "fn") if expected else ("fp" if flagged else "tn")
        counts[stage][key] += 1
        ok = flagged == expected
        what = "missed advice" if expected and not flagged else "false alarm" if flagged and not expected else ""
        report.cases.append(CaseResult(case["case_id"], ok, f"{what}: {case['text']}" if what else case["text"]))

    report.metrics = {"cases": sum(sum(c.values()) for c in counts.values()) + errors}
    for stage, c in counts.items():
        precision, recall, f1 = prf(c["tp"], c["fp"], c["fn"])
        label = "input (user's request)" if stage is AdviceStage.INPUT else "output (FinSight's answer)"
        report.metrics |= {f"{label}: recall (advice caught)": recall,
                           f"{label}: precision (flags that were right)": precision,
                           f"{label}: F1": f1}
    report.metrics["errors"] = errors
    report.notes.append("Recall matters most at input (an advice request must never be analysed); "
                        "precision matters for usability (normal claims must not be refused).")
    return report


def _ev(stance: Stance, tier: SourceTier) -> EvidenceItem:
    return EvidenceItem(evidence_id=f"{stance.value[:3]}-{tier.value}", sub_claim_id="sc1", stance=stance,
                        content="x", source_tier=tier, doc_type="10-Q" if tier is SourceTier.TIER_1 else "news",
                        company="Company X", publication_date=date(2025, 7, 25), produced_by="eval")


def _expected(proposed: Verdict, evidence: list[EvidenceItem], rule) -> Verdict:
    """The knowledge-layer rules as specified (cap, tier requirement, counter-evidence)."""
    verdict, _ = routing.cap_verdict(proposed, rule)
    verdict, _ = routing.enforce_tier_requirement(verdict, evidence, rule)
    verdict, _ = routing.enforce_counter_evidence(verdict, evidence, rule)
    return verdict


def evaluate_tier_policy(registry: AgentRegistry) -> AgentReport:
    """Cases generated from knowledge_layer.yaml, so they follow whatever content Irine finalises."""
    agent = registry.tier_policy
    report = AgentReport("tier_policy", "Person 4 (Irine)", implementation_name(agent))
    correct = total = raises = errors = 0
    scenarios = {
        "required-tier support": lambda tiers: [_ev(Stance.SUPPORTING, tiers[0])],
        "news-only support": lambda tiers: [_ev(Stance.SUPPORTING, SourceTier.TIER_3)],
        "support + counter-evidence": lambda tiers: [_ev(Stance.SUPPORTING, tiers[0]),
                                                     _ev(Stance.CHALLENGING, tiers[0])],
        "no evidence": lambda tiers: [],
    }
    for rule in default_knowledge_layer().rules:
        if rule.refuse or not rule.collectors or not rule.tier_requirement:
            continue
        sub_claim = SubClaim(sub_claim_id="sc1", text="eval", sub_claim_type=rule.sub_claim_type)
        for scenario, make in scenarios.items():
            evidence = make(rule.tier_requirement)
            for proposed in Verdict:
                case_id = f"{rule.sub_claim_type.value}/{scenario}/{proposed.value}"
                try:
                    got, _ = agent.apply(sub_claim, proposed, evidence, rule)
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    report.cases.append(CaseResult(case_id, False, f"raised {type(exc).__name__}: {exc}"))
                    continue
                expected = _expected(proposed, evidence, rule)
                illegal_raise = routing.is_raise(proposed, got)
                total, correct, raises = total + 1, correct + (got is expected), raises + illegal_raise
                detail = f"expected {expected.value}, got {got.value}" + (" (ILLEGAL RAISE)" if illegal_raise else "")
                report.cases.append(CaseResult(case_id, got is expected and not illegal_raise, detail))

    report.metrics = {
        "cases (generated from the knowledge layer)": total,
        "matches the knowledge-layer rules": ratio(correct, total),
        "illegal raises (verdict made stronger)": raises,
        "errors": errors,
    }
    report.notes.append("The orchestrator re-applies these rules as a safety net, so a mismatch here does not "
                        "reach users; it shows where the agent itself disagrees with the agreed rules.")
    return report
