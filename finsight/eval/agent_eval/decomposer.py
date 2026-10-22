"""Decomposer evaluation (Person 3, Amelia): does the claim split into the right sub-claims?

Gold: the dev split's `gold_sub_claims` (types + texts). Only the dev split is used: heldout stays frozen.
"""

from __future__ import annotations

from collections import Counter

from finsight.config import AgentRegistry
from finsight.contracts.schemas import SubClaim, SubClaimType
from finsight.eval.agent_eval.common import AgentReport, CaseResult, implementation_name, prf, ratio
from finsight.eval.metrics import PredictedSubClaim, align_sub_claims
from finsight.eval.run_eval import load_dataset


def evaluate(registry: AgentRegistry) -> AgentReport:
    del registry
    from finsight.agents.decomposition.stub import StubDecomposer

    agent = StubDecomposer()
    report = AgentReport("decomposer", "Person 3 (Amelia)", implementation_name(agent))
    report.notes.append("The running app reads stored claims. This score is the keyword splitter only.")
    tp = fp = fn = 0
    exact_count = analysed = gold_total = gold_matched = errors = 0
    advice_cases = advice_caught = 0

    for item in load_dataset("dev"):
        try:
            predicted: list[SubClaim] = agent.decompose(item.claim, item.cutoff_date)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            report.cases.append(CaseResult(item.item_id, False, f"raised {type(exc).__name__}: {exc}"))
            continue
        types = Counter(s.sub_claim_type for s in predicted)

        if item.category == "adversarial_advice":
            advice_cases += 1
            caught = types[SubClaimType.PERSONALIZED_ADVICE] > 0
            advice_caught += caught
            report.cases.append(CaseResult(item.item_id, caught, "advice detected" if caught
                                           else "advice request not flagged as PERSONALIZED_ADVICE"))
            continue
        if not item.gold_sub_claims or not all(g.sub_claim_type for g in item.gold_sub_claims):
            continue

        analysed += 1
        gold = Counter(g.sub_claim_type for g in item.gold_sub_claims)
        hit = sum((gold & types).values())
        tp, fp, fn = tp + hit, fp + sum(types.values()) - hit, fn + sum(gold.values()) - hit
        exact_count += len(predicted) == len(item.gold_sub_claims)
        aligned = align_sub_claims(item.gold_sub_claims,
                                   [PredictedSubClaim(text=s.text, verdict=None) for s in predicted])
        matched = sum(pred is not None for _, pred, _ in aligned)
        gold_total, gold_matched = gold_total + len(aligned), gold_matched + matched
        missing = sorted(t.value for t in (gold - types).elements())
        extra = sorted(t.value for t in (types - gold).elements())
        ok = not missing and not extra
        report.cases.append(CaseResult(item.item_id, ok, "types match" if ok
                                       else f"missing {missing or '-'}, extra {extra or '-'}"))

    precision, recall, f1 = prf(tp, fp, fn)
    report.metrics = {
        "claims evaluated": analysed,
        "sub-claim type precision": precision,
        "sub-claim type recall": recall,
        "sub-claim type F1": f1,
        "sub-claim count exactly right": ratio(exact_count, analysed),
        "gold sub-claims matched by text": ratio(gold_matched, gold_total),
        "advice requests flagged (PERSONALIZED_ADVICE)": ratio(advice_caught, advice_cases),
        "errors": errors,
    }
    report.notes.append("Gold labels are the dev set's; they were written alongside the stub corpus "
                        "(see docs/decisions.md, Known limitations).")
    return report
