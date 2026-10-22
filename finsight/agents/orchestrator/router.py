"""LLM routing: which evidence agent checks each sub-claim. Owner: Person 5 (Hung).

The orchestrator receives the sub-claims from claim extraction and asks the LLM, one sub-claim at a
time, whether the Financial Analysis Agent, the Narrative Evidence Agent, or both should gather
evidence. Each agent is then called as `gather(sub_claim, cutoff_date, feedback)`.

    decisions = route_sub_claims(sub_claims, kb)
    for d in decisions:
        print(d.sub_claim_id, d.collectors, d.reason)

# DECISION: the mock provider never calls the LLM. It routes by the knowledge-layer rule for the
# sub-claim type, so tests and free runs keep the old routing. An unreadable LLM reply falls back to
# the same rule. A refused type (personalized advice) is never sent to the LLM.
# The LLM only chooses collectors. Verdict caps and tier requirements still come from the knowledge layer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import get_args

from finsight.config import get_settings
from finsight.contracts.schemas import CollectorName, KnowledgeRule, SubClaim
from finsight.knowledge.loader import KnowledgeLayer
from finsight.llm.client import DATA_SAFETY_PREAMBLE, complete, data_block

COLLECTORS: tuple[str, ...] = get_args(CollectorName)  # ("financial", "narrative")

ROUTER_SYSTEM = DATA_SAFETY_PREAMBLE + """You are the router of FinSight, a system that fact-checks what \
finance influencers say about one listed company. You receive ONE sub-claim and decide which evidence \
agent should check it.

Agents:
- financial: the Financial Analysis Agent. Reads SEC / XBRL filing data and computes metrics in Python: \
revenue and its growth, net income, gross / operating / net margin, free cash flow, current ratio, \
debt to equity. Use it when the sub-claim states or implies a number, a growth rate, a margin, a ratio, \
or a direction of change in a reported financial figure.
- narrative: the Narrative Evidence Agent. Agentic RAG over 10-K / 10-Q text, earnings releases and \
investor-relations remarks. Finds supporting and counter-evidence for qualitative statements: what \
drives demand, management outlook, whether growth will continue, business risks, strategy.

Choose:
- ["financial"] for a pure reported-number claim.
- ["narrative"] for a qualitative or forward-looking claim.
- ["financial", "narrative"] only when the sub-claim needs BOTH a computed figure and a narrative \
explanation, e.g. "revenue grew 22% because of AI demand".
- [] when neither agent can check it, e.g. a ranking with no measure ("the biggest winner").

The team's default for this sub-claim type is given as guidance. Follow it unless the wording of the \
sub-claim clearly needs something else.

Reply with ONLY a JSON object, no markdown:
{"collectors": [...], "reason": "<one short sentence>"}"""


@dataclass(frozen=True)
class RouteDecision:
    """The orchestrator's routing for one sub-claim, plus the arguments each agent's gather() receives."""

    sub_claim: SubClaim
    collectors: list[str]
    reason: str
    source: str  # "llm" or "rules"
    cutoff_date: date | None = None
    feedback: str | None = None

    @property
    def sub_claim_id(self) -> str:
        return self.sub_claim.sub_claim_id

    @property
    def target(self) -> str:
        """One of: financial, narrative, both, none."""
        if len(self.collectors) == 2:
            return "both"
        return self.collectors[0] if self.collectors else "none"

    def agent_input(self) -> dict:
        """What the Financial and Narrative agents need (sub_claim fields + gather() arguments)."""
        return {
            "sub_claim_id": self.sub_claim.sub_claim_id,
            "company_ticker": self.sub_claim.company_ticker,
            "text": self.sub_claim.text,
            "cutoff_date": self.cutoff_date,
            "feedback": self.feedback,
        }


def router_prompt(sub_claim: SubClaim, rule: KnowledgeRule, feedback: str | None = None) -> str:
    """User prompt. The sub-claim and any auditor feedback are untrusted, so they go in data blocks."""
    payload = json.dumps({
        "text": sub_claim.text,
        "company_ticker": sub_claim.company_ticker,
        "sub_claim_type": sub_claim.sub_claim_type.value,
        "is_quantitative": sub_claim.is_quantitative,
    }, ensure_ascii=False)
    default = json.dumps(list(rule.collectors))
    parts = [
        data_block("sub_claim", payload),
        f"Team default collectors for {rule.sub_claim_type.value}: {default}",
    ]
    if rule.note:
        parts.append(f"Team note: {rule.note}")
    if feedback:
        parts.append("Auditor feedback from the failed first attempt:\n" + data_block("feedback", feedback))
    return "\n\n".join(parts)


def parse_route(text: str) -> tuple[list[str], str] | None:
    """(collectors, reason) from the LLM reply, or None when the reply is not a valid route."""
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    raw = data.get("collectors") if isinstance(data, dict) else None
    if isinstance(raw, str):
        raw = list(COLLECTORS) if raw.strip().lower() == "both" else [raw]
    if not isinstance(raw, list):
        return None
    names = [str(name).strip().lower() for name in raw]
    if "both" in names:
        names = list(COLLECTORS)
    if any(name not in COLLECTORS for name in names):
        return None
    collectors = [name for name in COLLECTORS if name in names]  # fixed order, no duplicates
    reason = str(data.get("reason") or "").strip()
    return collectors, reason


def _by_rule(sub_claim: SubClaim, rule: KnowledgeRule, why: str, cutoff: date | None,
             feedback: str | None) -> RouteDecision:
    note = f"{why}: knowledge-layer default for {rule.sub_claim_type.value}"
    return RouteDecision(sub_claim, list(rule.collectors), note, "rules", cutoff, feedback)


def route_sub_claim(sub_claim: SubClaim, rule: KnowledgeRule, *, cutoff_date: date | None = None,
                    feedback: str | None = None) -> RouteDecision:
    """Decide financial / narrative / both / none for one sub-claim."""
    if rule.refuse:
        return _by_rule(sub_claim, rule, "refused type", cutoff_date, feedback)
    if get_settings().llm_provider == "mock":
        return _by_rule(sub_claim, rule, "mock mode", cutoff_date, feedback)
    reply = complete(router_prompt(sub_claim, rule, feedback), system=ROUTER_SYSTEM,
                     max_tokens=2000, temperature=0.0)
    parsed = parse_route(reply.text)
    if parsed is None:
        return _by_rule(sub_claim, rule, "unreadable router reply", cutoff_date, feedback)
    collectors, reason = parsed
    return RouteDecision(sub_claim, collectors, reason or "no reason given", "llm", cutoff_date, feedback)


def route_sub_claims(sub_claims: list[SubClaim], kb: KnowledgeLayer, *, cutoff_date: date | None = None,
                     feedback: dict[str, str] | None = None) -> list[RouteDecision]:
    """Route every sub-claim from claim extraction, in order."""
    feedback = feedback or {}
    return [
        route_sub_claim(c, kb.rule_for(c.sub_claim_type), cutoff_date=cutoff_date,
                        feedback=feedback.get(c.sub_claim_id))
        for c in sub_claims
    ]
