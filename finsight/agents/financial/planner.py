""" Planner: the reasoning loop of the financial agent

The model reads the claim and decides, step by step, which figures to compute. It never computes a number and
never judges the claim: each step is one action that `agent.py` carries out with the deterministic tools in
`tools.py`, and the result (or the reason there is none) is shown to the model before its next step.

    model:  {"thought": "...", "action": "understand", "args": {"parts": [{"words": "revenue grew 22%", ...}]}}
    agent:  NOTED 1 part(s), 1 in scope. A rule-based reading of the same line, for comparison: ...
    model:  {"thought": "...", "action": "report_growth", "args": {"measure": "Revenue", "period": {...}}}
    agent:  OK revenue_growth_yoy = +21.8% (quarter ended 2026-03-31); amount $109.90bn
    model:  {"thought": "...", "action": "finish", "args": {}}

The first step is always `understand`: the model writes down, from the words alone, every financial statement
the line makes about the company (measure, period, stated figure, direction) and whether it is in this
sub-claim's scope. Only then does it see the rule-based reading, as a cross-check, so the rules cannot steer
its reading. Its reading is kept in `Plan.reading` for the decision log.

The loop uses the team's LLM client (finsight.llm.client.complete), so the provider (mock / anthropic / openai),
the spend limit, the usage meter and the data-block rule for untrusted text all apply. The client has no native
tool calling, so actions are JSON in plain text. A reply that is not a valid action is sent back once with the
error; a second bad reply in a row ends the loop with PlannerUnavailable, and the agent falls back to the
rule-based reading (claim_reader.py). MAX_TOKENS is generous because on reasoning models (gpt-5*, o*) the limit
also covers the model's hidden reasoning; a small limit can leave an empty reply. The default mock
provider never answers in this format, so tests and mock runs always use the rules unless a test scripts the
model with set_mock_responder.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from finsight.llm.client import DATA_SAFETY_PREAMBLE, complete, data_block

MAX_STEPS = 10
MAX_TOKENS = 4096        # includes hidden reasoning tokens on gpt-5* / o* models
MAX_BAD_REPLIES = 2     # bad replies in a row before giving up on the model

SYSTEM = DATA_SAFETY_PREAMBLE + """You are the Financial Evidence Agent of FinSight, which checks what finance \
influencers say against SEC filings. You get one claim about one company (and, when present, the user's prompt), \
plus the scope of this sub-claim. You never calculate a number yourself and you never say whether the claim is \
right: the Auditor does that from the figures you report.

Work in three stages:
1. Understand (your first action, always). Read the words, not keywords. For every statement the line makes about \
this company's financials, write down: the words; the measure (one of the names below, or "not computable" with \
why); the period it refers to; the figure stated, if any ("22%", "$110 billion"); the direction ("up", "down", \
"level"); and whether it is in this sub-claim's scope. Think about what the speaker means: "the top line jumped" \
is revenue growth; "they're printing cash" is free cash flow; "margins got crushed" is a margin falling; \
"earnings" next to "call" or "date" is an event, not a figure; a share price is not a company figure.
2. Compute. For each statement in scope that is computable, call the matching report action with the period you \
read. Look at each result before the next step. If the result does not fit what the words say (wrong period, the \
latest quarter is unclear), use list_periods and choose again only if the words allow it.
3. Finish, once every statement in scope has been reported or skipped. Statements out of scope belong to another \
sub-claim: leave them out of the computing, they are already noted.

Measures you can compute (knowledge layer v0.6):
- Growth, year on year: Revenue, OperatingIncome, NetIncome, OperatingCashFlow, DilutedEPS
- Levels, with the figure a year earlier: gross_margin, operating_margin, net_margin, free_cash_flow, fcf_margin,
  current_ratio, cash_ratio
- leverage (debt to equity): call skip; debt is not in the SEC data

How to read the claim:
- A claim normally holds one measure. If it holds more, report each. If a user prompt is given, report only what it
  asks about; if the claim does not state what the prompt asks about, skip it and say so.
- "Grew 24%" about the company with no measure word means Revenue. "Earnings" and "profits" mean NetIncome;
  "operating profit" means OperatingIncome. "Margins" with no type: operating_margin, then gross_margin.
  "Profitability": operating_margin and net_margin. "Cash flow" alone: free_cash_flow.
- A figure for another company in the same sentence is not this company's: ignore it.
- Skip (with a reason) anything you cannot compute: a segment or product (AWS, Azure, Google Cloud, data center,
  Services, subscriptions, search), a run rate, gross profit, EBITDA or adjusted / non-GAAP figures, net income
  excluding gains, a debt amount, a share price, valuation multiples, forecasts, and history longer than the latest
  annual report plus four quarters ("first time since 2004", "all-time high", "twelfth straight quarter").
- A phrase about the whole company with no measure ("financially strong", "a very strong financial position"):
  report operating_margin, free_cash_flow and current_ratio, and skip leverage.

Which period (in this order): the period in the user's prompt (only if a user_prompt block is given), then the one in
the claim, then one named in the preceding_sentences block (the speaker's own words just before the claim, e.g.
"Nvidia just reported its second quarter"), then: a margin or ratio stated in the present with a number ("is 34%")
= latest_quarter; otherwise none_named. Never take the period from the user_period_hint block: choose none_named and
the agent fills the gap itself (a stated amount, then that hint, then the latest fiscal year).
Period objects:
  {"type": "none_named"}                          nothing named: latest fiscal year, 12 months shown as context
  {"type": "latest_quarter"}                      "last quarter", "this quarter", "recently", "quarterly"
  {"type": "latest_fiscal_year"}                  "full year", "fiscal year", "in a year"
  {"type": "trailing_twelve_months"}              "trailing 12 months", "last twelve months", "TTM"
  {"type": "last_n_quarters", "n": 2}             "the last two quarters"
  {"type": "calendar_quarter", "year": 2026, "quarter": 1}       "Q1 2026", "first quarter of 2026"
  {"type": "fiscal_quarter", "year": 2026, "quarter": 1}         "fiscal Q1 2026", "Q1 FY2026"
  {"type": "quarter_ending_month", "year": 2026, "month": 3}     "the March quarter"
  {"type": "fiscal_year", "year": 2025}                          "fiscal 2025", "FY2025"
Use null for a year the words do not give.

Actions (reply with exactly ONE JSON object and nothing else):
  {"thought": "<your reasoning about the words>", "action": "understand", "args": {"parts": [
      {"words": "<quote>", "measure": "<measure name or not computable>", "period": {...} or null,
       "stated": "<figure as written>" or null, "direction": "up"|"down"|"level"|null, "in_scope": true|false,
       "why": "<one line: how you read it>"}]}}
  {"thought": "<why>", "action": "list_periods", "args": {}}
  {"thought": "<why>", "action": "report_growth", "args": {"measure": "<growth measure>", "period": {...}}}
  {"thought": "<why>", "action": "report_level", "args": {"measure": "<level measure>", "period": {...},
                                                         "with_growth": <true if the words give a % change>}}
  {"thought": "<why>", "action": "report_streak", "args": {"measure": "<growth measure>", "count": <n>,
                                                          "span": "quarter"|"annual", "threshold": <0.10>}}
  {"thought": "<why>", "action": "skip", "args": {"part": "<words>", "reason": "<why not checked>"}}
  {"thought": "<summary of what was reported and skipped>", "action": "finish", "args": {}}
Each result is shown to you before your next step. If a result says there is no figure, do not retry the same
action; choose another period only if the claim's words allow it, otherwise move on. Finish when every part of the
claim has been reported or skipped."""

ACTIONS = {"understand", "list_periods", "report_growth", "report_level", "report_streak", "skip", "finish"}


class PlannerUnavailable(RuntimeError):
    """The model did not answer with a valid action (or could not be called): use the rules instead."""


@dataclass
class Step:
    thought: str
    action: str
    args: dict
    observation: str = ""


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    finished: bool = False
    reading: list[dict] = field(default_factory=list)     # the model's `understand` parts

    @property
    def understood(self) -> bool:
        return any(s.action == "understand" for s in self.steps)

    def trace(self) -> str:
        """The reasoning, step by step, for the decision log."""
        lines = []
        for i, s in enumerate(self.steps, 1):
            lines.append(f"{i}. {s.thought} -> {s.action} {json.dumps(s.args, sort_keys=True)}")
            if s.observation:
                lines.append(f"   {s.observation}")
        return "\n".join(lines)


def period_spec(obj) -> tuple | None:
    """A period object from the model -> the period spec the agent uses (see claim_reader.named_period).
    Raises ValueError on an unknown type."""
    if obj in (None, {}, "none_named"):
        return None
    t = obj.get("type") if isinstance(obj, dict) else obj
    year = obj.get("year") if isinstance(obj, dict) else None
    year = int(year) if year not in (None, "") else None
    if t == "none_named":
        return None
    if t == "latest_quarter":
        return ("latest", "quarter")
    if t == "latest_fiscal_year":
        return ("latest", "annual")
    if t == "trailing_twelve_months":
        return ("ttm",)
    if t == "last_n_quarters":
        return ("last_n", max(1, min(4, int(obj.get("n", 2)))))
    if t == "calendar_quarter":
        return ("cal_q", year, int(obj["quarter"]))
    if t == "fiscal_quarter":
        return ("fiscal_q", year, int(obj["quarter"]))
    if t == "quarter_ending_month":
        return ("month_q", year, int(obj["month"]))
    if t == "fiscal_year":
        if year is None:
            return ("latest", "annual")
        return ("fy", year)
    raise ValueError(f"unknown period type {t!r}")


def parse_action(text: str) -> Step:
    """The first JSON object in the reply, as a Step. Raises PlannerUnavailable if there is none."""
    start = (text or "").find("{")
    if start < 0:
        raise PlannerUnavailable("the model's reply has no JSON action")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as e:
        raise PlannerUnavailable(f"the model's reply is not valid JSON ({e.msg})") from e
    action = obj.get("action") if isinstance(obj, dict) else None
    if action not in ACTIONS:
        raise PlannerUnavailable(f"unknown action {action!r}")
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        raise PlannerUnavailable("args must be an object")
    return Step(str(obj.get("thought", "")), action, args)


def build_prompt(context: dict, plan: Plan) -> str:
    """The user prompt for the next step: the claim and its context as data blocks, the sub-claim's scope, then
    the steps so far. The rule-based reading is shown only after the model's own `understand` step."""
    parts = [data_block("claim", context["line"])]
    if context.get("parent_span") and context["parent_span"] != context["line"]:
        parts.append(data_block("expanded_from", context["parent_span"]))
    if context.get("prompt"):
        parts.append(data_block("user_prompt", context["prompt"]))
    if context.get("feedback"):
        parts.append(data_block("auditor_feedback", context["feedback"]))
    if context.get("period_field"):
        parts.append(data_block("extracted_period", str(context["period_field"])))
    if context.get("preceding"):
        parts.append(data_block("preceding_sentences", context["preceding"]))
    if context.get("user_text"):
        parts.append(data_block("user_period_hint", context["user_text"]))
    parts.append(f"Company: {context['company']} ({context['ticker']}). Cut-off: {context['cutoff']}.\n"
                 f"Scope of this sub-claim: {context.get('scope') or 'every measure the line states'}"
                 + ("\nThe extracted_period block is how claim extraction read the period; use it only if the "
                    "words name no period." if context.get("period_field") else ""))
    if plan.steps:
        parts.append("Steps so far:\n" + plan.trace())
    if not plan.understood:
        parts.append("Your first action must be `understand`.")
    parts.append("Reply with your next action as one JSON object.")
    return "\n\n".join(parts)


def run(context: dict, execute: Callable[[Step], str], max_steps: int = MAX_STEPS) -> Plan:
    """The reasoning loop: ask the model for an action, carry it out, show the result, repeat until finish."""
    plan = Plan()
    bad = 0
    for _ in range(max_steps):
        try:
            reply = complete(build_prompt(context, plan), system=SYSTEM, max_tokens=MAX_TOKENS)
        except Exception as e:                                         # noqa: BLE001 - no key, spend limit, ...
            raise PlannerUnavailable(f"the model could not be called ({type(e).__name__}: {e})") from e
        try:
            step = parse_action(reply.text)
        except PlannerUnavailable as e:
            if reply.finish_reason in ("length", "max_tokens"):
                e = PlannerUnavailable(f"{e}; the reply hit the {MAX_TOKENS}-token limit")
            bad += 1
            if bad >= MAX_BAD_REPLIES:
                raise e
            plan.steps.append(Step("", "invalid_reply", {"reply": (reply.text or "")[:200]},
                                   f"ERROR: {e}. Reply with exactly one JSON object: "
                                   '{"thought": "...", "action": "...", "args": {...}}'))
            continue
        bad = 0
        plan.steps.append(step)
        if step.action == "finish":
            if not plan.understood:
                step.observation = "REFUSED: record your reading with `understand` before finishing."
                continue
            plan.finished = True
            break
        if step.action == "understand":
            raw = step.args.get("parts")
            plan.reading = [p for p in raw if isinstance(p, dict)] if isinstance(raw, list) else []
        elif not plan.understood:
            step.observation = "REFUSED: your first action must be `understand`."
            continue
        step.observation = execute(step)
    return plan
