"""Shared prompt, output format, parser and result type for the baselines. Owner: Person 5 (Hung)."""

from __future__ import annotations

import json
import re
from datetime import date
from importlib import resources

from pydantic import BaseModel, Field

from finsight.contracts.schemas import AgentCall, Verdict
from finsight.llm.client import DATA_SAFETY_PREAMBLE

_SCOPE = (
    "You evaluate a company-level investment claim using only information available on or before {cutoff}. "
    "Do NOT give personalized investment advice. Reply with exactly one line 'REFUSED: <reason>' and nothing "
    "else if the user asks what they personally should buy, sell or invest, if the claim is a stock price "
    "prediction, or if it is about the macro economy or a market index rather than a specific company.\n"
)
_CLARIFY_WITH_PERIOD_RULES = (
    "If the claim does not say which company it is about, or contains no checkable sub-claim, reply with exactly "
    "one line 'CLARIFY: <one question for the user>' and nothing else. Do not ask for anything else: in "
    "particular never ask which period is meant.\n"
    # DECISION (30 Oct 2026): the same period and evidence rules FinSight applies (knowledge_layer.md sections 11-12),
    # so a baseline loses on reliability, not on an instruction it was never given.
    "Periods: 'last quarter', 'this quarter', 'recent' or 'latest earnings' mean the latest quarter that ended on "
    "or before {cutoff}, compared with the same quarter a year earlier. A claim that names no period means the "
    "latest fiscal year that ended on or before {cutoff}, compared with the year before.\n"
    "If the information you have cannot verify a sub-claim, label it Insufficient Evidence; do not ask the user.\n"
)
# The prompt before 30 Oct 2026 (experiment: do the period rules matter?). FINSIGHT_BASELINE_PERIOD_RULES=0.
_CLARIFY_WITHOUT_PERIOD_RULES = (
    "If the claim is ambiguous (for example the company or period is unclear) or contains no checkable sub-claim, "
    "reply with exactly one line 'CLARIFY: <one question for the user>' and nothing else.\n"
)
_OUTPUT = (
    "Otherwise split the claim into atomic sub-claims (one checkable statement each, including sub-claims the "
    "claim implies) and output one line per sub-claim. Check the company facts only: who said the claim is not a "
    "sub-claim, so write each sub-claim without the speaker ('Microsoft revenue grew 18%', not 'X said "
    "Microsoft revenue grew 18%'):\n"
    "SUB-CLAIM: <sub-claim> | VERDICT: <Supported | Partially Supported | Contradicted | Insufficient Evidence>\n"
    "then one line 'OVERALL: <verdict>', then a short explanation citing the provided data."
)
# DECISION: baselines get the same refusal and scope policy as FinSight, so the comparison measures
# the architecture rather than which system was told the rules.
BASELINE_INSTRUCTIONS = _SCOPE + _CLARIFY_WITH_PERIOD_RULES + _OUTPUT
BASELINE_INSTRUCTIONS_NO_PERIOD_RULES = _SCOPE + _CLARIFY_WITHOUT_PERIOD_RULES + _OUTPUT


def period_rules_on() -> bool:
    """Experiment switch: FINSIGHT_BASELINE_PERIOD_RULES=0 gives the baselines the old prompt without them."""
    import os

    return os.getenv("FINSIGHT_BASELINE_PERIOD_RULES", "1").strip() not in ("0", "false", "no", "off")


def baseline_system_prompt(cutoff: date) -> str:
    """Trusted instructions only. The claim and documents go in data blocks in the user prompt."""
    template = BASELINE_INSTRUCTIONS if period_rules_on() else BASELINE_INSTRUCTIONS_NO_PERIOD_RULES
    return DATA_SAFETY_PREAMBLE + template.format(cutoff=cutoff.isoformat())


class ParsedSubClaim(BaseModel):
    text: str
    verdict: Verdict | None


class BaselineResult(BaseModel):
    system: str
    cutoff_date: date
    raw_output: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    clarification_question: str | None = None
    sub_claims: list[ParsedSubClaim] = Field(default_factory=list)
    overall_verdict: Verdict | None = None
    agent_calls: list[AgentCall] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    # What the model was given besides the claim (filings text, retrieved passages, the influencer's line).
    # The comparison report checks the answer's numbers against it (made-up number rate).
    sources_text: str = ""
    retrieved_after_cutoff: int = 0


_VERDICT_WORDS = [  # order matters: check the more specific phrase first
    (r"partial|mixed", Verdict.PARTIALLY_SUPPORTED),
    (r"insufficient|not enough|unclear|cannot (be )?(verified|determined)", Verdict.INSUFFICIENT_EVIDENCE),
    (r"contradict|refuted|false|unsupported", Verdict.CONTRADICTED),
    (r"support|true|confirmed", Verdict.SUPPORTED),
]


def parse_verdict(text: str) -> Verdict | None:
    for pattern, verdict in _VERDICT_WORDS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return verdict
    return None


def parse_output(text: str) -> tuple[bool, str | None, list[ParsedSubClaim], Verdict | None]:
    """Light parser: (refused, refusal_reason, sub-claims, overall). Tolerates free-form output."""
    refusal = re.search(r"^\s*REFUSED:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if refusal:
        return True, refusal.group(1).strip(), [], None
    sub_claims = [ParsedSubClaim(text=m.group(1).strip(), verdict=parse_verdict(m.group(2)))
              for m in re.finditer(r"(?:SUB-)?CLAIM:\s*(.+?)\s*\|\s*VERDICT:\s*(.+)", text, flags=re.IGNORECASE)]
    overall_match = re.search(r"OVERALL:\s*(.+)", text, flags=re.IGNORECASE)
    overall = parse_verdict(overall_match.group(1)) if overall_match else None
    if not sub_claims:
        # Fallback for free-form answers: any line that names a verdict becomes a sub-claim.
        for line in text.splitlines():
            if (v := parse_verdict(line)) and len(line.strip()) > 10:
                sub_claims.append(ParsedSubClaim(text=line.strip()[:200], verdict=v))
    return False, None, sub_claims, overall


def parse_clarification(text: str) -> str | None:
    match = re.search(r"^\s*CLARIFY:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None


def apply_output(result: BaselineResult, text: str) -> None:
    """Fill `result` from the model's raw text (shared by both baselines)."""
    result.raw_output = text
    result.clarification_question = parse_clarification(text)
    if result.clarification_question is None:
        result.refused, result.refusal_reason, result.sub_claims, result.overall_verdict = parse_output(text)


SEC_FACTS = "data/sec/processed/facts.csv"
_SEC_CONCEPTS = ("Revenue", "GrossProfit", "OperatingIncome", "NetIncome", "OperatingCashFlow", "CapitalExpenditure",
                 "DilutedEPS")


def sec_facts_as_text(ticker: str, cutoff: date, path: str = SEC_FACTS, quarters: int = 6,
                      years: int = 3) -> str | None:
    """The real XBRL facts for one company filed on or before the cut-off (latest quarters and fiscal years),
    as plain text. None when the SEC store is not on this machine or has nothing for the company."""
    import csv
    from pathlib import Path

    file = Path(path)
    if not file.exists():
        return None
    best: dict[tuple, dict] = {}
    with file.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["ticker"] != ticker or row["concept"] not in _SEC_CONCEPTS or row["period_class"] not in ("Q", "FY"):
                continue
            if date.fromisoformat(row["filed"]) > cutoff:
                continue
            key = (row["concept"], row["period_class"], row["period_end"])
            if key not in best or row["filed"] > best[key]["filed"]:
                best[key] = row
    if not best:
        return None
    lines = []
    for concept in _SEC_CONCEPTS:
        for cls, keep in (("Q", quarters), ("FY", years)):
            rows = sorted((r for k, r in best.items() if k[0] == concept and k[1] == cls),
                          key=lambda r: r["period_end"], reverse=True)[:keep]
            for r in rows:
                value = float(r["value"])
                shown = f"{value:.2f} USD/share" if concept == "DilutedEPS" else f"${value / 1e6:,.0f}m"
                lines.append(f"{ticker} {concept} {'quarter' if cls == 'Q' else 'fiscal year'} "
                             f"{r['period_start']}..{r['period_end']} = {shown} ({r['form']} filed {r['filed']})")
    return "\n".join(lines)


def xbrl_facts_as_text(cutoff: date, ticker: str | None = None) -> str:
    """XBRL facts filed on/before cutoff, as plain text for the LLM + filings baseline: the real SEC store for the
    claim's company when it is on this machine, otherwise the frozen placeholder facts."""
    if ticker:
        real = sec_facts_as_text(ticker, cutoff)
        if real:
            return real
    raw = json.loads(resources.files("finsight.data").joinpath("xbrl_facts.json").read_text("utf-8"))
    lines = []
    for ticker, block in raw.items():
        if ticker.startswith("_"):
            continue
        for f in block["facts"]:
            if date.fromisoformat(f["filed"]) <= cutoff:
                lines.append(f"{ticker} {f['metric']} {f['period']} = {f['value']} {f['unit']} "
                             f"({f['form']} filed {f['filed']})")
    return "\n".join(lines) or "(no XBRL facts available before the cut-off)"
