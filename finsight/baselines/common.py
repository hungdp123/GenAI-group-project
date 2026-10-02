"""Shared prompt, output format, parser and result type for the baselines. Owner: Person 5 (Hung)."""

from __future__ import annotations

import json
import re
from datetime import date
from importlib import resources

from pydantic import BaseModel, Field

from finsight.contracts.schemas import AgentCall, Verdict
from finsight.llm.client import DATA_SAFETY_PREAMBLE

# DECISION: baselines get the same refusal and scope policy as FinSight, so the comparison measures
# the architecture rather than which system was told the rules.
BASELINE_INSTRUCTIONS = (
    "You evaluate a company-level investment thesis using only information available on or before {cutoff}. "
    "Do NOT give personalized investment advice. Reply with exactly one line 'REFUSED: <reason>' and nothing "
    "else if the user asks what they personally should buy, sell or invest, if the thesis is a stock price "
    "prediction, or if it is about the macro economy or a market index rather than a specific company.\n"
    "Otherwise split the thesis into atomic claims (one checkable statement each, including claims the "
    "thesis implies) and output one line per claim:\n"
    "CLAIM: <claim> | VERDICT: <Supported | Partially Supported | Contradicted | Insufficient Evidence>\n"
    "then one line 'OVERALL: <verdict>', then a short explanation citing the provided data."
)


def baseline_system_prompt(cutoff: date) -> str:
    """Trusted instructions only. The thesis and documents go in data blocks in the user prompt."""
    return DATA_SAFETY_PREAMBLE + BASELINE_INSTRUCTIONS.format(cutoff=cutoff.isoformat())


class ParsedClaim(BaseModel):
    text: str
    verdict: Verdict | None


class BaselineResult(BaseModel):
    system: str
    cutoff_date: date
    raw_output: str = ""
    refused: bool = False
    refusal_reason: str | None = None
    claims: list[ParsedClaim] = Field(default_factory=list)
    overall_verdict: Verdict | None = None
    agent_calls: list[AgentCall] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


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


def parse_output(text: str) -> tuple[bool, str | None, list[ParsedClaim], Verdict | None]:
    """Light parser: (refused, refusal_reason, claims, overall). Tolerates free-form output."""
    refusal = re.search(r"^\s*REFUSED:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if refusal:
        return True, refusal.group(1).strip(), [], None
    claims = [ParsedClaim(text=m.group(1).strip(), verdict=parse_verdict(m.group(2)))
              for m in re.finditer(r"CLAIM:\s*(.+?)\s*\|\s*VERDICT:\s*(.+)", text, flags=re.IGNORECASE)]
    overall_match = re.search(r"OVERALL:\s*(.+)", text, flags=re.IGNORECASE)
    overall = parse_verdict(overall_match.group(1)) if overall_match else None
    if not claims:
        # Fallback for free-form answers: any line that names a verdict becomes a claim.
        for line in text.splitlines():
            if (v := parse_verdict(line)) and len(line.strip()) > 10:
                claims.append(ParsedClaim(text=line.strip()[:200], verdict=v))
    return False, None, claims, overall


def xbrl_facts_as_text(cutoff: date) -> str:
    """Frozen XBRL facts filed on/before cutoff, rendered as plain text for the direct baseline."""
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
