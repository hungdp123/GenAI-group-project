"""Advice screen. Input refuses a request for personal advice. Output deletes that wording from the report.

Section 2 of the rule book is the input list. The last screen uses the same kinds of advice
on the drafted report: buy, sell, hold, allocate, a price target, or a guaranteed return.
The claim text and the labels stay.
"""

from __future__ import annotations

import re

from finsight.contracts.schemas import AdviceDecision, AdviceStage

# The user is asking FinSight what to do with their money, or to predict a price.
# A price named inside a company claim, such as a finfluencer's target, is not this list.
_INPUT_PATTERNS = [
    r"\bshould i (buy|sell|invest|put|hold|add)\b",
    r"\bhow much (?:of my \S+ |should i )(?:should i )?(invest|put|buy)\b",
    r"\bmy (portfolio|savings|retirement|401k)\b",
    r"\b(good|better) buy\b",
    r"\b(better|good) pick for me\b",
    r"\bwhether to (buy|sell|hold|invest)\b",
    r"\bwhich stock will\b",
    r"\bmake the most money\b",
    r"\bhow (?:should i |do i |to )?(?:split|allocate)\b",
    r"\brisk tolerance\b",
    r"\bi(?:'m| am) \d{1,2}\b(?!\s*%)",
    r"\bguaranteed returns?\b",
    r"\bwhat price will\b",
    r"\bgive me (?:a |the )?price target\b",
    r"\bwhat(?:'s| is) (?:your |a )?price target\b",
]
# Sentences FinSight wrote. The quoted user claim is left out before this list is applied.
_OUTPUT_PATTERNS = [
    r"\byou should (buy|sell|invest|hold|add)\b",
    r"\bwe recommend (buying|selling|holding)\b",
    r"\bstrong buy\b",
    r"\bput your savings\b",
    r"\ballocat\w*\b.{0,80}\b(?:portfolio|money|savings|position)\b",
    r"\ballocat\w*\s+\d+",
    r"\b(?:price target|target price)\b",
    r"\bguaranteed returns?\b",
    r"\bguarantee(?:s|d)? a return\b",
]
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_CLAIM_LINE = re.compile(r"^\s*claim checked:", flags=re.IGNORECASE)


def _hits(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _report_body(text: str) -> str:
    """The drafted report without the quoted user claim. That line is not FinSight's advice."""
    return "\n".join(line for line in (text or "").splitlines() if not _CLAIM_LINE.match(line))


def without_advice(text: str) -> str:
    """Drop sentences that tell the reader what to do with their money. The claim text is not passed here."""
    kept: list[str] = []
    for match in _SENTENCE.findall(text or ""):
        sentence = match.strip()
        if not sentence or _hits(sentence, _OUTPUT_PATTERNS):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


class AdviceBoundaryAgent:
    name = "advice_boundary"

    def check(self, text: str, stage: AdviceStage) -> AdviceDecision:
        patterns = _INPUT_PATTERNS if stage is AdviceStage.INPUT else _OUTPUT_PATTERNS
        body = text if stage is AdviceStage.INPUT else _report_body(text)
        if _hits(body, patterns):
            reason = (
                "This asks what you personally should buy, sell, or invest in, or what price or return to expect. "
                "FinSight checks the evidence behind a claim, but doesn't give personal investment advice."
                if stage is AdviceStage.INPUT
                else "The answer contained wording that could read as personal investment advice."
            )
            return AdviceDecision(stage=stage, flagged=True, reason=reason)
        return AdviceDecision(stage=stage, flagged=False)
