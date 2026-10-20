# PLACEHOLDER: owned by Person 4 (Irine). Replace with real implementation. Do not change the signature.
"""Advice boundary stub: regex patterns for advice requests (input) and advice language (output)."""

from __future__ import annotations

import re

from finsight.contracts.schemas import AdviceDecision, AdviceStage

_INPUT_PATTERNS = [
    r"\bshould i (buy|sell|invest|put|hold|add)\b",
    r"\bhow much should i (invest|put|buy)\b",
    r"\bmy (portfolio|savings|retirement|401k)\b",
]
_OUTPUT_PATTERNS = [
    r"\byou should (buy|sell|invest|hold)\b",
    r"\bwe recommend (buying|selling)\b",
    r"\bstrong buy\b",
    r"\bput your savings\b",
]
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def without_advice(text: str) -> str:
    """Drop sentences that tell the reader to buy or sell. The claim text is not passed here."""
    kept: list[str] = []
    for match in _SENTENCE.findall(text or ""):
        sentence = match.strip()
        if not sentence:
            continue
        if any(re.search(pattern, sentence, flags=re.IGNORECASE) for pattern in _OUTPUT_PATTERNS):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


class StubAdviceBoundary:
    name = "advice_boundary_stub"

    def check(self, text: str, stage: AdviceStage) -> AdviceDecision:
        # `reason` is shown to end users, so keep it plain-English (no regexes or internal terms).
        patterns = _INPUT_PATTERNS if stage is AdviceStage.INPUT else _OUTPUT_PATTERNS
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                reason = ("This asks what you personally should buy, sell or invest in. FinSight checks the "
                          "evidence behind a claim, but doesn't give personal investment advice."
                          if stage is AdviceStage.INPUT else
                          "The answer contained wording that could read as personal investment advice.")
                return AdviceDecision(stage=stage, flagged=True, reason=reason)
        return AdviceDecision(stage=stage, flagged=False)
