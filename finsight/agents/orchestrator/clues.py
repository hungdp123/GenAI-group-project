"""Clues that help check a post line without changing what is checked.

1. context: the sentences before the line in the post. A speaker often names the period once ("Nvidia just
   reported its second quarter") and then gives figures without repeating it.
2. user_text: what the user typed. Used only as a period hint when the speaker named none.
3. prompt_conflicts: figures or a period the user typed that the checked line does not say ("you said 22%;
   the post says 28%"). The line is still what gets checked; the conflict is reported, never resolved quietly.
"""

from __future__ import annotations

import re

from finsight.agents.claim_extraction.fields import _values_match, parse_fields
from finsight.agents.financial.claim_reader import context_period, named_period
from finsight.contracts.schemas import SubClaim

CONTEXT_SENTENCES = 3
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%$.]+", " ", (text or "").lower()).strip()


def preceding_sentences(transcript: str, line: str, n: int = CONTEXT_SENTENCES) -> str | None:
    """The n sentences before `line` in the transcript, or None when the line is not found."""
    sentences = [s.strip() for s in _SENTENCE.split(transcript or "") if s.strip()]
    target = _norm(line)
    if not target:
        return None
    for index, sentence in enumerate(sentences):
        norm = _norm(sentence)
        if norm and (target in norm or norm in target) and len(norm) >= min(len(target), 20):
            before = sentences[max(0, index - n):index]
            return " ".join(before) or None
    return None


def with_clues(sub_claims: list[SubClaim], transcript: str, user_text: str | None) -> list[SubClaim]:
    """Each sub-claim with its preceding sentences and the user's words attached (only fields still empty)."""
    out = []
    for claim in sub_claims:
        update = {}
        split_fact = bool(claim.parent_span and claim.context == claim.parent_span and not claim.expansion_rule)
        if not claim.context or split_fact:
            context = preceding_sentences(transcript, claim.source_span or claim.text)
            # A fact split out of a longer line keeps the whole line as its nearest context, so a period named
            # elsewhere in the line ("reported earnings a week ago ... $81.6 billion") still applies.
            context = " ".join(part for part in (context, claim.context if split_fact else None) if part)
            if context and context != claim.context:
                update["context"] = context
        if user_text and not claim.user_text:
            update["user_text"] = user_text.strip()
        out.append(claim.model_copy(update=update) if update else claim)
    return out


def _show(value: float, unit: str) -> str:
    number = f"{value:g}"
    return {"percent": f"{number}%", "billion": f"${number} billion", "million": f"${number} million",
            "dollar": f"${number}"}.get(unit, number)


_PERIOD_NAMES = {"ttm": "the trailing twelve months", "fy": "fiscal year {1}",
                 "fiscal_q": "fiscal Q{2}", "cal_q": "Q{2}", "month_q": "the quarter ending in month {2}",
                 "last_n": "the last {1} quarters"}


def _period_name(spec: tuple) -> str:
    if spec[0] == "latest":
        return "the latest quarter" if spec[1] == "quarter" else "the latest fiscal year"
    template = _PERIOD_NAMES.get(spec[0], str(spec))
    return template.format(*spec, *([""] * 3)).replace("fiscal year None", "a fiscal year")


def prompt_conflicts(user_text: str | None, sub_claims: list[SubClaim], ticker: str | None) -> list[str]:
    """Plain sentences for each figure or period the user typed that the checked lines do not state."""
    if not user_text or not sub_claims:
        return []
    asked = parse_fields(user_text, company=ticker)
    lines = [parse_fields(c.text, company=ticker) for c in sub_claims]
    said_values = tuple(v for f in lines for v in f.values)
    notes = []
    # A typed "$110 billion" also parses as a bare "$110"; compare each figure once.
    asked_values = [(v, u) for v, u in asked.values
                    if not (u == "dollar" and any(u2 in ("billion", "million") and v2 == v for v2, u2 in asked.values))]
    missing = [(v, u) for v, u in asked_values if not _values_match(((v, u),), said_values)]
    if missing:
        stated = [(v, u) for v, u in said_values
                  if not (u == "dollar" and any(u2 in ("billion", "million") and v2 == v for v2, u2 in said_values))]
        says = (" and ".join(_show(v, u) for v, u in stated[:3])) if stated else "no figure"
        notes.append(f"You said {' and '.join(_show(v, u) for v, u in missing)}; the post says {says}. "
                     "The post's figure is what was checked.")
    asked_period = named_period(user_text)
    said_period = next((p for c in sub_claims for p in (named_period(c.text), context_period(c.context))
                        if p is not None), None)
    if asked_period and said_period and _clearly_different(asked_period, said_period):
        notes.append(f"You said {_period_name(asked_period)}; the post refers to {_period_name(said_period)}. "
                     "The post's period is what was checked.")
    return notes


def _span(spec: tuple) -> str:
    if spec[0] == "latest":
        return spec[1]
    return {"fy": "annual", "ttm": "ttm"}.get(spec[0], "quarter")


def _clearly_different(a: tuple, b: tuple) -> bool:
    """Only a real mismatch: a quarter vs a year, or two named quarters / years that differ.
    "Last quarter" vs "the second quarter" may well be the same quarter, so it is not flagged."""
    if _span(a) != _span(b):
        return True
    if a[0] == b[0] and a[0] in ("fiscal_q", "cal_q", "month_q"):
        quarter_differs = a[2] != b[2]
        year_differs = a[1] is not None and b[1] is not None and a[1] != b[1]
        return quarter_differs or year_differs
    if a[0] == b[0] == "fy":
        return a[1] is not None and b[1] is not None and a[1] != b[1]
    return False
