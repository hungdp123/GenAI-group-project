"""Match the user's prompt to the influencer line(s) it refers to.

    1. Filter  (catalog.lines_for): this influencer, this company, said on or before the cut-off.
    2. Rank    every line by how many of the user's facts it covers (facts.coverage), BM25 as a tie-break.
    3. Pick    the best video and the lines in it that cover the user's facts.
               FINSIGHT_MATCHER=llm (default in paid mode) asks a model to choose among the top
               numbered lines; it can only answer with ids from that list, so it cannot invent a quote.
               Any invalid answer falls back to the rules pick.
    4. Align   each user fact with each quote fact: both / conflict / quote_only / user_only.

The matcher never decides whether a figure is right. A different figure is a `conflict`: the
influencer's own line is still the one that gets verified, and the mismatch is shown to the user.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from finsight.agents.claim_extraction.fields import bm25_scores
from finsight.agents.claim_matching.catalog import Line
from finsight.agents.claim_matching.facts import (
    BOTH,
    CONFLICT,
    QUOTE_ONLY,
    USER_ONLY,
    Aligned,
    Fact,
    align,
    coverage,
    per_measure,
    split_facts,
)

MAX_LINES = 4          # lines from one video that one prompt can point to
MAX_CANDIDATES = 8     # lines shown to the LLM picker
TIE_MARGIN = 0.35      # two videos closer than this, both covering every fact, means "which one?"

NOT_FOUND = (
    "That line is not in this influencer's stored videos. "
    "Restate the line, or resubmit with the video id."
)


_LEAD = re.compile(
    r"^\W*(?:(?:has\s+|had\s+)?(?:said|says|saying|claimed|claims|claim|stated|states|thinks|believes|argued|"
    r"argues|mentioned|mentions|told viewers|noted|notes)\b\W*)(?:that\b\s*)?",
    flags=re.IGNORECASE,
)
_ORPHAN_POSSESSIVE = re.compile(r"(^|\s)['’]s\b")
_QUESTION_TAIL = re.compile(
    r"(?:[.?!]\s*|\s+)(?:is (?:that|this|he|she|it) (?:right|true|correct|accurate)|which (?:one )?is right|"
    r"who is right|are (?:they )?both right|is it true|can you check|check (?:this|that|it)|true or false)\b.*$",
    flags=re.IGNORECASE,
)
_PRICE_ONLY = re.compile(r"\b(?:could be worth|price target|stock price|share price)\b", flags=re.IGNORECASE)
_PRICE_FACT = re.compile(
    r"\b(?:stock|(?<!per )shares?|share price|stock price|price target|market cap|worth \$|could be worth|"
    r"valuation|p/?e\b|trading at)\b",
    flags=re.IGNORECASE,
)


def clean_user_claim(text: str) -> str:
    """The user's statement without "said that", a dangling 's, or a trailing "is that right?"."""
    text = _QUESTION_TAIL.sub("", text or "")
    text = _ORPHAN_POSSESSIVE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEAD.sub("", text)
    return text.strip(" ,")


def price_facts(facts: list[Fact]) -> tuple[list[Fact], list[Fact]]:
    """(checkable facts, share-price facts). A price view is never matched or verified."""
    keep, price = [], []
    for fact in facts:
        (price if _PRICE_FACT.search(fact.span) else keep).append(fact)
    return keep, price


@dataclass
class Scored:
    line: Line
    facts: list[Fact]
    score: float
    both: int
    conflict: int


@dataclass
class VideoPick:
    video_id: str
    lines: list[Scored]
    covered_both: int
    covered_any: int
    score: float


@dataclass
class MatchOutcome:
    status: str                      # MATCHED | NONE | ASK
    user_claim: str
    user_facts: list[Fact] = field(default_factory=list)
    skipped: list[Fact] = field(default_factory=list)  # share-price views in the prompt: not checked
    lines: list[Line] = field(default_factory=list)
    aligned: list[Aligned] = field(default_factory=list)
    method: str = "rules"
    considered: int = 0
    question: str | None = None

    @property
    def video_id(self) -> str | None:
        return self.lines[0].video_id if self.lines else None


def matcher_mode() -> str:
    """rules | llm. Default: llm in paid mode, rules in mock mode (tests and free runs)."""
    mode = os.getenv("FINSIGHT_MATCHER", "auto").strip().lower()
    if mode in ("rules", "llm"):
        return mode
    from finsight.config import get_settings

    return "rules" if get_settings().llm_provider == "mock" else "llm"


OTHER_COMPANY_PENALTY = 1.0


def score_lines(user_facts: list[Fact], lines: list[Line], user_claim: str,
                ticker: str | None = None) -> list[Scored]:
    """Lines that cover at least one of the user's facts, best first.

    A line stored under another company that only mentions this one ("ARM's chip has $2 billion of demand
    with Meta as lead partner") is kept but ranked lower."""
    if not lines:
        return []
    bm25 = bm25_scores(user_claim, [line.text for line in lines])
    top = max(bm25) or 1.0
    scored: list[Scored] = []
    for line, lexical in zip(lines, bm25):
        facts = split_facts(line.text)
        score, both, conflict = coverage(user_facts, facts)
        if both + conflict == 0:
            continue
        penalty = OTHER_COMPANY_PENALTY if ticker and line.ticker not in (ticker, "NONE", "") else 0.0
        scored.append(Scored(line, facts, score + 0.5 * lexical / top - penalty, both, conflict))
    scored.sort(key=lambda item: (item.both, item.score), reverse=True)
    return scored


def _video_picks(user_facts: list[Fact], scored: list[Scored]) -> list[VideoPick]:
    """Per video: the best line, plus lines that cover facts the best line misses."""
    wanted = per_measure(user_facts)
    by_video: dict[str, list[Scored]] = {}
    for item in scored:
        by_video.setdefault(item.line.video_id, []).append(item)
    picks: list[VideoPick] = []
    for video_id, items in by_video.items():
        chosen = [items[0]]
        state = _covered(wanted, chosen)
        for item in items[1:]:
            if len(chosen) >= MAX_LINES or state[0] == len(wanted):
                break
            trial = _covered(wanted, [*chosen, item])
            if trial[0] > state[0] or (trial[1] > state[1] and trial[0] == state[0]):
                chosen.append(item)
                state = trial
        picks.append(VideoPick(video_id, chosen, state[0], state[1], state[2]))
    picks.sort(key=lambda pick: (pick.covered_both, pick.covered_any, pick.score), reverse=True)
    return picks


def _covered(wanted: list[Fact], chosen: list[Scored]) -> tuple[int, int, float]:
    """(facts covered as both, facts covered at all, total score) by these lines together."""
    both = anything = 0
    total = 0.0
    facts = [fact for item in chosen for fact in item.facts]
    for user in wanted:
        score, b, c = coverage([user], facts)
        both += b
        anything += b or c
        total += score
    return both, anything, total


def match_claim(
    user_claim: str,
    lines: list[Line],
    *,
    video_id: str | None = None,
    mode: str | None = None,
    fallback: list[Line] | None = None,
    ticker: str | None = None,
) -> MatchOutcome:
    """Find the line(s) the user means. `lines` are already filtered to this influencer and company.

    `fallback` is searched only when no catalog line covers any fact: sentences from the stored transcripts,
    for a line the catalog missed.
    """
    user_claim = clean_user_claim(user_claim)
    user_facts, skipped = price_facts(split_facts(user_claim))
    outcome = MatchOutcome(status="NONE", user_claim=user_claim, user_facts=user_facts, skipped=skipped,
                           considered=len(lines))
    if not user_facts:
        outcome.question = NOT_FOUND
        return outcome
    pool = [line for line in lines if not video_id or line.video_id == video_id]
    pool = [line for line in pool if not _PRICE_ONLY.search(line.text)]
    scored = score_lines(user_facts, pool, user_claim, ticker)
    wanted = len(per_measure(user_facts))
    if fallback:
        # Transcript sentences the catalog did not store compete on the same terms (the catalog wins ties).
        stored = {(line.video_id, _key(line.text)) for line in pool}
        spare = [line for line in fallback
                 if (not video_id or line.video_id == video_id)
                 and (line.video_id, _key(line.text)) not in stored
                 and not _PRICE_ONLY.search(line.text)
                 and not line.text.rstrip().endswith("?")]  # a question is not a claim
        extra = score_lines(user_facts, spare, user_claim, ticker)
        for item in extra:
            item.score -= 0.05
        scored = sorted([*scored, *extra], key=lambda item: (item.both, item.score), reverse=True)
        pool = [*pool, *spare]
    picks = _video_picks(user_facts, scored)
    mode = mode or matcher_mode()
    if mode == "llm":
        chosen = _llm_pick(user_claim, _llm_candidates(scored, pool, user_claim))
        if chosen is not None:
            outcome.method = "llm"
            videos = {line.video_id for line in chosen}
            if not chosen:
                outcome.question = NOT_FOUND
                return outcome
            if len(videos) > 1 and not video_id:
                # The model often lists the same statement from every video that repeats it. Ask only when the
                # rules also see a tie; otherwise keep the video the rules rank highest (or the model's first pick
                # when the rules did not find it: a paraphrase only the model caught).
                ranked = [pick for pick in picks if pick.video_id in videos]
                if _is_tie(ranked, user_facts):
                    return _ask(outcome, ranked)
                keep = ranked[0].video_id if ranked else chosen[0].video_id
                chosen = [line for line in chosen if line.video_id == keep]
            return _matched(outcome, user_facts, chosen)
    if not picks:
        outcome.question = NOT_FOUND
        return outcome
    best = picks[0]
    if not video_id and _is_tie(picks, user_facts):
        return _ask(outcome, picks[:2])
    return _matched(outcome, user_facts, [item.line for item in best.lines])


def _is_tie(picks: list[VideoPick], user_facts: list[Fact]) -> bool:
    """Two videos both state every fact the user typed, with scores this close: ask which one."""
    if len(picks) < 2:
        return False
    wanted = len(per_measure(user_facts))
    best, second = picks[0], picks[1]
    return best.covered_both == second.covered_both == wanted and best.score - second.score < TIE_MARGIN


def _matched(outcome: MatchOutcome, user_facts: list[Fact], lines: list[Line]) -> MatchOutcome:
    aligned = align(user_facts, [split_facts(line.text) for line in lines])
    # A line kept only for quote_only facts is not what the user pointed at (the first line always stays).
    used = {item.quote_index for item in aligned if item.bucket in (BOTH, CONFLICT)}
    keep = [index for index in range(len(lines)) if index == 0 or index in used]
    if len(keep) != len(lines):
        lines = [lines[index] for index in keep]
        aligned = align(user_facts, [split_facts(line.text) for line in lines])
    outcome.status = "MATCHED"
    outcome.lines = lines
    outcome.aligned = aligned
    return outcome


def _ask(outcome: MatchOutcome, picks: list[VideoPick]) -> MatchOutcome:
    rows = []
    for pick in picks[:2]:
        line = pick.lines[0].line
        when = line.upload_date.isoformat() if line.upload_date else "undated"
        rows.append(f"{when}  {line.video_id}  {line.title} — \"{_short(line.text)}\"")
    outcome.status = "ASK"
    outcome.question = (
        "Two stored posts state this. Which post should FinSight check?\n" + "\n".join(rows)
        + "\nResubmit with the video id."
    )
    return outcome


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9%$.]+", " ", (text or "").lower()).strip()


def _short(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# -- LLM pick (paid mode) ----------------------------------------------------------

_PICK_SYSTEM = (
    "You match a user's paraphrase of what an influencer said to the influencer's actual stored lines. "
    "Pick the line(s) the user is referring to: the same statement, even if the user's numbers are wrong. "
    "Pick several only when the user's message refers to several different statements. "
    "If none of the lines is what the user means, return an empty list. "
    'Answer with JSON only: {"match_ids": ["L1", ...], "reason": "<one sentence>"}. '
    "Use only ids from the list."
)


def _llm_candidates(scored: list[Scored], pool: list[Line], user_claim: str) -> list[Scored]:
    """The rules' best lines, topped up with the lines that share the most words, so a paraphrase with no
    shared figure or measure word can still be offered to the model."""
    out = list(scored[: MAX_CANDIDATES - 2])
    seen = {id(item.line) for item in out}
    if pool:
        lexical = bm25_scores(user_claim, [line.text for line in pool])
        for _score, line in sorted(zip(lexical, pool), key=lambda pair: pair[0], reverse=True):
            if len(out) >= MAX_CANDIDATES or _score <= 0:
                break
            if id(line) not in seen:
                out.append(Scored(line, split_facts(line.text), 0.0, 0, 0))
                seen.add(id(line))
    return out


def _llm_pick(user_claim: str, scored: list[Scored]) -> list[Line] | None:
    """Lines the model picked, [] for "none of these", or None when the answer is unusable."""
    from finsight.llm.client import DATA_SAFETY_PREAMBLE, complete, data_block

    if not scored:
        return []
    ids = {f"L{index}": item.line for index, item in enumerate(scored, start=1)}
    listing = "\n".join(
        f"{key} ({line.video_id}, {line.upload_date}): {line.text}" for key, line in ids.items()
    )
    prompt = data_block("user_claim", user_claim) + "\n" + data_block("stored_lines", listing)
    try:
        response = complete(prompt, system=DATA_SAFETY_PREAMBLE + _PICK_SYSTEM, max_tokens=800)
        payload = _json(response.text)
    except Exception:  # noqa: BLE001 - a failed model call falls back to the rules pick
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("match_ids"), list):
        return None
    picked = [ids[key] for key in payload["match_ids"] if isinstance(key, str) and key in ids]
    if len(picked) != len(payload["match_ids"]):
        return None  # an id outside the list: do not trust the answer
    return picked[:MAX_LINES]


def _json(text: str) -> object:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    return json.loads(match.group(0)) if match else None


__all__ = ["MatchOutcome", "match_claim", "matcher_mode", "NOT_FOUND", "BOTH", "CONFLICT", "QUOTE_ONLY", "USER_ONLY"]
