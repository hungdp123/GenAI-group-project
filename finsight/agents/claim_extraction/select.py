"""What the prompt states, and why a line was not checked.

Matching the prompt to the influencer's stored lines lives in finsight/agents/claim_matching
(filter -> rank -> pick -> align). This module keeps the small helpers the extraction step still uses:
`stated_claim` (the prompt without names and instructions), `matching_claims` (stored rows about the same
point, for a post that has stored claims but no match), and `explain_miss` / `miss_text` (why a line was
not checked).
"""

from __future__ import annotations

import re

from finsight.agents.claim_extraction.prepare import _SENTENCE, _drop_reason
from finsight.agents.claim_extraction.thesaurus import PRODUCTS, without_urls
from finsight.agents.orchestrator.routing import INFLUENCERS
from finsight.contracts.schemas import SubClaim

NOT_IN_VIDEO = "That line is not in this video."
NOT_IN_STORED = (
    "That line is not in this influencer's stored videos. "
    "Restate the line, or resubmit with the video id."
)
CLOSEST_LINE = "The closest line is"
MATCH_NOTE = "Checked the lines in this post that match the claim in the prompt."

_FILLER = frozenset({
    "said", "says", "saying", "claim", "claims", "check", "checked", "about", "this", "that",
    "with", "from", "just", "have", "been", "will", "into", "your", "their", "what", "when",
    "where", "right", "correct", "true", "video", "post", "influencer", "finfluencer",
    "should", "would", "could", "really", "very", "also", "than", "then", "them", "they",
    "were", "was", "are", "and", "for", "his", "her", "its", "the", "does", "did", "has",
    "had", "how", "why", "who", "which", "whether", "approximately", "around", "nearly",
    "roughly", "company", "operating", "gross", "free", "total", "recent", "last",
    "quarterly", "annual", "current", "next", "year", "over", "yoy", "billion", "million",
    "percent", "percentage", "getting", "consistent", "consistency", "accurate", "accuracy",
})
_GROUPS = {
    "growth": frozenset({
        "grew", "grow", "growing", "growth", "grown", "rise", "rose", "risen", "rising",
        "increase", "increased", "increasing", "up", "higher", "double", "doubled", "doubling",
        "twofold", "expand", "expanded", "expanding", "expansion", "improved", "improving",
        "improvement", "strong", "stronger",
    }),
    "decline": frozenset({
        "fell", "fall", "fallen", "falling", "decline", "declined", "declining", "drop", "dropped",
        "down", "lower", "decrease", "decreased", "shrink", "shrank", "contract", "contracted",
    }),
    "revenue": frozenset({"revenue", "revenues", "sales", "topline"}),
    "profit": frozenset({"profit", "profits", "earnings", "earning", "income"}),
    "margin": frozenset({"margin", "margins", "profitability"}),
    "cash": frozenset({"cash", "fcf"}),
    "debt": frozenset({"debt"}),
    "leverage": frozenset({"leverage"}),
    "guidance": frozenset({
        "guidance", "guided", "forecast", "forecasts", "expects", "expected", "expect",
        "continue", "continued", "continuing", "started", "runway",
    }),
    "risk": frozenset({"risk", "risks", "threat", "threats"}),
    "compare": frozenset({"ahead", "winner", "leader", "better", "cheaper", "biggest"}),
}
_SHORT = frozenset({"up"})
_TOKEN = re.compile(r"[a-z0-9']+(?:\.[0-9]+)?%?", flags=re.IGNORECASE)
_PRICE = re.compile(
    r"\b(share price|stock price|price target|target price|forward p/?e|p/e|trading at|buy price|market price)\b"
    r"|\b(shares|the stock)\b.{0,40}\b(sold|fell|rose|up|down|plummet)",
    flags=re.IGNORECASE,
)
_ADVICE = re.compile(
    r"\b(you should|i(?:'m| am) (?:buying|investing|putting|willing)|"
    r"(?:buy|sell)(?:ing)? the (?:stock|shares)|i(?:'ll| will) (?:buy|sell)|committing)\b",
    flags=re.IGNORECASE,
)
_MEASURES = frozenset({"revenue", "profit", "margin", "cash", "debt", "leverage"})
_DIRECTIONS = frozenset({"growth", "decline"})
_TOPICS = frozenset({"guidance", "risk", "compare"})
_CUTOFF = re.compile(
    r"cut[\s-]?off(?:\s+date)?\s*[:=-]?\s*"
    r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})",
    flags=re.IGNORECASE,
)
_WHY = {
    "guest": "it is a guest line",
    "not the host": "it is not the host",
    "question": "it is a question",
    "hypothetical": "it is a hypothetical",
    "joke": "it is marked as a joke",
    "retracted": "the video takes it back",
    "list mention": "it only lists companies",
    "other company": "it is about another company",
    "unclear they": "it is not clear which company it is about",
}


_INSTRUCTION = re.compile(
    r"ignore previous instructions.*?(?:[.!]|$)"
    r"|\[?\s*note to the ai model:.*?(?:\]|$)",
    flags=re.IGNORECASE,
)


def stated_claim(prompt: str, influencer: str | None, aliases: list[str]) -> str | None:
    """The claim in the prompt, or None when the prompt only names the creator and the company."""
    cleaned = _strip_names(_INSTRUCTION.sub(" ", prompt), influencer, aliases)
    if not _features(cleaned):
        return None
    return cleaned


def matching_claims(focus: str, claims: list[SubClaim]) -> list[SubClaim]:
    """Stored claims that say the same thing as `focus`, including a paraphrase.

    A prompt can name several points. Each stored row is kept when it covers one
    of those points. A row about a different measure, or the opposite direction,
    is left out.
    """
    wanted = _features(focus)
    if not wanted:
        return []
    return [claim for claim in claims if _covers(wanted, _features(claim.text))]


def explain_miss(focus: str, transcript: str) -> str:
    """Why the prompt was not checked. The quoted line is not a claim to verify."""
    wanted = _features(focus)
    best: tuple[int, int, str] | None = None
    for index, sentence in enumerate(_sentences(transcript)):
        found = _features(sentence)
        if not wanted or not wanted <= found:
            continue
        extra = len(found - wanted)
        if best is None or (extra, index) < (best[0], best[1]):
            best = (extra, index, sentence)
    if best is None:
        return NOT_IN_VIDEO
    quote = re.sub(r"\s+", " ", best[2]).strip()
    if len(quote) > 180:
        quote = quote[:177].rstrip() + "..."
    return f'{CLOSEST_LINE} "{quote}". It was not checked because {_why(best[2])}.'


def miss_text(checked_post: str | None) -> str | None:
    """The miss sentence carried on the post label, if this run did not verify a claim."""
    if not checked_post:
        return None
    for marker in (NOT_IN_VIDEO, CLOSEST_LINE):
        start = checked_post.find(marker)
        if start >= 0:
            return checked_post[start:].strip()
    return None


_PRODUCT_NAMES = {name.casefold() for names in PRODUCTS.values() for name in names}


def _strip_names(text: str, influencer: str | None, aliases: list[str]) -> str:
    """Drop the influencer and the company name. Keep a product name such as AWS.

    "AWS grew 24%" is about that segment. Stripping AWS makes the AWS sentence look
    like a different fact from a company-wide revenue line.
    """
    cleaned = _CUTOFF.sub(" ", without_urls(text or ""))
    names = list(aliases)
    if influencer:
        names.extend(INFLUENCERS.get(influencer, ()))
    for name in sorted(set(names), key=len, reverse=True):
        if not name or name.casefold() in _PRODUCT_NAMES:
            continue
        cleaned = re.sub(rf"\b{re.escape(name)}\b", " ", cleaned, flags=re.IGNORECASE)
    return cleaned


def _features(text: str) -> set[str]:
    found: set[str] = set()
    for token in _TOKEN.findall(text or ""):
        word = token.lower().strip("'")
        if any(character.isdigit() for character in word):
            found.add("figure")
            continue
        canon = _canon(word)
        if canon:
            found.add(canon)
    return found


def _canon(word: str) -> str | None:
    if word in _FILLER:
        return None
    if len(word) <= 2 and word not in _SHORT:
        return None
    for name, group in _GROUPS.items():
        if word in group:
            return name
    if word.endswith("s") and len(word) > 4:
        stem = word[:-1]
        for name, group in _GROUPS.items():
            if stem in group:
                return name
        return stem
    return word


def _covers(wanted: set[str], found: set[str]) -> bool:
    """True when `found` is one point the prompt asked about."""
    if not wanted or not found:
        return False
    shared = wanted & found
    if (wanted & _DIRECTIONS) and (found & _DIRECTIONS) and not (shared & _DIRECTIONS):
        return False
    wanted_measures = wanted & _MEASURES
    found_measures = found & _MEASURES
    if wanted_measures and found_measures:
        return bool(wanted_measures & found_measures)
    leftovers = wanted - _MEASURES - _DIRECTIONS - _TOPICS - {"figure"}
    shared_topics = (wanted & _TOPICS) & found
    if leftovers and not (leftovers & found) and (not shared_topics or shared_topics <= {"risk"}):
        return False
    if wanted_measures:
        return bool(shared_topics or (leftovers & found))
    if wanted & _TOPICS:
        return bool(shared_topics)
    return bool(shared - {"figure"})


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE.split(text or "") if sentence.strip()]


def _why(sentence: str) -> str:
    reason = _drop_reason(sentence, None)
    if reason in _WHY:
        return _WHY[reason]
    if _PRICE.search(sentence):
        return "it is about a price"
    if _ADVICE.search(sentence):
        return "it is advice"
    return "it is not in the stored claim list"
