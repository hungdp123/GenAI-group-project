"""Choose which stored claims a prompt checks.

A stated claim is matched to stored rows by a local semantic embedding. The row
must share the measure, and a stated figure such as 24% must be in that row.
"Grew 24%" does not attach to "search revenue increased by 17%."

If no stored row is that close, the transcripts of that influencer's videos
about the company are searched. One close spoken sentence is shown and the user
is asked to confirm it. Several are shown and the user is asked which one.
Nothing close asks the user to restate the line or name the video.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from finsight.agents.claim_extraction.fields import (
    ClaimFields,
    bm25_scores,
    company_for_sentences,
    fields_match,
    parse_fields,
)
from finsight.agents.claim_extraction.prepare import _SENTENCE, _drop_reason
from finsight.agents.claim_extraction.thesaurus import PRODUCTS, find_mentions, without_urls
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


_DIM = 256
_MIN_SCORE = 0.32
_SOFT_MIN = 0.30
_GAP = 0.12
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", flags=re.IGNORECASE)
_FOLD = re.compile(r"\b(?:doubled|tripled|halved)\b", flags=re.IGNORECASE)
_MAGNITUDE = re.compile(
    r"\$?\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?:plus\s+)?(billion|million)\b",
    flags=re.IGNORECASE,
)
_DOLLARS = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\b")


@dataclass
class LineChoice:
    """What to check, or the question to ask before checking."""

    quotes: list[str]
    video_id: str | None = None
    question: str | None = None


def choose_line(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    *,
    confirmed_video: str | None = None,
    ticker: str | None = None,
) -> LineChoice:
    """Stored-claim match, then a transcript search. Each point is checked, not offered as a menu."""
    if confirmed_video:
        stored = _ranked(focus, claims, ticker=ticker)
        in_video = [item for item in stored if item[2] == confirmed_video]
        if in_video:
            return LineChoice([item[1].text for item in in_video], confirmed_video)
        spoken = _transcript_hits(focus, posts, confirmed_video, ticker)
        if spoken:
            return LineChoice([item.text for item in spoken], confirmed_video)
        lexical = _lexical_hits(focus, claims, posts, confirmed_video, ticker)
        if lexical:
            return LineChoice([item.text for item in lexical], confirmed_video)
        return LineChoice([], question=NOT_IN_STORED)
    other_fact = _figure_on_another_fact(focus, claims, posts, ticker)
    if other_fact:
        return LineChoice([], question=other_fact)
    if len(_parts(focus)) == 1:
        two_posts = _two_posts_question(focus, claims, posts, ticker)
        if two_posts:
            return LineChoice([], question=two_posts)
    picks = _lines_to_check(focus, claims, posts, ticker)
    if not picks:
        return LineChoice([], question=NOT_IN_STORED)
    return LineChoice([text for text, _video_id in picks], picks[0][1])


def _lines_to_check(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    ticker: str | None,
) -> list[tuple[str, str]]:
    """The best line for each point in the prompt. A 33% point and a $2,000 point stay separate."""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for part in _parts(focus):
        hit = _best_hit(part, claims, posts, ticker)
        if hit is None:
            continue
        key = re.sub(r"\s+", " ", hit[0]).casefold()
        if key in seen:
            continue
        seen.add(key)
        found.append(hit)
    return found[:8]


def _states_figure(focus: str) -> bool:
    """The prompt names a number. A different number is not checked until the user agrees."""
    return any(parse_fields(part).values for part in _parts(focus))


def _best_hit(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    ticker: str | None,
) -> tuple[str, str] | None:
    stored = _ranked(focus, claims, ticker=ticker)
    spoken = _transcript_hits(focus, posts, None, ticker)
    if not stored and not spoken and _states_figure(focus):
        # "said 22%" still means the measure named earlier. Use a line that has 22% and names no other measure
        # when nothing in the store repeats the measure ("22% is massive").
        stored = _ranked(focus, claims, ticker=ticker, metric_required=False)
        spoken = _transcript_hits(focus, posts, None, ticker, metric_required=False)
    if stored:
        # A stored claim wins ties, but not over a spoken line that agrees with the prompt more closely
        # (the catalog may not hold every sentence: "free cash flow is going down because ...").
        if spoken and agreement(focus, spoken[0].text, ticker) > agreement(focus, stored[0][1].text, ticker):
            return spoken[0].text, spoken[0].video_id
        return stored[0][1].text, stored[0][2]
    if not spoken and not _states_figure(focus):
        spoken = _transcript_hits(focus, posts, None, ticker, strict_figure=False)
    if spoken:
        return spoken[0].text, spoken[0].video_id
    lexical = _lexical_hits(focus, claims, posts, None, ticker)
    if lexical:
        return lexical[0].text, lexical[0].video_id
    return None


_SEGMENT = re.compile(
    r"\b(?:azure|aws|cloud|hardware|data\s*cent(?:er|re)|segment|advertising|youtube|instagram)\b",
    flags=re.IGNORECASE,
)


def _figure_on_another_fact(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    ticker: str | None,
) -> str | None:
    """The typed figure sits on a different fact from the named metric. Ask which line."""
    prompt = parse_fields(focus, company=ticker)
    if not prompt.metrics or not prompt.values:
        return None
    strict = _candidate_lines(focus, claims, posts, ticker, strict=True)
    if not strict:
        return None
    figure_text = strict[0][0]
    figure_fields = parse_fields(figure_text, company=ticker)
    metric_ok = bool(figure_fields.metrics) and prompt.metrics <= figure_fields.metrics
    segment_surprise = bool(_SEGMENT.search(figure_text)) and not _SEGMENT.search(focus)
    if metric_ok and not segment_surprise:
        return None
    if not segment_surprise and not (figure_fields.metrics and not metric_ok):
        return None
    alternate = _company_metric_line(focus, claims, posts, ticker, prompt, figure_text)
    if alternate is None:
        return None
    return (
        "Which line did you mean?\n"
        f"1. {alternate}\n"
        f"2. {figure_text}\n"
        "Resubmit with the line you meant."
    )


_METRIC_WORDS = {
    "revenue": frozenset({"revenue", "revenues", "sales", "topline"}),
    "profit": frozenset({"profit", "profits", "earnings", "income"}),
    "margin": frozenset({"margin", "margins"}),
    "cash": frozenset({"cash", "fcf"}),
    "debt": frozenset({"debt"}),
}
_WORD = re.compile(r"\d+(?:\.\d+)?%?|\$\d+(?:\.\d+)?|[a-z]+")
_BILLION = re.compile(r"\bbillion\b|\$\s*\d", flags=re.IGNORECASE)


def _value_near_metric(text: str, metrics: frozenset[str], units: set[str]) -> bool:
    """The named measure sits next to a figure of the same unit. A margin percent is not revenue."""
    tokens = _WORD.findall(text.lower())
    wanted = {word for metric in metrics for word in _METRIC_WORDS.get(metric, frozenset({metric}))}
    metric_at = [index for index, token in enumerate(tokens) if token in wanted]
    number_at: list[int] = []
    for index, token in enumerate(tokens):
        if "percent" in units and (token.endswith("%") or token == "percent"):
            number_at.append(index)
        if "billion" in units and token in {"billion", "bn"}:
            number_at.append(index)
        if "dollar" in units and token.startswith("$"):
            number_at.append(index)
    return any(abs(left - right) <= 12 for left in metric_at for right in number_at)


def _company_metric_line(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    ticker: str | None,
    prompt: ClaimFields,
    figure_text: str,
) -> str | None:
    figure_key = re.sub(r"\s+", " ", figure_text).casefold()
    units = {unit for _number, unit in prompt.values}
    passing: list[str] = []
    for text, _video_id in _candidate_lines(focus, claims, posts, ticker, strict=False):
        fields = parse_fields(text, company=ticker)
        if not fields.metrics or not prompt.metrics <= fields.metrics:
            continue
        if re.sub(r"\s+", " ", text).casefold() == figure_key:
            continue
        if _SEGMENT.search(text):
            continue
        if not _value_near_metric(text, prompt.metrics, units):
            continue
        passing.append(text)
    if not passing:
        return None
    passing = _same_phrase(focus, passing)
    # A revenue percent belongs on the company total, not on a smaller product line.
    if "percent" in units and "revenue" in prompt.metrics:
        totals = [text for text in passing if _BILLION.search(text)]
        if totals:
            return totals[0]
    return passing[0]


_PHRASES = (
    ("operating margin", r"operating\s+margins?"),
    ("gross margin", r"gross\s+margins?"),
    ("net margin", r"net\s+(?:income\s+)?margins?"),
    ("free cash", r"free\s+cash"),
)


def _same_phrase(focus: str, lines: list[str]) -> list[str]:
    """Keep lines that use the prompt's phrase, such as operating margin rather than gross margin."""
    lowered = focus.lower()
    for needle, pattern in _PHRASES:
        if needle not in lowered:
            continue
        matched = [text for text in lines if re.search(pattern, text, flags=re.IGNORECASE)]
        if matched:
            return matched
    return lines


def _two_posts_question(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    ticker: str | None,
) -> str | None:
    """Two videos state the same claim and the user named neither video."""
    stored = _ranked(focus, claims, ticker=ticker, strict_value=True)
    ranked = [(score, claim.text, video_id) for score, claim, video_id in stored]
    if not ranked:
        spoken = _transcript_hits(focus, posts, None, ticker, strict_figure=True)
        ranked = [(hit.score, hit.text, hit.video_id) for hit in spoken]
    if len(ranked) < 2 or ranked[0][0] <= 0:
        return None
    shown: list[tuple[str, str]] = []
    seen: set[str] = set()
    for _score, text, video_id in ranked:
        if video_id in seen or not _names_ticker(text, ticker):
            continue
        seen.add(video_id)
        shown.append((text, video_id))
        if len(shown) == 2:
            break
    if len(shown) < 2:
        return None
    lines = []
    for index, (text, video_id) in enumerate(shown, start=1):
        row = next((post for post in posts if str(post.get("video_id") or "") == video_id), {})
        title = row.get("title") or video_id
        when = row.get("upload_date") or ""
        lines.append(f"{index}. {text} — {title} ({video_id}, {when})")
    head = "Two stored posts state this. Which one should be checked?"
    return head + "\n" + "\n".join(lines) + "\nResubmit with the video id."


def _candidate_lines(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    ticker: str | None,
    *,
    strict: bool,
) -> list[tuple[str, str]]:
    stored = _ranked(focus, claims, ticker=ticker, strict_value=strict)
    if stored:
        return [(claim.text, video_id) for _score, claim, video_id in stored]
    spoken = _transcript_hits(focus, posts, None, ticker, strict_figure=strict)
    return [(hit.text, hit.video_id) for hit in spoken]


def _names_ticker(text: str, ticker: str | None) -> bool:
    """The sentence names this company. A carried label on another firm's line does not count."""
    if not ticker:
        return True
    return any(found == ticker for found, _surface, _start in find_mentions(text))


def shares_figure(user_text: str, other_text: str) -> bool:
    """A stated 24% matches a row that contains 24%, not a row that contains 17%."""
    user_pct, user_mag = _figures(user_text)
    other_pct, other_mag = _figures(other_text)
    if not user_pct and not user_mag:
        return True
    percent_hit = any(_close(left, right, True) for left in user_pct for right in other_pct)
    magnitude_hit = any(_close(left, right, False) for left in user_mag for right in other_mag)
    return percent_hit or magnitude_hit


def semantic_score(left: str, right: str) -> float:
    """Cosine similarity of local semantic embeddings. No separate vector store."""
    return _cosine(_embed(left), _embed(right))


def _ranked(
    focus: str,
    rows: list[tuple[SubClaim, str]],
    *,
    ticker: str | None = None,
    strict_value: bool = True,
    metric_required: bool = True,
) -> list[tuple[float, SubClaim, str]]:
    prompts = [parse_fields(part, company=ticker) for part in _parts(focus)]
    matched: list[tuple[SubClaim, str]] = []
    for claim, video_id in rows:
        candidate = parse_fields(claim.text, company=claim.company_ticker or ticker)
        if not candidate.metrics and claim.parent_span and re.search(
            r"\ball three\b|\ball of them\b", claim.text, flags=re.IGNORECASE,
        ):
            inherited = parse_fields(claim.parent_span).metrics
            if inherited:
                candidate = ClaimFields(
                    candidate.company, inherited, candidate.direction, candidate.values, candidate.period,
                )
        if any(
            fields_match(prompt, candidate, strict_value=strict_value, metric_required=metric_required)
            for prompt in prompts
        ):
            matched.append((claim, video_id))
    # A stated figure keeps every line that has it. BM25 only orders those lines.
    stated_value = any(prompt.values for prompt in prompts)
    return _order_by_bm25(focus, matched, cluster=not stated_value)


_BM25_FLOOR = 1.5


def _lexical_hits(
    focus: str,
    claims: list[tuple[SubClaim, str]],
    posts: list[dict],
    video_id: str | None,
    ticker: str | None,
) -> list["_Hit"]:
    """Company sentences ranked by BM25 when the prompt states no metric, value, or direction."""
    prompt = parse_fields(focus, company=ticker)
    if prompt.metrics or prompt.values or prompt.direction:
        return []
    company = prompt.company or ticker
    if not company:
        return []
    passed: list[tuple[str, str, str]] = []
    for claim, vid in claims:
        if video_id and vid != video_id:
            continue
        candidate = parse_fields(claim.text, company=claim.company_ticker or ticker)
        if candidate.company == company:
            passed.append((claim.text, vid, ""))
    for post in posts:
        vid = str(post.get("video_id") or "")
        if video_id and vid != video_id:
            continue
        when = str(post.get("upload_date") or post.get("published_at") or "")
        sentences = _sentences(str(post.get("transcript") or ""))
        companies = company_for_sentences(sentences)
        for sentence, named in zip(sentences, companies):
            if named == company:
                passed.append((sentence, vid, when))
    if not passed:
        return []
    scores = bm25_scores(focus, [sentence for sentence, _vid, _when in passed])
    found = [
        _Hit(score, sentence, vid, when)
        for score, (sentence, vid, when) in sorted(zip(scores, passed), key=lambda item: item[0], reverse=True)
        if score >= _BM25_FLOOR
    ]
    if not found:
        return []
    best = found[0].score
    kept = [item for item in found if _near_best(best, item.score)]
    unique: list[_Hit] = []
    seen: set[str] = set()
    for item in kept:
        key = re.sub(r"\s+", " ", item.text).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:8]


def agreement(focus: str, text: str, ticker: str | None = None) -> float:
    """How closely a line's fields agree with the prompt, beyond passing fields_match.

    Same direction stated (+2), same period stated (+1), each figure in common (+1), and -0.5 for each measure
    the line is about that the prompt is not. Word overlap (BM25) only breaks ties: a line that repeats
    "free cash flow" but is about margins must not beat "free cash flow is going down" for "went down"."""
    candidate = parse_fields(text, company=ticker)
    best = None
    for part in _parts(focus) or [focus]:
        prompt = parse_fields(part, company=ticker)
        score = 0.0
        if prompt.direction and candidate.direction == prompt.direction:
            score += 2
        if prompt.period and candidate.period and prompt.period.lower() == candidate.period.lower():
            score += 1
        score += sum(1 for value in prompt.values if value in candidate.values)
        if prompt.metrics:
            score -= 0.5 * len(candidate.metrics - prompt.metrics)
        best = score if best is None else max(best, score)
    return best or 0.0


def _order_by_bm25(
    focus: str, matched: list[tuple[SubClaim, str]], *, cluster: bool = True,
) -> list[tuple[float, SubClaim, str]]:
    if not matched:
        return []
    scores = bm25_scores(focus, [claim.text for claim, _video_id in matched])
    agrees = [agreement(focus, claim.text, claim.company_ticker) for claim, _video_id in matched]
    scored = sorted(zip(agrees, scores, matched), key=lambda item: (item[0], item[1]), reverse=True)
    top_agree, best = scored[0][0], scored[0][1]
    kept = [item for item in scored if not cluster or (item[0] == top_agree and _near_best(best, item[1]))]
    return [(score, claim, video_id) for _agree, score, (claim, video_id) in kept[:8]]


def _near_best(best: float, score: float) -> bool:
    if best <= 0:
        return True
    return score >= best * 0.72


def _transcript_hits(
    focus: str,
    posts: list[dict],
    video_id: str | None,
    ticker: str | None,
    *,
    strict_figure: bool = True,
    metric_required: bool = True,
) -> list["_Hit"]:
    prompts = [parse_fields(part, company=ticker) for part in _parts(focus)]
    passed: list[tuple[str, str, str]] = []
    for post in posts:
        vid = str(post.get("video_id") or "")
        if video_id and vid != video_id:
            continue
        when = str(post.get("upload_date") or post.get("published_at") or "")
        sentences = _sentences(str(post.get("transcript") or ""))
        companies = company_for_sentences(sentences)
        for index, sentence in enumerate(sentences):
            company = companies[index]
            if ticker and company != ticker:
                continue
            candidate = parse_fields(sentence, company=company)
            if any(
                fields_match(prompt, candidate, strict_value=strict_figure, metric_required=metric_required)
                for prompt in prompts
            ):
                passed.append((sentence, vid, when))
    if not passed:
        return []
    scores = bm25_scores(focus, [sentence for sentence, _vid, _when in passed])
    agrees = [agreement(focus, sentence, ticker) for sentence, _vid, _when in passed]
    ranked = sorted(zip(agrees, scores, passed), key=lambda item: (item[0], item[1]), reverse=True)
    found = [_Hit(score, sentence, vid, when) for _agree, score, (sentence, vid, when) in ranked]
    top_agree, best = ranked[0][0], found[0].score
    stated_value = any(prompt.values for prompt in prompts)
    kept = found if stated_value else [item for (agree, _s, _p), item in zip(ranked, found)
                                       if agree == top_agree and _near_best(best, item.score)]
    unique: list[_Hit] = []
    seen: set[str] = set()
    for item in kept:
        key = re.sub(r"\s+", " ", item.text).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:8]


def _about_ticker(sentences: list[str], ticker: str | None) -> list[bool]:
    """A company stays in force until another company is named.

    "Shares of Meta are down" makes the following sentences about Meta, including
    "a company that is investing leading to higher revenues, higher operating
    profits, and bigger operating margins." "Getting back to Amazon" ends that stretch.
    """
    if not ticker:
        return [True] * len(sentences)
    current: set[str] = set()
    about: list[bool] = []
    for sentence in sentences:
        mentioned = _mentioned(sentence)
        if mentioned:
            current = mentioned
        about.append(ticker in current)
    return about


def _mentioned(sentence: str) -> set[str]:
    return {found for found, _surface, _start in find_mentions(sentence)}


def _eligible_score(focus: str, text: str, *, strict_figure: bool = True) -> float:
    if not _same_measure(focus, text) or not _same_direction(focus, text):
        return 0.0
    if strict_figure and not shares_figure(focus, text):
        return 0.0
    if not strict_figure and not shares_figure(focus, text):
        # A misremembered percentage can still be the revenue line, not a different measure.
        if not (_features(focus) & _MEASURES & _features(text)):
            return 0.0
    return max(semantic_score(part, text) for part in _parts(focus))


def _parts(focus: str) -> list[str]:
    """Split a prompt that states two figured claims, such as 33% and $2,000.

    "Revenue, operating profit and operating margin all went up" stays one claim.
    A later figure keeps the measure from the rest of the prompt, so "grew 20% but
    said 22%" is scored as revenue growth at 22% as well as at 20%.
    """
    pieces = re.split(r"\b(?:and|but)\b", focus, flags=re.IGNORECASE)
    inherited = " ".join(sorted(_features(focus) & _MEASURES))
    figured = []
    for piece in pieces:
        piece = piece.strip(" ,.")
        percents, magnitudes = _figures(piece)
        if not percents and not magnitudes and not _FOLD.search(piece):
            continue
        if not (_features(piece) & _MEASURES) and inherited:
            piece = f"{inherited} {piece}"
        figured.append(piece)
    if len(figured) >= 2:
        return figured
    return [focus]


def _same_measure(focus: str, text: str) -> bool:
    """A margin sentence does not match a revenue claim. A sentence with no measure can."""
    wanted = _features(focus) & _MEASURES
    found = _features(text) & _MEASURES
    if not wanted or not found:
        return True
    return bool(wanted & found)


def _same_direction(focus: str, text: str) -> bool:
    wanted = _features(focus) & _DIRECTIONS
    found = _features(text) & _DIRECTIONS
    if wanted and found and not (wanted & found):
        return False
    return True


def _figures(text: str) -> tuple[list[float], list[float]]:
    percents = [float(match) for match in _PERCENT.findall(text or "")]
    magnitudes: list[float] = []
    for match in _MAGNITUDE.finditer(text or ""):
        value = float(match.group(1).replace(",", ""))
        if match.group(2).lower().startswith("mill"):
            value = value / 1000.0
        magnitudes.append(value)
    for match in _DOLLARS.finditer(text or ""):
        value = float(match.group(1).replace(",", ""))
        if value >= 100:
            magnitudes.append(value)
    return percents, magnitudes


def _close(left: float, right: float, percent: bool) -> bool:
    difference = abs(left - right)
    scale = max(left, right)
    if percent:
        # 24 matches 24.2, and 100 matches 107. 24 does not match 23 or 17.
        allowance = 0.5 if scale < 50 else max(2.0, 0.10 * scale)
    else:
        allowance = max(0.6, 0.08 * scale)
    return difference <= allowance


def _embed(text: str) -> list[float]:
    vector = [0.0] * _DIM
    tokens: list[str] = []
    for token in _TOKEN.findall((text or "").lower()):
        word = token.strip("'")
        if any(character.isdigit() for character in word):
            continue
        canon = _canon(word)
        if canon:
            tokens.append(canon)
    percents, magnitudes = _figures(text)
    for percent in percents:
        tokens.append(f"pct:{round(percent)}")
    for magnitude in magnitudes:
        tokens.append(f"mag:{round(magnitude)}")
    for token in tokens:
        _hash_into(vector, token, 1.0)
    for first, second in zip(tokens, tokens[1:]):
        _hash_into(vector, f"{first}_{second}", 0.6)
    return vector


def _hash_into(vector: list[float], key: str, weight: float) -> None:
    digest = hashlib.md5(key.encode(), usedforsecurity=False).digest()
    index = int.from_bytes(digest[:2], "little") % _DIM
    sign = 1.0 if digest[2] % 2 == 0 else -1.0
    vector[index] += sign * weight


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class _Hit:
    score: float
    text: str
    video_id: str
    when: str


def _as_hits(scored: list[tuple[float, SubClaim, str]], posts: list[dict]) -> list[_Hit]:
    when = {str(post.get("video_id") or ""): str(post.get("upload_date") or "") for post in posts}
    return [_Hit(score, claim.text, video_id, when.get(video_id, "")) for score, claim, video_id in scored]


def _quote(text: str) -> str:
    quote = re.sub(r"\s+", " ", text).strip()
    if len(quote) > 180:
        quote = quote[:177].rstrip() + "..."
    return quote


def _one_line(hit: _Hit) -> str:
    when = f"{hit.when}  " if hit.when else ""
    return (
        "Is this the line?\n"
        f"{when}{hit.video_id}  \"{_quote(hit.text)}\"\n"
        "Resubmit with that video id to check it."
    )


def _which_line(hits: list[_Hit]) -> str:
    lines = []
    for hit in hits[:8]:
        when = f"{hit.when}  " if hit.when else ""
        lines.append(f"{when}{hit.video_id}  \"{_quote(hit.text)}\"")
    return "Which line should FinSight check?\n" + "\n".join(lines) + "\nResubmit with that video id."


def _why(sentence: str) -> str:
    reason = _drop_reason(sentence, None)
    if reason in _WHY:
        return _WHY[reason]
    if _PRICE.search(sentence):
        return "it is about a price"
    if _ADVICE.search(sentence):
        return "it is advice"
    return "it is not in the stored claim list"
