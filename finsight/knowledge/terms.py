"""Shared word lists: what a sentence must NOT be for the financial agent to calculate it.

One list, used by claim extraction (the type, so the route), the financial agent's reader (what to compute) and
the Auditor (which metric the sentence is about), so the three cannot drift apart. Rule book sections 5 and 11.

    FORECAST          forward-looking wording -> continuation, never a past figure
    PRICE_MOVE        a share-price or return move ("Google is up 5% on the day") -> not a company figure
    NON_GAAP          EBITDA, adjusted / non-GAAP figures -> not in section 11
    MARGIN_OTHER      "margin" that is not a profit margin (margin buyers, margin of safety, by a wide margin)
    PERSONAL_GAIN     the influencer's own investment gain ("profits of $40,500 in my portfolio")
"""

from __future__ import annotations

import re

_I = re.IGNORECASE

# Forward-looking words. "Could", "might", "hoping" and "pricing in" were missing from extraction before 28 Oct.
FORECAST_WORDS = (
    r"\b(will|would|could|might|may be|going to|gonna|set to|expected to|expects?|expecting|forecast\w*|"
    r"projected|projects?|guid(?:ance|ed|es|ing)|outlook|next (?:year|quarter)|this coming|"
    r"by the end of (?:this|next) year|in (?:five|ten|\d+) years|potentially|hoping|hope to|pricing in|"
    r"priced in|forward (?:earnings|estimates?|p/?e)|just getting started|going forward|long-term target|"
    r"i think (?:they|it|the company|\w+)(?:'ll| will)|they'll|it'll|"
    # expectations, estimates, wishes and the speaker's own model assumptions are not reported figures
    r"(?:is|are|was|were) expected|expected (?:at|on|in|for)|expectations?|estimates?|consensus|"
    r"i want to see|i'd like to see|i look at|i'm (?:modeling|modelling|assuming|projecting)|assuming|"
    r"could have|would have)\b"
)
FORECAST = re.compile(FORECAST_WORDS, _I)

# A share-price or return move. "X is up 45%", "down 3.6% on the day", "up 251% year-to-date", "a 12% decline".
PRICE_MOVE_WORDS = (
    r"\b(stocks?|shares?|share price|stock price|market cap\w*|valuation|price target|ticker|"
    r"on the day|this week|in (?:the past )?\d+ days|year[- ]to[- ]date|ytd|since (?:the start|mid|early|late|"
    r"its|the)\b|from its \w+ (?:\d{4} )?(?:peak|high|low|surge)|all[- ]time high|one[- ]week high|"
    r"(?:\d+(?:\.\d+)?%|\d+ ?percent) (?:return|gain|loss)|returns?\b|portfolio|outperform\w*|underperform\w*|"
    r"(?:free cash flow|fcf|earnings|dividend) yield)"
    # "Google is up 45%", "Google's up 6.45%", "Intuit now down 11%", "lost almost 11%": a move with no measure word
    r"|(?:\b(?:is|are|was|were|be|been)|'s)\s+(?:\w+\s+){0,2}?(?:up|down)\s+(?:about |around |almost |over |nearly )?\d"
    r"|\b(?:now|only)\s+(?:up|down)\s+\d|\b(?:lost|dropped|plunged|rallied|soared|tumbled|sank|jumped)\s+"
    r"(?:about |around |almost |over |nearly |roughly )?\d+(?:\.\d+)?\s*%"
)
PRICE_MOVE = re.compile(PRICE_MOVE_WORDS, _I)

# A yield is a valuation ratio against the share price, not the company's figure ("a 7% free cash flow yield").
YIELD_WORDS = r"(?:free cash flow|fcf|earnings|dividend|cash flow) yields?"

NON_GAAP_WORDS = r"\b(ebitda(?: margins?)?|ebit\b|adjusted (?:\w+ )?(?:earnings|eps|income|profits?|margins?)|non-gaap)"
NON_GAAP = re.compile(NON_GAAP_WORDS, _I)

# "Margin" that is not a profit margin. Masked before any margin word is read as a measure.
MARGIN_OTHER_WORDS = (
    r"\b(margin (?:buyers?|calls?|debt|account|loans?|trading|of safety|of error|for error)|on margin|"
    r"buying on margin|by a (?:\w+ )?margin|(?:wide|wider|narrow|considerable|large|huge|big|slim) margin|"
    r"margin invested|margin loans?)\b"
)
MARGIN_OTHER = re.compile(MARGIN_OTHER_WORDS, _I)

PERSONAL_GAIN = re.compile(
    r"\b(my|i own|i'm up|i am up|portfolio|position|stake)\b.{0,80}\bprofits? (?:of|exceeding|around)\b"
    r"|\bprofits? of \$[\d,]+|\bunrealized (?:gains?|profits?)\b",
    _I,
)


def is_forecast(text: str) -> bool:
    return bool(FORECAST.search(text or ""))


def is_price_move(text: str) -> bool:
    return bool(PRICE_MOVE.search(text or ""))


def mask_other_margins(text: str) -> str:
    """Blank out 'margin' used in another sense, so it is not read as a profit margin."""
    return MARGIN_OTHER.sub(lambda m: " " * len(m.group(0)), text or "")
