"""One company, many names.

Tesla, TSLA, Tesla Inc, and Tesla Motors are the same company. The thesaurus maps
each of those to one ticker. A product name maps to the parent. YouTube maps to
Alphabet only when the sentence is about YouTube's business. A web address is not
a company mention.

Post matching uses the company names, the tickers, and the product names. It does
not use a bare "YouTube".
"""

from __future__ import annotations

import re

# Spoken name first. That name is what the report shows.
COMPANY_NAMES: dict[str, list[str]] = {
    "CMPX": ["Company X", "CMPX"],
    "AAPL": ["Apple", "AAPL", "Apple Inc", "Apple, Inc.", "Apple Inc."],
    "GOOGL": [
        "Alphabet", "Google", "GOOGL", "GOOG",
        "Alphabet Inc", "Alphabet, Inc.", "Alphabet Inc.",
        "Google Inc", "Google, Inc.",
    ],
    "MSFT": ["Microsoft", "MSFT", "Microsoft Corporation", "Microsoft Corp", "Microsoft Corp."],
    "NFLX": ["Netflix", "NFLX", "Netflix Inc", "Netflix, Inc.", "Netflix Inc."],
    "NVDA": ["NVIDIA", "Nvidia", "NVDA", "NVIDIA Corporation"],
    "TSLA": ["Tesla", "TSLA", "Tesla Inc", "Tesla, Inc.", "Tesla Inc.", "Tesla Motors"],
    "AMZN": ["Amazon", "AMZN", "Amazon.com", "Amazon.com, Inc.", "Amazon Inc"],
    "META": ["Meta", "META", "Meta Platforms", "Meta Platforms, Inc.", "Facebook", "Facebook, Inc."],
    "MELI": ["MercadoLibre", "Mercado Libre", "MELI"],
    "SBUX": ["Starbucks", "SBUX", "Starbucks Corporation"],
    "PLTR": ["Palantir", "PLTR", "Palantir Technologies"],
    "INTC": ["Intel", "INTC", "Intel Corporation"],
    "INTU": ["Intuit", "INTU", "Intuit Inc", "Intuit Inc."],
    "CRM": ["Salesforce", "CRM", "Salesforce.com", "Salesforce, Inc."],
    "ORCL": ["Oracle", "ORCL", "Oracle Corporation"],
    "AMD": ["AMD", "Advanced Micro Devices"],
}

# A sentence about the product is about the parent. These are not used for a bare platform mention.
PRODUCTS: dict[str, list[str]] = {
    "AAPL": ["iPhone"],
    "GOOGL": ["Waymo", "Google Cloud"],
    "MSFT": ["Azure", "Microsoft Azure"],
    "AMZN": ["AWS", "Amazon Web Services"],
    "META": ["Instagram"],
}

_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_COMPARATOR = re.compile(
    r"\b(than|versus|vs\.?|compared to|compared with|ahead of|behind)\s*$",
    flags=re.IGNORECASE,
)
_YOUTUBE_BUSINESS = re.compile(
    r"\bYouTube\b(?:\s+\w+){0,4}\s+\b(revenue|ad|ads|advertising|subscription|subscriptions|growth|grew|margin|sales)\b"
    r"|\b(revenue|ad|ads|advertising|subscription|subscriptions|growth|margin|sales)\b(?:\s+\w+){0,4}\s+\bYouTube\b",
    flags=re.IGNORECASE,
)


def corpus_aliases() -> dict[str, list[str]]:
    """Ticker to names a post search should match. YouTube is not in this list."""
    table: dict[str, list[str]] = {}
    for ticker, names in COMPANY_NAMES.items():
        table[ticker] = list(names) + list(PRODUCTS.get(ticker, ()))
    return table


def display_name(ticker: str | None) -> str:
    names = COMPANY_NAMES.get(ticker or "", [])
    return names[0] if names else (ticker or "")


def without_urls(text: str) -> str:
    return _URL.sub(" ", text or "")


def _pattern(alias: str) -> re.Pattern[str]:
    # A dot or a slash after the name is a web address, not the company.
    return re.compile(rf"(?<!\w){re.escape(alias)}(?![\w/.])", flags=re.IGNORECASE)


class _Alias:
    def __init__(self, alias: str, ticker: str) -> None:
        self.alias = alias
        self.ticker = ticker
        self.pattern = _pattern(alias)


def _all_names() -> dict[str, list[str]]:
    table = {ticker: list(names) for ticker, names in COMPANY_NAMES.items()}
    for ticker, names in PRODUCTS.items():
        table.setdefault(ticker, []).extend(names)
    return table


_PRODUCT_ALIASES = {alias.lower() for names in PRODUCTS.values() for alias in names}

_ALIAS_INDEX: list[_Alias] = sorted(
    (_Alias(alias, ticker) for ticker, names in _all_names().items() for alias in names),
    key=lambda item: len(item.alias),
    reverse=True,
)


def find_mentions(text: str, *, include_products: bool = True) -> list[tuple[str, str, int]]:
    """(ticker, surface, start) left to right. The longest name wins at each spot.

    Product names are included when `include_products` is set. A bare YouTube is
    never included. YouTube revenue or ads counts as Alphabet.
    """
    cleaned = without_urls(text)
    occupied = [False] * len(cleaned)
    hits: list[tuple[str, str, int]] = []
    for item in _ALIAS_INDEX:
        if not include_products and item.alias.lower() in _PRODUCT_ALIASES:
            continue
        for match in item.pattern.finditer(cleaned):
            start, end = match.span()
            if any(occupied[start:end]):
                continue
            for index in range(start, end):
                occupied[index] = True
            hits.append((item.ticker, match.group(0), start))
    if include_products:
        for match in _YOUTUBE_BUSINESS.finditer(cleaned):
            start = match.start()
            if not occupied[start]:
                hits.append(("GOOGL", "YouTube", start))
    hits.sort(key=lambda hit: hit[2])
    return hits


def has_alias(text: str, alias: str) -> bool:
    return _pattern(alias).search(without_urls(text)) is not None


def mentions_ticker(text: str, ticker: str) -> bool:
    return any(found == ticker for found, _surface, _start in find_mentions(text))


def subject_ticker(sentence: str, mentions: list[tuple[str, str, int]] | None = None) -> str | None:
    """The company the sentence is about. A name after than, versus, or ahead of is not the subject."""
    found = mentions if mentions is not None else find_mentions(sentence)
    for ticker, _surface, start in found:
        window = sentence[max(0, start - 24):start]
        if _COMPARATOR.search(window):
            continue
        # "Like Meta, Amazon is a big spender" is about Amazon. "I like Meta" stays Meta.
        later = any(other > start for _ticker, _surface, other in found)
        if later and re.search(r"\blike\s*$", window, flags=re.IGNORECASE):
            continue
        return ticker
    return None


def resolve_ticker(text: str) -> str | None:
    """The company this text is about: the subject named in the most sentences."""
    counts: dict[str, int] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        mentions = find_mentions(sentence)
        subject = subject_ticker(sentence, mentions)
        if subject:
            counts[subject] = counts.get(subject, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda ticker: counts[ticker])
