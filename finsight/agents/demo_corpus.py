"""PLACEHOLDER frozen corpus shared by the stubs (demo company "Company X", cut-off 31 Jul 2025).

Owner: Person 5 (Hung) for wiring; real content comes from Person 2's vector DB and
Person 1's XBRL tools. Nothing outside the stubs may import this module.
"""

from __future__ import annotations

import json
import re
from datetime import date
from importlib import resources
from typing import Any

from finsight.config import CORPUS_COMPANIES
from finsight.contracts.schemas import SourceTier, Stance

# Each entry: an evidence chunk plus the sub-claim topics it bears on, with a stance per topic.
NARRATIVE_CHUNKS: list[dict[str, Any]] = [
    {
        "evidence_id": "CMPX-10Q-2025Q2-MDA-DC",
        "content": "Data Center segment revenue increased 31% year over year, primarily driven by "
        "customer demand for AI infrastructure (accelerators and networking).",
        "source_tier": SourceTier.TIER_1, "doc_type": "10-Q", "period": "Q2 FY2025",
        "publication_date": date(2025, 7, 25),
        "url": "https://example.com/sec/cmpx/10-Q/2025-Q2",
        "citation_span": "Item 2. MD&A - Segment Results - Data Center",
        "topics": {"demand": Stance.SUPPORTING, "growth": Stance.SUPPORTING},
    },
    {
        "evidence_id": "CMPX-10Q-2025Q2-MDA-SEG",
        "content": "Client segment revenue grew 6% and Embedded segment revenue grew 4%, "
        "reflecting PC refresh and industrial recovery rather than AI infrastructure.",
        "source_tier": SourceTier.TIER_1, "doc_type": "10-Q", "period": "Q2 FY2025",
        "publication_date": date(2025, 7, 25),
        "url": "https://example.com/sec/cmpx/10-Q/2025-Q2",
        "citation_span": "Item 2. MD&A - Segment Results - Client and Embedded",
        "topics": {"demand": Stance.CHALLENGING},
    },
    {
        "evidence_id": "CMPX-CALL-2025Q2-OUTLOOK",
        "content": "On the Q2 earnings call, management said it 'expects continued strong demand' "
        "for AI infrastructure products through the next fiscal year.",
        "source_tier": SourceTier.TIER_2, "doc_type": "ir_remarks", "period": "Q2 FY2025",
        "publication_date": date(2025, 7, 25),
        "url": "https://example.com/ir/cmpx/2025-Q2-call",
        "citation_span": "Prepared remarks, CEO, paragraph 4",
        "topics": {"continuation": Stance.SUPPORTING},
    },
    {
        "evidence_id": "CMPX-ER-2025Q2-MARGIN",
        "content": "The Q2 earnings release reported gross margin of 52.1%, down from 54.3% a year "
        "earlier, and guided that capital expenditure will remain elevated.",
        "source_tier": SourceTier.TIER_2, "doc_type": "earnings_release", "period": "Q2 FY2025",
        "publication_date": date(2025, 7, 25),
        "url": "https://example.com/ir/cmpx/2025-Q2-release",
        "citation_span": "Financial highlights table; Outlook section",
        "topics": {"continuation": Stance.CHALLENGING, "risk": Stance.CHALLENGING},
    },
    {
        "evidence_id": "NEWS-2025-07-28-WINNER",
        "content": "Commentators called Company X 'the biggest winner' of the AI infrastructure boom.",
        "source_tier": SourceTier.TIER_3, "doc_type": "news", "period": None,
        "publication_date": date(2025, 7, 28),
        "url": "https://example.com/news/2025-07-28/company-x-biggest-winner",
        "citation_span": "Headline and first paragraph",
        "topics": {"demand": Stance.SUPPORTING, "comparative": Stance.SUPPORTING},
    },
]


def load_xbrl_facts(ticker: str, cutoff_date: date) -> list[dict[str, Any]]:
    """XBRL facts for `ticker` filed on or before the cut-off."""
    raw = json.loads(resources.files("finsight.data").joinpath("xbrl_facts.json").read_text("utf-8"))
    facts = raw.get(ticker, {}).get("facts", [])
    return [f for f in facts if date.fromisoformat(f["filed"]) <= cutoff_date]


def company_name(ticker: str | None) -> str:
    aliases = CORPUS_COMPANIES.get(ticker or "", [])
    return aliases[0] if aliases else (ticker or "UNKNOWN")


def detect_ticker(text: str) -> str | None:
    """First corpus company whose alias appears as a whole word in `text`."""
    for ticker, aliases in CORPUS_COMPANIES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE):
                return ticker
    return None
