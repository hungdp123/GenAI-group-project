"""Heuristic publication-date estimates for quarter-labelled IR documents.

Estimates remain separate from verified publication dates, but may be used as
explicitly marked effective dates for temporal filtering under project policy.
"""

from __future__ import annotations

import calendar
import re
from datetime import date
from typing import Any


# Fiscal year-end month. Week-based fiscal calendars still use the actual SEC
# period end date below; this map is only used to assign a fiscal-year label.
FISCAL_YEAR_END_MONTH = {
    "AAPL": 9, "AMD": 12, "AMC": 12, "AMZN": 12, "BIRD": 12,
    "CRM": 1, "DNUT": 12, "GME": 1, "GOOGL": 12, "GPRO": 12,
    "INTC": 12, "INTU": 7, "KSS": 1, "MCD": 12, "MELI": 12,
    "META": 12, "MSFT": 6, "NFLX": 12, "NKE": 5, "NVDA": 1,
    "OPEN": 12, "ORCL": 5, "PLTR": 12, "SBUX": 9, "TSLA": 12,
}


def build_quarter_period_lookup(core_manifest: dict[str, Any]) -> dict[tuple[str, int, int], date]:
    """Map (ticker, fiscal year, quarter) to actual SEC report period end."""
    by_ticker: dict[str, list[tuple[date, str]]] = {}
    for doc in core_manifest.get("documents", []):
        form = str(doc.get("form") or doc.get("document_type") or "").upper()
        if form not in {"10-Q", "10-K"}:
            continue
        try:
            period = date.fromisoformat(str(doc["period"])[:10])
        except (KeyError, ValueError):
            continue
        by_ticker.setdefault(str(doc["company_ticker"]).upper(), []).append((period, form))

    lookup: dict[tuple[str, int, int], date] = {}
    for ticker, periods in by_ticker.items():
        end_month = FISCAL_YEAR_END_MONTH.get(ticker)
        if end_month is None:
            continue
        grouped: dict[int, dict[str, list[date]]] = {}
        for period, form in set(periods):
            fiscal_year = period.year if period.month <= end_month else period.year + 1
            grouped.setdefault(fiscal_year, {"10-Q": [], "10-K": []})[form].append(period)
        for fiscal_year, forms in grouped.items():
            for quarter, period in enumerate(sorted(set(forms["10-Q"])), start=1):
                if quarter <= 3:
                    lookup[(ticker, fiscal_year, quarter)] = period
            annuals = sorted(set(forms["10-K"]))
            if annuals:
                lookup[(ticker, fiscal_year, 4)] = annuals[-1]
    return lookup


def _quarter_year(text: str) -> tuple[int, int] | None:
    value = text.upper().replace("%27", "'")
    value = re.sub(r"Q([1-4])[_\-\s]*27(20\d{2}|\d{2})(?!\d)", r"Q\1 FY\2", value)
    # Do not prefix a year that is already inside an FY label (e.g. FY2021 Q3).
    value = re.sub(r"(?<![A-Z0-9])(20\d{2})[_\-\s]*Q([1-4])", r"FY\1 Q\2", value)
    patterns = [
        r"(?:^|[^A-Z])Q([1-4])[_\-\s]*(?:FY|FISCAL\s*YEAR)?[_\-\s]*['’]?((?:20)?\d{2})(?!\d)",
        r"(?:^|[^A-Z])([1-4])Q[_\-\s]*['’]?((?:20)?\d{2})(?!\d)",
        r"(?:^|[^A-Z])FY[_\-\s]*['’]?((?:20)?\d{2})[_\-\s]*Q([1-4])(?!\d)",
    ]
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, value)
        if not match:
            continue
        if index == 2:
            year_text, quarter_text = match.groups()
        else:
            quarter_text, year_text = match.groups()
        year = int(year_text)
        if year < 100:
            year += 2000
        return int(quarter_text), year
    return None


def estimate_publication_date(
    ticker: str, label: str, local_path: str,
    period_lookup: dict[tuple[str, int, int], date],
) -> tuple[str | None, str | None]:
    """Estimate next-calendar-month-end for a clearly quarter-labelled record."""
    parsed = _quarter_year(f"{label} {local_path}")
    if parsed is None:
        return None, None
    quarter, fiscal_year = parsed
    period_end = period_lookup.get((ticker.upper(), fiscal_year, quarter))
    if period_end is None:
        return None, None
    year, month = period_end.year, period_end.month + 1
    if month == 13:
        year, month = year + 1, 1
    estimate = date(year, month, calendar.monthrange(year, month)[1])
    basis = (
        f"Heuristic: month-end after SEC {ticker.upper()} FY{fiscal_year} Q{quarter} "
        f"period end ({period_end.isoformat()}); verify against official IR publication date."
    )
    return estimate.isoformat(), basis
