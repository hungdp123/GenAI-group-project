""" 
Read-only fact store where tools.py gets the figures
"""

import csv
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from .concepts import CONCEPTS

SPAN_ORDER = ["Q", "H1", "9M", "FY"]
DEFAULT_DATA_DIR = Path("data") / "sec"   


def _iso(d):
    """A cut-off as 'YYYY-MM-DD' or None. Accepts a date, a datetime, or an ISO date string.
    Anything else is an error so that comparing against a badly formatted date will not silently leak or hide filings."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return date.fromisoformat(str(d)[:10]).isoformat()


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class FactStore:
    def __init__(self, facts, filings=()):
        self.filings = [dict(r) for r in filings]
        self.facts, self.stale_rows, self._by_key = [], 0, defaultdict(list)
        for r in (dict(r) for r in facts):
            if r["concept"] not in CONCEPTS or r["xbrl_tag"] not in CONCEPTS[r["concept"]][1]:
                self.stale_rows += 1
                continue
            r["value"], r["cik"] = float(r["value"]), int(r["cik"])
            r["period_start"] = r["period_start"] or None
            self.facts.append(r)
            self._by_key[(r["ticker"], r["concept"])].append(r)
        self.tickers = sorted({r["ticker"] for r in self.facts})

    @classmethod
    def from_dir(cls, path=DEFAULT_DATA_DIR / "processed"):
        path = Path(path)
        return cls(_read_csv(path / "facts.csv"), _read_csv(path / "filings.csv"))

    def visible(self, ticker, concept, cutoff=None):
        """Facts filed on or before the cut-off date (inclusive meaning a filing dated on the cut-off counts)."""
        c = _iso(cutoff)
        return [r for r in self._by_key.get((ticker, concept), ()) if c is None or r["filed"] <= c]

    def fact(self, ticker, concept, cls, cutoff=None, end=None, start=None):
        """Best fact for a period: highest-priority tag then the latest filing visible at the
        cut-off so restatements filed before the cut-off are used and later ones are not.
        Two filings on the same day are ordered by accession number so the result never depends on row order."""
        rows = [r for r in self.visible(ticker, concept, cutoff) if r["period_class"] == cls
                and (end is None or r["period_end"] == end) and (start is None or r["period_start"] == start)]
        if not rows:
            return None
        tags = CONCEPTS[concept][1]
        rows.sort(key=lambda r: (tags.index(r["xbrl_tag"]), r["filed"], r["accession"]))
        best_tag = rows[0]["xbrl_tag"]
        return [r for r in rows if r["xbrl_tag"] == best_tag][-1]

    def ends(self, ticker, concept, cutoff=None, classes=None):
        return sorted({r["period_end"] for r in self.visible(ticker, concept, cutoff)
                       if classes is None or r["period_class"] in classes})

    def filings_in_scope(self, ticker, cutoff=None):
        """Latest 10-K and last two 10-Qs filed on or before the cut-off for sanity checking."""
        c = _iso(cutoff)
        fs = sorted((f for f in self.filings if f["ticker"] == ticker and (c is None or f["filed"] <= c)),
                    key=lambda f: (f["filed"], f["accession"]), reverse=True)
        return [f for f in fs if f["form"] == "10-K"][:1] + [f for f in fs if f["form"] == "10-Q"][:2]
