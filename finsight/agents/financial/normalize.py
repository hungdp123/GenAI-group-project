""" 
Turns raw SEC Company Facts / submissions JSON into flat rows 
"""

from datetime import date

from . import config
from .concepts import TAG_TO_CONCEPT, kind, unit

_CLASSES = [("Q", 80, 100), ("H1", 170, 190), ("9M", 260, 285), ("FY", 350, 380)]


def period_class(start, end):
    """I (instant), Q, H1, 9M or FY by length in days; None for any other length."""
    if not start:
        return "I"
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    for name, lo, hi in _CLASSES:
        if lo <= days <= hi:
            return name
    return None


def filing_url(cik, accn):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{accn}-index.htm"


def normalize_companyfacts(raw: dict, ticker: str, stats: dict | None = None) -> list[dict]:
    """One row per usable fact. `stats`, if given, counts what was dropped and why."""
    stats = stats if stats is not None else {}
    scale = getattr(config, "VALUE_SCALE", {})          # (ticker, concept) -> multiplier, e.g. shares filed in millions
    cik, rows = int(raw["cik"]), []
    for tag, body in raw.get("facts", {}).get("us-gaap", {}).items():
        concept = TAG_TO_CONCEPT.get(tag)
        if not concept:
            continue
        for f in body.get("units", {}).get(unit(concept), []):
            try:
                end, filed = date.fromisoformat(f["end"]), date.fromisoformat(f["filed"])
                cls = period_class(f.get("start"), f["end"])
                value = float(f["val"])
            except (KeyError, TypeError, ValueError):
                stats["malformed"] = stats.get("malformed", 0) + 1
                continue
            if f.get("form") not in config.FORMS or end < config.MIN_PERIOD_END:
                continue
            if filed < end:                              # a period cannot be reported before it ends
                stats["filed_before_period_end"] = stats.get("filed_before_period_end", 0) + 1
                continue
            if cls is None:
                stats["unusual_period_length"] = stats.get("unusual_period_length", 0) + 1
                continue
            if (cls == "I") != (kind(concept) == "instant"):
                stats["wrong_period_type"] = stats.get("wrong_period_type", 0) + 1
                continue
            rows.append(dict(ticker=ticker, cik=cik, concept=concept, xbrl_tag=tag,
                             value=value * scale.get((ticker, concept), 1), unit=unit(concept),
                             period_start=f.get("start") or "", period_end=f["end"], period_class=cls,
                             form=f["form"], accession=f["accn"], filed=f["filed"],
                             source_url=filing_url(cik, f["accn"])))
    return rows


def normalize_submissions(raw: dict, ticker: str) -> list[dict]:
    r, cik, rows = raw["filings"]["recent"], int(raw["cik"]), []
    for i, form in enumerate(r["form"]):
        if form in ("10-K", "10-Q"):
            accn = r["accessionNumber"][i]
            doc_url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/"
                       f"{r['primaryDocument'][i]}")
            rows.append(dict(ticker=ticker, cik=cik, form=form, accession=accn, filed=r["filingDate"][i],
                             report_date=r["reportDate"][i], source_tier=1, url=doc_url))
    return rows
