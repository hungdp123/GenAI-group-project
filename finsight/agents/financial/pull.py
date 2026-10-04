""" 
Pulls SEC Company Facts data 
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .config import CIK_OVERRIDES, COMPANIES
from .normalize import normalize_companyfacts, normalize_submissions
from .store import DEFAULT_DATA_DIR

URLS = {"companyfacts": "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
        "submissions": "https://data.sec.gov/submissions/CIK{cik:010d}.json"}
TICKER_LIST_URL = "https://www.sec.gov/files/company_tickers.json"


def fetch(url, ua):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def resolve_ciks(ua):
    """ticker -> CIK from SEC's official list; CIK_OVERRIDES when CIK is in the config.py file."""
    listed = {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(fetch(TICKER_LIST_URL, ua)).values()}
    return {t: CIK_OVERRIDES.get(t, listed.get(t.upper())) for t in COMPANIES}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def download_company(ticker, cik, out, ua, manifest):
    """Save both raw files for one company and return them parsed."""
    raw = {}
    for kind, tmpl in URLS.items():
        time.sleep(0.3)   # SEC limit: 10 requests per second
        data = fetch(tmpl.format(cik=cik), ua)
        p = out / "raw" / kind / f"{ticker}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        manifest["files"][p.relative_to(out).as_posix()] = hashlib.sha256(data).hexdigest()
        raw[kind] = json.loads(data)
    return raw


def rebuild_offline(out):
    """Re-normalize the saved raw files after changing concepts.py"""
    facts, filings = [], []
    for path in sorted((out / "raw" / "companyfacts").glob("*.json")):
        ticker = path.stem
        facts += normalize_companyfacts(json.loads(path.read_text(encoding="utf-8")), ticker)
        subs = out / "raw" / "submissions" / path.name
        if subs.exists():
            filings += normalize_submissions(json.loads(subs.read_text(encoding="utf-8")), ticker)
    if not facts:
        sys.exit(f"No raw files under {out / 'raw'}. Run the pull without --offline first.")
    (out / "processed").mkdir(parents=True, exist_ok=True)
    write_csv(out / "processed" / "facts.csv", facts)
    write_csv(out / "processed" / "filings.csv", filings)
    print(f"Rebuilt {len(facts)} facts and {len(filings)} filings from {out / 'raw'} (manifest unchanged)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--offline", action="store_true", help="rebuild the tables from the saved raw files")
    args = ap.parse_args()
    out = Path(args.out)
    if args.offline:
        return rebuild_offline(out)
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit('Set SEC_USER_AGENT="Name email" first (SEC fair-access policy).')
    facts, filings, skipped = [], [], []
    manifest = {"pulled_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    ciks = resolve_ciks(ua)
    for ticker, industry in COMPANIES.items():
        cik = ciks[ticker]
        if cik is None:
            skipped.append(f"{ticker}: not in SEC ticker list - add its CIK to CIK_OVERRIDES in config.py")
            continue
        try:
            raw = download_company(ticker, cik, out, ua, manifest)
        except Exception as e:  # noqa: BLE001 - one bad company must not stop the pull
            skipped.append(f"{ticker} (CIK {cik}): download failed - {e}")
            continue
        dropped = {}
        f = normalize_companyfacts(raw["companyfacts"], ticker, dropped)
        s = normalize_submissions(raw["submissions"], ticker)
        facts += f
        filings += s
        print(f"{ticker}: {raw['companyfacts']['entityName']} ({industry}) - "
              f"{len(f)} facts, {len({r['concept'] for r in f})} concepts, {len(s)} 10-K/10-Q filings")
        if not f:
            print(f"  WARNING {ticker}: no us-gaap facts found (a foreign filer reporting under IFRS?)")
        if dropped:
            print("  dropped: " + ", ".join(f"{n} {why.replace('_', ' ')}" for why, n in sorted(dropped.items())))
            manifest.setdefault("dropped", {})[ticker] = dropped
    manifest["skipped"] = skipped
    for line in skipped:
        print("SKIPPED", line)
    if not facts:
        sys.exit("Nothing downloaded.")
    (out / "processed").mkdir(parents=True, exist_ok=True)
    write_csv(out / "processed" / "facts.csv", facts)
    write_csv(out / "processed" / "filings.csv", filings)
    (out / "processed" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(facts)} facts and {len(filings)} filings to {out / 'processed'}")


if __name__ == "__main__":
    main()
