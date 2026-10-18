"""Download public Q4 filing-platform copies of primary event reports, without SEC API access.

No email, account, SEC_USER_AGENT or paid service is used. Raw downloads and the
intake manifest are separate from canonical registration and retrieval indexes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup

from finsight.agents.narrative.acquisition.download_event_reports import _save_json


# Issuer identities, not accession-number/filing-agent prefixes.
ISSUERS = {'AAPL': '320193', 'AMD': '2488', 'AMZN': '1018724', 'CRM': '1108524',
           'GOOGL': '1652044', 'INTC': '50863', 'INTU': '896878', 'MELI': '1099590',
           'META': '1326801', 'MSFT': '789019', 'NFLX': '1065280', 'NVDA': '1045810',
           'ORCL': '1341439', 'PLTR': '1321655', 'SBUX': '829224', 'TSLA': '1318605'}
FEED = 'https://investor.nvidia.com/feed/SECFiling.svc/GetEdgarFilingList'
FORMS = {'8-K', '8-K/A', '6-K', '6-K/A'}


def extract_primary_report(record: dict, root: Path) -> dict:
    """Keep the original platform bundle and derive only its explicit primary form."""
    original_path = record.get('original_local_path') or record['local_path']
    source = root / original_path
    data = source.read_bytes()
    forms = list(re.finditer(rb'<efx_form(?:\s[^>]*)?>(.*?)</efx_form\s*>', data, re.I | re.S))
    if len(forms) != 1:
        record.update(primary_body_status='needs_structure_review', processing_status='downloaded_not_processed')
        return record
    body = forms[0].group(1)
    primary = b'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body>' + body + b'</body></html>'
    dest = source.with_name(source.stem + '.primary.html')
    dest.write_bytes(primary)
    record.update(original_local_path=original_path, source_sha256=hashlib.sha256(data).hexdigest(),
        local_path=dest.relative_to(root).as_posix(), sha256=hashlib.sha256(primary).hexdigest(),
        source_ref='efx_form in original platform HTML representation',
        content_role='primary_event_report_body', primary_body_status='extracted_explicit_primary_form_container',
        processing_status='downloaded_not_processed', accession_number=None,
        accession_number_status='Not supplied in platform feed; accession references in report text are not inferred as the current accession.')
    return record


def get(url: str) -> bytes:
    host = urllib.parse.urlsplit(url).hostname or ''
    if host == 'sec.gov' or host.endswith('.sec.gov'):
        raise ValueError('This alternate collector does not request SEC-hosted URLs')
    request = urllib.request.Request(url, headers={'User-Agent': 'FinSight research document collector'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(25_000_000)
            time.sleep(0.3)
            return data
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise RuntimeError('Download did not return')


def discover_one(task: tuple, intake: Path) -> dict:
    ticker, year = task
    cik = ISSUERS[ticker]
    query = dict(exchange='CIK', symbol=f'{int(cik):010d}', filingTypeList=','.join(sorted(FORMS)),
                 year=year, pageSize=-1, pageNumber=0, includeHtmlDocument='true',
                 excludeNoDocuments='true', LanguageId=1)
    url = FEED + '?' + urllib.parse.urlencode(query)
    cache = intake / 'feeds' / f'{ticker}_{year}.json'
    try:
        payload = json.loads(cache.read_text(encoding='utf-8')) if cache.exists() else json.loads(get(url))
        rows = payload['GetEdgarFilingListResult']
        for row in rows:
            if int(row['StockSymbol']) != int(cik) or row['StockExchange'] != 'CIK':
                raise ValueError('Filing feed returned a different issuer identity')
        _save_json(cache, payload)
        print(f'[{ticker} {year}] {len(rows)} event filings in public platform feed', flush=True)
        return dict(ticker=ticker, year=year, feed_url=url, filings=rows)
    except Exception as exc:
        print(f'[{ticker} {year}] Feed failed: {exc}', flush=True)
        return dict(ticker=ticker, year=year, feed_url=url, error=f'{type(exc).__name__}: {exc}', filings=[])


def download_one(task: dict, root: Path) -> dict:
    ticker, row = task['ticker'], task['filing']
    form, filing_id = row['FilingTypeMnemonic'], row['FilingId']
    day = datetime.strptime(row['FilingDate'], '%m/%d/%Y %H:%M:%S').date().isoformat()
    doc_id = f'{ticker}-IR-EVENT-{filing_id}-{form.replace("/", "-")}'
    result = dict(document_id=doc_id, company_ticker=ticker, filing_id=filing_id)
    try:
        choices = [d for d in row['DocumentList'] if d.get('DocumentType') == 'HTML' and d.get('Url')]
        if not choices:
            raise ValueError('No primary HTML representation in public feed')
        url = choices[0]['Url']
        if url.startswith('//'):
            url = 'https:' + url
        host = urllib.parse.urlsplit(url).hostname or ''
        if not host.endswith(('.cloudfront.net', '.q4cdn.com')):
            raise ValueError('Feed document is not hosted on the observed public filing-platform CDNs')
        dest = root / 'data/narrative/raw' / ticker / 'event_ir' / f'{day}_{form.replace("/", "-")}_{filing_id}.html'
        payload = dest.read_bytes() if dest.exists() else get(url)
        soup = BeautifulSoup(payload, 'html.parser')
        for element in soup.select('script, style'):
            element.decompose()
        text = soup.get_text(' ', strip=True)
        if len(text) < 250 or not re.search(rf'FORM\s+{re.escape(form.split("/")[0])}', text[:30000], re.I):
            raise ValueError('Expected event form header is absent')
        cik = ISSUERS[ticker]
        # Require independent identification inside the actual filing document.
        cik_match = re.search(r'CENTRAL\s+INDEX\s+KEY\s*:?\s*(\d+)', text[:20000], re.I)
        if cik_match and int(cik_match[1]) != int(cik):
            raise ValueError('Primary document header CIK does not match requested issuer')
        accession = re.search(r'(?:ACCESSION(?:\s+NUMBER)?\s*:?\s*)(\d{10}-\d{2}-\d{6})', text[:20000], re.I)
        if not accession:
            accession = re.search(r'(\d{10}-\d{2}-\d{6})', payload[:20000].decode('utf-8', errors='replace'))
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            staged = dest.with_suffix('.html.download_tmp')
            staged.write_bytes(payload)
            staged.replace(dest)
        result.update(status='downloaded', company_name=row['FilingAgentName'], cik=cik,
            document_type=form, form=form, publication_date=day, date_precision='day',
            publication_date_is_estimated=False, publication_date_basis='Filing date from public Q4 SEC filing feed.',
            accession_number=accession[1] if accession else None, source_tier=3,
            source_origin='Public Q4 regulatory filing platform copy, discovered through a public company IR feed; not downloaded from SEC EDGAR.',
            publisher='Q4 filing platform', source_url=url, source_page=task['feed_url'],
            feed_detail_url=urllib.parse.urljoin('https://investor.nvidia.com', row['LinkToDetailPage']),
            local_path=dest.relative_to(root).as_posix(), sha256=hashlib.sha256(payload).hexdigest(),
            retrieval_eligible=True, content_role='event_filing_html_representation',
            text_characters=len(text), is_amendment=form.endswith('/A'),
            header_excerpt=text[:1400])
        print(f'[{ticker}] {form} {day} {filing_id}: downloaded', flush=True)
    except Exception as exc:
        result.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(f'[{ticker}] {filing_id}: {result["error"]}', flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--tickers', nargs='+', choices=sorted(ISSUERS), default=sorted(ISSUERS))
    parser.add_argument('--from-date', type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument('--cutoff', type=date.fromisoformat, default=date(2026, 10, 4))
    parser.add_argument('--discover-only', action='store_true')
    args = parser.parse_args()
    root = args.workspace.resolve()
    intake = root / 'data/narrative/intake/event_ir'
    tasks = [(ticker, year) for ticker in args.tickers for year in range(args.from_date.year, args.cutoff.year + 1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        feeds = list(pool.map(lambda task: discover_one(task, intake), tasks))
    selected, seen = [], set()
    for feed in feeds:
        for row in feed['filings']:
            day = datetime.strptime(row['FilingDate'], '%m/%d/%Y %H:%M:%S').date()
            key = (feed['ticker'], row['FilingId'])
            if row['FilingTypeMnemonic'] not in FORMS or not args.from_date <= day <= args.cutoff or key in seen:
                continue
            seen.add(key)
            selected.append(dict(ticker=feed['ticker'], feed_url=feed['feed_url'], filing=row))
    discovery = dict(from_date=args.from_date.isoformat(), cutoff_date=args.cutoff.isoformat(),
        feeds=feeds, selected=selected, expected_filings=len(selected),
        discovery_errors=[f for f in feeds if 'error' in f],
        limitation='This is the public Q4 platform inventory, not an independently reconciled SEC submission inventory.')
    _save_json(intake / 'discovery.json', discovery)
    print('Selected', len(selected), 'event report representations:', dict(Counter(t['ticker'] for t in selected)), flush=True)
    if args.discover_only:
        return
    manifest = dict(schema_version='1.0', from_date=args.from_date.isoformat(), cutoff_date=args.cutoff.isoformat(),
        source_policy='Public Q4 filing platform representations, source tier 3. No personal email or SEC API access.',
        expected_filings=len(selected), documents=[], download_errors=[],
        discovery_errors=discovery['discovery_errors'], limitation=discovery['limitation'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda task: download_one(task, root), selected):
            manifest['documents' if result['status'] == 'downloaded' else 'download_errors'].append(result)
            manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
            _save_json(intake / 'manifest_event_ir.json', manifest)
    manifest['documents'] = [extract_primary_report(r, root) for r in manifest['documents']]
    manifest['companies_without_feed_records'] = sorted(set(args.tickers) - {r['company_ticker'] for r in manifest['documents']})
    manifest['status'] = 'complete_for_public_feed_inventory' if not manifest['download_errors'] and not manifest['discovery_errors'] else 'partial'
    _save_json(intake / 'manifest_event_ir.json', manifest)
    print(json.dumps(dict(status=manifest['status'], expected=len(selected), downloaded=len(manifest['documents']),
                         errors=len(manifest['download_errors'])), indent=2))


if __name__ == '__main__':
    main()
