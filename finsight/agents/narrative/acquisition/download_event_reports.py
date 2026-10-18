"""Download primary SEC 8-K/6-K event reports for the current company scope.

Discovery and downloading are resumable and separate from canonical promotion.
SEC_USER_AGENT must contain an application name and a real contact email.
No exhibits, images, complete submission bundles or model calls are requested.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse

from bs4 import BeautifulSoup

from finsight.agents.narrative.acquisition.download_corpus import SecClient, SEC_DATA, _archive_base, _safe_name


EVENT_FORMS = {'8-K', '8-K/A', '6-K', '6-K/A'}


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + '.tmp')
    staged.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    staged.replace(path)


def _rows(payload: dict) -> list[dict]:
    table = payload.get('filings', {}).get('recent', payload)
    accessions = table.get('accessionNumber', [])
    fields = ['accessionNumber', 'filingDate', 'reportDate', 'form', 'primaryDocument', 'items', 'acceptanceDateTime']
    return [{field: (table[field][i] if field in table else '') for field in fields}
            for i in range(len(accessions))]


def discover(client: SecClient, workspace: Path, tickers: list[str], start: date, cutoff: date,
             intake: Path) -> dict:
    # Resolve actual issuer CIKs from SEC's ticker map. Accession prefixes can
    # identify filing agents and must never be used as the issuer CIK.
    mapping = client.json('https://www.sec.gov/files/company_tickers.json')
    companies = {r['ticker'].upper(): r for r in mapping.values()}
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'from_date': start.isoformat(),
              'cutoff_date': cutoff.isoformat(), 'forms': sorted(EVENT_FORMS),
              'company_scope': tickers, 'companies': {}, 'filings': [], 'errors': []}
    for ticker in tickers:
        try:
            company = companies[ticker]
            cik = str(company['cik_str'])
            submissions = client.json(f'{SEC_DATA}/submissions/CIK{int(cik):010d}.json')
            if int(submissions['cik']) != int(cik) or ticker not in submissions.get('tickers', []):
                raise ValueError('SEC issuer CIK/ticker identity did not match')
            _save_json(intake / 'submissions' / f'{ticker}_recent.json', submissions)
            filings = _rows(submissions)
            shards = []
            for shard in submissions.get('filings', {}).get('files', []):
                if shard['filingTo'] < start.isoformat() or shard['filingFrom'] > cutoff.isoformat():
                    continue
                payload = client.json(f"{SEC_DATA}/submissions/{shard['name']}")
                _save_json(intake / 'submissions' / shard['name'], payload)
                filings.extend(_rows(payload))
                shards.append(shard['name'])
            unique = {}
            for filing in filings:
                if filing['form'] in EVENT_FORMS and start.isoformat() <= filing['filingDate'] <= cutoff.isoformat():
                    unique.setdefault(filing['accessionNumber'], filing)
            selected = sorted(unique.values(), key=lambda f: (f['filingDate'], f['accessionNumber']))
            for filing in selected:
                primary = filing['primaryDocument']
                if not primary or '/' in primary or '\\' in primary or primary.startswith('.'):
                    raise ValueError('Unexpected primary document filename')
                base = _archive_base(cik, filing['accessionNumber'])
                report['filings'].append(dict(company_ticker=ticker, company_name=submissions['name'], cik=cik,
                    **filing, source_url=f'{base}/{urllib.parse.quote(primary)}',
                    source_page=f"{base}/{filing['accessionNumber']}-index.html"))
            report['companies'][ticker] = dict(cik=cik, company_name=submissions['name'],
                selected=len(selected), form_counts=dict(Counter(f['form'] for f in selected)),
                historical_shards=shards, discovery_status='complete')
            print(f'[{ticker}] Discovered {len(selected)} primary event reports', flush=True)
        except Exception as exc:
            report['errors'].append(dict(company_ticker=ticker, error=f'{type(exc).__name__}: {exc}'))
            print(f'[{ticker}] Discovery failed: {exc}', flush=True)
        _save_json(intake / 'discovery.json', report)
    return report


def download(client: SecClient, workspace: Path, discovery: dict, intake: Path) -> dict:
    manifest_path = intake / 'manifest_event_sec.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
        'schema_version': '1.0', 'source_policy': 'Official SEC primary event-report documents; exhibits remain separate records.',
        'from_date': discovery['from_date'], 'cutoff_date': discovery['cutoff_date'],
        'documents': [], 'download_errors': []}
    if (manifest['from_date'], manifest['cutoff_date']) != (discovery['from_date'], discovery['cutoff_date']):
        raise ValueError('Intake manifest belongs to a different date window')
    by_id = {r['document_id']: r for r in manifest['documents']}
    manifest['download_errors'] = []
    targets = discovery['filings']
    for ordinal, filing in enumerate(targets, 1):
        ticker, form, accession = filing['company_ticker'], filing['form'], filing['accessionNumber']
        doc_id = f'{ticker}-{accession}-{form.replace("/", "-")}'
        existing = by_id.get(doc_id)
        if existing:
            path = workspace / existing['local_path']
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == existing['sha256']:
                continue
        try:
            payload = client.get(filing['source_url'])
            suffix = Path(filing['primaryDocument']).suffix.lower()
            if suffix not in {'.htm', '.html', '.pdf', '.txt'}:
                raise ValueError(f'Unsupported primary document extension: {suffix}')
            if suffix == '.pdf':
                if not payload.startswith(b'%PDF'):
                    raise ValueError('Primary document response is not PDF')
            else:
                soup = BeautifulSoup(payload, 'html.parser')
                text = soup.get_text(' ', strip=True)
                if len(text) < 200 or not re.search(rf'FORM\s+{re.escape(form.split("/")[0])}', text[:20000], re.I):
                    raise ValueError('Response does not contain the expected primary event form header')
                if any(message in text[:3000].lower() for message in ['request rate threshold exceeded', 'your request originates from an undeclared automated tool']):
                    raise ValueError('SEC returned an access/rate-limit page')
            filename = _safe_name(f"{filing['filingDate']}_{form}_{accession}_{filing['primaryDocument']}")
            dest = workspace / 'data/narrative/raw' / ticker / 'event_sec' / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise ValueError('Existing original file differs; refusing to overwrite')
            staged = dest.with_suffix(dest.suffix + '.download_tmp')
            staged.write_bytes(payload)
            staged.replace(dest)
            by_id[doc_id] = dict(document_id=doc_id, company_ticker=ticker, company_name=filing['company_name'],
                cik=filing['cik'], document_type=form, form=form, period=filing.get('reportDate') or None,
                publication_date=filing['filingDate'], publication_date_is_estimated=False, date_precision='day',
                publication_date_basis='SEC filing date, distinct from event reportDate.', accession_number=accession,
                acceptance_datetime=filing.get('acceptanceDateTime'), form_items=filing.get('items'),
                source_tier=1, source_origin='SEC EDGAR official primary event report',
                source_url=filing['source_url'], source_page=filing['source_page'],
                local_path=dest.relative_to(workspace).as_posix(), sha256=digest,
                retrieval_eligible=True, is_amendment=form.endswith('/A'),
                primary_document=filing['primaryDocument'], content_role='primary_event_report_body')
            print(f'[{ordinal}/{len(targets)}] {ticker} {form} {filing["filingDate"]}: downloaded', flush=True)
        except Exception as exc:
            manifest['download_errors'].append(dict(company_ticker=ticker, accession_number=accession,
                source_url=filing['source_url'], error=f'{type(exc).__name__}: {exc}'))
            print(f'[{ordinal}/{len(targets)}] {ticker} {accession}: failed: {exc}', flush=True)
        manifest['documents'] = sorted(by_id.values(), key=lambda r: (r['company_ticker'], r['publication_date'], r['document_id']))
        manifest['updated_at'] = datetime.now(timezone.utc).isoformat()
        _save_json(manifest_path, manifest)
        # Do not repeatedly hammer a blocked SEC endpoint across hundreds of files.
        if len(manifest['download_errors']) >= 3 and not by_id:
            break
    manifest['documents'] = sorted(by_id.values(), key=lambda r: (r['company_ticker'], r['publication_date'], r['document_id']))
    manifest['expected_filings'] = len(targets)
    manifest['discovery_errors'] = discovery['errors']
    manifest['status'] = 'complete' if len(by_id) == len(targets) and not discovery['errors'] and not manifest['download_errors'] else 'partial'
    _save_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--from-date', type=date.fromisoformat, default=date(2024, 1, 1))
    parser.add_argument('--cutoff', type=date.fromisoformat, default=date(2026, 10, 4))
    parser.add_argument('--tickers', nargs='+')
    parser.add_argument('--user-agent', default=os.getenv('SEC_USER_AGENT'))
    parser.add_argument('--discover-only', action='store_true')
    parser.add_argument('--use-discovery', action='store_true', help='Reuse the saved reviewed discovery inventory')
    parser.add_argument('--intake', type=Path, default=Path('data/narrative/intake/event_sec'))
    args = parser.parse_args()
    if not args.user_agent or not re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', args.user_agent) or 'example.com' in args.user_agent:
        parser.error('Set SEC_USER_AGENT to an application name and a real monitored contact email before downloading.')
    workspace = args.workspace.resolve()
    intake = (workspace / args.intake).resolve()
    if not intake.is_relative_to(workspace / 'data/narrative'):
        parser.error('Intake must remain within the workspace data/narrative directory.')
    core = json.loads((workspace / 'data/narrative/raw/manifest_core_sec.json').read_text(encoding='utf-8'))
    tickers = sorted(set(args.tickers or [r['company_ticker'] for r in core['documents']]))
    client = SecClient(args.user_agent, delay=0.3, timeout=30, attempts=3)
    discovery = json.loads((intake / 'discovery.json').read_text(encoding='utf-8')) if args.use_discovery else discover(client, workspace, tickers, args.from_date, args.cutoff, intake)
    if discovery['company_scope'] != tickers or discovery['from_date'] != args.from_date.isoformat() or discovery['cutoff_date'] != args.cutoff.isoformat():
        parser.error('Saved discovery does not match the requested companies/date window.')
    if args.discover_only:
        print(json.dumps({'selected': len(discovery['filings']), 'companies': discovery['companies'], 'errors': discovery['errors']}, indent=2))
        raise SystemExit(bool(discovery['errors']))
    manifest = download(client, workspace, discovery, intake)
    print(json.dumps({'status': manifest['status'], 'selected': manifest['expected_filings'],
        'downloaded': len(manifest['documents']), 'errors': len(manifest['download_errors']),
        'manifest': str(intake / 'manifest_event_sec.json')}, indent=2))
    raise SystemExit(manifest['status'] != 'complete')


if __name__ == '__main__':
    main()
