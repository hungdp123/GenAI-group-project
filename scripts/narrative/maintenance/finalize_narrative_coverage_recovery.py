"""Assemble intake downloads into recovery metadata before canonical promotion.

Historical intake finalizer. After merging, use the canonical raw manifests;
this script refuses to recreate the obsolete staging tree.
"""
from collections import Counter
from datetime import datetime
import hashlib, json, re, shutil, zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader
from finsight.agents.narrative.processing.corpus import CANONICAL_MANIFESTS
from finsight.agents.narrative.acquisition.download_ir_event_reports import ISSUERS


def main():
 root=Path.cwd(); intake=root/'data/narrative/intake/coverage_recovery'
 recovery_path=intake/'manifest_document_recovery.json'
 if recovery_path.exists():
  recovery=json.loads(recovery_path.read_text(encoding='utf-8'))
  if recovery.get('canonical_registration_status')=='merged_into_main_manifests':
   raise RuntimeError('Coverage recovery is already merged. Do not rerun this intake finalizer; the canonical manifests are under data/narrative/raw/.')
 original=[a for p in CANONICAL_MANIFESTS for a in json.loads((root/p).read_text(encoding='utf-8'))['documents']]
 event_path=root/'data/narrative/intake/event_ir/manifest_event_reports.json'
 event_manifest=json.loads(event_path.read_text(encoding='utf-8'))
 # The old Tesla PDF name records its event date, not its filing date. Both the
 # PDF's Hartung appointment body and Nasdaq's 2025-05-16 HTML identify this copy.
 corrections=[]
 for a in event_manifest['documents']:
  if a.get('accession_number')=='0001628280-25-026148' and a['publication_date']=='2025-05-15':
   backup=event_path.with_name('manifest_event_reports.before_coverage_recovery.json')
   if not backup.exists():shutil.copy2(event_path,backup)
   a.update(event_date='2025-05-15',publication_date='2025-05-16',
            publication_date_basis='Filing date in Nasdaq/QuoteMedia historical inventory; matched Hartung appointment text in supplied official IR PDF. Filename retains the event date.',
            alternate_source_url=next(t['url'] for t in json.loads((intake/'filing_candidates.json').read_text()) if t['provider_ref']=='319176415'))
   corrections.append(dict(document_id=a['document_id'],field='publication_date',old='2025-05-15',new='2025-05-16'))
 if corrections:event_path.write_text(json.dumps(event_manifest,indent=2)+'\n',encoding='utf-8')
 known=original+event_manifest['documents']; hashes={a['sha256']:a['document_id'] for a in known}
 docs=[]; duplicates=[]; errors=[]
 candidates=[]
 for name in ['presentation_candidates.json','msft_candidates.json','extra_candidates.json','meli_candidates.json','filing_candidates.json']:
  candidates+=json.loads((intake/name).read_text(encoding='utf-8'))
 seen=set()
 for c in candidates:
  if c['file'] in seen:continue
  seen.add(c['file']); src=intake/c['file']; t=c['ticker']; kind=c['document_type']
  try:
   if not src.is_file():raise ValueError('No downloaded source file')
   data=src.read_bytes(); digest=hashlib.sha256(data).hexdigest()
   if digest in hashes:
    duplicates.append(dict(file=c['file'],duplicate_of=hashes[digest]));continue
   if c.get('provider_ref')=='319176415':
    duplicates.append(dict(file=c['file'],duplicate_of='TSLA-0001628280-25-026148-8-K',reason='Same Hartung appointment filing in HTML and official IR PDF; filing date corrected to 2025-05-16.'));continue
   suffix=src.suffix.lower(); page_count=None; text=''
   if suffix=='.pdf':
    if not data.startswith(b'%PDF') or b'%%EOF' not in data[-4096:]:raise ValueError('Incomplete or non-PDF download')
    reader=PdfReader(src);page_count=len(reader.pages)
    text=' '.join(p.extract_text() or '' for p in reader.pages[:2])
   elif suffix=='.pptx':
    with zipfile.ZipFile(src) as z:
     slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
     if not slides:raise ValueError('No PowerPoint slides')
     page_count=len(slides)
     text=' '.join(BeautifulSoup(z.read(n),'xml').get_text(' ',strip=True) for n in sorted(slides)[:2])
   elif suffix=='.html':
    forms=list(re.finditer(rb'<html\b[^>]*>.*?</html\s*>',data,re.I|re.S))
    if not forms:raise ValueError('No complete HTML filing')
    primary=forms[0][0]
    text=BeautifulSoup(primary,'html.parser').get_text(' ',strip=True)
    if not re.search(r'FORM\s+'+re.escape(kind.split('/')[0]),text,re.I):raise ValueError('Main filing form header missing')
    if ISSUERS[t].zfill(10) not in text[:15000]:raise ValueError('Issuer CIK missing in primary filing')
   safe_date=c.get('publication_date') or c.get('period') or 'date_to_confirm'
   dest=root/'data/narrative/raw'/t/'coverage_recovery'/f'{safe_date}_{kind.replace("/","-")}_{digest[:16]}{suffix}'
   dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():shutil.copy2(src,dest)
   row=dict(document_id=f'{t}-COVERAGE-{digest[:16]}',company_ticker=t,document_type=kind,
       period=c.get('period'),publication_date=c.get('publication_date'),
       publication_date_basis='Official source event date or printed presentation date.' if c.get('publication_date') else 'Quarter identified; exact publication date not independently confirmed.',
       source_tier=c.get('source_tier',2),source_url=c['url'],source_page=c['source_page'],
       local_path=dest.relative_to(root).as_posix(),sha256=digest,pages_or_slides=page_count,
       availability_status='downloaded',processing_status='not_registered_or_processed',label=c.get('label'),
       publisher='Nasdaq / QuoteMedia' if suffix=='.html' else ('Seeking Alpha (company slide copy)' if c.get('source_tier')==3 else 'Official company investor relations'),
       source_inventory_path=c.get('inventory_path'),provider_ref=c.get('provider_ref'))
   if t=='NFLX':row.update(publication_date=None,publication_date_basis='Official archive labels this Content Accounting Overview as May 2025; exact day not confirmed.')
   if suffix=='.html':
    primary_dest=dest.with_suffix('.primary.html');primary_dest.write_bytes(primary)
    row.update(original_local_path=row['local_path'],source_sha256=digest,local_path=primary_dest.relative_to(root).as_posix(),sha256=hashlib.sha256(primary).hexdigest(),
               content_role='primary_event_report_body',source_ref='First complete HTML document in the preserved QuoteMedia filing bundle; form and issuer CIK confirmed.',publication_date_basis='Filing date in Nasdaq/QuoteMedia historical filing inventory.')
   docs.append(row);hashes[digest]=row['document_id']
  except Exception as e:errors.append(dict(file=c['file'],ticker=t,error=f'{type(e).__name__}: {e}'))
 manifest=dict(review_date='2026-10-17',scope='Additional downloaded documents for availability coverage; separate from canonical retrieval manifests.',documents=docs,duplicates=duplicates,errors=errors,metadata_corrections=corrections)
 (intake/'manifest_document_recovery.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 all_docs=known+docs
 counts=Counter((a['company_ticker'],a['document_type'],a.get('publication_date')) for a in all_docs if (root/a['local_path']).is_file())
 comparison=json.loads((intake/'filing_comparison.json').read_text())
 for a in comparison:
  p=intake/f'nasdaq_inventory/{a["ticker"]}_{a["year"]}_{a["category"]}.json'
  rows=json.loads(p.read_text())['data']['rows'];seen=Counter();missing=[]
  for row in rows:
   if row['formType'] not in ['10-K','10-Q','8-K','8-K/A','6-K','6-K/A']:continue
   day=datetime.strptime(row['filed'],'%m/%d/%Y').date().isoformat()
   if not '2024-01-01'<=day<='2026-10-04':continue
   key=(a['ticker'],row['formType'],day);seen[key]+=1
   if counts[key]<seen[key]:missing.append(dict(form=row['formType'],filing_date=day,url=row['view']['htmlLink']))
  a['missing_after_download']=missing
 (intake/'filing_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(new_documents=len(docs),by_type=dict(Counter(d['document_type'] for d in docs)),duplicates=len(duplicates),errors=errors,missing_reference_filings=sum(len(c['missing_after_download']) for c in comparison)),indent=2))

if __name__=='__main__':main()
