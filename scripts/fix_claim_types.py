# Usage (repo root): python scripts/fix_claim_types.py data/influencers/claims/claim_update_v3.jsonl claim_update_v3.fixed.jsonl
"""Write a corrected claim_update_v3.jsonl: same rows, same ids, fixed sub_claim_type (+ a type_fix audit field)."""
import json, re, sys, collections as C
from finsight.agents.claim_extraction.extract import _stored_type
from finsight.agents.claim_extraction.build import _classify, expand_stored_row
from finsight.agents.financial.agent import scope, XbrlFinancialAgent
from finsight.agents.financial.config import COMPANIES
from types import SimpleNamespace as _N
from finsight.agents.financial.claim_reader import read_claim, _SEGMENT_WORDS
_MEASURE_WORD = re.compile(r'\b(revenues?|sales|margins?|income|profits?|earnings|eps|cash flow|growth|grew|growing)\b', re.I)
def segment_only(text):
    clauses = [c for c in re.split(r'[,;]|\band\b', text) if _MEASURE_WORD.search(c)]
    return bool(clauses) and all(re.search(_SEGMENT_WORDS, c, re.I) for c in clauses)
import yaml
src, dst = sys.argv[1], sys.argv[2]
kl = yaml.safe_load(open('finsight/knowledge/knowledge_layer.yaml'))
ROUTE = {r['sub_claim_type']: tuple(r['collectors']) for r in kl['rules']}
route = lambda t: ROUTE.get(t, ('narrative',))
FIN = {t for t, c in ROUTE.items() if 'financial' in c}
MODEL_SET = lambda r: r.get('judge_source') == 'llm' or r.get('measure_source') in ('llm', 'block', 'judge')
REAL_LEVERAGE = re.compile(r"debt[- ]to[- ]equity|debt/equity|leverage ratio|(?:financial|balance[- ]sheet) leverage|"
                           r"leverage (?:is|was|remains|has been|looks) |leverage (?:is )?under control|"
                           r"(?:over|highly|heavily)[- ]?leveraged|de-?leverag", re.I)
OPERATING_LEVERAGE = re.compile(r"operating leverage|operational leverage", re.I)
PORTFOLIO = re.compile(r"\b(my|i own|portfolio|position|stake|fund)\b.*\bprofits? (?:of|exceeding)\b|\bprofits? of \$[\d,]+", re.I)
# Rows the dev gold set fixes by hand (context the rules cannot see).
MANUAL = {"GWUMX6S5fHE-GOOGL-t276": ("OTHER", "segment",
          "continues the Services margin line before it (a segment margin); dev item draft-40")}
NOT_COMPANY = "not a company figure; the narrative agent searches the stored documents"
def decide(r):
    """(new_type, new_measure, reason) or None."""
    sid, text, m = r['sub_claim_id'], r.get('text') or '', r.get('measure')
    if sid in MANUAL: return MANUAL[sid]
    eff = _stored_type(r).value
    if r.get('expansion_rule'):                      # section 6 children stay as built
        return (eff, m, 'loader re-type written into the file') if eff != r['sub_claim_type'] else None
    # 1. measure field says it is not computable
    if m == 'segment' and eff in FIN: return ('OTHER', m, 'measure=segment: a segment figure goes to the narrative agent (rule book 5, 11)')
    if m == 'guidance' and eff in FIN: return ('CONTINUATION', m, 'measure=guidance: forward-looking (rule book 5)')
    # 2. 'leverage' that is not debt-to-equity
    if eff == 'LEVERAGE' and not REAL_LEVERAGE.search(text):
        masked = re.sub(r'\b(?:operating |operational )?leverag\w*', ' ', text, flags=re.I)
        k = _classify(masked)
        t = k.type if k and k.type != 'LEVERAGE' else 'OTHER'
        return (t, (k.measure if k else None) or m, "'leverage' here is not debt-to-equity (verb or operating leverage)")
    # 3. the model's label set a financial type where the extraction rules give another route
    if eff in FIN and MODEL_SET(r):
        k = _classify(text)
        if (k is not None and not k.expansion and route(k.type) != route(eff)
                and not read_claim(text, r.get('company_ticker') or '').points):   # nothing computable either
            return (k.type, k.measure or m, f'the model label set {eff}; the extraction rules type it {k.type} (rule book 4: the model does not choose the route)')
    # 4. financial type the financial agent can never compute: a segment or a forecast (same lexicon as the agent)
    if eff in FIN:
        if PORTFOLIO.search(text): return ('OTHER', m, "the influencer's own investment gain, " + NOT_COMPANY)
        tk = r.get('company_ticker') or ''
        rd = read_claim(text, tk)
        if not rd.points and rd.notes:
            if any('segment' in n for n in rd.notes) and segment_only(text): return ('OTHER', m or 'segment', 'segment or product figure (rule book 5, 11)')
            if any('forecast' in n for n in rd.notes): return ('CONTINUATION', m, 'forward-looking wording (rule book 5)')
    # 5. otherwise write the loader's own re-type into the file so stored == used
    if eff != r['sub_claim_type']:
        return (eff, m, 'loader re-type written into the file')
    return None
FIG = re.compile(r"\d+(?:\.\d+)?\s*(?:%|percent)|\$\s*\d|\d+(?:\.\d+)?\s*(?:billion|million)", re.I)
BARE_MEASURES = {"revenue", "profit", "margin", "cash"}
def gate(r, t, m):
    """Step 1: a financial type only when the financial agent can calculate something in the sentence."""
    if t not in FIN or t == 'LEVERAGE' or r.get('expansion_rule') or r['company_ticker'] not in COMPANIES:
        return None
    if len(expand_stored_row({**r, 'sub_claim_type': t, 'measure': m})) > 1:
        return None                                   # split at load; each piece is typed by the rules
    rd = XbrlFinancialAgent._scoped(read_claim(r['text'], r['company_ticker']), scope(_N(measure=m)))
    if rd.points:
        return None
    if any('forecast' in n for n in rd.notes):
        return ('CONTINUATION', m, 'gate: the financial agent finds only a forecast to calculate')
    if rd.notes:
        return ('OTHER', m, 'gate: the financial agent cannot calculate it (' + rd.notes[0].split(': ', 1)[-1][:70] + ')')
    judged = (r.get('inherited_measure') or r.get('judge_measure') or '').lower()
    if FIG.search(r['text']) and judged in BARE_MEASURES:
        return None                                   # a bare figure labelled from context (rule book 4): keep
    return ('OTHER', m, 'gate: no measure the financial agent calculates is named')
rows = [json.loads(l) for l in open(src, encoding='utf-8')]
seen = C.Counter(); stats = C.Counter(); flips = []
out = []
for r in rows:
    d = decide(r)
    t0, m0 = (d[0], d[1]) if d else (r['sub_claim_type'], r.get('measure'))
    g = gate(r, t0, m0)
    if g:
        d = g
    if d:
        t, m, why = d
        if t != r['sub_claim_type'] or m != r.get('measure'):
            new = dict(r)
            new['type_fix'] = {'from_type': r['sub_claim_type'], 'from_measure': r.get('measure'), 'reason': why}
            new['sub_claim_type'], new['measure'] = t, m
            if _stored_type(new).value != t: flips.append((r['sub_claim_id'], t, _stored_type(new).value, r['text'][:90]))
            stats[(r['sub_claim_type'], t, why.split(':')[0].split(';')[0][:60])] += 1
            r = new
    seen[r['sub_claim_id']] += 1
    if seen[r['sub_claim_id']] > 1:                  # duplicate id: give the later row its own id
        r = {**r, 'sub_claim_id': f"{r['sub_claim_id']}-dup{seen[r['sub_claim_id']]}",
             'id_fix': f"duplicate of {r['sub_claim_id']}"}
    out.append(r)
with open(dst, 'w', encoding='utf-8', newline='\n') as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('rows', len(out), 'changed', sum(1 for r in out if 'type_fix' in r))
for k, v in stats.most_common(40): print(f'{v:5}  {k}')
print('loader would still flip', len(flips)); [print('   ', f) for f in flips[:15]]
