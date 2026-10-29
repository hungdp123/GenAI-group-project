"""Generation-only capture must preserve the saved evidence and never search again."""
from argparse import Namespace
import csv
import json
from pathlib import Path

import pytest

from scripts.narrative.evaluation.evaluate_rag_pilot import case_identity, evaluate, read_json, write_json
from scripts.narrative.evaluation.resume import LINE_ENDING_EQUIVALENTS
from scripts.narrative.evaluation.run_rag_pilot import sha


def csv_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    from finsight.contracts.schemas import EvidenceItem, Stance
    from finsight.agents.narrative.rag import collect_evidence, evidence_summary
    from finsight.agents.narrative.retrieval.retriever import CorpusRetriever
    from finsight.llm.client import LLMResponse

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('FINSIGHT_LLM_PROVIDER', 'openai')
    monkeypatch.setenv('FINSIGHT_LLM_MODEL', 'gpt-5.4')
    monkeypatch.delenv('FINSIGHT_NARRATIVE_RERANKER_MODEL', raising=False)
    corpus = tmp_path / 'data/narrative/processed/chunking_strategies/C1/full_v1'
    company = corpus / 'AAPL'
    parsed = tmp_path / 'data/narrative/processed/AAPL/documents.jsonl'
    parsed.parent.mkdir(parents=True)
    parsed.write_text(json.dumps({'document_id': 'doc', 'publication_date': '2024-01-01',
                                  'text': 'alpha beta.'}) + '\n', encoding='utf-8')
    chunks = [{'chunk_id': cid, 'document_id': 'doc', 'text': text, 'company_ticker': 'AAPL',
               'publication_date': '2024-01-01', 'source_tier': 'TIER_2',
               'document_type': 'earnings_release', 'company_name': 'Apple Inc.',
               'source_url': 'https://example.org/source', 'citation_span': 'block 1'}
              for cid, text in [('discarded', 'unrelated context.'), ('kept', 'alpha beta.')]]
    write_json(company / 'indexes/index.json', {'corpus_fingerprint': 'fingerprint', 'chunks': chunks})
    (company / 'source_map.jsonl').write_text('\n'.join(json.dumps({
        'chunk_id': c['chunk_id'], 'document_id': 'doc', 'block_line_1based': 1,
        'body_char_span': [0, len(c['text'])]}) for c in chunks) + '\n', encoding='utf-8')
    write_json(company / 'build_report.json', {'status': 'COMPLETE', 'corpus_fingerprint': 'fingerprint',
               'inputs': {'parsed_sha256': sha(parsed)}, 'index_sha256': sha(company / 'indexes/index.json')})
    for rel in ('agents/narrative/rag/basic_rag.py', 'agents/narrative/rag/collect_evidence.py',
                'agents/narrative/rag/evidence_summary.py', 'agents/narrative/retrieval/retriever.py',
                'agents/narrative/rag/verdict_baseline.py', 'llm/client.py'):
        path = tmp_path / 'finsight' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# test source identity\n', encoding='utf-8')
    cases = [{'id': f'AAPL-SUP-0{i}', 'company': 'AAPL', 'claims': 'alpha beta',
              'cut_off': '2024-01-01', 'type': 'supported', 'gold_snippet': 'alpha beta.',
              'gold_answer': 'SECRET GOLD ANSWER', 'source_doc': 'doc | source\nblock 1 | chars 0:11'}
             for i in (1, 2)]
    dataset = tmp_path / 'dataset.csv'
    csv_file(dataset, cases)
    source = tmp_path / 'retrieval'
    folder = source / 'configurations/C1/bm25'
    csv_file(folder / 'cases.csv', cases)
    write_json(source / 'matrix_configuration.json', {'retrieval_only': True,
               'configurations': [{'strategy': 'C1', 'backend': 'bm25', 'folder': 'configurations/C1/bm25'}]})
    write_json(folder / 'configuration.json', {
        'dataset_sha256': sha(dataset), 'selected_cases_sha256': case_identity(cases),
        'case_ids': [c['id'] for c in cases], 'backend': 'bm25', 'corpus_root': str(corpus),
        'top_k_per_direction': 10, 'candidate_k': 50, 'rerank_overflow': 'truncate',
        'retrieval_only': True, 'source_priority': 'P0',
        'reranker_model': 'cross-encoder/ms-marco-MiniLM-L-6-v2'})
    for case in cases:
        p = folder / 'cases' / case['id']
        index = {'corpus_fingerprint': 'fingerprint'}
        hits = [{'rank': n, 'chunk_id': c['chunk_id'], 'text': c['text'],
                 'doc_type': c['document_type'], 'publication_date': c['publication_date'],
                 'url': c['source_url'], 'citation_span': c['citation_span']}
                for n, c in enumerate(chunks, 1)]
        trace = {'ticker': 'AAPL', 'claim': case['claims'], 'cutoff': case['cut_off'],
                 'sub_claim_id': 'basic-rag-claim', 'backend': 'bm25', 'top_k': 10, 'candidate_k': 50,
                 'index': index, 'reranking': None, 'support': hits, 'counter': [],
                 'support_query': 'saved support query', 'counter_query': 'saved counter query'}
        item = EvidenceItem(evidence_id='kept', sub_claim_id='basic-rag-claim', stance=Stance.NEUTRAL,
                            content='alpha beta.', source_tier='TIER_2', doc_type='earnings_release',
                            company='Apple Inc.', publication_date='2024-01-01',
                            url='https://example.org/source', citation_span='block 1',
                            produced_by='narrative_retrieval_unassessed').model_dump(mode='json', exclude={'stance'})
        write_json(p / 'result.json', {'status': 'OK', 'pipeline': 'narrative_evidence_retrieval',
                   'ticker': 'AAPL', 'claim': case['claims'], 'cutoff_date': case['cut_off'], 'backend': 'bm25',
                   'candidate_evidence': [item], 'evidence': [item], 'selection_applied': False,
                   'selection': {'returned_count': 1, 'returned_evidence_ids': ['kept']},
                   'index': index, 'reranking': None,
                   'retrieval': {'count': 1, 'elapsed_ms': 1234.5, 'evidence_ids': ['kept'],
                                 'queries': {'support': trace['support_query'], 'counter': trace['counter_query']}}})
        write_json(p / 'retrieval.json', trace)
        write_json(p / 'preflight.json', {'status': 'PASS', 'parsed_sha256': sha(parsed),
                   'index_sha256': sha(company / 'indexes/index.json'), 'corpus_fingerprint': 'fingerprint',
                   'source_map_sha256': sha(company / 'source_map.jsonl'), 'gold_source_block_1based': 1,
                   'semantic_labels': 'NOT_INDEPENDENTLY_ADJUDICATED'})
    calls = []

    def fake_complete(*args, **kwargs):
        calls.append(kwargs)
        assert 'SECRET GOLD ANSWER' not in kwargs['prompt']
        assert 'unrelated context.' not in kwargs['prompt']
        assert 'alpha beta.' in kwargs['prompt']
        return LLMResponse(text=json.dumps({
            'selected_evidence_ids': ['kept'], 'stance': 'SUPPORTED',
            'stance_reason': 'The source states alpha beta.', 'stance_reason_code': 'ALL_MATERIAL_PARTS_SUPPORTED',
            'supporting_fact_indices': [0], 'contradicting_fact_indices': [], 'missing_claim_parts': [],
            'facts': [{'evidence_id': 'kept', 'quote': 'alpha beta.'}],
            'summary': [{'text': 'Source states alpha beta.', 'evidence_ids': ['kept']}]}),
            input_tokens=20, output_tokens=10, cost_usd=0, latency_ms=1,
            model='gpt-5.4', provider='openai', finish_reason='stop')

    monkeypatch.setattr(evidence_summary, 'complete', fake_complete)
    monkeypatch.setattr(collect_evidence.NarrativeEvidenceAgent, '__init__',
                        lambda *a, **k: pytest.fail('Live collector must not be initialized'))
    monkeypatch.setattr(CorpusRetriever, '__init__',
                        lambda *a, **k: pytest.fail('Retrieval backend must not be initialized'))
    args = Namespace(cases=dataset, limit=1, corpus_root=corpus, output_dir=tmp_path / 'generated',
                     backend='bm25', top_k=10, candidate_k=50, max_tokens=3000, ollama_model=None,
                     ollama_host='http://not-called', ollama_context=20000, rerank_overflow='truncate',
                     retrieval_source_dir=folder, resume=True)
    return Namespace(args=args, folder=folder, source=source, cases=cases, calls=calls, corpus=corpus)


def test_generation_preserves_filtered_candidates_and_rankings_and_resumes(saved):
    from scripts.narrative.evaluation.run_rag_pilot import run

    before = {p: p.read_bytes() for p in saved.source.rglob('*') if p.is_file()}
    report = run(saved.args)
    assert report['status'] == 'COMPLETE' and len(saved.calls) == 1
    current = saved.args.output_dir / 'cases/AAPL-SUP-01'
    result = read_json(current / 'result.json')
    assert result['stance'] == 'SUPPORTED'
    assert [c['evidence_id'] for c in result['candidate_evidence']] == ['kept']
    assert result['retrieval']['elapsed_ms'] == 1234.5
    assert report['new_retrieval_requests'] == result['retrieval_reuse']['new_retrieval_requests'] == 0
    assert (current / 'retrieval.json').read_bytes() == before[saved.folder / 'cases/AAPL-SUP-01/retrieval.json']
    assert (current / 'llm_input.json').exists() and (current / 'llm_response.json').exists()
    assert evaluate(saved.args.output_dir, Path.cwd())['stance_metrics']['accuracy'] == 1
    run(saved.args)
    assert len(saved.calls) == 1
    saved.args.limit = 2
    run(saved.args)
    assert len(saved.calls) == 2
    assert before == {p: p.read_bytes() for p in saved.source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation', ['settings', 'dataset', 'snapshot', 'missing', 'trace', 'candidate', 'preflight'])
def test_mismatched_or_corrupt_source_fails_before_generation(saved, mutation):
    from scripts.narrative.evaluation.run_rag_pilot import run

    case_dir = saved.folder / 'cases/AAPL-SUP-01'
    if mutation == 'settings':
        saved.args.top_k = 5
    elif mutation == 'dataset':
        saved.args.cases.write_bytes(saved.args.cases.read_bytes() + b'\n')
    elif mutation == 'snapshot':
        csv_file(saved.folder / 'cases.csv', [{**c, 'claims': 'changed'} for c in saved.cases])
    elif mutation == 'missing':
        (case_dir / 'retrieval.json').unlink()
    else:
        names = {'trace': 'retrieval.json', 'candidate': 'result.json', 'preflight': 'preflight.json'}
        path = case_dir / names[mutation]
        data = read_json(path)
        if mutation == 'trace':
            data['support'][0]['text'] = 'corrupted source'
        elif mutation == 'candidate':
            data['candidate_evidence'][0]['url'] = 'https://wrong.example'
        else:
            data['parsed_sha256'] = 'wrong'
        write_json(path, data)
    with pytest.raises((ValueError, FileNotFoundError)):
        run(saved.args)
    assert not saved.calls
    assert not list(saved.args.output_dir.glob('cases/*/llm_input.json'))


def test_pin_rejects_changed_source_on_resume(saved):
    from scripts.narrative.evaluation.run_rag_pilot import run

    saved.args.prepare_only = True
    run(saved.args)
    path = saved.folder / 'cases/AAPL-SUP-01/retrieval.json'
    data = read_json(path)
    data['at'] = 'new timestamp; same passages'
    write_json(path, data)
    saved.args.prepare_only = False
    with pytest.raises(ValueError, match='Pinned saved retrieval changed'):
        run(saved.args)
    assert not saved.calls


@pytest.mark.parametrize('backend', ['dense', 'r4'])
def test_saved_vectors_do_not_require_qdrant_storage(saved, backend):
    from scripts.narrative.evaluation.run_rag_pilot import run
    from finsight.agents.narrative.retrieval.retriever import normalize_backend

    saved.args.backend = backend
    for path in [saved.folder / 'configuration.json', *saved.folder.glob('cases/*/result.json'),
                 *saved.folder.glob('cases/*/retrieval.json')]:
        data = read_json(path)
        data['backend'] = normalize_backend(backend) if path.name == 'retrieval.json' else backend
        write_json(path, data)
    assert not (saved.corpus / 'AAPL/indexes/qdrant').exists()
    assert run(saved.args)['status'] == 'COMPLETE'
    assert len(saved.calls) == 1


def test_matrix_routes_saved_configuration_and_rejects_overlap_and_missing_pair(saved):
    from scripts.narrative.evaluation import run_rag_comparison as matrix

    args = Namespace(**{**vars(saved.args), 'output_dir': Path.cwd() / 'matrix',
                       'all_cases': False, 'strategies': ['C1'], 'backends': ['bm25'],
                       'processed_root': saved.corpus.parent.parent, 'build_version': 'full_v1',
                       'threads': 2, 'preflight_only': False, 'retrieval_run_dir': saved.source})
    assert matrix.run(args)['status'] == 'COMPLETE'
    assert len(saved.calls) == 1
    matrix.run(args)
    assert len(saved.calls) == 1
    args.output_dir = saved.source / 'inside-source'
    with pytest.raises(ValueError, match='overlap'):
        matrix.run(args)
    args.output_dir = Path.cwd() / 'another'
    args.backends = ['r4']
    with pytest.raises(ValueError, match='does not contain'):
        matrix.run(args)
    assert len(saved.calls) == 1


def test_matrix_validates_all_selected_cases_before_first_call(saved):
    from scripts.narrative.evaluation import run_rag_comparison as matrix

    (saved.folder / 'cases/AAPL-SUP-02/result.json').unlink()
    args = Namespace(**{**vars(saved.args), 'output_dir': Path.cwd() / 'matrix',
                       'all_cases': True, 'strategies': ['C1'], 'backends': ['bm25'],
                       'processed_root': saved.corpus.parent.parent, 'build_version': 'full_v1',
                       'threads': 2, 'preflight_only': False, 'retrieval_run_dir': saved.source})
    with pytest.raises(FileNotFoundError):
        matrix.run(args)
    assert not saved.calls


def test_empty_saved_candidates_produce_unsupported_without_a_model_call(saved):
    from scripts.narrative.evaluation.run_rag_pilot import run

    cases = [{**c, 'type': 'unsupported', 'source_doc': '', 'gold_snippet': ''} for c in saved.cases]
    csv_file(saved.args.cases, cases)
    csv_file(saved.folder / 'cases.csv', cases)
    config = read_json(saved.folder / 'configuration.json')
    config.update(dataset_sha256=sha(saved.args.cases), selected_cases_sha256=case_identity(cases))
    write_json(saved.folder / 'configuration.json', config)
    for case in cases:
        p = saved.folder / 'cases' / case['id']
        result = read_json(p / 'result.json')
        result.update(status='NO_EVIDENCE', candidate_evidence=[], evidence=[],
                      selection={'returned_count': 0, 'returned_evidence_ids': []})
        result['retrieval'].update(count=0, evidence_ids=[])
        write_json(p / 'result.json', result)
        trace = read_json(p / 'retrieval.json')
        trace.update(support=[], counter=[])
        write_json(p / 'retrieval.json', trace)
        preflight = read_json(p / 'preflight.json')
        preflight['gold_source_block_1based'] = None
        write_json(p / 'preflight.json', preflight)
    run(saved.args)
    result = read_json(saved.args.output_dir / 'cases/AAPL-SUP-01/result.json')
    assert result['status'] == 'NO_EVIDENCE' and result['stance'] == 'UNSUPPORTED'
    assert result['retrieval_reuse']['new_retrieval_requests'] == 0
    assert not saved.calls


def test_mode_cannot_change_inside_an_existing_generation_folder(saved):
    from scripts.narrative.evaluation.run_rag_pilot import run

    run(saved.args)
    saved.args.retrieval_source_dir = None
    with pytest.raises(ValueError, match='configuration/dataset/code differs'):
        run(saved.args)
    assert len(saved.calls) == 1


def test_missing_source_pin_does_not_relabel_existing_generated_work(saved):
    from scripts.narrative.evaluation.run_rag_pilot import run

    run(saved.args)
    (saved.args.output_dir / 'cases/AAPL-SUP-01/retrieval_source.json').unlink()
    with pytest.raises(ValueError, match='Missing saved retrieval provenance'):
        run(saved.args)
    assert len(saved.calls) == 1


@pytest.mark.parametrize('line_ending', [b'\n', b'\r\n'])
def test_audited_migration_preserves_existing_live_mode_only(line_ending):
    import hashlib

    from scripts.narrative.evaluation.resume import (
        SAVED_RETRIEVAL_CODE_MIGRATIONS, _migrated, capture_compatible, matrix_compatible,
    )

    cases = [{'id': 'one', 'claims': 'same'}]
    for name, before, after in SAVED_RETRIEVAL_CODE_MIGRATIONS:
        content = (Path('scripts/narrative/evaluation') / name).read_bytes()
        checkout_bytes = content.replace(b'\r\n', b'\n').replace(b'\n', line_ending)
        checkout_hash = hashlib.sha256(checkout_bytes).hexdigest()
        assert _migrated(name, after, checkout_hash)
        if name == 'run_rag_pilot.py':
            old = {'case_ids': ['one'], 'selected_cases_sha256': case_identity(cases),
                   'code_hashes': {name: before}, 'model': 'same-model'}
            new = {**old, 'code_hashes': {name: after}}
            assert capture_compatible(old, new, cases, cases)
            assert not capture_compatible(old, {**new, 'saved_retrieval': {'source': 'new-mode'}}, cases, cases)
            assert not capture_compatible(old, {**new, 'model': 'different'}, cases, cases)
        else:
            old = {'selected_case_ids': ['one'], 'case_count_per_configuration': 1, 'runner_sha256': before}
            new = {**old, 'runner_sha256': after}
            assert matrix_compatible(old, new)
            assert not matrix_compatible(old, {**new, 'saved_retrieval': {'source': 'new-mode'}})


@pytest.mark.parametrize('name,lf_hash,crlf_hash', sorted(LINE_ENDING_EQUIVALENTS))
def test_reviewed_line_endings_preserve_captures_but_unreviewed_code_is_rejected(name, lf_hash, crlf_hash):
    import hashlib

    from scripts.narrative.evaluation.resume import _migrated, capture_compatible

    # Test reviewed historical versions. A new substantive source revision may
    # intentionally require a fresh run, without making the CI suite invalid.
    cases = [{'id': 'one', 'claims': 'same'}]
    old = {'case_ids': ['one'], 'selected_cases_sha256': case_identity(cases),
           'code_hashes': {name: lf_hash}, 'model': 'same-model'}
    new = {**old, 'code_hashes': {name: crlf_hash}}
    assert capture_compatible(old, new, cases, cases)
    assert capture_compatible(new, old, cases, cases)
    changed = hashlib.sha256((lf_hash + 'unreviewed source').encode()).hexdigest()
    assert not capture_compatible(old, {**new, 'code_hashes': {name: changed}}, cases, cases)
    assert not capture_compatible(old, {**new, 'model': 'different'}, cases, cases)
    assert not _migrated(name, changed, lf_hash)
    assert not _migrated('unrelated.py', lf_hash, crlf_hash)


def test_formatting_migrations_preserve_captures_but_not_changed_settings():
    from scripts.narrative.evaluation.resume import (
        FORMATTING_CODE_MIGRATIONS, LINE_ENDING_EQUIVALENTS, capture_compatible,
    )

    cases = [{'id': 'one', 'claims': 'same'}]
    for name, before, after in FORMATTING_CODE_MIGRATIONS:
        old_crlf = next(crlf for module, lf, crlf in LINE_ENDING_EQUIVALENTS
                        if module == name and lf == before)
        new_crlf = next(crlf for module, lf, crlf in LINE_ENDING_EQUIVALENTS
                        if module == name and lf == after)
        old = {'case_ids': ['one'], 'selected_cases_sha256': case_identity(cases),
               'code_hashes': {name: old_crlf}, 'model': 'same-model'}
        for current_hash in (after, new_crlf):
            new = {**old, 'code_hashes': {name: current_hash}}
            assert capture_compatible(old, new, cases, cases)
            assert not capture_compatible(old, {**new, 'model': 'different'}, cases, cases)
        assert not capture_compatible(old, {**old, 'code_hashes': {name: 'unreviewed'}}, cases, cases)


def test_case_ranges_preserve_dataset_identity_and_reuse_only_requested_case(saved):
    from scripts.narrative.evaluation import run_rag_comparison as matrix
    from scripts.narrative.evaluation.evaluate_rag_comparison import score

    before = saved.args.cases.read_bytes()
    args = Namespace(**{**vars(saved.args), 'output_dir': Path.cwd() / 'matrix-range',
                       'limit': None, 'all_cases': False, 'case_ranges': ['2'],
                       'strategies': ['C1'], 'backends': ['bm25'],
                       'processed_root': saved.corpus.parent.parent, 'build_version': 'full_v1',
                       'threads': 2, 'preflight_only': False, 'retrieval_run_dir': saved.source})
    report = matrix.run(args)
    assert report['planned_case_runs'] == 1 and len(saved.calls) == 1
    config = read_json(args.output_dir / 'matrix_configuration.json')
    assert config['selected_case_ids'] == ['AAPL-SUP-02']
    assert config['dataset_sha256'] == sha(saved.args.cases)
    generated = args.output_dir / 'configurations/C1/bm25/cases'
    assert not (generated / 'AAPL-SUP-01').exists()
    assert (generated / 'AAPL-SUP-02/result.json').exists()
    assert score(args.output_dir, Path.cwd())['case_configuration_runs'] == 1
    matrix.run(args)
    assert len(saved.calls) == 1
    assert saved.args.cases.read_bytes() == before


def test_inclusive_case_ranges_match_requested_32_rows():
    from scripts.narrative.evaluation.run_rag_comparison import select_case_ranges

    cases = [{'id': str(i)} for i in range(1, 109)]
    selected = select_case_ranges(cases, ['1-20', '38-49'])
    assert [int(c['id']) for c in selected] == list(range(1, 21)) + list(range(38, 50))
    assert len(selected) == 32
    assert select_case_ranges(cases, ['38-49', '1-20']) == selected


@pytest.mark.parametrize('ranges', [[], ['0-2'], ['20-1'], ['1-109'], ['1-3', '3-4'], ['a'], ['1:20']])
def test_bad_case_ranges_are_rejected(ranges):
    from scripts.narrative.evaluation.run_rag_comparison import select_case_ranges

    with pytest.raises(ValueError):
        select_case_ranges([{'id': str(i)} for i in range(108)], ranges)


@pytest.mark.parametrize('conflict', ['--limit', '--all-cases'])
def test_case_ranges_are_mutually_exclusive_with_other_selection_flags(conflict):
    from scripts.narrative.evaluation.run_rag_comparison import build_parser

    argv = ['--case-ranges', '1-20', '38-49', conflict]
    if conflict == '--limit':
        argv.append('6')
    with pytest.raises(SystemExit):
        build_parser().parse_args([*argv, '--output-dir', 'unused'])
