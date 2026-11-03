"""Company workers must reuse parent caches and safely share only JSONL writes."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import errno
from pathlib import Path
import threading

import pytest

from test_c2_corpus_blurbs import corpus as corpus
from scripts.narrative.chunking import continue_c2_blurbs as worker


@pytest.fixture
def parent_build(corpus, monkeypatch):
    old_args, calls = corpus
    shared = worker.shared
    workspace = Path.cwd()
    cases = workspace / 'cases.csv'
    cases.write_text('id,claims\n1,test\n', encoding='utf-8')
    sources = {ticker: shared.load_source(workspace, old_args.source_root, ticker, 'blurb')
               for ticker in ('AAA', 'BBB')}
    config = {
        'protocol': shared.PROTOCOL, 'mode': 'blurb', 'source_root': str(old_args.source_root),
        'cases_sha256': shared.file_sha(cases), 'selected_case_ids': ['1'],
        'sources': {ticker: source['hashes'] for ticker, source in sources.items()},
        'embedding_model': 'same-source-model', 'embed': False, 'provider': 'mock',
        'blurb_model': 'test-model', 'blurb_max_tokens': 3000,
        'builder_sha256': shared.file_sha(Path(shared.__file__)),
        'context_helper_sha256': shared.file_sha(Path(shared.context_prefix.__code__.co_filename)),
        'blurb_helper_sha256': shared.file_sha(Path(shared.write_blurb.__code__.co_filename)),
    }
    root = old_args.output_root
    shared.write_json(root / 'enrichment_configuration.json', config)
    shared.write_json(root / 'selected_passages.json', {
        'configuration_sha256': shared.digest(config),
        'selected_chunk_ids': {ticker: [c['chunk_id'] for c in source['payload']['chunks']]
                               for ticker, source in sources.items()},
    })
    args = worker.build_parser().parse_args([
        '--output-root', str(root), '--cases', str(cases), '--companies', 'AAA', 'BBB', '--resume',
    ])
    args.blurb_max_tokens = config['blurb_max_tokens']  # Used only when seeding an original-builder response.
    monkeypatch.setattr(shared, 'shortlist', lambda *a, **k: pytest.fail('Must reuse the saved shortlist'))
    return args, config, sources, calls


def test_reuses_original_cache_and_publishes_original_signature(parent_build):
    args, config, sources, calls = parent_build
    shared = worker.shared
    cache = args.output_root / 'blurb_cache.jsonl'
    shared.generate_blurb(sources['AAA']['payload']['chunks'][0], args, config, cache, {}, set())
    manifest = args.output_root / 'enrichment_configuration.json'
    before = manifest.read_bytes()
    assert len(calls) == 1
    worker.run(args)
    assert len(calls) == 4
    assert manifest.read_bytes() == before
    done, pending = shared.cache_entries(cache)
    assert len(done) == 4 and not pending
    for ticker in sources:
        assert shared.already_built(args.output_root / ticker, sources[ticker], shared.digest(config), False)
    rows = worker.read_events(cache)
    assert sum(row.get('usage', {}).get('cost_usd', 0) for row in rows) == pytest.approx(0.04)
    worker.run(args)
    assert len(calls) == 4


def test_two_disjoint_workers_can_generate_concurrently(parent_build, monkeypatch):
    import finsight.llm.client as client

    args, _, _, calls = parent_build
    original = client.complete
    barrier = threading.Barrier(2)

    def simultaneous(**kwargs):
        barrier.wait(timeout=10)
        return original(**kwargs)

    monkeypatch.setattr(client, 'complete', simultaneous)
    first, second = deepcopy(args), deepcopy(args)
    first.companies, second.companies = ['AAA'], ['BBB']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker.run, part) for part in (first, second)]
        reports = [future.result() for future in futures]
    assert all(report['status'] == 'COMPLETE' for report in reports)
    assert len(calls) == 4
    rows = worker.read_events(args.output_root / 'blurb_cache.jsonl')
    assert len(rows) == len({row['event_id'] for row in rows}) == 8
    done, pending = worker.snapshot(rows)
    assert len(done) == 4 and not pending
    assert sum(row.get('usage', {}).get('cost_usd', 0) for row in rows) == pytest.approx(0.04)
    assert not list((args.output_root / '.workers/locks').glob('*.lock'))


def test_same_company_lock_blocks_a_second_worker(parent_build):
    args, _, _, calls = parent_build
    args.companies = ['AAA']
    with worker.file_lock(args.output_root / '.workers/locks/AAA.lock'):
        with pytest.raises(ValueError, match='Locked:'):
            worker.run(args)
    assert not calls


def test_success_checkpoint_is_recovered_without_repeating_model_call(parent_build):
    args, config, sources, calls = parent_build
    args.companies = ['AAA']
    chunk = sources['AAA']['payload']['chunks'][0]
    key = worker.shared.blurb_key(chunk, config)
    started = {'key': key, 'chunk_id': chunk['chunk_id'], 'status': 'STARTED', 'event_id': 'started'}
    completed = {'key': key, 'chunk_id': chunk['chunk_id'], 'status': 'DONE', 'event_id': 'completed',
                 'blurb': 'Previously generated sentence.', 'usage': {'cost_usd': 0.01}}
    worker.append_event(args.output_root / 'blurb_cache.jsonl', started)
    local = args.output_root / '.workers/AAA/blurb_cache.jsonl'
    worker.append_event(local, started)
    worker.append_event(local, completed)
    worker.run(args)
    assert len(calls) == 1  # Only the second AAA chunk needs generation.
    rows = worker.read_events(args.output_root / 'blurb_cache.jsonl')
    assert sum(row.get('usage', {}).get('cost_usd', 0) for row in rows) == pytest.approx(0.02)
    worker.sync_worker_events(args.output_root, 'AAA')
    assert worker.read_events(args.output_root / 'blurb_cache.jsonl') == rows


def test_unresolved_other_company_does_not_block_disjoint_worker(parent_build):
    args, config, sources, calls = parent_build
    chunk = sources['AAA']['payload']['chunks'][0]
    worker.append_event(args.output_root / 'blurb_cache.jsonl', {
        'key': worker.shared.blurb_key(chunk, config), 'chunk_id': chunk['chunk_id'], 'status': 'FAILED',
    })
    args.companies = ['BBB']
    worker.run(args)
    assert len(calls) == 2
    args.companies = ['AAA']
    with pytest.raises(ValueError, match='Unresolved request for AAA'):
        worker.run(args)
    assert len(calls) == 2


def test_failure_is_checkpointed_and_requires_explicit_retry(parent_build, monkeypatch):
    import finsight.llm.client as client

    args, _, _, calls = parent_build
    args.companies = ['AAA']
    original = client.complete

    def fail(**kwargs):
        raise RuntimeError('simulated provider failure')

    monkeypatch.setattr(client, 'complete', fail)
    with pytest.raises(RuntimeError, match='simulated provider failure'):
        worker.run(args)
    assert worker.read_events(args.output_root / 'blurb_cache.jsonl')[-1]['status'] == 'FAILED'
    assert not (args.output_root / '.workers/locks/AAA.lock').exists()
    with pytest.raises(ValueError, match='Unresolved request'):
        worker.run(args)
    monkeypatch.setattr(client, 'complete', original)
    args.retry_incomplete = True
    worker.run(args)
    assert len(calls) == 2


def test_prepare_only_does_not_generate_or_publish(parent_build):
    args, _, _, calls = parent_build
    args.prepare_only = True
    report = worker.run(args)
    assert report['status'] == 'PREPARED'
    assert not calls
    assert not (args.output_root / 'AAA').exists()
    assert not (args.output_root / 'BBB').exists()


@pytest.mark.parametrize('change', ['dataset', 'builder', 'provider', 'company'])
def test_rejects_parent_configuration_changes(parent_build, monkeypatch, change):
    from types import SimpleNamespace

    args, config, _, calls = parent_build
    if change == 'dataset':
        args.cases.write_text('changed dataset', encoding='utf-8')
    elif change == 'builder':
        config['builder_sha256'] = 'unreviewed-revision'
        worker.shared.write_json(args.output_root / 'enrichment_configuration.json', config)
    elif change == 'provider':
        monkeypatch.setattr('finsight.config.get_settings', lambda: SimpleNamespace(llm_provider='other'))
    else:
        args.companies = ['UNKNOWN']
    with pytest.raises(ValueError):
        worker.run(args)
    assert not calls


@pytest.mark.parametrize('lock_disappears', [False, True])
def test_windows_access_denied_during_lock_contention_is_retried(tmp_path, monkeypatch, lock_disappears):
    path = tmp_path / 'cache.lock'
    path.write_text('another-writer', encoding='utf-8')
    real_open = worker.os.open
    attempts = []

    def busy_open(target, flags, *args, **kwargs):
        if Path(target) == path:
            attempts.append(target)
            if len(attempts) == 1:
                if lock_disappears:
                    path.unlink()  # Owner released before the error handler checks exists().
                raise PermissionError(errno.EACCES, 'Sharing conflict', str(path))
        return real_open(target, flags, *args, **kwargs)

    def release_holder(delay):
        if path.exists():
            path.unlink()

    monkeypatch.setattr(worker, 'windows_sharing_error', lambda exc: True)
    monkeypatch.setattr(worker.os, 'open', busy_open)
    monkeypatch.setattr(worker.time, 'sleep', release_holder)
    with worker.file_lock(path, timeout=10):
        assert path.exists()
    assert len(attempts) == 2 and not path.exists()


def test_windows_permission_denial_without_a_lock_fails_after_one_race_retry(tmp_path, monkeypatch):
    path = tmp_path / 'cache.lock'
    attempts = []
    real_open = worker.os.open

    def denied(target, flags, *args, **kwargs):
        if Path(target) == path:
            attempts.append(target)
            raise PermissionError(errno.EACCES, 'Actual access denied', str(path))
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(worker, 'windows_sharing_error', lambda exc: True)
    monkeypatch.setattr(worker.os, 'open', denied)
    monkeypatch.setattr(worker.time, 'sleep', lambda delay: None)
    with pytest.raises(PermissionError, match='Actual access denied'):
        with worker.file_lock(path, timeout=10):
            pytest.fail('Access denial must not be bypassed')
    assert len(attempts) == 2


def test_windows_busy_lock_timeout_does_not_delete_another_owners_lock(tmp_path, monkeypatch):
    path = tmp_path / 'cache.lock'
    path.write_text('another-writer', encoding='utf-8')
    real_open = worker.os.open

    def denied(target, flags, *args, **kwargs):
        if Path(target) == path:
            raise PermissionError(errno.EACCES, 'Busy', str(path))
        return real_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(worker, 'windows_sharing_error', lambda exc: True)
    monkeypatch.setattr(worker.os, 'open', denied)
    with pytest.raises(ValueError, match='Locked:'):
        with worker.file_lock(path, timeout=0):
            pytest.fail('Busy lock must not be acquired')
    assert path.read_text(encoding='utf-8') == 'another-writer'


def test_windows_lock_release_retries_temporary_sharing_denial(tmp_path, monkeypatch):
    path = tmp_path / 'cache.lock'
    real_unlink = Path.unlink
    attempts = []

    def busy_unlink(target, *args, **kwargs):
        if target == path:
            attempts.append(target)
            if len(attempts) == 1:
                raise PermissionError(errno.EACCES, 'Temporary sharing denial', str(path))
        return real_unlink(target, *args, **kwargs)

    monkeypatch.setattr(worker, 'windows_sharing_error', lambda exc: True)
    monkeypatch.setattr(Path, 'unlink', busy_unlink)
    monkeypatch.setattr(worker.time, 'sleep', lambda delay: None)
    with worker.file_lock(path):
        assert path.exists()
    assert len(attempts) == 2 and not path.exists()


def test_response_saved_before_shared_write_failure_is_not_generated_again(parent_build, monkeypatch):
    args, config, sources, calls = parent_build
    chunk = sources['AAA']['payload']['chunks'][0]
    real_append = worker.append_event
    shared_cache = args.output_root / 'blurb_cache.jsonl'

    def fail_shared_done(path, event):
        if path == shared_cache and event['status'] == 'DONE':
            raise PermissionError(errno.EACCES, 'Simulated shared cache lock failure', str(path))
        return real_append(path, event)

    monkeypatch.setattr(worker, 'append_event', fail_shared_done)
    with pytest.raises(PermissionError):
        worker.generate_blurb(chunk, args, config, {}, set())
    assert len(calls) == 1
    local = worker.read_events(args.output_root / '.workers/AAA/blurb_cache.jsonl')
    assert local[-1]['status'] == 'DONE' and local[-1]['blurb']
    monkeypatch.setattr(worker, 'append_event', real_append)
    done, pending = worker.sync_worker_events(args.output_root, 'AAA')
    assert worker.generate_blurb(chunk, args, config, done, pending) == local[-1]['blurb']
    assert len(calls) == 1
    assert sum(row.get('usage', {}).get('cost_usd', 0)
               for row in worker.read_events(shared_cache)) == pytest.approx(0.01)
