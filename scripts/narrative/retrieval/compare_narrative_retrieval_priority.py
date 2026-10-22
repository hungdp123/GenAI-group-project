"""Isolated development-only R1-R4 x P0-P2 diagnostic on token-valid C4.

Reuses frozen Step 4 BM25 indexes/aliases, local Qdrant, Sentence Transformers,
existing fusion/reranking and Step 4 source-bundle evaluation. No production
index, heldout ranking, paid API, or final strategy selection is involved.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import shutil
import statistics
import time
from datetime import date
from pathlib import Path

from finsight.agents.narrative.corpus.processing.corpus import file_sha256
from finsight.agents.narrative.corpus.processing.index import search_index
from finsight.agents.narrative.retrieval.qdrant_index import build_qdrant_index, load_qdrant_index, search_qdrant, reciprocal_rank_fusion
from finsight.agents.narrative.retrieval.retrieval_experiments import SentenceTransformersBackend, SentenceTransformersReranker, _diverse
from finsight.agents.narrative.retrieval.source_priority import assign_priority, apply_bounded_priority, sufficiency_diagnostic, unchanged_trace
from scripts.narrative.chunking.compare_narrative_chunking import evaluate, evidence_bundles, summarize
from scripts.narrative.experiments.prepare_narrative_experiment import identity, write_json
from scripts.narrative.datasets.prepare_narrative_labels import jsonl, write_jsonl


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def milliseconds(start: float) -> float:
    return (time.perf_counter() - start) * 1000


class StrictEmbedding:
    def __init__(self, backend):
        self.base = backend
        self.model_name = backend.model_name
        self.document_count = 0
        self.query_count = 0
        self.max_tokens = 0

    def check(self, texts):
        ids = self.base.model.tokenizer(list(texts), add_special_tokens=True, truncation=False)['input_ids']
        counts = [len(i) for i in ids]
        self.max_tokens = max([self.max_tokens] + counts)
        if max(counts, default=0) > self.base.model.max_seq_length:
            raise RuntimeError('Actual embedding input exceeds effective model limit; truncation prohibited')

    def encode_documents(self, texts):
        vectors = []
        model = self.base.model
        encode = getattr(model, 'encode_document', model.encode)
        for start in range(0, len(texts), 512):
            batch = list(texts[start:start + 512])
            self.check(batch)
            vectors.extend(encode(batch, batch_size=64, normalize_embeddings=True, show_progress_bar=False).tolist())
            self.document_count += len(batch)
            if start % 2048 == 0 or start + 512 >= len(texts):
                print(f'  embedded {min(start + 512, len(texts))}/{len(texts)} chunks', flush=True)
        return vectors

    def encode_query(self, text):
        self.check([text])
        self.query_count += 1
        return self.base.encode_query(text)


class Features:
    """Cache primitives for quality runs; independent latency probes clear them."""
    def __init__(self, payload, vector, backend, ticker, cutoff):
        self.payload, self.vector, self.backend = payload, vector, backend
        self.ticker, self.cutoff = ticker, date.fromisoformat(cutoff)
        self.cache, self.events = {}, []

    def get(self, query, kind):
        key = (query, kind)
        if key in self.cache:
            self.events.append({'kind': kind, 'query_sha256': identity(query), 'cache_hit': True, 'ms': 0})
            return self.cache[key]
        started = time.perf_counter()
        if kind == 'bm25':
            result = search_index(self.payload, query, len(self.payload['chunks']), self.cutoff, self.ticker)
        else:
            result = search_qdrant(self.vector, self.payload, self.backend.encode_query(query),
                                    top_k=len(self.payload['chunks']), cutoff_date=self.cutoff, company_ticker=self.ticker)
        self.events.append({'kind': kind, 'query_sha256': identity(query), 'cache_hit': False, 'ms': milliseconds(started)})
        self.cache[key] = result
        return result


class Methods:
    def __init__(self, features, reranker, priorities, config):
        self.features, self.reranker, self.priorities, self.config = features, reranker, priorities, config
        self.score_cache = {}
        self.pair_calls, self.max_pair_tokens = 0, 0

    def candidates(self, query, method, scope=None):
        k = self.config['candidate_k']
        def scoped(kind):
            ranking = self.features.get(query, kind)
            if scope:
                ranking = [h for h in ranking if self.priorities[h['chunk_id']]['priority'] == scope]
            return ranking[:k]
        if method == 'R1':
            return scoped('bm25')
        if method == 'R2':
            return scoped('dense')
        return reciprocal_rank_fusion(scoped('bm25'), scoped('dense'), top_k=k, rrf_k=self.config['rrf_k'])

    def rerank(self, query, candidates, cache=True):
        missing = [c for c in candidates if not cache or (query, c['chunk_id']) not in self.score_cache]
        values = {}
        if missing:
            tokenizer = self.reranker.model.tokenizer
            token_ids = tokenizer([query] * len(missing), [c['text'] for c in missing], truncation=False, add_special_tokens=True)['input_ids']
            maximum = max(map(len, token_ids), default=0)
            self.max_pair_tokens = max(self.max_pair_tokens, maximum)
            limit = int(self.reranker.model.max_length or tokenizer.model_max_length)
            if maximum > limit:
                raise RuntimeError(f'Cross-encoder pair exceeds limit {limit}; truncation prohibited')
            scores = self.reranker.score(query, [c['text'] for c in missing])
            if len(scores) != len(missing):
                raise RuntimeError('Cross-encoder score count differs')
            self.pair_calls += len(missing)
            values = {c['chunk_id']: float(s) for c, s in zip(missing, scores)}
            if cache:
                self.score_cache.update({(query, cid): score for cid, score in values.items()})
        ranked = [{**c, 'retrieval_score': self.score_cache[(query, c['chunk_id'])] if cache else values[c['chunk_id']]} for c in candidates]
        ranked.sort(key=lambda c: (-c['retrieval_score'], c['chunk_id']))
        return _diverse(ranked, len(ranked))

    def search(self, query, method, policy, *, score_cache=True):
        begin = time.perf_counter()
        event_start = len(self.features.events)
        stages = []
        if policy == 'P1':
            high = self.candidates(query, method, 'HIGH')
            diagnostic = sufficiency_diagnostic(query, high)
            stages.append({'stage': 'high_first', 'candidate_ids': [h['chunk_id'] for h in high], 'sufficiency': diagnostic})
            counter_query = query + ' ' + self.config['counter_query_suffix']
            counter = self.candidates(counter_query, method)
            stages.append({'stage': 'mandatory_counter_probe_all_priorities', 'query': counter_query,
                           'candidate_ids': [h['chunk_id'] for h in counter], 'semantic_contradictions_identified': None})
            medium = self.candidates(query, method, 'MEDIUM')
            stages.append({'stage': 'expand_medium', 'reason': diagnostic['reason'], 'candidate_ids': [h['chunk_id'] for h in medium]})
            exposed = {h['chunk_id'] for h in high + medium + counter}
            # Global corpus statistics/scores stay unchanged across policies.
            # The final unrestricted relevance pass guarantees baseline breadth.
            # For hybrid RRF, separately truncated High/Medium candidate lists
            # are not additive: fusion can rank an item globally that falls
            # outside both scoped top-k lists. Record that gap instead of
            # treating it as a retrieval failure.
            ranking = self.candidates(query, method)
            baseline_only = [h['chunk_id'] for h in ranking if h['chunk_id'] not in exposed]
            stages.append({'stage': 'original_claim_relevance_order',
                           'reason': 'No semantic sufficiency gate; unrestricted final pass preserves P0 breadth, including hybrid-fusion candidates omitted by separately truncated priority views',
                           'exposed_unique_candidates': len(exposed),
                           'added_by_unrestricted_relevance_pass': len(baseline_only),
                           'added_candidate_ids': baseline_only})
        else:
            ranking = self.candidates(query, method)
        if method == 'R4':
            ranking = self.rerank(query, ranking, cache=score_cache)
        if policy == 'P2':
            ranking, trace = apply_bounded_priority(query, ranking, self.priorities, self.config)
        else:
            trace = unchanged_trace(query, ranking, self.priorities, policy)
        return ranking[:self.config['top_k']], {'method': method, 'policy': policy, 'stages': stages,
                'ranking': trace, 'component_events': self.features.events[event_start:], 'elapsed_ms': milliseconds(begin)}


def model_snapshot(name: str) -> Path:
    snapshots = Path.home() / '.cache/huggingface/hub' / ('models--' + name.replace('/', '--')) / 'snapshots'
    candidates = sorted(snapshots.iterdir())
    if len(candidates) != 1:
        raise RuntimeError('Choose a pinned local model snapshot explicitly: ' + name)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='20261018_all16_v1')
    parser.add_argument('--experiment-id', default='c4_dev_v1')
    parser.add_argument('--config', default='docs/narrative/config/narrative_priority_config_v1.json')
    parser.add_argument('--allow-draft', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    import re
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', s) for s in (args.run_id, args.experiment_id)):
        parser.error('Invalid version ID')
    root = Path.cwd().resolve()
    run = root / 'data/narrative/experiments/narrative_rag' / args.run_id
    output = run / 'retrieval_priority' / args.experiment_id
    if args.verify:
        m = read(output / 'experiment_manifest.json')
        failures = [f['path'] for f in m['files'] if not (output / f['path']).is_file() or file_sha256(output / f['path']) != f['sha256']]
        print(json.dumps({'status': 'verified' if not failures else 'failed', 'files_checked': len(m['files']), 'failures': failures}, indent=2))
        raise SystemExit(bool(failures))
    if output.exists() and not args.resume:
        parser.error('Experiment exists; use --resume/--verify or a new --experiment-id')
    dataset = run / 'labels/evidence_labels_v3'
    labels = read(dataset / 'dataset_manifest.json')
    if not labels['evaluation_ready'] and not args.allow_draft:
        parser.error('Draft labels require --allow-draft; no final selection is permitted')
    config = read(root / args.config)
    if config['p1_sufficiency'] != 'semantic_review_required_always_expand' or config['top_k'] != 10:
        parser.error('This diagnostic supports conservative P1 and top_k=10')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    torch.set_num_threads(4)
    embed_path = model_snapshot('sentence-transformers/all-MiniLM-L6-v2')
    rerank_path = model_snapshot('cross-encoder/ms-marco-MiniLM-L-6-v2')
    foundation = read(run / 'foundation_manifest.json')
    prior = run / 'comparisons/bm25_dev_v1'
    prior_report = read(prior / 'comparison.json')
    assert not prior_report['winner_selected'] and prior_report['configuration']['source_build_identity'] == foundation['build_identity']
    for folder, name in ((prior, 'comparison_manifest.json'), (run / 'validation/c4_v1', 'validation_manifest.json')):
        assert all(file_sha256(folder / f['path']) == f['sha256'] for f in read(folder / name)['files'])
    assert labels['foundation_build_identity'] == foundation['build_identity']
    assert file_sha256(dataset / 'dataset_manifest.json') == prior_report['configuration']['dataset_manifest_sha256']
    for f in labels['files']:
        assert file_sha256(dataset / f['path']) == f['sha256']
    questions, evidence = jsonl(dataset / 'dev/questions.jsonl'), jsonl(dataset / 'dev/evidence_labels.jsonl')
    assert len(questions) == 92 and all(q['split'] == 'dev' for q in questions)
    output.mkdir(parents=True, exist_ok=True)
    immutable = {'source_build_identity': foundation['build_identity'], 'dataset_manifest_sha256': file_sha256(dataset / 'dataset_manifest.json'),
                 'questions_sha256': file_sha256(dataset / 'dev/questions.jsonl'), 'evidence_sha256': file_sha256(dataset / 'dev/evidence_labels.jsonl'),
                 'chunking': 'C4 temporary token-valid diagnostic input; not selected winner', 'policies': config,
                 'embedding_snapshot': str(embed_path), 'reranker_snapshot': str(rerank_path),
                 'embedding_revision': embed_path.name, 'reranker_revision': rerank_path.name,
                 'embedding_weight_sha256': file_sha256(embed_path / 'model.safetensors'),
                 'reranker_weight_sha256': file_sha256(rerank_path / 'model.safetensors'),
                 'candidate_k': config['candidate_k'], 'query_field': 'unchanged claim; P1 additional configured counter probe',
                 'hybrid': 'existing Qdrant RRF k=60 with two BM25 anchors; no extra BM25+cross-encoder arm',
                 'filters': 'reuse identical Step 4 cutoff/company/eligibility-filtered indexes and exact-text aliases',
                 'heldout_evaluated': False, 'final_selection_permitted': False,
                 'P1_implementation_limit': 'All High/Medium views share exhaustive primitive scores; tests search order/traces, not server-side early-search savings',
                 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__, 'threads': 4, 'device': 'cpu'}}
    config_path = output / 'configuration.json'
    if config_path.exists():
        assert read(config_path) == immutable, 'Resume configuration differs'
    else:
        write_json(config_path, immutable)
        for p in (Path(__file__).resolve(), root / 'finsight/agents/narrative/retrieval/source_priority.py', root / 'finsight/agents/narrative/retrieval/qdrant_index.py',
                  root / 'finsight/agents/narrative/retrieval/retrieval_experiments.py', root / 'finsight/agents/narrative/corpus/processing/index.py', root / 'scripts/narrative/chunking/compare_narrative_chunking.py'):
            dest = output / 'code' / p.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    print('Loading pinned local embedding and cross-encoder models (offline, CPU, 4 threads)', flush=True)
    base = SentenceTransformersBackend(str(embed_path))
    base.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    backend = StrictEmbedding(base)
    reranker = SentenceTransformersReranker(str(rerank_path))
    print(f'Embedding limit {base.model.max_seq_length}; reranker pair limit {reranker.model.max_length}', flush=True)
    if base.model.max_seq_length != 256:
        raise RuntimeError('Embedding limit differs from validated C4 build')
    started = time.perf_counter()
    all_metrics, all_indexes, all_latency, all_pools = [], [], [], []
    for n, ticker in enumerate(foundation['companies'], 1):
        company = output / 'companies' / ticker
        checkpoint = company / 'checkpoint.json'
        if checkpoint.exists():
            saved = read(checkpoint)
            assert saved['configuration_sha256'] == file_sha256(config_path)
            assert all(file_sha256(company / f['path']) == f['sha256'] for f in saved['files'])
            all_metrics.extend(jsonl(company / 'per_query.jsonl'))
            all_indexes.append(read(company / 'indexing_report.json'))
            all_latency.extend(jsonl(company / 'latency_probes.jsonl'))
            all_pools.extend(jsonl(company / 'candidate_pool.jsonl'))
            print(f'[{n}/16] {ticker}: reusing verified checkpoint', flush=True)
            continue
        print(f'[{n}/16] {ticker}: preparing C4 indexes', flush=True)
        company.mkdir(parents=True, exist_ok=True)
        qrows = sorted((q for q in questions if q['ticker'] == ticker), key=lambda q: q['query_id'])
        cutoffs = {q['cutoff_date'] for q in qrows}
        assert cutoffs == {'2025-07-31'}
        cutoff = next(iter(cutoffs))
        input_dir = prior / 'indexes/C4' / ticker / cutoff
        payload = read(input_dir / 'index.json')
        original_chunks = {c['chunk_id']: c for c in jsonl(run / 'validation/c4_v1/arms/C4' / ticker / 'chunks.jsonl')}
        alias_rows = jsonl(input_dir / 'source_aliases.jsonl')
        aliases = {a['representative_chunk_id']: [original_chunks[i['chunk_id']] for i in a['aliases']] for a in alias_rows}
        priorities = {c['chunk_id']: assign_priority(c, aliases[c['chunk_id']], config) for c in payload['chunks']}
        write_jsonl(company / 'priority_assignments.jsonl', [{'chunk_id': cid, **p} for cid, p in priorities.items()])
        maps = {m['chunk_id']: m for m in jsonl(run / 'validation/c4_v1/arms/C4' / ticker / 'source_map.jsonl')}
        rows = jsonl(run / 'parsed' / ticker / 'documents.jsonl')
        vectors_dir = company / 'qdrant'
        build_start = time.perf_counter()
        vector_reused = (vectors_dir / 'qdrant_metadata.json').exists()
        if not vector_reused:
            if (vectors_dir / 'storage').exists():
                raise RuntimeError('Unfinished vector build retained; use a new experiment version rather than overwrite it')
            build_qdrant_index(payload, vectors_dir, backend)
        vector = load_qdrant_index(vectors_dir, payload, expected_model_name=backend.model_name, expected_dimension=384)
        build_ms = milliseconds(build_start)
        methods = Methods(Features(payload, vector, backend, ticker, cutoff), reranker, priorities, config)
        metrics, latency, ranked_cases, traces, pool = [], [], [], [], {}
        try:
            for qi, q in enumerate(qrows, 1):
                bundles = evidence_bundles(q, evidence)
                for method in ('R1', 'R2', 'R3', 'R4'):
                    for policy in ('P0', 'P1', 'P2'):
                        ranking, trace = methods.search(q['claim'], method, policy)
                        result = evaluate(q, bundles, ranking, aliases, maps, rows)
                        result.update(method=method, policy=policy, search_ms=trace['elapsed_ms'],
                                      timing_scope='feature/score-cache-dependent ranking assembly; not cold end-to-end latency')
                        metrics.append(result)
                        ranked_cases.append({'query_id': q['query_id'], 'ticker': ticker, 'query': q['claim'], 'cutoff': cutoff,
                                             'method': method, 'policy': policy, 'ranking': ranking})
                        traces.append({'query_id': q['query_id'], **trace})
                        assert len({h['text_sha256'] for h in ranking}) == len(ranking)
                        for h in ranking:
                            assert h['company_ticker'] == ticker and h['publication_date'] <= cutoff and h['citation_span']
                            key = (q['query_id'], h['text_sha256'])
                            candidate = pool.setdefault(key, {'query_id': q['query_id'], 'ticker': ticker, 'text': h['text'], 'judgment_status': 'UNJUDGED',
                                                               'sources': [a['chunk_id'] for a in aliases[h['chunk_id']]], 'appearances': []})
                            candidate['appearances'].append({'method': method, 'policy': policy, 'rank': next(i for i, r in enumerate(ranking, 1) if r['chunk_id'] == h['chunk_id'])})
                print(f'  {ticker} development claim {qi}/{len(qrows)}: 12 configurations complete', flush=True)
            # One fixed, label-independent query per company; cold primitive cache
            # then warm primitive cache. Both run real reranking, not score cache.
            probe = qrows[0]
            for method in ('R1', 'R2', 'R3', 'R4'):
                for policy in ('P0', 'P1', 'P2'):
                    probe_methods = Methods(Features(payload, vector, backend, ticker, cutoff), reranker, priorities, config)
                    for condition in ('cold_query_features', 'warm_query_features'):
                        ranking, trace = probe_methods.search(probe['claim'], method, policy, score_cache=False)
                        latency.append({'ticker': ticker, 'query_id': probe['query_id'], 'method': method, 'policy': policy,
                                        'condition': condition, 'elapsed_ms': trace['elapsed_ms'], 'component_events': trace['component_events'],
                                        'ranking_ids': [h['chunk_id'] for h in ranking], 'model_weights_loaded': True, 'real_reranking': method == 'R4'})
                    assert latency[-1]['ranking_ids'] == latency[-2]['ranking_ids']
                    methods.pair_calls += probe_methods.pair_calls
                    methods.max_pair_tokens = max(methods.max_pair_tokens, probe_methods.max_pair_tokens)
            assert len(metrics) == len(qrows) * 12 and len(latency) == 24
            write_jsonl(company / 'per_query.jsonl', metrics)
            write_jsonl(company / 'rankings.jsonl', ranked_cases)
            write_jsonl(company / 'policy_traces.jsonl', traces)
            write_jsonl(company / 'latency_probes.jsonl', latency)
            write_jsonl(company / 'candidate_pool.jsonl', list(pool.values()))
            # Persist compact complete primitive rankings for reproducible policy traces.
            with gzip.open(company / 'primitive_rankings.jsonl.gz', 'wt', encoding='utf-8') as handle:
                for (query, kind), ranking in methods.features.cache.items():
                    handle.write(json.dumps({'query': query, 'kind': kind, 'ranking': [{'chunk_id': h['chunk_id'], 'score': h['retrieval_score']} for h in ranking]}) + '\n')
            write_json(company / 'indexing_report.json', {'ticker': ticker, 'chunks': len(payload['chunks']), 'vector_count': len(vector.chunk_ids),
                'corpus_fingerprint': payload['corpus_fingerprint'], 'bm25_input_sha256': file_sha256(input_dir / 'index.json'),
                'alias_input_sha256': file_sha256(input_dir / 'source_aliases.jsonl'), 'qdrant_build_or_load_ms': build_ms, 'vector_reused': vector_reused,
                'vector_bytes': sum(p.stat().st_size for p in vectors_dir.rglob('*') if p.is_file()),
                'unknown_type_chunks': sum(p['needs_type_review'] for p in priorities.values()), 'cross_encoder_pair_inferences': methods.pair_calls,
                'maximum_cross_encoder_pair_tokens': methods.max_pair_tokens})
        finally:
            vector.client.close()
        files = [{'path': p.relative_to(company).as_posix(), 'sha256': file_sha256(p)} for p in sorted(company.rglob('*')) if p.is_file() and p != checkpoint and p.name != '.lock']
        write_json(checkpoint, {'status': 'complete', 'configuration_sha256': file_sha256(config_path), 'files': files})
        all_metrics.extend(metrics)
        all_indexes.append(read(company / 'indexing_report.json'))
        all_latency.extend(latency)
        all_pools.extend(pool.values())
        print(f'[{n}/16] {ticker}: checkpoint saved ({len(metrics)} cases)', flush=True)
    assert len(all_metrics) == 1104 and len(all_latency) == 384
    summaries = {}
    for method in ('R1', 'R2', 'R3', 'R4'):
        for policy in ('P0', 'P1', 'P2'):
            selected = [r for r in all_metrics if r['method'] == method and r['policy'] == policy]
            summary = summarize(selected)
            summary['cached_assembly_median_ms'] = summary.pop('search_median_ms')
            summary['cached_assembly_p95_ms'] = summary.pop('search_p95_ms')
            for condition in ('cold_query_features', 'warm_query_features'):
                samples = [r['elapsed_ms'] for r in all_latency if r['method'] == method and r['policy'] == policy and r['condition'] == condition]
                summary[condition + '_median_ms'] = statistics.median(samples)
                summary[condition + '_n'] = len(samples)
            summaries[method + '_' + policy] = summary
    # Exhaustive P1 must preserve P0 results for every method; prove, don't assume.
    identical = 0
    moves = 0
    for ticker in foundation['companies']:
        saved = jsonl(output / 'companies' / ticker / 'rankings.jsonl')
        cases = {(r['query_id'], r['method'], r['policy']): [h['chunk_id'] for h in r['ranking']] for r in saved}
        for q in (q for q in questions if q['ticker'] == ticker):
            for method in ('R1', 'R2', 'R3', 'R4'):
                assert cases[q['query_id'], method, 'P0'] == cases[q['query_id'], method, 'P1']
                identical += 1
                moves += cases[q['query_id'], method, 'P0'] != cases[q['query_id'], method, 'P2']
    write_jsonl(output / 'per_query.jsonl', all_metrics)
    with (output / 'per_query.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0])); writer.writeheader(); writer.writerows(all_metrics)
    write_jsonl(output / 'latency_probes.jsonl', all_latency)
    write_jsonl(output / 'development_candidate_pool.jsonl', all_pools)
    write_json(output / 'indexing_report.json', all_indexes)
    result = {'status': 'development_diagnostic_complete_formal_acceptance_pending', 'configuration': immutable,
              'companies': 16, 'development_claims': 92, 'retrieval_cases': len(all_metrics), 'heldout_evaluated': 0,
              'vector_count': sum(i['vector_count'] for i in all_indexes), 'summaries': summaries,
              'P1_equals_P0_cases': identical, 'P2_changed_top10_cases': moves, 'candidate_pool_size': len(all_pools),
              'winner_selected': False, 'labels_reviewed': labels['evaluation_ready'], 'precision_at_5': None, 'ndcg_at_10': None,
              'semantic_sufficiency_validated': False, 'semantic_near_duplicate_groups_reviewed': False,
              'api_requests': 0, 'api_cost_usd': 0, 'local_compute_cost_usd': None,
              'actual_embedding_inputs_this_invocation': {'documents': backend.document_count, 'queries': backend.query_count, 'max_tokens': backend.max_tokens},
              'elapsed_seconds_this_invocation': round(time.perf_counter() - started, 2),
              'strata': {origin: {key: summarize([r for r in all_metrics if r['origin'] == origin and r['method'] + '_' + r['policy'] == key]) for key in summaries}
                        for origin in sorted({q['origin'] for q in questions})}}
    write_json(output / 'comparison.json', result)
    prefix = output.relative_to(root).as_posix()
    table = '\n'.join(f"| {key} | {s['recall@5']:.2%} | {s['recall@10']:.2%} | {s['direct_counter_recall@10']:.2%} | {s['mrr@10']:.4f} | {s['cold_query_features_median_ms']:.1f} | {s['warm_query_features_median_ms']:.1f} |" for key, s in summaries.items())
    md = f"""# Step 5 - Retrieval and document priority: development diagnostic

Executed across **16 companies / 92 development claims / 1,104 cases**.
Status: **diagnostic complete; formal acceptance and final selection pending**.
No heldout quality cases or paid API requests. Production indexes remain unchanged.

## Representation and controls

C4 is a **temporary engineering input**, chosen because all chunks fit the
current 256-token embedding limit. It did not win Step 4 and is not promoted.
Reuse the same 679 cutoff-eligible documents, exact-text aliases, unchanged claim
queries, source-bundle labels, cutoff 2025-07-31, candidate K=50 and top K=5/10.
All 12 arms share identical C4 IDs and per-company BM25/Qdrant fingerprints.

R1 reuses `index.search_index`; R2 reuses persistent local Qdrant cosine search;
R3 reuses Qdrant RRF k=60 with two lexical anchors; R4 reranks R3's 50 candidates
with the pinned local cross-encoder and existing citation-location diversification.
This is the declared Qdrant hybrid variant, distinct from the older pure-RRF
experimental helper. No extra BM25+cross-encoder arm is computed.

## Provisional seed-reference results

| Arm | Recall@5 | Recall@10 | Direct-counter Recall@10 | MRR@10 | Cold-feature median ms | Warm-feature median ms |
|---|---:|---:|---:|---:|---:|---:|
{table}

Source-span bundle scoring is reused from Step 4. Header hits alone earn no fact
credit; multi-result completion is separate from MRR. Draft support/direct-counter
denominators and original versus source-drafted claim strata are in comparison.json.
Unjudged candidates remain unknown. Precision@5/nDCG are **unmeasured**, not zero.

## Priority policy and limitations

Document priority is separate from source tier. Underlying document types govern
exhibits: earnings releases High; presentation exhibits Medium; unknown exhibits
remain searchable at Medium and require content review. Main 8-K/6-K filings
are High. Prepared remarks are Medium. Exact-text aliases inherit the highest
known underlying-type priority, with each source assignment preserved.

P0 retains relevance order. P1 visits High, executes a configured counter probe
across all priorities, then expands Medium because semantic sufficiency remains
UNKNOWN. Keyword/value/unit presence cannot certify entailment, period, scope
or absence of counter-evidence. It returns original-claim relevance order after
full expansion. P1 equals P0 in **{identical}/368** method/claim cases, as expected.
It cannot demonstrate selective expansion or early-stopping cost savings yet.
Primitive scores are global, exhaustive and reused across High/Medium views;
this is a search-order diagnostic, not server-side High-only search optimization.
Counter-probe candidates are traced, not asserted to be actual contradictions.

P2 promotes High only across near-ties: normalized relevance gap <=0.03,
maximum two positions, and no worse explicit query-period match. Weights/bounds
are declared initial configuration, not optimized. It changed top-10 ordering
in **{moves}/368** method/claim cases. Period matching is a lexical guard, not
semantic reporting-period verification. Raw scores, priorities, rank changes and
selection/expansion reasons are saved for every candidate.

Exact-text evidence occupies one slot with all citable aliases retained. Ambiguous
semantic release/exhibit duplicates are not automatically merged; their review is
still an acceptance dependency. No claim that near-duplicate handling is complete.

## Timing, overhead and cost

Quality runs reuse primitive and pair-score caches; their cached assembly times
are not cold end-to-end latency. Separate probes use the first query ID per
company (label-independent selection): 16 claims x 12 configurations x cold/warm
= **384 probes**. Cold means cleared query-feature caches, warm reuses those
features. Both use already loaded model weights/OS caches and real uncached
cross-encoder inference. These are not machine-start cold timings. Each method
is measured separately; shared multi-method elapsed time is never assigned to
each method. Model-load, index-build, and query timings are distinct.

Qdrant: **{result['vector_count']:,} vectors** in isolated company stores.
Index time/storage and pair inference counts are in indexing_report.json.
Tokenization rejects oversized document/query/pair inputs before model calls.
API cost: $0; local CPU cost in dollars unmeasured. Device CPU, 4 Torch threads;
embedding revision `{embed_path.name}`, reranker revision `{rerank_path.name}`.
Weights, environment, corpus, labels, policies and code are recorded.

## Outputs and inspection - paths from project root

- `{prefix}/comparison.json`: all 12 aggregate results and strata.
- `{prefix}/per_query.csv`: all 1,104 cases.
- `{prefix}/companies/<TICKER>/rankings.jsonl`: passages per method/policy.
- `{prefix}/companies/<TICKER>/policy_traces.jsonl`: searches, sufficiency gaps, ranks and reasons.
- `{prefix}/companies/<TICKER>/priority_assignments.jsonl`: underlying-type and alias priorities.
- `{prefix}/companies/<TICKER>/qdrant/`: isolated persistent vectors and metadata.
- `{prefix}/latency_probes.jsonl`: 384 independent method/policy probes.
- `{prefix}/indexing_report.json`: vector build/storage and pair inference work.
- `{prefix}/development_candidate_pool.jsonl`: {len(all_pools):,} unjudged pooled passages.
- `{prefix}/configuration.json`, `code/`, `experiment_manifest.json`: reproducibility and integrity.

```powershell
Get-Content -Raw {prefix}/comparison.json
Import-Csv {prefix}/per_query.csv | Where-Object ticker -eq META
python -m scripts.narrative.retrieval.compare_narrative_retrieval_priority --run-id {args.run_id} --experiment-id {args.experiment_id} --verify
```

Resume an interrupted run with the same configuration and verified checkpoints:

```powershell
python -u -m scripts.narrative.retrieval.compare_narrative_retrieval_priority --run-id {args.run_id} --experiment-id {args.experiment_id} --allow-draft --resume
```

## Decision

**No strategy/policy selected or promoted.** Labels remain drafts, C4 is a
temporary representation, semantic context/citation review and duplicate-source
review remain, and P1 has no validated semantic sufficiency component. Review
questions, source-span/context labels and pooled judgments; settle representation
and semantic sufficiency rules; freeze a reviewed development release before a
formal comparison. Reserved evaluation remains untouched. This diagnostic does
not authorize downstream production promotion or a superiority claim.
"""
    (output / 'step_05_retrieval_priority.md').write_text(md, encoding='utf-8')
    inventory = [{'path': p.relative_to(output).as_posix(), 'sha256': file_sha256(p), 'bytes': p.stat().st_size}
                 for p in sorted(output.rglob('*')) if p.is_file() and p.name not in {'experiment_manifest.json', '.lock'}]
    write_json(output / 'experiment_manifest.json', {'files': inventory})
    (run / 'reports/step_05_retrieval_priority.md').write_text(md, encoding='utf-8')
    write_json(run / 'reports/step_05_retrieval_priority.json', result)
    print(json.dumps({'status': result['status'], 'cases': len(all_metrics), 'summaries': summaries}, indent=2))


if __name__ == '__main__':
    main()
