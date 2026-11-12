"""Resumable CPU MFA pseudo-reference evaluation on zero-free ROVER chunks.

Existing Qwen and hybrid predictions are reused. Each selected chunk is aligned
independently using its exact source samples/channel and complete ROVER text.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from compare_mfa_mae import align_words

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prepare(args):
    if (args.output / 'inventory.json').exists():
        return json.loads((args.output / 'inventory.json').read_text())
    started = time.perf_counter()
    batch_root = args.output / 'batches'
    batch_root.mkdir(parents=True, exist_ok=True)
    counts = dict(input_chunks=0, zero_chunks_excluded=0, input_failure_chunks=0,
                  empty_chunks=0, eligible_chunks=0, eligible_words=0,
                  eligible_audio_seconds=0., unchanged_hybrid_chunks=0)
    tasks, batch, sources = [], [], []
    def flush():
        if not batch:
            return
        directory = batch_root / f'b{len(tasks):05d}'
        directory.mkdir(exist_ok=True)
        with gzip.open(directory / 'input.jsonl.gz', 'wt', compresslevel=1) as f:
            for row in batch:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        tasks.append(dict(path=str(directory.resolve()), chunks=len(batch),
                          audio_seconds=sum(r['duration'] for r in batch)))
        batch.clear()
    manifest = json.loads((args.manifest_root / 'preparation.json').read_text())
    required_source = '/hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/Pipeline_github_v2/outputs/production/open_yap_1k/rover/rover/batch_results'
    assert manifest['input'] == required_source
    source_paths = sorted(args.existing.glob('gpu_*/blocks/b*/qwen_predictions.jsonl'))
    assert source_paths
    with gzip.open(args.output / 'excluded_chunks.jsonl.gz', 'wt') as excluded:
        for path in source_paths:
            qhash, hhash = hashlib.sha256(), hashlib.sha256()
            # These historical outputs use internal recording-time word dictionaries.
            hybrid_path = path.with_name('predictions.jsonl')
            with path.open('rb') as qf, hybrid_path.open('rb') as hf:
                for qline, hline in itertools.zip_longest(qf, hf):
                    assert qline is not None and hline is not None
                    qhash.update(qline); hhash.update(hline)
                    q, h = json.loads(qline), json.loads(hline)
                    assert q['id'] == h['id'] and q['text'] == h['text']
                    assert all(q[k] == h[k] for k in ['audio', 'start_sample', 'end_sample', 'channel', 'source_sample_rate'])
                    counts['input_chunks'] += 1
                    reason = None
                    if any(abs(w['end'] - w['start']) <= 1e-7 for w in q['words']):
                        counts['zero_chunks_excluded'] += 1
                        reason = 'qwen_zero_duration_chunk'
                    elif q.get('error') or h.get('error') or (q['text'].strip() and not q['words']):
                        counts['input_failure_chunks'] += 1
                        reason = 'input_alignment_failure'
                    elif not q['text'].strip():
                        counts['empty_chunks'] += 1
                        reason = 'empty_transcript'
                    if reason:
                        excluded.write(json.dumps(dict(id=q['id'], reason=reason)) + '\n')
                        continue
                    assert q['words'] == h['words'], 'Zero-free hybrid differs from Qwen: ' + q['id']
                    assert q['source_json'].startswith(required_source + '/')
                    counts['unchanged_hybrid_chunks'] += 1
                    counts['eligible_chunks'] += 1
                    counts['eligible_words'] += len(q['words'])
                    counts['eligible_audio_seconds'] += q['duration']
                    keep = ['id', 'audio', 'text', 'channel', 'source_sample_rate', 'start_sample',
                            'end_sample', 'chunk_start', 'chunk_end', 'duration', 'source_json', 'words']
                    batch.append({k: q[k] for k in keep})
                    if len(batch) == args.batch_size:
                        flush()
            sources.append(dict(qwen=str(path.resolve()), hybrid=str(hybrid_path.resolve()),
                                qwen_sha256=qhash.hexdigest(), hybrid_sha256=hhash.hexdigest()))
            print(f'Inventory blocks={len(sources)}/{len(source_paths)} selected={counts["eligible_chunks"]}', flush=True)
    flush()
    assert counts['input_chunks'] == manifest['unique_chunks']
    inventory = dict(counts=counts, batches=tasks, sources=sources,
                     source_input=manifest['input'], prepare_seconds=time.perf_counter()-started,
                     filter='Exclude whole chunks with any Qwen zero-duration word; epsilon=1e-7 seconds. Empty and failed inputs reported separately.',
                     hybrid_validation='Every eligible Qwen word dictionary exactly equals its hybrid counterpart.',
                     mfa_configuration=dict(jobs=args.jobs, workers=args.workers, beam=100, retry_beam=400,
                         model=str((ROOT/'models/mfa-english/english_us_arpa.zip').resolve()),
                         dictionary=str((ROOT/'models/mfa-english/dictionary/english_us_arpa.dict').resolve()),
                         model_sha256=hashlib.sha256((ROOT/'models/mfa-english/english_us_arpa.zip').read_bytes()).hexdigest(),
                         dictionary_sha256=hashlib.sha256((ROOT/'models/mfa-english/dictionary/english_us_arpa.dict').read_bytes()).hexdigest()),
                     scope='Pure MFA is run on all eligible evaluation chunks, not on excluded chunks.')
    dump(args.output / 'inventory.json', inventory)
    return inventory


def run_batch(task):
    directory, scratch_root, jobs = Path(task['path']), Path(task['scratch']), task['jobs']
    if (directory / 'summary.json').exists():
        return json.loads((directory / 'summary.json').read_text())
    import soundfile as sf
    import soxr
    started = time.perf_counter()
    scratch = scratch_root / directory.name
    if scratch.exists():
        shutil.rmtree(scratch)  # Only this incomplete task's rebuildable local scratch.
    corpus = scratch / 'corpus'
    corpus.mkdir(parents=True)
    rows = list(lines(directory / 'input.jsonl.gz'))
    failures, prepared = [], set()
    handle, current = None, None
    try:
        for index, row in enumerate(rows):
            try:
                if current != row['audio']:
                    if handle is not None:
                        handle.close()
                    handle = sf.SoundFile(row['audio']); current = row['audio']
                assert handle.samplerate == row['source_sample_rate']
                assert 0 <= row['channel'] < handle.channels
                handle.seek(row['start_sample'])
                samples = handle.read(row['end_sample']-row['start_sample'], dtype='float32', always_2d=True)
                assert len(samples) == row['end_sample']-row['start_sample']
                audio = samples[:, row['channel']]
                if handle.samplerate != 16000:
                    audio = soxr.resample(audio, handle.samplerate, 16000)
                sf.write(corpus / f'c{index:06d}.wav', audio, 16000, subtype='PCM_16')
                (corpus / f'c{index:06d}.lab').write_text(row['text'] + '\n')
                prepared.add(index)
            except Exception as exc:
                failures.append(dict(id=row['id'], stage='audio', reason=repr(exc)))
    finally:
        if handle is not None:
            handle.close()
    print(f'{directory.name}: audio prepared={len(prepared)}/{len(rows)}', flush=True)
    python = ROOT / 'models/mfa/bin/python'
    prefix = python.parent.parent
    env = dict(os.environ)
    env.update(PATH=str(prefix/'bin')+os.pathsep+env.get('PATH',''),
               LD_LIBRARY_PATH=str(prefix/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH',''),
               MFA_ROOT_DIR=str(scratch/'cache'), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [str(python), '-c', 'from montreal_forced_aligner.command_line.mfa import mfa_cli; mfa_cli()',
               'align', str(corpus), str(ROOT/'models/mfa-english/dictionary/english_us_arpa.dict'),
               str(ROOT/'models/mfa-english/english_us_arpa.zip'), str(scratch/'aligned'),
               '--output_format', 'json', '--num_jobs', str(jobs),
               '--temporary_directory', str(scratch/'work'), '--single_speaker', '--use_threading',
               '--no_use_postgres', '--beam', '100', '--retry_beam', '400', '--clean', '--overwrite']
    dump(directory/'command.json', dict(command=command, device='cpu', cuda_visible_devices='',
                                        input_sha256=hashlib.sha256((directory/'input.jsonl.gz').read_bytes()).hexdigest()))
    command_started = time.perf_counter()
    with (directory/'mfa.log').open('w') as log:
        status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=14400, check=False).returncode
    command_seconds = time.perf_counter()-command_started
    if status != 0:
        dump(directory/'failed_command.json', dict(exit_code=status, seconds=command_seconds))
        raise RuntimeError(f'{directory.name}: MFA exit={status}; see {directory}/mfa.log')
    files = {p.stem: p for p in (scratch/'aligned').rglob('*.json')}
    totals = dict(input_chunks=len(rows), audio_prepared=len(prepared), mfa_chunks=0, mfa_words=0,
                  zero_duration_mfa_words=0, invalid_mfa_words=0, oov_mfa_words=0,
                  matched_chunks=0, matched_words=0, excluded_qwen_words=0, excluded_mfa_words=0,
                  onset_abs_sum_ms=0., offset_abs_sum_ms=0., delta_sum_ms=0.)
    thresholds = {str(t): 0 for t in [25, 50, 100, 200]}
    with gzip.open(directory/'mfa_predictions.jsonl.gz', 'wt', compresslevel=3) as predictions, \
         gzip.open(directory/'per_word.jsonl.gz', 'wt', compresslevel=3) as words_out, \
         gzip.open(directory/'matching_audit.jsonl.gz', 'wt', compresslevel=3) as matching:
        for index, row in enumerate(rows):
            if index not in prepared:
                continue
            path = files.get(f'c{index:06d}')
            if path is None:
                failures.append(dict(id=row['id'], stage='mfa', reason='missing_alignment'))
                continue
            alignment = json.loads(path.read_text())
            tiers = [tier for name, tier in alignment['tiers'].items() if name.casefold().endswith('words')]
            if len(tiers) != 1:
                failures.append(dict(id=row['id'], stage='mfa', reason='no_unique_word_tier'))
                continue
            words = []
            for a, b, label in tiers[0]['entries']:
                if not label.strip() or label in {'<eps>', 'sil', 'sp', '<sil>'}:
                    continue
                assert math.isfinite(a) and math.isfinite(b)
                words.append(dict(text=label, start=a+row['chunk_start'], end=b+row['chunk_start'], source='mfa'))
                totals['zero_duration_mfa_words'] += abs(b-a) <= 1e-7
                totals['invalid_mfa_words'] += not (0 <= a < b <= row['duration']+1e-5)
                totals['oov_mfa_words'] += label in {'<unk>', 'spn'}
            predictions.write(json.dumps(dict(id=row['id'], audio=row['audio'], text=row['text'],
                chunk_start=row['chunk_start'], chunk_end=row['chunk_end'], channel=row['channel'],
                source_sample_rate=row['source_sample_rate'], start_sample=row['start_sample'],
                end_sample=row['end_sample'], words=words), ensure_ascii=False)+'\n')
            totals['mfa_chunks'] += 1; totals['mfa_words'] += len(words)
            if not words:
                failures.append(dict(id=row['id'], stage='mfa', reason='empty_word_output'))
                continue
            pairs, excluded, distance = align_words(row['words'], words)
            totals['matched_chunks'] += bool(pairs)
            totals['matched_words'] += len(pairs)
            totals['excluded_qwen_words'] += len(row['words'])-len(pairs)
            totals['excluded_mfa_words'] += len(words)-len(pairs)
            matching.write(json.dumps(dict(id=row['id'], matched_words=len(pairs),
                edit_distance=distance, excluded=excluded), ensure_ascii=False)+'\n')
            for i, j in pairs:
                q, ref = row['words'][i], words[j]
                onset, offset = abs(q['start']-ref['start'])*1000, abs(q['end']-ref['end'])*1000
                delta = (onset+offset)/2
                totals['onset_abs_sum_ms'] += onset; totals['offset_abs_sum_ms'] += offset
                totals['delta_sum_ms'] += delta
                for t in thresholds:
                    thresholds[t] += offset <= int(t)
                words_out.write(json.dumps(dict(id=row['id'], qwen_index=i, mfa_index=j,
                    word=q['text'], mfa_word=ref['text'], qwen=[q['start'], q['end']],
                    mfa=[ref['start'], ref['end']], boundary_mae_ms=delta), ensure_ascii=False)+'\n')
    with gzip.open(directory/'failures.jsonl.gz', 'wt') as f:
        for row in failures:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    n = totals['matched_words']
    summary = dict(batch=directory.name, totals=totals, offset_threshold_counts=thresholds,
        failed_chunks=len(failures), mfa_exit_code=status, mfa_command_seconds=command_seconds,
        total_seconds=time.perf_counter()-started, matched_mae_ms=totals['delta_sum_ms']/n if n else None)
    dump(directory/'summary.json', summary)
    shutil.rmtree(scratch)
    print(f'{directory.name}: completed chunks={len(rows)} matched={n} mae_ms={summary["matched_mae_ms"]}', flush=True)
    return summary


def aggregate(args, inventory):
    summaries = []
    for task in inventory['batches']:
        path = Path(task['path'])/'summary.json'
        if path.exists():
            summaries.append(json.loads(path.read_text()))
    totals, thresholds = {}, {str(t): 0 for t in [25, 50, 100, 200]}
    for summary in summaries:
        for k, v in summary['totals'].items():
            totals[k] = totals.get(k, 0) + v
        for k, v in summary['offset_threshold_counts'].items():
            thresholds[k] += v
    n = totals.get('matched_words', 0)
    metric = dict(matched_words=n, onset_mae_ms=totals.get('onset_abs_sum_ms',0)/n if n else None,
                  offset_mae_ms=totals.get('offset_abs_sum_ms',0)/n if n else None,
                  boundary_mae_ms=totals.get('delta_sum_ms',0)/n if n else None,
                  offset_accuracy_percent={k:100*v/n if n else None for k,v in thresholds.items()})
    result = dict(complete=len(summaries)==len(inventory['batches']),
        completed_batches=len(summaries), total_batches=len(inventory['batches']),
        inventory=inventory['counts'], totals=totals,
        failed_chunks=sum(s['failed_chunks'] for s in summaries),
        metrics=dict(qwen=metric, **{'qwen+mfa':metric},
                     mfa=dict(matched_words=n,onset_mae_ms=0.,offset_mae_ms=0.,boundary_mae_ms=0.)),
        reference='Pure MFA pseudo-reference, not manually annotated ground truth',
        metric_definition='Mean over matched words of (abs(onset error)+abs(offset error))/2; milliseconds',
        paper='https://arxiv.org/html/2610.22583v1#S4',
        matching='Canonical word forms, Levenshtein alignment, exact matches only; literal MFA <unk>/spn labels excluded',
        hybrid_validation=inventory['hybrid_validation'],
        scope=inventory['scope'], mfa_configuration=inventory['mfa_configuration'])
    dump(args.output/'summary.json', result)
    return result


def main():
    started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--existing', type=Path, default=ROOT/'results/open_yap_1k_rover_4gpu/qwen3-fa+mfa')
    p.add_argument('--manifest-root', type=Path, default=ROOT/'data/open_yap_1k_rover')
    p.add_argument('--output', type=Path, default=ROOT/'results/open_yap_1k_mfa_mae')
    p.add_argument('--scratch', type=Path, default=Path('/tmp/fa_open_yap_pure_mfa_mae'))
    p.add_argument('--batch-size', type=int, default=16384)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--jobs', type=int, default=32)
    p.add_argument('--limit-batches', type=int, help='Smoke run only; inventory still covers all inputs')
    args = p.parse_args()
    assert args.jobs>0 and args.workers>0 and args.batch_size>0
    args.output = args.output.resolve(); args.scratch = args.scratch.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    inventory = prepare(args)
    tasks = inventory['batches'][:args.limit_batches] if args.limit_batches else inventory['batches']
    todo = [dict(task, scratch=str(args.scratch), jobs=args.jobs) for task in tasks
            if not (Path(task['path'])/'summary.json').exists()]
    aggregate(args, inventory)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_batch, task) for task in todo]
        for future in as_completed(futures):
            future.result()
            result = aggregate(args, inventory)
            print(f'Completed {result["completed_batches"]}/{result["total_batches"]} batches', flush=True)
    result = aggregate(args, inventory)
    result['invocation_seconds'] = time.perf_counter()-started
    dump(args.output/'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
