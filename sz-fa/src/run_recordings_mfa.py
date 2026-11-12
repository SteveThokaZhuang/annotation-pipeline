"""Run the repository MFA baseline over completed ROVER recording transcripts.

ROVER recording JSON stores chunk offsets into enhanced full-recording WAVs,
whereas ``src/mfa/process.py`` expects one physical WAV per input row.  This
wrapper expands recordings, extracts chunk audio in bounded batches, invokes the
same MFA command, applies the shared MFA boundary normalization, and writes a
resumable gzip JSONL result.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/qwen3-fa'))
_mfa_spec = importlib.util.spec_from_file_location('_mfa_process_impl', ROOT / 'src/mfa/process.py')
_mfa_module = importlib.util.module_from_spec(_mfa_spec)
_mfa_spec.loader.exec_module(_mfa_module)
normalize_mfa_words = _mfa_module.normalize_mfa_words


def dump(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_rows(recordings):
    rows = []
    for path in sorted(Path(recordings).glob('*.json')):
        recording = json.loads(path.read_text())
        for index, item in enumerate(recording.get('chunks', [])):
            chunk = item.get('chunk', {})
            start = float(chunk.get('start', 0.0))
            end = float(chunk.get('end', start + float(chunk.get('duration', 0.0))))
            audio = chunk.get('audio_path') or chunk.get('path')
            if not audio:
                continue
            rows.append(dict(
                id=item.get('id') or chunk.get('id') or f'{path.stem}:{index}',
                text=item.get('text', '').strip(), audio=str(audio),
                chunk_start=start, chunk_end=end, duration=end - start,
                channel=int(chunk.get('channel', 0)),
                audio_start_sample=chunk.get('audio_start_sample'),
                audio_end_sample=chunk.get('audio_end_sample'),
                source_sample_rate=int(chunk.get('sample_rate', 44100)),
            ))
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ROVER chunk IDs')
    if not rows:
        raise ValueError(f'No chunks found under {recordings}')
    return rows


def run_batch(rows, start, output, jobs, mfa_python, dictionary, acoustic, epsilon):
    batch = rows[start:start + 5000]
    work_root = output / 'batches'
    batch_dir = work_root / f'b{start:07d}'
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    corpus = batch_dir / 'corpus'; aligned = batch_dir / 'aligned'; temporary = batch_dir / 'work'
    corpus.mkdir(parents=True)
    handles = {}
    try:
        for index, row in enumerate(batch):
            source = Path(row['audio'])
            if source not in handles:
                handles[source] = sf.SoundFile(source)
            handle = handles[source]
            a, b = row['audio_start_sample'], row['audio_end_sample']
            if a is None or b is None:
                a = round(row['chunk_start'] * handle.samplerate)
                b = round(row['chunk_end'] * handle.samplerate)
            handle.seek(int(a))
            samples = handle.read(int(b) - int(a), dtype='float32', always_2d=True)
            if len(samples) == 0:
                raise ValueError(f'Empty audio for {row["id"]}')
            channel = row['channel'] if row['channel'] < samples.shape[1] else 0
            audio = samples[:, channel]
            if handle.samplerate != 16000:
                audio = soxr.resample(audio, handle.samplerate, 16000)
            sf.write(corpus / f'c{index:06d}.wav', audio, 16000, subtype='PCM_16')
            (corpus / f'c{index:06d}.lab').write_text(row['text'] + '\n')
    finally:
        for handle in handles.values():
            handle.close()

    env = dict(os.environ)
    prefix = mfa_python.resolve().parent.parent
    env.update(PATH=str(prefix / 'bin') + os.pathsep + env.get('PATH', ''),
               LD_LIBRARY_PATH=str(prefix / 'lib') + os.pathsep + env.get('LD_LIBRARY_PATH', ''),
               MFA_ROOT_DIR=str(batch_dir / 'cache'), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [str(mfa_python), '-c',
               'from montreal_forced_aligner.command_line.mfa import mfa_cli; mfa_cli()',
               'align', str(corpus), str(dictionary), str(acoustic), str(aligned),
               '--output_format', 'json', '--num_jobs', str(jobs),
               '--temporary_directory', str(temporary), '--single_speaker', '--use_threading',
               '--no_use_postgres', '--beam', '100', '--retry_beam', '400', '--clean', '--overwrite']
    log = (output / 'logs' / f'b{start:07d}.log').open('w')
    command_started = time.perf_counter()
    status = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                            timeout=14400, check=False).returncode
    command_seconds = time.perf_counter() - command_started
    log.close()
    if status != 0:
        raise RuntimeError(f'MFA batch {start} failed with exit {status}')
    files = {p.stem: p for p in aligned.rglob('*.json')}
    predictions = []
    modes = {'unchanged': 0, 'epsilon_clamp': 0, 'bounded_compression': 0}
    failures = []
    for index, row in enumerate(batch):
        words = []
        path = files.get(f'c{index:06d}')
        try:
            if path is None:
                raise ValueError('Missing MFA alignment')
            data = json.loads(path.read_text())
            tiers = [v for k, v in data['tiers'].items() if k.casefold().endswith('words')]
            if len(tiers) != 1:
                raise ValueError('No unique MFA word tier')
            for a, b, label in tiers[0]['entries']:
                if not label.strip() or label in {'<eps>', 'sil', 'sp', '<sil>'}:
                    continue
                a, b = float(a) + row['chunk_start'], float(b) + row['chunk_start']
                if not all(math.isfinite(x) for x in (a, b)):
                    raise ValueError('Nonfinite MFA timestamp')
                words.append(dict(text=label, start=a, end=b, source='mfa'))
            if row['text'] and not words:
                raise ValueError('MFA produced no words')
            words, repair = normalize_mfa_words(words, row['chunk_start'], row['chunk_end'], epsilon)
            modes[repair['mode']] += 1
            predictions.append(dict(row, words=words, mfa_boundary_repair=repair))
        except Exception as exc:
            failures.append(dict(id=row['id'], error=str(exc)))
            predictions.append(dict(row, words=[], error=str(exc)))
    output_batch = output / 'batch_results' / f'b{start:07d}.jsonl.gz'
    with gzip.open(output_batch, 'wt', compresslevel=3) as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    shutil.rmtree(batch_dir)
    return dict(start=start, chunks=len(batch), words=sum(len(r['words']) for r in predictions),
                failed_chunks=len(failures), failures=failures, boundary_modes=modes,
                mfa_command_seconds=command_seconds, output=str(output_batch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recordings', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mfa-jobs', type=int, default=32)
    parser.add_argument('--mfa-python', type=Path, default=ROOT / 'models/mfa/bin/python')
    parser.add_argument('--dictionary', type=Path, default=ROOT / 'models/mfa-english/dictionary/english_us_arpa.dict')
    parser.add_argument('--acoustic-model', type=Path, default=ROOT / 'models/mfa-english/english_us_arpa.zip')
    parser.add_argument('--mfa-boundary-epsilon', type=float, default=1e-5)
    args = parser.parse_args()
    started = time.perf_counter()
    rows = load_rows(args.recordings)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'logs').mkdir(exist_ok=True)
    (args.output / 'batch_results').mkdir(exist_ok=True)
    manifest = args.output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    state_path = args.output / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'next': 0, 'batches': []}
    for start in range(state['next'], len(rows), 5000):
        result = run_batch(rows, start, args.output, args.mfa_jobs, args.mfa_python,
                           args.dictionary, args.acoustic_model, args.mfa_boundary_epsilon)
        state['next'] = start + result['chunks']; state['batches'].append(result)
        dump(state_path, state)
        print(json.dumps({'dataset': str(args.recordings), 'done': state['next'], 'total': len(rows),
                          'batch': result['start'], 'failed': result['failed_chunks']}), flush=True)
    totals = dict(chunks=len(rows), words=sum(x['words'] for x in state['batches']),
                  failed_chunks=sum(x['failed_chunks'] for x in state['batches']),
                  audio_seconds=sum(x['duration'] for x in rows),
                  mfa_command_seconds=sum(x['mfa_command_seconds'] for x in state['batches']))
    totals['rtf_wall'] = (time.perf_counter() - started) / totals['audio_seconds']
    totals['rtf_mfa_commands'] = totals['mfa_command_seconds'] / totals['audio_seconds']
    totals['boundary_modes'] = {k: sum(x['boundary_modes'].get(k, 0) for x in state['batches'])
                                for k in ('unchanged', 'epsilon_clamp', 'bounded_compression')}
    summary = dict(method='mfa', device='cpu', recordings=str(args.recordings), output=str(args.output),
                   complete=state['next'] == len(rows), totals=totals,
                   elapsed_seconds=time.perf_counter() - started,
                   code_sha256=hashlib.sha256((ROOT / 'src/mfa/process.py').read_bytes()).hexdigest())
    dump(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if totals['failed_chunks']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
