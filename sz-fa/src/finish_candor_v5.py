"""Prepare fresh v5 ROVER input, run two FA routes, and export both JSON layouts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/qwen3-fa'))
from core import read_jsonl, write_json, write_jsonl
from prediction_format import export_json
from prepare_rover import canonical
from smoke_test_annotations import check_output, gpu_snapshot

MODES = [('qwen3-fa+mfa', 'qwen+mfa'), ('qwen3-fa+mfa+nyra', 'qwen+mfa+nyra')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audio_coordinates(row, chunk, info):
    """Resolve v5 VAD and enhanced-audio sample coordinate systems explicitly."""
    start = chunk.get('audio_start_sample', round(chunk['start'] * info.samplerate))
    end = chunk.get('audio_end_sample', round(chunk['end'] * info.samplerate))
    if not 0 <= start < end <= info.frames:
        raise ValueError('Invalid processed audio bounds: ' + row['id'])
    if not (math.isclose(start / info.samplerate, chunk['start'], abs_tol=1 / info.samplerate)
            and math.isclose(end / info.samplerate, chunk['end'], abs_tol=1 / info.samplerate)):
        raise ValueError('Processed audio times differ from VAD times: ' + row['id'])
    row['vad_sample_coordinates'] = dict(sample_rate=row['source_sample_rate'],
        start_sample=row['start_sample'], end_sample=row['end_sample'])
    row.update(source_sample_rate=info.samplerate, start_sample=start, end_sample=end,
               chunk_start=start / info.samplerate, chunk_end=end / info.samplerate,
               duration=(end - start) / info.samplerate)


def prepare(output, native, selected):
    """Archive ASR and its exact enhanced waveforms before leaving local scratch."""
    import soundfile as sf
    asr = output / 'asr'
    asr.mkdir(exist_ok=True)
    reports = []
    for run in sorted((native / 'runs').iterdir()):
        archive = asr / 'reports' / run.name
        archive.mkdir(parents=True, exist_ok=True)
        for source in sorted(run.glob('*-report.json')):
            shutil.copyfile(source, archive / source.name)
            value = json.loads(source.read_text())
            reports.append(dict(run=run.name, stage=value['stage'], status=value['status'],
                                tasks=value['tasks'], completed=value['completed'], failed=value['failed']))
        if (run / 'effective-configuration.json').exists():
            shutil.copyfile(run / 'effective-configuration.json', archive / 'effective-configuration.json')
        for source in run.glob('*-failures.jsonl'):
            if source.stat().st_size:
                shutil.copyfile(source, archive / source.name)
    write_json(asr / 'stage_reports.json', reports)
    latest = {}
    for report in reports:
        latest[report['stage']] = report
    for stage in ['ingest', 'dnsmos', 'vad', 'enhance', 'asr1', 'asr2', 'disagreement', 'asr3', 'rover']:
        if stage not in latest or latest[stage]['status'] != 'passed' or latest[stage]['failed']:
            raise RuntimeError('Native stage has not completed successfully: ' + stage)
    # Preserve all small native stage results for reproducibility; model/cache files stay local.
    for stage in ['vad', 'enhance', 'asr1', 'asr2', 'disagreement', 'asr3', 'rover']:
        for source in sorted((native / stage).glob('*/batch_results/*/*.json')):
            destination = asr / 'native' / source.relative_to(native)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    items = []
    for source in sorted((asr / 'native/rover').glob('*/batch_results/*/*.json')):
        items.extend((item, source) for item in json.loads(source.read_text())['results'])
    if not items:
        raise ValueError('Native ROVER produced no chunks')
    selected_by_id = {row['id']: row for row in selected}
    copied, sources, rows = {}, [], []
    (output / 'audio').mkdir(exist_ok=True)
    for item, source in items:
        row = canonical(item, source)
        identity = row['recording_id']
        if identity not in selected_by_id:
            raise ValueError('Unexpected recording: ' + identity)
        original = row['audio']
        if original not in copied:
            destination = output / 'audio' / (identity + '.wav')
            if destination.exists() and sha(destination) != sha(original):
                raise ValueError('Persistent processed audio differs from native source')
            if not destination.exists():
                shutil.copyfile(original, destination)
            info = sf.info(destination)
            copied[original] = str(destination.resolve())
            sources.append(dict(native_audio_path=original, audio_path=str(destination.resolve()),
                sha256=sha(destination), sample_rate=info.samplerate, duration=info.duration,
                selection=selected_by_id[identity]))
        row.update(audio=copied[original], source_audio=copied[original], native_audio_path=original,
                   original_input=selected_by_id[identity])
        # v5 retains VAD's original 16 kHz sample indices in chunk.start_sample,
        # while audio_path can reference enhanced 48 kHz audio. Use the explicit
        # audio_* indices with the actual persisted waveform's sample rate.
        info = sf.info(row['audio'])
        chunk = item['chunk']
        audio_coordinates(row, chunk, info)
        rows.append(row)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate ROVER chunks')
    if {r['recording_id'] for r in rows} != set(selected_by_id):
        raise ValueError('ROVER is missing one of the three selected audio files')
    rows.sort(key=lambda r: (r['recording_id'], r['channel'], r['start_sample']))
    write_jsonl(output / 'fa_input.jsonl', rows)
    write_json(asr / 'sources.json', sources)
    write_jsonl(asr / 'transcripts.jsonl', [dict(id=r['id'], recording_id=r['recording_id'],
        audio=r['audio'], start=r['chunk_start'], end=r['chunk_end'], text=r['text']) for r in rows])
    return rows


def finalize(directory, rows):
    blocks = sorted((directory / 'blocks').glob('b*'))
    if not blocks:
        raise ValueError('No completed FA blocks')
    for name in ['predictions.jsonl', 'alignment_predictions.jsonl', 'qwen_predictions.jsonl', 'mfa_predictions.jsonl']:
        if not all((block / name).exists() for block in blocks):
            continue
        with (directory / name).open('wb') as out:
            for block in blocks:
                with (block / name).open('rb') as inp:
                    shutil.copyfileobj(inp, out)
    shutil.copyfile(blocks[0] / 'annotation_schema.json', directory / 'annotation_schema.json')
    validation = check_output(directory, rows)
    routes = [json.loads((block / 'validation.json').read_text()) for block in blocks]
    assert all(v['status'] == 'passed' for checks in routes for v in checks.values())
    unresolved = dict(rejected_sentences=[], residual_chunks=[])
    for block in blocks:
        for kind, values in json.loads((block / 'unresolved.json').read_text()).items():
            unresolved[kind].extend(values)
    write_json(directory / 'unresolved.json', unresolved)
    normalized = read_jsonl(directory / 'predictions.jsonl')
    if not (directory / 'export_index.json').exists():
        export_json(directory, normalized, 'both')
    index = json.loads((directory / 'export_index.json').read_text())
    by_id = {r['chunk_id']: r for r in normalized}
    seen = set()
    for item in index['files']:
        value = json.loads((directory / item['path']).read_text())
        if item['unit'] == 'chunk':
            assert value == by_id[item['chunk_id']]
            continue
        for segment in value['annotated_segments']:
            chunk = by_id[segment['chunk_id']]
            seen.add(segment['chunk_id'])
            assert segment['text'] == chunk['text']
            assert segment['start'] == chunk['audio_start_sec'] and segment['end'] == chunk['audio_end_sec']
            a, b = segment['word_start_index'], segment['word_end_index']
            assert value['force_align']['words'][a:b] == chunk['force_align']['words']
            for full, local in zip(value['force_align']['timestamps'][a:b], chunk['force_align']['timestamps']):
                assert all(math.isclose(x, y + segment['start'], abs_tol=1e-9, rel_tol=0) for x, y in zip(full, local))
    assert seen == set(by_id)
    assert index['file_counts'] == dict(chunk=len(rows), audio=len({r['audio'] for r in rows}))
    validation.update(json_exports=index['file_counts'], export_text_and_timestamps_verified=True,
                      sentence_replacement_validation=routes)
    write_json(directory / 'format_validation.json', validation)
    return dict(validation=validation, errors=json.loads((directory / 'error_stats.json').read_text()),
                runtime=json.loads((directory / 'runtime.json').read_text()), exports=index['file_counts'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/smoke-test/candor_v5')
    parser.add_argument('--python', default='/mnt/data/L202500293/obed/envs/qwenasr/bin/python')
    parser.add_argument('--gpu', type=int, help='Optional fixed GPU; otherwise select an idle GPU before each route')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--preprocess', action='store_true', help='Run/resume the saved native v5 command first')
    args = parser.parse_args()
    output = args.output.resolve()
    job = json.loads((output / 'job.json').read_text())
    if args.preprocess:
        if args.finalize_only:
            parser.error('--preprocess cannot be combined with --finalize-only')
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   TOKENIZERS_PARALLELISM='false')
        with (output / 'preprocess_resume.log').open('w') as log:
            subprocess.run(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = read_jsonl(output / 'fa_input.jsonl') if args.finalize_only else prepare(
        output, Path(job['native_output']), read_jsonl(job['selected_inputs']))
    results = {}
    for folder, method in MODES:
        directory = output / folder
        if not (directory / 'runtime.json').exists() and not args.finalize_only:
            snapshots = [gpu_snapshot()]
            time.sleep(2)
            snapshots.append(gpu_snapshot())
            idle = [r for r in snapshots[-1] if r['utilization'] == 0 and
                r['memory_total_mib'] - r['memory_used_mib'] >= 32768 and
                any(old['gpu'] == r['gpu'] and old['utilization'] == 0 for old in snapshots[0]) and
                (args.gpu is None or args.gpu == r['gpu'])]
            if not idle:
                raise RuntimeError('No selected GPU is idle with at least 32 GiB free')
            gpu = min(idle, key=lambda r: r['memory_used_mib'])['gpu']
            command = [args.python, str(ROOT / 'src/rover_worker.py'), '--input', str(output / 'fa_input.jsonl'),
                '--output', str(directory), '--scratch', str(Path(job['work_root']) / 'fa' / folder),
                '--method', method, '--batch-size', '16', '--block-size', '1024', '--mfa-jobs', '8',
                '--export-unit', 'both']
            write_json(output / (folder + '.command.json'), dict(command=command, gpu_snapshots=snapshots,
                       cuda_visible_devices=str(gpu)))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
            with (output / (folder + '.log')).open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[folder] = finalize(directory, rows)
        print(json.dumps(dict(mode=folder, status='complete', exports=results[folder]['exports'])), flush=True)
    summary = dict(status='complete', source='fresh Pipeline_github_v5 preprocessing and ASR/ROVER',
        input_audio_files=len({r['audio'] for r in rows}), input_audio_seconds=360., chunks=len(rows),
        annotated_audio_seconds=sum(r['duration'] for r in rows), results=results)
    write_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
