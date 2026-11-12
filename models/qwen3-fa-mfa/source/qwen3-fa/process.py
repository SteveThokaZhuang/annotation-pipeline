"""Run Qwen FA independently, or with a sentence repair hook."""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from core import QwenAligner, expand_rows, read_jsonl, write_json, write_jsonl
from sentences import sentence_spans
from prediction_format import FORMAT, chunk_prediction, export_json, write_predictions

ROOT = Path(__file__).resolve().parents[2]


def word_issues(words, start, end, epsilon=1e-7):
    """Check every output unit: Chinese characters / space-language words."""
    issues = []
    previous_start = previous_end = start
    for i, word in enumerate(words):
        a, b = word['start'], word['end']
        reasons = []
        if not (math.isfinite(a) and math.isfinite(b)):
            reasons.append('nonfinite')
        else:
            if abs(b - a) <= epsilon:
                reasons.append('zero_duration')
            elif b < a:
                reasons.append('negative_duration')
            if a < start - epsilon or b > end + epsilon:
                reasons.append('out_of_bounds')
            if a < previous_start - epsilon:
                reasons.append('nonmonotonic')
            if i and a < previous_end - epsilon:
                reasons.append('overlap')
        issues.extend(dict(index=i, text=word['text'], start=a, end=b, reason=r) for r in reasons)
        previous_start, previous_end = a, b
    return issues


def audit(predictions):
    counts = collections.Counter()
    total_words = bad_chunks = failed_chunks = empty_chunks = bad_words = sentences = bad_sentences = 0
    for row in predictions:
        issues = word_issues(row['words'], row['chunk_start'], row['chunk_end'])
        row['issues'] = issues
        counts.update(x['reason'] for x in issues)
        bad_indices = {x['index'] for x in issues}
        bad_words += len(bad_indices)
        for span in row.get('sentences', []):
            sentences += 1
            bad_sentences += bool(row.get('error') or not row['words'] or
                any(span['start_index'] <= i < span['end_index'] for i in bad_indices))
        total_words += len(row['words'])
        failed_chunks += bool(row.get('error'))
        empty_chunks += bool(row['text'].strip() and not row['words'])
        bad_chunks += bool(issues or row.get('error') or (row['text'].strip() and not row['words']))
    return dict(chunks=len(predictions), words=total_words, bad_chunks=bad_chunks,
                failed_chunks=failed_chunks, empty_nonempty_text_chunks=empty_chunks,
                zero_duration_words=counts['zero_duration'],
                zero_duration_rate=counts['zero_duration'] / total_words if total_words else 0,
                invalid_words=bad_words, invalid_word_rate=bad_words/total_words if total_words else 0,
                chunk_error_rate=bad_chunks/len(predictions) if predictions else 0,
                sentences=sentences, bad_sentences=bad_sentences,
                sentence_error_rate=bad_sentences/sentences if sentences else 0,
                issues_by_type=dict(counts))


def load_rows(path):
    import soundfile as sf
    path = Path(path).resolve()
    data = json.loads(path.read_text()) if path.suffix == '.json' else read_jsonl(path)
    rows = expand_rows(data if isinstance(data, list) else [data])
    for i, row in enumerate(rows):
        row.setdefault('id', f'row:{i}')
        audio = Path(row['audio'])
        row['audio'] = str(audio if audio.is_absolute() else (path.parent / audio).resolve())
        actual = float(sf.info(row['audio']).duration)
        row.setdefault('duration', actual)
        row.setdefault('chunk_start', 0.0)
        row.setdefault('chunk_end', row['chunk_start'] + row['duration'])
        if not math.isclose(actual, row['duration'], abs_tol=0.08):
            raise ValueError(f"{row['id']}: supply a chunk audio file matching duration ({actual} != {row['duration']})")
        if not math.isclose(row['chunk_end'] - row['chunk_start'], actual, abs_tol=0.08):
            raise ValueError(f"{row['id']}: chunk bounds disagree with audio duration")
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input ids')
    if not rows:
        raise ValueError('Input contains no audio chunks')
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'models/qwen3-forced-aligner/checkpoints')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--backend', choices=['public', 'native'], default='public')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--language', default='English')
    p.add_argument('--export-unit', choices=['chunk', 'audio', 'both'],
                   help='Also save individual JSON files; existing JSONL/audit outputs are retained')
    return p


def run(args, repair_factory=None):
    started = time.perf_counter()
    method = getattr(args, 'method', 'qwen+mfa' if repair_factory else 'qwen')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    rows = load_rows(args.input)
    args.output.mkdir(parents=True, exist_ok=False)
    aligner = QwenAligner(args.checkpoint, device=args.device, backend=args.backend, language=args.language)
    if args.device.startswith('cuda'):
        aligner.torch.cuda.synchronize()
    loaded = time.perf_counter()
    predictions, failures = [], []
    gpu_ms = 0.0
    def infer(batch):
        nonlocal gpu_ms
        a = b = None
        if args.device.startswith('cuda'):
            a = aligner.torch.cuda.Event(enable_timing=True)
            b = aligner.torch.cuda.Event(enable_timing=True)
            a.record()
        try:
            return aligner.batch(batch, strict=False)
        finally:
            if b is not None:
                b.record(); b.synchronize()
                gpu_ms += a.elapsed_time(b)
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        try:
            values = infer(batch)
        except Exception as exc:
            print(f'batch {offset}: retrying individually: {exc!r}', flush=True)
            values = []
            for row in batch:
                try:
                    values.extend(infer([row]))
                except Exception as single_exc:
                    failures.append(dict(id=row['id'], error=repr(single_exc)))
                    values.append((([], args.language), row))
        for (words, language), row in values:
            value = dict(row, words=[dict(w, source='qwen') for w in words], language=language,
                         backend=args.backend, method=method,
                         sentences=sentence_spans(row['text'], language, aligner.model.aligner_processor))
            error = next((f['error'] for f in failures if f['id'] == row['id']), None)
            if error:
                value['error'] = error
            predictions.append(value)
        print(f'Qwen {min(offset + len(batch), len(rows))}/{len(rows)} chunks', flush=True)
    qwen_finished = time.perf_counter()
    before = audit(predictions)
    write_jsonl(args.output / 'qwen_predictions.jsonl', predictions)
    repair_stats = {}
    if repair_factory:
        repair = repair_factory(args, aligner)
        repair_stats = repair.repair(predictions)
    after = audit(predictions)
    serialization = write_predictions(args.output, predictions)
    if getattr(args, 'export_unit', None):
        exported = export_json(args.output, (chunk_prediction(row) for row in predictions), args.export_unit)
        serialization['json_exports'] = exported['file_counts']
    write_jsonl(args.output / 'failures.jsonl', failures)
    write_json(args.output / 'error_stats.json', dict(before=before, after=after, repair=repair_stats))
    if not repair_factory:
        write_json(args.output / 'unresolved.json', [dict(id=r['id'], audio=r['audio'], issues=r['issues'],
                    error=r.get('error')) for r in predictions if r['issues'] or r.get('error')])
    write_json(args.output / 'metadata.json', dict(
        command=sys.argv, checkpoint=str(args.checkpoint.resolve()), device=args.device,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), backend=args.backend,
        batch_size=args.batch_size, input=str(args.input.resolve()),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        prediction_format=FORMAT, serialization=serialization,
        timestamp_unit='seconds', timestamp_origin='selected audio segment start',
        audit_file='alignment_predictions.jsonl', audit_timestamp_origin='source recording',
        zero_epsilon=1e-7, python=sys.executable,
        gpu_name=aligner.torch.cuda.get_device_name() if args.device.startswith('cuda') else None))
    elapsed = time.perf_counter() - started
    duration = sum(r['duration'] for r in rows)
    additional_load = repair_stats.get('nyra_model_load_seconds', 0.0)
    runtime = dict(method=method, samples=len(rows),
        audio_seconds=duration, device=args.device, seconds=elapsed,
        setup_seconds=loaded-started, qwen_seconds=qwen_finished-loaded,
        repair_and_output_seconds=time.perf_counter()-qwen_finished,
        gpu_stream_seconds=gpu_ms/1000, rtf=elapsed/duration,
        nyra_model_load_seconds=additional_load,
        processing_rtf=(elapsed-(loaded-started)-additional_load)/duration,
        qwen_gpu_stream_rtf=gpu_ms/1000/duration,
        scope='input validation + all model loading + Qwen + optional sentence fallbacks + QA + output IO; shared MP3 preparation excluded')
    write_json(args.output / 'runtime.json', runtime)
    print(json.dumps(runtime, ensure_ascii=False), flush=True)
    return runtime


if __name__ == '__main__':
    run(parser().parse_args())
