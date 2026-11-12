"""Smoke-test MFA and MFA-to-Nyra cascades on three Candor entries."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/qwen3-fa'))
from process import audit
from compare_results import read_predictions, read_rows, validate_repairs

MODES = [('qwen3-fa+mfa', 'Qwen＋MFA'), ('qwen3-fa+mfa+nyra', 'Qwen＋MFA＋Nyra')]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return [dict(zip(['gpu', 'utilization', 'memory_used_mib', 'memory_total_mib'],
                     [int(part.strip()) for part in line.split(',')])) for line in result.stdout.splitlines()]


def check_output(directory, expected):
    final = read_predictions(directory)
    normalized = read_rows(directory / 'predictions.jsonl')
    stats = json.loads((directory / 'error_stats.json').read_text())
    after = stats['after'] if 'after' in stats else stats['final']
    assert len(final) == len(normalized) == len(expected)
    computed = audit(copy.deepcopy(final))
    assert all(computed[key] == after[key] for key in computed)
    assert [row['id'] for row in final] == [row['id'] for row in expected]
    for original, raw, value in zip(expected, final, normalized):
        assert value['chunk_id'] == original['id'] and value['text'] == original['text']
        assert value['audio_path'] == original['audio']
        assert value['turns'] == [] and value['turns_status'] == 'not_available'
        assert value['annotation_unit'] == 'chunk' and 'user_audio_force_align' not in value
        audio_start = original['chunk_start'] if original.get('audio_scope') == 'recording' else 0.
        assert value['audio_start_sec'] == audio_start and value['audio_duration_sec'] == original['duration']
        alignment = value['force_align']
        assert alignment['words'] == [word['text'] for word in raw['words']]
        assert len(alignment['timestamps']) == len(alignment['words'])
        for pair, word in zip(alignment['timestamps'], raw['words']):
            assert len(pair) == 2
            for stamp, field in zip(pair, ['start', 'end']):
                assert math.isclose(stamp + original['chunk_start'], word[field], abs_tol=1e-9, rel_tol=0)
    return dict(status='passed', samples=len(final), words=after['words'],
        text_and_word_order_preserved=True, timestamp_origin_verified=True,
        audit_statistics_match=True, missing_turns_explicit=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/candor_full/manifest.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/smoke-test')
    parser.add_argument('--python', default='/mnt/data/L202500293/obed/envs/qwenasr/bin/python')
    parser.add_argument('--report', type=Path, default=ROOT / 'docs/smoke_test_annotation_format.md')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        gpu = json.loads((args.output / 'selection.json').read_text())['gpu']
        summarize(args, read_rows(args.output / 'input.jsonl'), gpu)
        return
    samples = [gpu_snapshot()]
    time.sleep(2)
    samples.append(gpu_snapshot())
    idle = [gpu for gpu in samples[-1] if gpu['utilization'] == 0 and
            gpu['memory_total_mib'] - gpu['memory_used_mib'] >= 32768 and
            any(old['gpu'] == gpu['gpu'] and old['utilization'] == 0 for old in samples[0])]
    if not idle:
        raise RuntimeError('No idle GPU with 32 GiB free; inspect nvidia-smi before retrying')
    gpu = min(idle, key=lambda value: value['memory_used_mib'])['gpu']
    rows = read_rows(args.input)[:3]
    if len(rows) != 3:
        raise ValueError('Need at least three input entries')
    args.output.mkdir(parents=True, exist_ok=True)
    protected = ['input.jsonl', 'selection.json', 'summary.json']
    protected += [name for name, _ in MODES] + [name + '.log' for name, _ in MODES]
    existing = [name for name in protected if (args.output / name).exists()]
    if existing:
        raise FileExistsError('Choose a new output directory; existing experiment artifacts: ' + ', '.join(existing))
    manifest = args.output / 'input.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    write(args.output / 'selection.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_manifest=str(args.input.resolve()), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        selection='first three manifest entries, not first three complete recordings',
        formal_modes=[name for name, _ in MODES],
        ids=[row['id'] for row in rows], audio_seconds=sum(row['duration'] for row in rows),
        chunking='fixed 120-second ASR chunks; no turn/speaker metadata', gpu=gpu, gpu_selection_samples=samples))
    results = []
    for name, label in MODES:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        command = [args.python, '-u', str(ROOT / 'src' / name / 'process.py'), '--input', str(manifest.resolve()),
            '--output', str((args.output / name).resolve()), '--device', 'cuda:0', '--batch-size', '4']
        print(f'START {name} GPU {gpu}', flush=True)
        with (args.output / (name + '.log')).open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        directory = args.output / name
        check = check_output(directory, rows)
        write(directory / 'format_validation.json', check)
        result = dict(mode=name, label=label, validation=check,
            errors=json.loads((directory / 'error_stats.json').read_text())['after'],
            runtime=json.loads((directory / 'runtime.json').read_text()))
        results.append(result)
        write(args.output / 'summary.json', dict(status='running', gpu=gpu, results=results))
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summarize(args, rows, gpu)


def summarize(args, rows, gpu):
    results = []
    for name, label in MODES:
        directory = args.output / name
        check = check_output(directory, rows)
        write(directory / 'format_validation.json', check)
        results.append(dict(mode=name, label=label, validation=check,
            errors=json.loads((directory / 'error_stats.json').read_text())['after'],
            runtime=json.loads((directory / 'runtime.json').read_text())))
    baseline = read_rows(args.output / 'qwen3-fa+mfa/qwen_predictions.jsonl')
    for name, _ in MODES:
        qwen = read_rows(args.output / name / 'qwen_predictions.jsonl')
        assert [row['words'] for row in qwen] == [row['words'] for row in baseline]
    validations = {}
    for name, source in [('qwen3-fa+mfa', 'mfa')]:
        directory = args.output / name
        validations[name] = validate_repairs(baseline, read_predictions(directory),
            read_rows(directory / source / 'repairs.jsonl'),
            json.loads((directory / 'unresolved.json').read_text()), source=source)
    directory = args.output / 'qwen3-fa+mfa+nyra'
    intermediate = read_rows(directory / 'mfa_predictions.jsonl')
    validations['cascade_mfa'] = validate_repairs(baseline, intermediate,
        read_rows(directory / 'mfa/repairs.jsonl'), json.loads((directory / 'mfa/unresolved.json').read_text()))
    nyra_records = read_rows(directory / 'nyra/repairs.jsonl')
    assert {record['task_id'] for record in nyra_records} == {
        record['task_id'] for record in read_rows(directory / 'mfa/repairs.jsonl') if record['status'] == 'unresolved'}
    validations['cascade_nyra'] = validate_repairs(intermediate, read_predictions(directory),
        nyra_records, json.loads((directory / 'unresolved.json').read_text()), source='nyra')
    summary = dict(status='complete', gpu=gpu, results=results, paired_qwen_identical=True, route_validation=validations)
    write(args.output / 'summary.json', summary)
    lines = ['# Candor 标注格式灰度测试', '',
        '输入为 `data/candor_full/manifest.jsonl` 按顺序前 3 条，分别覆盖原录音 0–120、120–240、240–360 秒，总长 360 秒。', '',
        '这些是定长 chunk，没有说话人或 turn 元数据。本轮使用 HLT-FD 的 `force_align: {timestamps, words}` 约定，'
        '以 `hlt-fd-chunk-v1` 明确保留 chunk 粒度；`turns: []` 配合 `turns_status: "not_available"`，不生成 user/assistant。', '',
        '`predictions.jsonl` 是新格式，时间戳相对于选定音频区间；Candor 的物理 WAV 切片均从 0 秒起。'
        '`alignment_predictions.jsonl` 保留原录音坐标及词级来源、句界和异常，供修复审计及旧统计工具使用。', '',
        f'运行在采样时计算负载为 0% 的物理 GPU {gpu}，Qwen＋MFA、Qwen＋MFA＋Nyra 依次执行。选卡快照、输入选择和哈希见 `results/smoke-test/selection.json`。', '',
        '| 方案 | 字词数 | 零时长字词 | 零时长率 | RTF | 格式校验 |',
        '|---|---:|---:|---:|---:|---|']
    for result in results:
        errors, runtime = result['errors'], result['runtime']
        lines.append(f'| {result["label"]} | {errors["words"]} | {errors["zero_duration_words"]} | '
            f'{errors["zero_duration_rate"]:.6%} | {runtime["rtf"]:.8f} | 通过 |')
    lines.extend(['', 'RTF 包含模型加载、对齐、回退及新旧格式输出；小样本用于验证格式和运行链路。', '',
        '两组 Qwen 输出完全一致，两组新旧坐标与词元覆盖核验通过；整句回退和边界校验通过。', '',
        '正式目录：`results/smoke-test/qwen3-fa+mfa/`、`results/smoke-test/qwen3-fa+mfa+nyra/`。', '',
        '三级方案第一条实际新格式预测：', '', '```json',
        json.dumps(read_rows(directory / 'predictions.jsonl')[0], ensure_ascii=False, indent=2), '```', ''])
    args.report.write_text('\n'.join(lines))
    print(f'Complete: {args.report}', flush=True)


if __name__ == '__main__':
    main()
