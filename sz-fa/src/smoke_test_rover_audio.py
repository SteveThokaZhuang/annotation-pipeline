"""Run one source WAV through Qwen -> MFA -> Nyra using its existing ROVER ASR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/qwen3-fa'))
from prediction_format import force_align
from process import word_issues
from smoke_test_annotations import check_output, gpu_snapshot, read_predictions, read_rows, write


def recover_archive(directory, scratch):
    """Resume an archive-copy failure from completed, validated local blocks."""
    from rover_worker import archive_block, statistics, Totals
    totals = {name: Totals() for name in ['qwen', 'mfa', 'final']}
    timings = []
    for block in sorted(scratch.glob('b*')):
        validation = json.loads((block / 'validation.json').read_text())
        if not all(check['status'] == 'passed' for check in validation.values()):
            raise ValueError('Cannot recover an unvalidated alignment block')
        destination = directory / 'blocks' / block.name
        if destination.exists():
            shutil.move(str(destination), str(scratch / (block.name + '_incomplete_shared_archive')))
        final = read_rows(block / 'alignment_predictions.jsonl')
        archive_block(block, destination, final)
        for stage, name in [('qwen', 'qwen_predictions.jsonl'), ('mfa', 'mfa_predictions.jsonl'),
                            ('final', 'alignment_predictions.jsonl')]:
            totals[stage].add(statistics(read_rows(block / name)))
        timings.append(json.loads((block / 'runtime.json').read_text()))
    if not timings:
        raise ValueError('No completed block found for archive recovery')
    write(directory / 'error_stats.json', {stage: value.value() for stage, value in totals.items()})
    write(directory / 'archive_recovery.json', dict(status='complete', source_scratch=str(scratch),
        inference_repeated=False, block_timings=timings,
        timing_scope='completed block timing only; total wall RTF unavailable after archive recovery'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--manifests', type=Path, default=ROOT / 'data/open_yap_1k_rover')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--python', default='/mnt/data/L202500293/obed/envs/qwenasr/bin/python')
    parser.add_argument('--finalize-only', action='store_true')
    parser.add_argument('--recover-scratch', type=Path)
    args = parser.parse_args()
    import soundfile as sf
    audio = args.audio.resolve()
    output = args.output or ROOT / 'results/smoke-test/open_yap_1k' / audio.parent.name / audio.stem
    if args.finalize_only or args.recover_scratch:
        directory = output / 'qwen3-fa+mfa+nyra'
        if args.recover_scratch:
            recover_archive(directory, args.recover_scratch)
        finalize(output, directory, read_rows(output / 'input.jsonl'), sf.info(audio),
                 json.loads((output / 'selection.json').read_text()))
        return
    rows, sources = [], []
    for manifest in sorted(args.manifests.glob('gpu_*.jsonl')):
        found = 0
        with manifest.open() as stream:
            for line in stream:
                if str(audio) not in line:
                    continue
                row = json.loads(line)
                if row['audio'] == str(audio):
                    rows.append(row)
                    found += 1
        if found:
            sources.append(dict(manifest=str(manifest.resolve()), selected_chunks=found))
    rows.sort(key=lambda row: (row['start_sample'], row['id']))
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('No matching ROVER ASR or duplicate source chunks')
    info = sf.info(audio)
    if any(row['source_sample_rate'] != info.samplerate or row['end_sample'] > info.frames for row in rows):
        raise ValueError('Source audio and ROVER sample boundaries disagree')
    snapshots = [gpu_snapshot()]
    time.sleep(2)
    snapshots.append(gpu_snapshot())
    candidates = [gpu for gpu in snapshots[-1] if gpu['utilization'] == 0 and
        gpu['memory_total_mib'] - gpu['memory_used_mib'] >= 32768 and
        any(old['gpu'] == gpu['gpu'] and old['utilization'] == 0 for old in snapshots[0])]
    if not candidates:
        raise RuntimeError('No idle GPU with 32 GiB free')
    gpu = min(candidates, key=lambda value: value['memory_used_mib'])['gpu']
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / 'input.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    selection = dict(audio=str(audio), audio_seconds=info.duration, sample_rate=info.samplerate,
        frames=info.frames, channels=info.channels, chunks=len(rows),
        annotated_audio_seconds=sum(row['duration'] for row in rows), sources=sources,
        transcript='existing fused ROVER text, unchanged; no new ASR',
        input_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), gpu=gpu, gpu_selection_samples=snapshots)
    write(output / 'selection.json', selection)
    directory = output / 'qwen3-fa+mfa+nyra'
    scratch = Path('/tmp') / ('fa_single_audio_smoke_' + str(os.getpid()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    command = [args.python, '-u', str(ROOT / 'src/rover_worker.py'), '--input', str(manifest),
        '--output', str(directory), '--scratch', str(scratch), '--method', 'qwen+mfa+nyra',
        '--batch-size', '128', '--block-size', '32768', '--mfa-jobs', '8']
    print(f'START triple FA GPU {gpu}: {len(rows)} chunks, {selection["annotated_audio_seconds"]:.3f} annotated seconds', flush=True)
    with (output / 'qwen3-fa+mfa+nyra.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    finalize(output, directory, rows, info, selection)


def finalize(output, directory, rows, info, selection):
    audio = Path(selection['audio'])
    gpu = selection['gpu']
    blocks = sorted((directory / 'blocks').glob('b*'))
    for name in ['predictions.jsonl', 'alignment_predictions.jsonl', 'qwen_predictions.jsonl', 'mfa_predictions.jsonl']:
        with (directory / name).open('wb') as destination:
            for block in blocks:
                with (block / name).open('rb') as source:
                    shutil.copyfileobj(source, destination)
    shutil.copyfile(blocks[0] / 'annotation_schema.json', directory / 'annotation_schema.json')
    validation = check_output(directory, rows)
    write(directory / 'format_validation.json', validation)
    routes = [json.loads((block / 'validation.json').read_text()) for block in blocks]
    assert all(check['status'] == 'passed' for block in routes for check in block.values())
    unresolved = dict(rejected_sentences=[], residual_chunks=[])
    for block in blocks:
        for kind, records in json.loads((block / 'unresolved.json').read_text()).items():
            unresolved[kind].extend(records)
    write(directory / 'unresolved.json', unresolved)
    raw = read_predictions(directory)
    words = [word for row in raw for word in row['words']]
    recording = dict(conversation_id=audio.parent.name, speaker_id=audio.stem, annotation_unit='recording',
        audio_path=str(audio), audio_start_sec=0., audio_end_sec=info.duration, audio_duration_sec=info.duration,
        text=' '.join(row['text'] for row in rows), force_align=force_align(words), turns=[], turns_status='not_available',
        annotated_segments=[dict(chunk_id=row['id'], start=row['chunk_start'], end=row['chunk_end'],
                                 text=row['text']) for row in rows],
        alignment_status='needs_review' if any(row.get('issues') for row in raw) or
                         word_issues(words, 0, info.duration) else 'aligned')
    assert len(recording['force_align']['words']) == validation['words']
    assert all(pair == [word['start'], word['end']] for pair, word in zip(recording['force_align']['timestamps'], words))
    write(output / 'recording_prediction.json', recording)
    stats = json.loads((directory / 'error_stats.json').read_text())
    summary = dict(status='complete', method='qwen+mfa+nyra', selection=selection,
        format_validation=validation, route_validation=routes, errors=stats,
        recording_prediction=str(output / 'recording_prediction.json'),
        chunk_predictions=str(directory / 'predictions.jsonl'))
    write(output / 'summary.json', summary)
    lines = ['# OpenYap 单音频三 FA 格式灰度', '',
        f'输入：`{audio}`。物理 GPU {gpu}，使用连续两次采样计算负载为 0% 且空闲显存足够的卡。', '',
        f'全文件 {info.duration:.6f} 秒、{info.samplerate} Hz、{info.channels} 通道；匹配 '
        f'{len(rows)} 个原有 ROVER 区间，标注区间总长 {selection["annotated_audio_seconds"]:.6f} 秒。'
        '沿用融合 ASR 原文，按 Qwen → MFA → Nyra 路由执行一次，没有重跑 ASR。', '',
        '本次目的是验证标注 pipeline 的输出格式。源文件是 speaker_b 音轨，但没有真实 turn/角色元数据；'
        '保留 `turns_status: "not_available"`，不把 VAD 区间写成 user/assistant turn。', '',
        f'完成 {validation["samples"]} 个区间、{validation["words"]} 个字词；Qwen/MFA/最终零时长数：'
        f'{stats["qwen"]["zero_duration_words"]}/{stats["mfa"]["zero_duration_words"]}/{stats["final"]["zero_duration_words"]}。', '',
        f'Nyra 实际请求 {sum(block["nyra"]["sentence_groups"] for block in routes)} 个句组；仅在 MFA 未通过时触发。', '',
        '- `recording_prediction.json`：整条音轨的并列 words/timestamps，时间戳对应完整 WAV；未标注的静音区间不补造词。',
        '- `qwen3-fa+mfa+nyra/predictions.jsonl`：逐区间标准预测，时间戳对应各音频选定区间的起点。',
        '- `alignment_predictions.jsonl`：原录音坐标及完整修复审计；块内保留 MFA/Nyra 接受、拒绝与未解决记录。',
        '- `summary.json`：格式、词元/时间偏移、覆盖及整句回退校验结果，均通过。', '',
        ('归档时共享盘曾返回 ENOSPC，使用完整本机结果恢复，没有重复推理；见 `archive_recovery.json`。'
         'Candor 的可重建 MFA cache/work 移至本机并保留链接，主预测及审计文件保存于共享项目目录，见 `results/smoke-test/storage.json`。'
         if (directory / 'archive_recovery.json').exists() else '主预测及审计结果保存在上述项目输出目录。'), '',
        f'结果目录：`{output}`。', '', '第一条实际区间结果：', '', '```json',
        json.dumps(read_rows(directory / 'predictions.jsonl')[0], ensure_ascii=False, indent=2), '```', '']
    report = ROOT / 'docs/open_yap_single_audio_smoke.md'
    report.write_text('\n'.join(lines))
    print(f'COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
