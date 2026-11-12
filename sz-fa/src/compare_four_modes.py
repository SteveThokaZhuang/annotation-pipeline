"""Audit four fresh Candor runs and report zero-duration severity and RTF."""
import argparse
import collections
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/qwen3-fa'))
from process import audit
from compare_results import read_rows, read_predictions, validate_repairs

MODES=[('qwen-fa','Qwen'),('qwen3-fa+mfa','Qwen＋MFA'),
       ('qwen3-fa+nyra','Qwen＋Nyra'),('qwen3-fa+mfa+nyra','Qwen＋MFA＋Nyra')]


def read(path):
    return json.loads(path.read_text())


def example(directory, records, predictions):
    accepted=[r for r in records if r['status']=='repaired']
    options=accepted or records
    if not options:
        return None
    chosen=next((r for r in options if len(r['sentences'])==1 and 3<=len(r['original_words'])<=12),
                min(options,key=lambda r:len(r['original_words'])))
    row=next(r for r in predictions if r['id']==chosen['id'])
    value=dict(id=row['id'],source_audio=row.get('source_audio'),text=chosen['text'],
               timestamp_origin='source recording',status=chosen['status'],
               input_zero_words=len(chosen['zero_indices']),
               words=row['words'][chosen['start_index']:chosen['end_index']],
               boundary_limits=chosen.get('boundary_limits'),reason=chosen.get('reason'),
               mfa_rejection_reason=chosen.get('mfa_rejection_reason'))
    (directory/'example.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


def display(value):
    if isinstance(value,dict): return {k:display(v) for k,v in value.items()}
    if isinstance(value,list): return [display(v) for v in value]
    return round(value,6) if isinstance(value,float) else value


def zero_severity(rows):
    chunks = sentences = longest = 0
    for row in rows:
        indices={i for i,w in enumerate(row['words']) if abs(w['end']-w['start'])<=1e-7}
        chunks += bool(indices)
        sentences += sum(any(s['start_index']<=i<s['end_index'] for i in indices) for s in row['sentences'])
        run=0
        for i in range(len(row['words'])):
            run=run+1 if i in indices else 0
            longest=max(longest,run)
    return dict(chunks_with_zero_duration=chunks,sentences_with_zero_duration=sentences,
                longest_consecutive_zero_words=longest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'results/candor_four_modes')
    p.add_argument('--report',type=Path,default=ROOT/'docs/candor_four_modes.md')
    args=p.parse_args()
    data={}
    for name,label in MODES:
        directory=args.root/name
        rows=read_predictions(directory)
        raw=read_rows(directory/'qwen_predictions.jsonl')
        stats=read(directory/'error_stats.json')
        assert audit(copy.deepcopy(rows))==stats['after'],name+': incorrect final statistics'
        assert audit(copy.deepcopy(raw))==stats['before'],name+': incorrect raw statistics'
        data[name]=dict(label=label,rows=rows,raw=raw,runtime=read(directory/'runtime.json'),
                        stats=stats,metadata=read(directory/'metadata.json'))
    baseline=data['qwen-fa']
    base_ids=[r['id'] for r in baseline['rows']]
    for name,value in data.items():
        assert [r['id'] for r in value['rows']]==base_ids
        assert [r['id'] for r in value['raw']]==base_ids
        assert value['metadata']['input_sha256']==baseline['metadata']['input_sha256']
        assert [r['words'] for r in value['raw']]==[r['words'] for r in baseline['rows']],name+': Qwen outputs differ'
        assert [r['sentences'] for r in value['rows']]==[r['sentences'] for r in baseline['rows']]
        assert value['runtime']['audio_seconds']==baseline['runtime']['audio_seconds']
    validation={}
    plain_mfa=data['qwen3-fa+mfa']
    directory=args.root/'qwen3-fa+mfa'
    validation['qwen+mfa']=validate_repairs(baseline['rows'],plain_mfa['rows'],
        read_rows(directory/'mfa/repairs.jsonl'),read(directory/'unresolved.json'),
        cross_chunk_seams=read_rows(directory/'mfa/cross_chunk_seams.jsonl') if (directory/'mfa/cross_chunk_seams.jsonl').exists() else [])
    direct=data['qwen3-fa+nyra']
    directory=args.root/'qwen3-fa+nyra'
    direct_records=read_rows(directory/'nyra/repairs.jsonl')
    validation['qwen+nyra']=validate_repairs(baseline['rows'],direct['rows'],direct_records,
        read(directory/'unresolved.json'),source='nyra')
    cascade=data['qwen3-fa+mfa+nyra']
    directory=args.root/'qwen3-fa+mfa+nyra'
    intermediate=read_rows(directory/'mfa_predictions.jsonl')
    assert [r['words'] for r in intermediate]==[r['words'] for r in plain_mfa['rows']], 'MFA stages differ'
    validation['cascade_mfa']=validate_repairs(baseline['rows'],intermediate,
        read_rows(directory/'mfa/repairs.jsonl'),read(directory/'mfa/unresolved.json'),
        cross_chunk_seams=read_rows(directory/'mfa/cross_chunk_seams.jsonl') if (directory/'mfa/cross_chunk_seams.jsonl').exists() else [])
    cascade_records=read_rows(directory/'nyra/repairs.jsonl')
    validation['cascade_nyra']=validate_repairs(intermediate,cascade['rows'],cascade_records,
        read(directory/'unresolved.json'),source='nyra')
    mfa_rejected={r['task_id'] for r in read_rows(directory/'mfa/repairs.jsonl') if r['status']=='unresolved'}
    assert {r['task_id'] for r in cascade_records}==mfa_rejected,'Nyra cascade must retry exactly MFA rejections'
    for name in ['qwen3-fa+nyra','qwen3-fa+mfa+nyra']:
        assert data[name]['stats']['repair']['nyra_load_error'] is None,'Nyra did not load'
    examples={
        'qwen+nyra':example(args.root/'qwen3-fa+nyra',direct_records,direct['rows']),
        'qwen+mfa+nyra':example(directory,cascade_records,cascade['rows'])}
    summary=dict(input_sha256=baseline['metadata']['input_sha256'],
        chunks=len(base_ids),audio_seconds=baseline['runtime']['audio_seconds'],
        qwen_stages_identical=True,mfa_stages_identical=True,validation=validation,examples=examples,modes={})
    zero_before=baseline['stats']['after']['zero_duration_words']
    for name,value in data.items():
        sources=collections.Counter(w['source'] for r in value['rows'] for w in r['words'])
        after=value['stats']['after']
        summary['modes'][name]=dict(label=value['label'],errors=after,runtime=value['runtime'],
            repair=value['stats']['repair'],word_sources=dict(sources),
            zero_severity=zero_severity(value['rows']),
            zero_duration_reduction=(zero_before-after['zero_duration_words'])/zero_before)
    summary['cascade_stages']=dict(qwen_zero_words=zero_before,
        after_mfa_zero_words=read(directory/'mfa_error_stats.json')['zero_duration_words'],
        final_zero_words=cascade['stats']['after']['zero_duration_words'])
    (args.root/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.root/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    percent=lambda x:f'{100*x:.4f}%'
    lines=['# Candor 四种强制对齐模式实验','',
        '新增 Qwen＋Nyra、Qwen＋MFA＋Nyra。本轮重新运行四种模式，均从 Qwen 全量推理开始；没有复用预测结果来缩短任一方案的计时。', '',
        '## 数据、模式与验收','',
        f'- 同一批 4 段 Candor 原始音频，共 70 个片段、{summary["audio_seconds"]:.4f} 秒、{baseline["stats"]["after"]["words"]} 个字词、{baseline["stats"]["after"]["sentences"]} 句话。复用已有 ASR，不重新识别。',
        f'- 输入：`data/candor_full/manifest.jsonl`，SHA-256：`{summary["input_sha256"]}`。',
        '- 四种模式的 Qwen 阶段逐词结果完全一致；两次 MFA 阶段逐词结果也完全一致，已自动核验。',
        '- 任一字词零时长即触发完整句子回退；相邻触发句按原规则合为句组，组内全部词一起替换。所有模式使用相同 0.35 秒裁剪余量和边界门槛。',
        '- Qwen＋Nyra：380 个 Qwen 异常句组直接送 Nyra。Qwen＋MFA＋Nyra：先完成 MFA，只将被 MFA 验收拒绝的完整句组送 Nyra，保留已接受的 MFA 结果。',
        '- 候选须文本覆盖及词序一致、无 skipped、未解析词或漏词、每个词为正时长，且不与未修改邻词重叠、不越出切片。Nyra 可通过 espeak 为词典外词生成发音。失败不伪造时长，保持上一阶段原结果，记录到单独的 `unresolved.json`。',
        '- Nyra 0.2.2，英文 standard 权重，FP32；WavLM 在第一次对齐前完整加载。独立的 um/uh 映射到官方 [UM]/[UH] 输入，输出恢复原文本。模型文件及 SHA-256 见 `models/nyra-forced-aligner/checkpoints/download-provenance.json`。',
        '- Qwen 为本地 0.6B、BF16、SDPA、public backend、batch size 4；MFA 3.4.3、English US ARPA、8 CPU jobs、beam 100/retry 400。', '',
        '## 零时长程度与错误率','',
        '错误率为时间戳结构异常率，没有人工词级时间真值，不能解释为边界准确率。零时长阈值为 `abs(end-start) <= 1e-7`；字词率分母是全部输出字词，句子/片段率分母分别是全部非空句子/全部片段。异常词按词索引去重。', '',
        '| 指标 | Qwen | Qwen＋MFA | Qwen＋Nyra | Qwen＋MFA＋Nyra |',
        '|---|---:|---:|---:|---:|']
    for label,key in [('零时长字词数','zero_duration_words'),('零时长字词率','zero_duration_rate'),
                      ('全部异常字词数','invalid_words'),('全部异常字词率','invalid_word_rate'),
                      ('含错误句子数','bad_sentences'),('句子错误率','sentence_error_rate'),
                      ('含错误片段数','bad_chunks'),('片段错误率','chunk_error_rate'),
                      ('推理失败片段','failed_chunks')]:
        vals=[v['stats']['after'][key] for v in data.values()]
        lines.append('| '+label+' | '+' | '.join(percent(v) if key.endswith('rate') else str(v) for v in vals)+' |')
    for reason in ['out_of_bounds','negative_duration','nonfinite','nonmonotonic','overlap']:
        lines.append('| '+reason+' | '+' | '.join(str(v['stats']['after']['issues_by_type'].get(reason,0)) for v in data.values())+' |')
    for label,key in [('含零时长词的句子数','sentences_with_zero_duration'),
                      ('含零时长词的片段数','chunks_with_zero_duration'),
                      ('最长连续零时长词数','longest_consecutive_zero_words')]:
        lines.append('| '+label+' | '+' | '.join(str(v['zero_severity'][key]) for v in summary['modes'].values())+' |')
    lines.extend(['','## RTF 与阶段耗时','',
        'RTF = 每个进程实测墙钟 / 同一份音频总时长。端到端包括输入检查、全部模型加载、Qwen、MFA/Nyra、被拒绝的尝试、验证与输出，不含共用切片准备和首次权重下载。处理 RTF 排除 Qwen 输入/加载与 Nyra 显式加载；保留各后端首条推理及 Nyra JIT、MFA CLI 生命周期，不是稳定热推理速度。', '',
        '| 指标 | Qwen | Qwen＋MFA | Qwen＋Nyra | Qwen＋MFA＋Nyra |',
        '|---|---:|---:|---:|---:|'])
    for label,key in [('总墙钟（秒）','seconds'),('Qwen 输入/加载（秒）','setup_seconds'),
                      ('Qwen 阶段（秒）','qwen_seconds'),('Nyra 模型加载（秒）','nyra_model_load_seconds'),
                      ('端到端 RTF','rtf'),('排除显式神经模型加载的处理 RTF','processing_rtf')]:
        lines.append('| '+label+' | '+' | '.join(f'{v["runtime"].get(key,0):.6f}' for v in data.values())+' |')
    gpu_values=[v['metadata']['cuda_visible_devices'] for v in data.values()]
    lines.extend(['',
        '运行两轮：先 Qwen（GPU 0）和 Qwen＋MFA（GPU 1），再 Qwen＋Nyra（GPU 0）和 Qwen＋MFA＋Nyra（GPU 1）；每个神经方案只使用一张 H100。GPU 映射依次为：`'+str(gpu_values)+'`。这是单轮测量，共享 CPU/文件系统和缓存会影响耗时，不表示统计显著性。','',
        '## Nyra 新方案阶段统计','',
        '| 指标 | Qwen＋Nyra 的 Nyra 阶段 | Qwen＋MFA＋Nyra 的 Nyra 阶段 |','|---|---:|---:|'])
    for label,key in [('请求句组','sentence_groups'),('请求完整句子','triggered_sentences'),
                      ('接受句组','accepted_groups'),('拒绝句组','rejected_groups'),('修复完整句子','repaired_sentences'),
                      ('修复零时长字词','fixed_zero_words'),('替换全部字词','replaced_words'),
                      ('Nyra 显式加载秒数','nyra_model_load_seconds'),('Nyra 对齐和验收秒数','nyra_inference_seconds')]:
        vals=[v['stats']['repair'][key] for v in [direct,cascade]]
        lines.append('| '+label+' | '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in vals)+' |')
    stages=summary['cascade_stages']
    lines.extend(['',
        f'三级方案零时长字词轨迹：Qwen {stages["qwen_zero_words"]} → MFA 后 {stages["after_mfa_zero_words"]} → Nyra 后 {stages["final_zero_words"]}；Nyra 在 MFA 基础上再减少 {stages["after_mfa_zero_words"]-stages["final_zero_words"]} 个。','',
        'Nyra 拒绝原因：',''])
    for value in [direct,cascade]:
        lines.append('- '+value['label']+'：`'+json.dumps(value['stats']['repair']['rejection_reasons'],ensure_ascii=False)+'`。')
    lines.extend(['','## 两个新方案的实际例子','',
        '以下分别展示 Nyra 直接回退和 MFA 拒绝后由 Nyra 接手的实际完整句子。为便于阅读，文档时间戳显示至 6 位小数，原始文件保留完整精度。',''])
    for name,value in examples.items():
        lines.extend(['### '+name,'','```json',json.dumps(display(value),ensure_ascii=False,indent=2),'```',''])
    lines.extend(['## 结果与复现','',
        '本轮结果根目录：`results/candor_four_modes/`。四个模式各有独立目录：`qwen-fa/`、`qwen3-fa+mfa/`、`qwen3-fa+nyra/`、`qwen3-fa+mfa+nyra/`。旧的双方案结果及记录保留。', '',
        '- 每个模式有 `predictions.jsonl`、`qwen_predictions.jsonl`、`error_stats.json`、`runtime.json`、`metadata.json`、`unresolved.json`。',
        '- 两个新增模式各有 `nyra/tasks.jsonl`、`nyra/predictions.jsonl`（原始裁剪内时间）、`nyra/repairs.jsonl` 和 `nyra/runtime.json`；`example.json` 是上文实际例子。',
        '- 三级方案另存 `mfa_predictions.jsonl`、`mfa_error_stats.json`、`mfa/unresolved.json` 和 MFA 的全部输入/输出及拒绝原因，能追溯每次转交。',
        '- 汇总：`comparison.json`；全量整句替换、边界、词序及未解决记录校验：`validation.json`。', '',
        '下载与启动命令见 [design.md](design.md)。运行 `python src/compare_four_modes.py` 可重新核验并生成本记录。`results/nyra-tests.log` 保存 16 项回归检查。',''])
    args.report.write_text('\n'.join(lines))
    print(json.dumps(dict(cascade_stages=stages,modes={k:dict(zero=v['errors']['zero_duration_words'],
        zero_rate=v['errors']['zero_duration_rate'],rtf=v['runtime']['rtf']) for k,v in summary['modes'].items()}),indent=2))
    print(args.report)


if __name__=='__main__':
    main()
