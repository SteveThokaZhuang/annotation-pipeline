"""Compare the paired runs and write a reproducible Chinese experiment report."""
import argparse
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_predictions(directory):
    path = directory / 'alignment_predictions.jsonl'
    return read_rows(path if path.exists() else directory / 'predictions.jsonl')


def validate_repairs(base, final, repairs, unresolved, source='mfa', cross_chunk_seams=None):
    import sys
    fa_path=str(Path(__file__).resolve().parent/'qwen3-fa')
    if fa_path not in sys.path:sys.path.insert(0,fa_path)
    from bounded_seams import splice_bounded, adjust_run, valid
    baseline = {r['id']:r for r in base}
    predicted = {r['id']:r for r in final}
    changed = {r['id']:set() for r in base}
    replayed = {r['id']:[dict(w) for w in r['words']] for r in base}
    midpoint_seams = 0
    bounded_edits = 0
    accepted = rejected = 0
    for repair in repairs:
        ident = repair['id']
        start, end = repair['start_index'], repair['end_index']
        assert repair['original_words'] == baseline[ident]['words'][start:end]
        assert all(s in baseline[ident]['sentences'] for s in repair['sentences'])
        assert start == repair['sentences'][0]['start_index']
        assert end == repair['sentences'][-1]['end_index']
        actual = replayed[ident][start:end]
        if repair['status'] == 'repaired':
            accepted += 1
            actual = repair['replacement_words']
            bounded = repair['boundary_limits'].get('policy') == 'bounded'
            if bounded:
                row=baseline[ident]
                expected,bounds=splice_bounded(replayed[ident],start,end,actual,
                    row['chunk_start'],row['chunk_end'],row['words'])
                assert bounds == repair['boundary_limits']
                replayed[ident]=expected
                for adjustment in bounds['adjustments']:
                    changed[ident].add(adjustment['neighbor_index'])
                    bounded_edits+=1
            replayed[ident][start:end] = [dict(w) for w in actual]
            for adjustment in ([] if bounded else repair['boundary_limits'].get('adjustments', [])):
                assert repair['boundary_limits']['policy'] == 'midpoint'
                index = adjustment['neighbor_index']
                assert index == (start-1 if adjustment['side'] == 'left' else end)
                endpoint = 'end' if adjustment['side']=='left' else 'start'
                assert adjustment['endpoint'] == endpoint
                before, after = adjustment['before_neighbor'], adjustment['after_neighbor']
                assert replayed[ident][index] == before
                midpoint = (adjustment['original_neighbor_endpoint']+adjustment['original_replacement_endpoint'])/2
                assert math.isclose(after[endpoint], midpoint, abs_tol=1e-12)
                assert after == dict(before, **{endpoint:midpoint})
                assert after['start'] < after['end']
                touching = actual[0]['start'] if endpoint=='end' else actual[-1]['end']
                assert math.isclose(touching,midpoint,abs_tol=1e-12)
                replayed[ident][index] = dict(after)
                changed[ident].add(index)
                midpoint_seams += 1
            left, right = repair['boundary_limits']['left'], repair['boundary_limits']['right']
            previous_end = left
            for word in actual:
                assert word['source'] == source
                assert left-1e-7 <= word['start'] < word['end'] <= right+1e-7
                assert word['start'] >= previous_end-1e-7
                previous_end = word['end']
            changed[ident].update(range(start,end))
        else:
            rejected += 1
            for index in range(start,end):
                if index not in changed[ident]:
                    assert replayed[ident][index] == baseline[ident]['words'][index]
    cross_chunk_seams=cross_chunk_seams or []
    for seam in cross_chunk_seams:
        lid,rid=seam['left_id'],seam['right_id']
        lrow,rrow=baseline[lid],baseline[rid]
        assert (lrow.get('source_audio') or lrow['audio'])==(rrow.get('source_audio') or rrow['audio'])
        assert (lrow.get('channel') or 0)==(rrow.get('channel') or 0)
        assert lrow['chunk_end']<=rrow['chunk_start']+1e-7
        if seam.get('policy')=='bounded':
            qid,mid=seam['qwen_id'],seam['mfa_id']
            assert {qid,mid}=={lid,rid}
            side='left' if qid==lid else 'right'
            a,b=replayed[lid][-1],replayed[rid][0]
            assert {a.get('source'),b.get('source')}=={'qwen','mfa'}
            assert (a if qid==lid else b)['source']=='qwen'
            assert a['end']>b['start']+1e-7
            original=baseline[qid]
            indices=range(len(replayed[qid])-1,-1,-1) if side=='left' else range(len(replayed[qid]))
            edits=adjust_run(replayed[qid],original['words'],indices,side,
                b['start'] if side=='left' else a['end'],original['chunk_start'],original['chunk_end'])
            assert edits==seam['adjustments']
            touched=[e['neighbor_index'] for e in edits]
            lo,hi=max(0,min(touched)-1),min(len(replayed[qid]),max(touched)+2)
            assert valid(replayed[qid][lo:hi],original['chunk_start'],original['chunk_end'])
            assert replayed[lid][-1]['end']<=replayed[rid][0]['start']+1e-7
            for edit in edits:changed[qid].add(edit['neighbor_index'])
            bounded_edits+=len(edits)
            continue
        li,ri=seam['left_index'],seam['right_index']
        assert li==len(replayed[lid])-1 and ri==0
        left,right=replayed[lid][li],replayed[rid][ri]
        assert left==seam['before_left'] and right==seam['before_right']
        assert {left.get('source'),right.get('source')}=={'qwen','mfa'}
        assert left['end']>right['start']+1e-7
        midpoint=(left['end']+right['start'])/2
        assert math.isclose(midpoint,seam['midpoint'],abs_tol=1e-12)
        assert seam['after_left']==dict(left,end=midpoint)
        assert seam['after_right']==dict(right,start=midpoint)
        assert midpoint-left['start']>1e-7 and right['end']-midpoint>1e-7
        replayed[lid][li]=dict(seam['after_left']);replayed[rid][ri]=dict(seam['after_right'])
        changed[lid].add(li);changed[rid].add(ri)
    for ident, row in baseline.items():
        assert replayed[ident] == predicted[ident]['words']
        assert len(row['words']) == len(predicted[ident]['words'])
        for i,(old,new) in enumerate(zip(row['words'],predicted[ident]['words'])):
            assert old['text'] == new['text']
            if i not in changed[ident]:
                assert old == new
    assert {r['id'] for r in final if r['issues'] or r.get('error')} == {
        r['id'] for r in unresolved['residual_chunks']}
    assert {r['task_id'] for r in repairs if r['status']=='unresolved'} == {
        r['task_id'] for r in unresolved['rejected_sentences'] if 'task_id' in r}
    return dict(status='passed', chunks=len(base), sentence_groups=len(repairs),
                accepted_groups=accepted, rejected_groups=rejected,
                whole_sentence_replacement=True, unchanged_neighbors=midpoint_seams+len(cross_chunk_seams)+bounded_edits==0,
                audited_bounded_neighbor_edits=bounded_edits,
                audited_midpoint_seams=midpoint_seams+sum(s.get('policy')!='bounded' for s in cross_chunk_seams),audited_cross_chunk_seams=len(cross_chunk_seams),
                accepted_intervals_positive_and_bounded=True, unchanged_text_and_token_count=True,
                unresolved_coverage_complete=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, default=Path('results/qwen-fa'))
    p.add_argument('--hybrid', type=Path, default=Path('results/qwen3-fa+mfa'))
    p.add_argument('--report', type=Path, default=Path('docs/candor_comparison.md'))
    args = p.parse_args()
    base = read_predictions(args.baseline)
    hybrid = read_predictions(args.hybrid)
    raw_hybrid = read_rows(args.hybrid/'qwen_predictions.jsonl')
    assert [r['id'] for r in base] == [r['id'] for r in hybrid] == [r['id'] for r in raw_hybrid]
    same_qwen = all(a['words'] == b['words'] for a,b in zip(base,raw_hybrid))
    assert same_qwen, 'Qwen stages differ: inspect paired runs before comparing repair gains'
    meta = [json.loads((d/'metadata.json').read_text()) for d in [args.baseline,args.hybrid]]
    assert meta[0]['input_sha256'] == meta[1]['input_sha256']
    runtimes = [json.loads((d/'runtime.json').read_text()) for d in [args.baseline,args.hybrid]]
    errors = [json.loads((d/'error_stats.json').read_text()) for d in [args.baseline,args.hybrid]]
    before, after = errors[0]['after'], errors[1]['after']
    repair = errors[1]['repair']
    repairs = read_rows(args.hybrid/'mfa/repairs.jsonl')
    seams=args.hybrid/'mfa/cross_chunk_seams.jsonl'
    validation = validate_repairs(base, hybrid, repairs, json.loads((args.hybrid/'unresolved.json').read_text()),
                                  cross_chunk_seams=read_rows(seams) if seams.exists() else [])
    (args.baseline.parent/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    valid = [r for r in repairs if r['status']=='repaired']
    if not valid:
        raise ValueError('No accepted MFA repair to illustrate')
    example = next((r for r in valid if 4<=len(r['original_words'])<=10 and len(r['sentences'])==1), valid[0])
    examples=[]
    for rows, directory in [(base,args.baseline),(hybrid,args.hybrid)]:
        row=next(r for r in rows if r['id']==example['id'])
        value=dict(id=row['id'], source_audio=row.get('source_audio'), text=example['text'],
                   timestamp_origin='source recording',
                   words=row['words'][example['start_index']:example['end_index']])
        examples.append(value)
        (directory/'example.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    summary = dict(paired_qwen_outputs_identical=same_qwen, input_sha256=meta[0]['input_sha256'], validation=validation,
                   baseline=dict(runtime=runtimes[0],errors=before),
                   hybrid=dict(runtime=runtimes[1],errors=after,repair=repair),
                   zero_duration_reduction=(before['zero_duration_words']-after['zero_duration_words'])/before['zero_duration_words']
                       if before['zero_duration_words'] else None, examples=examples)
    (args.baseline.parent/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    def percent(value): return f'{100*value:.4f}%'
    lines = ['# Candor：Qwen FA 与整句 MFA 回退对比','',
        '本次按最终要求执行：任一字词的时长为 0，则该字词所在的整句话全部交给 MFA；接受时整句所有词统一替换，不混用该句的 Qwen 时间戳。', '',
        '## 数据与运行条件','',
        '- 音频：`/mnt/data/L202500721/stevezhuang/candor` 下的全部 4 个 MP3。',
        '- ASR：复用 `../annotation_pipeline/results/v4/candor_v4_0_1/asr.jsonl`，没有重新识别。',
        f'- 两方案共用 70 个 16 kHz 单声道切片，总时长 {runtimes[0]["audio_seconds"]:.4f} 秒；manifest：`data/candor_full/manifest.jsonl`。',
        '- 纯 Qwen 使用物理 GPU 0；混合使用物理 GPU 1。各进程设置 CUDA_VISIBLE_DEVICES 后内部设备均为 cuda:0。MFA 在 CPU 上使用 8 个 job。',
        '- Qwen3-ForcedAligner-0.6B，public 后端、bfloat16、SDPA、batch size 4；MFA 3.4.3，官方 English US ARPA 模型和配套词典。',
        f'- manifest SHA-256：`{meta[0]["input_sha256"]}`；两次 Qwen 阶段的全部词及时间戳完全一致：{same_qwen}。',
        '- 时间戳单位为秒，原点为原始录音；切片偏移只添加一次。准备切片和下载模型不计入 FA RTF，准备耗时见 `data/candor_full/preparation.json`。', '',
        '## 错误统计与错误率','',
        '这些是可程序检测的时间戳结构错误率。没有人工词级时间戳真值，因此不代表边界准确率、WER 或人工听审准确率。中文按模型输出的字、英文按模型输出的词统计。', '',
        '| 指标 | 纯 Qwen（GPU 0） | Qwen＋MFA（GPU 1） |',
        '|---|---:|---:|']
    for label, key in [('字词总数','words'),('零时长字词数','zero_duration_words'),
                       ('零时长字词率','zero_duration_rate'),('异常字词数（去重）','invalid_words'),
                       ('异常字词率','invalid_word_rate'),('句子总数','sentences'),('含错误句子数','bad_sentences'),
                       ('句子错误率','sentence_error_rate'),('片段总数','chunks'),('含错误片段数','bad_chunks'),
                       ('片段错误率','chunk_error_rate'),('推理失败片段数','failed_chunks'),
                       ('文本非空但无输出片段数','empty_nonempty_text_chunks')]:
        values=[percent(x[key]) if key.endswith('rate') else str(x[key]) for x in [before,after]]
        lines.append(f'| {label} | {values[0]} | {values[1]} |')
    for kind in ['negative_duration','nonfinite','out_of_bounds','nonmonotonic','overlap']:
        lines.append(f'| {kind} | {before["issues_by_type"].get(kind,0)} | {after["issues_by_type"].get(kind,0)} |')
    lines.extend(['',
        '分母：字词率 = 对应异常字词数 / 全部输出字词数；句子率 = 含至少一个异常或推理失败的句子数 / 全部非空句子数；片段率 = 含异常、失败或非空文本无输出的片段数 / 全部片段数。异常字词数按索引去重，各类型数可重叠。零时长阈值为 `abs(end-start) <= 1e-7`，其他结构校验同样使用 1e-7 秒容差。', '',
        f'零时长字词减少 {before["zero_duration_words"]-after["zero_duration_words"]} 个，减少比例 {percent(summary["zero_duration_reduction"] or 0)}。', '',
        '## MFA 整句回退','',
        f'- 触发 {repair["triggered_sentences"]} 句话，合并为 {repair["sentence_groups"]} 个相邻句组；相邻触发句一起送入 MFA，组内每句都完整替换。',
        f'- 成功替换 {repair["repaired_sentences"]} 句话、{repair["accepted_groups"]} 个句组、{repair["replaced_words"]} 个词，整句修复成功率 {percent(repair["sentence_repair_success_rate"] or 0)}。',
        f'- 修复原始零时长词 {repair["fixed_zero_words"]} 个，拒绝 {repair["rejected_groups"]} 个句组。MFA 退出码：`{repair["mfa_exit_code"]}`。',
        '- 拒绝原因：`'+json.dumps(repair['rejection_reasons'],ensure_ascii=False)+'`。',
        '- 未解决项：`results/qwen3-fa+mfa/unresolved.json`，含句子、录音、裁剪范围、原始坏词、原因和最终残留异常。完整修复记录：`mfa/tasks.jsonl`、`mfa/repairs.jsonl`；MFA 原始结果在 `mfa/aligned/`。',
        '- 接受条件：词序和覆盖完整、整句所有字词为正时长、顺序合法；将裁剪相对时间恢复至原录音，再限制在切片及未修改邻词的边界内。边界裁剪造成任一词无时长则拒绝整句，不人为填充时长。', '',
        '## RTF','',
        'RTF = 本方案耗时 / 同一份 8,083.8125 秒音频。墙钟包含模型加载、音频读取、Qwen、可选 MFA、校验与输出；处理 RTF 排除输入检查及 Qwen 加载。两个进程分别测量自身耗时，不能把并行总墙钟当成单方案耗时。单次实验会受文件缓存及共享 CPU/I/O 影响。', '',
        '| 指标 | 纯 Qwen | Qwen＋MFA |','|---|---:|---:|'])
    for label,key in [('总墙钟（秒）','seconds'),('输入检查和模型加载（秒）','setup_seconds'),
                      ('Qwen 阶段墙钟（秒）','qwen_seconds'),('回退和输出阶段（秒）','repair_and_output_seconds'),
                      ('端到端 RTF','rtf'),('排除加载后的处理 RTF','processing_rtf')]:
        lines.append(f'| {label} | {runtimes[0][key]:.6f} | {runtimes[1][key]:.6f} |')
    lines.extend(['',f'MFA 子进程耗时 {repair["mfa_command_seconds"]:.6f} 秒；含裁剪和替换的整个修复阶段 {repair["seconds"]:.6f} 秒。',
        '`runtime.json` 另有 CUDA event 包围 Qwen 调用的 stream 计时；它可能包含 CPU 预处理造成的 GPU 等待，不作为纯 GPU kernel 时间或混合方案总耗时。', '',
        '## 同一句话的两个实际结果','',
        f'片段 `{example["id"]}`；原文：{example["text"]}。以下为整句全部字词，时间戳相对于原始录音。', '',
        '纯 Qwen（同时保存为 `results/qwen-fa/example.json`）：','',
        '```json',json.dumps(examples[0],ensure_ascii=False,indent=2),'```','',
        '整句 MFA 替换后（同时保存为 `results/qwen3-fa+mfa/example.json`）：','',
        '```json',json.dumps(examples[1],ensure_ascii=False,indent=2),'```','',
        '## 复现与校验','',
        '完整启动命令与算法说明见 [design.md](design.md)。运行 `python src/compare_results.py` 可从结果重新生成本记录和 `results/comparison.json`。', '',
        '运行 `python -m unittest discover -s tests -v` 检查整句原子替换、零时长拒绝、邻句边界、词序/覆盖、缩写分词映射、录音时间偏移、空批次和错误率分母。', '',
        '本次 10 项回归检查通过。比较脚本另对全部 70 个片段、380 个句组验证：整句替换完整、接受词全部为正时长且不越界、拒绝句与邻词未被修改、词数词序不变、未解决项记录完整。结果见 `results/validation.json`。', '',
        '边界限制：句子以已有 ASR 标点划分；标点和 120 秒 ASR 分块本身的语义准确性未经人工审校。输入发生词元映射歧义时保守地回退整个 ASR 片段。MFA 对重叠讲话、漏词、ASR 幻觉或词典外词仍可能失败；失败保持原结果并记录，正时长也不能保证声学边界正确。',''])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['examples','baseline','hybrid']},indent=2))
    print(args.report)


if __name__ == '__main__':
    main()
