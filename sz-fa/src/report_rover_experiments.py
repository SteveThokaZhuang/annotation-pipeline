"""Generate an auditable report from completed streaming ROVER experiments."""
import argparse
from collections import Counter
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODES=[('qwen+mfa+nyra','Qwen＋MFA＋Nyra','qwen3-fa+mfa+nyra'),
       ('qwen+mfa','Qwen＋MFA','qwen3-fa+mfa'),('qwen+nyra','Qwen＋Nyra','qwen3-fa+nyra')]


def choose_example(directory):
    for backend in ['nyra','mfa']:
        for path in sorted(directory.glob(f'gpu_*/blocks/*/{backend}/repairs.jsonl')):
            with path.open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['status']=='repaired' and 3<=len(row['original_words'])<=10:
                        return dict(id=row['id'],text=row['text'],source_audio=row.get('source_audio'),
                            backend=backend,crop_start=row['crop_start'],crop_end=row['crop_end'],
                            original_words=row['original_words'],replacement_words=row['replacement_words'],
                            boundary_limits=row['boundary_limits'],record=str(path),task_id=row['task_id'])
    return None


def summarize_unresolved(directory):
    reasons=Counter()
    nontriggered=Counter()
    zero_chunks=[]
    for path in sorted(directory.glob('gpu_*/blocks/*/unresolved.json')):
        data=json.loads(path.read_text())
        for row in data.get('rejected_sentences',[]):
            reason=row.get('reason','unspecified')
            (nontriggered if reason=='qwen_error_without_zero_trigger' else reasons)[reason]+=1
        for row in data.get('residual_chunks',[]):
            zeros=[issue for issue in row.get('issues',[]) if issue['reason']=='zero_duration']
            if zeros:
                zero_chunks.append(dict(id=row['id'],text=row['text'],zero_words=len(zeros),
                    source_audio=row.get('source_audio',row.get('audio')),record=str(path)))
    return dict(rejected_reasons=dict(reasons.most_common()),nontriggered_issue_reasons=dict(nontriggered),zero_chunk_count=len(zero_chunks),
                largest_zero_chunks=sorted(zero_chunks,key=lambda row:(-row['zero_words'],row['id']))[:5])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/tmp/fa_open_yap_results'))
    p.add_argument('--output',type=Path,default=ROOT/'docs/open_yap_1k_experiments.md')
    a=p.parse_args()
    preparation=json.loads((ROOT/'data/open_yap_1k_rover/preparation.json').read_text())
    suite=json.loads((a.root/'suite_summary.json').read_text())
    results={r['method']:r for r in suite['experiments']}
    persistent=ROOT/'results/open_yap_1k_summary';persistent.mkdir(exist_ok=True)
    percent=lambda value:f'{value*100:.5f}%'
    lines=['# OpenYap ROVER 四卡流式强制对齐实验','',
        f'输入共 {preparation["files"]:,} 个 ROVER 批结果，{preparation["unique_chunks"]:,} 个唯一片段，'
        f'{preparation["recording_channels"]:,} 个录音通道，标注区间总长 {preparation["audio_seconds"]:.6f} 秒'
        f'（{preparation["audio_seconds"]/3600:.3f} 小时）。文本及原音频不修改。','',
        '原始输入：`'+preparation['input']+'`。四卡清单及来源文件 SHA-256 位于 `data/open_yap_1k_rover/preparation.json`。', '',
        f'本轮按要求依次执行三级方案、MFA 双路由消融、Nyra 双路由消融；当前完整完成 {len(results)}/3 组。每组都独立使用 GPU 0–3，四卡按标注时长分片，分片清单相同。', '',
        '## RTF 与最终零时长情况','',
        '主 RTF 包含四卡加载模型、读取/重采样、全部成功和失败回退、边界检查、验证、输出归档与汇总。分母为全部唯一标注区间秒数之和。RTF 用协调器真实墙钟计算，不能用各卡耗时相加替代。共享的输入清点、磁盘空间恢复及分片导出不计入各组 FA 墙钟。', '',
        '| 方案 | 完成片段 | 总字词 | 零时长字词 | 零时长率 | 四卡墙钟秒 | 四卡 RTF |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,label,_ in MODES:
        if method not in results:
            lines.append(f'| {label} | 待完成 | — | — | — | — | — |');continue
        r=results[method];e=r['errors']['final']
        lines.append(f'| {label} | {r["samples"]:,} | {e["words"]:,} | {e["zero_duration_words"]:,} | '
            f'{percent(e["zero_duration_rate"])} | {r["wall_seconds"]:.3f} | {r["wall_rtf"]:.8f} |')
    lines.extend(['','错误率为时间戳结构异常率，没有人工词级时间真值，因此不当作声学边界准确率。零时长阈值为 `abs(end-start) <= 1e-7`。字词/句子/片段百分比分别用全量字词/非空句子/片段计数重算，不平均各卡或各块的比例。',''])
    if len(results)==3:
        cascade=results['qwen+mfa+nyra'];mfa=results['qwen+mfa'];nyra=results['qwen+nyra']
        cz=cascade['errors']['final']['zero_duration_words'];mz=mfa['errors']['final']['zero_duration_words'];nz=nyra['errors']['final']['zero_duration_words']
        lines.extend([
            f'本次三级方案零时长残留最少：比 Qwen＋MFA 少 {mz-cz:,} 个（降低 {(mz-cz)/mz:.2%}），'
            f'比 Qwen＋Nyra 少 {nz-cz:,} 个（降低 {(nz-cz)/nz:.2%}）。Qwen＋Nyra 的四卡 RTF 最小，'
            f'比三级方案墙钟短 {cascade["wall_seconds"]-nyra["wall_seconds"]:.3f} 秒。', '',
            f'三级中的 Nyra 仅收到 {cascade["routes"]["nyra_requested"]:,} 个回退组，接受 '
            f'{cascade["routes"]["nyra_accepted"]:,} 个；直接 Qwen＋Nyra 收到 '
            f'{nyra["routes"]["nyra_requested"]:,} 个回退组。三组 Qwen 输出的逐卡 SHA-256 一致：'
            f'`{all(r["qwen_matches_first_experiment"] for r in results.values())}`。', '',
            '这些耗时来自各方案的一次全量顺序运行；共享主机负载可能变化，墙钟差额不能完全归因于单一回退模型。',''])
    lines.extend(['## 分阶段及残留程度',''])
    for method,label,folder in MODES:
        if method not in results:continue
        r=results[method]
        lines.extend(['### '+label,'',
            '| 阶段 | 总词数 | 零时长词 | 零时长率 | 含零词句子 | 含零词片段 | 最长连续零词 | 全部异常词率 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|'])
        for stage,stats in r['errors'].items():
            lines.append(f'| {stage} | {stats["words"]:,} | {stats["zero_duration_words"]:,} | '
                f'{percent(stats["zero_duration_rate"])} | {stats.get("zero_sentences",0):,} | '
                f'{stats.get("zero_chunks",0):,} | {stats.get("longest_consecutive_zero_words",0)} | '
                f'{percent(stats["invalid_word_rate"])} |')
        final=r['errors']['final']
        lines.extend(['',f'- 推理失败 {final["failed_chunks"]} 个片段；非空文本无输出 {final["empty_nonempty_text_chunks"]} 个片段。',
            '- 最终各类异常：`'+json.dumps(final['issues_by_type'],ensure_ascii=False)+'`。',
            '- 请求及接受数：`'+json.dumps(r['routes'],ensure_ascii=False)+'`。',
            '- 四卡阶段耗时之和（秒）：`'+json.dumps(r['phases_sum_worker_seconds'],ensure_ascii=False)+'`。',
            f'- 四卡 worker 总耗时 {r["sum_worker_seconds"]:.3f} 秒，归一化工作量 RTF {r["worker_work_rtf"]:.8f}；它不是四卡吞吐 RTF。',
            f'- Qwen 输出与第一组逐卡 SHA-256 一致：`{r.get("qwen_matches_first_experiment")}`。',''])
        if 'mfa' in r['errors'] and method=='qwen+mfa+nyra':
            before=r['errors']['qwen']['zero_duration_words'];middle=r['errors']['mfa']['zero_duration_words'];after=final['zero_duration_words']
            lines.extend([f'三级零时长轨迹：**{before:,} → {middle:,} → {after:,}**；Nyra 在 MFA 后再修复 {middle-after:,} 个。',''])
        residual=summarize_unresolved(a.root/folder)
        (persistent/(folder+'_failure_analysis.json')).write_text(json.dumps(residual,ensure_ascii=False,indent=2)+'\n')
        lines.extend([f'仍含零时长的片段共 {residual["zero_chunk_count"]:,} 个；回退拒绝原因前五项：`'+
            json.dumps(dict(list(residual['rejected_reasons'].items())[:5]),ensure_ascii=False)+'`。',''])
        if residual['nontriggered_issue_reasons']:
            lines.extend(['另有未触发零时长回退的原始异常记录：`'+
                json.dumps(residual['nontriggered_issue_reasons'],ensure_ascii=False)+'`；它们不计入后端回退拒绝数。',''])
        if residual['largest_zero_chunks']:
            lines.extend(['零时长字词最多的残留片段（完整原因和原时间戳在单独未解决记录中）：','',
                '```json',json.dumps(residual['largest_zero_chunks'][:3],ensure_ascii=False,indent=2),'```',''])
        example=choose_example(a.root/folder)
        if example:
            (persistent/(folder+'_example.json')).write_text(json.dumps(example,ensure_ascii=False,indent=2)+'\n')
            lines.extend(['实际完整回退例子（原始录音秒坐标）：','','```json',json.dumps(example,ensure_ascii=False,indent=2),'```',''])
    lines.extend(['## 配置、数据适配和记录位置','',
        '- 设计与全部代码入口：[open_yap_streaming_design.md](open_yap_streaming_design.md)。Qwen batch 128，块大小 32,768，块内按时长分桶，结果恢复 manifest 顺序；每卡 MFA 32 jobs。Qwen 和 Nyra 在一组实验内常驻，跨块复用。',
        '- 测量在共享主机进行，期间观察到其他 CPU/I/O 数据转换作业；RTF 为该实际运行环境的结果。',
        '- ROVER 去标点片段缺少可靠自然句界时，回退整个已有 VAD/ROVER 片段；不改写融合文本。缺失的单系统 ASR hypothesis 允许为 null，266 条相关数据已完整补回。',
        '- 校验包括整句/片段原子替换、正时长与邻词边缘、文本/词数不变、未解决记录覆盖。各块 `validation.json` 保存实际校验结果。',
        '- 每组目录含 `summary.json`、`results.json` 和四个 GPU 目录。`results.json` 列出全部最终预测分片；块内保留 Qwen、中间 MFA、最终预测及 `unresolved.json`。',
        '- 大结果实际在 `/tmp/fa_open_yap_results/`，项目 `results/open_yap_1k_rover_4gpu` 是访问链接。输入大索引也在本地磁盘。原因是共享项目盘出现 ENOSPC；这些本机文件不是共享盘持久归档，迁移前请保留所在机器。',
        '- 完整最终预测与未解决记录另压缩导出到项目 `results/open_yap_1k_summary/predictions/`，逐组实际存储路径与复制状态见下面的归档表。',
        '- 小体积的汇总、输入元数据和真实例子保存在项目 `results/open_yap_1k_summary/`，代码保存在 `src/`；结果根目录还保存了代码快照及 SHA-256。',
        '- 最终归档核验代码为 `src/verify_rover_outputs.py`；[final_audit.json](../results/open_yap_1k_summary/final_audit.json) 保存覆盖回执、路由校验、代码快照与归档完整性复核。',
        '- `results/open_yap_4gpu_suite.log` 记录三组先后顺序；`status.json` 为当前进度，`suite_summary.json` 为完成组的机器可读汇总。',''])
    archive_index=persistent/'predictions/archives.json'
    if archive_index.exists():
        lines.extend(['## 最终压缩预测导出','',
            '以下是完整最终预测的 gzip JSONL。该步骤在三组基准全部结束后执行，单独记录压缩和共享盘复制耗时，未混入主表的 FA 墙钟。共享复制若失败，会保留本地完整压缩文件并记录错误。','',
            '| 方案 | 片段数 | 压缩字节 | 导出秒数 | 加上导出后的 RTF |','|---|---:|---:|---:|---:|'])
        for entry in json.loads(archive_index.read_text()):
            r=results[entry['method']]
            lines.append(f'| {entry["method"]} | {entry["samples"]:,} | {entry["bytes"]:,} | '
                f'{entry["seconds"]:.3f} | {(r["wall_seconds"]+entry["seconds"])/r["audio_seconds"]:.8f} |')
            lines.extend(['',f'文件：`{entry["path"]}`；SHA-256：`{entry["sha256"]}`；共享复制错误：`{entry["shared_copy_error"]}`。',''])
            lines.extend(['导出核验：`'+json.dumps(entry.get('validation',{}),ensure_ascii=False)+'`。',''])
            lines.extend([f'单独的未解决记录：`{entry["unresolved_path"]}`；数量：`{entry["unresolved_counts"]}`。',''])
        lines.extend(['归档兼容字段 `rejected_sentences` 含后端拒绝组及没有零时长触发的原始异常记录；实际后端拒绝数与未触发记录数已另外列出。',''])
    a.output.write_text('\n'.join(lines))
    print(a.output)


if __name__=='__main__':main()
