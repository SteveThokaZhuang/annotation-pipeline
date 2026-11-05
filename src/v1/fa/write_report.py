"""Render Chinese result documentation from saved experiment artifacts."""
import json
from pathlib import Path
from common import ROOT

def main():
    base=ROOT/'results/fa'
    e=json.loads((base/'evaluation.json').read_text());i=json.loads((base/'insights.json').read_text());p=json.loads((base/'policy.json').read_text())
    lines=['# AMI-100 实测问题与修复结果','','本表全部来自本次运行；时间准确性以 AMI 银标准计。独立人工逐词金标准尚未建立，详见 [参考标准与协议](01_reference_and_protocol.md)。','',f"数据：{e['dataset']['samples']} 条、{e['dataset']['unique_events']} 个独立事件、{e['dataset']['audio_seconds']:.2f} 秒、{e['dataset']['words']} 个参考词。",'']
    for split,title in [('all','全量集'),('held_out','原验证集 IS1009a（V2 已看过其 V1 失败，不是全新盲测）')]:
        lines += [f'## {title}','','| 系统 | 参考口径 | 匹配/参考词数 | MAE ms | P95 ms | IoU | >100ms 或未匹配 | 零时长（参考匹配词） |','|---|---|---:|---:|---:|---:|---:|---:|']
        for b in ['qwen','mfa','nyra','hybrid']:
            for ref,label in [('all_reference','完整'),('filtered_reference','过滤')]:
                m=e['systems'][b][split][ref]
                lines.append(f"| {b} | {label} | {m['matched_words']}/{m['reference_words']} | {m['mae_ms']:.2f} | {m['p95_ms']:.2f} | {m['mean_iou']:.3f} | {m['bad_or_missing']} | {m['categories'].get('zero_duration',0)} |")
        lines+=['']
    lines+=['## Qwen 问题分类与替代模型修复','','类别可重叠；下面是完整参考 758 词口径。非零仅是结构修复，不自动等于时间正确。','','| Qwen 类别 | Qwen 问题词 | MFA 消除该类 / 达到≤100ms | Nyra 消除该类 / 达到≤100ms | 混合消除该类 / 达到≤100ms |','|---|---:|---:|---:|---:|']
    for cat in ['zero_duration','short_duration','long_duration','missing_or_substituted','boundary_mae_gt100ms','boundary_mae_gt200ms','iou_lt0.5']:
        cells=[]
        for b in ['mfa','nyra','hybrid']:
            m=e['paired_rescue'][b][cat];cells.append(f"{m['category_removed']} / {m['timing_rescued_le100ms']}")
        n=e['paired_rescue']['mfa'][cat]['qwen_cases']
        lines.append(f'| {cat} | {n} | '+ ' | '.join(cells)+' |')
    lines += ['','## 验证集上的新增退化','','仅过滤参考；好词=MAE≤100ms，坏词=MAE>100ms或未匹配。按同一参考词配对。','','| 替代系统 | 修复 Qwen 坏词 | 将 Qwen 好词变坏 | Qwen 零时长词中消除零时长 |','|---|---:|---:|---:|']
    for b,m in i['held_out_paired'].items():lines.append(f"| {b} | {m['rescued']} | {m['regressed']} | {m['zero_removed']}/{m['zero_cases']} |")
    lines+=['','## 输入文本完整性与失败','','下面按输入转录 token 与输出配对，和上表“参考词”分母不同。MFA 的拆词、复合词及 OOV 都可能影响该项。所有模型异常均保留，没有按成功样本筛除。','','| 系统 | 预测词 | 输入词 | 匹配输入词 | 输入遗漏/替换 | 额外输出 | 执行错误 | 输出零时长 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in ['qwen','mfa','nyra','hybrid']:
        c=i[b]['counts'];lines.append(f"| {b} | {c['predicted_tokens']} | {c['input_tokens']} | {c['input_tokens_matched']} | {c['input_tokens_deleted']}/{c['input_tokens_substituted']} | {c['extra_output_tokens']} | {c['execution_errors']} | {c.get('zero_duration',0)} |")
    lines+=['','## 复核产物','','- `results/fa/word_diagnostics.jsonl`：四系统逐词误差、类别、参考排除标记。','- `results/fa/review.csv`：Qwen 异常及参考可疑词的人工复核队列，人工边界尚空。','- `results/fa/insights.json`：输入完整性、零时长具体例子、验证集配对修复/退化。','- `results/fa/evaluation.json`：完整/过滤、开发/验证、三种音频条件的全部统计。','- `results/fa/*runtime.json`、`mfa_utterance/runtime.json`：运行配置和耗时；环境安装及下载时间不当作模型推理时间。','','不能把以上异常全视为模型错误。长词可能是拖音，重叠音频可能包含其他说话人，参考词自身也可能吸收静音。最终边界判断需要独立人工复核。']
    (ROOT/'docs/fa/02_results.md').write_text('\n'.join(lines)+'\n')
    lines=['# 以 Qwen 为主的混合 FA 路由','','策略由 `evaluate.py` 只在 ES2004a 上拟合参数；V2 的防退化设计源自 V1 验证失败，因此 IS1009a 的 V2 分数仅为诊断性复测。结果保存为 `results/fa/policy.json`。IS1009a 不参与参数搜索；ASR 轨道复用这一冻结策略。另从 ES2014a 和 TS3003a 选取此前未运行的 253 条音频，冻结 V2 后进行新会议审计。','', '## 固定选型约束','','以句段为单位回退，避免逐词替换造成前后边界冲突。最多两个有先后顺序的规则；开发集实际回退比例≤40%，同分优先更少回退。开发目标为过滤参考上未截断的词 MAE 秒数之和（未匹配按1秒惩罚），加 0.1×未匹配/MAE>100ms词数。保留均值风险，避免首版目标掩盖数秒级退化。完整集和验证集继续报告均值与退化。','', '候选信号：硬结构错误/输入词不完整、短词比例、超长词、输入填充词比例、直接重复词、相邻词最大间隔。候选阈值写死在代码中，未用验证集搜索。仅使用原始输入 text、音频时长和 Qwen 输出；禁止使用参考词时间、reference flags、condition、会议 ID 或参考误差选后端。','', '## 本次冻结规则','','```json',json.dumps(p,ensure_ascii=False,indent=2),'```','','按数组顺序命中第一条；全部不命中则保留 Qwen。Qwen 异常会影响整段风险，即使参考标注被过滤，仍保留输入及路由。','', '## 执行与验收','','1. Qwen 对全部输入运行。','2. 根据冻结规则，仅将命中的音频与原文本发给指定后端。Nyra 对 um/uh 做约定映射；不向其传入参考标签。','3. V2 额外拒绝相对 Qwen 新增的>1.5秒长词；对 Qwen 中40ms–1.5秒的结构稳定词，若候选任一边界跳变>500ms，也拒绝整段回退。这是保守退化门，不证明 Qwen 锚点准确。回退输出还必须：非空；与输入 token 完全保序匹配；所有时间有限、正时长、在片段范围内、无起点逆序和超过20ms的词间重叠。','4. 回退不满足结构验收则保留 Qwen 原结果、标 `needs_review=true`，不拉伸零时长、不制造边界、不删除输入词。未命中规则但 Qwen 自身不满足验收，也保留输出并标复核。','5. `needs_review=false` 只代表通过结构验收，不能解释为通过人工准确性验收。ASR 文本错误、其他说话人内容与无法分辨的截断词，需转录/说话人复核，FA 本身不能保证恢复真值。','', '## 实际路由量','','| 集合 | 请求后端 | 实际选择 | 需复核句段 |','|---|---|---|---:|']
    for split,m in e['routing'].items():lines.append(f"| {split} | {m['requested']} | {m['selected']} | {m['needs_review']} |")
    lines+=['','开发集的40%约束不保证新域回退比例；生产分布变化需要监控。单机离线评测为先计算三模型再重放策略，保存完整对照；`hybrid.py` 实际只调用被路由的后端。必须另行检查真实按需执行与重放是否一致，尤其 MFA 的语料级统计。','', '当前 MFA 每条音频使用独立 speaker 目录，使 CMVN 不依赖同批其他请求；固定 `--single_speaker`，关闭说话人自适应；`--use_threading`、beam400/retry4000。Qwen BF16+SDPA，Nyra WavLM FP32，确保本次配置明确。','', 'Nyra 模型及其输出采用非商业研究许可；商业使用需另行取得对应授权，不能因推理代码 MIT 而推断权重/时间戳也是 MIT。见[原始许可](https://huggingface.co/nyralabs/nyra_forced_aligner_en/blob/main/LICENSE.md)。当前只使用公开标准模型，未使用未公开 Pro。']
    (ROOT/'docs/fa/03_hybrid_routing.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
