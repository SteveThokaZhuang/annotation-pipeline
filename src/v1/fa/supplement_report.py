"""Render ASR findings and frozen new-meeting audit without refitting policy."""
import collections
import hashlib
import json
from pathlib import Path
from common import ROOT,DATASET,load_manifest,read_jsonl,match_words

def table(e,i):
    lines=['| 系统 | 完整参考 MAE ms | 过滤参考匹配/总词 | 过滤 MAE ms | 过滤坏词/未匹配 | 输出零时长 |','|---|---:|---:|---:|---:|---:|']
    for b in ['qwen','mfa','nyra','hybrid']:
        a=e['systems'][b]['all']['all_reference'];f=e['systems'][b]['all']['filtered_reference'];c=i[b]['counts']
        lines.append(f"| {b} | {a['mae_ms']:.2f} | {f['matched_words']}/{f['reference_words']} | {f['mae_ms']:.2f} | {f['bad_or_missing']} | {c.get('zero_duration',0)} |")
    return lines

def main():
    base=ROOT/'results/fa';docs=ROOT/'docs/fa'
    load=lambda p:json.loads(p.read_text())
    ae=load(base/'asr_eval/evaluation.json');ai=load(base/'asr_eval/insights.json')
    rows=load_manifest();asr={r['id']:r for r in read_jsonl(base/'asr.jsonl')};c=collections.Counter()
    for r in rows:
        m=match_words(r['words'],[dict(text=t) for t in asr[r['id']]['text'].split()]);c.update(reference=len(r['words']),matched=len(m['pairs']),deleted=len(m['deleted']),inserted=len(m['inserted']),substituted=len(m['substituted']))
    lines=['# 真实 ASR → FA 评测','','本次用本地 Qwen3-ASR-1.7B 对全部100条音频重新识别，再把同一份识别文本送三套 FA；V2 使用参考文本开发得到的冻结路由，未在 ASR 结果上重调。输入和三方结果位于 `results/fa/asr.jsonl`、`asr_manifest.jsonl`、`asr_eval/`。','',f"参考758词，ASR输入874个空白分词 token。与参考严格词形 DP：匹配{c['matched']}，删除{c['deleted']}，插入{c['inserted']}，替换{c['substituted']}；表观词错误率{(c['deleted']+c['inserted']+c['substituted'])/c['reference']:.2%}。该值包含目标说话人范围和分词差异，不能等同于独立听审的识别错误率。",'']
    lines+=table(ae,ai)
    lines+=['','零时长输出口径：Qwen85个，混合73个；Nyra/MFA无零时长，但不因此认定对齐正确。过滤参考上混合 MAE 从102.22ms降至95.72ms；纯Nyra为60.01ms，明显优于本次Qwen主导策略。这说明Qwen主导约束存在准确性代价，不能宣称混合优于所有单模型。','', 'Nyra 在 `ES2004a_A_004_mixed` 跳过 `1.4`，造成1个输入 token 没有时间；数字的实际读法可能是 one point four 等，需要转录规范化/人工核对，不能随意假定。MFA 输出有拆分/替换 token，详见 insights 的输入完整性统计。','', '## 内容范围问题','','`IS1009a_C_013_mixed`：参考目标说话人文本结束于 user interface design；ASR 还识别出其他人的 Very good、项目经理介绍等内容。`IS1009a_B_013_mixed` 和 `ES2004a_C_013_mixed` 也出现超出目标参考文本的内容。这些内容可能真实来自其他说话人；本次没有听审证据把它们叫作“幻觉”。FA无法仅靠转录确定目标说话人，应先做说话人约束/转录复核。','', '不能仅对参考匹配词的 MAE 排名而隐去覆盖率：本轨道过滤参考中Qwen、Nyra、混合均只匹配409/468词，59词无匹配。MFA的410词覆盖差异还受到分词影响，并不表示FA修复了ASR识别。','', '参考文字轨道与真实ASR轨道的误差不能直接相减当作ASR造成的损失，因为计分词集合不同。']
    (docs/'04_asr_findings.md').write_text('\n'.join(lines)+'\n')
    e=load(base/'audit/evaluation.json');i=load(base/'audit/insights.json');freeze=load(base/'audit_freeze.json');assert hashlib.sha256((base/'policy.json').read_bytes()).hexdigest()==freeze['policy_sha256']
    live=load(base/'live/replay_comparison.json')
    lines=['# 冻结后的新会议审计与按需执行验证','','V1在IS1009a失败后修改了防退化门和目标，因此V2在该会议的复测不能当作独立验证。为此使用本机已有 `ami-extended-517` 中全部 ES2014a、TS3003a 两个新会议，共253条、983.97秒、1740参考词。它们与AMI-100两个会议完全不相交；未根据审计结果再选规则或阈值。','',f"推理前冻结的 policy SHA256：`{freeze['policy_sha256']}`；source manifest SHA256：`{freeze['source_sha256']}`。见 `results/fa/audit_freeze.json`。来源仍为AMI银标准，不是新增人工金标准。",'']
    lines+=table(e,i)
    q=e['systems']['qwen']['all']['filtered_reference'];h=e['systems']['hybrid']['all']['filtered_reference']
    lines+=['',f"过滤参考 MAE 相对下降{1-h['mae_ms']/q['mae_ms']:.2%}；坏词/未匹配从{q['bad_or_missing']}减至{h['bad_or_missing']}。只有两个新会议，未作显著性或广泛跨域泛化声明。纯Nyra过滤MAE更低；本实现满足Qwen主导约束，并非追求无约束的最优单模型替代。",'',f"实际选择：{e['routing']['all']['selected']}；请求：{e['routing']['all']['requested']}；需复核{e['routing']['all']['needs_review']}段。Qwen最终占199/253（78.66%）。拒绝回退的段落仍可含错误，不伪造修复。",'', '## 真正按需执行','','另对AMI-100运行 `hybrid.py`：Qwen100条，Nyra只请求16条，MFA只请求19条；最后选Qwen77、MFA15、Nyra8。回退请求数大于最终选择数，因为候选可能被验收门拒绝。', '',f"与三模型全量对照的离线重放逐词比较：{live['samples']}条，差异{len(live['differences'])}条（时间容差1e-5秒）。对应 `results/fa/live/replay_comparison.json`；fallback requests中只有输入音频/文本/时长/ID/语言，没有参考词时间。",'', 'MFA逐段独立speaker目录和关闭自适应使按需子集不依赖整批说话人统计。新会议MFA CLI耗时约70.4秒；首次导出曾因重复遍历TextGrid目录较慢，保留已完成的CLI结果后用 `--export-only` 导出。没有重复推理或替换模型预测，最终导出代码已改为一次构建索引。']
    (docs/'05_external_audit.md').write_text('\n'.join(lines)+'\n')
    diag=read_jsonl(base/'word_diagnostics.jsonl');q=[w for w in diag if w['backend']=='qwen'];zeros=[w for w in q if 'zero_duration' in w['categories']]
    lines=['# 问题分类、机制和未解决项','','## 零时长机制','','实际下载的Qwen config中 `timestamp_segment_time=80`（毫秒）。本次qwen-asr0.0.6源码先把预测类别乘这一量化步长，再调用 `fix_timestamp` 做非严格单调修正，允许相同起止时间。它不保证每个词正时长。可核对 `models/qwen3-forced-aligner/checkpoints/config.json` 以及运行环境 `qwen_asr/inference/qwen3_forced_aligner.py`。','',f"77个零时长词对应的参考时长中，只有{sum(w['ref_end']-w['ref_start']<.08+1e-6 for w in zeros)}个≤80ms。其余不是简单的“真实短词小于时间分辨率”即可解释；参考本身可能不精确，模型压缩、弱语音或内容不匹配也可能参与。未保存原始未修正logit，不能定量拆分后处理与声学预测各自贡献。",'', '最长连续塌缩例为 `ES2004a_D_000_noise10` 的5词；不应把同一时刻的词均分成假定时长。fillers只占6个零时长词，其余71个是普通词，因此不能只路由um/uh就声称解决零时长。','', '| 音频条件 | 参考词 | 零时长词 | 词MAE>100ms |','|---|---:|---:|---:|']
    for condition in ['headset','noise10','mixed']:
        ww=[w for w in q if w['condition']==condition];lines.append(f"| {condition} | {len(ww)} | {sum('zero_duration' in w['categories'] for w in ww)} | {sum(w['mae'] is not None and w['mae']>.1 for w in ww)} |")
    lines+=['','## 汇总分类','','| 类别 | 证据/检测 | 处置 |','|---|---|---|','| 时间结构 | 零/负/非有限、逆序、超界、输出词间重叠 | 规则回退；失败后复核，禁止捏造时间 |','| 精细边界偏差 | MAE、IoU、100/200ms尾部 | 使用双口径参考计分；保留替代模型退化 |','| 时长异常 | <40ms或>1.5秒 | 仅为可疑；回退不得新增超长词 |','| 序列完整性 | DP遗漏、替换、额外token、Nyra skipped | 数字/分词/OOV规范化或转录复核 |','| 内容/说话人范围 | 混音ASR多转录其他人的语音 | 上游说话人约束；FA不负责判断真实说话人 |','| 参考不确定性 | 290词带可疑时长/截断/重叠等排除原因 | 双口径报告，独立人工盲标后再定结论 |','| 候选回退退化 | 新增长词、稳定锚点跳变>500ms | 拒绝整段回退并标needs_review |','','剩余：AMI-100混合仍有52个零时长输出，真实ASR轨道仍有73个。结构正常也可能声学错位；guard可能挡住正确的大幅修复，不能将它称为真值判别器。应优先人工复核 `review.csv` 中的连续塌缩、数秒偏移、说话人范围及参考异常，再扩充独立金标准。']
    (docs/'06_issue_taxonomy.md').write_text('\n'.join(lines)+'\n')
    lines=['# AMI-100 / Qwen 主导混合 FA 实验','','已完成原始100条的参考文本三模型对照、真实ASR轨道、混合实现、实际按需运行，以及冻结后的253条新会议审计。现有AMI词边界是自动对齐银标准；独立人工金标准仍待建立，未伪称完成听审。','','- [参考标准与评判协议](01_reference_and_protocol.md)','- [AMI-100当前结果与修复统计](02_results.md)','- [最终V2混合路由](03_hybrid_routing.md)','- [真实ASR转录问题](04_asr_findings.md)','- [新会议审计和实际按需执行](05_external_audit.md)','- [问题汇总分类与机制](06_issue_taxonomy.md)','- [V1失败结果](02_v1_failed_results.md)及[失败路由](03_v1_failed_routing.md)','- [代码入口与复现](../../src/v1/fa/README.md)','','结果目录 `results/fa/`，模型权重在 `models/` 对应模型的 `checkpoints/`。Qwen主导V2在原集过滤MAE为75.41ms（Qwen88.27ms）；新会议过滤MAE为144.26ms（Qwen163.20ms）。Nyra在部分轨道整体更好，MFA也存在严重尾部偏移，不能宣称一个后端全面解决另一后端的问题。']
    (docs/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
