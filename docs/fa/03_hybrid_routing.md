# 以 Qwen 为主的混合 FA 路由

策略由 `evaluate.py` 只在 ES2004a 上拟合参数；V2 的防退化设计源自 V1 验证失败，因此 IS1009a 的 V2 分数仅为诊断性复测。结果保存为 `results/fa/policy.json`。IS1009a 不参与参数搜索；ASR 轨道复用这一冻结策略。另从 ES2014a 和 TS3003a 选取此前未运行的 253 条音频，冻结 V2 后进行新会议审计。

## 固定选型约束

以句段为单位回退，避免逐词替换造成前后边界冲突。最多两个有先后顺序的规则；开发集实际回退比例≤40%，同分优先更少回退。开发目标为过滤参考上未截断的词 MAE 秒数之和（未匹配按1秒惩罚），加 0.1×未匹配/MAE>100ms词数。保留均值风险，避免首版目标掩盖数秒级退化。完整集和验证集继续报告均值与退化。

候选信号：硬结构错误/输入词不完整、短词比例、超长词、输入填充词比例、直接重复词、相邻词最大间隔。候选阈值写死在代码中，未用验证集搜索。仅使用原始输入 text、音频时长和 Qwen 输出；禁止使用参考词时间、reference flags、condition、会议 ID 或参考误差选后端。

## 本次冻结规则

```json
{
  "version": 2,
  "guarded": true,
  "development_meeting": "ES2004a",
  "held_out_meeting": "IS1009a",
  "reference_status": "AMI forced-alignment silver; not independent human gold",
  "max_development_routing_fraction": 0.4,
  "objective": "filtered uncapped MAE seconds (missing=1s) + 0.1*bad_or_missing count",
  "development_baseline_loss": 40.59000000000079,
  "development_selected_loss": 33.43499900000179,
  "rules": [
    {
      "feature": "fillers",
      "threshold": 0.05,
      "backend": "mfa"
    },
    {
      "feature": "gap",
      "threshold": 0.5,
      "backend": "nyra"
    }
  ]
}
```

按数组顺序命中第一条；全部不命中则保留 Qwen。Qwen 异常会影响整段风险，即使参考标注被过滤，仍保留输入及路由。

## 执行与验收

1. Qwen 对全部输入运行。
2. 根据冻结规则，仅将命中的音频与原文本发给指定后端。Nyra 对 um/uh 做约定映射；不向其传入参考标签。
3. V2 额外拒绝相对 Qwen 新增的>1.5秒长词；对 Qwen 中40ms–1.5秒的结构稳定词，若候选任一边界跳变>500ms，也拒绝整段回退。这是保守退化门，不证明 Qwen 锚点准确。回退输出还必须：非空；与输入 token 完全保序匹配；所有时间有限、正时长、在片段范围内、无起点逆序和超过20ms的词间重叠。
4. 回退不满足结构验收则保留 Qwen 原结果、标 `needs_review=true`，不拉伸零时长、不制造边界、不删除输入词。未命中规则但 Qwen 自身不满足验收，也保留输出并标复核。
5. `needs_review=false` 只代表通过结构验收，不能解释为通过人工准确性验收。ASR 文本错误、其他说话人内容与无法分辨的截断词，需转录/说话人复核，FA 本身不能保证恢复真值。

## 实际路由量

| 集合 | 请求后端 | 实际选择 | 需复核句段 |
|---|---|---|---:|
| all | {'nyra': 16, 'qwen': 65, 'mfa': 19} | {'nyra': 8, 'qwen': 77, 'mfa': 15} | 29 |
| development | {'qwen': 30, 'mfa': 8, 'nyra': 12} | {'qwen': 35, 'mfa': 8, 'nyra': 7} | 15 |
| held_out | {'nyra': 4, 'qwen': 35, 'mfa': 11} | {'nyra': 1, 'qwen': 42, 'mfa': 7} | 14 |

开发集的40%约束不保证新域回退比例；生产分布变化需要监控。单机离线评测为先计算三模型再重放策略，保存完整对照；`hybrid.py` 实际只调用被路由的后端。必须另行检查真实按需执行与重放是否一致，尤其 MFA 的语料级统计。

当前 MFA 每条音频使用独立 speaker 目录，使 CMVN 不依赖同批其他请求；固定 `--single_speaker`，关闭说话人自适应；`--use_threading`、beam400/retry4000。Qwen BF16+SDPA，Nyra WavLM FP32，确保本次配置明确。

Nyra 模型及其输出采用非商业研究许可；商业使用需另行取得对应授权，不能因推理代码 MIT 而推断权重/时间戳也是 MIT。见[原始许可](https://huggingface.co/nyralabs/nyra_forced_aligner_en/blob/main/LICENSE.md)。当前只使用公开标准模型，未使用未公开 Pro。

当前执行架构、完整路由/验收流程与实测高RTF原因，另见 [V2具体设计与耗时拆解](08_hybrid_v2_design_and_rtf.md)。
