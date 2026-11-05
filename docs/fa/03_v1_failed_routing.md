# V1 失败路由归档

此版本未通过原验证集；不能当作最终策略。回退新增严重时间偏移，后续加入防退化门，并重新设置不截断的误差目标。

策略由 `evaluate.py` 只在 ES2004a 上选取，结果保存为 `results/fa/v1/policy.json`。IS1009a 不参与选规则；ASR 轨道复用这一冻结策略。

## 固定选型约束

以句段为单位回退，避免逐词替换造成前后边界冲突。最多两个有先后顺序的规则；开发集实际回退比例≤40%，同分优先更少回退。开发目标为过滤参考上“未匹配或词 MAE>100ms”的词数，加 0.1×截断在 1 秒内的词 MAE 总和。它偏重减少大错，**不承诺整体 MAE 最优**。完整集和验证集继续报告均值与退化。

候选信号：硬结构错误/输入词不完整、短词比例、超长词、输入填充词比例、直接重复词、相邻词最大间隔。候选阈值写死在代码中，未用验证集搜索。仅使用原始输入 text、音频时长和 Qwen 输出；禁止使用参考词时间、reference flags、condition、会议 ID 或参考误差选后端。

## 本次冻结规则

```json
{
  "version": 1,
  "development_meeting": "ES2004a",
  "held_out_meeting": "IS1009a",
  "reference_status": "AMI forced-alignment silver; not independent human gold",
  "max_development_routing_fraction": 0.4,
  "objective": "filtered missing/MAE>100ms count + 0.1*clipped MAE seconds",
  "development_baseline_loss": 72.91850000000008,
  "development_selected_loss": 54.26850010000013,
  "rules": [
    {
      "feature": "fillers",
      "threshold": 0.05,
      "backend": "mfa"
    },
    {
      "feature": "gap",
      "threshold": 2,
      "backend": "nyra"
    }
  ]
}
```

按数组顺序命中第一条；全部不命中则保留 Qwen。Qwen 异常会影响整段风险，即使参考标注被过滤，仍保留输入及路由。

## 执行与验收

1. Qwen 对全部输入运行。
2. 根据冻结规则，仅将命中的音频与原文本发给指定后端。Nyra 对 um/uh 做约定映射；不向其传入参考标签。
3. 回退输出必须：非空；与输入 token 完全保序匹配；所有时间有限、正时长、在片段范围内、无起点逆序和超过20ms的词间重叠。
4. 回退不满足结构验收则保留 Qwen 原结果、标 `needs_review=true`，不拉伸零时长、不制造边界、不删除输入词。未命中规则但 Qwen 自身不满足验收，也保留输出并标复核。
5. `needs_review=false` 只代表通过结构验收，不能解释为通过人工准确性验收。ASR 文本错误、其他说话人内容与无法分辨的截断词，需转录/说话人复核，FA 本身不能保证恢复真值。

## 实际路由量

| 集合 | 请求后端 | 实际选择 | 需复核句段 |
|---|---|---|---:|
| all | {'qwen': 74, 'mfa': 19, 'nyra': 7} | {'qwen': 74, 'mfa': 19, 'nyra': 7} | 23 |
| development | {'qwen': 37, 'mfa': 8, 'nyra': 5} | {'qwen': 37, 'mfa': 8, 'nyra': 5} | 15 |
| held_out | {'qwen': 37, 'mfa': 11, 'nyra': 2} | {'qwen': 37, 'mfa': 11, 'nyra': 2} | 8 |

开发集的40%约束不保证新域回退比例；生产分布变化需要监控。单机离线评测为先计算三模型再重放策略，保存完整对照；`hybrid.py` 实际只调用被路由的后端。必须另行检查真实按需执行与重放是否一致，尤其 MFA 的语料级统计。

当前 MFA 配置固定 `--single_speaker`，关闭说话人自适应；`--use_threading`、beam400/retry4000。Qwen BF16+SDPA，Nyra WavLM FP32，确保本次配置明确。

Nyra 模型及其输出采用非商业研究许可；商业使用需另行取得对应授权，不能因推理代码 MIT 而推断权重/时间戳也是 MIT。见[原始许可](https://huggingface.co/nyralabs/nyra_forced_aligner_en/blob/main/LICENSE.md)。当前只使用公开标准模型，未使用未公开 Pro。
