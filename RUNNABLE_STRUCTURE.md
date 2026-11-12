# 当前可运行代码结构

本目录是从 `Pipeline_github_v3_Merge` 提取的当前运行代码，实际链路为：

`ingest -> dnsmos -> vad -> enhance_parts -> enhance -> asr1 -> asr2 -> disagreement -> asr3 -> rover -> pure MFA CPU`

## 上游阶段

- `run.py`、`run_pipeline.sh`：V3 控制器入口。
- `stages/*.py`：各阶段独立入口。
- `pipeline/audio_pipeline/cli.py`：阶段编排、依赖检查、结果汇总和失败记录。
- `pipeline/audio_pipeline/worker.py`：worker 执行、结果持久化和重试。
- `models/audio_ingestion/final_script/adapter.py`：ingest。
- `models/dnsmos/final_script/adapter.py`：DNSMOS。
- `models/VAD/FireRedVAD/final_script/adapter.py`：VAD。
- `pipeline/audio_pipeline/enhancement_dispatch.py`、`enhancement_chunks_v1.py`、`models/deepfilternet3/final_script/adapter_chunked.py`：`enhance_parts` 分块增强。
- `models/deepfilternet3/final_script/adapter.py`、`adapter_auto.py`：完整增强。
- `models/qwen3-asr/final_script/adapter.py`：ASR1，Qwen3-ASR。
- `models/funasr/final_script/adapter.py`：ASR2，FunASR。
- `models/disagreement/final_script/adapter.py`：ASR1/ASR2 差异计算和 ASR3 路由判定。
- `models/firered-asr2/final_script/adapter.py`：ASR3，FireRedASR 路由补充。
- `models/roverfusion/final_script/adapter.py`、`models/roverfusion/fuse.py`：ROVER 融合和 recording JSON 导出。

## 当前 FA 阶段：纯 MFA CPU

FA 阶段当前使用纯 MFA CPU 方案。

- `sz-fa/run_fa_pipeline.py`：先运行上游九阶段，取得 ROVER recording JSON，再启动纯 MFA。
- `sz-fa/src/run_recordings_mfa.py`：展开 recording JSON，按批次裁取 chunk 音频，调用 MFA 并生成批次结果。
- `sz-fa/src/mfa/process.py`：纯 MFA 主流程、MFA JSON 解析、边界归一化和结构审计。
- `sz-fa/src/qwen3-fa/core.py`：输入展开、chunk 元数据和通用 JSON 工具；这里不代表 FA 阶段会加载 Qwen 模型。
- `sz-fa/src/qwen3-fa/prediction_format.py`：预测、审计和输出 JSON 格式。
- `sz-fa/src/setup_mfa_models.py`：MFA 环境、声学模型和词典准备。

运行入口：

```bash
cd pipeline_runnable_subset_20261111/sz-fa
python3 run_fa_pipeline.py --output-root runs/oto141_pure_mfa
```

主要输出：

- `upstream_pipeline.log`
- `pure_mfa.log`
- `pure_mfa/summary.json`
- `pure_mfa/runtime.json`
- `pure_mfa/batch_results/*.jsonl.gz`
- `pure_mfa/failures.jsonl`
- `pure_mfa/unresolved.json`
- `fa_report.json`
- `call_stack.json`

## 空字符串 ASR 结果的当前处理

当前代码把 `{"text": ""}` 当作字符串类型的有效 ASR 结果，不会在通用失败策略中自动判定为失败。

- [failure_policy.py](pipeline/audio_pipeline/failure_policy.py) 的 `hypothesis_error()` 只拒绝缺失结果、`transcription_error`、`fallback_requested` 和非字符串文本；空字符串返回 `None`。
- [disagreement adapter](models/disagreement/final_script/adapter.py) 会将空 hypothesis 标为 `reason="empty"`、`high=true`，从而触发 ASR3 路由。
- [planning.py](pipeline/audio_pipeline/planning.py) 会把 ASR3 结果追加到 ROVER hypotheses；ASR3 仍为空时，空结果仍被保留。
- [ROVER adapter](models/roverfusion/final_script/adapter.py) 只排除 `None` 或 `hypothesis_error()` 非空的 hypothesis。空字符串会作为空投票进入融合；全空时可能得到 `text=""` 且 `transcript_status="fused"`。
- 只有所有 hypothesis 都缺失或失败时，ROVER 才返回 `transcript_status="no_transcript"`。
- [finalization.py](pipeline/audio_pipeline/finalization.py) 的 `no_transcript_chunks` 只统计显式 `no_transcript`，不会自动统计空文本 fused 结果。

因此，空 ASR 结果目前会被识别为高风险并尝试 ASR3，但不会被统一升级成失败；ASR3 仍为空时，结果可以继续进入 ROVER 和后续 FA。

纯 MFA 对空文本的处理不同：

- 输入文本本身为空时，`src/mfa/process.py` 保留空 words，并记录 `alignment_note`。
- 输入文本非空但 MFA 没产生 words 时，记录失败并写入 `failures.jsonl`、`unresolved.json`。
- 纯 MFA 不会为原本为空的 ASR 文本自动生成转写内容。

## 依赖和边界

模型权重、数据集和 MFA Python 环境没有复制到本目录。配置中的 checkpoint 仍指向现有模型路径；MFA 声学模型、词典和 MFA 运行环境必须在运行主机上可用。
