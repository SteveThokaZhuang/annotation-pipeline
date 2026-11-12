# complex_handling

独立的复杂场景检测与规则路由包，主类为 `ComplexHandler`。公共 FA 入口和 `rover_worker.py` 通过 class 加载；不开启时保持原有回退链路。首版提供 `shadow` 模式：风险检测在原整句回退前执行，额外候选在回退后生成并单独保存，不自动覆盖主预测。

## 使用

在原 FA 命令后增加以下参数；四种入口、纯 Qwen 和 streaming worker 均支持：

```bash
--complex-handling shadow \
--complex-config src/complex_handling/example.json \
--complex-evidence /absolute/path/evidence.jsonl
```

后两个参数可省略。配置路径中模型文件的相对路径以配置文件所在目录为原点。默认不加载事件或说话人模型；只运行文本、已有质量/事件证据和时间戳检查。没有某项证据时记录 `not_provided`，不能解释为已排除该类风险。`candidate_backend: "none"` 可仅记录风险和复核决定，不运行额外 Nyra。

Python 接口（先将项目 `src` 加入 Python 模块路径）：

```python
from complex_handling import ComplexConfig, ComplexHandler, EvidenceStore

handler = ComplexHandler(
    config=ComplexConfig(),
    evidence_store=EvidenceStore("evidence.jsonl"),
    candidate_backend=my_backend,  # 实现 align(task)，返回 candidate_words / skipped 等
)
handler.prepare(predictions, audio_loader=load_selected_segment)
# 原回退类可用 handler.baseline_rejection(row, "mfa"/"nyra") 做任务门控。
# 此处执行原整句回退；不改写原 ASR 文本。
stats = handler.run(predictions, output_dir, audio_loader=load_selected_segment)
```

`load_selected_segment(row)` 返回当前完整 chunk 的 `(mono_audio, actual_sample_rate)`；不能返回整段录音。主干会自动接入现有 loader 和 Nyra 类，streaming 时复用已加载的模型。自定义候选后端返回录音绝对坐标的完整 `candidate_words`，每项包含 `text/start/end`。

## 证据格式

每条 JSONL 对应一个 FA 输入 `id`。也可直接在 manifest 行内提供 `complex_evidence`；同一行不能同时使用两种来源。示例：

```json
{
  "id": "recording:chunk_1",
  "timestamp_origin": "chunk",
  "events": [
    {"start": 0.4, "end": 0.9, "tag": "laughter", "score": 0.85,
     "source": "my_event_detector", "detector_version": "v1"}
  ],
  "speech": [{"start": 0.0, "end": 1.8}],
  "speakers": [
    {"start": 0.0, "end": 1.8, "speaker": "A"},
    {"start": 0.8, "end": 1.2, "speaker": "B"}
  ],
  "dnsmos": {"BAK": 2.2},
  "hypotheses": [{"text": "well um yes"}, {"text": "well yes"}],
  "target_isolated": false
}
```

有时间区间时必须声明 `timestamp_origin: "chunk" | "recording"`。`chunk` 指 FA 选定音频区间的起点；统一输出为原录音坐标，转换只执行一次。录音级事件可在上游连续检测，再按 chunk ID 送入，区间在读取时裁到该 chunk。重复证据 ID、非法区间和检测器错误均显式记录。

`events` 表示上游已通过其检测阈值的事件区间；`score` 只保留原始分数，不视作校准概率。支持 `laughter/cough/breath/noise/overlap/transcript_uncertain`。`speech` 是语音活动区间；显式 `[]` 表示检测器认为该区间没有语音，缺少字段表示没有证据。

`target_isolated: true` 仅适用于 FA 实际输入已经是目标说话人独立音轨的情况；只有一个 speaker 标签不满足此条件。短句、`yeah` 或“嗯”不会自动变成 backchannel/打断标签。当前只检测其相关的重叠风险，未实现交互功能的语义分类。

ROVER 新建 manifest 会保留精简的各路 `hypotheses` 和输入语言来源。旧 manifest 不重写，可用 sidecar 补齐。DNSMOS 使用 `BAK/bak/background` 做背景质量初筛，不换算为 SNR；ASR 分歧使用最长序列归一化的 token 编辑距离，中文按字比较，不能直接套用上游单向 WER 阈值。

## 可选声学适配器

- `PannsEventDetector`：配置本地 `panns_checkpoint`（decision-level 模型），安装兼容的 `panns-inference`、PyTorch、NumPy、SciPy 后启用。使用模型本身加载本地权重，避开官方推理 wrapper 的自动下载。内部重采样到 32 kHz、约 10 秒窗口/5 秒步长，按模型的帧步长合并窗口，平滑、双阈值和短事件处理后输出风险区间。当前类别为笑声、咳嗽、呼吸、音乐及白噪声，不能覆盖所有噪声。
- `PyannoteSpeakerDetector`：配置已完整下载的本地 `pyannote_model`，使用普通 `speaker_diarization`，不使用消除重叠的 exclusive 结果。启用配置时主入口在模型加载前设置离线环境；本地 pipeline 的依赖权重也必须已准备好。当前按 FA chunk 检测，speaker ID 仅在本次检测中有效；需要跨 chunk 说话人身份或完整交互上下文时，应提供录音级预计算证据。

适配器均惰性加载，缺依赖或加载失败会记录为复核，不冒充无风险。配置中不填路径时不会导入这两个模型。此轮没有安装依赖、下载权重或运行真实神经网络评测。

默认对实际 FA 音频检测。可在 manifest 中提供明确的原音频选择器 `risk_audio`，让声学检测读取增强前音频：

```json
{"path": "/absolute/original.wav", "sample_rate": 16000,
 "start_sample": 160000, "end_sample": 208000, "channel": 0}
```

其时长必须与当前 chunk 一致。`prepare_rover.canonical` 为新 manifest 保留这一信息；原音频与增强音频各自使用真实采样率和样本范围。候选对齐始终使用 FA 实际选定音频。

接口依据：[PANNs 官方代码](https://github.com/qiuqiangkong/panns_inference)、[pyannote 官方模型说明](https://huggingface.co/pyannote/speaker-diarization-community-1)。

## 规则与语言门控

优先级为：检测失败 → 转录不确定 → 未隔离的重叠 → 语言能力 → 风险句候选。填充词、连续重复、词尾截断只作为规则风险信号。没有零时长的句子同样可触发；相邻选中完整句子合并，同一句多个事件只运行一次。已经完整使用 Nyra 的句子不重复对齐。

开启模块后，语言门控也作用于原零时长回退；转录不确定和目标音轨不明确时保留原结果并记录原因。默认 MFA 资源语言为英语，Nyra 仅允许英语。`input_language: null`、未知语言、显式 `language_verified: false` 不能继承 Qwen 的缺省 English；明显中英混说进入复核。输入行应明确提供其真实 `language`。语言门控不是音频语言识别器。

中文 MFA 必须显式配置一对真实可用、匹配的资源；该声明会覆盖入口的英文默认路径：

```json
{"version": 1, "mfa_language": "zh",
 "mfa_dictionary": "/absolute/chinese.dict",
 "mfa_acoustic_model": "/absolute/chinese.zip"}
```

只有语言声明而没有资源对时直接报错，避免把中文交给默认英文 MFA。`mfa_language: "disabled"` 可屏蔽 MFA。此模块没有新增中文模型或保证中文零时长全部修复。

## 输出与验证

每次调用保存 `complex_handling/config.json`、`detections.jsonl`、`routes.jsonl`、`tasks.jsonl`、`candidates.jsonl` 和 `runtime.json`。词序、文本覆盖、正时长、裁剪区间和未修改邻句边界全部合法时，候选记为 `shadow_accepted`，并始终有 `applied: false`。非法边界直接拒绝，不裁边凑合法结果；原始候选及拒绝原因保留。事件标签暂不自动插入对齐文本，以免在没有验证的情况下改变词表。

检测缺失与检测错误分别记录；审计中的非有限数值序列化为 `null`，对应的失败/拒绝原因保留。阈值目前是初始参数，没有经过数据集校准；没有人工真值时不能把候选结构验收当成边界准确率。

streaming worker 将上述 JSON/JSONL 归档到每个 block 的同名目录；临时裁剪 WAV 不归档，任务保留源音频、通道、采样率和区间以便重建。原预测与审计文件继续按现有格式保存。

```bash
python -m unittest discover -s tests -v
```

测试使用模拟检测/对齐模型，验证规则、时间坐标、语言门控、整句去重、拒绝与主入口/归档接入；不加载真实模型权重。
