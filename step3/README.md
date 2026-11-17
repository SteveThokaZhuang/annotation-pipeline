# Step 3 全双工后处理

本次发布保留原有 Qwen＋音频／规则判断逻辑。版本依据、实测范围和风险标记说明见
[RELEASE_NOTES.md](RELEASE_NOTES.md)。独立保存的文本 SFT 实验不替换本管线。
本地展示工具为 [demo/index.html](demo/index.html)，支持导入标注展示包、按版本/任务/风险筛选和原始时间轴。

推理结果另外包含 `quality` 与 `risk_items`；`risk_subsets/*.json` 单独保存疑似错判及潜在漏标。
质量标记不覆盖业务标签，也不是已校准的错误概率。筛选出低风险子集不代表整体准确率提高。

以仓库 `ff4270d` 的 V3 `align/recordings/<recording_id>_c<channel>.json`
为输入。逐 chunk 读取 `words`，用 `chunk.start + word.start/end` 恢复录音时间。
词 ID 由上游录音声道、chunk ID 和词序组成。能与 FA 词序完整匹配时，从上游转录
恢复句末标点。已有对齐直接读取。缺失、部分发布或异常的 FA 会记录原因，
需要在对应的上游 chunk 修复后再运行 Step 3。

每行配对清单指定一个原始对话 ID 与两位参与者。两者可来自同一双声道文件的不同声道，
也可来自两个同步单声道文件。FA 指向的源文件默认沿用 chunk 的 `path`。
迁移数据时可用 `audio_path` 指定文件，必要时用 `audio_channel` 指定源文件的物理声道。
相对路径以配对清单目录为基准。代入上游的真实 recording ID 和声道号。

```json
{"conversation_id":"seamless-session-id","language":"en","participants":[{"speaker":"A","recording_id":"raw-left-id","channel":0},{"speaker":"B","recording_id":"raw-right-id","channel":0}]}
```

```bash
python step3/pipeline.py --align-recordings /path/to/run/align/recordings \
  --pairs /path/to/pairs.jsonl --output /path/to/step3-precheck --prepare-only

python step3/pipeline.py --align-recordings /path/to/run/align/recordings \
  --pairs /path/to/pairs.jsonl --output /path/to/step3-run \
  --jev-checkpoint /path/to/DuplexJev-32B-Para --jev-device cuda:0 \
  --qwen-endpoint http://127.0.0.1:18770/v1/chat/completions --qwen-model qwen38-27b
```

预检查需要 NumPy、SciPy 和 SoundFile。推理还需要可导入的 `duplexjev`、
对应 checkpoint 和可用的 Qwen 服务。模型推理用新输出目录运行。

- 重叠分支使用双轨混音和 JEV 选项评分，Qwen 结合双方转录和原始时间输出
  有方向的 Interrupt / Backchannel。8 秒核心窗口各拥有自己的事件起点，
  窗口两侧各保留最多 4 秒上下文。分别核对双向打断、嵌套反馈和普通接话误报。
- 空缺分支扫描相邻同说话人词，严格大于 0.5 秒才启动判断。正式标签为
  `PAUSE`（同一发言内停顿）和 `TURN_TRANSITION`（话轮交接间隔）。
  同一说话人开启独立表达也可以是话轮交接。特别长的空缺会标明省略了中段上下文。
  每段录音按 `ceil(空缺数 × review_fraction)` 的额度复判高风险空缺；默认比例为 20%，
  小样本向上取整，所以实际复判率可能超过 20%。没有风险项则不强行复判。
  记录复判意见，最终沿用首判策略；复判当前是旁路检查，不自动覆盖首判。
  模型请求失败另记质量状态。
- `prepare_report.json` 记录接入检查。推理运行输出
  `recordings/*.json` 和 `summary.json`，保存源对话 ID、原始时间、候选判定、
  模型请求信息、故障和首判／含复判的墙钟时间与吞吐。
  参考一致率和专项误报用事先锁定的参考标签另行核对。

`summary.json` 每完成一段录音原子更新。`run_state=running` 表示尚未正常收尾，
`finished` 表示遍历完清单；仍需检查 `failures_by_stage` 和 `completion`。
`reviewed_gaps` 统计启动复判的空缺，`completed_reviews` 只统计两次复判调用都成功的空缺。
`review_rate` 的分母是全部空缺；`review_completion` 的分母是已启动复判的空缺，两者不同。
吞吐按实际墙钟时间计算；模型加载时间单列，不包含在 `total_wall_seconds` 内。
录音串行处理，同一录音的空缺请求按 `--workers` 并发。JEV 数量按候选评分调用计，
单次内部使用 4 次选项排列，不等同于 HTTP 请求数。
`provenance.json` 保存清单哈希、代码/提示词哈希及运行参数。每次使用新的输出目录；
当前没有跨进程断点续跑，不会覆盖旧结果。

可在不加载大模型的环境运行代码回归测试（需要 NumPy、SciPy、SoundFile）：

```bash
python -m unittest discover -s step3 -p 'test_*.py' -v
```

测试包括双向打断与嵌套反馈的事件保留、窗口边界、词 ID 与绝对时间、异常模型输出、
失败复判统计、双轨混音，以及上游 FA 对双声道文件的声道选择和裁切。
随机时间窗口测试比较 1,000 组查询与穷举结果；它检验代码一致性，不代表模型语义准确率。

Step 2 的 FA 适配器对多声道音频先选择目标声道，再截取并对齐。此前生成的多声道 FA
如需确认没有串轨，应先抽查。由两份独立单声道文件生成的 FA 可直接做 Step 3 预检查。
首轮真实数据验证需提供有同步双轨或明确声道归属的录音。来源只有单轨混音的数据
还需提供可用的词级说话人归属或接入单独的归属流程。

Seamless 发布的词级对齐也可直接输入。把 `--align-recordings` 换成
`--seamless-root /path/to/seamless_fd`，配对清单每行提供 `conversation_id`。
程序从官方索引获取双轨路径、说话人和原始词序。可选字段 `interval` 指定
读取的音频区间，`core` 指定区间内纳入测试的事件起点或空缺中心。
这条入口使用数据集发布的 FA，V3 生成的录音导出仍用 `--align-recordings`。
