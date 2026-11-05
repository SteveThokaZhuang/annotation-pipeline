# Qwen 主导的混合强制对齐

入口 `hybrid.py` 对全部请求运行 Qwen，只对命中规则的请求运行指定回退。输入 JSONL 每行含 `id, audio, text, duration`；可选 `speaker, language`。`audio` 为绝对路径或相对 manifest 的路径，时间单位秒。运行接口不需要 words/参考边界。

```bash
export QWEN_PYTHON=/mnt/data/L202500293/obed/envs/qwenasr/bin/python
python src/v1/fa/hybrid.py \
  --manifest /path/to/manifest.jsonl \
  --output /path/to/new-output-directory \
  --policy results/fa/policy.json \
  --qwen-python "$QWEN_PYTHON"
```

使用全新输出目录。已有模型输出会被视为续跑检查点，不可在更换输入、模型或策略后复用同一输出目录。最终结果 `hybrid.jsonl` 含 `words, selected_backend, route_reason, needs_review`。候选模型失败/输出不合格会保留 Qwen 原始结果并标复核，不修改零时长为虚假正时长。当前要求英文单段音频，其他语言需重新验证回退后端。

运行环境：本次复用指定外部 Qwen Python 3.12 环境（torch2.6.0/cu124、transformers4.57.6、qwen-asr0.0.6）；Nyra 依赖放在 `models/fa-runtime/nyra`，没有改动外部环境。MFA 环境独立安装于 `models/fa-runtime/mfa`，版本快照保存在 `models/fa-runtime/*freeze*` 和 `mfa-explicit.txt`。系统需 espeak-ng。

模型：`models/qwen3-forced-aligner/checkpoints`、`models/nyra-forced-aligner/checkpoints`、`models/Montreal-Forced-Aligner/checkpoints`。`download_models.py` 下载固定 revision/版本并校验 MFA 摘要；设置 `HF_ENDPOINT=https://huggingface.co` 后运行。真实 ASR 的1.7B模型另放在 `models/qwen3-asr/checkpoints`，为已有本地文件的独立副本，来源与非固定 revision 状态已记录。

新建环境可用独立 Python3.12 venv 安装 qwen-asr0.0.6、transformers4.57.6、CUDA兼容 torch2.6.0 和本地 Nyra 包；不要在 MFA 的 Python3.14 环境混装 Qwen。MFA 可用 `conda create -p <绝对路径>/models/fa-runtime/mfa --override-channels -c conda-forge montreal-forced-aligner`，精确重建则使用本次 explicit 文件。

组件：

- `run_aligners.py`：Qwen/Nyra/ASR 的可续跑全量推理，`--retry-errors` 只重试失败行。
- `run_mfa.py`：独立 corpus、逐段 CMVN、禁用说话人自适应的 MFA 推理，保留 TextGrid 和运行命令。
- `router.py`：参考无关特征、路由及回退验收。
- `evaluate.py`：按词 DP、双参考口径、开发集选规则、重放混合结果。`--policy` 固定外部规则，跳过参数搜索。
- `insights.py`：输入完整性、问题例子、配对修复与退化；`--hash` 生成来源摘要。
- `freeze_audit.py`：只在新会议审计推理前运行，冻结 manifest 和 policy 摘要，已有预测后拒绝重新冻结。
- `validate.py`：检查模型输出齐全、ID 唯一、无执行错误。
- `write_report.py`：根据结果生成 docs/fa 中的当前报告。

离线全量对比按下列次序运行（每个轨道使用不同目录）：

```bash
HF_HUB_OFFLINE=1 "$QWEN_PYTHON" src/v1/fa/run_aligners.py --backend qwen --manifest DATA.jsonl --output RUN/qwen.jsonl
PYTHONPATH=models/fa-runtime/nyra HF_HUB_OFFLINE=1 "$QWEN_PYTHON" src/v1/fa/run_aligners.py --backend nyra --manifest DATA.jsonl --output RUN/nyra.jsonl
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 models/fa-runtime/mfa/bin/python src/v1/fa/run_mfa.py --manifest DATA.jsonl --output RUN/mfa_utterance
cp RUN/mfa_utterance/predictions.jsonl RUN/mfa.jsonl
python src/v1/fa/evaluate.py --manifest DATA.jsonl --results RUN
python src/v1/fa/insights.py --manifest DATA.jsonl --results RUN
python src/v1/fa/validate.py --manifest DATA.jsonl --results RUN
python -m unittest discover -s src/v1/fa -p 'test_*.py'
```

审计或 ASR 轨道的 `evaluate.py` 必须额外传 `--policy results/fa/policy.json`，不得重新搜索路由。基准 `evaluate.py` 可读取参考；`hybrid.py/router.py` 推理逻辑不读取参考时间。完整证据、失败的 V1、V2 修订及新会议审计均见 docs/fa。

## 单张H100四方案基准

```bash
python src/v1/fa/benchmark_single_h100.py \
  --output results/fa/ami100_single_h100_v2-rerun --gpu 0 --repeats 3
```

固定一个H100 UUID，四种方案串行、每次新进程和新预测；冻结现有V2规则，不调参。主速度为包含启动/加载/全部处理及输出的端到端RTF；Qwen和Nyra另记录加载后处理区间，该区间没有预热，不能称作稳定热推理。MFA明确仅使用CPU，8 jobs、每job BLAS/OMP线程1。

结果位于 `results/fa/ami100_single_h100_v2/`，文档见 `docs/fa/07_single_h100_v2_benchmark.md`。所有推理输入均去除了参考词边界；评分单独执行。`test_benchmark.py` 检查准确率分母、零时长失败、双边界要求和阈值容差。
