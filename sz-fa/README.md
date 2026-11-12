# Pure MFA pipeline

此目录的 FA 部分已替换为纯 MFA CPU 基线。`run_fa_pipeline.py` 先运行上游 V3 的 ingest、DNSMOS、VAD、enhance、ASR1、ASR2、disagreement、ASR3、ROVER，然后从 ROVER recording JSON 读取 chunk，调用 `src/run_recordings_mfa.py` 分批执行 MFA。

纯 MFA 的核心代码来自 `/mnt/data/L202500721/stevezhuang/fa/src/mfa/process.py` 及 `/mnt/data/L202500721/stevezhuang/fa/src/run_recordings_mfa.py`。它不加载 Qwen forced aligner，也不启动 GPU FA worker；MFA 运行在 CPU 上，使用 `models/mfa/bin/python`、English US ARPA acoustic model 和 dictionary。

```bash
python3 run_fa_pipeline.py --output-root runs/oto141_pure_mfa
```

输出包括 `upstream_pipeline.log`、`pure_mfa.log`、`pure_mfa/summary.json`、`pure_mfa/runtime.json`、分批预测、`fa_report.json` 和 `call_stack.json`。
