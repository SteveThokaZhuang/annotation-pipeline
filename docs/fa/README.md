# AMI-100 / Qwen 主导混合 FA 实验

已完成原始100条的参考文本三模型对照、真实ASR轨道、混合实现、实际按需运行，以及冻结后的253条新会议审计。现有AMI词边界是自动对齐银标准；独立人工金标准仍待建立，未伪称完成听审。

- [最新：V2单张H100准确度和RTF对比](07_single_h100_v2_benchmark.md)
- [V2具体设计与高RTF原因](08_hybrid_v2_design_and_rtf.md)
- [Qwen前预路由与问题段粗定位调研](09_pre_qwen_routing_research.md)
- [参考标准与评判协议](01_reference_and_protocol.md)
- [AMI-100当前结果与修复统计](02_results.md)
- [最终V2混合路由](03_hybrid_routing.md)
- [真实ASR转录问题](04_asr_findings.md)
- [新会议审计和实际按需执行](05_external_audit.md)
- [问题汇总分类与机制](06_issue_taxonomy.md)
- [V1失败结果](02_v1_failed_results.md)及[失败路由](03_v1_failed_routing.md)
- [代码入口与复现](../../src/v1/fa/README.md)

结果目录 `results/fa/`，模型权重在 `models/` 对应模型的 `checkpoints/`。Qwen主导V2在原集过滤MAE为75.41ms（Qwen88.27ms）；新会议过滤MAE为144.26ms（Qwen163.20ms）。Nyra在部分轨道整体更好，MFA也存在严重尾部偏移，不能宣称一个后端全面解决另一后端的问题。
