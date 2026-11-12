# ASR engine integration

`models.NAME.inference.engine` selects `transformers` or `vllm`. There is no automatic engine fallback. Use separate environments for native and vLLM profiles because their Torch/Transformers requirements can conflict.

Qwen calls `Qwen3ASRModel.LLM`, as supported by [the official implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py). The adapter collects completion metadata before the wrapper reduces request outputs to text. A token-limited row is marked failed rather than published as a successful transcript.

FunASR and FireRed use `pipeline/audio_pipeline/vllm_decoder.py`. The bridge exports the **loaded trained decoder**, merging PEFT LoRA when present, under `outputs/model_exports`. It retains the native audio encoder/projector, trained token embedding table, prompts and output formatting. Exports are atomic and locked by checkpoint/source provenance. Native checkpoints are never modified. First use needs host RAM and output disk space to export the decoder; subsequent workers reuse the export. Both bridges passed real English/Chinese GPU batches, forced token-limit failure checks and a complete pipeline run. See [measured results](../validation/README.md).

The bridge strips padding from each prompt and sends [embedding prompts to vLLM](https://docs.vllm.ai/en/latest/features/prompt_embeds/). It runs greedy decoding, verifies one output per input, and preserves stop/length handling. It rejects beam search, sampling and unsupported decoding constraints explicitly. The [vLLM implementation](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/generate/beam_search/offline.py) rejects embedding prompts in beam search. The supplied vLLM profile sets FireRed `beam_size: 1`; the native profile uses 3. Compare WER/CER before adopting that change.

One pipeline worker receives a single visible GPU. `tensor_parallel_size` must be 1. Use `resources.4gpu.yaml` for four independent replicas. Do not nest vLLM tensor parallelism or internal data-parallel workers under this scheduler.

vLLM's `gpu_memory_utilization` includes decoder weights/KV cache allocations, not a guarantee that the other native encoder fits. The bridge defaults to 0.6 to leave encoder headroom; Qwen's integrated wrapper defaults to 0.8. A lower fraction may be needed for long batches.

The vendored FunASR source also contains its upstream split-engine implementation, documented in [the official guide](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md). This pipeline's bridge deliberately reuses its established native prompt/encoder path instead. The separately converted native vLLM checkpoint has a different interface; do not point the native FunASR adapter at it.

The GPU validation exercised mixed-duration English/Chinese batches, completion limits, four Qwen replicas, full-pipeline resume and explicit engine shutdown. All allocations were released after the tests. The tested profile uses eager execution, `max_model_len: 4096` and disabled prefix caching (also disabled multimodal processor caching for Qwen). CUDA graph mode has not been validated. Upstream vLLM can still print an NCCL shutdown warning after successful cleanup.

Batch size and backend can change transcripts. Validate WER/CER against human labels before selecting a production engine, especially when changing FireRed beam size. The small repeated-clip benchmark is not a production-scale accuracy or scaling study.
