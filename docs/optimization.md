# Stage-by-stage performance audit

These are implemented mechanisms and remaining opportunities. Actual H100 GPU measurements are in [validation](../validation/README.md). Tune from worker report timings, recording lengths and actual available RAM/VRAM.

| Module | Implemented path | Remaining bottlenecks / next measurement |
| --- | --- | --- |
| Dataset discovery | Parallel top-level directory traversal; bounded concurrent stat snapshots; optional JSONL entries | One flat network directory still enumerates serially; use a manifest for repeated large scans. Controller retains the task list in RAM. |
| Ingest | Header metadata only; parallel worker shards; no payload hashing, peak scan or full decode | Metadata server latency and final state publication. |
| DNSMOS | Decode multichannel source once; parallel recording preparation; batched windows; CUDA OOM splitting | Profile ORT launch time vs window batch size; windows within one short recording are not pooled with other recordings. |
| VAD | Spawned CPU feature workers; local NumPy scratch; GPU inference overlaps prepared recordings | Native recording detection remains sequential per GPU worker; multichannel preparation still reads channels separately. The VAD spool admits an oversized recording alone. |
| Enhancement | Parallel decode/resample of future recordings; bounded queue; complete-context inference; optional channel batches; carried GRU state | Native STFT/features and synthesis remain inside enhancement inference calls. Parallelizing these needs independent DF state and waveform equivalence tests. Very long recordings retain existing CPU fallback policy. |
| ASR1 Qwen | Parallel waveform slices; recording locality; duration batches; optional official vLLM engine; per-row completion checks | Qwen's processor frontend still executes inside its wrapper. Profile processor time before replacing it; the measured large batch improved throughput, while processor overlap remains an opportunity. |
| ASR2 FunASR | Spawned native frontend/tokenization; bounded future-feature queue; no temporary WAVs in prepared path; encoder microbatches; batched decoder; optional vLLM decoder bridge | CPU-worker startup, IPC arrays and encoder-to-decoder synchronization. Native/vLLM throughput was compared on real models; human-labeled accuracy is still needed. Dither distribution is retained; `frontend_seed` optionally makes per-item RNG repeatable. |
| Disagreement | Multiple CPU workers; exact bit-parallel edit distance; batch checkpoints; parallel saved-result scans | Manifest validation and persistence can dominate after distance acceleration. WER/CER normalization and directional denominator are unchanged. |
| ASR3 FireRed | Spawned Kaldi frontend; no WAV write/read roundtrip; native padded encoder batches; bounded feature lookahead; optional trained-decoder vLLM bridge | Native projector/padding behavior is retained. Greedy-vs-beam quality must be measured. Native and vLLM GPU batches passed; beam-dependent accuracy remains unmeasured. |
| ROVER | Official NIST C binary; multiple CPU workers; grouped requests/checkpoints | Native fusion subprocess and text formatting. Benchmark larger result groups rather than replacing voting semantics. |
| Alignment | Native list inference; parallel cached waveform preparation; duration limits and OOM recovery | Per-batch text/audio padding and timestamp validation; not an autoregressive ASR decoder and does not need vLLM. |
| Resume / writer / exports | Parallel source/result scans; shared batch result indexing; local scan spool; async durable writes; bounded pending write; cancellation checks | Shared-filesystem metadata and final exports still require I/O; do not remove durability checks to hide latency. |

## Resource controls

`prepare_workers` and `prepare_max_gib` in resource YAML control waveform preparation. ASR2/3 use their model's `frontend_workers` and `frontend_max_gib` settings for spawned CPU feature workers. DNSMOS and enhancement use `settings.STAGE.prepare_workers`, `prepare_lookahead` and `prepare_max_gib`. Those recording queues use threads because decode/resample primarily call native libraries. VAD has its own process/spool budget in resource YAML.

Queued-byte bounds are based on conservative audio/feature estimates. They are not a hard limit on total process RSS: the active batch, temporary native arrays, imported worker runtimes, GPU model memory and decoder exports are separate. An item larger than the generic queue budget is loaded as the current item, without scheduling it ahead. Cancellation of a running loader retains its queue reservation until it finishes. VAD retains its existing single oversized recording policy.

`batch_size` is a maximum item count, not a guarantee. Duration padding, free VRAM, calibration (if supplied), and CUDA OOM recovery can reduce it. The four-GPU profile allows 6,400 padded seconds but this is a starting point, not a universal optimum. Compare audio-seconds/wall-second and successful sample counts, not only GPU-utilization percentages.

## Data movement

Source filesystem → leased local source cache (copy once on miss) → bounded CPU decode/resample/features → current inference batch → GPU encoder/decoder → bounded pending durable write → output filesystem. Preparation of future work overlaps the current model call. The source cache stores encoded recordings, not GPU tensors. It is shared across audio-consuming stages and evicts unleased entries by budget. Ingest and text-only stages do not copy audio into it.

Cache cleanup is budget-driven. Process temporary directories clean up normally on exit; a killed process can leave scratch behind. `/tmp` retention across container replacement/reboots is platform-dependent. Do not delete cache entries or scratch directories while active workers hold them. Stop workers before removing this installation's disposable scratch. Decoder exports live in `outputs/model_exports` and are not part of the source-cache eviction budget.

The source cache uses a local SQLite LRU inventory, so new cache entries do not rescan every stored recording. Existing managed cache files are inventoried once; file leases still prevent eviction during use. Regression checks cover inventory reuse, migration and repair of missing publication reservations. The 1,200 GiB current-host budget permits retention of a roughly 1 TB dataset, subject to actual encoded size and other local disk usage.
