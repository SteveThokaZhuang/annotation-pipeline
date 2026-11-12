# Isolated performance version 2

The code baseline is the read-only dated snapshot named in `BASELINE.json`. Original production code and outputs remain in `Pipeline_github`. This copy writes to `Pipeline_github_v2/outputs`; its temporary work uses `/tmp/shared-storage/audio-pipeline-github-v2/scratch`. Existing Python environments, checkpoint snapshots, source cache and cooperative GPU locks are referenced, not duplicated. Original and v2 run histories are separate.

## What changed

| Area | Implementation |
| --- | --- |
| Checkpoint persistence | All stages support grouped results independently of inference batch size. A bounded buffer fills a checkpoint group while one prior group is saved. Single-recording DNSMOS and enhancement can overlap inference with saving. Alignment runs batches of 8 but saves groups of 128. Errors and final partial groups remain resumable. |
| State updates | v2 resource profiles disable the redundant pre-inference `running` database transaction. Completed/failed checkpoints remain durable. Progress reports include active worker counts. Enable `optimization.record_running_state` to restore per-batch running rows. |
| Ingest | Up to four concurrent metadata readers per worker; 128-file checkpoint groups. It still reads headers, not whole-file checksums or decoded audio. |
| VAD | Up to 8 independent feature sequences per model forward, with padding masks at every FSMN layer. Native long-recording split boundaries and per-recording postprocessing are retained. Prepared arrays survive batch failure so smaller/single-item retries can work after spool cleanup. |
| Decoded RAM cache | A per-process native-rate float32 recording LRU avoids reopening and decoding a recording for every ASR/alignment slice. Slice selection and resampling remain in the original order, preserving boundary-filter behavior. Original size/mtime changes invalidate entries. Concurrent preparation threads share one decode. |
| ASR preparation | Four CPU helpers and a 4 GiB queue budget in v2 profiles, with additional lookahead. The model/decoder algorithms, token limits and recovery rules are unchanged. |
| Enhancement | CPU preparation can compute STFT and normalized spectral features ahead of GPU work using independent recording states. A background publisher writes enhanced audio through local scratch, durably publishes it, and seeds the source cache from the existing local WAV. The completed checkpoint waits for publication. Whole-recording enhancement and recurrent context are retained. |
| Controller | A bounded in-process cache reuses validated upstream results across stages. Manifest, result-file, original-source and artifact metadata are rechecked before reuse; changed files trigger normal reloading/validation. |
| ROVER and alignment exports | Larger ROVER fusion groups and up to eight parallel final recording exports. The final head is published only after successful export. NIST voting order and alignment validation rules are unchanged. |
| Diagnostics | The doctor now respects stage checkpoint overrides. Worker reports include decoded-cache metrics; enhancement reports separate preparation waits, feature work, model/transfer, synthesis and native fallback timings. |

`model_and_transfer_seconds` includes GPU synchronization through the existing `.cpu()` result transfer. Feature-work totals can overlap inference and are not additive wall-time components. Decoded-cache statistics in the main worker do not aggregate caches inside spawned frontend helpers.

## Resource settings

In the resource YAML:

```yaml
optimization:
  async_write: true
  decoded_ram_gib_per_process: 2
  manifest_cache_gib: 8
  export_workers: 8
  record_running_state: false
  cache:
    enabled: true
    root: /tmp/shared-storage/audio-pipeline-github/cache
    max_gib: 6144
  scratch_root: /tmp/shared-storage/audio-pipeline-github-v2/scratch
stages:
  vad:
    inference_batch_size: 8
    result_batch_size: 64
    prepare_workers: 4
    prepare_lookahead: 16
    prepare_max_gib: 8
  align:
    inference_batch_size: 8
    result_batch_size: 128
```

This is an excerpt: edit the existing complete resource file, retaining devices, workers and other required settings. Public profiles retain a 50 GiB disk-cache default; v2 current-host local profiles allow 6,144 GiB (6 TiB), without allocating it in advance. This leaves room for source recordings plus larger float32 enhanced WAVs across stages; the local volume had about 50 TiB available when configured. The original profile stays at 1,200 GiB. `decoded_ram_gib_per_process` applies independently to CPU frontend helpers too; it is not a node-wide allocation. Manifest-cache accounting conservatively estimates Python object memory from serialized result sizes. Preparation budgets use estimates and exclude active model allocations. These budgets reserve no RAM in advance.

The current container has a 512 GiB memory limit and a CPU quota equivalent to 32 cores. Its host-level `free`/`nproc` numbers are larger. The selected profiles allow at most 24 model/preparation CPU slots per stage and keep stages sequential. The decoded RAM cache stores native-rate samples: caching fully resampled recordings instead would change the filter behavior at slice boundaries.

Enhancement options in pipeline YAML are `settings.enhance.prepare_spectral_features`, `async_audio_write`, `prepare_workers`, `prepare_lookahead` and `prepare_max_gib`. Oversized recordings fall back to native whole-recording preparation under the existing RAM/VRAM limits. DNSMOS keeps its previously selected eight-window model batch; checkpoint grouping is separate.

## Run this copy

On this installation, after choosing a dataset:

```bash
cd /hbt_share02/202610/ob/ob_ppl_env_data/audio_pipeline_optimized/Pipeline_github_v2
./run_v2.sh smoothconv-2spks --dry-run
./run_v2.sh smoothconv-2spks --background
```

`run_v2.sh` selects the current-host vLLM and four-GPU profiles, enhancement and resume. It takes the dataset name as its first argument. It never points outputs at the original installation. Model environments and checkpoint locations must exist; these are recorded in the ignored local profiles. Shared GPU locks coordinate execution with the original installation.

## Validation and limits

Machine-readable measurements are in `outputs/benchmarks/v2_resource_validation/`. `validation/v2-results.json` summarizes the checks and their scope. Tests cover cache identity/eviction/concurrency, exact slice/resampling output, malformed-sample handling, grouped checkpoints and unchanged cached timestamps on resume, durable enhancement publication, cache seeding, parallel exports, native frontends, and real VAD padding/splitting/recovery parity.

The 128-recording ingest component test used the same real source-file metadata and an isolated shared-filesystem output directory: old worker time 8.12 seconds, v2 0.67 seconds. This is approximately 12.1 times faster for that component and sample; it is not a whole-pipeline speedup. The RAM-cache experiment used 128 real ASR chunks from three recordings and verified bit-identical inputs; warmed times were 1.02 versus 0.84 seconds. Cold timings include import/cache effects and should not be used as a headline speedup.

Real GPU checks use only idle cooperative GPU slots. VAD regions matched on 16 real Smoothconv recordings. Enhancement publication matched native waveforms on real clips. CPU checks also exercise padded sequences, multichannel inputs and retry after spool deletion. These functional checks do not constitute human-labeled recognition-quality validation. A full-dataset v2 throughput/RTF comparison has not been run, and no fixed speedup is promised for every stage.

Model startup optimization and persistent model servers are intentionally outside this version. FireRed decoder retry behavior is unchanged. Larger buffers and more helpers cannot by themselves guarantee full GPU utilization for a small model or eliminate device synchronization inside its backend.

Reproduce the isolated ingest component comparison with the controller environment:

```bash
python benchmarks/v2_resources.py --baseline ../Pipeline_github \
  --tasks /path/to/an/existing/ingest-tasks-0.jsonl --count 128 \
  --output-root outputs/benchmarks/ingest-comparison
```

Choose the output root on the filesystem you intend to measure. This benchmark uses standalone workers, grouped durable checkpoints and real header reads; its timings do not represent the entire pipeline or model inference.

## Completed real execution checks

The final v2 CPU regression run executed 120 tests with zero failures and four optional skips. Matching environments separately passed the two native ASR frontend tests and the real DeepFilterNet spectral-preparation test. Nine dedicated resource/integration tests passed, including publication failure handling.

A real eight-clip end-to-end GPU run passed all ten stages with zero failures. ASR3 processed six high-disagreement clips; fusion and alignment produced eight final recording JSON files. Repeating the run reused all 78 completed task rows, preserved their output references and update timestamps, and launched no model workers. The cache budget changed from 1,200 to 6,144 GiB between the run and its successful cached resume.

A separate four-GPU VAD run completed on devices 0, 1, 2 and 3 with zero failures. All eight recordings had the same regions and chunk boundaries as the single-GPU reference. Real GPU enhancement checks covered five mono/stereo inputs with bit-identical output after parallel spectral preparation and asynchronous publication. These checks establish functional behavior on this sample; they do not replace a full production throughput/accuracy evaluation.

Source-cache and decoded-cache controls are separate: `--no-cache` disables the disk source cache, while `optimization.decoded_ram_gib_per_process: 0` disables decoded recording retention. Bounded preparation queues still operate independently.
