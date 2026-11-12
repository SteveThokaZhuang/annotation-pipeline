# Resumable audio pipeline — Version 3

Use `bash run_v3.sh` for V3. See [V3 configuration, execution and fidelity](docs/v3-performance.md) for the new controls and cache-only defaults. The commands below describe the inherited launchers.

See [v2 changes, resource settings and validation](docs/v2-performance.md) before running this version. The dated baseline is recorded in `BASELINE.json`; the original installation remains separate.

An independent copy of the existing pipeline, with portable dataset YAML, local source caching, parallel CPU preparation, batched inference and durable resume. The original installation is not used as an output destination.

## Run

Use Python 3.10+ for the controller. ASR environments can use Python 3.12. Install the stage dependencies and checkpoints described in [setup](docs/setup.md).

```bash
python3 -m pip install -r requirements/controller.txt
python3 models/roverfusion/build_rover.py
cp config/datasets.example.yaml config/datasets.local.yaml
# Edit paths/languages in datasets.local.yaml and Python/model paths in pipeline.yaml.
./run_pipeline.sh all --dataset example_english --datasets config/datasets.local.yaml --dry-run
./run_pipeline.sh all --dataset example_english --datasets config/datasets.local.yaml
```

Pipeline/model paths in configuration are relative to this repository. Paths in a dataset catalog are relative to that catalog. JSONL manifest audio paths are relative to the manifest. Environment variables such as `${DATA_ROOT}` are supported; unset variables fail early.

For four independent GPU workers per inference stage:

```bash
./run_pipeline.sh all --dataset example_english --datasets config/datasets.local.yaml \
  --resources config/resources.4gpu.yaml
```

This profile permits larger duration batches and needs hardware-specific validation. One worker owns one GPU; it does not tensor-shard a model. CPU budgets include preparation helpers. Concurrent stages remain disabled by default to avoid GPU contention.

## Dataset entries

```yaml
datasets:
  english_calls:
    path: /datasets/english_calls
    language: English
    enhance: false
    extensions: [.wav, .flac, .ogg]
    scan_workers: 16
  chinese_calls:
    manifest: /datasets/chinese_calls.jsonl
    language: Chinese
    enhance: true
```

A manifest contains one object per line, for example `{"path":"audio/001.wav","id":"call001"}`. IDs are optional; default IDs derive from the relative source path. Duplicate IDs/files are rejected. Directory entries support `include` and `exclude` lists of globs. Ingest reads headers and source size/mtime; consumers detect decoding errors while doing their work.

## Stages and resume

The order is `ingest → dnsmos / vad / enhance → asr1 / asr2 → disagreement → asr3 → rover → align`. Enhancement is optional. DNSMOS scores recordings; it does not gate VAD by default.

English disagreement is directional WER; Chinese disagreement is CER. `settings.disagreement.reference` selects ASR1 or ASR2 as the reference. Only scores **greater than** the configurable default `0.1` are routed to ASR3; missing/empty hypotheses are also routed. Scores can exceed 1.

Stop with Ctrl+C, or use the same configuration and `./run_pipeline.sh stop --dataset NAME`. Resume with the same run command and output root. Completed valid results are reused automatically. A single-stage command, such as `disagreement`, reads saved upstream manifests. Changing worker counts/cache settings does not require an ingest restart. Changing models, engine settings or inference code invalidates affected results deliberately.

FunASR with vLLM can overlap the next batch's native encoding with the current batch's decoding. Enable `optimization.funasr_pipeline.enabled` in the resource YAML; the current-host profile enables it with `lookahead_batches: 2` (current batch plus one ahead) and `max_gib: 4` for estimated queued embedding/feature memory per worker. Oversized batches run alone. The encoder has one owner thread, and the decoder retains the existing greedy generation, EOS recovery and result schema. This scheduling option does not change inference signatures: existing individual/batched checkpoints remain valid. Only the normal worker marks results complete after durable saving; queued preparation is disposable on stop or crash. Toggle the option off to use sequential execution without invalidating completed results. Restart workers to apply a change. New `pipeline_prepare_seconds`, `pipeline_capture_seconds`, `pipeline_decode_seconds`, and `pipeline_wait_seconds` metrics distinguish preparation, host transfer, decoding and waiting; preparation and decoding overlap, so their totals need not sum to wall time.

The current-host profile also enables `optimization.stage_pipelines` entries for `qwen3_asr`, `fireredasr`, `align`, `vad` and `dnsmos`. Each entry has an `enabled` switch and `max_gib` estimated queue budget per worker. Qwen, FireRed, alignment and VAD use `lookahead_batches: 2`; oversized batches run alone. Qwen prepares normalized audio/prompts ahead of its synchronous vLLM request; the encoder inside vLLM is still managed by vLLM. FireRed overlaps its native encoder with vLLM decoding. Alignment prepares audio/text/features ahead of one GPU forward. VAD overlaps the next GPU batch with CPU probability/region processing and serializes its singleton fallback. DNSMOS overlaps window construction and calibration with one serialized ONNX session runner, retaining the configured window batch size and exact calibration. Enhancement keeps its existing spectral-feature prefetch and asynchronous audio publication; recurrent model calls remain ordered. These options keep existing inference signatures and durable checkpoint formats. Disable any entry to return that stage to its former scheduling without discarding completed results. They do not enable simultaneous pipeline stages: `scheduler.max_concurrent_stages` remains separate.

Default outputs are under `outputs/production/DATASET`. `test STAGE --limit N` uses a separate `outputs/tests/` namespace. `test` mode still runs real model inference when that stage requires it. Run `./test.sh` for CPU regressions and fixture-based integration tests.

## Engines, performance and validation

Native Transformers is the default. `config/pipeline.vllm.yaml` is an alternative tested on four H100 GPUs: Qwen uses its official vLLM wrapper; FunASR and FireRed retain their native audio encoders and prompts and export the loaded trained decoder for vLLM. FireRed's vLLM profile explicitly selects beam size 1 because vLLM rejects beam search for embedding prompts. It is not equivalent to the native beam-3 profile.

Native and vLLM full-pipeline runs passed, including enhancement, ASR3 routing and alignment. Four Qwen vLLM replicas completed 768 samples with zero failures; stop/resume preserved completed checkpoints. These are functional and throughput checks, not labeled recognition-accuracy validation. See [the stage audit](docs/optimization.md), [engine details](docs/engines.md), and [validation](validation/README.md).

The default source cache is **50 GiB** under `/tmp/shared-storage/audio-pipeline-github/cache`; scratch is `/tmp/shared-storage/audio-pipeline-github/scratch`. The ignored current-host profile raises the source-cache budget to **1,200 GiB**, to retain roughly a 1 TB dataset between stages when space permits. The budget does not allocate space in advance. Model snapshots and decoder exports have separate directories and consume additional space. These paths are configurable. `/tmp` is only fast if its actual backing filesystem is fast; use an NVMe-backed mount on your machine when available. RAM preparation limits cover the queued work; active batches and model allocations need additional memory.

The code layout is `pipeline/audio_pipeline/` for orchestration, `models/*/final_script/` for adapters, `vendor/` and model source directories for reused implementations, `config/`, `tests/`, `benchmarks/`, and `docs/`. Model weights, datasets, outputs, environments and binaries are excluded from Git. Third-party notices are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). A license for project-specific code has not been selected by its owner.

V2 also supports a persistent working cache for generated output. The dedicated
`resources.duplexconv.v1-calibrated.local.yaml` enables it under
`/tmp/shared-storage/audio-pipeline-github-v2-duplexconv/working`. Results,
manifests, enhanced audio, task files, and the working SQLite database stay there
across stops and resumes. Reads prefer the working copy and fall back to the
configured final output tree. Existing final state is imported with SQLite's
backup API on the first launch; existing inference signatures remain valid.
Generated audio already in this working tree bypasses the disposable source cache.

`optimization.working_cache` controls `root`, `sync_interval_seconds` (30),
`sync_mib_per_second` (64), and `min_free_gib` (8). One background worker streams
copies with bounded buffers, publishes files atomically, and publishes a consistent
SQLite snapshot only after the artifacts. Cached state is authoritative while
unsynchronized results exist. Keep the same final output and working-cache roots
when resuming; don't run another pipeline against this same dataset/output tree.
The disposable source-cache `max_gib` does not limit or evict the working tree.
Low free space or synchronization failure stops processing with cached data retained.

A graceful stop retains pending transfers for the next launch. Finishing alignment
waits for the remaining copies. Working files are retained even after alignment;
there is no automatic deletion. If fast storage is lost, only results already
synchronized to the final disk can be recovered. To finish pending transfers
without running inference, use the matching configuration:

```bash
bash run_pipeline.sh sync --dataset duplexconv \
  --config config/pipeline.vllm.duplexconv.v1-calibrated.local.yaml \
  --resources config/resources.duplexconv.v1-calibrated.local.yaml
```

The feature takes effect on the next launch, not in workers already running.

V3 defaults to `optimization.resume.validation: trusted`. Successful stages
record small completion receipts under `control/resume` in the working output
tree. Matching inference settings, dataset policy, upstream manifest metadata,
stage publication and SQLite row versions let subsequent resumes skip task
planning, result-file scans, manifest rebuilding and repeated recording exports.
Resource-only changes such as worker counts do not invalidate these receipts.

Older completed runs adopt their existing successful manifests after matching
task signatures against SQLite once, without reopening every result. The first
adoption may still load upstream inputs for task planning. Interrupted stages
trust committed completion rows and run pending, failed or changed-signature
tasks; finalization validates the resulting outputs once. New downstream stages
still load the upstream results they need. Failed/degraded publications are not
treated as successful stage receipts.

Trusted mode also saves immutable task plans under `control/resume/plans`.
An unfinished stage with matching settings and upstream heads can reload its
inputs and signatures sequentially from local storage, without reopening all
upstream results or rebuilding the plan. Existing installations create these
plans once on the next launch. Changed settings/upstream manifests and full
validation bypass the saved plan. Stage issues and intentionally skipped inputs
are retained with the plan. A damaged or incomplete plan falls back to planning.

ASR1/2/3 continuation retains unchanged manifest entries and validates only
changed, retried or previously unsuccessful chunks during finalization. VAD
coverage counts for ledger updates are cached by the VAD manifest version.
Full validation still rechecks all chunks. New stages and non-ASR stages that
assemble routing/recording outputs can still require normal finalization;
publication of large JSON manifests and coverage ledgers still takes time.

To continue at a known stage without retrying earlier stages' reported errors,
select a starting stage explicitly (earlier outputs must already exist):

```bash
# After the current run has stopped:
bash run_duplexconv.sh \
  --resources config/resources.v3.vad-throughput.yaml \
  --resume-from-stage asr2 --background
```

The default `all` sequence still revisits earlier degraded stages so their
failed chunks can be retried. `--resume-from-stage` applies only to `all` and
also works with the concurrent scheduler. Durable result commits determine the
resume point; an in-flight, uncommitted batch may run again after interruption.
Only one controller can own an output tree; starting another background command
while one is live is rejected.

Trusted mode assumes source data and saved artifacts are unchanged outside the
pipeline. It deliberately skips source-fingerprint and artifact-integrity
checks; external additions, modifications, deletions or corruption may not be
noticed. After editing inputs/artifacts, or to audit/recover a run, use full
validation. Full mode bypasses completion receipts and rechecks individual
results, source metadata and checksums while loading upstream inputs:

```bash
bash run_duplexconv.sh \
  --resources config/resources.v3.cache-only.yaml \
  --resume-validation full
```

Use `--resume-validation trusted` to explicitly select fast resume, or set
`optimization.resume.validation` to `trusted` or `full` in the resource YAML.
Result signatures and checkpoint formats remain compatible. Changed processing
settings and upstream publications invalidate receipts automatically. Large full
scans report both validation progress and results delivered, so overall scan
progress reaches 100% after the ordered processing pass finishes.

V3 host-memory admission respects the container limit and host `MemAvailable`.
Its estimate credits a configurable fraction of clean file cache, including
active cache, while excluding shared/tmpfs, dirty, writeback and unevictable
pages. Cache accounting is an estimate of reclaimable capacity, not a memory
reservation. The default 50% credit retains a margin for cache that cannot be
reclaimed immediately. The kernel reclaims pages as allocations need them;
the pipeline does not clear system caches or delete cached recordings.

```yaml
optimization:
  memory:
    min_free_gib: 16
    cache_reclaim_fraction: 0.5
    wait_timeout_seconds: 300
    wait_poll_seconds: 2
```

All stages check admission before model loading/bundle execution. Low headroom
emits `memory_wait` progress and waits up to five minutes, with interruptible
polling. If pressure persists, the worker publishes an infrastructure failure;
the controller stops peer workers and skips expensive finalization and downstream
stages. Committed results and the previous manifest remain available for resume.
Unstarted chunks are left pending rather than being counted as millions of
transcription failures. Resource-only memory policy changes preserve existing
inference signatures. Repeated real pressure calls for smaller preparation
queues or fewer concurrent model workers, while retaining the 16 GiB minimum.

V3 output storage is selected with two configuration flags:

| `optimization.working_cache.enabled` | `sync_enabled` | Generated output behavior |
| --- | --- | --- |
| `true` | `true` (default) | Write to local working cache and synchronize to final storage. |
| `true` | `false` | Write to local working cache only, including at alignment completion. |
| `false` | ignored | Write directly to `pipeline.output_root`. |

Use `config/resources.v3.cache-only.yaml` to keep generated output on the large
local volume at `/tmp/shared-storage/ap3/working`. The working tree is retained
across stops and resumes, with no eviction or source-cache size limit. Free-space
checks still enforce `min_free_gib: 16`. No background transfer or final drain
runs in this mode. All stages write results, reports, manifests, stage locks,
stop files and run registrations to local storage. Registrations live under
`<working_root>/<output-id>/control/runs`; stage locks live under the cached
`files/locks` directory. The working-cache ownership lock is also local.
An existing final ownership lock is honored read-only for compatibility with
older runs. Cached reads can fall back to existing final results. No output or
control files are created or updated in the final output directory.

`status` and `stop` search both final and local registrations. Use the same
configuration and working-cache root as the run (the standard V3 profiles share
the same root). Global GPU reservation locks remain at `scheduler.gpu_lock_root`
so other installations still coordinate GPU ownership.

```bash
bash run_v3.sh all --dataset duplexconv --resume \
  --config config/pipeline.v3.sync.yaml \
  --resources config/resources.v3.cache-only.yaml

# Control a cache-only run, including one started with --background.
bash run_duplexconv.sh status --resources config/resources.v3.cache-only.yaml
bash run_duplexconv.sh stop --resources config/resources.v3.cache-only.yaml

# Explicitly transfer retained output later, even with sync_enabled: false.
bash run_v3.sh sync --dataset duplexconv \
  --config config/pipeline.v3.sync.yaml \
  --resources config/resources.v3.cache-only.yaml
```

Use `config/resources.v3.direct.yaml` for direct final output writes and direct
source-audio reads. This sets both `optimization.working_cache.enabled: false`
and `optimization.cache.enabled: false`. To keep source caching with direct
output writes, leave `optimization.cache.enabled: true` instead. CLI `--cache`
and `--no-cache` override only source-audio caching; scratch and model caches
are separate. For example:

```bash
bash run_v3.sh all --dataset duplexconv --resume \
  --config config/pipeline.v3.sync.yaml \
  --resources config/resources.v3.direct.yaml
```

Synchronize pending cached output before switching an existing output tree to
direct writes. Retain the same output and working-cache roots when resuming or
synchronizing. Cache-only results depend on the local volume surviving; they
have no final-storage copy until explicitly synchronized.

For CPU-limited VAD, `config/resources.v3.vad-throughput.yaml` extends cache-only
mode with eight preparation processes per GPU (32 total), a 32-recording
lookahead per worker, an 8 GiB preparation budget per worker, two overlapping
stage batches and 128-recording dispatch bundles. The configured CPU cost is
40 of 56 cores, leaving room for controller and I/O work. GPU inference settings,
result signatures, output roots and source-cache identity are unchanged, so
completed VAD results remain reusable. Other stages retain their existing settings.

```bash
# Stop the old run and wait for status to show live: false before restarting.
bash run_duplexconv.sh stop --resources config/resources.v3.cache-only.yaml
bash run_duplexconv.sh status --resources config/resources.v3.cache-only.yaml
bash run_duplexconv.sh \
  --resources config/resources.v3.vad-throughput.yaml --background
```

A CPU-only warm-cache probe on 32 recent DuplexConv recordings, while production
was running, measured median preparation times of 2.357 seconds with four
helpers, 1.249 seconds with eight and 0.947 seconds with twelve. Imports and
worker warmup were outside timing; timed work included source checks, audio
reads/resampling, native features and local array publication. Each setting ran
twice. This establishes preparation scaling on that sample, not an end-to-end
GPU speedup or a full-dataset optimum. Twelve helpers per GPU would reserve the
entire 56-core budget; the supplied profile uses eight for headroom.

`config/resources.v3.vad-max-cpu.yaml` selects twelve preparation helpers per GPU
(48 total) and a 48-recording lookahead. With the four GPU workers' two threads
each, it reserves the full 56-core scheduling budget. It preserves inference
signatures and changes only VAD resource allocation. Select it through
`--resources config/resources.v3.vad-max-cpu.yaml` after gracefully stopping the
previous run. The CPU-only sample was approximately 32% faster with twelve
helpers than eight; end-to-end scaling with all four GPUs remains unmeasured.
Higher GPU utilization is possible but is not guaranteed by more CPU helpers.

The report is retained at
`/tmp/shared-storage/ap3/benchmarks/vad-prepare-scaling-6d0006qt/report.json`.
`benchmarks/vad_preparation_scaling.py --jobs BUNDLE_JOB_JSON ...` reproduces the
CPU-only probe without reserving GPUs or modifying production state. Measure
successful audio-seconds per wall-second after restarting; low GPU utilization
can persist for this small model even when overall throughput improves.

The opt-in `vad_chunked_v1` enhancement method is isolated from the original
whole-recording adapter. For basis-conversations-1500, use
`bash run_v2_com_2_chunked.sh` and stop it with `bash run_stop_2_chunked.sh`.
The original `run_v2_com_2.sh`, its configuration, and `adapter.py` are unchanged.
The new profile is in `config/pipeline.vllm.basis-conversations-1500.vad-chunked-v1.local.yaml`
and its matching resources file. It runs English then Chinese with explicit languages.

The profile writes to `/mnt/data/L202500721/Pipeline_v1_output_storage/vad_chunked_v1`
and uses `/tmp/shared-storage/audio-pipeline-github-v2-basis-conversations-1500-vad-chunked-v1/working`.
It shares the existing source-audio cache and model exports, but never shares the
working database, enhanced outputs, ASR results, or stage heads with the original
profile. Before inference, the launcher copies available completed ingest,
DNSMOS, and VAD checkpoints into the isolated working tree. The copy prefers the
original working cache, opens its database read-only, and leaves original files
unchanged. Missing upstream heads run normally. Imported checkpoints still pass
the normal signature/resume checks. Upstream copy can be stopped with the matching
stop script and resumes safely on the next launch.

Chunk planning uses the union of raw VAD speech intervals across all channels;
it prefers shared pauses and falls back to bounded cuts during continuous speech.
No silence or other samples are removed. Defaults are a 120-second target,
180-second maximum model input (including context/overlap), 2 seconds of context
on either side, and a 250 ms complementary crossfade. Segment reads preserve the
global rational resampling phase and discard extra FIR guard samples. Each model
call gets fresh transform state and native delay compensation. The method differs
from whole-recording inference because model context is reset per segment; listen
to representative speech and joins before drawing quality conclusions.

Each enhanced segment/channel has a durable checkpoint. Missing or invalid parts
are recomputed; valid parts survive stops and can be restored from the final disk
if their local copies are absent. Assembly streams into one atomic `enhanced.wav`
with the same timeline/channel count and the existing 48 kHz downstream contract.
RF64 is used when float output exceeds the WAV/RIFF size limit. Segment parts,
plans, and complete recordings use the persistent working-cache synchronizer.
The segment method/settings, VAD result and boundary plan, native implementation,
and new helper code are included in enhancement identity. Other stages retain
existing inference identities; ASR and later results naturally change when their
enhanced input changes. No enhancement or downstream heads are imported from the
old profile. Validation covers synthetic joins/resampling/resume and short real
DeepFilterNet CPU calls; it does not establish real-speech quality or GPU throughput.
