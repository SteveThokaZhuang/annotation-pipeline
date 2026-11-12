# Version 3: configuration and execution

V3 keeps checkpoints in their existing locations and runs entirely in the local
cache by default. It does not synchronize results to the dataset directory.
The inherited V2 launchers and profiles remain available for comparison; use
`run_v3.sh` to select the new configuration.

## Start here

```bash
bash run_v3.sh --explain-config
bash run_v3.sh all --dataset duplexconv --dry-run
bash run_v3.sh all --dataset duplexconv
bash run_v3.sh all --dataset-list duplexconv basis-conversations-1500-en
bash run_v3.sh workers-stop
```

Put `all` before `--dataset-list`: the list consumes subsequent positional names.
The list mode processes each stage across the datasets before moving to the next
stage, allowing compatible models and CPU frontends to stay loaded. A normal
single-dataset `all` invocation changes model roles between stages. Resident
workers retain models until the configured idle timeout or `workers-stop`.

| File | Purpose |
| --- | --- |
| `config/launch.v3.json` | Controller interpreter, default YAML files, Python import paths |
| `config/pipeline.v3.yaml` | Checkpoint references, stage interpreters, model semantics, segmentation geometry, output root |
| `config/datasets.v3.local.yaml` | Source paths/manifests, languages, enhancement selection |
| `config/resources.v3.yaml` | GPU assignments, CPU limits, batching, preparation, memory, cache, dispatch, checkpoints |
| `config/resources.v3.concurrent.yaml` | Optional three-stage overlap with separate GPU assignments |

YAML supports recursive `extends` with deep mapping merge; lists replace their
parents. Relative inheritance paths are relative to the containing YAML file.
Runtime paths follow the existing repository/dataset path rules. `${VARIABLE}`
expansion is supported. Use an extending file for local overrides:

```yaml
extends: /absolute/path/to/config/resources.v3.yaml
gpu:
  devices: [0, 1]
stages:
  vad:
    workers: 2
  dnsmos:
    workers: 2
  enhance:
    workers: 2
  asr1:
    workers: 2
  asr2:
    workers: 2
  asr3:
    workers: 2
  align:
    workers: 2
```

Pass it with `--resources /path/to/override.yaml`. `gpu_devices: all` expands to
the configured device list. Worker counts remain explicit. `cpu.cores: auto`
uses affinity and cgroup quota. Worker threads and preparation helpers count
toward that budget. `--explain-config` reports resolved allocations and the
source of batch/preparation overrides; each run saves effective configuration.
CLI controls win over stage resources. Conflicting explicit model/stage batch
sizes fail with `configuration.conflicts: error`; `resources_win` permits an
intentional resource override. Unknown V3 resource and stage-setting keys fail.

## Independent controls

| Control | Meaning |
| --- | --- |
| `stages.<stage>.inference_batch_size` | Maximum items in a GPU inference batch; DNSMOS uses this for recordings, with `settings.dnsmos.window_batch_size` controlling ONNX windows |
| `stages.<stage>.batch_policy.max_padded_seconds` | Limits duration multiplied by batch count, reducing padding for mixed lengths |
| `stages.<stage>.prepare_workers` | CPU preparation helpers per inference worker |
| `prepare_lookahead`, `prepare_max_gib`, `prefetch_batches` | Bound outstanding preparation and buffering |
| `optimization.task_dispatch` / stage override | Dynamically claimed GPU bundles, limited by items and audio seconds; `target_bundles_per_worker` spreads small workloads and `order` selects input or longest-first dispatch |
| `result_batch_size` and `checkpoint.max_seconds/max_mib/poll_seconds` | Persistence frequency, independent of inference batch size |
| `optimization.stage_pipelines` / `funasr_pipeline` | Bounded preparation/inference/postprocessing overlap |
| `optimization.transfers` | Pinned host buffers, asynchronous H2D copies, per-worker pinned memory cap |
| `optimization.feature_transport` | `memmap` passes array descriptors between CPU helpers and inference; `pickle` restores ordinary IPC |
| `optimization.memory` | Queue/cache reservation budget, automatic host-budget calculation, minimum available RAM |
| `optimization.monitoring` | GPU sampling and progress intervals |

An individual oversized work item runs alone; a queue budget is a scheduling
estimate, not a hard operating-system RSS limit. GPU batch feedback uses driver
headroom for vLLM, whose reserved KV memory makes PyTorch-only feedback misleading.
OOM recovery still reduces batches and isolates bad inputs. Scientific model
constants and checkpoint-imposed input constraints remain in model code.

VAD can consume completed CPU preparation in readiness order, preserving source
IDs and channel/sample coordinates. CPU frontends and GPU models persist across
compatible task bundles. DNSMOS batches independent fixed windows across files
without concatenating recordings into a synthetic speech stream. ASR keeps
independent utterances, prompts, IDs and outputs in each model batch.

The memmap and pinned-buffer paths reduce repeated serialization and allocation;
they do not remove all copies. vLLM multimodal bridges still transfer encoder
features through their supported CPU interface. A long VAD recording still
requires its native feature preparation; readiness scheduling removes waiting
for unrelated files rather than making that work disappear. Discovery caches
validate directory membership and source metadata, so a changed dataset is
rescanned. Controller manifests still hold the dataset task list in memory.

## Long recordings and fidelity

`settings.enhance.method_version: auto_v3` selects whole-recording enhancement
only when duration and estimated CPU/GPU memory fit `auto_policy`. Otherwise,
VAD-guided sample-coordinate plans split processing into bounded segments with
context and crossfades. `target_seconds`, minimum/maximum segment duration,
boundary search, context and crossfade geometry are configurable. `host_headroom_fraction`,
`gpu_headroom_fraction` and `assembly_block_seconds` control memory safety
margins and output assembly buffers. Silence stays
in the output timeline. Channels are processed independently and restored in
their original order; output rate, frame count and finite samples are checked.

`parallel_segments: true` exposes durable segment tasks to the allocated GPUs
before recording assembly. Parts are keyed by source, plan, checkpoint and
inference settings; resume can reuse valid parts. A failed part fails the parent
recording; it is not silently omitted. Segment processing changes recurrent
context relative to whole-recording enhancement, so exact waveform identity or
identical ASR accuracy is not promised. Listening and labeled validation are
needed to assess enhancement quality at joins.

`recordings-ledger.json` tracks each active recording through stages, including
failures, partial coverage, no detected speech and unrouted third-ASR chunks.
ROVER/alignment recording exports contain chunk coverage and missing chunk IDs.
All-failed/no-speech recordings remain visible in the ledger even when they
have no transcript export. Failure CSV/JSONL files categorize token limits,
invalid alignment timestamps, transport failures and other failures. Decoder
termination and timestamp validation remain strict.

## Runtime paths and ownership

`bash run_duplexconv.sh` selects the synchronization profiles
`config/pipeline.v3.sync.yaml` and `config/resources.v3.sync.yaml`. They extend
the base profiles below. Processing uses `/tmp/shared-storage/ap3/working`,
while outputs synchronize to
`/mnt/data/L202500721/Pipeline_v1_output_storage/v3/production/duplexconv`.
Edit the sync pipeline's `output_root` to change the final storage location.
Edit `optimization.working_cache` in the sync resources to change the cache root,
`sync_interval_seconds` (30), `sync_mib_per_second` (64), or `min_free_gib` (16).
The interval is the wait between synchronization passes; it is not a strict
per-file replication deadline. A final pass drains pending outputs when a
successful run reaches alignment. The local cache is retained for resume.

To synchronize that retained working tree separately:

```bash
bash run_v3.sh sync --dataset duplexconv \
  --config config/pipeline.v3.sync.yaml --resources config/resources.v3.sync.yaml
```

`working_cache.enabled: false` writes directly to `output_root`; it does not
keep local processing while disabling replication. For cache-only execution,
select the base profiles with `bash run_v3.sh all --dataset duplexconv`.
Changing profiles does not relocate earlier cache-only results automatically,
and it does not alter settings already loaded by a running controller.
Persistent workers evict models/helpers when their working-tree mapping changes,
so subsequent datasets cannot inherit the previous dataset's output mapping.

The default output root is `/tmp/shared-storage/ap3/results`, with source cache,
scratch files, discovery catalogs, feature spools and decoder exports nearby.
`optimization.working_cache.enabled: false` means this output root is the working
destination itself; there is no final-destination synchronization. Change these
paths in YAML. Keep `optimization.ipc_root` short: Linux Unix socket names have
a 107-byte usable path limit, and vLLM appends its generated name. IPC paths are
validated before model startup.

GPU owners use the same cooperative locks as the previous pipeline. One owner
serves each GPU; incompatible Python environments restart it, and model changes
evict old weights. Dynamic bundle claims are atomic. A worker crash stops that
run rather than automatically duplicating a potentially active claim; rerun
the command to resume from durable results. `stop` requests normal cancellation;
the configured shutdown grace bounds an unresponsive resident owner.

The optional concurrent profile overlaps stages whose dependency, CPU and queue
budgets permit it. It assigns ASR1/2/3 to separate GPUs and DSP/alignment to a
fourth GPU. The default profile uses sequential stages on all four GPUs.

### DNSMOS preparation across bundles

The DNSMOS stage now defaults to this dispatch policy in `resources.v3.yaml`:

```yaml
stages:
  dnsmos:
    task_dispatch:
      max_items: 32
      max_audio_seconds: 4800
      prefetch_bundles: 1
```

`max_audio_seconds` is the sum of audio durations per dispatch bundle; it does
not change the model's 9.01-second windows or inference batch limits. Increasing
it from 1200 to 4800 keeps more work inside each preparation/inference session.
`prefetch_bundles: 1` atomically reserves one upcoming bundle per GPU worker and
exposes it to the resident preparation queue while the current bundle executes.
Preparation threads and ready arrays survive the bundle boundary. Current and
upcoming work share the same `prepare_lookahead` and `prepare_max_gib` limits;
there is no second full-size preparation queue. Set `prefetch_bundles: 0` to use
ordinary bundle-local preparation. This option currently supports DNSMOS only.

Prefetching does not commit future work. A stop/crash may leave a speculative
bundle leased in the old run's queue; the next run reconstructs pending bundles
from normal durable results. Completed items retain their inference identities.
For this change, all ten inference descriptors and 100 sampled task signatures
from the active DuplexConv run were checked and remained unchanged. Model code,
window geometry and score calibration are unchanged. Changing checkpoints or
actual inference settings can still invalidate cached work as before.

A running controller keeps its existing allocation. To apply the new policy:

```bash
bash run_duplexconv.sh stop
bash run_duplexconv.sh status
# Wait until the previous run reports live: false, then resume:
bash run_duplexconv.sh --background
```

Resume validates and reuses completed work. Unfinished or not-yet-checkpointed
items can run again; no existing cache/database should be deleted. Source-cache
and final synchronization paths remain unchanged.

## Validation of this upgrade

The CPU regression run completed 177 tests with 13 skips. The final V3 test
run completed all 18 tests, including actual resident-process reuse over a Unix
socket, language changes between jobs, timed checkpoint flushing during blocked
inference, concurrent task claims, shared-array transport, and coverage exports.

Real GPU probes used two devices for DuplexConv while the other devices handled
Basis. All ten stages passed on 100 DuplexConv recordings: 1,209 chunks through
ASR1, ASR2, ROVER and alignment, plus 399 chunks routed to ASR3, with zero failures.
All 100 alignment recording exports have complete coverage. Enhancement frame
counts, channel counts and sample finiteness were checked for all 100 recordings.

DNSMOS and VAD passed on 100 Basis recordings containing 101 hours of audio.
VAD boundaries match the prior probe exactly on both datasets: 1,209 DuplexConv
chunks and 16,133 Basis chunks. DuplexConv DNSMOS window coordinates are unchanged;
the maximum difference in aggregate scores is 0.000767, attributable to different
batch composition/numerical execution.

The longest Basis recording (13,624.8 seconds) enhanced successfully: 114 segments
ran across two GPUs, and assembly reused all 114 parts. Its 653,990,400 frames at
48 kHz are finite and bit-for-bit identical to the previous sequential segmented
output. Enhancement including dispatch and assembly took 69.39 seconds; the
previous single-GPU segmented probe took 98.94 seconds. These correctness probes
used evolving configurations and are not a controlled benchmark of the final
profile. Basis downstream ASR was not rerun on all 101 hours. No labeled accuracy
or listening evaluation was performed.

Reports and effective configurations are retained under
`/tmp/shared-storage/ap3/validation`; the consolidated report is
`validation-report.json`. The full changed/new file list is `V3_UPGRADE.json` in
the repository. Probe GPU owners were shut down after validation.

## Ingest finalization on resume

The V3 resources enable:

```yaml
optimization:
  finalization:
    reuse_unchanged_ingest_manifest: true
    incremental_recording_ledger: true
```

An unchanged ingest resume still discovers sources, calculates current task
signatures and validates each saved result once. During that pass it compares
IDs, signatures, result locations and result digests with the published ingest
manifest. It also checks dataset selection, limits and semantic settings. Only
an exact, successful match reuses the existing manifest. A membership, source,
settings or result change falls back to normal processing/finalization. Empty or
failed previous publications cannot bypass normal validity checks.

A reusable manifest avoids the second ingest validation pass, rebuilding all
manifest entries, and hashing/writing a replacement large manifest. Its existing
path and modification time are preserved. Reports identify this path with
`manifest_reused: true`; progress names distinguish `reuse_manifest`,
`write_manifest` and `update_ledger` instead of implying that scan completion
means all publication work is finished.

Incremental ledger mode caches each recording's ingest-result digest in the
`ledger_sources` SQLite table. Subsequent updates reread source metadata only
for changed/new results or after loss of trust in the ledger snapshot. A small
`recordings-ledger-index.json` associates the snapshot with applied manifests
and its file fingerprint. An unchanged publication skips reading and rewriting
the large ledger JSON. A changed stage updates its recording counts, and the
compatibility JSON snapshot is rewritten only if its row data changes. This
is not an append-only JSON format: a changed snapshot is still written atomically
as a complete JSON file. Concurrent stage updates share a ledger lock.

If a known ingest-result digest changes, prior stage coverage for that recording
is marked `stale_source` until those stages publish updated counts. Missing or
malformed ledger JSON can be rebuilt from authoritative ingest results. The
first run with this feature may need a one-time metadata-index build. The SQLite
cache and small index are included in the existing output synchronization path.
No completed inference signatures are changed by these scheduling/publication
controls. Set either option to `false` to restore its original behavior.

Validation includes unchanged-resume file timestamp preservation, a single
validation scan, changed-source fallback, reading metadata for only changed
recordings, ledger repair, and working-cache synchronization/path mapping. A
read-only replay of all 209,060 existing DuplexConv ingest records confirmed that
the current published manifest is reusable. That replay took 39.63 seconds;
it excluded source rediscovery and ledger updates, so it is not a measurement of
total production resume time. No active production run was stopped for validation.
