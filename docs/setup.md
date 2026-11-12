# Installation and deployment

## Controller and native stages

Create new environments; do not install into an active production environment. The controller needs Python 3.10+ and `requirements/controller.txt`. GPU model workers may need Python 3.12 and compatible CUDA wheels.

`requirements/*.observed.txt` records package versions from the source installation, with local package references removed. These are reconstruction inputs, not dependency lockfiles or clean-install guarantees. In particular, editable vendored Python code is provided in this repository; native extensions such as `libdf`, TorchAudio and Kaldi must match the selected Python/Torch ABI. Select CUDA wheels for your GPU and driver before installing stage requirements. Run `python -m pip check` in each newly created environment.

| Environment key | Stages | Observed native requirements |
| --- | --- | --- |
| `firered-vad` | controller, ingest, VAD, disagreement, ROVER | `firered-vad.observed.txt` |
| `dnsmos` | DNSMOS | `dnsmos.observed.txt` |
| `denoising-comparison` | enhancement | `denoising-comparison.observed.txt` |
| `qwen3-asr` | Qwen ASR and forced alignment | `qwen3-asr.observed.txt` |
| `funasr` | FunASR | `funasr.observed.txt` |
| `firered-asr2` | FireRed ASR | `firered-asr2.observed.txt` |

Map these keys to absolute Python executables under `environments:` in your pipeline YAML. An absent entry uses the controller's Python. Alternatively export `PIPELINE_PYTHON_QWEN3_ASR`, `PIPELINE_PYTHON_FUNASR`, `PIPELINE_PYTHON_FIRERED_ASR2`, `PIPELINE_PYTHON_FIRERED_VAD`, `PIPELINE_PYTHON_DNSMOS`, and `PIPELINE_PYTHON_DENOISING_COMPARISON`. `PIPELINE_PYTHON` selects only the shell entrypoint/controller Python.

Run scripts from this repository with `run_pipeline.sh`: it sets the vendored module paths for all worker environments. Do not invoke an arbitrary installed `audio_pipeline` package from the old installation.

## Checkpoints

Download checkpoints into the paths in `config/pipeline.yaml`, or change those paths. No weights are bundled and no symlinks to the original installation are needed. Preserve model-license terms and record download revisions separately.

| Stage | Upstream artifact | Expected local layout |
| --- | --- | --- |
| ASR1 | [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Hugging Face model/tokenizer directory |
| ASR2 | [Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512) | `model.pt`, `config.yaml`, CMVN assets, `Qwen3-0.6B/` |
| ASR3 | [FireRedASR2-LLM](https://huggingface.co/FireRedTeam/FireRedASR2-LLM) | `model.pth.tar`, `asr_encoder.pth.tar`, `cmvn.ark`, complete `Qwen2-7B-Instruct/` |
| Alignment | [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Hugging Face model/tokenizer directory |
| VAD | [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) | Native VAD checkpoint plus `cmvn.ark` |
| DNSMOS | [DNS-Challenge DNSMOS](https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS) | `sig_bak_ovr.onnx`; optional `model_v8.onnx` for separate P.808 script |
| Enhancement | [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | `config.ini`, native `checkpoints/` directory |

For non-ASR stages, override a checkpoint through `stage_models: {vad: {checkpoint: /path/to/VAD}}` (similarly `dnsmos`, `enhance`, `align` and `rover`). ASR model entries each have their own `checkpoint` field.

Build ROVER with `python models/roverfusion/build_rover.py`. This compiles exported NIST C source and does not require nested `.git` directories. GCC is required. Do not copy a platform-specific binary into Git.

## vLLM

Use a separate environment for vLLM and the native encoder/frontend dependencies. Real GPU validation used vLLM 0.14.0, Torch 2.9.1, Transformers 4.57.6 and PEFT 0.15.2. It used an isolated supplemental environment inheriting the existing Qwen environment; this proves that combination executed the models, but is not a pristine-install lockfile. `requirements/vllm.txt` records selected versions. Do not install the observed native requirement files wholesale over this environment because their Torch/Transformers pins can conflict. Preserve the configured virtual environment's Python executable path, including its symlink: resolving it to a base interpreter can discard the environment's dependencies.

Read [engine constraints](engines.md), especially the FireRed beam change, export disk/RAM needs and one-GPU-per-worker rule. Keep native and vLLM output roots separate during comparison.

## Checks and storage

```bash
python scripts/doctor.py --config config/pipeline.yaml --datasets config/datasets.local.yaml
./test.sh
python benchmarks/cpu.py --output validation/cpu-benchmarks.json
python benchmarks/read_audio.py /path/to/representative.wav --scratch /your/local/nvme
```

The doctor reports missing model paths without loading models. On a source-only checkout it is expected to report missing checkpoints. The CPU suite builds no models; optional real frontend tests use `PIPELINE_TEST_VAD_CHECKPOINT`, `PIPELINE_TEST_FUNASR_CHECKPOINT`, and `PIPELINE_TEST_FIRERED_CHECKPOINT` in their matching environments. Build ROVER before running integration tests. Tests generate temporary audio and do not need production datasets.

Set `optimization.cache.root` and `optimization.scratch_root` in resource YAML to a fast local mount. All cooperating installations must use the same scheduler GPU lock directory; `PIPELINE_GPU_LOCK_ROOT` overrides it. These cooperative locks do not protect against unrelated programs that do not use them. Test execution must use unoccupied GPUs.

## Before publishing

Review project-specific licensing, model licenses and retained third-party notices. No project-wide license is assumed. Git excludes weights, generated audio, logs, exports, environments, local configuration and output state. Initialize/push a GitHub remote only after reviewing the source tree and reviewing GPU validation and performing labeled accuracy checks appropriate to your release.

The ignored current-host deployment profiles use verified native or vLLM environments, read-only model snapshots under `/tmp/shared-storage/audio-pipeline-github/model_cache`, local decoder exports, and the original installation's cooperative GPU locks. Native and vLLM full-pipeline runs and their cached resumes passed. The source cache is `/tmp/shared-storage/audio-pipeline-github/cache` with a 1,200 GiB budget; scratch is the sibling `scratch` directory. `/tmp/shared-storage` is writable XFS backed by an eight-NVMe RAID0 volume on this host. `/tmp/acs` is read-only here. These deployment profiles are excluded from Git and need recreating on another machine.

From this workspace, after configuring your dataset catalog:

```bash
PIPELINE_PYTHON=/hbt_share02/202610/ob/ob_ppl_env_data/envs/firered-vad/bin/python \
  ./run_pipeline.sh all --dataset YOUR_DATASET --datasets config/datasets.local.yaml \
  --config config/pipeline.vllm.current-host.local.yaml \
  --resources config/resources.current-host.local.yaml
```

Use `pipeline.current-host.local.yaml` for the native backend. The current-host resource profile schedules four replicas per GPU stage. Rerun the same command to resume. The cache budget can be lowered without discarding inference results.

The optional `PIPELINE_TEST_DECODER_EXPORT=1` test requires Transformers/PEFT/Safetensors and uses a tiny CPU Qwen2 model to verify that exported weights include learned LoRA updates. It does not load production ASR models or test vLLM execution.
