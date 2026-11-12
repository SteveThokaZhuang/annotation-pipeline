# Third-party source

Project-specific code has no newly selected license. This file does not relicense it or the dependencies. Model weights are excluded and retain their own download terms.

| Source in this repository | Upstream | Retained license |
| --- | --- | --- |
| `vendor/qwen_asr` | https://github.com/QwenLM/Qwen3-ASR | `vendor/qwen_asr/LICENSE` (Apache-2.0) |
| `vendor/fireredvad` | https://github.com/FireRedTeam/FireRedVAD | `vendor/fireredvad/LICENSE` (Apache-2.0) |
| `models/funasr/source` | https://github.com/modelscope/FunASR | `models/funasr/source/LICENSE` (MIT) |
| `models/firered-asr2/scripts/FireRedASR2S` | https://github.com/FireRedTeam/FireRedASR2S | Native LICENSE; SOURCE_COMMIT retained |
| `models/deepfilternet3/DeepFilterNet/df` | https://github.com/Rikorose/DeepFilterNet | `models/deepfilternet3/LICENSE`, LICENSE-MIT, LICENSE-APACHE |
| `models/dnsmos/scripts/upstream` | https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS | Native `upstream/LICENSE` retained |
| `models/roverfusion/vendor/SCTK` | https://github.com/usnistgov/SCTK | Native LICENSE.md and DISCLAIMER retained |
| `models/roverfusion/vendor/hystoc` | https://github.com/BUTSpeechFIT/hystoc | Native LICENSE retained |

Python source was copied from the existing installation. The source snapshot in `validation/original-source-hashes.json` identifies the files inherited from it; it is not a claim that the snapshot matches the latest upstream revisions. Historical source-only compatibility fixes in those directories are retained. The new pipeline-specific integration changes live primarily in `pipeline/audio_pipeline/` and the model `final_script` adapters; DNSMOS session/path handling and ROVER source building are also adapted for this repository. Training tools in vendored dependencies are not the supported entrypoint for this pipeline.

The license files for Qwen, FireRedVAD and the full DeepFilterNet license alternatives were restored from their upstream repositories where the installation's source snapshots lacked them. Preserve upstream copyright headers and these notices when distributing.
