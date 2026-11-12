# Qwen3 Forced Alignment + MFA

The `source/` directory vendors the upstream implementation from
`FA_awaiting` without runtime changes. `final_script/adapter.py` is the only
Pipeline V3 integration layer: it converts Rover alignment tasks to the
upstream row format, runs Qwen forced alignment and the optional MFA sentence
repair, and returns the existing V3 `{text, chunk, words, language}` result.

Model checkpoints and MFA runtime assets remain external and are configured by
the pipeline YAML.
