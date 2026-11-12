#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/pipeline:$ROOT/vendor:$ROOT/models/funasr/source:$ROOT/models/deepfilternet3/DeepFilterNet${PYTHONPATH:+:$PYTHONPATH}"
exec "${PIPELINE_PYTHON:-python3}" "$ROOT/run.py" "$@"
