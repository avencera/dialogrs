#!/usr/bin/env bash
# scoring venv in _scratch (wiped weekly; rerun to recreate)
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 _scratch/score-venv
VIRTUAL_ENV=_scratch/score-venv uv pip install meeteval jiwer whisper-normalizer pyannote.metrics
