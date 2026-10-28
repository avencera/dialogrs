#!/usr/bin/env bash
# WhisperX baseline on gpu0: 2 Earnings-21 calls as a smoke test, then all 44
set -euo pipefail
cd "$(dirname "$0")/../.."
WAVS=/home/praveen/code/speakrs/fixtures/datasets/earnings21/wav
PY="_scratch/whisperx-venv/bin/python -I"
$PY bench/whisperx/run.py --wavs $WAVS --out _scratch/whisperx-smoke --limit 2
$PY bench/whisperx/run.py --wavs $WAVS --out _scratch/whisperx-e21
