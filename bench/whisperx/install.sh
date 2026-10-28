#!/usr/bin/env bash
# WhisperX baseline venv in _scratch (wiped weekly; rerun to recreate)
set -euo pipefail
cd "$(dirname "$0")/../.."
uv venv --python 3.12 _scratch/whisperx-venv
VIRTUAL_ENV=_scratch/whisperx-venv uv pip install whisperx soundfile
_scratch/whisperx-venv/bin/python -c "import whisperx, torch, importlib.metadata as m; print('whisperx', m.version('whisperx'), 'torch', torch.__version__, 'cuda', torch.version.cuda, 'pyannote', m.version('pyannote.audio'))"
