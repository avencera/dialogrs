#!/usr/bin/env bash
# The mixed benchmark (bench/sets.py) on gpu0: `mixed.sh dialogrs` runs speakrs
# `cuda` and `cuda-fast`, `mixed.sh whisperx` the WhisperX baseline. Every set is
# English, so both decode English only. A set with a summary.json is skipped, so a
# preempted job resumes where it stopped
set -euo pipefail
cd "$(dirname "$0")/.."
SETS=_scratch/sets
O=_scratch/mixed
MODELS=/shared/models/scriptrs-native
SET_NAMES="apptek edacc chime6 ami-ihm ami-sdm earnings21"

done_already() { [ -f "$1/summary.json" ]; }

case "${1:?dialogrs or whisperx}" in
  dialogrs)
    cargo build --release --features cuda --example bench
    for mode in cuda cuda-fast; do
      for set in $SET_NAMES; do
        out=$O/dialogrs-$mode/$set
        done_already $out && continue
        echo "== dialogrs $mode $set"
        target/release/examples/bench --models $MODELS --language en --diarization $mode \
          --wavs $SETS/$set/wav --out $out | tail -1
      done
    done
    ;;
  whisperx)
    for set in $SET_NAMES; do
      out=$O/whisperx/$set
      done_already $out && continue
      echo "== whisperx $set"
      _scratch/whisperx-venv/bin/python -I bench/whisperx/run.py --wavs $SETS/$set/wav --out $out | tail -1
    done
    ;;
esac
