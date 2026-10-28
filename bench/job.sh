#!/usr/bin/env bash
# dialogrs on Earnings-21 (gpu0): 2-file smoke run, then all 44 calls per config,
# each scored for WER, cpWER and DER
set -euo pipefail
cd "$(dirname "$0")/.."
WAVS=/home/praveen/code/speakrs/fixtures/datasets/earnings21/wav
MODELS=/shared/models/scriptrs-native
O=_scratch/runs
cargo build --release --features cuda --example bench
B="target/release/examples/bench --models $MODELS --wavs $WAVS"
$B --limit 2 --out $O/smoke
run() { name=$1; shift; $B "$@" --out $O/$name | tail -1; }
run v3-fp32-cuda
run v3-tf32-cuda --tf32
run v3-tf32-cudafast --tf32 --diarization cuda-fast
run v2-tf32-cuda --version v2 --tf32
_scratch/score-venv/bin/python -I bench/score.py $O/v3-fp32-cuda $O/v3-tf32-cuda $O/v3-tf32-cudafast $O/v2-tf32-cuda
