# dialogrs

**Work in progress**

Speaker-attributed transcription in Rust: [scriptrs](https://github.com/avencera/scriptrs)
(Parakeet TDT) for what was said, [speakrs](https://github.com/avencera/speakrs)
for who said it. Both run natively on an NVIDIA GPU, without Python, PyTorch or
ONNX Runtime.

`DialogPipeline` transcribes and diarizes the same audio at the same time, splits
the transcript into words with timestamps, and gives each word the speaker whose
turn it falls in:

```rust,ignore
use dialogrs::DialogPipeline;
use scriptrs::{LongFormTranscriptionPipeline, ModelBundle, ParakeetDeployment};
use speakrs::{ExecutionMode, OwnedDiarizationPipeline};

let bundle = ModelBundle::from_native_dir("models", ParakeetDeployment::v3());
let transcriber = LongFormTranscriptionPipeline::from_bundle(bundle)?;
let diarizer = OwnedDiarizationPipeline::from_pretrained(ExecutionMode::Cuda)?;
let mut pipeline = DialogPipeline::new(transcriber, diarizer);

// mono f32 at 16 kHz
let dialog = pipeline.run(&audio)?;
for turn in &dialog.turns {
    println!("{:?} [{:.1}-{:.1}] {}", turn.speaker, turn.start, turn.end, turn.text());
}
```

It builds against local checkouts of scriptrs and speakrs for now
(`../scriptrs`, `../speakrs`), with the `cuda` feature on Linux.

## Benchmark

`examples/bench.rs` runs a directory of 16 kHz mono wavs, and
`bench/whisperx/run.py` runs WhisperX on the same files with the same output
layout. `bench/score.py` scores both for WER, cpWER (speaker-attributed WER) and
DER on Earnings-21.

On all 44 Earnings-21 earnings calls (39 hours) on one RTX 5090, end to end
(transcription and diarization, audio already in memory):

| Pipeline | WER | cpWER | DER | Speed (x real time) |
|---|---|---|---|---|
| dialogrs, Parakeet v3 + speakrs `cuda-fast` | 8.24% | **15.77%** | **9.21%** | **1822x** |
| dialogrs, Parakeet v3 + speakrs `cuda` | 8.24% | 16.66% | 9.69% | 1401x |
| dialogrs, Parakeet v2 + speakrs `cuda` | 10.17% | 18.09% | 9.69% | 1425x |
| WhisperX 3.8.6, large-v3 (float16, batch 16) + pyannote community-1 | 9.74% | 17.82% | 9.70% | 79x |

WER and cpWER use Whisper's English normalizer. cpWER pools each speaker's words
against the reference speakers, since the Earnings-21 references have no word
times. DER uses a zero collar and scores overlapped speech.

### Speed relative to WhisperX

Wall time for all 44 calls, with WhisperX as 1x:

| Pipeline | Wall time | vs WhisperX |
|---|---:|---:|
| WhisperX 3.8.6, large-v3 + pyannote community-1 | 29 min 44 s | 1x |
| dialogrs, Parakeet v3 + speakrs `cuda` | 1 min 41 s | **17.7x** |
| dialogrs, Parakeet v2 + speakrs `cuda` | 1 min 39 s | **18.0x** |
| dialogrs, Parakeet v3 + speakrs `cuda-fast` | 1 min 18 s | **23.0x** |
