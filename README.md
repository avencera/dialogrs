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
