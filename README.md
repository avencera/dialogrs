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
layout. `bench/sets.py` builds the benchmark sets, `bench/mixed.sh` runs both
pipelines over them, and `bench/score.py` scores WER, cpWER (speaker-attributed
WER) and DER.

The benchmark mixes six English sets, 212 recordings and 114 hours in all, so no
one kind of audio decides the result:

| Set | Recordings | Hours | Audio |
|---|---:|---:|---|
| [Earnings-21](https://github.com/revdotcom/speech-datasets/tree/main/earnings21) | 44 | 39.3 | earnings calls, up to 20 speakers |
| [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) headset mix | 34 | 18.7 | 3 or 4 person meetings |
| AMI far-field | 34 | 18.7 | the same meetings on one room microphone |
| [EdAcc](https://groups.inf.ed.ac.uk/edacc/) test | 40 | 18.2 | accented English conversations |
| [CHiME-6](https://www.chimechallenge.org/challenges/chime6/) eval | 4 | 9.7 | dinner parties, far-field, overlapping speech |
| [AppTek call center](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues) | 56 | 9.4 | customer calls, 4 per accent across 14 accents |

Both pipelines decode English only. On one RTX 5090, end to end (transcription
and diarization, audio already in memory), averaged over the six sets:

| Pipeline | WER | cpWER | DER | Speed (x real time) |
|---|---:|---:|---:|---:|
| dialogrs, Parakeet v3 + speakrs `cuda` | **20.7%** | **28.0%** | 21.9% | 1355x |
| dialogrs, Parakeet v3 + speakrs `cuda-fast` | **20.7%** | 29.0% | 22.4% | **1617x** |
| WhisperX 3.8.6, large-v3 (float16, batch 16) + pyannote community-1 | 23.1% | 30.5% | **21.8%** | 77x |

dialogrs transcribes more accurately than WhisperX on every set and attributes
words to speakers more accurately (lower cpWER) on every set. Diarization alone
(DER) is within a point of WhisperX's pyannote on every set, slightly behind on
average.

speakrs `cuda` is the default. `cuda-fast` diarizes with a 2 s step instead of
1 s, which makes the whole pipeline about 19% faster but less accurate on every
set except Earnings-21. It also gets the number of speakers right less often:
137 of 212 recordings against 163 for `cuda`.

### Per set

| Set | WER dialogrs | WER WhisperX | cpWER `cuda` | cpWER `cuda-fast` | cpWER WhisperX | DER `cuda` | DER `cuda-fast` | DER WhisperX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Earnings-21 | **8.2%** | 9.7% | 16.7% | **15.8%** | 17.8% | 9.7% | **9.2%** | 9.7% |
| AMI headset mix | **21.5%** | 23.8% | **27.5%** | 28.3% | 30.7% | **17.0%** | 17.5% | **17.0%** |
| AMI far-field | **23.2%** | 28.3% | **30.7%** | 31.9% | 35.2% | 19.7% | 20.6% | **19.3%** |
| EdAcc | **16.0%** | 19.2% | **24.7%** | 27.6% | 25.7% | 22.8% | 23.7% | **22.1%** |
| CHiME-6 | **44.0%** | 44.7% | **54.8%** | 55.9% | 58.1% | 49.4% | 49.7% | **49.3%** |
| AppTek call center | **11.2%** | 12.6% | **13.6%** | 14.3% | 15.3% | **13.1%** | 13.5% | 13.2% |

Both dialogrs modes share one transcriber, so they have the same WER. WER and
cpWER use Whisper's English normalizer. cpWER pools each speaker's words against
the reference speakers. DER uses a zero collar and scores overlapped speech.

### Speed relative to WhisperX

Wall time for all 212 recordings (114 hours), with WhisperX as 1x:

| Pipeline | Wall time | vs WhisperX |
|---|---:|---:|
| WhisperX 3.8.6, large-v3 + pyannote community-1 | 1 h 29 min | 1x |
| dialogrs, Parakeet v3 + speakrs `cuda` | 5 min 3 s | **17.7x** |
| dialogrs, Parakeet v3 + speakrs `cuda-fast` | 4 min 14 s | **21.1x** |
