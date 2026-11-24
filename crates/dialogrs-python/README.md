# dialogrs

Speaker-attributed transcription for Python: [scriptrs](https://github.com/avencera/scriptrs)
(NVIDIA Parakeet TDT) for what was said, [speakrs](https://github.com/avencera/speakrs)
for who said it. Both run natively, without PyTorch or ONNX Runtime: on an NVIDIA GPU
on Linux and Windows x86_64, and on Core ML on Apple silicon Macs.

```sh
pip install dialogrs
```

```python
import dialogrs

pipeline = dialogrs.Pipeline.from_pretrained()  # Parakeet v3, 25 European languages
dialog = pipeline.transcribe_file("meeting.wav")
for turn in dialog.turns:
    print(f"{turn.speaker} [{turn.start:.1f}-{turn.end:.1f}] {turn.text}")
```

Transcription and diarization run at the same time with the GIL released. The first
`from_pretrained` downloads the models from Hugging Face into its hub cache: about
2.5 GB of Parakeet weights on Linux and Windows or 0.6 GB of Core ML models on macOS,
plus the speakrs diarization models. Setting `SCRIPTRS_MODELS_DIR` loads the Parakeet
model from that local directory instead.

## Results

`transcribe_file` decodes WAV, FLAC, MP3, Ogg Vorbis and AAC in MP4/M4A, and
`transcribe` takes 16 kHz mono samples as a float32 NumPy array or any 1-D float32
buffer. Both return a `Dialog`:

- `turns`: consecutive words from one speaker, each with `speaker` (such as
  `"SPEAKER_00"`), `start`, `end`, `text` and its `words`, which carry their own
  times and confidence
- `text`: the transcript without speakers
- `segments`: the diarization, overlapping speech kept, for RTTM output and DER
- `speakers`: a voice embedding per speaker for matching the same person across
  recordings: the L2-normalised mean of the speaker's 256-d chunk embeddings, with
  `chunk_count` and `speech_seconds` behind it. A speaker with no usable embedding,
  which is rare, has none, so look labels up rather than assume one per speaker
- `duration_seconds`

## Options

```python
pipeline = dialogrs.Pipeline.from_pretrained(
    "parakeet/v2",  # English only; the default "parakeet/v3" detects 25 languages
    mode="cuda-fast",  # diarize with a 2 s step: faster, slightly less accurate
    language="en",  # transcribe one language instead of detecting it
)
```

`mode` is `"cuda"` (the default on Linux and Windows), `"cuda-fast"`, `"coreml"` (the
default on macOS), `"coreml-fast"` or `"cpu"`; `dialogrs.available_modes()` lists the
ones the installed wheel has. `precision` sets the CUDA speech-to-text precision:
`"fp16"` (default), `"tf32"` or `"fp32"`. `Pipeline.from_dir` loads both models from
local directories instead of downloading them.

Failures raise `dialogrs.DialogrsError`, whose `code` is a stable string such as
`"cuda"` (no NVIDIA driver), `"model_download"` or `"audio_decode"`.

## Combining your own results

The two steps the pipeline runs after transcription and diarization are available on
their own, and take any objects with the right attributes, such as results from the
`scriptrs` and `speakrs` packages:

```python
words = dialogrs.words_from_tokens(transcription.tokens)
turns = dialogrs.assign_speakers(words, diarization.exclusive_segments)
```

## CUDA

On Linux and Windows the wheel needs an NVIDIA driver; pip installs cuBLAS 12 with it.
speakrs runs its own kernels and falls back to cuDNN 9 for a few layers on some GPUs:
`pip install "dialogrs[cudnn]"` adds it. A CUDA toolkit is not needed.
