"""WhisperX baseline for speaker-attributed transcription on a directory of wavs

WhisperX's defaults (batched faster-whisper, wav2vec2 alignment, pyannote VAD and
pyannote speaker-diarization-community-1) with large-v3 in float16. Two choices
that can only help it: batch size 16 (the README example; the CLI default is 8)
and English fixed instead of detected. Models load once; the first file runs once
untimed as warmup; each file's wall time covers transcribe, align, diarize and
word assignment on preloaded audio.

Writes, per file, `<id>.json` (turns: speaker, start, end, text) and `<id>.rttm`
(diarization, for DER), plus `texts.json` ({id: text}, for long_wer.py) and
`summary.json`, the same layout as dialogrs' bench example
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile
import whisperx
from whisperx.diarize import DiarizationPipeline

SAMPLE_RATE = 16000


def load(path: Path) -> np.ndarray:
    audio, rate = soundfile.read(path, dtype="float32")
    assert rate == SAMPLE_RATE and audio.ndim == 1, f"{path}: need 16 kHz mono"
    return audio


def turns_from_words(segments: list[dict]) -> list[dict]:
    turns: list[dict] = []
    speaker = None
    for segment in segments:
        # a segment whose alignment failed has text but no words; keep its text so
        # scoring doesn't count correctly transcribed speech as deletions
        words = segment.get("words") or [
            {"word": segment["text"].strip(), "start": segment["start"], "end": segment["end"]}
        ]
        for word in words:
            # unaligned words (digits, symbols) have no times or speaker of their own
            speaker = word.get("speaker", segment.get("speaker", speaker))
            start = word.get("start", turns[-1]["end"] if turns else segment["start"])
            end = word.get("end", start)
            if turns and turns[-1]["speaker"] == speaker:
                turns[-1]["words"].append(word["word"])
                turns[-1]["end"] = end
            else:
                turns.append({"speaker": speaker, "start": start, "end": end, "words": [word["word"]]})
    return [
        {"speaker": t["speaker"], "start": t["start"], "end": t["end"], "text": " ".join(t["words"])}
        for t in turns
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.wavs.glob("*.wav"))[: args.limit]
    audios = {path.stem: load(path) for path in paths}
    device = "cuda"
    model = whisperx.load_model(args.model, device, compute_type="float16", language="en")
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    diarizer = DiarizationPipeline(device=device)

    def run(audio: np.ndarray) -> tuple[dict, object]:
        result = model.transcribe(audio, batch_size=args.batch_size, language="en")
        result = whisperx.align(result["segments"], align_model, metadata, audio, device)
        diarization = diarizer(audio)
        result = whisperx.assign_word_speakers(diarization, result)
        return result, diarization

    run(next(iter(audios.values())))

    texts, walls = {}, {}
    for stem, audio in audios.items():
        start = time.perf_counter()
        result, diarization = run(audio)
        walls[stem] = time.perf_counter() - start

        turns = turns_from_words(result["segments"])
        (args.out / f"{stem}.json").write_text(json.dumps({"turns": turns}, indent=1))
        texts[stem] = " ".join(segment["text"].strip() for segment in result["segments"])
        with open(args.out / f"{stem}.rttm", "w") as rttm:
            for row in diarization.itertuples():
                rttm.write(
                    f"SPEAKER {stem} 1 {row.start:.3f} {row.end - row.start:.3f} <NA> <NA> {row.speaker} <NA> <NA>\n"
                )
        print(f"{stem}: {len(audio) / SAMPLE_RATE:.0f} s in {walls[stem]:.1f} s", flush=True)

    audio_s = sum(len(audio) for audio in audios.values()) / SAMPLE_RATE
    wall_s = sum(walls.values())
    (args.out / "texts.json").write_text(json.dumps(texts, indent=1))
    summary = {"backend": f"whisperx-{args.model}", "files": len(audios), "audio_s": audio_s, "wall_s": wall_s, "rtfx": audio_s / wall_s, "walls_s": walls}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "walls_s"}))


if __name__ == "__main__":
    main()
