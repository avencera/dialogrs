"""Score speaker-attributed transcripts on Earnings-21: WER, cpWER and DER

Each run directory holds `<id>.json` (turns: speaker, start, end, text) and
`<id>.rttm`, as `examples/bench.rs` and `bench/whisperx/run.py` write them.

- WER: all turns' text against the .nlp reference, pooled over files, both through
  Whisper's English normalizer with the reference built as scriptrs'
  `dev-tools/nemo/long_wer.py` builds it
- cpWER: meeteval's concatenated minimum-permutation WER, each speaker's words
  joined in time order against the reference speakers from the .nlp speaker
  column, pooled over files (the .nlp has no word times, so no tcpWER)
- DER: pyannote.metrics against the RTTM references, collar 0 and overlap scored,
  speakrs' convention

Every run scores only the files all runs share, so the numbers compare the same
recordings
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jiwer
import meeteval.wer
from meeteval.wer.matching.cy_levenshtein import levenshtein_distance
from meeteval.wer.wer.cp import _minimum_permutation_assignment
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from whisper_normalizer.english import EnglishTextNormalizer

# meeteval's sanity limit; Earnings-21 has a 20-speaker call, so a diarizer can
# legitimately exceed it
MEETEVAL_MAX_SPEAKERS = 20


def cp_errors(ref: dict[str, str], hyp: dict[str, str]) -> tuple[int, int]:
    """cpWER errors and reference word count for one file"""
    if max(len(ref), len(hyp)) <= MEETEVAL_MAX_SPEAKERS:
        cp = meeteval.wer.cp_word_error_rate(ref, hyp)
        return cp.errors, cp.length

    # the same Hungarian assignment and Levenshtein distance meeteval uses behind
    # the guard; with string inputs that is all cp_word_error_rate adds
    ref_words = {speaker: text.split() for speaker, text in ref.items()}
    hyp_words = {speaker: text.split() for speaker, text in hyp.items()}
    _, errors, _ = _minimum_permutation_assignment(ref_words, hyp_words, levenshtein_distance, [])
    return errors, sum(len(t.split()) for t in ref.values())


NLP = Path("/shared/datasets/earnings21/nlp_references")
RTTM = Path("/home/praveen/code/speakrs/fixtures/datasets/earnings21/rttm")


def reference(stem: str) -> list[tuple[str, str]]:
    """(speaker, token with punctuation) per .nlp row"""
    rows = (NLP / f"{stem}.nlp").read_text().splitlines()[1:]
    fields = [row.split("|") for row in rows if row.strip()]
    return [(f[1], f[0] + f[4]) for f in fields]


def rttm(path: Path) -> Annotation:
    annotation = Annotation()
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "SPEAKER":
            start, duration = float(parts[3]), float(parts[4])
            # one track per speaker, so two speakers with identical times both survive
            annotation[Segment(start, start + duration), parts[7]] = parts[7]
    return annotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--per-file", action="store_true")
    args = parser.parse_args()

    norm = EnglishTextNormalizer()
    stems = sorted(
        set.intersection(*({p.stem for p in run.glob("*.json") if p.stem.isdigit()} for run in args.runs))
    )
    refs = {stem: reference(stem) for stem in stems}
    ref_text = {stem: norm(" ".join(token for _, token in refs[stem])) for stem in stems}
    ref_speakers = {}
    for stem in stems:
        by_speaker = defaultdict(list)
        for speaker, token in refs[stem]:
            by_speaker[speaker].append(token)
        ref_speakers[stem] = {s: norm(" ".join(t)) for s, t in by_speaker.items()}

    for run in args.runs:
        hyp_text, cp_error_total, cp_length, lines = {}, 0, 0, []
        speaker_counts = []
        der = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        for stem in stems:
            turns = json.loads((run / f"{stem}.json").read_text())["turns"]
            hyp_text[stem] = norm(" ".join(turn["text"] for turn in turns))
            by_speaker = defaultdict(list)
            for turn in turns:
                by_speaker[str(turn["speaker"])].append(turn["text"])
            hyp_speakers = {s: norm(" ".join(t)) for s, t in by_speaker.items()}
            errors, length = cp_errors(ref_speakers[stem], hyp_speakers)
            cp_error_total += errors
            cp_length += length
            speaker_counts.append((len(ref_speakers[stem]), len(hyp_speakers)))
            der(rttm(RTTM / f"{stem}.rttm"), rttm(run / f"{stem}.rttm"))
            if args.per_file:
                lines.append(f"  {stem} cpWER {100 * errors / max(length, 1):.2f}% speakers {len(ref_speakers[stem])} -> {len(hyp_speakers)}")

        wer = jiwer.process_words([ref_text[s] for s in stems], [hyp_text[s] for s in stems]).wer
        exact = sum(ref == hyp for ref, hyp in speaker_counts)
        summary = run / "summary.json"
        summary = json.loads(summary.read_text()) if summary.exists() else {}
        # the summary's speed covers every file the run timed, so it only describes
        # these scores when the run timed exactly the scored files
        timed_same = set(summary.get("walls_s", {})) == set(stems)
        speed = f"  RTFx {summary['rtfx']:.0f}x" if timed_same and summary.get("rtfx") else ""
        print(
            f"{run}: WER {100 * wer:.2f}%  cpWER {100 * cp_error_total / cp_length:.2f}%  "
            f"DER {100 * abs(der):.2f}%  speaker count exact {exact}/{len(stems)}"
            f"{speed} over {len(stems)} files"
        )
        if lines:
            print("\n".join(lines))


if __name__ == "__main__":
    main()
