"""Score speaker-attributed transcripts on the mixed benchmark: WER, cpWER and DER

`bench/sets.py` builds the sets: `<sets>/<set>/{ref,rttm}/<id>.*`. Each run is a
directory with one subdirectory per set holding `<id>.json` (turns: speaker,
start, end, text), `<id>.rttm` and `summary.json`, as `examples/bench.rs` and
`bench/whisperx/run.py` write them.

- WER: all turns' text against all reference text, pooled over files, both
  through Whisper's English normalizer
- cpWER: meeteval's concatenated minimum-permutation WER, each speaker's words
  joined in time order against each reference speaker's, pooled over files
  (Earnings-21 has no word times, so no tcpWER)
- DER: pyannote.metrics against the RTTM references, collar 0 and overlap scored,
  speakrs' convention

Each set reports its own numbers; "all" pools every file, and "mean" averages the
sets so the long Earnings-21 calls don't dominate. Every run scores only the files
all runs share, so the numbers compare the same recordings
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
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

NORM = EnglishTextNormalizer()

# cpWER never needs times, so meeteval's note for every untimed file is noise
logging.getLogger("preprocess").setLevel(logging.ERROR)


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


def by_speaker(segments: list[dict]) -> dict[str, str]:
    texts = defaultdict(list)
    for segment in segments:
        texts[str(segment["speaker"])].append(segment["text"])
    return {speaker: NORM(" ".join(t)) for speaker, t in texts.items()}


def rttm(path: Path) -> Annotation:
    annotation = Annotation()
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "SPEAKER":
            start, duration = float(parts[3]), float(parts[4])
            # one track per speaker, so two speakers with identical times both survive
            annotation[Segment(start, start + duration), parts[7]] = parts[7]
    return annotation


@dataclass
class Score:
    """Error counts for a group of files, so groups can be pooled"""

    refs: list[str] = field(default_factory=list)
    hyps: list[str] = field(default_factory=list)
    cp_errors: int = 0
    cp_length: int = 0
    der: DiarizationErrorRate = field(default_factory=lambda: DiarizationErrorRate(collar=0.0, skip_overlap=False))
    files: int = 0
    exact_speakers: int = 0
    audio_s: float = 0.0
    wall_s: float | None = 0.0

    def wer(self) -> float:
        return jiwer.process_words(self.refs, self.hyps).wer

    def cpwer(self) -> float:
        return self.cp_errors / self.cp_length

    def rtfx(self) -> float | None:
        return self.audio_s / self.wall_s if self.wall_s else None

    def absorb(self, other: "Score") -> None:
        self.refs += other.refs
        self.hyps += other.hyps
        self.cp_errors += other.cp_errors
        self.cp_length += other.cp_length
        for name, value in other.der.accumulated_.items():
            self.der.accumulated_[name] = self.der.accumulated_.get(name, 0.0) + value
        self.files += other.files
        self.exact_speakers += other.exact_speakers
        self.audio_s += other.audio_s
        self.wall_s = None if self.wall_s is None or other.wall_s is None else self.wall_s + other.wall_s


def score_set(sets: Path, name: str, run: Path, stems: list[str]) -> Score:
    score = Score()
    for stem in stems:
        reference = json.loads((sets / name / "ref" / f"{stem}.json").read_text())["segments"]
        turns = json.loads((run / name / f"{stem}.json").read_text())["turns"]
        score.refs.append(NORM(" ".join(s["text"] for s in reference)))
        score.hyps.append(NORM(" ".join(t["text"] for t in turns)))
        ref_speakers, hyp_speakers = by_speaker(reference), by_speaker(turns)
        errors, length = cp_errors(ref_speakers, hyp_speakers)
        score.cp_errors += errors
        score.cp_length += length
        score.der(rttm(sets / name / "rttm" / f"{stem}.rttm"), rttm(run / name / f"{stem}.rttm"))
        score.files += 1
        score.exact_speakers += len(ref_speakers) == len(hyp_speakers)

    summary_path = run / name / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    walls = summary.get("walls_s", {})
    # speed only describes these scores when the run timed exactly the scored files
    if set(walls) == set(stems) and summary.get("audio_s"):
        score.audio_s, score.wall_s = summary["audio_s"], sum(walls.values())
    else:
        score.wall_s = None
    return score


def row(label: str, score: Score) -> dict:
    return {
        "set": label,
        "files": score.files,
        "wer": score.wer(),
        "cpwer": score.cpwer(),
        "der": abs(score.der),
        "exact_speakers": score.exact_speakers,
        "audio_s": score.audio_s,
        "wall_s": score.wall_s,
        "rtfx": score.rtfx(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="also write every row here")
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()

    names = sorted(
        set.intersection(*({p.name for p in run.iterdir() if p.is_dir()} for run in args.runs))
        & {p.name for p in args.sets.iterdir() if p.is_dir()}
    )
    results = {}
    for run in args.runs:
        rows, total = [], Score()
        for name in names:
            stems = sorted(
                set.intersection(*({p.stem for p in (r / name).glob("*.json") if p.stem != "summary" and p.stem != "texts"} for r in args.runs))
            )
            score = score_set(args.sets, name, run, stems)
            rows.append(row(name, score))
            total.absorb(score)

        rows.append(row("all", total))
        per_set = rows[:-1]
        rows.append(
            {
                "set": "mean",
                "files": total.files,
                **{k: sum(r[k] for r in per_set) / len(per_set) for k in ("wer", "cpwer", "der")},
                "exact_speakers": total.exact_speakers,
                "audio_s": total.audio_s,
                "wall_s": total.wall_s,
                "rtfx": total.rtfx(),
            }
        )
        results[str(run)] = rows

        print(f"\n{run}")
        print(f"  {'set':<11} {'files':>5} {'WER':>7} {'cpWER':>7} {'DER':>7} {'spk ok':>7} {'RTFx':>7}")
        for r in rows:
            rtfx = f"{r['rtfx']:.0f}x" if r["rtfx"] else "-"
            print(
                f"  {r['set']:<11} {r['files']:>5} {100 * r['wer']:>6.2f}% {100 * r['cpwer']:>6.2f}% "
                f"{100 * r['der']:>6.2f}% {r['exact_speakers']:>3}/{r['files']:<3} {rtfx:>7}"
            )

    if args.json:
        args.json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
