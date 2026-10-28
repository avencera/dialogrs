"""Build the mixed benchmark: English recordings with speaker-attributed references

Each set becomes `<out>/<set>/` with `wav/<id>.wav` (symlinks to 16 kHz mono
sources), `rttm/<id>.rttm` and `ref/<id>.json`: {"segments": [{"speaker", "start",
"end", "text"}]} in time order, with times null where the reference has none.
`bench/score.py` reads this layout.

| Set | Recordings | Kind |
|---|---|---|
| earnings21 | 44 calls | earnings calls, many speakers, long turns |
| ami-ihm | 34 meetings | 4-5 person meetings, headset mix |
| ami-sdm | 34 meetings | the same meetings, one far-field array mic |
| edacc | 40 test recordings | accented English conversations, 2 people |
| chime6 | 4 eval sessions | dinner parties, far-field, 4 people |
| apptek | 4 calls per accent, 14 accents | call-center calls, 2 people |

AMI and Earnings-21 audio and RTTMs come from speakrs' staged datasets, so DER
matches speakrs' own benchmark; AMI words come from the AMI manual annotations
1.6.2 (CC BY 4.0)
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

SPEAKRS = Path("/home/praveen/code/speakrs/fixtures/datasets")
EARNINGS_NLP = Path("/shared/datasets/earnings21/nlp_references")
AMI_WORDS = Path("/shared/datasets/ami-annotations/extracted/words")
FREE_EVAL = Path("/shared/datasets/cachedisk-mirror/dev-cache/free-eval-2026-09-13/sources")
APPTEK_CALLS_PER_ACCENT = 4
NITE = "{http://nite.sourceforge.net/}"


def earnings21(stem: str) -> list[dict]:
    rows = (EARNINGS_NLP / f"{stem}.nlp").read_text().splitlines()[1:]
    segments: list[dict] = []
    for fields in (row.split("|") for row in rows if row.strip()):
        token, speaker = fields[0] + fields[4], fields[1]
        if segments and segments[-1]["speaker"] == speaker:
            segments[-1]["text"] += " " + token
        else:
            segments.append({"speaker": speaker, "start": None, "end": None, "text": token})
    return segments


def ami(stem: str) -> list[dict]:
    words = []
    for path in sorted(AMI_WORDS.glob(f"{stem}.*.words.xml")):
        speaker = path.name.split(".")[1]
        for w in ET.parse(path).getroot().iter("w"):
            # punctuation is its own <w>; the normalizer would drop it anyway
            if w.get("punc") == "true" or not (w.text or "").strip() or w.get("starttime") is None:
                continue
            words.append((float(w.get("starttime")), float(w.get("endtime")), speaker, w.text.strip()))
    if not words:
        raise SystemExit(f"no AMI words for {stem}")
    return [{"speaker": s, "start": a, "end": b, "text": t} for a, b, s, t in sorted(words)]


def edacc_refs() -> dict[str, list[dict]]:
    refs: dict[str, list[dict]] = defaultdict(list)
    for line in (FREE_EVAL / "edacc/eval/test/stm").read_text().splitlines():
        parts = line.split(maxsplit=6)
        if len(parts) < 7:
            continue
        stem, _, speaker, start, end, _, text = parts
        if "IGNORE_TIME_SEGMENT_IN_SCORING" in text:
            continue
        refs[stem].append({"speaker": speaker, "start": float(start), "end": float(end), "text": text})
    return refs


def clock(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def chime6(stem: str) -> list[dict]:
    rows = json.loads((FREE_EVAL / f"chime6-eval/metadata/{stem}.json").read_text())
    segments = [
        {"speaker": r["speaker"], "start": clock(r["start_time"]), "end": clock(r["end_time"]), "text": r["words"]}
        for r in rows
    ]
    return sorted(segments, key=lambda s: s["start"])


def apptek_calls() -> dict[str, list[dict]]:
    calls = {}
    for path in sorted((FREE_EVAL / "apptek/metadata").glob("*.jsonl")):
        accent = sorted(
            (json.loads(line) for line in path.read_text().splitlines() if line.strip()),
            key=lambda call: Path(call["file_name"]).stem,
        )
        for call in accent[:APPTEK_CALLS_PER_ACCENT]:
            calls[Path(call["file_name"]).stem] = [
                {"speaker": s["speaker_id"], "start": s["start"], "end": s["end"], "text": s["text"]}
                for s in call["segments"]
            ]
    return calls


def speakrs_set(name: str, ref) -> dict[str, tuple[Path, Path, list[dict]]]:
    stems = sorted(p.stem for p in (SPEAKRS / name / "wav").glob("*.wav"))
    return {s: (SPEAKRS / name / "wav" / f"{s}.wav", SPEAKRS / name / "rttm" / f"{s}.rttm", ref(s)) for s in stems}


def write(out: Path, name: str, items: dict[str, tuple[Path, Path, list[dict]]]) -> None:
    root = out / name
    for sub in ("wav", "rttm", "ref"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    hours = 0.0
    for stem, (wav, rttm, segments) in sorted(items.items()):
        for link, target in ((root / "wav" / f"{stem}.wav", wav), (root / "rttm" / f"{stem}.rttm", rttm)):
            if not target.exists():
                raise SystemExit(f"missing {target}")
            link.unlink(missing_ok=True)
            link.symlink_to(target)
        (root / "ref" / f"{stem}.json").write_text(json.dumps({"segments": segments}))
        # 16-bit mono at 16 kHz after a 44-byte header
        hours += (wav.stat().st_size - 44) / 32000 / 3600
    print(f"{name}: {len(items)} recordings, {hours:.1f} h")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    write(args.out, "earnings21", speakrs_set("earnings21", earnings21))
    write(args.out, "ami-ihm", speakrs_set("ami-ihm", ami))
    write(args.out, "ami-sdm", speakrs_set("ami-sdm", ami))

    edacc = FREE_EVAL / "edacc/eval/test"
    write(
        args.out,
        "edacc",
        {s: (edacc / "wav" / f"{s}.wav", edacc / "rttm" / f"{s}.rttm", ref) for s, ref in edacc_refs().items()},
    )
    chime = FREE_EVAL / "chime6-eval"
    write(
        args.out,
        "chime6",
        {s: (chime / "canonical_wav" / f"{s}.wav", chime / "rttm" / f"{s}.rttm", chime6(s)) for s in ["S01", "S02", "S09", "S21"]},
    )
    apptek = FREE_EVAL / "apptek"
    write(
        args.out,
        "apptek",
        {s: (apptek / "wav" / f"{s}.wav", apptek / "rttm" / f"{s}.rttm", ref) for s, ref in apptek_calls().items()},
    )


if __name__ == "__main__":
    main()
