"""Integration tests for the dialogrs Python package."""

from __future__ import annotations

import platform
import sys
from types import SimpleNamespace

import pytest

import dialogrs
from conftest import CLIP, CLIP_TEXT, backend_available, normalize

_CUDA_PLATFORM = sys.platform in ("linux", "win32") and platform.machine() in ("x86_64", "AMD64")


def token(text: str, start: float, confidence: float) -> SimpleNamespace:
    # any object with these attributes works, as a scriptrs.TimedToken does
    return SimpleNamespace(text=text, start=start, end=start + 0.08, confidence=confidence)


def word(text: str, start: float, end: float) -> dialogrs.Word:
    return dialogrs.Word(text, start, end, 0.9)


def test_backend_and_modes_match_platform():
    if sys.platform == "darwin":
        assert dialogrs.BACKEND == "coreml"
        assert dialogrs.available_modes() == ["CPU", "CoreML", "CoreMLFast"]
    elif _CUDA_PLATFORM:
        assert dialogrs.BACKEND == "cuda"
        assert dialogrs.available_modes() == ["CPU", "CUDA", "CUDAFast"]
    else:
        assert dialogrs.BACKEND is None
        assert dialogrs.available_modes() == ["CPU"]


def test_model_names_resolve_to_one_enum():
    for name in ("parakeet/v3", "Parakeet/V3", " parakeet/v3 ", "nvidia/parakeet-tdt-0.6b-v3"):
        assert dialogrs._model(name) is dialogrs.Model.PARAKEET_V3
    with pytest.raises(ValueError, match="'parakeet/v2', 'parakeet/v3'"):
        dialogrs._model("v3")
    assert dialogrs.Model.PARAKEET_V2.languages == ["en"]
    assert {"en", "de", "fr"} <= set(dialogrs.Model.PARAKEET_V3.languages)


def test_words_from_tokens_merges_pieces_and_punctuation():
    tokens = [
        token(" Good", 0.0, 0.9),
        token(" after", 0.2, 0.8),
        token("noon", 0.3, 0.6),
        token(",", 0.4, 0.95),
        token(" ", 0.6, 0.9),
        token("everyone", 0.7, 0.7),
    ]
    words = dialogrs.words_from_tokens(tokens)
    assert [w.text for w in words] == ["Good", "afternoon,", "everyone"]
    assert (words[1].start, words[1].end) == pytest.approx((0.2, 0.48))
    assert words[1].confidence == pytest.approx(0.6)
    # an iterator works as well as a list
    assert dialogrs.words_from_tokens(iter(tokens)) == words


def test_words_from_tokens_names_a_bad_token():
    with pytest.raises(TypeError, match="token 1"):
        dialogrs.words_from_tokens([token(" a", 0.0, 0.9), SimpleNamespace(text="b")])


def test_assign_speakers_groups_words_into_turns():
    words = [word("Hello", 0.0, 0.4), word("there.", 0.5, 0.9), word("Hi!", 1.6, 1.9), word("Bye.", 3.0, 3.4)]
    segments = [
        dialogrs.Segment(0.0, 1.0, "SPEAKER_00"),
        dialogrs.Segment(1.5, 2.0, "SPEAKER_01"),
        # any object with start, end and speaker works, as a speakrs.Segment does
        SimpleNamespace(start=2.9, end=3.5, speaker="SPEAKER_00"),
    ]
    turns = dialogrs.assign_speakers(words, segments)
    assert [(t.speaker, t.text) for t in turns] == [
        ("SPEAKER_00", "Hello there."),
        ("SPEAKER_01", "Hi!"),
        ("SPEAKER_00", "Bye."),
    ]
    assert (turns[0].start, turns[0].end) == (0.0, 0.9)
    assert turns[1].words == [words[2]]


def test_assign_speakers_without_speech_keeps_words_unlabelled():
    turns = dialogrs.assign_speakers([word("Hello", 0.0, 0.4)], [])
    assert [(t.speaker, t.text) for t in turns] == [(None, "Hello")]


def test_assign_speakers_rejects_unsorted_input():
    with pytest.raises(ValueError, match="segment 1 starts before segment 0 ends"):
        dialogrs.assign_speakers(
            [word("a", 0.0, 0.1)],
            [dialogrs.Segment(0.0, 2.0, "SPEAKER_00"), dialogrs.Segment(1.0, 3.0, "SPEAKER_01")],
        )
    with pytest.raises(ValueError, match="word 1 starts before word 0"):
        dialogrs.assign_speakers([word("b", 1.0, 1.1), word("a", 0.0, 0.1)], [])


def test_bad_settings_fail_before_the_driver_check_and_any_download():
    with pytest.raises(ValueError, match="precision"):
        dialogrs.Pipeline.from_pretrained(precision="int8")  # type: ignore[arg-type]
    with pytest.raises(dialogrs.DialogrsError) as raised:
        dialogrs.Pipeline.from_pretrained(mode="gpu")  # type: ignore[arg-type]
    assert raised.value.code == "configuration"
    with pytest.raises(dialogrs.DialogrsError) as raised:
        dialogrs.Pipeline.from_pretrained("parakeet/v2", language="de")
    assert raised.value.code == "unsupported_language"
    assert "en" in raised.value.message


@pytest.mark.skipif(not _CUDA_PLATFORM, reason="a Linux or Windows build has no Core ML")
def test_mode_missing_from_this_build_is_rejected():
    with pytest.raises(dialogrs.DialogrsError) as raised:
        dialogrs.Pipeline.from_pretrained(mode="coreml")
    assert raised.value.code == "unsupported_mode"
    assert str(raised.value).endswith("[unsupported_mode]")


@pytest.mark.skipif(
    dialogrs.BACKEND != "cuda" or backend_available(),
    reason="checks the error on a Linux or Windows machine without an NVIDIA driver",
)
def test_missing_driver_fails_before_download():
    with pytest.raises(dialogrs.DialogrsError) as raised:
        dialogrs.Pipeline.from_pretrained()
    assert raised.value.code == "cuda"


@pytest.mark.model
def test_transcribes_and_diarizes_file_and_samples(pipeline, clip):
    dialog = pipeline.transcribe_file(CLIP)
    assert normalize(dialog.text) == CLIP_TEXT
    assert str(dialog) == dialog.text
    assert dialog.duration_seconds == pytest.approx(len(clip) / 16_000)

    # one reader, so every word lands in a labelled turn
    assert dialog.turns
    assert all(turn.speaker is not None for turn in dialog.turns)
    assert normalize(" ".join(turn.text for turn in dialog.turns)) == CLIP_TEXT
    labels = {segment.speaker for segment in dialog.segments}
    assert {turn.speaker for turn in dialog.turns} <= labels

    assert dialog.speakers
    for centroid in dialog.speakers:
        assert centroid.label in labels
        assert len(centroid.embedding) == 256
        assert sum(value * value for value in centroid.embedding) == pytest.approx(1.0, abs=1e-3)
        assert centroid.chunk_count > 0 and centroid.speech_seconds > 0

    from_samples = pipeline.transcribe(clip)
    assert from_samples.text == dialog.text
    assert [(t.speaker, t.text) for t in from_samples.turns] == [(t.speaker, t.text) for t in dialog.turns]


@pytest.mark.model
def test_empty_audio_raises(pipeline):
    with pytest.raises(dialogrs.DialogrsError) as raised:
        pipeline.transcribe([])
    assert raised.value.code == "empty_audio"
