"""Shared fixtures: the test clip and one loaded pipeline."""

from __future__ import annotations

import ctypes
import os
import re
import wave
from pathlib import Path

import numpy as np
import pytest

import dialogrs
from dialogrs._cuda import driver_path

DATA = Path(__file__).resolve().parent / "data"
# LibriSpeech test-clean 1089-134686-0007, CC BY 4.0
CLIP = DATA / "librispeech-1089-134686-0007.wav"
CLIP_TEXT = "a cold lucid indifference reigned in his soul"
# DIALOGRS_TEST_MODEL=parakeet/v2 keeps CI's download smaller than the v3 default
TEST_MODEL = os.environ.get("DIALOGRS_TEST_MODEL", "parakeet/v3")
# GitHub's macOS runners diarize on the CPU; a Mac with a working Core ML can pass coreml
TEST_MODE = os.environ.get("DIALOGRS_TEST_MODE") or None


def read_wav(path: Path) -> np.ndarray:
    """16-bit mono 16 kHz WAV as float32."""
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) == (1, 16_000, 2)
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def normalize(text: str) -> str:
    """Lowercase words without punctuation, to compare with LibriSpeech references."""
    return " ".join(re.findall(r"[a-z']+", text.lower()))


def backend_available() -> bool:
    if dialogrs.BACKEND == "cuda":
        try:
            ctypes.CDLL(driver_path())
        except OSError:
            return False
    return dialogrs.BACKEND is not None


@pytest.fixture(scope="session")
def pipeline() -> dialogrs.Pipeline:
    if not backend_available():
        pytest.skip(f"no dialogrs backend on this machine (BACKEND={dialogrs.BACKEND})")
    return dialogrs.Pipeline.from_pretrained(TEST_MODEL, mode=TEST_MODE)


@pytest.fixture(scope="session")
def clip() -> np.ndarray:
    return read_wav(CLIP)
