"""Speaker-attributed transcription: scriptrs for what was said, speakrs for who said it.

Load the pipeline once, then transcribe files or 16 kHz mono samples::

    import dialogrs

    pipeline = dialogrs.Pipeline.from_pretrained()
    dialog = pipeline.transcribe_file("meeting.wav")
    for turn in dialog.turns:
        print(f"{turn.speaker} [{turn.start:.1f}-{turn.end:.1f}] {turn.text}")

Transcription and diarization run at the same time in Rust with the GIL
released, so other Python threads keep running while a call is in progress.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from importlib import metadata
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from . import _cuda, _native
from ._errors import DialogrsError
from ._native import (
    BACKEND,
    Dialog,
    Segment,
    SpeakerCentroid,
    Turn,
    Word,
    assign_speakers,
    available_modes,
    words_from_tokens,
)

if TYPE_CHECKING:
    from typing import Any, Protocol

    from typing_extensions import Buffer

    class _ArrayInterface(Protocol):
        # NumPy only declares __buffer__ for Python 3.12+, so match its array
        # interface too
        @property
        def __array_interface__(self) -> dict[str, Any]: ...

    Samples = Buffer | _ArrayInterface | Sequence[float]
else:
    # a runtime name too, so typing.get_type_hints resolves the annotations
    Samples = object

try:
    __version__ = metadata.version("dialogrs")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"


class Model(str, Enum):
    """Which speech-to-text model to run.

    Anywhere a model is taken, its value, such as ``"parakeet/v3"``, or the
    upstream model id, such as ``"nvidia/parakeet-tdt-0.6b-v3"``, works too.
    """

    PARAKEET_V2 = "parakeet/v2"
    """NVIDIA Parakeet TDT 0.6B v2: English only, slightly more accurate on English."""
    PARAKEET_V3 = "parakeet/v3"
    """NVIDIA Parakeet TDT 0.6B v3: 25 European languages, detected automatically."""

    @property
    def languages(self) -> list[str]:
        """ISO 639-1 codes of the languages this model transcribes."""
        return _native.languages(self.value)


ModelName = Model | Literal["parakeet/v2", "parakeet/v3"] | str
"""A :class:`Model`, its value, or its upstream model id."""

Mode = Literal["cuda", "cuda-fast", "coreml", "coreml-fast", "cpu"]
"""How speakrs diarizes; :func:`available_modes` lists the ones this build has.

``"cuda"`` is the default on Linux and Windows and ``"coreml"`` on macOS. The
``-fast`` modes step the diarization window 2 s at a time instead of 1 s: faster,
slightly less accurate. ``"cpu"`` diarizes on the CPU while transcription stays on
the GPU or Core ML. Case, ``-`` and ``_`` are ignored, so ``"CUDAFast"`` works too.
"""

Precision = Literal["fp16", "tf32", "fp32"]
"""Precision of the CUDA speech-to-text encoder's matrix multiplies; Core ML ignores it.

``"fp16"`` is the default and the fastest. ``"tf32"`` keeps the weights in FP32,
and ``"fp32"`` matches NVIDIA NeMo's FP32 output token for token.
"""

StrPath = str | PathLike[str]

# upstream model ids, so a name copied from Hugging Face or NeMo works as is
_UPSTREAM_IDS = {
    "nvidia/parakeet-tdt-0.6b-v2": "parakeet/v2",
    "nvidia/parakeet-tdt-0.6b-v3": "parakeet/v3",
}
_PRECISIONS = ("fp16", "tf32", "fp32")
_DEFAULT_MODES = {"cuda": "cuda", "coreml": "coreml"}


def _model(model: ModelName) -> Model:
    if isinstance(model, Model):
        return model
    name = str(model).strip().lower()
    try:
        return Model(_UPSTREAM_IDS.get(name, name))
    except ValueError:
        known = ", ".join(repr(known.value) for known in Model)
        raise ValueError(f"unknown model {model!r}; use one of {known}") from None


def _precision(precision: str) -> str:
    # checked before the driver check and any download, on every platform
    if precision not in _PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; use one of {', '.join(_PRECISIONS)}")
    return precision


def _mode(mode: Mode | str | None) -> str:
    if mode is None:
        return _DEFAULT_MODES.get(BACKEND or "", "cpu")
    return mode


class Pipeline:
    """A loaded speech-to-text model and diarization models.

    Audio of any length works. Calls on one pipeline run one at a time.
    """

    def __init__(self, native: _native._Pipeline) -> None:
        """Use :meth:`from_pretrained` or :meth:`from_dir` instead."""
        self._native = native

    @classmethod
    def from_pretrained(
        cls,
        model: ModelName = Model.PARAKEET_V3,
        *,
        mode: Mode | None = None,
        precision: Precision = "fp16",
        language: str | None = None,
    ) -> Pipeline:
        """Download the models from Hugging Face if needed, then load them.

        The first call downloads the Parakeet model, about 0.6 GB of Core ML
        models on macOS or 2.5 GB of weights on Linux and Windows, and the
        speakrs diarization models into the Hugging Face hub cache; later calls
        load from the cache. Use :meth:`from_dir` to load models from local
        directories without the network.

        ``language`` is an ISO 639-1 code from :attr:`Model.languages`, such as
        ``"de"``; without it, Parakeet v3 detects the language and v2
        transcribes English.

        Raises :class:`DialogrsError` with code ``cuda`` when Linux or Windows
        has no NVIDIA driver, before anything is downloaded, or
        ``model_download`` when a download fails.
        """
        model = _model(model)
        precision = _precision(precision)
        mode = _mode(mode)
        _native._check_settings(model.value, mode, precision, language)
        _cuda.prepare()
        return cls(_native._from_pretrained(model.value, mode, precision, language))

    @classmethod
    def from_dir(
        cls,
        transcription_dir: StrPath,
        diarization_dir: StrPath,
        model: ModelName = Model.PARAKEET_V3,
        *,
        mode: Mode | None = None,
        precision: Precision = "fp16",
        language: str | None = None,
    ) -> Pipeline:
        """Load the models from local directories instead of downloading them.

        ``transcription_dir`` holds the Parakeet model in the layout the
        scriptrs README describes: Core ML models on macOS, native weights on
        Linux and Windows. ``diarization_dir`` holds the speakrs models.
        """
        model = _model(model)
        precision = _precision(precision)
        mode = _mode(mode)
        _native._check_settings(model.value, mode, precision, language)
        _cuda.prepare()
        return cls(
            _native._from_dir(Path(transcription_dir), Path(diarization_dir), model.value, mode, precision, language)
        )

    @property
    def model(self) -> Model:
        """The loaded speech-to-text model."""
        return Model(self._native.model)

    @property
    def mode(self) -> str:
        """The diarization mode, as speakrs names it, such as ``"CUDA"``."""
        return self._native.mode

    @property
    def language(self) -> str | None:
        """The language every call transcribes, or ``None`` when the model picks."""
        return self._native.language

    @property
    def languages(self) -> list[str]:
        """ISO 639-1 codes of the languages the loaded model transcribes."""
        return self._native.languages

    def transcribe(self, samples: Samples) -> Dialog:
        """Transcribe and diarize 16 kHz mono audio, giving each word a speaker.

        ``samples`` is a 1-D float32 buffer, such as a NumPy array, or a
        sequence of floats; it is copied before the GIL is released.
        """
        return self._native.transcribe(samples)

    def transcribe_file(self, path: StrPath) -> Dialog:
        """Decode an audio file, resample it to 16 kHz mono, then transcribe and diarize it.

        WAV, FLAC, MP3, Ogg Vorbis, and AAC in MP4/M4A are supported. The whole
        file is decoded into memory.
        """
        return self._native.transcribe_file(Path(path))

    def __repr__(self) -> str:
        return f"Pipeline(model={self.model.value!r}, backend={BACKEND!r}, mode={self.mode!r})"


__all__ = [
    "BACKEND",
    "Dialog",
    "DialogrsError",
    "Mode",
    "Model",
    "ModelName",
    "Pipeline",
    "Precision",
    "Segment",
    "SpeakerCentroid",
    "Turn",
    "Word",
    "assign_speakers",
    "available_modes",
    "words_from_tokens",
]
