"""Type stubs for the ``dialogrs._native`` extension module."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, final

BACKEND: str | None
"""Speech-to-text backend in this build: ``"coreml"`` on macOS, ``"cuda"`` on
Linux and Windows x86_64, or ``None`` where dialogrs cannot transcribe."""

class _TokenLike(Protocol):
    """A timed subword token, such as ``scriptrs.TimedToken``."""

    @property
    def text(self) -> str: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def confidence(self) -> float: ...

class _WordLike(Protocol):
    """A timed word, such as :class:`Word`."""

    @property
    def text(self) -> str: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def confidence(self) -> float: ...

class _SegmentLike(Protocol):
    """A span of one speaker's speech, such as :class:`Segment` or ``speakrs.Segment``."""

    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...
    @property
    def speaker(self) -> str: ...

@final
class Word:
    """A word with its time span, merged from the ASR's subword tokens."""

    def __init__(self, text: str, start: float, end: float, confidence: float) -> None: ...
    @property
    def text(self) -> str:
        """The word as transcribed, punctuation included."""
    @property
    def start(self) -> float:
        """Start time in seconds."""
    @property
    def end(self) -> float:
        """End time in seconds."""
    @property
    def confidence(self) -> float:
        """The lowest confidence of the word's tokens, so one doubtful piece flags the word."""

@final
class Turn:
    """Consecutive words from one speaker."""

    @property
    def speaker(self) -> str | None:
        """The diarization label, such as ``SPEAKER_00``; ``None`` when diarization found no speech at all."""
    @property
    def start(self) -> float:
        """Start of the first word in seconds."""
    @property
    def end(self) -> float:
        """End of the last word in seconds."""
    @property
    def words(self) -> list[Word]:
        """The words in time order."""
    @property
    def text(self) -> str:
        """The turn's words joined by spaces."""

@final
class Segment:
    """A span of speech from one speaker."""

    def __init__(self, start: float, end: float, speaker: str) -> None: ...
    @property
    def start(self) -> float:
        """Start time in seconds."""
    @property
    def end(self) -> float:
        """End time in seconds."""
    @property
    def speaker(self) -> str:
        """Speaker label, such as ``SPEAKER_00``."""

@final
class SpeakerCentroid:
    """A speaker's voice embedding, for matching the same person across recordings."""

    @property
    def label(self) -> str:
        """The label this speaker's turns and segments carry, such as ``SPEAKER_00``."""
    @property
    def embedding(self) -> list[float]:
        """L2-normalised mean of the speaker's 256-d chunk embeddings, so the cosine
        similarity of two centroids is their dot product."""
    @property
    def chunk_count(self) -> int:
        """Number of chunk embeddings averaged into :attr:`embedding`."""
    @property
    def speech_seconds(self) -> float:
        """Seconds of the speaker's speech, overlapped speech included."""

@final
class Dialog:
    """Speaker-attributed transcript of one recording."""

    @property
    def turns(self) -> list[Turn]:
        """Consecutive words from one speaker, in time order."""
    @property
    def text(self) -> str:
        """The transcript without speakers."""
    @property
    def segments(self) -> list[Segment]:
        """Diarization with overlapping speech kept, for RTTM output and DER."""
    @property
    def speakers(self) -> list[SpeakerCentroid]:
        """Voice embeddings in label order. A speaker with no usable embedding,
        which is rare, has none, so look labels up rather than assume one per speaker."""
    @property
    def duration_seconds(self) -> float:
        """Input duration in seconds."""

@final
class _Pipeline:
    @property
    def model(self) -> str: ...
    @property
    def mode(self) -> str: ...
    @property
    def language(self) -> str | None: ...
    @property
    def languages(self) -> list[str]: ...
    def transcribe(self, samples: Any) -> Dialog: ...
    def transcribe_file(self, path: Path) -> Dialog: ...

def _check_settings(model: str, mode: str, precision: str, language: str | None = None) -> None: ...
def _from_pretrained(model: str, mode: str, precision: str, language: str | None = None) -> _Pipeline: ...
def _from_dir(
    transcription_dir: Path,
    diarization_dir: Path,
    model: str,
    mode: str,
    precision: str,
    language: str | None = None,
) -> _Pipeline: ...
def languages(model: str) -> list[str]:
    """ISO 639-1 codes of the languages ``model`` transcribes."""

def available_modes() -> list[str]:
    """Diarization modes compiled into this build, as speakrs names them, such as
    ``["CPU", "CUDA", "CUDAFast"]``. A CUDA mode still needs an NVIDIA driver."""

def words_from_tokens(tokens: Iterable[_TokenLike]) -> list[Word]:
    """Merge subword tokens, such as a ``scriptrs.TranscriptionResult``'s ``tokens``, into words.

    A token whose text starts with a space begins a word; any other token,
    punctuation included, extends the word before it.
    """

def assign_speakers(words: Iterable[_WordLike], segments: Iterable[_SegmentLike]) -> list[Turn]:
    """Give each word the speaker of the segment it overlaps most and group the words into turns.

    ``segments`` must be sorted without overlap, as a speakrs result's
    ``exclusive_segments`` are, and ``words`` sorted by start; otherwise this
    raises ``ValueError``. A word in a pause goes to the nearest segment.
    """
