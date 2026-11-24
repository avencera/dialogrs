"""The exception type raised for dialogrs failures."""

from __future__ import annotations


class DialogrsError(Exception):
    """A dialogrs operation failed; invalid arguments raise ``ValueError`` or ``TypeError`` instead.

    ``code`` is one of the stable values below, so callers can branch on it
    without parsing messages.

    Loading: ``configuration``, ``unsupported_language``, ``unsupported_mode``,
    ``unsupported_platform``, ``model_download``, ``missing_model``,
    ``invalid_model``, ``model_load``, ``cuda``, ``coreml``.

    Running: ``empty_audio``, ``audio_too_long``, ``invalid_audio``,
    ``audio_decode``, ``io``, ``transcription``, ``diarization``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)

    @property
    def code(self) -> str:
        """Stable failure code."""
        return self.args[0]

    @property
    def message(self) -> str:
        """Human-readable description."""
        return self.args[1]

    def __str__(self) -> str:
        return f"{self.message} [{self.code}]"
