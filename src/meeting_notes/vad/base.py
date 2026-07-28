"""Voice Activity Detection base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VADSegment:
    """A speech segment detected by VAD."""

    start: float
    end: float
    confidence: float = 0.0


class VADBackend(ABC):
    """Abstract base class for VAD backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is usable."""

    @abstractmethod
    def detect(
        self,
        audio_path: Path,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 200,
    ) -> list[VADSegment]:
        """Detect speech segments in audio.

        Args:
            audio_path: Path to 16kHz mono WAV file.
            threshold: Speech probability threshold.
            min_speech_ms: Minimum speech segment duration.
            min_silence_ms: Minimum silence gap duration.
            speech_pad_ms: Padding around speech segments.

        Returns:
            List of VADSegment with absolute timestamps.
        """
