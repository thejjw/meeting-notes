"""Diarization base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class DiarizationTurn:
    """A single speaker turn from diarization."""

    turn_id: str
    start: float
    end: float
    speaker: str
    confidence: float | None = None
    source: str = ""


@dataclass
class DiarizationResult:
    """Complete diarization result."""

    turns: list[DiarizationTurn]
    duration: float = 0.0
    backend: str = ""
    model: str = ""
    device: str = ""
    speakers: list[str] = field(default_factory=list)


class DiarizationBackend(ABC):
    """Abstract base class for diarization backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is usable."""

    @abstractmethod
    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """Run speaker diarization.

        Args:
            audio_path: Path to audio file.
            num_speakers: Known number of speakers (None = auto-detect).
            min_speakers: Minimum speaker count.
            max_speakers: Maximum speaker count (None = unbounded).

        Returns:
            DiarizationResult with speaker turns.
        """
