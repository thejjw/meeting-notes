"""No-op diarization backend."""

from __future__ import annotations

from pathlib import Path

from meeting_notes.diarization.base import DiarizationBackend, DiarizationResult


class NoDiarizationBackend(DiarizationBackend):
    """Diarization backend that returns no speaker turns (disables diarization)."""

    @property
    def name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return True

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int = 8,
    ) -> DiarizationResult:
        return DiarizationResult(turns=[], backend="none")
