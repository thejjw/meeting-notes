"""No-op VAD backend (passes all audio through)."""

from __future__ import annotations

from pathlib import Path

from meeting_notes.vad.base import VADBackend, VADSegment


class NoVADBackend(VADBackend):
    """VAD backend that returns no speech segments (disables VAD)."""

    @property
    def name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return True

    def detect(
        self,
        audio_path: Path,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 200,
    ) -> list[VADSegment]:
        return []
