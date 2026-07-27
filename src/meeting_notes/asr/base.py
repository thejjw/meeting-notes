"""Base class for ASR (speech recognition) backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ASRSegment:
    """A single transcript segment from ASR."""

    id: str
    start: float
    end: float
    text: str
    language: str | None = None
    speaker: str | None = None
    confidence: float | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class ASRResult:
    """Complete ASR result."""

    segments: list[ASRSegment]
    language: str = ""
    duration: float = 0.0
    backend: str = ""
    model: str = ""
    device: str = ""
    raw_output: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ASRReadiness:
    """Structured availability result used by preflight checks and diagnostics."""

    available: bool
    detail: str
    version: str = ""
    device: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'whisper_cpp')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is usable in the current environment."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str = "medium",
        model_path: Path | None = None,
        language: str = "ko",
        task: str = "transcribe",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        threads: int = 0,
        extra_args: list[str] | None = None,
    ) -> ASRResult:
        """Run transcription on an audio file.

        Args:
            audio_path: Path to normalized WAV file.
            model: Model name (e.g., 'medium', 'large-v3').
            model_path: Optional explicit path to model file.
            language: Language hint (e.g., 'ko', 'en', 'auto').
            task: 'transcribe' or 'translate'.
            initial_prompt: Optional initial prompt for the model.
            word_timestamps: Whether to request word-level timestamps.
            threads: Number of CPU threads (0 = auto).
            extra_args: Additional backend-specific arguments.

        Returns:
            ASRResult with segments and metadata.
        """

    @abstractmethod
    def get_version(self) -> str:
        """Return the version string of the installed backend."""

    def check_readiness(
        self,
        *,
        model: str = "",
        expected_device: str = "",
        allow_provision: bool = False,
    ) -> ASRReadiness:
        """Return actionable backend readiness information.

        Local adapters retain their existing availability behavior. Service
        adapters can override this to inspect models and accelerator state.
        """
        del model, allow_provision
        available = self.is_available()
        return ASRReadiness(
            available=available,
            detail="backend is ready" if available else "backend is unavailable",
            version=self.get_version(),
            device=expected_device,
        )
