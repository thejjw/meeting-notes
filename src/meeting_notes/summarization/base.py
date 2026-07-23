"""Summarizer base class and adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SummaryResult:
    """Result from a summarization backend."""

    data: dict[str, Any]
    backend: str = ""
    model: str = ""
    raw_output: str = ""
    warnings: list[str] = field(default_factory=list)


class SummarizerBackend(ABC):
    """Abstract base class for summarization backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is usable."""

    @abstractmethod
    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str,
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> SummaryResult:
        """Generate a structured meeting summary from transcript text.

        Args:
            transcript_text: The transcript content to summarize.
            prompt: Summarization instruction/prompt.
            schema_path: Optional JSON schema for output validation.
            timeout_seconds: Maximum time for the summarizer.

        Returns:
            SummaryResult with structured data.
        """
