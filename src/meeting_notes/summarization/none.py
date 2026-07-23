"""No-op summarizer backend."""

from __future__ import annotations

from pathlib import Path

from meeting_notes.summarization.base import SummarizerBackend, SummaryResult


class NoSummarizerBackend(SummarizerBackend):
    @property
    def name(self) -> str:
        return "none"

    def is_available(self) -> bool:
        return True

    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str = "",
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> SummaryResult:
        return SummaryResult(data={}, backend="none", warnings=["Summarization disabled"])
