"""Local command summarization backend."""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from meeting_notes.subprocess_utils import run_command
from meeting_notes.summarization.base import SummarizerBackend, SummaryResult

log = structlog.get_logger()


class LocalCommandBackend(SummarizerBackend):
    """Summarization backend that invokes a user-configured local command."""

    def __init__(
        self,
        command: list[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self._command = command or []
        self._environment = environment or {}

    @property
    def name(self) -> str:
        return "local_command"

    def is_available(self) -> bool:
        return bool(self._command)

    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str,
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> SummaryResult:
        if not self._command:
            raise RuntimeError("No local command configured")

        log.info(
            "local_command.summarize",
            command=self._command[0] if self._command else "none",
            transcript_length=len(transcript_text),
        )

        # The command receives prompt via args, transcript via stdin
        result = run_command(
            self._command,
            timeout=timeout_seconds,
            input_text=transcript_text,
            env=self._environment or None,
            label="local-command-summarize",
        )

        if not result.success:
            raise RuntimeError(
                f"Local command failed (exit {result.returncode}):\n"
                f"  stderr: {result.stderr[:500]}"
            )

        output = result.stdout.strip()
        data = json.loads(output) if output else {}

        return SummaryResult(
            data=data,
            backend=self.name,
            raw_output=output[:5000],
        )
