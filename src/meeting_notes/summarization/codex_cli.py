"""Codex CLI summarization backend."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import structlog

from meeting_notes.subprocess_utils import run_command
from meeting_notes.summarization.base import SummarizerBackend, SummaryResult

log = structlog.get_logger()


class CodexCliBackend(SummarizerBackend):
    """Summarization backend using Codex CLI (codex exec)."""

    def __init__(
        self,
        executable: str = "codex",
        model: str | None = None,
        reasoning_effort: str | None = None,
        ephemeral: bool = True,
        skip_git_repo_check: bool = True,
        extra_args: list[str] | None = None,
    ) -> None:
        self._executable = executable
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._ephemeral = ephemeral
        self._skip_git_repo_check = skip_git_repo_check
        self._extra_args = extra_args or []

    @property
    def name(self) -> str:
        return "codex_cli"

    def is_available(self) -> bool:
        try:
            result = run_command(
                [self._executable, "--version"],
                timeout=5.0,
                label="codex-check",
            )
            return result.returncode == 0
        except RuntimeError:
            return False

    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str,
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> SummaryResult:
        """Invoke codex exec with transcript on stdin and schema-constrained output."""
        args = [self._executable, "exec"]

        if self._ephemeral:
            args.append("--ephemeral")

        if self._skip_git_repo_check:
            args.append("--skip-git-repo-check")

        if schema_path and schema_path.exists():
            args.extend(["--output-schema", str(schema_path)])

        if self._model:
            args.extend(["--model", self._model])

        if self._reasoning_effort:
            args.extend(["--reasoning-effort", self._reasoning_effort])

        args.extend(self._extra_args)
        args.append(prompt)

        log.info(
            "codex_cli.summarize",
            prompt_length=len(prompt),
            transcript_length=len(transcript_text),
            timeout=timeout_seconds,
        )

        # Create a temporary working directory for codex
        with tempfile.TemporaryDirectory(prefix="meeting-notes-codex-") as tmp_dir:
            result = run_command(
                args,
                timeout=timeout_seconds,
                cwd=tmp_dir,
                input_text=transcript_text,
                label="codex-summarize",
            )

            if not result.success:
                log.error(
                    "codex_cli.failed",
                    returncode=result.returncode,
                    stderr=result.stderr[:1000],
                )
                raise RuntimeError(
                    f"Codex CLI failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:500]}"
                )

            # Parse JSON output from stdout
            output = result.stdout.strip()
            if not output:
                raise RuntimeError("Codex CLI returned empty output")

            try:
                data = json.loads(output)
            except json.JSONDecodeError:
                # Try to extract JSON from the output (Codex may include other text)
                data = self._extract_json(output)

            return SummaryResult(
                data=data,
                backend=self.name,
                raw_output=output[:5000],
            )

    def _extract_json(self, text: str) -> dict:
        """Try to extract JSON from Codex output that may contain other text."""
        # Look for JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Look for JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return {"items": json.loads(text[start : end + 1])}
            except json.JSONDecodeError:
                pass

        raise RuntimeError(f"Could not extract JSON from Codex output:\n{text[:500]}")
