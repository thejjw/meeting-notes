"""Interchangeable summarizer adapters for multiple AI CLI tools.

Supports: Codex CLI, OpenCode, Mimo Code, Claude Code, and custom local commands.
Each adapter implements the same interface so they can be swapped via config.
"""

from __future__ import annotations

import json
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from meeting_notes.subprocess_utils import run_command

log = structlog.get_logger()


@dataclass
class SummaryResult:
    """Result from a summarization backend."""

    data: dict[str, Any]
    backend: str = ""
    raw_output: str = ""
    warnings: list[str] = field(default_factory=list)


class SummarizerAdapter(ABC):
    """Abstract base class for all summarizer adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'codex', 'opencode', 'claude')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this CLI tool is installed and authenticated."""

    @abstractmethod
    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str,
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
    ) -> SummaryResult:
        """Generate a structured meeting summary from transcript text."""

    def _parse_json_output(self, output: str) -> dict:
        """Try to extract JSON from CLI output that may contain other text."""
        # Try direct JSON parse
        try:
            return json.loads(output.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in output
        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(output[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Try JSON array
        start = output.find("[")
        end = output.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return {"items": json.loads(output[start : end + 1])}
            except json.JSONDecodeError:
                pass

        raise RuntimeError(f"Could not extract JSON from output:\n{output[:500]}")


class CodexAdapter(SummarizerAdapter):
    """Codex CLI adapter (openai/codex)."""

    def __init__(
        self,
        executable: str = "codex",
        model: str | None = None,
        reasoning_effort: str | None = None,
        ephemeral: bool = True,
        skip_git_repo_check: bool = True,
        ignore_user_config: bool = False,
        ignore_rules: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        self._executable = executable
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._ephemeral = ephemeral
        self._skip_git_repo_check = skip_git_repo_check
        self._ignore_user_config = ignore_user_config
        self._ignore_rules = ignore_rules
        self._extra_args = extra_args or []

    @property
    def name(self) -> str:
        return "codex"

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
        with tempfile.TemporaryDirectory(prefix="meeting-notes-codex-") as tmp_dir:
            tmp = Path(tmp_dir)
            output_file = tmp / "summary.json"

            args = [self._executable, "exec"]

            if self._ephemeral:
                args.append("--ephemeral")
            if self._skip_git_repo_check:
                args.append("--skip-git-repo-check")
            if self._ignore_user_config:
                args.append("--ignore-user-config")
            if self._ignore_rules:
                args.append("--ignore-rules")
            if schema_path and schema_path.exists():
                args.extend(["--output-schema", str(schema_path.resolve())])
            if self._model:
                args.extend(["--model", self._model])
            if self._reasoning_effort:
                effort = json.dumps(self._reasoning_effort)
                args.extend(["--config", f"model_reasoning_effort={effort}"])

            args.extend(self._extra_args)
            args.extend(["--output-last-message", str(output_file), prompt])

            result = run_command(
                args,
                timeout=timeout_seconds,
                cwd=tmp_dir,
                input_text=transcript_text,
                label="codex-summarize",
                redact_args={len(args) - 1},
            )

            if not result.returncode == 0:
                raise RuntimeError(
                    f"Codex CLI failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:2000]}"
                )

            output = (
                output_file.read_text(encoding="utf-8") if output_file.exists() else result.stdout
            )
            if not output.strip():
                raise RuntimeError("Codex CLI returned empty output")

            return SummaryResult(
                data=self._parse_json_output(output),
                backend=self.name,
                raw_output=output[:5000],
            )


class OpenCodeAdapter(SummarizerAdapter):
    """OpenCode CLI adapter."""

    def __init__(
        self,
        executable: str = "opencode",
        model: str | None = None,
    ) -> None:
        self._executable = executable
        self._model = model

    @property
    def name(self) -> str:
        return "opencode"

    def is_available(self) -> bool:
        try:
            result = run_command(
                [self._executable, "--version"],
                timeout=5.0,
                label="opencode-check",
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
        # OpenCode uses -p for non-interactive prompt mode
        args = [self._executable, "run"]

        if self._model:
            args.extend(["--model", self._model])

        # Build full prompt with schema instruction
        full_prompt = prompt
        if schema_path and schema_path.exists():
            schema_text = schema_path.read_text(encoding="utf-8")
            full_prompt += f"\n\nOutput schema:\n{schema_text}"

        args.extend(["--prompt", full_prompt])

        with tempfile.TemporaryDirectory(prefix="meeting-notes-opencode-") as tmp_dir:
            result = run_command(
                args,
                timeout=timeout_seconds,
                cwd=tmp_dir,
                input_text=transcript_text,
                label="opencode-summarize",
                redact_args={args.index("--prompt") + 1},
            )

            if not result.returncode == 0:
                raise RuntimeError(
                    f"OpenCode CLI failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:500]}"
                )

            return SummaryResult(
                data=self._parse_json_output(result.stdout),
                backend=self.name,
                raw_output=result.stdout[:5000],
            )


class MimoCodeAdapter(SummarizerAdapter):
    """Mimo Code CLI adapter (mimocode)."""

    def __init__(
        self,
        executable: str = "mimo",
        model: str | None = None,
    ) -> None:
        self._executable = executable
        self._model = model

    @property
    def name(self) -> str:
        return "mimo"

    def is_available(self) -> bool:
        try:
            result = run_command(
                [self._executable, "--version"],
                timeout=5.0,
                label="mimo-check",
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
        # Mimo Code uses -p for print mode (non-interactive)
        args = [self._executable, "-p"]

        if self._model:
            args.extend(["--model", self._model])

        full_prompt = prompt
        if schema_path and schema_path.exists():
            schema_text = schema_path.read_text(encoding="utf-8")
            full_prompt += f"\n\nOutput schema:\n{schema_text}"

        args.append(full_prompt)

        with tempfile.TemporaryDirectory(prefix="meeting-notes-mimo-") as tmp_dir:
            result = run_command(
                args,
                timeout=timeout_seconds,
                cwd=tmp_dir,
                input_text=transcript_text,
                label="mimo-summarize",
                redact_args={len(args) - 1},
            )

            if not result.returncode == 0:
                raise RuntimeError(
                    f"Mimo Code failed (exit {result.returncode}):\n  stderr: {result.stderr[:500]}"
                )

            return SummaryResult(
                data=self._parse_json_output(result.stdout),
                backend=self.name,
                raw_output=result.stdout[:5000],
            )


class ClaudeCodeAdapter(SummarizerAdapter):
    """Claude Code CLI adapter (claude)."""

    def __init__(
        self,
        executable: str = "claude",
        model: str | None = None,
    ) -> None:
        self._executable = executable
        self._model = model

    @property
    def name(self) -> str:
        return "claude"

    def is_available(self) -> bool:
        try:
            result = run_command(
                [self._executable, "--version"],
                timeout=5.0,
                label="claude-check",
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
        # Claude Code uses -p for print mode
        args = [self._executable, "-p"]

        if self._model:
            args.extend(["--model", self._model])

        full_prompt = prompt
        if schema_path and schema_path.exists():
            schema_text = schema_path.read_text(encoding="utf-8")
            full_prompt += f"\n\nOutput schema:\n{schema_text}"

        args.append(full_prompt)

        with tempfile.TemporaryDirectory(prefix="meeting-notes-claude-") as tmp_dir:
            result = run_command(
                args,
                timeout=timeout_seconds,
                cwd=tmp_dir,
                input_text=transcript_text,
                label="claude-summarize",
                redact_args={len(args) - 1},
            )

            if not result.returncode == 0:
                raise RuntimeError(
                    f"Claude Code failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:500]}"
                )

            return SummaryResult(
                data=self._parse_json_output(result.stdout),
                backend=self.name,
                raw_output=result.stdout[:5000],
            )


class LocalCommandAdapter(SummarizerAdapter):
    """Generic local command adapter.

    For custom AI CLIs not covered above. Configure the command and
    arguments in the config file.
    """

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

        result = run_command(
            self._command,
            timeout=timeout_seconds,
            input_text=transcript_text,
            env=self._environment or None,
            label="local-command-summarize",
        )

        if not result.returncode == 0:
            raise RuntimeError(
                f"Local command failed (exit {result.returncode}):\n  stderr: {result.stderr[:500]}"
            )

        return SummaryResult(
            data=self._parse_json_output(result.stdout),
            backend=self.name,
            raw_output=result.stdout[:5000],
        )


# --- Registry ---

_adapters: dict[str, type[SummarizerAdapter]] = {
    "codex": CodexAdapter,
    "opencode": OpenCodeAdapter,
    "mimo": MimoCodeAdapter,
    "claude": ClaudeCodeAdapter,
    "local_command": LocalCommandAdapter,
}

_adapter_aliases = {
    "codex_cli": "codex",
}


def register_adapter(name: str, cls: type[SummarizerAdapter]) -> None:
    """Register a custom summarizer adapter."""
    _adapters[name] = cls


def get_adapter(name: str, **kwargs: Any) -> SummarizerAdapter:
    """Get a summarizer adapter by name.

    Falls back to 'none' if the requested adapter is not available.
    """
    canonical_name = _adapter_aliases.get(name, name)
    if canonical_name not in _adapters:
        raise ValueError(
            f"Unknown summarizer adapter: '{name}'. "
            f"Available: {', '.join([*_adapters.keys(), *_adapter_aliases.keys()])}"
        )

    return _adapters[canonical_name](**kwargs)


def configured_adapter_options(config: Any) -> dict[str, Any]:
    """Build provider-specific adapter arguments from summarization config."""
    backend = config.backend
    if backend in {"codex", "codex_cli"}:
        options = config.codex
        return {
            "executable": options.executable,
            "model": options.model,
            "reasoning_effort": options.reasoning_effort,
            "ephemeral": options.ephemeral,
            "skip_git_repo_check": options.skip_git_repo_check,
            "ignore_user_config": options.ignore_user_config,
            "ignore_rules": options.ignore_rules,
            "extra_args": options.extra_args,
        }
    if backend == "claude":
        options = config.claude
        return {"executable": options.executable, "model": options.model}
    if backend == "local_command":
        options = config.local_command
        return {"command": options.command, "environment": options.environment}
    return {}


def summarizer_provenance(config: Any) -> dict[str, str | None]:
    """Return the requested provider settings without claiming effective defaults."""
    backend = str(config.backend)
    model: str | None = None
    reasoning_effort: str | None = None
    if backend in {"codex", "codex_cli"}:
        model = config.codex.model
        reasoning_effort = config.codex.reasoning_effort
    elif backend == "claude":
        model = config.claude.model
    return {
        "backend": backend,
        "requested_model": model,
        "requested_reasoning_effort": reasoning_effort,
    }


def detect_available_adapters() -> dict[str, bool]:
    """Detect which summarizer adapters are available."""
    results = {}
    for name, cls in _adapters.items():
        try:
            adapter = cls()
            results[name] = adapter.is_available()
        except Exception:
            results[name] = False
    return results
