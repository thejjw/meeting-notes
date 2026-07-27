"""Interchangeable summarizer adapters for multiple AI CLI tools.

Supports: Codex CLI, OpenCode, Mimo Code, Claude Code, and custom local commands.
Each adapter implements the same interface so they can be swapped via config.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import tempfile
import time
from abc import ABC, abstractmethod
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog

from meeting_notes.subprocess_utils import run_command

log = structlog.get_logger()
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_SHELL_COMMAND = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_CLAUDE_STDIN_PROMPT = "Follow the task instructions and transcript provided via stdin."
_LOCAL_REASONING_MARKERS = ("<|thought|>", "<|final|>", "</think>")


def _strip_local_reasoning(output: str) -> str:
    """Remove known local-model reasoning envelopes without rewriting Markdown."""
    cleaned = output.strip()
    marker_positions = [
        (cleaned.rfind(marker), marker) for marker in _LOCAL_REASONING_MARKERS if marker in cleaned
    ]
    if marker_positions:
        position, marker = max(marker_positions)
        cleaned = cleaned[position + len(marker) :].strip()
    elif cleaned.lower().startswith(("--- thought", "<think>", "thought:")):
        heading = re.search(r"(?m)^#\s+\S.+$", cleaned)
        if heading:
            cleaned = cleaned[heading.start() :].strip()
    return cleaned


@dataclass
class SummaryResult:
    """Result from a summarization backend."""

    data: dict[str, Any] | None = None
    backend: str = ""
    raw_output: str = ""
    warnings: list[str] = field(default_factory=list)
    output_format: str = "structured_json"
    markdown: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


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
        metadata: dict[str, Any] | None = None,
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

    def _validate_output(self, data: dict[str, Any], schema_path: Path | None) -> None:
        """Validate adapter output against the authoritative configured schema."""
        if not schema_path or not schema_path.exists():
            return
        from jsonschema import ValidationError, validate

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        try:
            validate(instance=data, schema=schema)
        except ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            raise RuntimeError(
                f"Summary schema validation failed at {location}: {exc.message}"
            ) from exc


def _resolve_environment(values: dict[str, str]) -> dict[str, str]:
    """Resolve exact ${NAME} references without interpolating arbitrary text."""
    resolved: dict[str, str] = {}
    for key, value in values.items():
        match = _ENV_REFERENCE.fullmatch(value)
        if not match:
            resolved[key] = value
            continue
        source = match.group(1)
        if source not in os.environ:
            raise RuntimeError(f"Adapter environment variable {key} references missing ${source}")
        resolved[key] = os.environ[source]
    return resolved


def _powershell_args(script: str) -> list[str]:
    encoded = b64encode(script.encode("utf-16-le")).decode("ascii")
    return ["powershell.exe", "-NoLogo", "-EncodedCommand", encoded]


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
        metadata: dict[str, Any] | None = None,
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

            data = self._parse_json_output(output)
            self._validate_output(data, schema_path)
            return SummaryResult(
                data=data,
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
        metadata: dict[str, Any] | None = None,
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

            data = self._parse_json_output(result.stdout)
            self._validate_output(data, schema_path)
            return SummaryResult(
                data=data,
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
        metadata: dict[str, Any] | None = None,
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

            data = self._parse_json_output(result.stdout)
            self._validate_output(data, schema_path)
            return SummaryResult(
                data=data,
                backend=self.name,
                raw_output=result.stdout[:5000],
            )


class ClaudeCodeAdapter(SummarizerAdapter):
    """Claude Code CLI adapter (claude)."""

    def __init__(
        self,
        executable: str = "claude",
        model: str | None = None,
        effort: str | None = None,
        environment: dict[str, str] | None = None,
        launcher_execution: str = "direct",
        launcher_command: str | None = None,
    ) -> None:
        self._executable = executable
        self._model = model
        self._effort = effort
        self._environment = environment or {}
        self._launcher_execution = launcher_execution
        self._launcher_command = launcher_command

    def _shell_command_name(self) -> str:
        command = self._launcher_command or ""
        if not _SHELL_COMMAND.fullmatch(command):
            raise RuntimeError(
                "Claude shell launcher_command must be a single function or command name"
            )
        return command

    @property
    def name(self) -> str:
        return "claude"

    def is_available(self) -> bool:
        try:
            if self._launcher_execution == "powershell":
                command = self._shell_command_name().replace("'", "''")
                args = _powershell_args(
                    f"if (Get-Command '{command}' -ErrorAction SilentlyContinue) "
                    "{ exit 0 } else { exit 1 }"
                )
            elif self._launcher_execution == "posix_shell":
                command = shlex.quote(self._shell_command_name())
                args = ["bash", "-lc", f"command -v {command} >/dev/null"]
            else:
                args = [self._executable, "--version"]
            result = run_command(
                args,
                timeout=5.0,
                label="claude-check",
                env=_resolve_environment(self._environment) or None,
            )
            return result.returncode == 0
        except RuntimeError:
            return False

    def _parse_claude_output(self, output: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        envelopes: list[dict[str, Any]] = []
        for index, character in enumerate(output):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(output[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                envelopes.append(value)
        for envelope in reversed(envelopes):
            structured = envelope.get("structured_output")
            if isinstance(structured, dict):
                return structured
            if envelope.get("is_error"):
                detail = envelope.get("result") or envelope.get("error") or "unknown error"
                raise RuntimeError(f"Claude Code returned an error: {detail}")
        return self._parse_json_output(output)

    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str,
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
        metadata: dict[str, Any] | None = None,
    ) -> SummaryResult:
        provider_args = [
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--permission-mode",
            "dontAsk",
        ]
        if self._model:
            provider_args.extend(["--model", self._model])
        if self._effort:
            provider_args.extend(["--effort", self._effort])
        schema_arg_index: int | None = None
        if schema_path and schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema.pop("$schema", None)
            schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            if self._launcher_execution == "powershell":
                # Windows PowerShell 5.1 loses native-argument quoting when a
                # value contains both embedded quotes and spaces. Compact JSON
                # has spaces only inside strings, where \u0020 is equivalent.
                schema_text = schema_text.replace(" ", "\\u0020").replace('"', '\\"')
            provider_args.extend(["--json-schema", schema_text])
            schema_arg_index = len(provider_args) - 1
        # The instructions and transcript are piped via stdin (they can exceed
        # CLI argument-length limits); a short positional prompt is still
        # required, matching every documented `claude -p` usage pattern.
        provider_args.append(_CLAUDE_STDIN_PROMPT)
        request_text = f"{prompt}\n\nTranscript:\n{transcript_text}"

        environment = _resolve_environment(self._environment)
        redacted: set[int]
        if self._launcher_execution == "powershell":
            command = self._shell_command_name()
            escaped_command = command.replace("'", "''")
            environment["MEETING_NOTES_CLAUDE_ARGS"] = json.dumps(provider_args, ensure_ascii=False)
            script = (
                "$ErrorActionPreference = 'Stop'\n"
                "$invokeArgs = @($env:MEETING_NOTES_CLAUDE_ARGS | ConvertFrom-Json)\n"
                f"$command = Get-Command '{escaped_command}' "
                "-ErrorAction Stop\n"
                "& $command @invokeArgs\n"
                "exit $LASTEXITCODE\n"
            )
            args = _powershell_args(script)
            redacted = set()
        elif self._launcher_execution == "posix_shell":
            command = shlex.quote(self._shell_command_name())
            args = ["bash", "-lc", f'{command} "$@"', "meeting-notes", *provider_args]
            offset = len(args) - len(provider_args)
            redacted = {offset + schema_arg_index} if schema_arg_index is not None else set()
        else:
            args = [self._executable, *provider_args]
            offset = len(args) - len(provider_args)
            redacted = {offset + schema_arg_index} if schema_arg_index is not None else set()

        with tempfile.TemporaryDirectory(prefix="meeting-notes-claude-") as tmp_dir:
            result = run_command(
                args,
                timeout=timeout_seconds,
                cwd=tmp_dir,
                input_text=request_text,
                env=environment or None,
                label="claude-summarize",
                redact_args=redacted,
            )

            if not result.returncode == 0:
                raise RuntimeError(
                    f"Claude Code failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:2000]}\n"
                    f"  stdout: {result.stdout[:1000]}"
                )

            data = self._parse_claude_output(result.stdout)
            self._validate_output(data, schema_path)
            return SummaryResult(
                data=data,
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
        protocol: str = "request_json_v1",
        execution: str = "direct",
        script: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        self._command = command or []
        self._protocol = protocol
        self._execution = execution
        self._script = script
        self._environment = environment or {}

    @property
    def name(self) -> str:
        return "local_command"

    def is_available(self) -> bool:
        return bool(self._command) if self._execution == "direct" else bool(self._script)

    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str,
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
        metadata: dict[str, Any] | None = None,
    ) -> SummaryResult:
        if not self.is_available():
            raise RuntimeError("No local command or shell script configured")

        if self._protocol == "request_json_v1":
            schema = (
                json.loads(schema_path.read_text(encoding="utf-8"))
                if schema_path and schema_path.exists()
                else None
            )
            input_text = json.dumps(
                {
                    "protocol_version": 1,
                    "task": "meeting_summary",
                    "prompt": prompt,
                    "transcript": transcript_text,
                    "schema": schema,
                    "metadata": metadata or {},
                },
                ensure_ascii=False,
            )
        else:
            input_text = transcript_text

        if self._execution == "powershell":
            args = _powershell_args(self._script or "")
        elif self._execution == "posix_shell":
            args = ["bash", "-lc", self._script or ""]
        else:
            args = self._command

        result = run_command(
            args,
            timeout=timeout_seconds,
            input_text=input_text,
            env=_resolve_environment(self._environment) or None,
            label="local-command-summarize",
        )

        if not result.returncode == 0:
            raise RuntimeError(
                f"Local command failed (exit {result.returncode}):\n"
                f"  stderr: {result.stderr[:2000]}\n"
                f"  stdout: {result.stdout[:1000]}"
            )

        data = self._parse_json_output(result.stdout)
        self._validate_output(data, schema_path)
        return SummaryResult(
            data=data,
            backend=self.name,
            raw_output=result.stdout[:5000],
        )


class LemonadeAdapter(SummarizerAdapter):
    """Best-effort Markdown summarizer using a running AMD Lemonade Server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:13305",
        model_id: str = "Gemma-4-26B-A4B-it-MTP-GGUF",
        api_key_env: str = "LEMONADE_API_KEY",
        connect_timeout_seconds: float = 5.0,
        request_timeout_seconds: int = 7200,
        provisioning_timeout_seconds: int = 3600,
        max_completion_tokens: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.connect_timeout_seconds = connect_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.provisioning_timeout_seconds = provisioning_timeout_seconds
        self.max_completion_tokens = max_completion_tokens

    @property
    def name(self) -> str:
        return "lemonade"

    def _headers(self) -> dict[str, str]:
        token = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _timeout(self, total_seconds: float) -> httpx.Timeout:
        return httpx.Timeout(total_seconds, connect=self.connect_timeout_seconds)

    def _get_json(self, path: str) -> dict[str, Any]:
        try:
            response = httpx.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                timeout=self._timeout(self.connect_timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(f"Lemonade request failed: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Lemonade returned a non-object response.")
        return payload

    def is_available(self) -> bool:
        try:
            return self._get_json("/v1/health").get("status") == "ok"
        except RuntimeError:
            return False

    def model_info(self) -> dict[str, Any] | None:
        payload = self._get_json("/v1/models?show_all=true")
        models = payload.get("data")
        if not isinstance(models, list):
            return None
        return next(
            (item for item in models if isinstance(item, dict) and item.get("id") == self.model_id),
            None,
        )

    def pull_model(self) -> None:
        """Download the configured catalogue model through the running server."""
        info = self.model_info()
        if info is None:
            raise RuntimeError(f"Lemonade model '{self.model_id}' is not registered.")
        if info.get("downloaded"):
            return
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/v1/pull",
                json={"model_name": self.model_id, "stream": True},
                headers=self._headers(),
                timeout=self._timeout(self.provisioning_timeout_seconds),
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = json.loads(line.partition(":")[2].strip())
                    if isinstance(payload, dict) and payload.get("error"):
                        raise RuntimeError(str(payload["error"]))
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"Failed to download Lemonade model '{self.model_id}': {error}"
            ) from error
        refreshed = self.model_info()
        if not refreshed or not refreshed.get("downloaded"):
            raise RuntimeError(f"Lemonade did not report '{self.model_id}' as downloaded.")

    def ensure_model_ready(self) -> dict[str, Any]:
        info = self.model_info()
        if info is None:
            raise RuntimeError(f"Lemonade model '{self.model_id}' is not registered.")
        if not info.get("downloaded"):
            raise RuntimeError(
                f"Lemonade model '{self.model_id}' is not downloaded. "
                "Run `meeting-notes configure --provision --yes` after selecting "
                "Lemonade summarization."
            )
        health = self._get_json("/v1/health")
        loaded = health.get("all_models_loaded")
        if isinstance(loaded, list) and any(
            isinstance(item, dict)
            and item.get("model_name") == self.model_id
            and item.get("status") in {"ready", "in_use"}
            for item in loaded
        ):
            return info
        try:
            response = httpx.post(
                f"{self.base_url}/v1/load",
                json={"model_name": self.model_id, "save_options": False},
                headers=self._headers(),
                timeout=self._timeout(self.provisioning_timeout_seconds),
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise RuntimeError(
                f"Failed to load Lemonade model '{self.model_id}': {error}"
            ) from error
        return info

    def summarize(
        self,
        transcript_text: str,
        *,
        prompt: str,
        schema_path: Path | None = None,
        timeout_seconds: int = 1800,
        metadata: dict[str, Any] | None = None,
    ) -> SummaryResult:
        del schema_path, metadata
        info = self.ensure_model_ready()
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Transcript:\n{transcript_text}"},
            ],
            "stream": True,
        }
        if self.max_completion_tokens is not None:
            request["max_tokens"] = self.max_completion_tokens

        content_parts: list[str] = []
        reasoning_characters = 0
        usage: dict[str, Any] = {}
        started = time.monotonic()
        effective_timeout = self.request_timeout_seconds or timeout_seconds
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=request,
                headers=self._headers(),
                timeout=self._timeout(effective_timeout),
            ) as response:
                if response.status_code in {400, 413}:
                    detail = response.read().decode("utf-8", errors="replace")[:1000]
                    raise RuntimeError(
                        "Lemonade rejected the complete transcript request. It may exceed "
                        f"the model's {info.get('max_context_window') or 'reported'}-token "
                        f"context window or the server payload limit: {detail}"
                    )
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line.partition(":")[2].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    event = json.loads(raw)
                    if not isinstance(event, dict):
                        continue
                    if isinstance(event.get("usage"), dict):
                        usage.update(event["usage"])
                    choices = event.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta")
                    if not isinstance(delta, dict):
                        continue
                    content = delta.get("content")
                    if isinstance(content, str):
                        content_parts.append(content)
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str):
                        reasoning_characters += len(reasoning)
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(f"Lemonade summarization failed: {error}") from error

        markdown = _strip_local_reasoning("".join(content_parts))
        if not markdown:
            raise RuntimeError("Lemonade returned an empty Markdown summary.")
        return SummaryResult(
            backend=self.name,
            raw_output=markdown[:5000],
            output_format="markdown",
            markdown=markdown,
            metrics={
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "reasoning_characters": reasoning_characters,
                "max_context_window": info.get("max_context_window"),
                **usage,
            },
        )


# --- Registry ---

_adapters: dict[str, type[SummarizerAdapter]] = {
    "codex": CodexAdapter,
    "opencode": OpenCodeAdapter,
    "mimo": MimoCodeAdapter,
    "claude": ClaudeCodeAdapter,
    "local_command": LocalCommandAdapter,
    "lemonade": LemonadeAdapter,
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
        return {
            "executable": options.executable,
            "model": options.model,
            "effort": options.effort,
            "environment": options.environment,
            "launcher_execution": options.launcher_execution,
            "launcher_command": options.launcher_command,
        }
    if backend == "local_command":
        options = config.local_command
        return {
            "command": options.command,
            "protocol": options.protocol,
            "execution": options.execution,
            "script": options.script,
            "environment": options.environment,
        }
    if backend == "lemonade":
        options = config.lemonade
        return {
            "base_url": options.base_url,
            "model_id": options.model_id,
            "api_key_env": options.api_key_env,
            "connect_timeout_seconds": options.connect_timeout_seconds,
            "request_timeout_seconds": options.request_timeout_seconds,
            "provisioning_timeout_seconds": options.provisioning_timeout_seconds,
            "max_completion_tokens": options.max_completion_tokens,
        }
    return {}


def summarizer_provenance(config: Any) -> dict[str, Any]:
    """Return the requested provider settings without claiming effective defaults."""
    backend = str(config.backend)
    model: str | None = None
    reasoning_effort: str | None = None
    execution: str | None = None
    launcher: str | None = None
    if backend in {"codex", "codex_cli"}:
        model = config.codex.model
        reasoning_effort = config.codex.reasoning_effort
    elif backend == "claude":
        model = config.claude.model
        reasoning_effort = config.claude.effort
        execution = config.claude.launcher_execution
        launcher = config.claude.launcher_command
    elif backend == "local_command":
        execution = config.local_command.execution
    elif backend == "lemonade":
        model = config.lemonade.model_id
        execution = "http"
        launcher = config.lemonade.base_url
    return {
        "backend": backend,
        "requested_model": model,
        "requested_reasoning_effort": reasoning_effort,
        "execution": execution,
        "launcher": launcher,
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
