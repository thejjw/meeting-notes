"""Safe subprocess invocation utilities."""

from __future__ import annotations

import subprocess
import threading
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

log = structlog.get_logger()


class SubprocessResult:
    """Result from a subprocess call."""

    def __init__(
        self,
        returncode: int,
        stdout: str,
        stderr: str,
        args: list[str],
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.args = args

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def check(self, label: str = "command") -> None:
        """Raise on failure."""
        if not self.success:
            raise RuntimeError(
                f"{label} failed (exit {self.returncode}):\n"
                f"  args: {' '.join(self.args)}\n"
                f"  stderr: {self.stderr[:500]}"
            )


def run_command(
    args: list[str],
    *,
    timeout: float | None = 120.0,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    label: str = "command",
    input_text: str | None = None,
    capture_stderr: bool = True,
    redact_args: set[int] | None = None,
) -> SubprocessResult:
    """Run a subprocess with argument array (never shell=True).

    Args:
        args: Command and arguments as a list.
        timeout: Timeout in seconds. None for no timeout.
        cwd: Working directory.
        env: Extra environment variables (merged with current env).
        label: Human-readable label for error messages.
        input_text: Text to pipe to stdin.
        capture_stderr: Whether to capture stderr.
        redact_args: Argument indexes replaced in diagnostic logs.

    Returns:
        SubprocessResult with returncode, stdout, stderr.
    """
    import os

    merged_env = dict(os.environ) if env else None
    if merged_env and env:
        merged_env.update(env)

    redacted = redact_args or set()
    display_args = " ".join(
        (
            "<redacted>"
            if index in redacted
            else argument if " " not in argument else f'"{argument}"'
        )
        for index, argument in enumerate(args)
    )
    log.debug("subprocess.run", args=display_args, label=label)

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=merged_env,
            shell=False,
            input=input_text,
        )
        return SubprocessResult(
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            args=args,
        )
    except subprocess.TimeoutExpired:
        log.error("subprocess.timeout", args=display_args, timeout=timeout, label=label)
        raise RuntimeError(f"{label} timed out after {timeout}s") from None
    except FileNotFoundError:
        log.error("subprocess.not_found", args=display_args, label=label)
        raise RuntimeError(f"{label}: executable not found: {args[0]}") from None
    except OSError as e:
        log.error("subprocess.os_error", args=display_args, error=str(e), label=label)
        raise RuntimeError(f"{label}: {e}") from e


def run_command_streaming(
    args: list[str],
    *,
    on_output: Callable[[str, str], None],
    timeout: float | None = 120.0,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    label: str = "command",
) -> SubprocessResult:
    """Run a command while reporting and retaining stdout/stderr lines."""
    import os

    merged_env = dict(os.environ) if env else None
    if merged_env and env:
        merged_env.update(env)

    display_args = " ".join(a if " " not in a else f'"{a}"' for a in args)
    log.debug("subprocess.Popen", args=display_args, label=label)

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            env=merged_env,
            shell=False,
        )
    except FileNotFoundError:
        log.error("subprocess.not_found", args=display_args, label=label)
        raise RuntimeError(f"{label}: executable not found: {args[0]}") from None
    except OSError as e:
        log.error("subprocess.os_error", args=display_args, error=str(e), label=label)
        raise RuntimeError(f"{label}: {e}") from e

    captured: dict[str, list[str]] = {"stdout": [], "stderr": []}

    def _read_stream(name: str, stream: Any) -> None:
        try:
            for line in stream:
                captured[name].append(line)
                on_output(name, line.rstrip("\r\n"))
        finally:
            stream.close()

    readers = [
        threading.Thread(target=_read_stream, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=_read_stream, args=("stderr", proc.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        log.error("subprocess.timeout", args=display_args, timeout=timeout, label=label)
        raise RuntimeError(f"{label} timed out after {timeout}s") from None
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=5.0)

    return SubprocessResult(
        returncode=returncode,
        stdout="".join(captured["stdout"]),
        stderr="".join(captured["stderr"]),
        args=args,
    )


def run_command_background(
    args: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    label: str = "command",
) -> subprocess.Popen[str]:
    """Start a subprocess in the background (non-blocking).

    Returns the Popen object for the caller to manage.
    """
    import os

    merged_env = dict(os.environ) if env else None
    if merged_env and env:
        merged_env.update(env)

    display_args = " ".join(a if " " not in a else f'"{a}"' for a in args)
    log.debug("subprocess.Popen", args=display_args, label=label)

    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=merged_env,
        shell=False,
    )


def format_command_display(args: list[str]) -> str:
    """Format command for display in logs, escaping special characters."""
    parts = []
    for arg in args:
        if any(c in arg for c in " \t\n\"'\\|&;<>"):
            escaped = arg.replace("\\", "\\\\").replace('"', '\\"')
            parts.append(f'"{escaped}"')
        else:
            parts.append(arg)
    return " ".join(parts)
