"""Tests for subprocess utilities."""

from __future__ import annotations

import pytest

from meeting_notes.subprocess_utils import (
    SubprocessResult,
    format_command_display,
    run_command,
)


class TestSubprocessResult:
    """Test SubprocessResult properties."""

    def test_success(self) -> None:
        result = SubprocessResult(returncode=0, stdout="ok", stderr="", args=["cmd"])
        assert result.success is True

    def test_failure(self) -> None:
        result = SubprocessResult(returncode=1, stdout="", stderr="error", args=["cmd"])
        assert result.success is False

    def test_check_raises_on_failure(self) -> None:
        result = SubprocessResult(returncode=1, stdout="", stderr="error details", args=["cmd"])
        with pytest.raises(RuntimeError, match="cmd failed"):
            result.check("cmd")

    def test_check_succeeds(self) -> None:
        result = SubprocessResult(returncode=0, stdout="ok", stderr="", args=["cmd"])
        result.check("cmd")  # Should not raise


class TestFormatCommand:
    """Test command display formatting."""

    def test_simple_args(self) -> None:
        result = format_command_display(["ffmpeg", "-i", "file.wav"])
        assert result == "ffmpeg -i file.wav"

    def test_args_with_spaces(self) -> None:
        result = format_command_display(["ffmpeg", "-i", "my file.wav"])
        assert '"my file.wav"' in result

    def test_args_with_quotes(self) -> None:
        result = format_command_display(["echo", 'he said "hello"'])
        assert '"' in result


class TestRunCommand:
    """Test subprocess execution (using echo as safe command)."""

    def test_run_echo(self) -> None:
        result = run_command(["echo", "hello"], timeout=5.0, label="test")
        assert result.success
        assert "hello" in result.stdout

    def test_run_nonexistent_command(self) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            run_command(["nonexistent-tool-xyz"], timeout=5.0, label="test")

    def test_run_command_timeout(self) -> None:
        # Use a command that takes longer than the timeout
        # On Windows, use PowerShell Start-Sleep; on Linux, use sleep
        import platform
        if platform.system() == "Windows":
            with pytest.raises(RuntimeError, match="timed out"):
                run_command(["powershell", "-Command", "Start-Sleep -Seconds 10"], timeout=0.5, label="test")
        else:
            with pytest.raises(RuntimeError, match="timed out"):
                run_command(["sleep", "10"], timeout=0.5, label="test")
