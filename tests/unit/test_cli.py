"""Tests for CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from meeting_notes.cli import app

runner = CliRunner()


class TestCLIBasic:
    """Test basic CLI commands."""

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "meeting-notes" in result.output

    def test_configure_help(self) -> None:
        result = runner.invoke(app, ["configure", "--help"])
        assert result.exit_code == 0

    def test_config_help(self) -> None:
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0

    def test_doctor_help(self) -> None:
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0

    def test_process_help(self) -> None:
        result = runner.invoke(app, ["process", "--help"])
        assert result.exit_code == 0

    def test_models_help(self) -> None:
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0

    def test_resources_help(self) -> None:
        result = runner.invoke(app, ["resources", "--help"])
        assert result.exit_code == 0


class TestCLIConfigCommands:
    """Test config subcommands."""

    def test_config_status_no_config(self) -> None:
        result = runner.invoke(app, ["config", "status"])
        # Should not crash, even without config
        assert result.exit_code == 0 or "No configuration" in result.output


class TestCLIDoctor:
    """Test doctor command."""

    def test_doctor_runs(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Detected system" in result.output

    def test_doctor_json(self) -> None:
        result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0


class TestCLIModels:
    """Test models subcommands."""

    def test_models_list(self) -> None:
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "tiny" in result.output
        assert "large-v3" in result.output


class TestCLIProcess:
    """Test process command."""

    def test_process_no_config_fails(self) -> None:
        # Without config, process should fail with config error
        result = runner.invoke(app, ["process", "test.m4a", "--dry-run"])
        assert result.exit_code == 1
