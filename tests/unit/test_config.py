"""Tests for configuration discovery, persistence, and validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from meeting_notes.config import (
    MeetingNotesConfig,
    SetupConfig,
    load_config,
    save_config,
)
from meeting_notes.configure import _prompt_summarization_config
from meeting_notes.errors import ConfigNotFoundError, ConfigValidationError
from meeting_notes.resources import SystemDiagnostics


class TestConfigModels:
    """Test Pydantic config model construction and defaults."""

    def test_default_config_is_valid(self) -> None:
        config = MeetingNotesConfig()
        assert config.version == 1
        assert config.setup.completed is False
        assert config.runtime.device == "cpu"
        assert config.asr.model == "medium"
        assert config.asr.language == "ko"

    def test_config_from_dict(self) -> None:
        data = {
            "version": 1,
            "setup": {"completed": True, "profile": "safe-cpu"},
            "runtime": {"device": "vulkan"},
            "asr": {"model": "large-v3"},
        }
        config = MeetingNotesConfig(**data)
        assert config.setup.completed is True
        assert config.runtime.device == "vulkan"
        assert config.asr.model == "large-v3"

    def test_config_ignores_unknown_fields(self) -> None:
        # Pydantic v2 ignores extra fields by default
        config = MeetingNotesConfig(unknown_field="value")  # type: ignore[arg-type]
        assert config.version == 1

    def test_setup_completed_must_be_true(self) -> None:
        config = MeetingNotesConfig(setup=SetupConfig(completed=True))
        assert config.setup.completed is True

    def test_config_serialization_roundtrip(self) -> None:
        config = MeetingNotesConfig(
            setup=SetupConfig(completed=True, profile="safe-cpu"),
            runtime={"device": "cpu"},
        )
        data = config.model_dump()
        restored = MeetingNotesConfig(**data)
        assert restored.setup.profile == "safe-cpu"
        assert restored.runtime.device == "cpu"


class TestConfigSaveAndLoad:
    """Test atomic config write and discovery."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config = MeetingNotesConfig(
            setup=SetupConfig(completed=True, profile="test"),
        )
        save_config(config, config_path)
        assert config_path.exists()

        loaded = load_config(str(config_path))
        assert loaded.setup.profile == "test"

    def test_atomic_write_cleans_temp(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config = MeetingNotesConfig(setup=SetupConfig(completed=True))
        save_config(config, config_path)

        # No temp files should remain
        temp_files = list(tmp_path.glob("*.tmp"))
        assert len(temp_files) == 0

    def test_load_config_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad.yaml"
        config_path.write_text("not: [valid: yaml: {{", encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            load_config(str(config_path))

    def test_load_config_incomplete_setup(self, tmp_path: Path) -> None:
        config_path = tmp_path / "incomplete.yaml"
        config = MeetingNotesConfig(setup=SetupConfig(completed=False))
        save_config(config, config_path)
        with pytest.raises(ConfigNotFoundError, match="not been completed"):
            load_config(str(config_path))


class TestConfigDiscovery:
    """Test config discovery order."""

    def test_explicit_path_takes_priority(self, tmp_path: Path) -> None:
        from meeting_notes.config import _resolve_config_path

        config_path = tmp_path / "explicit.yaml"
        config = MeetingNotesConfig(setup=SetupConfig(completed=True))
        save_config(config, config_path)

        result = _resolve_config_path(str(config_path))
        assert result == config_path

    def test_env_var_fallback(self, tmp_path: Path) -> None:
        from meeting_notes.config import _resolve_config_path

        config_path = tmp_path / "env-config.yaml"
        config = MeetingNotesConfig(setup=SetupConfig(completed=True))
        save_config(config, config_path)

        with patch.dict(os.environ, {"MEETING_NOTES_CONFIG": str(config_path)}):
            result = _resolve_config_path(None)
            assert result == config_path

    def test_explicit_nonexistent_path_returns_none(self) -> None:
        from meeting_notes.config import _resolve_config_path

        result = _resolve_config_path("/nonexistent/path/config.yaml")
        assert result is None


class TestProfiles:
    """Test profile-based config overrides."""

    def test_safe_cpu_profile(self, tmp_path: Path) -> None:
        profile_path = Path(__file__).parent.parent.parent / "config" / "profiles" / "safe-cpu.yaml"
        if profile_path.exists():
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            assert data["runtime"]["device"] == "cpu"
            assert data["asr"]["model"] == "medium"

    def test_vulkan_profile(self, tmp_path: Path) -> None:
        profile_path = Path(__file__).parent.parent.parent / "config" / "profiles" / "vulkan.yaml"
        if profile_path.exists():
            data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            assert data["runtime"]["device"] == "vulkan"
            assert data["asr"]["model"] == "large-v3"


class TestSummarizationWizard:
    def test_recommends_terra_for_codex(self) -> None:
        diagnostics = SystemDiagnostics()
        diagnostics.tools.codex_available = True
        with patch("typer.prompt", side_effect=[1, 1]):
            assert _prompt_summarization_config(diagnostics) == (
                "codex",
                "gpt-5.6-terra",
                None,
            )

    def test_recommends_sonnet_for_claude(self) -> None:
        diagnostics = SystemDiagnostics()
        diagnostics.tools.claude_available = True
        with patch("typer.prompt", side_effect=[1, 1]):
            assert _prompt_summarization_config(diagnostics) == (
                "claude",
                "sonnet",
                None,
            )

    def test_provider_default_is_null(self) -> None:
        diagnostics = SystemDiagnostics()
        diagnostics.tools.codex_available = True
        with patch("typer.prompt", side_effect=[1, 3]):
            assert _prompt_summarization_config(diagnostics) == ("codex", None, None)

    def test_disabled_remains_default_choice(self) -> None:
        diagnostics = SystemDiagnostics()
        diagnostics.tools.codex_available = True
        diagnostics.tools.claude_available = True
        with patch("typer.prompt", return_value=3):
            assert _prompt_summarization_config(diagnostics) == ("none", None, None)
