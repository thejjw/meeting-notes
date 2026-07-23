"""Tests for interchangeable summarizer adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_notes.summarization.adapters import (
    ClaudeCodeAdapter,
    CodexAdapter,
    LocalCommandAdapter,
    MimoCodeAdapter,
    OpenCodeAdapter,
    SummaryResult,
    detect_available_adapters,
    get_adapter,
    register_adapter,
)


class TestSummaryResult:
    """Test SummaryResult dataclass."""

    def test_basic_creation(self) -> None:
        result = SummaryResult(data={"title": "Test"}, backend="codex")
        assert result.data == {"title": "Test"}
        assert result.backend == "codex"
        assert result.warnings == []


class TestAdapterRegistry:
    """Test adapter registry and detection."""

    def test_get_adapter_codex(self) -> None:
        adapter = get_adapter("codex")
        assert isinstance(adapter, CodexAdapter)
        assert adapter.name == "codex"

    def test_get_adapter_opencode(self) -> None:
        adapter = get_adapter("opencode")
        assert isinstance(adapter, OpenCodeAdapter)

    def test_get_adapter_mimo(self) -> None:
        adapter = get_adapter("mimo")
        assert isinstance(adapter, MimoCodeAdapter)

    def test_get_adapter_claude(self) -> None:
        adapter = get_adapter("claude")
        assert isinstance(adapter, ClaudeCodeAdapter)

    def test_get_adapter_local_command(self) -> None:
        adapter = get_adapter("local_command")
        assert isinstance(adapter, LocalCommandAdapter)

    def test_get_unknown_adapter_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown summarizer adapter"):
            get_adapter("nonexistent")

    def test_register_custom_adapter(self) -> None:
        class CustomAdapter(CodexAdapter):
            @property
            def name(self) -> str:
                return "custom_test"

        register_adapter("custom_test", CustomAdapter)
        adapter = get_adapter("custom_test")
        assert adapter.name == "custom_test"

    def test_detect_available(self) -> None:
        available = detect_available_adapters()
        assert isinstance(available, dict)
        assert "codex" in available


class TestJSONParsing:
    """Test JSON output parsing from adapters."""

    def test_parse_direct_json(self) -> None:
        adapter = CodexAdapter()
        result = adapter._parse_json_output('{"title": "Test"}')
        assert result == {"title": "Test"}

    def test_parse_json_in_text(self) -> None:
        adapter = CodexAdapter()
        output = 'Here is the result:\n{"title": "Test"}\nDone.'
        result = adapter._parse_json_output(output)
        assert result == {"title": "Test"}

    def test_parse_invalid_json_raises(self) -> None:
        adapter = CodexAdapter()
        with pytest.raises(RuntimeError, match="Could not extract JSON"):
            adapter._parse_json_output("no json here")


class TestLocalCommandAdapter:
    """Test local command adapter."""

    def test_not_available_without_command(self) -> None:
        adapter = LocalCommandAdapter()
        assert adapter.is_available() is False

    def test_available_with_command(self) -> None:
        adapter = LocalCommandAdapter(command=["echo", "hello"])
        assert adapter.is_available() is True
