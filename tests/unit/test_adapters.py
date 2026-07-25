"""Tests for interchangeable summarizer adapters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from meeting_notes.config import MeetingNotesConfig
from meeting_notes.pipeline import _format_summary_transcript
from meeting_notes.summarization.adapters import (
    ClaudeCodeAdapter,
    CodexAdapter,
    LocalCommandAdapter,
    MimoCodeAdapter,
    OpenCodeAdapter,
    SummaryResult,
    configured_adapter_options,
    detect_available_adapters,
    get_adapter,
    register_adapter,
    summarizer_provenance,
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

    def test_get_adapter_accepts_legacy_codex_cli_name(self) -> None:
        adapter = get_adapter("codex_cli")
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


def test_codex_transcript_uses_stdin_not_windows_command_line(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text("{}", encoding="utf-8")
    transcript = "x" * 150_000
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object):
        from meeting_notes.subprocess_utils import SubprocessResult

        observed["args"] = args
        observed["input_text"] = kwargs.get("input_text")
        observed["redact_args"] = kwargs.get("redact_args")
        output_index = args.index("--output-last-message") + 1
        Path(args[output_index]).write_text('{"title":"ok"}', encoding="utf-8")
        return SubprocessResult(0, "", "", args)

    adapter = CodexAdapter(
        reasoning_effort="high",
        ignore_user_config=True,
        ignore_rules=True,
        extra_args=["--color", "never"],
    )
    with patch("meeting_notes.summarization.adapters.run_command", side_effect=fake_run):
        result = adapter.summarize(transcript, prompt="Summarize.", schema_path=schema)

    args = observed["args"]
    assert isinstance(args, list)
    assert transcript not in args
    assert observed["input_text"] == transcript
    assert observed["redact_args"] == {len(args) - 1}
    assert "--output-schema" in args
    assert "--ignore-user-config" in args
    assert "--ignore-rules" in args
    assert 'model_reasoning_effort="high"' in args
    assert result.data == {"title": "ok"}


def test_provider_options_preserve_null_defaults() -> None:
    config = MeetingNotesConfig()
    config.summarization.backend = "codex"
    assert configured_adapter_options(config.summarization)["model"] is None
    assert configured_adapter_options(config.summarization)["reasoning_effort"] is None
    assert summarizer_provenance(config.summarization) == {
        "backend": "codex",
        "requested_model": None,
        "requested_reasoning_effort": None,
        "execution": None,
        "launcher": None,
    }


def test_claude_configured_model_and_effort_are_forwarded(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object):
        from meeting_notes.subprocess_utils import SubprocessResult

        observed["args"] = args
        return SubprocessResult(0, '{"title":"ok"}', "", args)

    config = MeetingNotesConfig(
        summarization={
            "backend": "claude",
            "claude": {"model": "sonnet", "effort": "medium"},
        }
    )
    adapter = get_adapter(
        config.summarization.backend,
        **configured_adapter_options(config.summarization),
    )
    with patch("meeting_notes.summarization.adapters.run_command", side_effect=fake_run):
        result = adapter.summarize("transcript", prompt="Summarize.")

    args = observed["args"]
    assert isinstance(args, list)
    assert args[:5] == ["claude", "-p", "--output-format", "json", "--no-session-persistence"]
    assert args[args.index("--model") + 1] == "sonnet"
    assert args[args.index("--effort") + 1] == "medium"
    assert result.data == {"title": "ok"}


def test_claude_null_model_omits_model_flag() -> None:
    config = MeetingNotesConfig(summarization={"backend": "claude"})
    options = configured_adapter_options(config.summarization)
    assert options == {
        "executable": "claude",
        "model": None,
        "effort": None,
        "environment": {},
        "launcher_execution": "direct",
        "launcher_command": None,
    }


def test_claude_effort_is_recorded_in_provenance() -> None:
    config = MeetingNotesConfig(
        summarization={"backend": "claude", "claude": {"effort": "high"}}
    )

    assert summarizer_provenance(config.summarization) == {
        "backend": "claude",
        "requested_model": None,
        "requested_reasoning_effort": "high",
        "execution": "direct",
        "launcher": None,
    }


def test_summary_transcript_includes_stable_evidence_ids() -> None:
    formatted = _format_summary_transcript(
        [
            {
                "id": "seg-000123",
                "start": 65.0,
                "speaker": "SPEAKER_01",
                "text": "Decision text",
            }
        ]
    )

    assert formatted == "[seg-000123] [00:01:05] [SPEAKER_01] Decision text"


class TestLocalCommandAdapter:
    """Test local command adapter."""

    def test_not_available_without_command(self) -> None:
        adapter = LocalCommandAdapter()
        assert adapter.is_available() is False

    def test_available_with_command(self) -> None:
        adapter = LocalCommandAdapter(command=["echo", "hello"])
        assert adapter.is_available() is True

    def test_v1_protocol_sends_complete_request(self) -> None:
        observed: dict[str, object] = {}

        def fake_run(args: list[str], **kwargs: object):
            from meeting_notes.subprocess_utils import SubprocessResult

            observed["input"] = kwargs["input_text"]
            return SubprocessResult(0, '{"title":"ok"}', "", args)

        adapter = LocalCommandAdapter(command=["agent"])
        with patch("meeting_notes.summarization.adapters.run_command", side_effect=fake_run):
            adapter.summarize(
                "회의 내용",
                prompt="요약하세요",
                metadata={"language": "ko"},
            )

        import json

        request = json.loads(str(observed["input"]))
        assert request == {
            "protocol_version": 1,
            "task": "meeting_summary",
            "prompt": "요약하세요",
            "transcript": "회의 내용",
            "schema": None,
            "metadata": {"language": "ko"},
        }

    def test_legacy_protocol_sends_only_transcript(self) -> None:
        observed: dict[str, object] = {}

        def fake_run(args: list[str], **kwargs: object):
            from meeting_notes.subprocess_utils import SubprocessResult

            observed["input"] = kwargs["input_text"]
            return SubprocessResult(0, '{"title":"ok"}', "", args)

        adapter = LocalCommandAdapter(
            command=["agent"],
            protocol="transcript_stdin_v0",
        )
        with patch("meeting_notes.summarization.adapters.run_command", side_effect=fake_run):
            adapter.summarize("transcript", prompt="ignored")

        assert observed["input"] == "transcript"

    def test_powershell_execution_uses_encoded_command(self) -> None:
        observed: dict[str, object] = {}

        def fake_run(args: list[str], **kwargs: object):
            from meeting_notes.subprocess_utils import SubprocessResult

            observed["args"] = args
            return SubprocessResult(0, '{"title":"ok"}', "", args)

        adapter = LocalCommandAdapter(
            execution="powershell",
            script="custom-agent",
        )
        with patch("meeting_notes.summarization.adapters.run_command", side_effect=fake_run):
            adapter.summarize("transcript", prompt="prompt")

        args = observed["args"]
        assert isinstance(args, list)
        assert args[:3] == ["powershell.exe", "-NoLogo", "-EncodedCommand"]

    def test_schema_validation_reports_json_path(self, tmp_path: Path) -> None:
        schema = tmp_path / "schema.json"
        schema.write_text(
            '{"type":"object","required":["title"],'
            '"properties":{"title":{"type":"string"}}}',
            encoding="utf-8",
        )

        def fake_run(args: list[str], **kwargs: object):
            from meeting_notes.subprocess_utils import SubprocessResult

            return SubprocessResult(0, '{"title":42}', "", args)

        adapter = LocalCommandAdapter(command=["agent"])
        with (
            patch("meeting_notes.summarization.adapters.run_command", side_effect=fake_run),
            pytest.raises(RuntimeError, match="validation failed at title"),
        ):
            adapter.summarize("transcript", prompt="prompt", schema_path=schema)

    def test_missing_environment_reference_fails_without_exposing_value(self) -> None:
        adapter = LocalCommandAdapter(
            command=["agent"],
            environment={"AGENT_TOKEN": "${MEETING_NOTES_TEST_MISSING_TOKEN}"},
        )
        with pytest.raises(RuntimeError, match="references missing"):
            adapter.summarize("transcript", prompt="prompt")


def test_claude_structured_output_and_powershell_launcher(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(
        '{"$schema":"https://json-schema.org/draft/2020-12/schema",'
        '"type":"object","required":["title"],'
        '"properties":{"title":{"type":"string"}}}',
        encoding="utf-8",
    )
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object):
        from meeting_notes.subprocess_utils import SubprocessResult

        observed["args"] = args
        observed["env"] = kwargs["env"]
        observed["input"] = kwargs["input_text"]
        return SubprocessResult(
            0,
            'setup notice\n{"structured_output":{"title":"성공"},"is_error":false}',
            "",
            args,
        )

    adapter = ClaudeCodeAdapter(
        model=None,
        launcher_execution="powershell",
        launcher_command="claudemm",
    )
    with patch("meeting_notes.summarization.adapters.run_command", side_effect=fake_run):
        result = adapter.summarize("transcript", prompt="prompt", schema_path=schema)

    args = observed["args"]
    assert isinstance(args, list)
    assert args[:3] == ["powershell.exe", "-NoLogo", "-EncodedCommand"]
    env = observed["env"]
    assert isinstance(env, dict)
    assert "--model" not in str(env["MEETING_NOTES_CLAUDE_ARGS"])
    import json

    provider_args = json.loads(str(env["MEETING_NOTES_CLAUDE_ARGS"]))
    assert '\\"type\\"' in provider_args[provider_args.index("--json-schema") + 1]
    assert observed["input"] == "prompt\n\nTranscript:\ntranscript"
    assert result.data == {"title": "성공"}
