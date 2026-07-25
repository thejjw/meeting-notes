"""Tests for the non-publishing summarizer probe."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from meeting_notes.summarization.adapters import SummaryResult
from meeting_notes.summarizer_probe import run_summarizer_test

if TYPE_CHECKING:
    from pathlib import Path


class _ProbeAdapter:
    def is_available(self) -> bool:
        return True

    def summarize(self, transcript: str, **kwargs: object) -> SummaryResult:
        assert "seg-000000" in transcript
        assert kwargs["metadata"] == {
            "language": "ko",
            "speaker_resolution": "none",
            "probe": True,
        }
        return SummaryResult(
            data={"title": "Probe", "short_title": "probe"},
            backend="local_command",
        )


def test_probe_invokes_adapter_without_creating_publication(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
setup:
  completed: true
summarization:
  backend: local_command
  prompt_path: missing-prompt.md
  output_schema_path: missing-schema.json
  local_command:
    protocol: request_json_v1
    command: [probe]
""",
        encoding="utf-8",
    )

    with patch(
        "meeting_notes.summarizer_probe.get_adapter",
        return_value=_ProbeAdapter(),
    ):
        payload = run_summarizer_test(config_path=str(config), output_json=True)

    assert payload["success"] is True
    assert payload["title"] == "Probe"
    assert list(tmp_path.iterdir()) == [config]
