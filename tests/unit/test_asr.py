"""Tests for ASR backend registry and transcript rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from meeting_notes.asr.base import ASRResult, ASRSegment
from meeting_notes.asr.registry import get_backend, list_backends
from meeting_notes.transcript.render import (
    format_timestamp,
    render_json,
    render_markdown,
    render_srt,
    render_vtt,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestASRRegistry:
    """Test ASR backend registry."""

    def test_list_backends(self) -> None:
        backends = list_backends()
        assert "whisper_cpp" in backends

    def test_get_whisper_cpp_backend(self) -> None:
        backend = get_backend("whisper_cpp")
        assert backend.name == "whisper_cpp"

    def test_get_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ASR backend"):
            get_backend("nonexistent_backend")


class TestTimestampFormatting:
    """Test timestamp formatting utilities."""

    def test_format_hhmmss_mmm(self) -> None:
        assert format_timestamp(3723.456, "HH:MM:SS.mmm") == "01:02:03.456"

    def test_format_hhmmss(self) -> None:
        assert format_timestamp(3723.0, "HH:MM:SS") == "01:02:03"

    def test_format_srt(self) -> None:
        assert format_timestamp(3723.456, "HH:MM:SS,mmm") == "01:02:03,456"

    def test_format_zero(self) -> None:
        assert format_timestamp(0.0) == "00:00:00.000"


class TestTranscriptRendering:
    """Test transcript output rendering."""

    @pytest.fixture
    def sample_result(self) -> ASRResult:
        return ASRResult(
            segments=[
                ASRSegment(
                    id="seg-000000",
                    start=5.0,
                    end=12.0,
                    text="Hello world",
                    language="en",
                    source={"backend": "whisper_cpp", "raw_segment_index": 0},
                ),
                ASRSegment(
                    id="seg-000001",
                    start=15.0,
                    end=22.0,
                    text="Korean text here",
                    language="ko",
                    speaker="SPEAKER_00",
                    source={"backend": "whisper_cpp", "raw_segment_index": 1},
                ),
            ],
            language="ko",
            duration=22.0,
            backend="whisper_cpp",
            model="medium",
            device="cpu",
        )

    def test_render_json(self, sample_result: ASRResult, tmp_path: Path) -> None:
        output = tmp_path / "transcript.json"
        result = render_json(sample_result, output)
        assert result.exists()
        import json

        data = json.loads(result.read_text(encoding="utf-8"))
        assert len(data["segments"]) == 2
        assert data["segments"][0]["text"] == "Hello world"

    def test_render_markdown(self, sample_result: ASRResult, tmp_path: Path) -> None:
        output = tmp_path / "transcript.md"
        result = render_markdown(sample_result, output, source_filename="test.wav")
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "# Transcript" in content
        assert "Hello world" in content
        assert "Korean text here" in content
        assert "SPEAKER_00" in content

    def test_render_srt(self, sample_result: ASRResult, tmp_path: Path) -> None:
        output = tmp_path / "transcript.srt"
        result = render_srt(sample_result, output)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "00:00:05,000 --> 00:00:12,000" in content
        assert "Hello world" in content

    def test_render_vtt(self, sample_result: ASRResult, tmp_path: Path) -> None:
        output = tmp_path / "transcript.vtt"
        result = render_vtt(sample_result, output)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "WEBVTT" in content
        assert "00:00:05.000 --> 00:00:12.000" in content
