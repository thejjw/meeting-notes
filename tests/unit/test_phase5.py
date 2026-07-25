"""Tests for summarization, minutes rendering, and filename finalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_notes.minutes.render import render_minutes, save_minutes
from meeting_notes.naming import (
    generate_filenames,
    resolve_collision,
    resolve_date,
    sanitize_short_title,
)
from meeting_notes.summarization.chunking import chunk_transcript, format_chunk_for_summarization
from meeting_notes.transcript.models import TranscriptDocument, TranscriptSegment


class TestSanitizeShortTitle:
    """Test short title sanitization."""

    def test_basic_title(self) -> None:
        assert sanitize_short_title("API Auth Review") == "API-Auth-Review"

    def test_korean_title(self) -> None:
        result = sanitize_short_title("API \uc778\uc99d \ubc29\uc2dd \ud68c\uc758")
        assert "\uc778\uc99d" in result

    def test_removes_path_separators(self) -> None:
        result = sanitize_short_title("test/file\\name")
        assert "/" not in result
        assert "\\" not in result

    def test_collapses_separators(self) -> None:
        result = sanitize_short_title("hello   world")
        assert "  " not in result

    def test_trims_trailing_dots(self) -> None:
        result = sanitize_short_title("test...")
        assert not result.endswith(".")

    def test_windows_reserved(self) -> None:
        result = sanitize_short_title("CON")
        assert result != "CON"

    def test_empty_returns_default(self) -> None:
        assert sanitize_short_title("") == "meeting"

    def test_max_length(self) -> None:
        result = sanitize_short_title("a" * 100, max_length=20)
        assert len(result) <= 20


class TestResolveDate:
    """Test date resolution from multiple sources."""

    def test_summary_date_first(self) -> None:
        summary = {"meeting_date": "2026-07-22"}
        date, source = resolve_date(summary)
        assert date == "2026-07-22"
        assert source == "summary_meeting_date"

    def test_fallback_to_mtime(self) -> None:
        date, source = resolve_date(None, source_mtime=1721644800.0)
        assert source == "source_mtime"

    def test_fallback_to_processing_date(self) -> None:
        date, source = resolve_date(None)
        assert source == "processing_date"


class TestGenerateFilenames:
    """Test filename generation."""

    def test_basic_generation(self) -> None:
        files = generate_filenames("2026-07-22", "api-auth-review", ".m4a")
        assert files["recording"] == "2026-07-22_api-auth-review.m4a"
        assert files["minutes"] == "2026-07-22_api-auth-review_meeting-notes.md"

    def test_korean_title(self) -> None:
        files = generate_filenames("2026-07-22", "API-\uc778\uc99d-\ud68c\uc758", ".wav")
        assert "2026-07-22" in files["recording"]
        assert ".wav" in files["recording"]


class TestResolveCollision:
    """Test filename collision handling."""

    def test_no_collision(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        result = resolve_collision(target)
        assert result == target

    def test_increment_policy(self, tmp_path: Path) -> None:
        existing = tmp_path / "file.txt"
        existing.write_text("existing")
        result = resolve_collision(existing, policy="increment")
        assert result.name == "file_02.txt"
        assert not result.exists()

    def test_error_policy(self, tmp_path: Path) -> None:
        existing = tmp_path / "file.txt"
        existing.write_text("existing")
        with pytest.raises(FileExistsError):
            resolve_collision(existing, policy="error")


class TestMinutesRendering:
    """Test deterministic meeting minutes rendering."""

    def test_render_basic_summary(self) -> None:
        summary = {
            "title": "API Auth Meeting",
            "short_title": "API Auth Review",
            "meeting_date": "2026-07-22",
            "executive_summary": ["Discussed API authentication approach"],
            "decisions": [
                {
                    "decision": "Use API key for initial implementation",
                    "status": "confirmed",
                    "evidence": ["seg-000042"],
                }
            ],
            "action_items": [
                {
                    "task": "Document OAuth transition plan",
                    "owner": "John",
                    "due_date": None,
                    "status": "open",
                    "confidence": "high",
                    "evidence": ["seg-000043"],
                }
            ],
        }
        md = render_minutes(summary, source_filename="meeting.m4a")
        assert "# " in md
        assert "API Auth Meeting" in md
        assert "API key" in md
        assert "John" in md
        assert "\ubbf8\uc815" in md  # unknown_value_text for None due_date

    def test_render_empty_summary(self) -> None:
        md = render_minutes({})
        assert "# " in md

    def test_render_summary_with_user_clarifications(self) -> None:
        summary = {
            "title": "API Auth Meeting",
            "short_title": "API Auth Review",
            "meeting_date": "2026-07-22",
            "user_clarifications": [
                {
                    "category": "asr_correction",
                    "question": "Is 'ArgoCD' the correct spelling for '아르고 시디'?",
                    "suggested_correction": "ArgoCD",
                    "evidence": ["seg-000012"],
                }
            ],
        }
        md = render_minutes(summary)
        assert "## 사용자 확인 및 정정" in md
        assert "[ASR 정정]" in md
        assert "ArgoCD" in md
        assert "seg-000012" in md


class TestTranscriptChunking:
    """Test transcript chunking for hierarchical summarization."""

    def test_short_doc_single_chunk(self) -> None:
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(id="s1", start=0, end=5, text="hello world"),
            ]
        )
        chunks = chunk_transcript(doc, target_characters=1000)
        assert len(chunks) == 1

    def test_long_doc_multiple_chunks(self) -> None:
        segments = [
            TranscriptSegment(id=f"s{i}", start=i * 10, end=(i + 1) * 10, text="x" * 1000)
            for i in range(100)
        ]
        doc = TranscriptDocument(segments=segments)
        chunks = chunk_transcript(doc, target_characters=5000)
        assert len(chunks) > 1

    def test_format_chunk(self) -> None:
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(id="s1", start=65, end=70, text="hello", speaker="SPEAKER_00"),
            ]
        )
        text = format_chunk_for_summarization(doc, 0, 3)
        assert "Chunk 1 of 3" in text
        assert "[00:01:05]" in text
        assert "hello" in text
