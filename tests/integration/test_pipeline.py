"""Integration tests for meeting-notes pipeline."""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING

import pytest

from meeting_notes.audio.inspect import inspect_media
from meeting_notes.audio.normalize import normalize_audio
from meeting_notes.jobs import create_job_dir, load_manifest, make_job_slug, save_manifest
from meeting_notes.minutes.render import render_minutes
from meeting_notes.naming import generate_filenames, sanitize_short_title

if TYPE_CHECKING:
    from pathlib import Path


def _create_test_wav(path: Path, duration_sec: float = 2.0, sample_rate: int = 16000) -> Path:
    """Create a minimal WAV file for testing."""
    n_samples = int(sample_rate * duration_sec)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


@pytest.mark.integration
class TestAudioPipeline:
    """Integration tests for audio processing pipeline."""

    def test_inspect_and_normalize(self, tmp_path: Path) -> None:
        """Test full inspect → normalize pipeline."""
        # Create test audio
        source = tmp_path / "test.wav"
        _create_test_wav(source, duration_sec=3.0)

        # Inspect
        info = inspect_media(source)
        assert info.has_audio
        assert info.duration_seconds > 0
        assert info.file_hash

        # Normalize
        output = tmp_path / "output" / "normalized.wav"
        normalize_audio(source, output)
        assert output.exists()
        assert output.stat().st_size > 0


@pytest.mark.integration
class TestJobPipeline:
    """Integration tests for job directory management."""

    def test_create_and_load_job(self, tmp_path: Path) -> None:
        """Test job creation, manifest, and slug generation."""
        source = tmp_path / "recording.m4a"
        source.write_bytes(b"fake audio")

        slug = make_job_slug(source)
        job_dir = create_job_dir(tmp_path / "data", slug)
        assert job_dir.exists()

        manifest = load_manifest(job_dir)
        assert manifest["version"] == 1

        manifest["source"]["original_path"] = str(source)
        save_manifest(job_dir, manifest)

        loaded = load_manifest(job_dir)
        assert loaded["source"]["original_path"] == str(source)


@pytest.mark.integration
class TestMinutesRendering:
    """Integration tests for meeting minutes rendering."""

    def test_render_full_summary(self) -> None:
        """Test rendering a complete meeting summary."""
        summary = {
            "title": "API Authentication Review Meeting",
            "short_title": "API-Auth-Review",
            "meeting_date": "2026-07-22",
            "participants": [
                {"name": "John Kim", "role": "Tech Lead"},
                {"name": "Sarah Park", "role": "Developer"},
            ],
            "executive_summary": [
                "Discussed API authentication approach",
                "Decided on API key for initial implementation",
            ],
            "agenda": ["API authentication strategy", "OAuth transition plan"],
            "discussion_topics": [
                {
                    "topic": "Authentication Method",
                    "summary": ["Team discussed API key vs OAuth approaches"],
                    "evidence": ["seg-000042"],
                }
            ],
            "decisions": [
                {
                    "decision": "Use API key for initial implementation",
                    "status": "confirmed",
                    "evidence": ["seg-000043"],
                    "timestamps": ["00:05:12"],
                }
            ],
            "action_items": [
                {
                    "task": "Document OAuth transition plan",
                    "owner": "John Kim",
                    "due_date": "2026-07-29",
                    "status": "open",
                    "confidence": "high",
                    "evidence": ["seg-000044"],
                    "timestamps": ["00:05:20"],
                }
            ],
            "open_questions": [
                {"question": "What is the rate limiting strategy?", "evidence": ["seg-000050"]}
            ],
            "risks": [
                {
                    "risk": "API key exposure in client-side code",
                    "impact": "High",
                    "mitigation": "Use server-side proxy",
                    "evidence": ["seg-000051"],
                }
            ],
        }

        md = render_minutes(summary, source_filename="meeting.m4a", duration_timestamp="01:12:44")

        assert "API Authentication Review Meeting" in md
        assert "2026-07-22" in md
        assert "John Kim" in md
        assert "API key" in md
        assert "Document OAuth transition" in md
        assert "API key exposure" in md


@pytest.mark.integration
class TestFilenameFinalization:
    """Integration tests for filename generation."""

    def test_generate_and_sanitize(self) -> None:
        """Test filename generation with Korean and English."""
        title = sanitize_short_title("API \uc778\uc99d \ubc29\uc2dd \ud68c\uc758")
        files = generate_filenames("2026-07-22", title, ".m4a")

        assert "2026-07-22" in files["recording"]
        assert ".m4a" in files["recording"]
        assert "meeting-notes" in files["minutes"]

    def test_collision_handling(self, tmp_path: Path) -> None:
        """Test filename collision resolution."""
        existing = tmp_path / "file.txt"
        existing.write_text("existing")

        from meeting_notes.naming import resolve_collision
        result = resolve_collision(existing, policy="increment")
        assert result.name == "file_02.txt"
