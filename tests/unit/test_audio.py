"""Tests for audio inspection and normalization."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from meeting_notes.audio.inspect import MediaInfo, inspect_media
from meeting_notes.audio.normalize import create_normalized_path, normalize_audio


def _create_test_wav(path: Path, duration_sec: float = 1.0, sample_rate: int = 16000) -> Path:
    """Create a minimal WAV file for testing."""
    n_samples = int(sample_rate * duration_sec)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Write silence
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


class TestMediaInfo:
    """Test MediaInfo dataclass."""

    def test_duration_timestamp(self) -> None:
        info = MediaInfo(file_path="test.wav", duration_seconds=3723.5)
        assert info.duration_timestamp == "01:02:03"

    def test_has_audio(self) -> None:
        from meeting_notes.audio.inspect import StreamInfo

        info = MediaInfo(
            file_path="test.wav",
            streams=[StreamInfo(index=0, codec_type="audio", codec_name="pcm_s16le")],
        )
        assert info.has_audio is True
        assert info.has_video is False

    def test_has_video(self) -> None:
        from meeting_notes.audio.inspect import StreamInfo

        info = MediaInfo(
            file_path="test.mp4",
            streams=[
                StreamInfo(index=0, codec_type="video", codec_name="h264"),
                StreamInfo(index=1, codec_type="audio", codec_name="aac"),
            ],
        )
        assert info.has_video is True
        assert info.has_audio is True


class TestAudioInspection:
    """Test FFprobe-based media inspection."""

    def test_inspect_wav(self, tmp_path: Path) -> None:
        wav = _create_test_wav(tmp_path / "test.wav", duration_sec=2.0)
        info = inspect_media(wav)
        assert info.has_audio
        assert info.file_size_bytes > 0
        assert info.file_hash
        assert info.duration_seconds > 0

    def test_inspect_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            inspect_media(tmp_path / "nonexistent.wav")


class TestAudioNormalization:
    """Test FFmpeg normalization."""

    def test_normalize_creates_output(self, tmp_path: Path) -> None:
        wav = _create_test_wav(tmp_path / "input.wav", duration_sec=1.0)
        output = tmp_path / "output" / "normalized.wav"
        result = normalize_audio(wav, output)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_normalize_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            normalize_audio(
                tmp_path / "nonexistent.wav",
                tmp_path / "output.wav",
            )

    def test_create_normalized_path(self, tmp_path: Path) -> None:
        path = create_normalized_path(tmp_path / "job", "recording.m4a")
        assert path == tmp_path / "job" / "audio" / "normalized.wav"
