"""Tests for benchmark system and VAD backends."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_notes.benchmark.runner import (
    BenchmarkRun,
    load_benchmark_matrix,
    render_benchmark_report,
)
from meeting_notes.vad.base import VADBackend, VADSegment
from meeting_notes.vad.none import NoVADBackend


class TestBenchmarkRun:
    """Test BenchmarkRun dataclass."""

    def test_default_values(self) -> None:
        run = BenchmarkRun(name="test", backend="whisper_cpp", model="medium", device="cpu")
        assert run.name == "test"
        assert run.transcription_seconds == 0.0
        assert run.error is None


class TestBenchmarkMatrix:
    """Test benchmark matrix loading."""

    def test_load_matrix(self, tmp_path: Path) -> None:
        matrix_file = tmp_path / "matrix.yaml"
        matrix_file.write_text(
            "runs:\n"
            "  - name: test-run\n"
            "    runtime:\n"
            "      asr_backend: whisper_cpp\n"
            "      device: cpu\n"
            "    asr:\n"
            "      model: small\n",
            encoding="utf-8",
        )
        matrix = load_benchmark_matrix(matrix_file)
        assert len(matrix.runs) == 1
        assert matrix.runs[0]["name"] == "test-run"


class TestBenchmarkReport:
    """Test benchmark report rendering."""

    def test_render_json(self, tmp_path: Path) -> None:
        results = [
            BenchmarkRun(
                name="test",
                backend="whisper_cpp",
                model="medium",
                device="cpu",
                audio_duration_seconds=100.0,
                transcription_seconds=50.0,
                real_time_factor=0.5,
                speed_multiple=2.0,
                segment_count=10,
                character_count=500,
                peak_ram_mb=1024.0,
            )
        ]
        paths = render_benchmark_report(results, tmp_path)
        assert "json" in paths
        assert paths["json"].exists()

        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["name"] == "test"

    def test_render_csv(self, tmp_path: Path) -> None:
        results = [
            BenchmarkRun(name="test", backend="whisper_cpp", model="medium", device="cpu")
        ]
        paths = render_benchmark_report(results, tmp_path)
        assert "csv" in paths
        content = paths["csv"].read_text(encoding="utf-8")
        assert "name,backend" in content

    def test_render_markdown(self, tmp_path: Path) -> None:
        results = [
            BenchmarkRun(name="test", backend="whisper_cpp", model="medium", device="cpu")
        ]
        paths = render_benchmark_report(results, tmp_path)
        assert "markdown" in paths
        content = paths["markdown"].read_text(encoding="utf-8")
        assert "Benchmark Results" in content


class TestVADSegment:
    """Test VAD segment dataclass."""

    def test_segment_creation(self) -> None:
        seg = VADSegment(start=0.0, end=1.5, confidence=0.9)
        assert seg.start == 0.0
        assert seg.end == 1.5
        assert seg.confidence == 0.9


class TestNoVAD:
    """Test no-op VAD backend."""

    def test_always_available(self) -> None:
        assert NoVADBackend().is_available() is True

    def test_returns_empty(self, tmp_path: Path) -> None:
        dummy = tmp_path / "test.wav"
        dummy.write_bytes(b"\x00" * 100)
        result = NoVADBackend().detect(dummy)
        assert result == []
