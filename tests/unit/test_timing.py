"""Tests for ASR/diarization time estimation from historical job manifests."""

from __future__ import annotations

import json
from pathlib import Path

from meeting_notes.config import MeetingNotesConfig
from meeting_notes.timing import (
    build_time_estimate_lines,
    collect_real_time_factors,
    estimate_stage_seconds,
    format_duration,
)

MATCH = {"backend": "whisper_cpp", "device": "cpu", "model": "large-v3-turbo"}


def _write_job(
    data_dir: Path,
    name: str,
    *,
    stage: str = "transcribe",
    status: str = "completed",
    runtime: dict | None = None,
    duration_seconds: float | None = 600.0,
    started_at: str = "2026-07-25T00:00:00+00:00",
    ended_at: str = "2026-07-25T00:10:00+00:00",
) -> Path:
    job_dir = data_dir / "meetings" / name
    job_dir.mkdir(parents=True)
    manifest = {
        "source": {"duration_seconds": duration_seconds},
        "stages": {
            stage: {
                "status": status,
                "started_at": started_at,
                "ended_at": ended_at,
                "runtime": runtime if runtime is not None else dict(MATCH),
            }
        },
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return job_dir


class TestCollectRealTimeFactors:
    def test_matching_job_contributes_a_factor(self, tmp_path: Path) -> None:
        _write_job(tmp_path, "job1")  # 600s audio, 600s wall -> RTF 1.0
        factors = collect_real_time_factors(tmp_path, "transcribe", MATCH)
        assert factors == [1.0]

    def test_mismatched_runtime_excluded(self, tmp_path: Path) -> None:
        other = {"backend": "openai_whisper", "device": "cpu", "model": "large-v3-turbo"}
        _write_job(tmp_path, "job1", runtime=other)
        assert collect_real_time_factors(tmp_path, "transcribe", MATCH) == []

    def test_incomplete_stage_excluded(self, tmp_path: Path) -> None:
        _write_job(tmp_path, "job1", status="running")
        assert collect_real_time_factors(tmp_path, "transcribe", MATCH) == []

    def test_missing_duration_excluded(self, tmp_path: Path) -> None:
        _write_job(tmp_path, "job1", duration_seconds=None)
        assert collect_real_time_factors(tmp_path, "transcribe", MATCH) == []

    def test_no_meetings_dir_returns_empty(self, tmp_path: Path) -> None:
        assert collect_real_time_factors(tmp_path, "transcribe", MATCH) == []

    def test_corrupt_manifest_skipped(self, tmp_path: Path) -> None:
        job_dir = tmp_path / "meetings" / "broken"
        job_dir.mkdir(parents=True)
        (job_dir / "manifest.json").write_text("not json", encoding="utf-8")
        _write_job(tmp_path, "job1")
        assert collect_real_time_factors(tmp_path, "transcribe", MATCH) == [1.0]

    def test_excluded_job_dir_skipped(self, tmp_path: Path) -> None:
        job_dir = _write_job(tmp_path, "job1")
        factors = collect_real_time_factors(
            tmp_path, "transcribe", MATCH, exclude_job_dir=job_dir
        )
        assert factors == []

    def test_multiple_jobs_all_contribute(self, tmp_path: Path) -> None:
        _write_job(tmp_path, "job1", duration_seconds=600.0, ended_at="2026-07-25T00:10:00+00:00")
        _write_job(tmp_path, "job2", duration_seconds=600.0, ended_at="2026-07-25T00:20:00+00:00")
        factors = sorted(collect_real_time_factors(tmp_path, "transcribe", MATCH))
        assert factors == [1.0, 2.0]


class TestEstimateStageSeconds:
    def test_no_samples_returns_none(self, tmp_path: Path) -> None:
        estimate, count = estimate_stage_seconds(tmp_path, "transcribe", MATCH, 1200.0)
        assert estimate is None
        assert count == 0

    def test_uses_median_real_time_factor(self, tmp_path: Path) -> None:
        _write_job(tmp_path, "job1", ended_at="2026-07-25T00:10:00+00:00")  # RTF 1.0
        _write_job(tmp_path, "job2", ended_at="2026-07-25T00:20:00+00:00")  # RTF 2.0
        _write_job(tmp_path, "job3", ended_at="2026-07-25T00:30:00+00:00")  # RTF 3.0
        estimate, count = estimate_stage_seconds(tmp_path, "transcribe", MATCH, 1000.0)
        assert count == 3
        assert estimate == 2000.0  # median RTF 2.0 * 1000s


class TestFormatDuration:
    def test_seconds_only(self) -> None:
        assert format_duration(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        assert format_duration(125) == "2m 5s"

    def test_whole_minutes(self) -> None:
        assert format_duration(120) == "2m"

    def test_hours_and_minutes(self) -> None:
        assert format_duration(3661) == "1h 1m"

    def test_zero(self) -> None:
        assert format_duration(0) == "0s"


class TestBuildTimeEstimateLines:
    def test_no_audio_duration_returns_empty(self, tmp_path: Path) -> None:
        config = MeetingNotesConfig()
        assert build_time_estimate_lines(config, ["transcribe"], None, data_dir=tmp_path) == []

    def test_no_matching_stages_returns_empty(self, tmp_path: Path) -> None:
        config = MeetingNotesConfig()
        assert build_time_estimate_lines(config, ["merge"], 600.0, data_dir=tmp_path) == []

    def test_reports_estimate_and_total_when_history_exists(self, tmp_path: Path) -> None:
        config = MeetingNotesConfig(
            runtime={"asr_backend": "whisper_cpp", "device": "cpu"},
            asr={"model": "large-v3-turbo"},
        )
        _write_job(tmp_path, "job1", duration_seconds=600.0, ended_at="2026-07-25T00:10:00+00:00")

        lines = build_time_estimate_lines(
            config, ["transcribe"], 1200.0, data_dir=tmp_path
        )

        assert any("Transcribe (ASR)" in line and "20m" in line for line in lines)
        assert any(line.strip().startswith("Total:") for line in lines)

    def test_reports_no_history_message_when_no_samples(self, tmp_path: Path) -> None:
        config = MeetingNotesConfig(
            runtime={"asr_backend": "whisper_cpp", "device": "cpu"},
            asr={"model": "large-v3-turbo"},
        )
        lines = build_time_estimate_lines(config, ["transcribe"], 600.0, data_dir=tmp_path)
        assert any("No timing history" in line for line in lines)

    def test_excludes_current_job_dir(self, tmp_path: Path) -> None:
        config = MeetingNotesConfig(
            runtime={"asr_backend": "whisper_cpp", "device": "cpu"},
            asr={"model": "large-v3-turbo"},
        )
        job_dir = _write_job(tmp_path, "job1")
        lines = build_time_estimate_lines(
            config, ["transcribe"], 600.0, data_dir=tmp_path, exclude_job_dir=job_dir
        )
        assert any("No timing history" in line for line in lines)
