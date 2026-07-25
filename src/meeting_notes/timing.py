"""ASR/diarization time estimates, derived from this machine's own job history.

Every completed job already records `started_at`/`ended_at` per stage and the
source audio duration in its manifest. This module mines that history for jobs
that ran the same backend/device/model, so a new job can be given a
real-time-factor-based ETA before the slow stages begin, instead of guessing.
"""

from __future__ import annotations

import json
from datetime import datetime
from statistics import median
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from meeting_notes.config import MeetingNotesConfig


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _stage_wall_seconds(stage: dict[str, Any]) -> float | None:
    started = _parse_iso(stage.get("started_at"))
    ended = _parse_iso(stage.get("ended_at"))
    if started is None or ended is None:
        return None
    seconds = (ended - started).total_seconds()
    return seconds if seconds > 0 else None


def _runtime_matches(stage: dict[str, Any], match: dict[str, str]) -> bool:
    runtime = stage.get("runtime") or {}
    return all(runtime.get(key) == value for key, value in match.items())


def collect_real_time_factors(
    data_dir: Path,
    stage_name: str,
    match: dict[str, str],
    *,
    exclude_job_dir: Path | None = None,
) -> list[float]:
    """Real-time factors (stage wall seconds / audio seconds) from past completed jobs."""
    meetings_dir = data_dir / "meetings"
    if not meetings_dir.is_dir():
        return []

    factors = []
    for job_dir in meetings_dir.iterdir():
        if not job_dir.is_dir() or job_dir == exclude_job_dir:
            continue
        manifest_path = job_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        stage = manifest.get("stages", {}).get(stage_name, {})
        if stage.get("status") != "completed" or not _runtime_matches(stage, match):
            continue

        audio_seconds = manifest.get("source", {}).get("duration_seconds")
        wall_seconds = _stage_wall_seconds(stage)
        if not audio_seconds or audio_seconds <= 0 or wall_seconds is None:
            continue

        factors.append(wall_seconds / audio_seconds)

    return factors


def estimate_stage_seconds(
    data_dir: Path,
    stage_name: str,
    match: dict[str, str],
    audio_duration_seconds: float,
    *,
    exclude_job_dir: Path | None = None,
) -> tuple[float | None, int]:
    """Estimated wall-clock seconds for a stage, and the sample count behind it."""
    factors = collect_real_time_factors(
        data_dir, stage_name, match, exclude_job_dir=exclude_job_dir
    )
    if not factors:
        return None, 0
    return median(factors) * audio_duration_seconds, len(factors)


def format_duration(seconds: float) -> str:
    """Render seconds as a compact duration, e.g. '1h 12m', '3m 5s', '45s'."""
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    return f"{secs}s"


def build_time_estimate_lines(
    config: MeetingNotesConfig,
    stages: list[str],
    audio_duration_seconds: float | None,
    *,
    data_dir: Path,
    exclude_job_dir: Path | None = None,
) -> list[str]:
    """Build the ASR/diarization ETA lines shown before the heavy stages start."""
    if not audio_duration_seconds:
        return []

    stage_matches: dict[str, tuple[str, dict[str, str]]] = {}
    if "transcribe" in stages:
        stage_matches["transcribe"] = (
            "Transcribe (ASR)",
            {
                "backend": config.runtime.asr_backend,
                "device": config.runtime.device,
                "model": config.asr.model,
            },
        )
    if "diarize" in stages:
        stage_matches["diarize"] = (
            "Diarize",
            {
                "backend": config.diarization.backend,
                "device": config.diarization.device,
                "model": config.diarization.model,
            },
        )
    if not stage_matches:
        return []

    lines: list[str] = []
    unestimated: list[str] = []
    total_seconds = 0.0

    for stage_name, (label, match) in stage_matches.items():
        estimate, count = estimate_stage_seconds(
            data_dir,
            stage_name,
            match,
            audio_duration_seconds,
            exclude_job_dir=exclude_job_dir,
        )
        if estimate is None:
            unestimated.append(label)
            continue
        total_seconds += estimate
        run_word = "run" if count == 1 else "runs"
        lines.append(f"{label}: ~{format_duration(estimate)} (from {count} prior {run_word})")

    if not lines and not unestimated:
        return []

    report = [f"Estimated time (audio length: {format_duration(audio_duration_seconds)}):"]
    report.extend(f"  {line}" for line in lines)
    if lines:
        report.append(f"  Total: ~{format_duration(total_seconds)}")
    if unestimated:
        report.append(
            f"  No timing history yet for: {', '.join(unestimated)}"
            " -- this run will establish a baseline for next time."
        )
    return report
