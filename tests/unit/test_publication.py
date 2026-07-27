"""Tests for finalized publication layout and compact reports."""

from __future__ import annotations

import json
from pathlib import Path

from meeting_notes.config import MeetingNotesConfig
from meeting_notes.jobs import load_manifest, update_stage_status
from meeting_notes.pipeline import _run_finalize
from meeting_notes.publication import write_run_report


def _finalize_job(tmp_path: Path) -> tuple[Path, Path, dict, MeetingNotesConfig]:
    source_dir = tmp_path / "input"
    source_dir.mkdir()
    source = source_dir / "recording.m4a"
    source.write_bytes(b"audio")
    job = tmp_path / "job"
    for directory in ("source", "summary", "output", "transcript"):
        (job / directory).mkdir(parents=True, exist_ok=True)
    (job / "source" / source.name).write_bytes(b"audio")
    summary = {
        "title": "Demo",
        "short_title": "demo",
        "meeting_date": "2026-07-25",
    }
    (job / "summary" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (job / "output" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (job / "output" / "minutes.md").write_text("# Demo", encoding="utf-8")
    transcript = {
        "metadata": {"language": "ko"},
        "segments": [{"id": "seg-1", "start": 0, "end": 1, "text": "Hello"}],
    }
    (job / "transcript" / "transcript.merged.json").write_text(
        json.dumps(transcript), encoding="utf-8"
    )
    manifest = load_manifest(job)
    manifest["source"].update({"original_filename": source.name, "original_path": str(source)})
    config = MeetingNotesConfig(summarization={"enabled": False})
    return job, source, manifest, config


def test_initial_finalize_uses_nested_human_first_layout(tmp_path: Path) -> None:
    job, source, manifest, config = _finalize_job(tmp_path)
    result = _run_finalize(
        job,
        manifest,
        config,
        source,
        copy_to_input=True,
        run_id="run-1",
        started_at="2026-07-25T00:00:00+00:00",
    )

    root = job / "output" / "finalized" / "run-1"
    assert {path.name for path in root.iterdir()} == {
        "2026-07-25_demo.m4a",
        "2026-07-25_demo_meeting-notes.md",
        "2026-07-25_demo_transcript.md",
        "json",
        "subtitles",
        "run",
    }
    assert (root / "json" / "2026-07-25_demo_meeting-notes.json").exists()
    assert (root / "json" / "2026-07-25_demo_transcript.json").exists()
    assert (root / "subtitles" / "2026-07-25_demo_transcript.srt").exists()
    assert (root / "subtitles" / "2026-07-25_demo_transcript.vtt").exists()
    assert (root / "run" / "report.md").exists()
    assert (source.parent / "json" / "2026-07-25_demo_meeting-notes.json").exists()
    assert all(Path(path).is_file() for path in result["finalized"]["managed_paths"])
    assert all(Path(path).is_file() for path in result["finalized"]["external_paths"])


def test_local_markdown_finalize_uses_h1_and_omits_summary_json(tmp_path: Path) -> None:
    job, source, manifest, _ = _finalize_job(tmp_path)
    (job / "summary" / "summary.json").unlink()
    (job / "output" / "summary.json").unlink()
    markdown = (
        "> [!WARNING]\n> **Local AI — best-effort summary**\n\n"
        "# Local Planning Review\n\n## Executive summary\n\nUseful result.\n"
    )
    (job / "summary" / "summary.md").write_text(markdown, encoding="utf-8")
    (job / "output" / "minutes.md").write_text(markdown, encoding="utf-8")
    manifest["source"]["creation_time"] = "2026-07-25T10:00:00+09:00"
    manifest["stages"]["summarize"] = {
        "provider": {
            "output_format": "markdown",
            "quality_tier": "best_effort_local",
        }
    }
    config = MeetingNotesConfig(summarization={"enabled": True, "backend": "lemonade"})

    _run_finalize(
        job,
        manifest,
        config,
        source,
        run_id="run-local",
        started_at="2026-07-25T00:00:00+00:00",
    )

    root = job / "output" / "finalized" / "run-local"
    assert (root / "2026-07-25_Local-Planning-Review_meeting-notes.md").exists()
    assert not list((root / "json").glob("*meeting-notes.json"))
    report = (root / "run" / "report.md").read_text(encoding="utf-8")
    assert "Summary format: `markdown`" in report
    assert "Summary quality tier: `best_effort_local`" in report


def test_run_report_redacts_configured_secret(tmp_path: Path) -> None:
    config = MeetingNotesConfig(
        summarization={"claude": {"environment": {"ANTHROPIC_AUTH_TOKEN": "super-secret-token"}}}
    )
    report = write_run_report(
        tmp_path / "report.md",
        run_id="run-1",
        operation="pipeline",
        status="failed",
        started_at="2026-07-25T00:00:00+00:00",
        manifest={"stages": {}},
        config=config,
        error=RuntimeError("request failed for super-secret-token"),
    )
    text = report.read_text(encoding="utf-8")
    assert "super-secret-token" not in text
    assert "<redacted>" in text


def test_run_report_does_not_show_stale_errors_on_completed_stages(tmp_path: Path) -> None:
    report = write_run_report(
        tmp_path / "report.md",
        run_id="run-1",
        operation="pipeline",
        status="success",
        started_at="2026-07-25T00:00:00+00:00",
        manifest={
            "stages": {
                "prepare": {
                    "status": "completed",
                    "error": "an error from an earlier attempt",
                }
            }
        },
        config=MeetingNotesConfig(),
    )

    text = report.read_text(encoding="utf-8")
    assert "`prepare`: completed" in text
    assert "an error from an earlier attempt" not in text


def test_successful_stage_update_clears_previous_error() -> None:
    manifest = {"stages": {"prepare": {"status": "failed", "error": "old failure"}}}

    update_stage_status(manifest, "prepare", "completed")

    assert manifest["stages"]["prepare"] == {
        "status": "completed",
        "ended_at": manifest["stages"]["prepare"]["ended_at"],
    }
