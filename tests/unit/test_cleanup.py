"""Tests for safe final-only job cleanup."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from meeting_notes.cleanup import CleanupError, resolve_final_deliverables, run_final_only_cleanup
from meeting_notes.cli import app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _generation(job: Path, generation_id: str, title: str, *, recording: bool = True) -> Path:
    root = job / "output" / "finalized" / generation_id
    root.mkdir(parents=True)
    if recording:
        (root / f"{title}.m4a").write_bytes(f"audio-{title}".encode())
    (root / f"{title}_meeting-notes.md").write_text(f"notes-{title}", encoding="utf-8")
    (root / f"{title}_transcript.md").write_text(f"transcript-{title}", encoding="utf-8")
    (root / "json").mkdir()
    (root / "json" / "summary.json").write_text("{}", encoding="utf-8")
    return root


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "2026-07-28-demo"
    (job / "source").mkdir(parents=True)
    (job / "source" / "original.m4a").write_bytes(b"source-audio")
    initial_id = "20260728T010101000000Z"
    initial = _generation(job, initial_id, "initial")
    (job / "audio").mkdir()
    (job / "audio" / "normalized.wav").write_bytes(b"derived")
    manifest = {
        "version": 1,
        "source": {"original_filename": "original.m4a"},
        "finalized": {
            "generation_id": initial_id,
            "managed_paths": [str(path) for path in initial.rglob("*") if path.is_file()],
        },
    }
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return job


def _set_active(
    job: Path,
    key: str,
    generation_id: str,
    root: Path,
) -> None:
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    manifest[key] = {
        "active_generation": generation_id,
        "generations": [
            {
                "id": generation_id,
                "state": "active",
                "managed_paths": [str(path) for path in root.rglob("*") if path.is_file()],
            }
        ],
    }
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_resolve_uses_newest_active_publication(tmp_path: Path) -> None:
    job = _job(tmp_path)
    speaker_id = "20260728T020202000000Z"
    clarification_id = "20260728T030303000000Z"
    speaker = _generation(job, speaker_id, "speaker")
    clarification = _generation(job, clarification_id, "clarified")
    _set_active(job, "speaker_publications", speaker_id, speaker)
    _set_active(job, "clarification_publications", clarification_id, clarification)

    selected = resolve_final_deliverables(job)

    assert selected.recording.name == "clarified.m4a"
    assert selected.minutes.name == "clarified_meeting-notes.md"
    assert selected.transcript.name == "clarified_transcript.md"


def test_final_only_leaves_exactly_three_flat_files(tmp_path: Path) -> None:
    job = _job(tmp_path)

    run_final_only_cleanup(str(job), yes=True)

    assert sorted(path.name for path in job.iterdir()) == [
        "initial.m4a",
        "initial_meeting-notes.md",
        "initial_transcript.md",
    ]
    assert (job / "initial.m4a").read_bytes() == b"audio-initial"


def test_final_only_uses_source_recording_when_publication_has_none(tmp_path: Path) -> None:
    job = _job(tmp_path)
    generation_id = "20260728T040404000000Z"
    root = _generation(job, generation_id, "no-recording", recording=False)
    _set_active(job, "clarification_publications", generation_id, root)

    run_final_only_cleanup(str(job), yes=True)

    assert (job / "original.m4a").read_bytes() == b"source-audio"
    assert (job / "no-recording_meeting-notes.md").is_file()
    assert (job / "no-recording_transcript.md").is_file()


def test_dry_run_and_cancel_do_not_change_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(tmp_path)
    before = sorted(str(path.relative_to(job)) for path in job.rglob("*"))

    run_final_only_cleanup(str(job), dry_run=True)
    monkeypatch.setattr("meeting_notes.cleanup.typer.confirm", lambda prompt: False)
    run_final_only_cleanup(str(job))

    assert sorted(str(path.relative_to(job)) for path in job.rglob("*")) == before


def test_incomplete_newest_publication_fails_without_fallback(tmp_path: Path) -> None:
    job = _job(tmp_path)
    generation_id = "20260728T050505000000Z"
    root = job / "output" / "finalized" / generation_id
    root.mkdir(parents=True)
    (root / "broken_meeting-notes.md").write_text("notes", encoding="utf-8")
    _set_active(job, "speaker_publications", generation_id, root)

    with pytest.raises(CleanupError, match="Markdown transcript"):
        run_final_only_cleanup(str(job), yes=True)

    assert (job / "manifest.json").is_file()
    assert (job / "source" / "original.m4a").is_file()


def test_external_finalized_source_is_copied_but_not_deleted(tmp_path: Path) -> None:
    job = _job(tmp_path)
    external = tmp_path / "outside.m4a"
    external.write_bytes(b"outside")
    (job / "source" / "original.m4a").unlink()
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    manifest["source"]["finalized_path"] = str(external)
    initial = job / "output" / "finalized" / manifest["finalized"]["generation_id"]
    (initial / "initial.m4a").unlink()
    (job / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    run_final_only_cleanup(str(job), yes=True)

    assert external.read_bytes() == b"outside"
    assert (job / "outside.m4a").read_bytes() == b"outside"


def test_failed_directory_swap_restores_original_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(tmp_path)
    real_replace = __import__("os").replace
    calls = 0

    def fail_install(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated install failure")
        real_replace(source, destination)

    monkeypatch.setattr("meeting_notes.cleanup.os.replace", fail_install)

    with pytest.raises(CleanupError, match="original restored"):
        run_final_only_cleanup(str(job), yes=True)

    assert (job / "manifest.json").is_file()
    assert (job / "audio" / "normalized.wav").is_file()
    assert not list(tmp_path.glob(f".{job.name}.final-only-*"))
    assert not list(tmp_path.glob(f".{job.name}.cleanup-backup-*"))


def test_already_compact_job_is_a_noop(tmp_path: Path) -> None:
    job = tmp_path / "compact"
    job.mkdir()
    (job / "meeting.m4a").write_bytes(b"audio")
    (job / "meeting_meeting-notes.md").write_text("notes", encoding="utf-8")
    (job / "meeting_transcript.md").write_text("transcript", encoding="utf-8")

    run_final_only_cleanup(str(job), yes=True)

    assert len(list(job.iterdir())) == 3


def test_cli_validates_final_only_options(tmp_path: Path) -> None:
    job = _job(tmp_path)

    conflict = runner.invoke(
        app,
        ["clean", str(job), "--final-only", "--stage", "asr", "--yes"],
    )
    unsupported_dry_run = runner.invoke(app, ["clean", str(job), "--dry-run"])
    preview = runner.invoke(app, ["clean", str(job), "--final-only", "--dry-run"])

    assert conflict.exit_code == 2
    assert "mutually exclusive" in conflict.output
    assert unsupported_dry_run.exit_code == 2
    assert "requires --final-only" in unsupported_dry_run.output
    assert preview.exit_code == 0
    assert "Dry run only" in preview.output
    assert (job / "manifest.json").is_file()
