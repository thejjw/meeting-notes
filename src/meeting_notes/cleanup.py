"""Safe cleanup of completed meeting jobs."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from meeting_notes.jobs import file_sha256

console = Console(stderr=True)

_GENERATION_ID = re.compile(r"^\d{8}T\d{12,}Z$")
_MINUTES_SUFFIX = "_meeting-notes.md"
_TRANSCRIPT_SUFFIX = "_transcript.md"


class CleanupError(RuntimeError):
    """Raised when a cleanup cannot be completed safely."""


@dataclass(frozen=True)
class FinalDeliverables:
    """The three files retained by final-only cleanup."""

    recording: Path
    minutes: Path
    transcript: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("recording", self.recording),
            ("meeting notes", self.minutes),
            ("transcript", self.transcript),
        )


def _read_manifest(job_dir: Path) -> dict[str, Any]:
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.is_file():
        raise CleanupError(f"Job manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CleanupError(f"Cannot read job manifest: {error}") from error
    if not isinstance(data, dict):
        raise CleanupError("Job manifest must contain a JSON object.")
    return cast("dict[str, Any]", data)


def _generation_ids(manifest: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    finalized = manifest.get("finalized")
    if isinstance(finalized, dict):
        finalized_data = cast("dict[str, Any]", finalized)
        generation_id = finalized_data.get("generation_id")
        if isinstance(generation_id, str):
            ids.append(generation_id)
    for key in ("speaker_publications", "clarification_publications"):
        publications = manifest.get(key)
        if isinstance(publications, dict):
            publication_data = cast("dict[str, Any]", publications)
            active = publication_data.get("active_generation")
            if isinstance(active, str):
                ids.append(active)
    if not ids:
        raise CleanupError("The manifest does not contain a finalized publication.")
    invalid = sorted({item for item in ids if not _GENERATION_ID.fullmatch(item)})
    if invalid:
        raise CleanupError(f"Unsafe publication generation ID in manifest: {invalid[0]}")
    return ids


def _one_file(paths: list[Path], label: str, generation: Path) -> Path:
    if len(paths) != 1:
        raise CleanupError(
            f"Expected exactly one {label} in {generation}, found {len(paths)}."
        )
    path = paths[0]
    if path.is_symlink() or not path.is_file():
        raise CleanupError(f"The selected {label} is not a regular file: {path}")
    return path


def _source_recording(job_dir: Path, manifest: dict[str, Any]) -> Path | None:
    source = manifest.get("source")
    if not isinstance(source, dict):
        return None
    source_data = cast("dict[str, Any]", source)
    original_filename = source_data.get("original_filename")
    if isinstance(original_filename, str) and original_filename:
        candidate = job_dir / "source" / Path(original_filename).name
        if candidate.is_file():
            return candidate
    finalized_path = source_data.get("finalized_path")
    if isinstance(finalized_path, str) and finalized_path:
        candidate = Path(finalized_path).expanduser()
        if candidate.is_file():
            return candidate
    return None


def resolve_final_deliverables(job_dir: Path) -> FinalDeliverables:
    """Resolve the newest finalized recording, notes, and Markdown transcript."""
    job_dir = job_dir.resolve()
    manifest = _read_manifest(job_dir)
    generation_id = max(_generation_ids(manifest))
    generation = job_dir / "output" / "finalized" / generation_id
    if not generation.is_dir() or generation.is_symlink():
        raise CleanupError(f"Newest finalized publication is missing: {generation}")

    top_level = [path for path in generation.iterdir() if path.is_file()]
    minutes = _one_file(
        [path for path in top_level if path.name.endswith(_MINUTES_SUFFIX)],
        "Markdown meeting-notes file",
        generation,
    )
    transcript = _one_file(
        [path for path in top_level if path.name.endswith(_TRANSCRIPT_SUFFIX)],
        "Markdown transcript file",
        generation,
    )
    recordings = [
        path
        for path in top_level
        if path not in {minutes, transcript} and path.suffix.lower() != ".md"
    ]
    if len(recordings) == 1:
        recording = _one_file(recordings, "recording", generation)
    elif len(recordings) > 1:
        raise CleanupError(
            f"Expected at most one recording in {generation}, found {len(recordings)}."
        )
    else:
        recording = _source_recording(job_dir, manifest)
        if recording is None:
            raise CleanupError("No finalized or manifest-tracked source recording was found.")
        if recording.is_symlink():
            raise CleanupError(f"The selected recording is a symlink: {recording}")

    return FinalDeliverables(recording, minutes, transcript)


def _already_compact(job_dir: Path) -> bool:
    if not job_dir.is_dir() or job_dir.is_symlink():
        return False
    entries = list(job_dir.iterdir())
    if len(entries) != 3 or any(not path.is_file() or path.is_symlink() for path in entries):
        return False
    names = [path.name for path in entries]
    return (
        sum(name.endswith(_MINUTES_SUFFIX) for name in names) == 1
        and sum(name.endswith(_TRANSCRIPT_SUFFIX) for name in names) == 1
        and sum(not name.endswith(".md") for name in names) == 1
    )


def _validate_job_path(job_dir: Path) -> Path:
    if job_dir.is_symlink():
        raise CleanupError(f"Refusing to clean a symlinked job directory: {job_dir}")
    resolved = job_dir.resolve()
    if not resolved.is_dir():
        raise CleanupError(f"Job directory not found: {resolved}")
    if resolved.parent == resolved:
        raise CleanupError("Refusing to clean a filesystem root.")
    return resolved


def _show_preview(job_dir: Path, deliverables: FinalDeliverables) -> None:
    console.print("[bold]Final-only cleanup preview[/bold]")
    console.print("\n[green]Retain in job root:[/green]")
    for label, source in deliverables.items():
        console.print(f"  {label}: {source} -> {job_dir / source.name}")
    console.print("\n[red]Remove from the current job:[/red]")
    for path in sorted(job_dir.rglob("*"), key=lambda item: str(item).lower()):
        console.print(f"  {path.relative_to(job_dir)}")


def _copy_and_verify(deliverables: FinalDeliverables, staging: Path) -> None:
    staging.mkdir()
    names: set[str] = set()
    for label, source in deliverables.items():
        if source.name in names:
            raise CleanupError(f"Retained filenames collide at {source.name!r}.")
        names.add(source.name)
        target = staging / source.name
        try:
            shutil.copy2(source, target)
        except OSError as error:
            raise CleanupError(f"Could not stage {label}: {error}") from error
        if source.stat().st_size != target.stat().st_size:
            raise CleanupError(f"Size verification failed while staging {label}.")
        if file_sha256(source) != file_sha256(target):
            raise CleanupError(f"SHA-256 verification failed while staging {label}.")


def run_final_only_cleanup(job_dir: str, *, yes: bool = False, dry_run: bool = False) -> None:
    """Flatten a completed job to its recording, meeting notes, and transcript."""
    job_path = _validate_job_path(Path(job_dir))
    if _already_compact(job_path):
        console.print("[green]Job is already reduced to the three final files.[/green]")
        return

    deliverables = resolve_final_deliverables(job_path)
    _show_preview(job_path, deliverables)
    if dry_run:
        console.print("\n[yellow]Dry run only; no files were changed.[/yellow]")
        return
    if not yes and not typer.confirm("Replace this job with only the three final files?"):
        console.print("[yellow]Cleanup cancelled; no files were changed.[/yellow]")
        return

    token = uuid.uuid4().hex
    staging = job_path.parent / f".{job_path.name}.final-only-{token}"
    backup = job_path.parent / f".{job_path.name}.cleanup-backup-{token}"
    swapped = False
    try:
        _copy_and_verify(deliverables, staging)
        os.replace(job_path, backup)
        try:
            os.replace(staging, job_path)
            swapped = True
        except OSError as error:
            os.replace(backup, job_path)
            raise CleanupError(
                f"Could not install compact job; original restored: {error}"
            ) from error
        try:
            shutil.rmtree(backup)
        except OSError as error:
            raise CleanupError(
                f"Compact job created, but the recoverable backup remains at {backup}: {error}"
            ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if not swapped and backup.exists() and not job_path.exists():
            os.replace(backup, job_path)

    console.print("[green]Final-only cleanup complete.[/green]")
    for path in sorted(job_path.iterdir()):
        console.print(f"  {path}")
