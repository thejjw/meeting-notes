"""Speaker identification templates and downstream-only regeneration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console

from meeting_notes.config import MeetingNotesConfig, load_config
from meeting_notes.jobs import atomic_write_text, file_sha256, load_manifest, save_manifest
from meeting_notes.minutes.render import render_minutes
from meeting_notes.naming import generate_filenames, resolve_date, sanitize_short_title
from meeting_notes.pipeline import _format_summary_transcript
from meeting_notes.publication import (
    managed_files,
    publication_paths,
    render_transcript_variants,
    write_run_report,
)
from meeting_notes.transcript.render import format_timestamp

console = Console(stderr=True)
TEMPLATE_VERSION = 1


class SpeakerMapError(ValueError):
    """An invalid or stale speaker map."""


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _mapping_hash(mapping: dict[str, str]) -> str:
    value = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    atomic_write_text(path, text)


def _transcript(job_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = job_dir / "transcript" / "transcript.merged.json"
    if not path.is_file():
        raise SpeakerMapError(f"Anonymous merged transcript not found: {path}. Run merge first.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise SpeakerMapError(f"Cannot read merged transcript: {error}") from error
    if not isinstance(data.get("segments"), list):
        raise SpeakerMapError(f"Merged transcript has no segments: {path}")
    return path, data


def _speaker_ids(data: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(segment["speaker"])
            for segment in data["segments"]
            if segment.get("speaker") and str(segment["speaker"]).upper() != "UNKNOWN"
        }
    )


def _examples(segments: list[dict[str, Any]], speaker_id: str) -> list[dict[str, str]]:
    candidates = [
        segment
        for segment in segments
        if segment.get("speaker") == speaker_id and len(str(segment.get("text", "")).strip()) >= 12
    ]
    if not candidates:
        candidates = [s for s in segments if s.get("speaker") == speaker_id]
    if len(candidates) <= 5:
        selected = candidates
    else:
        indexes = [round(i * (len(candidates) - 1) / 4) for i in range(5)]
        selected = [candidates[index] for index in indexes]
    return [
        {
            "timestamp": format_timestamp(float(segment.get("start", 0)), "HH:MM:SS"),
            "segment_id": str(segment.get("id", "")),
            "text": " ".join(str(segment.get("text", "")).split())[:300],
        }
        for segment in selected
    ]


def build_template(job_dir: Path, preserved_names: dict[str, str] | None = None) -> dict[str, Any]:
    path, data = _transcript(job_dir)
    preserved_names = preserved_names or {}
    speakers: dict[str, Any] = {}
    for speaker_id in _speaker_ids(data):
        segments = [s for s in data["segments"] if s.get("speaker") == speaker_id]
        speakers[speaker_id] = {
            "name": preserved_names.get(speaker_id, ""),
            "segment_count": len(segments),
            "total_seconds": round(
                sum(max(0.0, float(s.get("end", 0)) - float(s.get("start", 0))) for s in segments),
                1,
            ),
            "examples": _examples(data["segments"], speaker_id),
        }
    return {
        "version": TEMPLATE_VERSION,
        "job_id": job_dir.name,
        "transcript_sha256": _sha256(path),
        "speakers": speakers,
    }


def _read_names(path: Path) -> dict[str, str]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SpeakerMapError(f"Malformed speaker map {path}: {error}") from error
    if not isinstance(value, dict):
        raise SpeakerMapError(f"Speaker map must be a YAML mapping: {path}")
    source = value.get("speakers", value)
    if not isinstance(source, dict):
        raise SpeakerMapError("The 'speakers' value must be a mapping.")
    names: dict[str, str] = {}
    for speaker_id, entry in source.items():
        if speaker_id in {"version", "job_id", "transcript_sha256"}:
            continue
        if isinstance(entry, dict):
            entry = entry.get("name", "")
        if entry is None:
            entry = ""
        if not isinstance(entry, str):
            raise SpeakerMapError(f"Name for {speaker_id} must be a string.")
        names[str(speaker_id)] = entry.strip()
    return names


def write_template(
    job_dir: Path,
    output: Path | None = None,
    *,
    force: bool = False,
    automatic: bool = False,
) -> tuple[Path | None, str | None]:
    output = output or job_dir / "speakers.yaml"
    _, transcript = _transcript(job_dir)
    if not _speaker_ids(transcript):
        return None, None
    if output.exists() and not force:
        current = yaml.safe_load(output.read_text(encoding="utf-8")) or {}
        fingerprint = current.get("transcript_sha256") if isinstance(current, dict) else None
        expected = _sha256(job_dir / "transcript" / "transcript.merged.json")
        if fingerprint != expected:
            candidate = output.with_name(f"{output.stem}.candidate{output.suffix}")
            _atomic_text(
                candidate,
                yaml.safe_dump(build_template(job_dir), allow_unicode=True, sort_keys=False),
            )
            return (
                candidate,
                f"existing template is stale ({fingerprint or 'no fingerprint'} != {expected})",
            )
        return output, None

    names: dict[str, str] = {}
    if output.exists():
        names = _read_names(output)
        backup = output.with_name(
            f"{output.name}.bak-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        shutil.copy2(output, backup)
    template = build_template(job_dir, names)
    _atomic_text(output, yaml.safe_dump(template, allow_unicode=True, sort_keys=False))
    return output, None


def load_mapping(job_dir: Path, map_path: Path) -> tuple[dict[str, str], dict[str, Any], str]:
    transcript_path, transcript = _transcript(job_dir)
    try:
        document = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SpeakerMapError(f"Malformed speaker map {map_path}: {error}") from error
    if not isinstance(document, dict):
        raise SpeakerMapError(f"Speaker map must be a YAML mapping: {map_path}")
    rich = "speakers" in document
    if rich:
        if document.get("version") != TEMPLATE_VERSION:
            raise SpeakerMapError(
                f"Unsupported speaker template version {document.get('version')!r}; "
                "run `meeting-notes speakers template --force JOB_DIR`."
            )
        if document.get("job_id") != job_dir.name:
            raise SpeakerMapError(
                f"Speaker map belongs to job {document.get('job_id')!r}, not {job_dir.name!r}."
            )
        actual_hash = _sha256(transcript_path)
        if document.get("transcript_sha256") != actual_hash:
            raise SpeakerMapError(
                "Speaker map is stale for the current transcript; run "
                "`meeting-notes speakers template --force JOB_DIR`, review it, then apply again."
            )
    mapping = _read_names(map_path)
    known = set(_speaker_ids(transcript))
    unknown = set(mapping) - known
    if unknown:
        raise SpeakerMapError(f"Unknown speaker IDs in map: {', '.join(sorted(unknown))}.")
    return mapping, transcript, _sha256(transcript_path)


def _summarize(
    segments: list[dict[str, Any]],
    roster: list[str],
    config: MeetingNotesConfig,
    local_only: bool,
    *,
    speaker_resolution: str = "mapped",
) -> dict[str, Any]:
    if not config.summarization.enabled:
        return {
            "short_title": "meeting",
            "participants": [{"name": name} for name in roster],
        }
    backend = config.summarization.backend
    if local_only and backend in {"codex", "codex_cli", "opencode", "mimo", "claude"}:
        raise SpeakerMapError(
            f"Summarization adapter '{backend}' is not allowed with --local-only."
        )
    from meeting_notes.summarization.adapters import (
        configured_adapter_options,
        get_adapter,
    )

    kwargs = configured_adapter_options(config.summarization)
    adapter = get_adapter(backend, **kwargs)
    if not adapter.is_available():
        raise SpeakerMapError(f"Summarization adapter '{backend}' is not available.")
    prompt_path = Path(config.summarization.prompt_path)
    prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else "Summarize this meeting."
    )
    if speaker_resolution == "disabled":
        speaker_context = (
            "Speaker diarization was intentionally disabled because its attribution was "
            "unreliable. The participant roster is unavailable. Do not infer attendees, "
            "speaker identities, roles, or action-item owners from unattributed speech."
        )
    else:
        speaker_context = f"Resolved participant roster: {', '.join(roster) or '(none)'}"
    prompt = f"{speaker_context}\n\n{prompt}"
    schema = (
        Path(config.summarization.output_schema_path)
        if config.summarization.output_schema_path
        else None
    )
    summary = adapter.summarize(
        _format_summary_transcript(segments),
        prompt=prompt,
        schema_path=schema,
        timeout_seconds=config.summarization.timeout_seconds,
        metadata={
            "language": config.summarization.language,
            "speaker_resolution": speaker_resolution,
        },
    ).data
    if speaker_resolution == "disabled":
        summary["participants"] = []
        action_items = summary.get("action_items", [])
        if isinstance(action_items, list):
            for item in action_items:
                if isinstance(item, dict):
                    item["owner"] = None
    return summary


def _publish_recording(job_dir: Path, target: Path, manifest: dict[str, Any]) -> Path | None:
    source_name = manifest.get("source", {}).get("original_filename")
    if not source_name:
        return None
    source = job_dir / "source" / str(source_name)
    if not source.exists():
        publications = manifest.get("speaker_publications", {})
        active_id = publications.get("active_generation")
        active = next(
            (
                generation
                for generation in publications.get("generations", [])
                if generation.get("id") == active_id
            ),
            None,
        )
        extension = Path(str(source_name)).suffix.lower()
        if active:
            source = next(
                (
                    Path(path)
                    for path in active.get("managed_paths", [])
                    if Path(path).is_file() and Path(path).suffix.lower() == extension
                ),
                source,
            )
    if not source.exists():
        return None
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return target


def _cleanup_paths(paths: list[str], job_dir: Path, *, allow_external: bool = True) -> list[str]:
    residual: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = job_dir / path
        try:
            if not allow_external and job_dir.resolve() not in path.resolve().parents:
                continue
            if path.is_file() or path.is_symlink():
                path.unlink()
                if job_dir.resolve() in path.resolve().parents:
                    parent = path.parent
                    while parent != job_dir and parent.is_dir():
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError:
            residual.append(str(path))
    return residual


def _speaker_apply_report_stages(
    without_diarization: bool,
) -> dict[str, dict[str, str]]:
    """Describe the complete pipeline while distinguishing reused work."""
    diarization_message = (
        "existing labels removed"
        if without_diarization
        else "reused existing speaker labels"
    )
    return {
        "prepare": {"status": "skipped", "message": "reused existing job"},
        "transcribe": {"status": "skipped", "message": "reused merged transcript"},
        "diarize": {"status": "skipped", "message": diarization_message},
        "merge": {"status": "skipped", "message": "reused merged transcript"},
        "named transcript": {"status": "completed"},
        "summarize": {"status": "completed"},
        "render": {"status": "completed"},
        "finalize": {"status": "completed"},
    }


def apply_speakers(
    job_dir: Path,
    map_path: Path | None,
    config: MeetingNotesConfig,
    *,
    cleanup: bool = False,
    cleanup_all: bool = False,
    local_only: bool = False,
    without_diarization: bool = False,
) -> dict[str, Any]:
    from meeting_notes.summarization.adapters import summarizer_provenance

    if without_diarization:
        transcript_path, anonymous = _transcript(job_dir)
        transcript_hash = _sha256(transcript_path)
        mapping: dict[str, str] = {}
        speaker_resolution = "disabled"
    else:
        if map_path is None:
            raise SpeakerMapError("A speaker map is required unless --without-diarization is used.")
        mapping, anonymous, transcript_hash = load_mapping(job_dir, map_path)
        unresolved = [speaker for speaker in _speaker_ids(anonymous) if not mapping.get(speaker)]
        if unresolved:
            console.print(f"[yellow]Unresolved speakers retained: {', '.join(unresolved)}[/yellow]")
        speaker_resolution = "mapped"
    named = json.loads(json.dumps(anonymous))
    roster: list[str] = []
    for segment in named["segments"]:
        speaker_id = segment.get("speaker")
        if without_diarization:
            segment.pop("speaker", None)
            segment.pop("speaker_id", None)
        elif speaker_id:
            segment["speaker_id"] = speaker_id
            segment["speaker"] = mapping.get(str(speaker_id)) or speaker_id
            if segment["speaker"] not in roster and str(speaker_id).upper() != "UNKNOWN":
                roster.append(segment["speaker"])
    map_hash = None if without_diarization else _mapping_hash(mapping)
    named.setdefault("metadata", {}).update(
        {
            "speaker_resolution": speaker_resolution,
            "speaker_template_version": TEMPLATE_VERSION,
            "anonymous_transcript_sha256": transcript_hash,
            "speaker_mapping_sha256": map_hash,
            "participants": roster,
        }
    )

    manifest = load_manifest(job_dir)
    old_generations = manifest.get("speaker_publications", {}).get("generations", [])
    generation_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    started_at = datetime.now(UTC).isoformat()
    staging = job_dir / "output" / f".speakers-{generation_id}"
    generation = job_dir / "output" / "finalized" / generation_id
    staging.mkdir(parents=True)
    try:
        render_dir = staging / ".render"
        transcript_paths = render_transcript_variants(named, render_dir)
        summary = _summarize(
            named["segments"],
            roster,
            config,
            local_only,
            speaker_resolution=speaker_resolution,
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "minutes.md").write_text(
            render_minutes(
                summary, source_filename=manifest.get("source", {}).get("original_filename", "")
            ),
            encoding="utf-8",
        )
        title = sanitize_short_title(
            summary.get("short_title", "meeting"),
            max_length=config.naming.max_short_title_characters,
        )
        original = str(manifest.get("source", {}).get("original_filename") or "recording.m4a")
        source_path = Path(manifest.get("source", {}).get("original_path") or original)
        date, date_source = resolve_date(
            summary,
            media_creation_time=manifest.get("source", {}).get("creation_time", ""),
            source_mtime=source_path.stat().st_mtime if source_path.exists() else 0,
            source_order=config.naming.date_source_order,
        )
        names = generate_filenames(
            date,
            title,
            Path(original).suffix,
            recording_template=config.naming.recording_template,
            minutes_template=config.naming.minutes_template,
            json_template=config.naming.json_export_template,
            transcript_json_template=config.naming.transcript_json_template,
            transcript_markdown_template=config.naming.transcript_markdown_template,
            transcript_srt_template=config.naming.transcript_srt_template,
            transcript_vtt_template=config.naming.transcript_vtt_template,
        )
        layout = publication_paths(staging, names)
        for key, source in transcript_paths.items():
            target = layout[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        shutil.rmtree(render_dir)
        (staging / "minutes.md").rename(layout["minutes"])
        layout["json_export"].parent.mkdir(parents=True, exist_ok=True)
        (staging / "summary.json").rename(layout["json_export"])
        _publish_recording(job_dir, layout["recording"], manifest)
        relative_outputs = [
            str(path.relative_to(staging))
            for key, path in layout.items()
            if key != "run_report" and path.exists()
        ]
        write_run_report(
            layout["run_report"],
            run_id=generation_id,
            operation="speakers apply",
            status="success",
            started_at=started_at,
            manifest=manifest,
            config=config,
            transcript_sha256=transcript_hash,
            mapping_sha256=map_hash,
            speaker_resolution=speaker_resolution,
            outputs=relative_outputs,
            messages=[f"Published {len(relative_outputs)} output files."],
            stages=_speaker_apply_report_stages(without_diarization),
            asr_activity="not run (reused merged transcript)",
            diarization_activity=(
                "not run (speaker labels removed)"
                if without_diarization
                else "not run (reused existing speaker labels)"
            ),
        )
        generation.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, generation)
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        with suppress(OSError):
            write_run_report(
                job_dir / "output" / "runs" / generation_id / "report.md",
                run_id=generation_id,
                operation="speakers apply",
                status="failed",
                started_at=started_at,
                manifest=manifest,
                config=config,
                transcript_sha256=transcript_hash,
                mapping_sha256=map_hash,
                speaker_resolution=speaker_resolution,
                error=error,
                stages={},
                asr_activity="not run (reused merged transcript)",
                diarization_activity=(
                    "not run (speaker labels removed)"
                    if without_diarization
                    else "not run (reused existing speaker labels)"
                ),
            )
        raise

    final_layout = publication_paths(generation, names)
    managed = managed_files(generation)
    canonical_sources = {
        job_dir / "transcript" / "transcript.named.json": final_layout["transcript_json"],
        job_dir / "transcript" / "transcript.named.md": final_layout["transcript_markdown"],
        job_dir / "transcript" / "transcript.named.srt": final_layout["transcript_srt"],
        job_dir / "transcript" / "transcript.named.vtt": final_layout["transcript_vtt"],
        job_dir / "summary" / "summary.json": final_layout["json_export"],
        job_dir / "output" / "minutes.md": final_layout["minutes"],
        job_dir / "output" / "summary.json": final_layout["json_export"],
    }
    for target, source in canonical_sources.items():
        _atomic_text(target, source.read_text(encoding="utf-8"))

    generations = []
    for item in old_generations:
        copied = dict(item)
        copied["state"] = "superseded"
        generations.append(copied)
    generation_record = {
        "id": generation_id,
        "state": "active",
        "created_at": datetime.now(UTC).isoformat(),
        "mapping_sha256": map_hash,
        "transcript_sha256": transcript_hash,
        "speaker_resolution": speaker_resolution,
        "summarizer": summarizer_provenance(config.summarization),
        "managed_paths": managed,
        "external_paths": [],
        "date_source": date_source,
    }
    generations.append(generation_record)
    template_provenance = manifest.setdefault("speaker_template", {})
    template_provenance["version"] = TEMPLATE_VERSION
    template_provenance["transcript_sha256"] = transcript_hash
    template_provenance["active_mapping_sha256"] = map_hash
    if map_path is not None:
        template_provenance["path"] = str(map_path)
    manifest["speaker_publications"] = {
        "active_generation": generation_id,
        "generations": generations,
    }
    save_manifest(job_dir, manifest)

    residual: list[str] = []
    if cleanup or cleanup_all:
        obsolete = [
            path
            for item in generations[:-1]
            for path in item.get("managed_paths", []) + item.get("external_paths", [])
        ]
        residual.extend(_cleanup_paths(obsolete, job_dir))
    if cleanup_all:
        preserve = {
            (job_dir / "manifest.json").resolve(),
            (job_dir / "transcript" / "transcript.merged.json").resolve(),
            *(path.resolve() for path in canonical_sources),
            generation.resolve(),
        }
        if map_path is not None:
            preserve.add(map_path.resolve())
        candidates = [
            job_dir / "source",
            job_dir / "audio",
            job_dir / "asr",
            job_dir / "diarization",
            job_dir / "logs",
        ]
        for candidate in candidates:
            if candidate.resolve() not in preserve:
                residual.extend(_cleanup_paths([str(candidate)], job_dir, allow_external=False))
    if cleanup or cleanup_all:
        generation_record["cleanup"] = {
            "mode": "all" if cleanup_all else "superseded",
            "completed": not residual,
            "residual_paths": residual,
        }
        save_manifest(job_dir, manifest)
        cleanup_mode = "all reproducible artifacts" if cleanup_all else "superseded outputs"
        write_run_report(
            final_layout["run_report"],
            run_id=generation_id,
            operation="speakers apply",
            status="success" if not residual else "success_with_cleanup_errors",
            started_at=started_at,
            manifest=manifest,
            config=config,
            transcript_sha256=transcript_hash,
            mapping_sha256=map_hash,
            speaker_resolution=speaker_resolution,
            outputs=relative_outputs,
            error=(
                "Cleanup residual paths: " + ", ".join(residual)
                if residual
                else None
            ),
            messages=[
                f"Published {len(relative_outputs)} output files.",
                f"Cleanup of {cleanup_mode} completed."
                if not residual
                else f"Cleanup of {cleanup_mode} left {len(residual)} residual path(s).",
            ],
            stages={
                **_speaker_apply_report_stages(without_diarization),
                "cleanup": {"status": "completed" if not residual else "failed"},
            },
            asr_activity="not run (reused merged transcript)",
            diarization_activity=(
                "not run (speaker labels removed)"
                if without_diarization
                else "not run (reused existing speaker labels)"
            ),
        )
    if residual:
        raise SpeakerMapError(
            "Publication succeeded, but cleanup failed for: " + ", ".join(residual)
        )
    return generation_record


def load_meeting_config(config_path: str | None) -> MeetingNotesConfig:
    return load_config(config_path)


def command_template(job_dir: str, output: str | None, force: bool) -> None:
    path, warning = write_template(
        Path(job_dir).resolve(), Path(output).resolve() if output else None, force=force
    )
    if warning:
        console.print(f"[yellow]{warning}; wrote replacement candidate: {path}[/yellow]")
    elif path:
        console.print(str(path))
    else:
        console.print("[yellow]No stable diarization speaker labels found.[/yellow]")


def command_apply(
    job_dir: str,
    map_path: str | None,
    config_path: str | None,
    cleanup: bool,
    cleanup_all: bool,
    yes: bool,
    local_only: bool,
    without_diarization: bool,
) -> None:
    path = Path(job_dir).resolve()
    if without_diarization and map_path is not None:
        raise SpeakerMapError("--map cannot be combined with --without-diarization.")
    mapping = None
    if not without_diarization:
        mapping = Path(map_path).resolve() if map_path else path / "speakers.yaml"
    if (cleanup or cleanup_all) and not yes:
        if not os.isatty(0):
            raise SpeakerMapError("Cleanup requires --yes in non-interactive use.")
        if not typer.confirm("Delete tracked superseded artifacts after successful publication?"):
            raise typer.Abort()
    record = apply_speakers(
        path,
        mapping,
        load_meeting_config(config_path),
        cleanup=cleanup,
        cleanup_all=cleanup_all,
        local_only=local_only,
        without_diarization=without_diarization,
    )
    console.print(f"[green]Activated speaker publication {record['id']}[/green]")
