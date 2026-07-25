"""Finalized publication layout and compact run reporting."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meeting_notes.transcript.render import format_timestamp

if TYPE_CHECKING:
    from meeting_notes.config import MeetingNotesConfig


def publication_paths(root: Path, names: dict[str, str]) -> dict[str, Path]:
    """Return the conventional human-first finalized output layout."""
    return {
        "recording": root / names["recording"],
        "minutes": root / names["minutes"],
        "transcript_markdown": root / names["transcript_markdown"],
        "json_export": root / "json" / names["json_export"],
        "transcript_json": root / "json" / names["transcript_json"],
        "transcript_srt": root / "subtitles" / names["transcript_srt"],
        "transcript_vtt": root / "subtitles" / names["transcript_vtt"],
        "run_report": root / "run" / "report.md",
    }


def managed_files(root: Path) -> list[str]:
    """List exact managed files recursively, excluding directory entries."""
    return [str(path) for path in sorted(root.rglob("*")) if path.is_file()]


def render_transcript_variants(data: dict[str, Any], directory: Path) -> dict[str, Path]:
    """Render transcript JSON, Markdown, SRT, and VTT into temporary flat files."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "transcript_json": directory / "transcript.json",
        "transcript_markdown": directory / "transcript.md",
        "transcript_srt": directory / "transcript.srt",
        "transcript_vtt": directory / "transcript.vtt",
    }
    paths["transcript_json"].write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = ["# Transcript", ""]
    srt: list[str] = []
    vtt = ["WEBVTT", ""]
    for index, segment in enumerate(data.get("segments", []), 1):
        name = str(segment.get("speaker") or segment.get("speaker_id") or "")
        text = str(segment.get("text", ""))
        start = float(segment.get("start", 0))
        end = float(segment.get("end", start))
        speaker_suffix = f" {name}" if name else ""
        md.extend(
            [
                f"**[{format_timestamp(start)}--{format_timestamp(end)}]{speaker_suffix}**",
                "",
                text,
                "",
                f"<!-- {segment.get('id', '')} -->",
                "",
            ]
        )
        subtitle = f"{name}: {text}" if name else text
        srt.extend(
            [
                str(index),
                (
                    f"{format_timestamp(start, 'HH:MM:SS,mmm')} --> "
                    f"{format_timestamp(end, 'HH:MM:SS,mmm')}"
                ),
                subtitle,
                "",
            ]
        )
        vtt.extend(
            [
                f"{format_timestamp(start)} --> {format_timestamp(end)}",
                subtitle,
                "",
            ]
        )
    paths["transcript_markdown"].write_text("\n".join(md), encoding="utf-8")
    paths["transcript_srt"].write_text("\n".join(srt), encoding="utf-8")
    paths["transcript_vtt"].write_text("\n".join(vtt), encoding="utf-8")
    return paths


def _path_fingerprint(raw: str | None) -> str | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_error(
    error: BaseException | str | None,
    sensitive_values: list[str] | None = None,
) -> str | None:
    if error is None:
        return None
    message = str(error).replace("\r", " ").replace("\n", " ").strip()
    for value in sensitive_values or []:
        if len(value) >= 4:
            message = message.replace(value, "<redacted>")
    return message[:500] if message else type(error).__name__


def _sensitive_values(config: MeetingNotesConfig) -> list[str]:
    values: list[str] = []
    environments = [
        config.summarization.claude.environment,
        config.summarization.local_command.environment,
    ]
    for environment in environments:
        for value in environment.values():
            match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
            resolved = os.environ.get(match.group(1), "") if match else value
            if resolved:
                values.append(resolved)
    return values


def write_run_report(
    path: Path,
    *,
    run_id: str,
    operation: str,
    status: str,
    started_at: str,
    manifest: dict[str, Any],
    config: MeetingNotesConfig,
    transcript_sha256: str | None = None,
    mapping_sha256: str | None = None,
    speaker_resolution: str | None = None,
    outputs: list[str] | None = None,
    error: BaseException | str | None = None,
    messages: list[str] | None = None,
    stages: dict[str, dict[str, Any]] | None = None,
    asr_activity: str | None = None,
    diarization_activity: str | None = None,
) -> Path:
    """Write a concise, sanitized Markdown record for one attempt."""
    from meeting_notes.summarization.adapters import summarizer_provenance

    ended_at = datetime.now(UTC).isoformat()
    provenance = summarizer_provenance(config.summarization)
    sensitive_values = _sensitive_values(config)
    reported_stages = manifest.get("stages", {}) if stages is None else stages
    asr_description = (
        asr_activity
        or f"{config.runtime.asr_backend} / {config.asr.model} / {config.runtime.device}"
    )
    diarization_description = diarization_activity or (
        "enabled" if config.diarization.enabled else "disabled"
    )
    lines = [
        "# Run Report",
        "",
        f"- Run ID: `{run_id}`",
        f"- Operation: `{operation}`",
        f"- Status: `{status}`",
        f"- Started: `{started_at}`",
        f"- Ended: `{ended_at}`",
        f"- ASR: `{asr_description}`",
        f"- Diarization: `{diarization_description}`",
        f"- Summarizer: `{provenance.get('backend')}`",
        f"- Requested model: `{provenance.get('requested_model') or 'provider default'}`",
        (
            "- Requested reasoning: "
            f"`{provenance.get('requested_reasoning_effort') or 'provider default'}`"
        ),
        f"- Launcher: `{provenance.get('launcher') or provenance.get('execution') or 'direct'}`",
    ]
    prompt_hash = _path_fingerprint(config.summarization.prompt_path)
    schema_hash = _path_fingerprint(config.summarization.output_schema_path)
    for label, value in (
        ("Transcript SHA-256", transcript_sha256),
        ("Speaker mapping SHA-256", mapping_sha256),
        ("Speaker resolution", speaker_resolution),
        ("Prompt SHA-256", prompt_hash),
        ("Schema SHA-256", schema_hash),
    ):
        if value is not None:
            lines.append(f"- {label}: `{value}`")

    lines.extend(["", "## Stages", ""])
    if reported_stages:
        for name, stage in reported_stages.items():
            message = _clean_error(stage.get("message"), sensitive_values)
            if not message and stage.get("status") == "failed":
                message = _clean_error(stage.get("error"), sensitive_values)
            suffix = f" — {message}" if message else ""
            lines.append(f"- `{name}`: {stage.get('status', 'unknown')}{suffix}")
    else:
        lines.append("- No pipeline stages were run.")

    clean_error = _clean_error(error, sensitive_values)
    if clean_error:
        lines.extend(["", "## Error", "", clean_error])

    if messages:
        lines.extend(["", "## Results", ""])
        lines.extend(f"- {message}" for message in messages)

    if outputs:
        lines.extend(["", "## Published Files", ""])
        lines.extend(f"- `{item}`" for item in outputs)

    lines.extend(
        [
            "",
            "This report intentionally excludes secrets, environment values, prompts, "
            "transcript content, raw subprocess output, and stack traces.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
