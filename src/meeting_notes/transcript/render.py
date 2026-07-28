"""Transcript output rendering (JSON, Markdown, SRT, VTT)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from meeting_notes.asr.base import ASRResult


def format_timestamp(seconds: float, fmt: str = "HH:MM:SS.mmm") -> str:
    """Format seconds as a timestamp string.

    Supported formats:
    - HH:MM:SS.mmm (default)
    - HH:MM:SS
    - HH:MM:SS,mmm (SRT format)
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60

    if fmt == "HH:MM:SS.mmm":
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    elif fmt == "HH:MM:SS":
        return f"{h:02d}:{m:02d}:{int(s):02d}"
    elif fmt == "HH:MM:SS,mmm":
        ms = int((s % 1) * 1000)
        return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"
    else:
        return f"{h:02d}:{m:02d}:{s:06.3f}"


def render_json(result: ASRResult, output_path: Path) -> Path:
    """Write transcript as normalized JSON."""
    data = {
        "metadata": {
            "language": result.language,
            "backend": result.backend,
            "model": result.model,
            "device": result.device,
            "duration": result.duration,
        },
        "segments": [
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "language": seg.language,
                "speaker": seg.speaker,
                "confidence": seg.confidence,
                "metrics": seg.metrics,
                "source": seg.source,
            }
            for seg in result.segments
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def render_markdown(
    result: ASRResult,
    output_path: Path,
    *,
    source_filename: str = "",
    title: str = "Transcript",
) -> Path:
    """Write transcript as timestamped Markdown."""
    lines = [
        f"# {title}",
        "",
    ]

    if source_filename:
        lines.append(f"- Source: `{source_filename}`")
    if result.duration > 0:
        lines.append(f"- Duration: {format_timestamp(result.duration, 'HH:MM:SS')}")
    lines.append(f"- ASR: `{result.backend} / {result.model} / {result.device}`")
    lines.append(f"- Language: `{result.language}`")
    lines.append("")
    lines.append("## Transcript")
    lines.append("")

    for seg in result.segments:
        ts_start = format_timestamp(seg.start, "HH:MM:SS.mmm")
        ts_end = format_timestamp(seg.end, "HH:MM:SS.mmm")
        speaker = f" {seg.speaker}" if seg.speaker else ""
        lines.append(f"**[{ts_start}--{ts_end}]{speaker}**")
        lines.append("")
        lines.append(seg.text)
        lines.append("")
        lines.append(f"<!-- {seg.id} -->")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def render_srt(result: ASRResult, output_path: Path) -> Path:
    """Write transcript as SRT subtitle file."""
    lines: list[str] = []

    for i, seg in enumerate(result.segments, 1):
        ts_start = format_timestamp(seg.start, "HH:MM:SS,mmm")
        ts_end = format_timestamp(seg.end, "HH:MM:SS,mmm")
        lines.append(str(i))
        lines.append(f"{ts_start} --> {ts_end}")
        lines.append(seg.text)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def render_vtt(result: ASRResult, output_path: Path) -> Path:
    """Write transcript as WebVTT subtitle file."""
    lines = ["WEBVTT", ""]

    for seg in result.segments:
        ts_start = format_timestamp(seg.start, "HH:MM:SS.mmm")
        ts_end = format_timestamp(seg.end, "HH:MM:SS.mmm")
        lines.append(f"{ts_start} --> {ts_end}")
        lines.append(seg.text)
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def render_all_formats(
    result: ASRResult,
    asr_dir: Path,
    *,
    source_filename: str = "",
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """Render transcript in all requested formats.

    Args:
        result: ASR result to render.
        asr_dir: Output directory (e.g., job_dir/asr/).
        source_filename: Original source filename for metadata.
        formats: List of formats to render. Default: json, md, srt, vtt.

    Returns:
        Dict mapping format name to output path.
    """
    if formats is None:
        formats = ["json", "md", "srt", "vtt"]

    output_paths: dict[str, Path] = {}

    if "json" in formats:
        output_paths["json"] = render_json(result, asr_dir / "transcript.raw.json")

    if "md" in formats:
        output_paths["md"] = render_markdown(
            result,
            asr_dir / "transcript.raw.md",
            source_filename=source_filename,
        )

    if "srt" in formats:
        output_paths["srt"] = render_srt(result, asr_dir / "transcript.srt")

    if "vtt" in formats:
        output_paths["vtt"] = render_vtt(result, asr_dir / "transcript.vtt")

    return output_paths
