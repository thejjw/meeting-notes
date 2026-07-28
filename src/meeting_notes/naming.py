"""Post-summary filename finalization with date and topic."""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()

# Windows reserved device names
_WIN_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def sanitize_short_title(
    title: str,
    *,
    max_length: int = 48,
    preserve_unicode: bool = True,
    whitespace_replacement: str = "-",
) -> str:
    """Sanitize a short title for use in filenames.

    Rules:
    - Remove path separators and control characters
    - Replace whitespace with configured separator
    - Collapse repeated separators
    - Trim trailing spaces and dots
    - Avoid Windows reserved device names
    - Preserve Korean and English text
    """
    if not title:
        return "meeting"

    # Remove path separators and control characters
    sanitized = re.sub(r'[/\\:*?"<>|\x00-\x1f]', "", title)

    # Replace whitespace
    sanitized = re.sub(r"\s+", whitespace_replacement, sanitized.strip())

    # Collapse repeated separators
    sanitized = re.sub(
        r"[" + re.escape(whitespace_replacement) + r"]{2,}", whitespace_replacement, sanitized
    )

    # Trim trailing spaces and dots (Windows issue)
    sanitized = sanitized.rstrip(". ")

    # Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip(". ")

    # Check Windows reserved names
    stem = sanitized.split(".")[0].upper()
    if stem in _WIN_RESERVED:
        sanitized = f"file-{sanitized}"

    return sanitized or "meeting"


def resolve_date(
    summary: dict | None,
    media_creation_time: str = "",
    source_mtime: float = 0.0,
    source_order: list[str] | None = None,
) -> tuple[str, str]:
    """Resolve the meeting date from multiple sources.

    Returns (date_string, source_name) tuple.
    """
    if source_order is None:
        source_order = [
            "summary_meeting_date",
            "media_creation_time",
            "source_mtime",
            "processing_date",
        ]

    for source in source_order:
        if source == "summary_meeting_date" and summary:
            date = summary.get("meeting_date")
            if date:
                return str(date), "summary_meeting_date"

        elif source == "media_creation_time" and media_creation_time:
            # Parse ISO format
            try:
                dt = datetime.fromisoformat(media_creation_time.replace("Z", "+00:00"))
                return dt.strftime("%Y-%m-%d"), "media_creation_time"
            except (ValueError, AttributeError):
                continue

        elif source == "source_mtime" and source_mtime:
            try:
                dt = datetime.fromtimestamp(source_mtime)
                return dt.strftime("%Y-%m-%d"), "source_mtime"
            except (ValueError, OSError):
                continue

        elif source == "processing_date":
            return datetime.now().strftime("%Y-%m-%d"), "processing_date"

    return datetime.now().strftime("%Y-%m-%d"), "processing_date"


def generate_filenames(
    date: str,
    short_title: str,
    original_extension: str,
    *,
    recording_template: str = "{date}_{short_title}{extension}",
    minutes_template: str = "{date}_{short_title}_meeting-notes.md",
    json_template: str = "{date}_{short_title}_meeting-notes.json",
    transcript_json_template: str = "{date}_{short_title}_transcript.json",
    transcript_markdown_template: str = "{date}_{short_title}_transcript.md",
    transcript_srt_template: str = "{date}_{short_title}_transcript.srt",
    transcript_vtt_template: str = "{date}_{short_title}_transcript.vtt",
) -> dict[str, str]:
    """Generate finalized filenames from date and title.

    Returns dict with keys: 'recording', 'minutes', 'json_export'.
    """
    ext = original_extension if original_extension.startswith(".") else f".{original_extension}"

    return {
        "recording": recording_template.format(date=date, short_title=short_title, extension=ext),
        "minutes": minutes_template.format(date=date, short_title=short_title),
        "json_export": json_template.format(date=date, short_title=short_title),
        "transcript_json": transcript_json_template.format(date=date, short_title=short_title),
        "transcript_markdown": transcript_markdown_template.format(
            date=date, short_title=short_title
        ),
        "transcript_srt": transcript_srt_template.format(date=date, short_title=short_title),
        "transcript_vtt": transcript_vtt_template.format(date=date, short_title=short_title),
    }


def resolve_collision(
    target_path: Path,
    policy: str = "increment",
) -> Path:
    """Handle filename collisions by applying the configured policy.

    Policies:
    - 'increment': append _02, _03, etc.
    - 'short_hash': append _<hash>
    - 'error': raise FileExistsError
    """
    if not target_path.exists():
        return target_path

    if policy == "error":
        raise FileExistsError(f"File already exists: {target_path}")

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent

    if policy == "increment":
        counter = 2
        while True:
            new_name = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1

    elif policy == "short_hash":
        import hashlib

        hash_val = hashlib.md5(str(target_path).encode()).hexdigest()[:8]
        return parent / f"{stem}_{hash_val}{suffix}"

    return target_path


def finalize_recording(
    source_path: Path,
    target_dir: Path,
    filename: str,
    mode: str = "managed_copy",
    copy_method: str = "auto",
) -> Path:
    """Finalize recording file based on the configured mode.

    Modes:
    - 'managed_copy': copy/link to job output dir (non-destructive)
    - 'in_place': rename original in its directory
    - 'none': leave recording unchanged
    """
    if mode == "none":
        return source_path

    target = resolve_collision(target_dir / filename)

    if mode == "in_place":
        # Rename in original directory
        new_path = source_path.parent / filename
        new_path = resolve_collision(new_path)
        os.rename(str(source_path), str(new_path))
        log.info("naming.in_place", old=str(source_path), new=str(new_path))
        return new_path

    # managed_copy: copy or hardlink to target dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if copy_method == "hardlink" or (
        copy_method == "auto" and _same_filesystem(source_path, target)
    ):
        try:
            os.link(str(source_path), str(target))
            log.info("naming.hardlink", source=str(source_path), target=str(target))
            return target
        except OSError:
            pass

    shutil.copy2(source_path, target)
    log.info("naming.copy", source=str(source_path), target=str(target))
    return target


def _same_filesystem(path1: Path, path2: Path) -> bool:
    """Check if two paths are on the same filesystem."""
    try:
        return path1.stat().st_dev == path2.stat().st_dev
    except OSError:
        return False
