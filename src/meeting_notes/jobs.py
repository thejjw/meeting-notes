"""Job directory management and stage tracking."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

# Allowed characters for job slug (keep it filesystem-safe)
_SLUG_PATTERN = re.compile(r"[^a-z0-9\-]")


def _file_hash(path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:16]


def _text_hash(text: str) -> str:
    """Compute SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_job_slug(source_path: Path, title_hint: str | None = None) -> str:
    """Create a filesystem-safe job slug from source path and optional title.

    Format: YYYY-MM-DD-<short-slug>-<hash>
    Uses source file modification date for consistency across runs.
    """
    # Use source file's modification date for consistent slugs
    if source_path.exists():
        mtime = source_path.stat().st_mtime
        from datetime import datetime as dt
        date_str = dt.fromtimestamp(mtime).strftime("%Y-%m-%d")
    else:
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")

    if title_hint:
        slug_part = _SLUG_PATTERN.sub("-", title_hint.lower().strip())[:40]
        slug_part = re.sub(r"-+", "-", slug_part).strip("-")
    else:
        stem = source_path.stem[:30]
        slug_part = _SLUG_PATTERN.sub("-", stem.lower().strip())[:40]
        slug_part = re.sub(r"-+", "-", slug_part).strip("-")

    path_hash = _file_hash(source_path) if source_path.exists() else _text_hash(str(source_path))

    return f"{date_str}-{slug_part}-{path_hash}"


def create_job_dir(data_dir: Path, slug: str, *, resume: bool = True) -> Path:
    """Create or reuse a job directory with standard subdirectories.

    Returns the job directory path.
    """
    job_dir = data_dir / "meetings" / slug

    if job_dir.exists() and resume:
        log.info("job.resuming", job_dir=str(job_dir))
        return job_dir

    # Create all subdirectories
    subdirs = [
        "source",
        "audio",
        "audio/chunks",
        "asr",
        "diarization",
        "transcript",
        "summary",
        "output",
        "output/finalized",
        "logs",
    ]
    for sub in subdirs:
        (job_dir / sub).mkdir(parents=True, exist_ok=True)

    log.info("job.created", job_dir=str(job_dir))
    return job_dir


def _empty_manifest() -> dict[str, Any]:
    """Create an empty manifest structure."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "source": {
            "original_path": None,
            "original_filename": None,
            "hash": None,
        },
        "stages": {},
        "config_fingerprint": None,
        "artifacts": {},
    }


def load_manifest(job_dir: Path) -> dict[str, Any]:
    """Load or create a manifest for a job directory."""
    manifest_path = job_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return _empty_manifest()


def save_manifest(job_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically save manifest to disk."""
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = job_dir / "manifest.json"
    tmp_path = manifest_path.with_suffix(".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Atomic rename
    if manifest_path.exists():
        manifest_path.unlink()
    os.rename(str(tmp_path), str(manifest_path))


def update_stage_status(
    manifest: dict[str, Any],
    stage_name: str,
    status: str,
    *,
    config_fingerprint: str | None = None,
    tool_version: str | None = None,
    input_hash: str | None = None,
    output_hash: str | None = None,
    error: str | None = None,
) -> None:
    """Update a stage's status in the manifest."""
    now = datetime.now(timezone.utc).isoformat()
    stages = manifest.setdefault("stages", {})
    stage = stages.setdefault(stage_name, {})

    stage["status"] = status

    if status == "running":
        stage["started_at"] = now
        stage["ended_at"] = None
    elif status in ("completed", "failed", "skipped", "cancelled"):
        stage["ended_at"] = now

    if config_fingerprint:
        stage["config_fingerprint"] = config_fingerprint
    if tool_version:
        stage["tool_version"] = tool_version
    if input_hash:
        stage["input_hash"] = input_hash
    if output_hash:
        stage["output_hash"] = output_hash
    if error:
        stage["error"] = error


def compute_stage_fingerprint(
    source_hash: str,
    config: Any,
    model_hash: str = "",
    glossary_hash: str = "",
    prompt_hash: str = "",
) -> str:
    """Compute a fingerprint for a pipeline stage based on inputs.

    Used to determine if a stage needs re-running when config changes.
    """
    parts = [source_hash]
    if model_hash:
        parts.append(model_hash)
    if glossary_hash:
        parts.append(glossary_hash)
    if prompt_hash:
        parts.append(prompt_hash)
    return _text_hash("|".join(parts))


def stage_is_stale(manifest: dict[str, Any], stage_name: str, current_fingerprint: str) -> bool:
    """Check if a stage needs re-running based on fingerprint change."""
    stages = manifest.get("stages", {})
    stage = stages.get(stage_name, {})
    saved_fp = stage.get("config_fingerprint", "")
    saved_status = stage.get("status", "")

    if saved_status != "completed":
        return True
    if saved_fp and saved_fp != current_fingerprint:
        return True
    return False
