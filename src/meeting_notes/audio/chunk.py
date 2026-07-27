"""Audio chunking with absolute timestamp preservation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from meeting_notes.subprocess_utils import run_command

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()


@dataclass
class AudioChunk:
    """A chunk of audio with absolute source timestamps."""

    chunk_id: str
    source_start: float
    source_end: float
    overlap_before: float = 0.0
    overlap_after: float = 0.0
    path: str = ""

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


def compute_chunks(
    total_duration: float,
    *,
    mode: str = "auto",
    max_chunk_minutes: float = 20.0,
    overlap_seconds: float = 2.0,
    trigger_duration_minutes: float = 45.0,
) -> list[AudioChunk]:
    """Compute chunk boundaries for a recording.

    Args:
        total_duration: Total audio duration in seconds.
        mode: 'auto' (chunk only if long), 'fixed', or 'none'.
        max_chunk_minutes: Maximum chunk duration in minutes.
        overlap_seconds: Overlap between adjacent chunks.
        trigger_duration_minutes: Duration threshold for auto chunking.

    Returns:
        List of AudioChunk objects.
    """
    max_chunk_sec = max_chunk_minutes * 60
    trigger_sec = trigger_duration_minutes * 60

    if mode == "none" or (mode == "auto" and total_duration <= trigger_sec):
        return [
            AudioChunk(
                chunk_id="chunk-0000",
                source_start=0.0,
                source_end=total_duration,
            )
        ]

    chunks: list[AudioChunk] = []
    pos = 0.0
    chunk_idx = 0

    while pos < total_duration:
        end = min(pos + max_chunk_sec, total_duration)
        overlap_before = overlap_seconds if chunk_idx > 0 else 0.0
        overlap_after = overlap_seconds if end < total_duration else 0.0

        chunks.append(
            AudioChunk(
                chunk_id=f"chunk-{chunk_idx:04d}",
                source_start=max(0.0, pos - overlap_before),
                source_end=min(total_duration, end + overlap_after),
                overlap_before=overlap_before,
                overlap_after=overlap_after,
            )
        )
        pos = end
        chunk_idx += 1

    return chunks


def save_chunks_manifest(chunks: list[AudioChunk], output_path: Path) -> Path:
    """Save chunk metadata to JSON."""
    data = [
        {
            "chunk_id": c.chunk_id,
            "source_start": c.source_start,
            "source_end": c.source_end,
            "overlap_before": c.overlap_before,
            "overlap_after": c.overlap_after,
            "path": c.path,
        }
        for c in chunks
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def load_chunks_manifest(path: Path) -> list[AudioChunk]:
    """Load chunk metadata from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        AudioChunk(
            chunk_id=c["chunk_id"],
            source_start=c["source_start"],
            source_end=c["source_end"],
            overlap_before=c.get("overlap_before", 0.0),
            overlap_after=c.get("overlap_after", 0.0),
            path=c.get("path", ""),
        )
        for c in data
    ]


def materialize_audio_chunks(
    input_path: Path,
    chunks: list[AudioChunk],
    output_dir: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> list[AudioChunk]:
    """Extract timestamped PCM WAV chunks with FFmpeg."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Normalized audio not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        output_path = output_dir / f"{chunk.chunk_id}.wav"
        args = [
            ffmpeg_path,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{chunk.source_start:.3f}",
            "-t",
            f"{chunk.duration:.3f}",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]
        result = run_command(args, timeout=300.0, label=f"ffmpeg-{chunk.chunk_id}")
        if not result.success:
            raise RuntimeError(
                f"FFmpeg chunk extraction failed for {chunk.chunk_id}: "
                f"{result.stderr[:500]}"
            )
        chunk.path = str(output_path)
    return chunks
