"""Transcript data models (shared between ASR output and merge stage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptSegment:
    """A single transcript segment with full metadata."""

    id: str
    start: float
    end: float
    text: str
    language: str | None = None
    speaker: str | None = None
    confidence: float | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    # Merge metadata
    chunk_id: str | None = None
    original_segment_id: str | None = None
    merge_warnings: list[str] = field(default_factory=list)


@dataclass
class TranscriptDocument:
    """Complete transcript with metadata and segments."""

    segments: list[TranscriptSegment]
    language: str = ""
    duration: float = 0.0
    backend: str = ""
    model: str = ""
    device: str = ""
    source_file: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
