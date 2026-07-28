"""Speaker-to-transcript segment reconciliation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from meeting_notes.diarization.base import DiarizationTurn
    from meeting_notes.transcript.models import TranscriptSegment

log = structlog.get_logger()


def assign_speakers(
    segments: list[TranscriptSegment],
    turns: list[DiarizationTurn],
    *,
    assignment_method: str = "maximum_overlap",
    minimum_overlap_ratio: float = 0.15,
    nearest_tolerance_seconds: float = 1.0,
    unknown_label: str = "UNKNOWN",
) -> list[TranscriptSegment]:
    """Assign speaker labels to transcript segments using diarization turns.

    Assignment order:
    1. Speaker with greatest duration overlap
    2. If overlap below threshold, nearest diarization turn within tolerance
    3. Otherwise UNKNOWN

    Args:
        segments: Transcript segments to label.
        turns: Diarization turns with speaker labels.
        assignment_method: 'maximum_overlap' or 'nearest'.
        minimum_overlap_ratio: Minimum overlap ratio to accept assignment.
        nearest_tolerance_seconds: Max gap for nearest-turn fallback.
        unknown_label: Label for unmatched segments.

    Returns:
        Same segments list with speaker field updated (in-place modification).
    """
    if not turns:
        return segments

    for seg in segments:
        speaker = _find_best_speaker(
            seg,
            turns,
            assignment_method=assignment_method,
            minimum_overlap_ratio=minimum_overlap_ratio,
            nearest_tolerance_seconds=nearest_tolerance_seconds,
        )
        seg.speaker = speaker or unknown_label

    return segments


def _find_best_speaker(
    segment: TranscriptSegment,
    turns: list[DiarizationTurn],
    *,
    assignment_method: str = "maximum_overlap",
    minimum_overlap_ratio: float = 0.15,
    nearest_tolerance_seconds: float = 1.0,
) -> str | None:
    """Find the best speaker for a transcript segment."""
    seg_duration = segment.end - segment.start
    if seg_duration <= 0:
        return None

    # Method 1: Maximum overlap
    best_speaker = None
    best_overlap = 0.0

    for turn in turns:
        overlap_start = max(segment.start, turn.start)
        overlap_end = min(segment.end, turn.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

    overlap_ratio = best_overlap / seg_duration if seg_duration > 0 else 0.0

    if overlap_ratio >= minimum_overlap_ratio:
        return best_speaker

    # Method 2: Nearest turn fallback
    min_distance = float("inf")
    nearest_speaker = None

    for turn in turns:
        # Distance from segment center to turn center
        seg_center = (segment.start + segment.end) / 2
        turn_center = (turn.start + turn.end) / 2
        distance = abs(seg_center - turn_center)

        if distance < min_distance and distance <= nearest_tolerance_seconds:
            min_distance = distance
            nearest_speaker = turn.speaker

    if nearest_speaker:
        return nearest_speaker

    return None


def apply_speaker_map(
    segments: list[TranscriptSegment],
    speaker_map: dict[str, str],
) -> list[TranscriptSegment]:
    """Apply a user-provided speaker mapping to rename labels.

    Args:
        segments: Transcript segments with speaker labels.
        speaker_map: Mapping from anonymous labels to real names.

    Returns:
        Same segments with speaker labels renamed.
    """
    for seg in segments:
        if seg.speaker and seg.speaker in speaker_map:
            seg.speaker = speaker_map[seg.speaker]

    return segments


def load_speaker_map(speaker_map_path: str | None) -> dict[str, str]:
    """Load speaker mapping from YAML file.

    Args:
        speaker_map_path: Path to speakers.yaml. None = empty map.

    Returns:
        Dict mapping anonymous labels to real names.
    """
    if not speaker_map_path:
        return {}

    from pathlib import Path

    path = Path(speaker_map_path)
    if not path.exists():
        return {}

    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not data:
        return {}

    return {str(k): str(v) for k, v in data.items()}
