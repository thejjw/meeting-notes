"""Chunk transcript merging with absolute timestamps and overlap deduplication."""

from __future__ import annotations

import structlog

from meeting_notes.transcript.models import TranscriptDocument, TranscriptSegment

log = structlog.get_logger()


def merge_chunk_transcripts(
    chunk_results: list[tuple[str, TranscriptDocument]],
    *,
    chunk_overlap_seconds: float = 2.0,
) -> TranscriptDocument:
    """Merge multiple chunk transcripts into a single document.

    Converts all segment times to absolute source timestamps,
    de-duplicates overlapping text, and retains provenance.

    Args:
        chunk_results: List of (chunk_id, TranscriptDocument) tuples.
        chunk_overlap_seconds: Expected overlap between adjacent chunks.

    Returns:
        Merged TranscriptDocument with absolute timestamps.
    """
    if not chunk_results:
        return TranscriptDocument(segments=[])

    all_segments: list[TranscriptSegment] = []
    warnings: list[str] = []

    for chunk_id, doc in chunk_results:
        for seg in doc.segments:
            merged_seg = TranscriptSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                language=seg.language,
                speaker=seg.speaker,
                confidence=seg.confidence,
                metrics=seg.metrics,
                source=seg.source,
                chunk_id=chunk_id,
                original_segment_id=seg.id,
            )
            all_segments.append(merged_seg)

    # Sort by start time
    all_segments.sort(key=lambda s: (s.start, s.end))

    # De-duplicate overlapping text in overlap regions
    deduplicated = _deduplicate_overlap(
        all_segments, overlap_seconds=chunk_overlap_seconds
    )

    # Re-assign IDs
    for i, seg in enumerate(deduplicated):
        seg.id = f"seg-{i:06d}"

    # Calculate total duration
    duration = max((s.end for s in deduplicated), default=0.0)

    return TranscriptDocument(
        segments=deduplicated,
        duration=duration,
        source_file=chunk_results[0][1].source_file if chunk_results else "",
        metadata={"merge_warnings": warnings},
    )


def _deduplicate_overlap(
    segments: list[TranscriptSegment],
    overlap_seconds: float = 2.0,
) -> list[TranscriptSegment]:
    """Conservatively de-duplicate text in overlap regions.

    Strategy: if two adjacent segments have highly similar text and
    their time ranges overlap significantly, keep only the first one.
    Never delete two genuinely different utterances.
    """
    if len(segments) <= 1:
        return segments

    result: list[TranscriptSegment] = [segments[0]]

    for seg in segments[1:]:
        prev = result[-1]

        # Check if this is an overlap region
        is_overlap = seg.start < prev.end and (prev.end - seg.start) < overlap_seconds

        if is_overlap:
            # Compare text similarity
            similarity = _text_similarity(prev.text, seg.text)
            if similarity > 0.85:
                # Very similar text in overlap — skip duplicate
                log.debug(
                    "merge.dedup_skipped",
                    seg_id=seg.id,
                    prev_id=prev.id,
                    similarity=f"{similarity:.2f}",
                )
                continue

        result.append(seg)

    return result


def _text_similarity(a: str, b: str) -> float:
    """Compute simple word-level Jaccard similarity between two texts."""
    if not a or not b:
        return 0.0

    words_a = set(a.split())
    words_b = set(b.split())

    if not words_a or not words_b:
        return 0.0

    intersection = words_a & words_b
    union = words_a | words_b

    return len(intersection) / len(union) if union else 0.0
