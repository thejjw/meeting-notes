"""Long transcript chunking for hierarchical summarization."""

from __future__ import annotations

import structlog

from meeting_notes.transcript.models import TranscriptDocument, TranscriptSegment

log = structlog.get_logger()


def chunk_transcript(
    doc: TranscriptDocument,
    *,
    target_characters: int = 60000,
    overlap_segments: int = 2,
) -> list[TranscriptDocument]:
    """Split a transcript into chunks for hierarchical summarization.

    Chunks are split at segment boundaries to preserve context.

    Args:
        doc: Full transcript document.
        target_characters: Target character count per chunk.
        overlap_segments: Number of segments to overlap between chunks.

    Returns:
        List of TranscriptDocument chunks.
    """
    if not doc.segments:
        return [doc]

    total_chars = sum(len(s.text) for s in doc.segments)
    if total_chars <= target_characters:
        return [doc]

    chunks: list[TranscriptDocument] = []
    current_segments: list[TranscriptSegment] = []
    current_chars = 0

    for seg in doc.segments:
        seg_chars = len(seg.text)

        if current_chars + seg_chars > target_characters and current_segments:
            # Save current chunk
            chunks.append(
                TranscriptDocument(
                    segments=list(current_segments),
                    language=doc.language,
                    backend=doc.backend,
                    model=doc.model,
                    source_file=doc.source_file,
                )
            )

            # Start new chunk with overlap
            overlap_start = max(0, len(current_segments) - overlap_segments)
            current_segments = current_segments[overlap_start:]
            current_chars = sum(len(s.text) for s in current_segments)

        current_segments.append(seg)
        current_chars += seg_chars

    # Final chunk
    if current_segments:
        chunks.append(
            TranscriptDocument(
                segments=current_segments,
                language=doc.language,
                backend=doc.backend,
                model=doc.model,
                source_file=doc.source_file,
            )
        )

    log.info(
        "transcript.chunked",
        total_segments=len(doc.segments),
        total_chars=total_chars,
        chunks=len(chunks),
    )

    return chunks


def format_chunk_for_summarization(
    chunk: TranscriptDocument,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Format a transcript chunk as text for the summarizer.

    Includes metadata and segment timestamps for context.
    """
    lines = [
        f"Chunk {chunk_index + 1} of {total_chunks}",
        f"Language: {chunk.language}",
        "",
    ]

    for seg in chunk.segments:
        h = int(seg.start // 3600)
        m = int((seg.start % 3600) // 60)
        s = int(seg.start % 60)
        ts = f"{h:02d}:{m:02d}:{s:02d}"
        speaker = f" [{seg.speaker}]" if seg.speaker else ""
        lines.append(f"[{ts}]{speaker} {seg.text}")

    return "\n".join(lines)
