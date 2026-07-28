"""Tests for transcript chunk merging and overlap deduplication."""

from __future__ import annotations

from meeting_notes.transcript.merge import (
    _deduplicate_overlap,
    _text_similarity,
    merge_chunk_transcripts,
)
from meeting_notes.transcript.models import TranscriptDocument, TranscriptSegment


class TestTextSimilarity:
    """Test text similarity calculation."""

    def test_identical_texts(self) -> None:
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self) -> None:
        assert _text_similarity("hello", "goodbye") == 0.0

    def test_partial_overlap(self) -> None:
        sim = _text_similarity("hello world test", "hello world foo")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self) -> None:
        assert _text_similarity("", "") == 0.0
        assert _text_similarity("hello", "") == 0.0


class TestOverlapDeduplication:
    """Test overlap deduplication logic."""

    def test_no_overlap(self) -> None:
        segments = [
            TranscriptSegment(id="s1", start=0, end=5, text="hello"),
            TranscriptSegment(id="s2", start=6, end=10, text="world"),
        ]
        result = _deduplicate_overlap(segments, overlap_seconds=2.0)
        assert len(result) == 2

    def test_overlap_similar_text_deduplicated(self) -> None:
        segments = [
            TranscriptSegment(id="s1", start=0, end=5, text="hello world test sentence"),
            TranscriptSegment(id="s2", start=4, end=10, text="hello world test sentence"),
        ]
        result = _deduplicate_overlap(segments, overlap_seconds=2.0)
        assert len(result) == 1

    def test_overlap_different_text_kept(self) -> None:
        segments = [
            TranscriptSegment(id="s1", start=0, end=5, text="hello world"),
            TranscriptSegment(id="s2", start=4, end=10, text="completely different text here"),
        ]
        result = _deduplicate_overlap(segments, overlap_seconds=2.0)
        assert len(result) == 2

    def test_empty_segments(self) -> None:
        result = _deduplicate_overlap([], overlap_seconds=2.0)
        assert len(result) == 0

    def test_single_segment(self) -> None:
        segments = [TranscriptSegment(id="s1", start=0, end=5, text="hello")]
        result = _deduplicate_overlap(segments, overlap_seconds=2.0)
        assert len(result) == 1


class TestMergeChunkTranscripts:
    """Test full chunk transcript merging."""

    def test_merge_empty(self) -> None:
        result = merge_chunk_transcripts([])
        assert len(result.segments) == 0

    def test_merge_single_chunk(self) -> None:
        doc = TranscriptDocument(
            segments=[
                TranscriptSegment(id="s1", start=0, end=5, text="hello"),
                TranscriptSegment(id="s2", start=6, end=10, text="world"),
            ]
        )
        result = merge_chunk_transcripts([("chunk-0001", doc)])
        assert len(result.segments) == 2
        assert result.segments[0].chunk_id == "chunk-0001"

    def test_merge_two_chunks_no_overlap(self) -> None:
        doc1 = TranscriptDocument(
            segments=[TranscriptSegment(id="s1", start=0, end=5, text="hello")]
        )
        doc2 = TranscriptDocument(
            segments=[TranscriptSegment(id="s1", start=10, end=15, text="world")]
        )
        result = merge_chunk_transcripts([
            ("chunk-0001", doc1),
            ("chunk-0002", doc2),
        ])
        assert len(result.segments) == 2
        # Segments should be sorted by start time
        assert result.segments[0].start == 0
        assert result.segments[1].start == 10

    def test_merge_preserves_provenance(self) -> None:
        doc = TranscriptDocument(
            segments=[TranscriptSegment(id="orig-s1", start=0, end=5, text="test")]
        )
        result = merge_chunk_transcripts([("chunk-0001", doc)])
        seg = result.segments[0]
        assert seg.chunk_id == "chunk-0001"
        assert seg.original_segment_id == "orig-s1"

    def test_merge_reassigns_ids(self) -> None:
        doc1 = TranscriptDocument(
            segments=[TranscriptSegment(id="s1", start=0, end=5, text="a")]
        )
        doc2 = TranscriptDocument(
            segments=[TranscriptSegment(id="s1", start=10, end=15, text="b")]
        )
        result = merge_chunk_transcripts([
            ("chunk-0001", doc1),
            ("chunk-0002", doc2),
        ])
        assert result.segments[0].id == "seg-000000"
        assert result.segments[1].id == "seg-000001"
