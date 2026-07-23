"""Tests for audio chunking, glossary, VAD, and diarization."""

from __future__ import annotations

from pathlib import Path

import pytest

from meeting_notes.audio.chunk import (
    AudioChunk,
    compute_chunks,
    load_chunks_manifest,
    save_chunks_manifest,
)
from meeting_notes.transcript.glossary import (
    Glossary,
    GlossaryTerm,
    apply_glossary_corrections,
    build_initial_prompt,
    load_glossary,
)
from meeting_notes.diarization.reconcile import (
    apply_speaker_map,
    assign_speakers,
    load_speaker_map,
)
from meeting_notes.transcript.models import TranscriptSegment
from meeting_notes.diarization.base import DiarizationTurn
from meeting_notes.vad.none import NoVADBackend


class TestChunking:
    """Test audio chunking logic."""

    def test_short_recording_no_chunks(self) -> None:
        chunks = compute_chunks(600.0, mode="auto", trigger_duration_minutes=45)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chunk-0000"
        assert chunks[0].source_start == 0.0
        assert chunks[0].source_end == 600.0

    def test_long_recording_chunked(self) -> None:
        chunks = compute_chunks(5400.0, mode="auto", max_chunk_minutes=20, trigger_duration_minutes=45)
        assert len(chunks) > 1
        # First chunk should start at 0
        assert chunks[0].source_start == 0.0
        # Last chunk should end at total duration
        assert chunks[-1].source_end == 5400.0

    def test_overlap_applied(self) -> None:
        chunks = compute_chunks(5400.0, mode="auto", max_chunk_minutes=20, overlap_seconds=5.0)
        # Second chunk should have overlap_before
        if len(chunks) > 1:
            assert chunks[1].overlap_before == 5.0
            assert chunks[1].source_start < chunks[1].source_end

    def test_mode_none_returns_single(self) -> None:
        chunks = compute_chunks(10000.0, mode="none")
        assert len(chunks) == 1

    def test_chunks_manifest_roundtrip(self, tmp_path: Path) -> None:
        chunks = [
            AudioChunk(chunk_id="chunk-0000", source_start=0, source_end=600),
            AudioChunk(chunk_id="chunk-0001", source_start=598, source_end=1200),
        ]
        path = tmp_path / "chunks.json"
        save_chunks_manifest(chunks, path)
        loaded = load_chunks_manifest(path)
        assert len(loaded) == 2
        assert loaded[0].chunk_id == "chunk-0000"
        assert loaded[1].overlap_before == 0.0


class TestGlossary:
    """Test glossary loading and correction."""

    def test_load_empty_glossary(self) -> None:
        g = load_glossary(None)
        assert len(g.terms) == 0

    def test_load_glossary_from_file(self, tmp_path: Path) -> None:
        glossary_path = tmp_path / "glossary.yaml"
        glossary_path.write_text(
            "terms:\n"
            "  - canonical: Z3Soft\n"
            "    aliases:\n"
            "      - 지쓰리소프트\n"
            "  - canonical: API\n"
            "    aliases:\n"
            "      - 에이피아이\n",
            encoding="utf-8",
        )
        g = load_glossary(glossary_path)
        assert len(g.terms) == 2
        assert g.terms[0].canonical == "Z3Soft"

    def test_build_initial_prompt(self) -> None:
        g = Glossary(terms=[
            GlossaryTerm(canonical="Z3Soft"),
            GlossaryTerm(canonical="API"),
        ])
        prompt = build_initial_prompt(g)
        assert "Z3Soft" in prompt
        assert "API" in prompt

    def test_apply_corrections(self) -> None:
        g = Glossary(terms=[
            GlossaryTerm(canonical="API", aliases=["에이피아이"]),
        ])
        corrected, corrections = apply_glossary_corrections(
            "에이피아이 방식으로 진행합니다",
            g,
        )
        assert "API" in corrected
        assert len(corrections) == 1
        assert corrections[0].rule_canonical == "API"

    def test_no_correction_when_no_match(self) -> None:
        g = Glossary(terms=[
            GlossaryTerm(canonical="Z3Soft", aliases=["지쓰리소프트"]),
        ])
        corrected, corrections = apply_glossary_corrections(
            "일반적인 회의 내용입니다",
            g,
        )
        assert corrected == "일반적인 회의 내용입니다"
        assert len(corrections) == 0

    def test_whole_word_only(self) -> None:
        g = Glossary(terms=[
            GlossaryTerm(canonical="API", aliases=["에이피"]),
        ])
        corrected, _ = apply_glossary_corrections(
            "에이피아이 방식",  # 에이피 inside 에이피아이
            g,
        )
        # Should NOT replace because 에이피 is inside 에이피아이
        assert "에이피아이" in corrected


class TestDiarizationReconcile:
    """Test speaker assignment and mapping."""

    def test_assign_speakers_maximum_overlap(self) -> None:
        segments = [
            TranscriptSegment(id="s1", start=0, end=5, text="hello"),
            TranscriptSegment(id="s2", start=6, end=10, text="world"),
        ]
        turns = [
            DiarizationTurn(turn_id="t1", start=0, end=5, speaker="SPEAKER_00"),
            DiarizationTurn(turn_id="t2", start=5.5, end=10, speaker="SPEAKER_01"),
        ]
        result = assign_speakers(segments, turns)
        assert result[0].speaker == "SPEAKER_00"
        assert result[1].speaker == "SPEAKER_01"

    def test_assign_unknown_when_no_overlap(self) -> None:
        segments = [
            TranscriptSegment(id="s1", start=100, end=105, text="distant"),
        ]
        turns = [
            DiarizationTurn(turn_id="t1", start=0, end=5, speaker="SPEAKER_00"),
        ]
        result = assign_speakers(segments, turns, nearest_tolerance_seconds=1.0)
        assert result[0].speaker == "UNKNOWN"

    def test_empty_turns(self) -> None:
        segments = [
            TranscriptSegment(id="s1", start=0, end=5, text="hello"),
        ]
        result = assign_speakers(segments, [])
        assert result[0].speaker is None

    def test_apply_speaker_map(self) -> None:
        segments = [
            TranscriptSegment(id="s1", start=0, end=5, text="hello", speaker="SPEAKER_00"),
        ]
        speaker_map = {"SPEAKER_00": "John"}
        result = apply_speaker_map(segments, speaker_map)
        assert result[0].speaker == "John"

    def test_load_speaker_map(self, tmp_path: Path) -> None:
        map_path = tmp_path / "speakers.yaml"
        map_path.write_text(
            "SPEAKER_00: John\nSPEAKER_01: Jane\n",
            encoding="utf-8",
        )
        sm = load_speaker_map(str(map_path))
        assert sm["SPEAKER_00"] == "John"
        assert sm["SPEAKER_01"] == "Jane"

    def test_load_speaker_map_none(self) -> None:
        assert load_speaker_map(None) == {}


class TestVADNone:
    """Test no-op VAD backend."""

    def test_no_vad_always_available(self) -> None:
        assert NoVADBackend().is_available() is True

    def test_no_vad_returns_empty(self, tmp_path: Path) -> None:
        # Create a dummy file (VAD none won't actually read it)
        dummy = tmp_path / "dummy.wav"
        dummy.write_bytes(b"\x00" * 100)
        result = NoVADBackend().detect(dummy)
        assert result == []
