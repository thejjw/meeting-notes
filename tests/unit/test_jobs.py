"""Tests for job directory management and stage tracking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from meeting_notes.jobs import (
    compute_stage_fingerprint,
    create_job_dir,
    load_manifest,
    make_job_slug,
    save_manifest,
    stage_is_stale,
    update_stage_status,
)


class TestJobSlug:
    """Test job slug generation."""

    def test_slug_contains_date(self, tmp_path: Path) -> None:
        source = tmp_path / "recording.m4a"
        source.write_bytes(b"fake audio data")
        slug = make_job_slug(source)
        assert slug.startswith("2026-") or slug.startswith("2025-")

    def test_slug_with_title_hint(self, tmp_path: Path) -> None:
        source = tmp_path / "recording.m4a"
        source.write_bytes(b"fake audio data")
        slug = make_job_slug(source, title_hint="Customer Review")
        assert "customer-review" in slug

    def test_slug_is_filesystem_safe(self, tmp_path: Path) -> None:
        source = tmp_path / "recording.m4a"
        source.write_bytes(b"fake audio data")
        slug = make_job_slug(source, title_hint="API Auth/Review & Planning!")
        assert "/" not in slug
        assert "\\" not in slug
        assert " " not in slug


class TestJobDirectory:
    """Test job directory creation."""

    def test_create_job_dir(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        job_dir = create_job_dir(data_dir, "2026-07-22-test-abc123")
        assert job_dir.exists()
        assert (job_dir / "source").exists()
        assert (job_dir / "audio").exists()
        assert (job_dir / "asr").exists()
        assert (job_dir / "diarization").exists()
        assert (job_dir / "transcript").exists()
        assert (job_dir / "summary").exists()
        assert (job_dir / "output").exists()
        assert (job_dir / "output" / "runs").exists()
        assert (job_dir / "logs").exists()

    def test_create_job_dir_resume(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        job_dir = create_job_dir(data_dir, "2026-07-22-test-abc123")
        # Create a marker file
        marker = job_dir / "marker.txt"
        marker.write_text("existing")

        # Resume should reuse existing dir
        resumed = create_job_dir(data_dir, "2026-07-22-test-abc123", resume=True)
        assert resumed == job_dir
        assert marker.exists()


class TestManifest:
    """Test manifest loading and saving."""

    def test_load_manifest_creates_empty(self, tmp_path: Path) -> None:
        manifest = load_manifest(tmp_path)
        assert manifest["version"] == 1
        assert "stages" in manifest

    def test_save_and_load_manifest(self, tmp_path: Path) -> None:
        manifest = load_manifest(tmp_path)
        manifest["source"]["original_path"] = "/path/to/file.m4a"
        manifest["source"]["hash"] = "abc123"
        save_manifest(tmp_path, manifest)

        loaded = load_manifest(tmp_path)
        assert loaded["source"]["original_path"] == "/path/to/file.m4a"
        assert loaded["source"]["hash"] == "abc123"
        assert loaded["updated_at"] is not None


class TestStageTracking:
    """Test stage status updates and fingerprinting."""

    def test_update_stage_status(self, tmp_path: Path) -> None:
        manifest = load_manifest(tmp_path)
        update_stage_status(manifest, "transcribe", "running")
        assert manifest["stages"]["transcribe"]["status"] == "running"
        assert manifest["stages"]["transcribe"]["started_at"] is not None

        update_stage_status(manifest, "transcribe", "completed", output_hash="xyz")
        assert manifest["stages"]["transcribe"]["status"] == "completed"
        assert manifest["stages"]["transcribe"]["ended_at"] is not None
        assert manifest["stages"]["transcribe"]["output_hash"] == "xyz"

    def test_stage_is_stale_when_not_completed(self) -> None:
        manifest = {"stages": {"transcribe": {"status": "running"}}}
        assert stage_is_stale(manifest, "transcribe", "fp1") is True

    def test_stage_is_stale_when_fingerprint_changes(self) -> None:
        manifest = {
            "stages": {
                "transcribe": {"status": "completed", "config_fingerprint": "old_fp"}
            }
        }
        assert stage_is_stale(manifest, "transcribe", "new_fp") is True

    def test_stage_is_fresh_when_fingerprint_matches(self) -> None:
        manifest = {
            "stages": {
                "transcribe": {"status": "completed", "config_fingerprint": "same_fp"}
            }
        }
        assert stage_is_stale(manifest, "transcribe", "same_fp") is False


class TestStageFingerprint:
    """Test stage fingerprint computation."""

    def test_fingerprint_deterministic(self) -> None:
        fp1 = compute_stage_fingerprint("src_hash", None, model_hash="m1")
        fp2 = compute_stage_fingerprint("src_hash", None, model_hash="m1")
        assert fp1 == fp2

    def test_fingerprint_differs_with_model_change(self) -> None:
        fp1 = compute_stage_fingerprint("src_hash", None, model_hash="m1")
        fp2 = compute_stage_fingerprint("src_hash", None, model_hash="m2")
        assert fp1 != fp2

    def test_fingerprint_differs_with_glossary_change(self) -> None:
        fp1 = compute_stage_fingerprint("src_hash", None, glossary_hash="g1")
        fp2 = compute_stage_fingerprint("src_hash", None, glossary_hash="g2")
        assert fp1 != fp2
