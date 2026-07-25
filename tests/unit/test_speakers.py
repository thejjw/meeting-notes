"""Tests for speaker identification and downstream regeneration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from meeting_notes.config import MeetingNotesConfig
from meeting_notes.jobs import load_manifest, save_manifest
from meeting_notes.speakers import (
    SpeakerMapError,
    _summarize,
    apply_speakers,
    load_mapping,
    write_template,
)


def _job(tmp_path: Path) -> Path:
    job = tmp_path / "2026-07-23-demo"
    (job / "transcript").mkdir(parents=True)
    (job / "source").mkdir()
    (job / "source" / "recording.m4a").write_bytes(b"recording")
    data = {
        "metadata": {"language": "ko"},
        "segments": [
            {
                "id": f"seg-{index:06d}",
                "start": index * 70.0,
                "end": index * 70.0 + 10,
                "speaker": "SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
                "text": f"Representative sentence number {index} for this meeting.",
            }
            for index in range(12)
        ],
    }
    (job / "transcript" / "transcript.merged.json").write_text(json.dumps(data), encoding="utf-8")
    manifest = load_manifest(job)
    manifest["source"].update(
        {"original_filename": "recording.m4a", "original_path": str(job / "recording.m4a")}
    )
    save_manifest(job, manifest)
    return job


def test_template_is_deterministic_and_preserves_names(tmp_path: Path) -> None:
    job = _job(tmp_path)
    path, warning = write_template(job)
    assert warning is None
    assert path == job / "speakers.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert len(document["speakers"]["SPEAKER_00"]["examples"]) == 5
    document["speakers"]["SPEAKER_00"]["name"] = "홍길동"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")

    write_template(job, force=True)
    regenerated = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert regenerated["speakers"]["SPEAKER_00"]["name"] == "홍길동"
    assert list(job.glob("speakers.yaml.bak-*"))


def test_stale_and_unknown_maps_fail_with_remediation(tmp_path: Path) -> None:
    job = _job(tmp_path)
    path, _ = write_template(job)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["transcript_sha256"] = "stale"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(SpeakerMapError, match="template --force"):
        load_mapping(job, path)

    path.write_text("SPEAKER_99: Someone\n", encoding="utf-8")
    with pytest.raises(SpeakerMapError, match="Unknown speaker IDs"):
        load_mapping(job, path)


def test_apply_writes_named_formats_and_generation(tmp_path: Path) -> None:
    job = _job(tmp_path)
    path, _ = write_template(job)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["speakers"]["SPEAKER_00"]["name"] = "민지"
    document["speakers"]["SPEAKER_01"]["name"] = "민지"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    config = MeetingNotesConfig(summarization={"enabled": False})

    generation = apply_speakers(job, path, config)

    named = json.loads((job / "transcript" / "transcript.named.json").read_text(encoding="utf-8"))
    assert named["segments"][0]["speaker_id"] == "SPEAKER_00"
    assert named["segments"][0]["speaker"] == "민지"
    assert named["metadata"]["participants"] == ["민지"]
    assert "민지:" in (job / "transcript" / "transcript.named.srt").read_text(encoding="utf-8")
    manifest = load_manifest(job)
    assert manifest["speaker_publications"]["active_generation"] == generation["id"]
    assert all(Path(item).exists() for item in generation["managed_paths"])


def test_apply_without_diarization_needs_no_map(tmp_path: Path) -> None:
    job = _job(tmp_path)
    config = MeetingNotesConfig(summarization={"enabled": False})

    generation = apply_speakers(job, None, config, without_diarization=True)

    named = json.loads((job / "transcript" / "transcript.named.json").read_text(encoding="utf-8"))
    assert named["metadata"]["speaker_resolution"] == "disabled"
    assert named["metadata"]["speaker_mapping_sha256"] is None
    assert named["metadata"]["participants"] == []
    assert all("speaker" not in segment for segment in named["segments"])
    assert all("speaker_id" not in segment for segment in named["segments"])
    subtitles = (job / "transcript" / "transcript.named.srt").read_text(encoding="utf-8")
    assert "SPEAKER_" not in subtitles
    manifest = load_manifest(job)
    assert generation["speaker_resolution"] == "disabled"
    assert generation["mapping_sha256"] is None
    assert manifest["speaker_template"]["active_mapping_sha256"] is None


def test_disabled_mode_preserves_template_and_can_switch_back(tmp_path: Path) -> None:
    job = _job(tmp_path)
    map_path, _ = write_template(job)
    assert map_path is not None
    before = map_path.read_bytes()
    config = MeetingNotesConfig(summarization={"enabled": False})

    disabled = apply_speakers(job, None, config, without_diarization=True)
    mapped = apply_speakers(job, map_path, config)

    assert map_path.read_bytes() == before
    assert disabled["speaker_resolution"] == "disabled"
    assert mapped["speaker_resolution"] == "mapped"
    manifest = load_manifest(job)
    states = [item["state"] for item in manifest["speaker_publications"]["generations"]]
    assert states == ["superseded", "active"]


def test_disabled_summary_prompt_and_identity_fields_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResult:
        def __init__(self) -> None:
            self.data = {
                "participants": [{"name": "Guessed Person"}],
                "action_items": [{"task": "Follow up", "owner": "Guessed Person"}],
            }

    class FakeAdapter:
        def is_available(self) -> bool:
            return True

        def summarize(self, transcript: str, **kwargs: object) -> FakeResult:
            del transcript
            prompt = str(kwargs["prompt"])
            assert "participant roster is unavailable" in prompt
            assert "Do not infer attendees" in prompt
            return FakeResult()

    monkeypatch.setattr(
        "meeting_notes.summarization.adapters.get_adapter",
        lambda backend, **kwargs: FakeAdapter(),
    )
    config = MeetingNotesConfig(
        summarization={
            "enabled": True,
            "backend": "none",
            "prompt_path": "does-not-exist.md",
        }
    )

    summary = _summarize(
        [{"id": "seg-1", "start": 0, "end": 1, "text": "Hello"}],
        [],
        config,
        False,
        speaker_resolution="disabled",
    )

    assert summary["participants"] == []
    assert summary["action_items"][0]["owner"] is None
