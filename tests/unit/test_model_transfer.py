"""Tests for portable managed-model archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
from typing import TYPE_CHECKING

import pytest

from meeting_notes.config import MeetingNotesConfig, SetupConfig, load_config, save_config
from meeting_notes.diarization.setup import run_diarization_setup
from meeting_notes.model_transfer import (
    MANIFEST_NAME,
    ModelTransferError,
    backup_diarization,
    backup_whisper,
    restore_archive,
)

if TYPE_CHECKING:
    from pathlib import Path


def _config(path: Path) -> Path:
    config = MeetingNotesConfig(setup=SetupConfig(completed=True))
    save_config(config, path)
    return path


def _artifact(monkeypatch: pytest.MonkeyPatch, name: str, payload: bytes) -> None:
    from meeting_notes import artifacts, model_transfer

    metadata = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    monkeypatch.setitem(artifacts.MODEL_ARTIFACTS, name, metadata)
    monkeypatch.setitem(model_transfer.MODEL_ARTIFACTS, name, metadata)


def test_whisper_backup_and_restore_updates_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "fixture-model"
    payload = b"portable whisper weights"
    _artifact(monkeypatch, name, payload)
    source_cache = tmp_path / "source-cache"
    source = source_cache / "models" / f"ggml-{name}.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    config_path = _config(tmp_path / "config.yaml")
    config = load_config(str(config_path))
    config.asr.model = name
    config.asr.model_path = str(source)
    save_config(config, config_path)
    monkeypatch.setattr("meeting_notes.model_transfer.model_path", lambda value: source)

    archive, sidecar = backup_whisper(
        config_path=str(config_path),
        archive_path=tmp_path / "whisper.zip",
        compression_level="none",
    )

    assert archive.is_file()
    assert sidecar.is_file()
    with zipfile.ZipFile(archive) as value:
        manifest = json.loads(value.read(MANIFEST_NAME))
        assert manifest["kind"] == "whisper"
        assert manifest["model"]["name"] == name
        assert all("source-cache" not in item for item in value.namelist())

    destination_cache = tmp_path / "destination-cache"
    monkeypatch.setattr(
        "meeting_notes.model_transfer.cache_root", lambda: destination_cache
    )
    destination = restore_archive("whisper", archive, config_path=str(config_path))

    assert destination.read_bytes() == payload
    restored = load_config(str(config_path))
    assert restored.asr.model == name
    assert restored.asr.model_path == str(destination)


def test_whisper_restore_refuses_collision_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "fixture-model"
    payload = b"new weights"
    _artifact(monkeypatch, name, payload)
    source = tmp_path / "source" / f"ggml-{name}.bin"
    source.parent.mkdir()
    source.write_bytes(payload)
    config_path = _config(tmp_path / "config.yaml")
    config = load_config(str(config_path))
    config.asr.model = name
    save_config(config, config_path)
    monkeypatch.setattr("meeting_notes.model_transfer.model_path", lambda value: source)
    archive, _ = backup_whisper(
        config_path=str(config_path), archive_path=tmp_path / "model.zip"
    )
    destination_cache = tmp_path / "cache"
    destination = destination_cache / "models" / f"ggml-{name}.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        "meeting_notes.model_transfer.cache_root", lambda: destination_cache
    )

    with pytest.raises(ModelTransferError, match="Use -Force"):
        restore_archive("whisper", archive, config_path=str(config_path))
    assert destination.read_bytes() == b"old"

    restore_archive("whisper", archive, config_path=str(config_path), force=True)
    assert destination.read_bytes() == payload


def test_diarization_backup_excludes_hugging_face_metadata_and_restores_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source-diarization"
    (source / "embedding").mkdir(parents=True)
    (source / ".cache" / "huggingface").mkdir(parents=True)
    (source / "config.yaml").write_text("pipeline: fixture\n", encoding="utf-8")
    (source / "embedding" / "model.bin").write_bytes(b"embedding")
    (source / ".cache" / "huggingface" / "token-metadata").write_text(
        "not portable", encoding="utf-8"
    )
    (source / ".meeting-notes-manifest.json").write_text(
        json.dumps(
            {
                "repo_id": "pyannote/fixture",
                "revision": "abc123",
                "authentication": "saved Hugging Face login",
            }
        ),
        encoding="utf-8",
    )
    config_path = _config(tmp_path / "config.yaml")
    config = load_config(str(config_path))
    config.diarization.model = "pyannote/fixture"
    config.diarization.model_path = str(source)
    save_config(config, config_path)

    archive, _ = backup_diarization(
        config_path=str(config_path), archive_path=tmp_path / "diarization.zip"
    )

    with zipfile.ZipFile(archive) as value:
        names = value.namelist()
        assert "payload/config.yaml" in names
        assert "payload/embedding/model.bin" in names
        assert not any(".cache" in name for name in names)
        assert not any(".meeting-notes-manifest" in name for name in names)
        assert b"saved Hugging Face login" not in value.read(MANIFEST_NAME)

    destination = tmp_path / "restored-diarization"
    monkeypatch.setattr(
        "meeting_notes.model_transfer.managed_diarization_model_dir",
        lambda config, repo_id: destination,
    )
    restored_path = restore_archive(
        "diarization", archive, config_path=str(config_path)
    )

    assert restored_path == destination.resolve()
    assert (destination / "embedding" / "model.bin").read_bytes() == b"embedding"
    install = json.loads(
        (destination / ".meeting-notes-manifest.json").read_text(encoding="utf-8")
    )
    assert install["authentication"] == "restored offline archive"
    restored = load_config(str(config_path))
    assert restored.diarization.enabled is True
    assert restored.diarization.model_path == str(destination.resolve())


def test_diarization_setup_accepts_offline_model_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source-model"
    source.mkdir()
    (source / "config.yaml").write_text("pipeline: fixture\n", encoding="utf-8")
    (source / "weights.bin").write_bytes(b"weights")
    config_path = _config(tmp_path / "config.yaml")
    config = load_config(str(config_path))
    config.project.cache_dir = str(tmp_path / "project-cache")
    config.diarization.model = "pyannote/fixture"
    config.diarization.model_path = str(source)
    save_config(config, config_path)
    archive, _ = backup_diarization(
        config_path=str(config_path), archive_path=tmp_path / "backup.zip"
    )
    config.diarization.model_path = None
    save_config(config, config_path)

    monkeypatch.setattr("meeting_notes.diarization.setup.version", lambda name: "4.0.7")
    run_diarization_setup(
        config_path=str(config_path), model_archive=archive, acceleration="cpu", yes=True
    )

    restored = load_config(str(config_path))
    expected = (
        tmp_path / "project-cache" / "diarization" / "models" / "pyannote--fixture"
    ).resolve()
    assert restored.diarization.model_path == str(expected)
    assert restored.diarization.device == "cpu"
    assert (expected / "weights.bin").read_bytes() == b"weights"


def test_restore_rejects_sidecar_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "fixture-model"
    payload = b"weights"
    _artifact(monkeypatch, name, payload)
    source = tmp_path / f"ggml-{name}.bin"
    source.write_bytes(payload)
    config_path = _config(tmp_path / "config.yaml")
    config = load_config(str(config_path))
    config.asr.model = name
    save_config(config, config_path)
    monkeypatch.setattr("meeting_notes.model_transfer.model_path", lambda value: source)
    archive, sidecar = backup_whisper(
        config_path=str(config_path), archive_path=tmp_path / "model.zip"
    )
    sidecar.write_text(f"{'0' * 64}  {archive.name}\n", encoding="ascii")

    with pytest.raises(ModelTransferError, match="Archive checksum mismatch"):
        restore_archive("whisper", archive, config_path=str(config_path))


def test_restore_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    manifest = {
        "version": 1,
        "kind": "diarization",
        "model": {"repo_id": "pyannote/fixture"},
        "files": [{"path": "payload/../escape", "size": 1, "sha256": "0" * 64}],
    }
    with zipfile.ZipFile(archive, "w") as value:
        value.writestr(MANIFEST_NAME, json.dumps(manifest))
        value.writestr("payload/../escape", b"x")

    with pytest.raises(ModelTransferError, match="Unsafe archive member"):
        restore_archive(
            "diarization", archive, config_path=str(_config(tmp_path / "config.yaml"))
        )


def test_forced_restore_rolls_back_model_when_config_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "fixture-model"
    payload = b"new weights"
    _artifact(monkeypatch, name, payload)
    source = tmp_path / f"ggml-{name}.bin"
    source.write_bytes(payload)
    config_path = _config(tmp_path / "config.yaml")
    config = load_config(str(config_path))
    config.asr.model = name
    save_config(config, config_path)
    monkeypatch.setattr("meeting_notes.model_transfer.model_path", lambda value: source)
    archive, _ = backup_whisper(
        config_path=str(config_path), archive_path=tmp_path / "model.zip"
    )
    destination_cache = tmp_path / "cache"
    destination = destination_cache / "models" / f"ggml-{name}.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old weights")
    monkeypatch.setattr(
        "meeting_notes.model_transfer.cache_root", lambda: destination_cache
    )
    monkeypatch.setattr(
        "meeting_notes.model_transfer.save_config",
        lambda config, path: (_ for _ in ()).throw(OSError("read only")),
    )

    with pytest.raises(OSError, match="read only"):
        restore_archive(
            "whisper", archive, config_path=str(config_path), force=True
        )

    assert destination.read_bytes() == b"old weights"
