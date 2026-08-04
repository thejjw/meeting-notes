"""Tests for project-local first-party storage and legacy migration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from meeting_notes import artifacts
from meeting_notes.config import MeetingNotesConfig, SetupConfig, load_config, save_config
from meeting_notes.models import model_path
from meeting_notes.runtime import runtime_dir
from meeting_notes.storage import (
    StorageMigrationError,
    cache_inventory,
    migrate_legacy_cache,
    project_cache_root,
)


def _config(tmp_path: Path) -> tuple[Path, MeetingNotesConfig]:
    path = tmp_path / "config.yaml"
    config = MeetingNotesConfig(setup=SetupConfig(completed=True))
    config.project.cache_dir = str(tmp_path / "project-cache")
    config.runtime.asr_backend = "lemonade"
    config.runtime.device = "npu"
    save_config(config, path)
    return path, config


def _legacy_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str, bytes, Path]:
    legacy = tmp_path / "legacy-cache"
    monkeypatch.setattr("meeting_notes.storage.legacy_user_cache_root", lambda: legacy)
    name = "fixture-model"
    payload = b"verified whisper weights"
    monkeypatch.setitem(
        artifacts.MODEL_ARTIFACTS,
        name,
        {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
    )
    model = legacy / "models" / f"ggml-{name}.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(payload)
    runtime = legacy / "runtimes" / "v1.9.1" / "windows-x86_64-cpu"
    executable = runtime / "Release" / "whisper-cli.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"executable")
    (runtime / "manifest.json").write_text(
        json.dumps(
            {
                "version": "v1.9.1",
                "platform": "windows",
                "architecture": "x86_64",
                "backend": "cpu",
                "executable_path": str(executable),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("meeting_notes.runtime.validate_executable", lambda path: None)
    return legacy, name, payload, executable


def test_managed_paths_require_project_cache(tmp_path: Path) -> None:
    config = MeetingNotesConfig(project={"cache_dir": str(tmp_path / "cache")})
    cache = project_cache_root(config)
    assert model_path("large-v3-turbo", cache_dir=cache).parent == cache / "models"
    assert runtime_dir("v1.9.1", "cpu", cache_dir=cache).is_relative_to(cache / "runtimes")


def test_migration_moves_verified_assets_without_changing_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    legacy, name, payload, executable = _legacy_assets(tmp_path, monkeypatch)
    config.asr.model = name
    config.asr.model_path = str(legacy / "models" / f"ggml-{name}.bin")
    config.runtime.whisper_cpp_path = str(executable)
    save_config(config, config_path)

    result = migrate_legacy_cache(config, config_path)

    restored = load_config(str(config_path))
    project = tmp_path / "project-cache"
    model = project / "models" / f"ggml-{name}.bin"
    runtime = project / "runtimes" / "v1.9.1" / "windows-x86_64-cpu"
    assert model.read_bytes() == payload
    assert Path(restored.asr.model_path) == model.resolve()
    assert Path(restored.runtime.whisper_cpp_path).is_relative_to(runtime.resolve())
    assert restored.runtime.asr_backend == "lemonade"
    assert restored.runtime.device == "npu"
    assert restored.project.cache_dir == str(project.resolve())
    assert restored.asr.model_cache_dir == str((project / "models").resolve())
    assert not (legacy / "models" / f"ggml-{name}.bin").exists()
    assert not (legacy / "runtimes").exists()
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["executable_path"]).is_relative_to(runtime.resolve())
    assert result["asr_backend"] == "lemonade"


def test_migration_rolls_back_project_copy_when_config_save_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    legacy, name, _payload, _executable = _legacy_assets(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "meeting_notes.config.save_config",
        lambda config, path: (_ for _ in ()).throw(OSError("read only")),
    )

    with pytest.raises(OSError, match="read only"):
        migrate_legacy_cache(config, config_path)

    assert (legacy / "models" / f"ggml-{name}.bin").is_file()
    assert not (tmp_path / "project-cache" / "models" / f"ggml-{name}.bin").exists()
    assert (legacy / "runtimes").is_dir()


def test_migration_refuses_conflicting_project_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, config = _config(tmp_path)
    legacy, name, _payload, _executable = _legacy_assets(tmp_path, monkeypatch)
    destination = tmp_path / "project-cache" / "models" / f"ggml-{name}.bin"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"conflict")

    with pytest.raises(StorageMigrationError, match="conflicts"):
        migrate_legacy_cache(config, config_path)

    assert (legacy / "models" / f"ggml-{name}.bin").is_file()
    assert destination.read_bytes() == b"conflict"


def test_cache_inventory_separates_project_and_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, config = _config(tmp_path)
    legacy = tmp_path / "legacy"
    monkeypatch.setattr("meeting_notes.storage.legacy_user_cache_root", lambda: legacy)
    (tmp_path / "project-cache" / "models").mkdir(parents=True)
    (tmp_path / "project-cache" / "models" / "one").write_bytes(b"123")
    (legacy / "runtimes").mkdir(parents=True)
    (legacy / "runtimes" / "two").write_bytes(b"45")

    inventory = cache_inventory(config)

    assert inventory["project"]["total_bytes"] == 3  # type: ignore[index]
    assert inventory["legacy"]["total_bytes"] == 2  # type: ignore[index]
