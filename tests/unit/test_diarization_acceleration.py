"""Tests for project-local ROCm diarization runtime management."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from meeting_notes.config import MeetingNotesConfig
from meeting_notes.diarization.acceleration import (
    ROCM_PYANNOTE_VERSION,
    ROCM_TORCH_VERSION,
    RocmRuntimeError,
    _migrate_runtime_manifest,
    default_runtime_dir,
    diarization_cache_root,
    directory_size,
    model_dir,
    probe_rocm,
    runtime_environment,
    validate_runtime,
)


def _config(tmp_path: Path) -> MeetingNotesConfig:
    return MeetingNotesConfig(project={"cache_dir": str(tmp_path / "cache")})


def test_project_local_diarization_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert diarization_cache_root(config) == (tmp_path / "cache" / "diarization")
    assert model_dir(config, "pyannote/fixture") == (
        tmp_path / "cache" / "diarization" / "models" / "pyannote--fixture"
    )
    assert default_runtime_dir(config).parent == (tmp_path / "cache" / "runtimes")


def test_directory_size_counts_nested_files(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "one").write_bytes(b"123")
    (tmp_path / "nested" / "two").write_bytes(b"4567")
    assert directory_size(tmp_path) == 7


def test_runtime_environment_isolates_managed_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("PYTHONPATH", "leaked-packages")
    monkeypatch.setenv("VIRTUAL_ENV", "main-project-venv")
    monkeypatch.setenv("PATH", "host-path")

    env = runtime_environment(runtime)

    assert "PYTHONPATH" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["VIRTUAL_ENV"] == str(runtime.resolve())
    assert env["PATH"].split(os.pathsep)[0] == str(runtime.resolve() / "Scripts")


def test_worker_does_not_shadow_third_party_pyannote(tmp_path: Path) -> None:
    fake_packages = tmp_path / "packages"
    (fake_packages / "pyannote" / "audio").mkdir(parents=True)
    (fake_packages / "pyannote" / "__init__.py").write_text("", encoding="utf-8")
    (fake_packages / "pyannote" / "audio" / "__init__.py").write_text(
        "class Pipeline: pass\n", encoding="utf-8"
    )
    (fake_packages / "torch.py").write_text(
        "class cuda:\n    @staticmethod\n    def is_available(): return False\n"
        "class version:\n    hip = None\n",
        encoding="utf-8",
    )
    worker = Path(__file__).parents[2] / "src" / "meeting_notes" / "diarization" / "worker.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(fake_packages)

    result = subprocess.run(
        [sys.executable, str(worker)],
        input="{}",
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 1
    assert "cannot access an AMD HIP device" in result.stderr


def test_probe_reports_eligible_host_without_runtime(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with (
        patch("meeting_notes.diarization.acceleration.os.name", "nt"),
        patch("meeting_notes.diarization.acceleration.platform.machine", return_value="AMD64"),
        patch("meeting_notes.diarization.acceleration._host_hip") as host,
    ):
        host.return_value = (Path("hipInfo.exe"), "gfx1151", "AMD Radeon 8060S")
        probe = probe_rocm(config)
    assert probe.state == "eligible"
    assert probe.architecture == "gfx1151"


def test_validate_runtime_requires_expected_torch(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    python = runtime / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    payload = {
        "available": True,
        "torch": "wrong",
        "hip": "7.2",
        "pyannote": ROCM_PYANNOTE_VERSION,
        "device": "AMD",
    }
    result = type(
        "Result",
        (),
        {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""},
    )()
    with (
        patch("meeting_notes.diarization.acceleration.subprocess.run", return_value=result),
        pytest.raises(RocmRuntimeError, match="Expected torch"),
    ):
        validate_runtime(runtime)

    payload["torch"] = ROCM_TORCH_VERSION
    result.stdout = json.dumps(payload)
    with patch("meeting_notes.diarization.acceleration.subprocess.run", return_value=result):
        assert validate_runtime(runtime)["device"] == "AMD"


def test_qwen_runtime_profile_is_migrated_without_reinstall(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    manifest = runtime / ".meeting-notes-runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "profiles": {
                    "diarization": {"pyannote_audio": ROCM_PYANNOTE_VERSION},
                    "qwen3_asr": {"transformers": "5.14.1", "soynlp": "0.0.493"},
                },
            }
        ),
        encoding="utf-8",
    )

    _migrate_runtime_manifest(
        runtime,
        {"transformers": "5.14.1", "soynlp": "0.0.493"},
    )

    migrated = json.loads(manifest.read_text(encoding="utf-8"))
    assert migrated["version"] == 3
    assert "qwen3_asr" not in migrated["profiles"]
    assert migrated["profiles"]["qwen3_alignment"]["transformers"] == "5.14.1"
