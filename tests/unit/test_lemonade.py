"""Tests for the AMD Lemonade ASR adapter and configuration integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest

from meeting_notes.asr.base import ASRReadiness
from meeting_notes.asr.lemonade import LemonadeASRBackend
from meeting_notes.asr.registry import get_configured_backend, list_backends
from meeting_notes.config import MeetingNotesConfig
from meeting_notes.configure import _build_backend_options, _provision_lemonade_model
from meeting_notes.resources import SystemDiagnostics
from meeting_notes.timing import build_time_estimate_lines

if TYPE_CHECKING:
    from pathlib import Path


def _response(payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "http://127.0.0.1:13305/test"),
    )


def _config() -> MeetingNotesConfig:
    return MeetingNotesConfig(
        setup={"completed": True, "profile": "amd-lemonade"},
        runtime={"asr_backend": "lemonade", "device": "npu"},
        asr={
            "model": "large-v3-turbo",
            "model_path": None,
            "backend_options": {
                "lemonade": {
                    "base_url": "http://127.0.0.1:13305",
                    "model_id": "Whisper-Large-v3-Turbo",
                }
            },
        },
    )


def test_lemonade_is_registered_and_cpu_default_is_unchanged() -> None:
    assert "lemonade" in list_backends()
    defaults = MeetingNotesConfig()
    assert defaults.runtime.asr_backend == "whisper_cpp"
    assert defaults.runtime.device == "cpu"
    assert defaults.asr.backend_options.lemonade.base_url == "http://127.0.0.1:13305"


def test_get_configured_lemonade_backend() -> None:
    configured = get_configured_backend(_config())
    assert isinstance(configured.backend, LemonadeASRBackend)
    assert configured.runtime_identity["device"] == "npu"
    assert configured.runtime_identity["lemonade_model_id"] == "Whisper-Large-v3-Turbo"
    assert configured.transcribe_kwargs["model_path"] is None


def test_unreachable_server_has_manual_start_remediation() -> None:
    backend = LemonadeASRBackend()
    with patch.object(backend, "is_available", return_value=False):
        readiness = backend.check_readiness(expected_device="npu")
    assert not readiness.available
    assert "Start Lemonade Server manually" in readiness.detail
    assert "http://127.0.0.1:13305" in readiness.detail


def test_readiness_rejects_wrong_device() -> None:
    backend = LemonadeASRBackend()
    with (
        patch.object(backend, "is_available", return_value=True),
        patch.object(
            backend,
            "model_info",
            return_value={
                "id": backend.model_id,
                "downloaded": True,
                "size": 1.51,
                "labels": ["transcription"],
            },
        ),
        patch.object(
            backend,
            "_loaded_model",
            return_value=(
                {
                    "model_name": backend.model_id,
                    "device": "cpu",
                    "status": "ready",
                    "backend_alive": True,
                },
                {"version": "11.5.0"},
            ),
        ),
    ):
        readiness = backend.check_readiness(expected_device="npu")
    assert not readiness.available
    assert "expected ready on npu" in readiness.detail


def test_verbose_json_maps_to_timestamped_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")
    backend = LemonadeASRBackend()
    monkeypatch.setattr(
        backend,
        "load_model",
        lambda: ASRReadiness(
            True,
            "ready",
            version="11.5.0",
            device="npu",
        ),
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: _response(
            {
                "detected_language": "ko",
                "duration": 12.5,
                "task": "transcribe",
                "text": "안녕하세요",
                "segments": [
                    {
                        "id": 0,
                        "start": 1.25,
                        "end": 3.5,
                        "text": " 안녕하세요 ",
                        "avg_logprob": -0.2,
                        "no_speech_prob": 0.1,
                    }
                ],
            }
        ),
    )
    result = backend.transcribe(wav, model="large-v3-turbo", language="ko")
    assert result.backend == "lemonade"
    assert result.device == "npu"
    assert result.duration == 12.5
    assert result.segments[0].start == 1.25
    assert result.segments[0].end == 3.5
    assert result.segments[0].text == "안녕하세요"
    assert result.segments[0].source["lemonade_model_id"] == backend.model_id


def test_lemonade_rejects_unsupported_initial_prompt(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="initial prompt"):
        LemonadeASRBackend().transcribe(wav, initial_prompt="terms")


def test_provision_requires_running_server() -> None:
    config = _config()
    with (
        patch.object(LemonadeASRBackend, "is_available", return_value=False),
        pytest.raises(RuntimeError, match="Start Lemonade Server manually"),
    ):
        _provision_lemonade_model(config, yes=True)


def test_provision_pulls_and_loads_registered_model() -> None:
    config = _config()
    ready = ASRReadiness(True, "ready", version="11.5.0", device="npu")
    with (
        patch.object(LemonadeASRBackend, "is_available", return_value=True),
        patch.object(
            LemonadeASRBackend,
            "model_info",
            return_value={"id": "Whisper-Large-v3-Turbo", "downloaded": False, "size": 1.51},
        ),
        patch.object(LemonadeASRBackend, "pull_model") as pull,
        patch.object(LemonadeASRBackend, "load_model", return_value=ready) as load,
    ):
        assert _provision_lemonade_model(config, yes=True) == "Whisper-Large-v3-Turbo"
    pull.assert_called_once()
    load.assert_called_once()


def test_wizard_backend_options_include_default_lemonade_url() -> None:
    with patch.object(LemonadeASRBackend, "is_available", return_value=False):
        options = _build_backend_options(SystemDiagnostics())
    lemonade = next(item for item in options if item["runtime_asr_backend"] == "lemonade")
    assert "http://127.0.0.1:13305" in lemonade["notes"]


def test_lemonade_first_run_estimate_uses_npu_seed(tmp_path: Path) -> None:
    lines = build_time_estimate_lines(
        _config(),
        ["transcribe"],
        300.0,
        data_dir=tmp_path,
    )
    assert any("33s" in line and "generic estimate" in line for line in lines)


def test_cpu_history_does_not_leak_into_lemonade_estimate(tmp_path: Path) -> None:
    job = tmp_path / "meetings" / "cpu-job"
    job.mkdir(parents=True)
    (job / "manifest.json").write_text(
        json.dumps(
            {
                "source": {"duration_seconds": 300.0},
                "stages": {
                    "transcribe": {
                        "status": "completed",
                        "started_at": "2026-07-01T00:00:00+00:00",
                        "ended_at": "2026-07-01T00:06:42+00:00",
                        "runtime": {
                            "backend": "whisper_cpp",
                            "device": "cpu",
                            "model": "large-v3-turbo",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    lines = build_time_estimate_lines(
        _config(),
        ["transcribe"],
        300.0,
        data_dir=tmp_path,
    )
    assert any("33s" in line and "generic estimate" in line for line in lines)
    assert not any("rough guess" in line for line in lines)
