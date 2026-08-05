"""Tests for the AMD Lemonade ASR adapter and configuration integration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest

from meeting_notes.asr.base import ASRReadiness, ASRResult, ASRSegment
from meeting_notes.asr.lemonade import LemonadeASRBackend
from meeting_notes.asr.registry import get_configured_backend, list_backends
from meeting_notes.audio.chunk import AudioChunk
from meeting_notes.config import MeetingNotesConfig
from meeting_notes.configure import (
    _build_backend_options,
    _provision_lemonade_model,
    _provision_lemonade_summarizer,
)
from meeting_notes.pipeline import _merge_asr_chunks, _transcription_chunks
from meeting_notes.resources import SystemDiagnostics
from meeting_notes.summarization.adapters import LemonadeAdapter
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
    assert defaults.asr.backend_options.lemonade.max_upload_mib == 100.0


def test_get_configured_lemonade_backend() -> None:
    configured = get_configured_backend(_config())
    assert isinstance(configured.backend, LemonadeASRBackend)
    assert configured.runtime_identity["device"] == "npu"
    assert configured.runtime_identity["lemonade_model_id"] == "Whisper-Large-v3-Turbo"
    assert configured.runtime_identity["whispercpp_backend"] == "npu"
    assert configured.transcribe_kwargs["model_path"] is None


def test_get_configured_lemonade_vulkan_backend() -> None:
    config = _config()
    config.runtime.device = "vulkan"
    configured = get_configured_backend(config)
    assert isinstance(configured.backend, LemonadeASRBackend)
    assert configured.backend.expected_device == "vulkan"
    assert configured.runtime_identity["whispercpp_backend"] == "vulkan"


def test_lemonade_rejects_rocm_and_preserves_local_cpu_baseline() -> None:
    config = _config()
    config.runtime.device = "rocm"
    with pytest.raises(ValueError, match="npu, vulkan"):
        get_configured_backend(config)

    defaults = MeetingNotesConfig()
    assert defaults.runtime.asr_backend == "whisper_cpp"
    assert defaults.runtime.device == "cpu"


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
                    "recipe_options": {"whispercpp_backend": "npu"},
                },
                {"version": "11.5.0"},
            ),
        ),
    ):
        readiness = backend.check_readiness(expected_device="npu")
    assert not readiness.available
    assert "expected ready with whisper.cpp npu" in readiness.detail


def test_vulkan_readiness_uses_exact_recipe_backend_not_generic_gpu() -> None:
    backend = LemonadeASRBackend(expected_device="vulkan")
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
                    "device": "gpu",
                    "status": "ready",
                    "backend_alive": True,
                    "recipe_options": {"whispercpp_backend": "vulkan"},
                },
                {"version": "11.5.1"},
            ),
        ),
    ):
        readiness = backend.check_readiness()
    assert readiness.available
    assert readiness.device == "vulkan"
    assert readiness.metadata["reported_device"] == "gpu"
    assert readiness.metadata["whispercpp_backend"] == "vulkan"


def test_load_model_requests_selected_whispercpp_backend() -> None:
    backend = LemonadeASRBackend(expected_device="vulkan")
    not_loaded = ASRReadiness(
        True,
        "downloaded",
        device="vulkan",
        metadata={"loaded": False},
    )
    ready = ASRReadiness(
        True,
        "ready",
        device="vulkan",
        metadata={"loaded": True, "whispercpp_backend": "vulkan"},
    )
    with (
        patch.object(backend, "model_info", return_value={"downloaded": True}),
        patch.object(backend, "check_readiness", side_effect=[not_loaded, ready]),
        patch.object(backend, "_post_json", return_value={}) as post,
    ):
        assert backend.load_model() == ready
    assert post.call_args.args[1]["whispercpp_backend"] == "vulkan"


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


def test_lemonade_413_has_chunking_remediation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF")
    backend = LemonadeASRBackend()
    monkeypatch.setattr(
        backend,
        "load_model",
        lambda: ASRReadiness(True, "ready", device="npu"),
    )
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            413,
            request=httpx.Request("POST", "http://127.0.0.1:13305/test"),
        ),
    )
    with pytest.raises(RuntimeError, match="split and merge"):
        backend.transcribe(wav)


def test_large_lemonade_wav_is_chunked_even_when_chunking_is_disabled(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "normalized.wav"
    normalized.write_bytes(b"\0" * (2 * 1024 * 1024))
    config = _config()
    config.audio.chunking.mode = "none"
    config.audio.chunking.max_chunk_minutes = 20
    config.asr.backend_options.lemonade.max_upload_mib = 1.5

    chunks = _transcription_chunks(
        normalized,
        {"source": {"duration_seconds": 1000.0}},
        config,
    )

    assert len(chunks) > 1
    assert chunks[0].source_start == 0
    assert chunks[-1].source_end == 1000
    bytes_per_second = normalized.stat().st_size / 1000.0
    safe_bytes = 1.5 * 1024 * 1024 * 0.9
    assert all(chunk.duration * bytes_per_second <= safe_bytes for chunk in chunks)


def test_lemonade_wav_below_safe_upload_budget_is_not_chunked(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "normalized.wav"
    normalized.write_bytes(b"\0" * (1024 * 1024))
    config = _config()
    config.audio.chunking.mode = "none"
    config.asr.backend_options.lemonade.max_upload_mib = 1.5

    chunks = _transcription_chunks(
        normalized,
        {"source": {"duration_seconds": 1000.0}},
        config,
    )

    assert len(chunks) == 1
    assert chunks[0].path == str(normalized)


def test_chunk_results_merge_to_absolute_timestamps_and_remove_overlap() -> None:
    chunks = [
        AudioChunk(
            "chunk-0000",
            source_start=0,
            source_end=12,
            overlap_after=2,
        ),
        AudioChunk(
            "chunk-0001",
            source_start=8,
            source_end=20,
            overlap_before=2,
        ),
    ]
    first = ASRResult(
        segments=[
            ASRSegment("local-0", 8, 9, "before boundary"),
            ASRSegment("local-1", 10, 11, "discarded overlap"),
        ],
        language="ko",
        backend="lemonade",
        model="large-v3-turbo",
        device="npu",
    )
    second = ASRResult(
        segments=[
            ASRSegment("local-0", 0, 1, "duplicate overlap"),
            ASRSegment("local-1", 3, 4, "after boundary"),
            ASRSegment("local-2", 11.5, 12.5, "clamped ending"),
        ],
        language="ko",
        backend="lemonade",
        model="large-v3-turbo",
        device="npu",
    )

    merged = _merge_asr_chunks(list(zip(chunks, [first, second], strict=True)))

    assert [(item.id, item.start, item.text) for item in merged.segments] == [
        ("seg-000000", 8, "before boundary"),
        ("seg-000001", 11, "after boundary"),
        ("seg-000002", 19.5, "clamped ending"),
    ]
    assert merged.segments[-1].end == 20
    assert merged.duration == 20
    assert merged.raw_output["chunk_count"] == 2


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


def test_summarizer_provision_requires_running_server() -> None:
    config = MeetingNotesConfig(summarization={"enabled": True, "backend": "lemonade"})
    with (
        patch.object(LemonadeAdapter, "is_available", return_value=False),
        pytest.raises(RuntimeError, match="Start Lemonade Server manually"),
    ):
        _provision_lemonade_summarizer(config, yes=True)


def test_summarizer_provision_pulls_and_loads_model() -> None:
    config = MeetingNotesConfig(summarization={"enabled": True, "backend": "lemonade"})
    with (
        patch.object(LemonadeAdapter, "is_available", return_value=True),
        patch.object(
            LemonadeAdapter,
            "model_info",
            return_value={
                "id": "Gemma-4-26B-A4B-it-MTP-GGUF",
                "downloaded": False,
                "size": 17.3,
            },
        ),
        patch.object(LemonadeAdapter, "pull_model") as pull,
        patch.object(LemonadeAdapter, "ensure_model_ready") as ready,
    ):
        assert _provision_lemonade_summarizer(config, yes=True) == "Gemma-4-26B-A4B-it-MTP-GGUF"
    pull.assert_called_once()
    ready.assert_called_once()


def test_wizard_backend_options_include_default_lemonade_url() -> None:
    with patch.object(LemonadeASRBackend, "is_available", return_value=False):
        options = _build_backend_options(SystemDiagnostics())
    lemonade = next(item for item in options if item["runtime_asr_backend"] == "lemonade")
    assert "http://127.0.0.1:13305" in lemonade["notes"]


def test_wizard_offers_lemonade_vulkan_and_npu_but_not_rocm() -> None:
    diagnostics = SystemDiagnostics()
    diagnostics.gpu.vulkan_devices = [{"name": "AMD Radeon"}]
    with patch.object(LemonadeASRBackend, "is_available", return_value=True):
        options = _build_backend_options(diagnostics)
    choices = {(item["runtime_asr_backend"], item["runtime_device"]) for item in options}
    assert ("whisper_cpp", "cpu") in choices
    assert ("lemonade", "vulkan") in choices
    assert ("lemonade", "npu") in choices
    assert not any(item["runtime_device"] == "rocm" for item in options)


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
