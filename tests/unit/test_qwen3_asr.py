"""Tests for Lemonade Qwen3-ASR and native ROCm forced alignment."""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING, Any

import pytest

from meeting_notes.asr.base import ASRReadiness, ASRResult
from meeting_notes.asr.qwen3_lemonade import (
    QWEN_GGUF_CHECKPOINT,
    Qwen3ASRLemonadeBackend,
    _segments_from_words,
    managed_aligner_dir,
)
from meeting_notes.asr.registry import get_configured_backend, list_backends
from meeting_notes.asr.setup import _remove_legacy_native_weights
from meeting_notes.audio.chunk import AudioChunk
from meeting_notes.config import MeetingNotesConfig, load_config
from meeting_notes.errors import ConfigValidationError
from meeting_notes.pipeline import _merge_asr_chunks

if TYPE_CHECKING:
    from pathlib import Path


def _wav(path: Path, seconds: float = 1.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * int(16_000 * seconds))
    return path


def _backend(tmp_path: Path) -> Qwen3ASRLemonadeBackend:
    python = tmp_path / "runtime" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    aligner = managed_aligner_dir(tmp_path / "models")
    aligner.mkdir(parents=True)
    (aligner / "config.json").write_text("{}", encoding="utf-8")
    return Qwen3ASRLemonadeBackend(
        python_executable=str(python),
        aligner_path=aligner,
    )


def test_qwen_lemonade_backend_is_registered_without_native_alias() -> None:
    assert "qwen3_asr_lemonade" in list_backends()
    assert "qwen3_asr" not in list_backends()


def test_only_aligner_is_managed_in_project_cache(tmp_path: Path) -> None:
    path = managed_aligner_dir(tmp_path / "models")
    assert path == (tmp_path / "models" / "qwen3-asr" / "Qwen--Qwen3-ForcedAligner-0.6B-hf")


def test_aligned_words_become_readable_segments() -> None:
    segments = _segments_from_words(
        [
            {"text": "first", "start": 0.1, "end": 0.3},
            {"text": "sentence.", "start": 0.3, "end": 0.7},
            {"text": "next", "start": 1.8, "end": 2.1},
            {"text": "sentence", "start": 2.1, "end": 2.5},
        ],
        "en",
    )
    assert [segment.text for segment in segments] == ["first sentence.", "next sentence"]
    assert [(segment.start, segment.end) for segment in segments] == [(0.1, 0.7), (1.8, 2.5)]
    assert segments[0].source["backend"] == "qwen3_asr_lemonade"


def test_aligned_words_are_owned_once_at_chunk_boundaries() -> None:
    chunks = [
        AudioChunk("chunk-0000", 0, 12, overlap_after=2),
        AudioChunk("chunk-0001", 8, 20, overlap_before=2),
    ]
    first = ASRResult(
        segments=_segments_from_words(
            [
                {"text": "before", "start": 8.0, "end": 9.0},
                {"text": "first-overlap", "start": 10.0, "end": 11.0},
            ],
            "en",
        )
    )
    second = ASRResult(
        segments=_segments_from_words(
            [
                {"text": "duplicate", "start": 0.0, "end": 1.0},
                {"text": "after", "start": 3.0, "end": 4.0},
            ],
            "en",
        )
    )

    merged = _merge_asr_chunks(list(zip(chunks, [first, second], strict=True)))

    assert [(segment.start, segment.end, segment.text) for segment in merged.segments] == [
        (8.0, 9.0, "before"),
        (11.0, 12.0, "after"),
    ]


def test_non_monotonic_alignment_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="non-monotonic"):
        _segments_from_words(
            [
                {"text": "one", "start": 1.0, "end": 2.0},
                {"text": "two", "start": 1.0, "end": 1.5},
            ],
            "en",
        )


def test_readiness_requires_project_alignment_runtime(tmp_path: Path) -> None:
    backend = Qwen3ASRLemonadeBackend(
        python_executable=str(tmp_path / "missing-python.exe"),
        aligner_path=tmp_path / "aligner",
    )
    readiness = backend.check_readiness()
    assert not readiness.available
    assert "alignment runtime is missing" in readiness.detail


def test_model_discovery_prefers_exact_1_7b_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    backend.model_id = "wrong-0.6b-display-name"
    monkeypatch.setattr(
        backend,
        "list_models",
        lambda **_kwargs: [
            {
                "id": "wrong-0.6b-display-name",
                "checkpoint": "unslothai/Qwen3-ASR-0.6B-GGUF:Q8_0",
            },
            {"id": "resolved-1.7b", "checkpoint": QWEN_GGUF_CHECKPOINT},
        ],
    )

    info = backend.model_info()

    assert info is not None
    assert info["id"] == "resolved-1.7b"


def test_readiness_rejects_non_vulkan_lemonade_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    monkeypatch.setattr(backend, "is_available", lambda: True)
    monkeypatch.setattr(
        backend,
        "model_info",
        lambda: {"id": backend.model_id, "checkpoint": backend.checkpoint, "downloaded": True},
    )
    monkeypatch.setattr(
        backend,
        "_loaded_model",
        lambda: (
            {
                "model_name": backend.model_id,
                "device": "cpu",
                "status": "ready",
                "backend_alive": True,
                "recipe_options": {"llamacpp_backend": "rocm"},
            },
            {"version": "11.5.1"},
        ),
    )
    readiness = backend.check_readiness()
    assert not readiness.available
    assert "not with llama.cpp vulkan" in readiness.detail


def test_registry_resolves_gpu_only_qwen_backend(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    config = MeetingNotesConfig(
        project={"cache_dir": str(tmp_path / "cache")},
        runtime={"asr_backend": "qwen3_asr_lemonade", "device": "rocm"},
        asr={"backend_options": {"qwen3_asr_lemonade": {"rocm_gpu_runtime_path": str(runtime)}}},
    )
    configured = get_configured_backend(config)
    assert configured.backend.name == "qwen3_asr_lemonade"
    assert configured.runtime_identity["checkpoint"] == QWEN_GGUF_CHECKPOINT
    assert configured.runtime_identity["llamacpp_backend"] == "vulkan"


def test_config_rejects_lemonade_rocm_for_qwen() -> None:
    with pytest.raises(ValueError, match="llamacpp_backend"):
        MeetingNotesConfig(
            asr={
                "backend_options": {
                    "qwen3_asr_lemonade": {"llamacpp_backend": "rocm"}
                }
            }
        )


def test_registry_rejects_cpu_for_lemonade_qwen() -> None:
    config = MeetingNotesConfig(runtime={"asr_backend": "qwen3_asr_lemonade", "device": "cpu"})
    with pytest.raises(ValueError, match="GPU-only"):
        get_configured_backend(config)


def test_chat_audio_transcript_is_forced_to_korean_and_aligned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    wav = _wav(tmp_path / "sample.wav")
    requests: list[dict[str, Any]] = []
    monkeypatch.setattr(
        backend,
        "load_model",
        lambda: ASRReadiness(True, "ready", version="11.5.1", device="rocm"),
    )

    def post(path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        del timeout
        assert path == "/v1/chat/completions"
        requests.append(payload)
        return {
            "choices": [{"message": {"content": "language Korean<asr_text>테스트 문장."}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }

    monkeypatch.setattr(backend, "_post_json", post)
    monkeypatch.setattr(
        backend,
        "_align",
        lambda *_args: {
            "results": [
                {
                    "words": [{"text": "테스트 문장.", "start": 0.1, "end": 0.8}],
                    "alignment_seconds": 0.2,
                }
            ],
            "metrics": {"aligner_load_seconds": 0.1},
        },
    )

    result = backend.transcribe(wav, language="ko", initial_prompt="product name")

    assert result.backend == "qwen3_asr_lemonade"
    assert result.device == "vulkan+rocm"
    assert result.segments[0].text == "테스트 문장."
    messages = requests[0]["messages"]
    assert messages[-1] == {"role": "assistant", "content": "language Korean<asr_text>"}
    assert requests[0]["continue_final_message"] is True
    assert requests[0]["add_generation_prompt"] is False
    assert "product name" in messages[0]["content"][1]["text"]


def test_auto_language_omits_prefill_and_detects_per_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    wav = _wav(tmp_path / "auto.wav")
    requests: list[dict[str, Any]] = []

    def post(_path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        del timeout
        requests.append(payload)
        return {"choices": [{"message": {"content": "language English<asr_text>hello world"}}]}

    monkeypatch.setattr(backend, "_post_json", post)
    transcript, language, _response, _elapsed = backend._transcribe_text(
        wav, language_code="auto", initial_prompt=None
    )

    assert (transcript, language) == ("hello world", "en")
    assert len(requests[0]["messages"]) == 1
    assert "continue_final_message" not in requests[0]
    assert "add_generation_prompt" not in requests[0]


def test_explicit_language_accepts_code_or_full_name() -> None:
    assert Qwen3ASRLemonadeBackend._normalize_language("ja") == ("ja", "Japanese")
    assert Qwen3ASRLemonadeBackend._normalize_language(" Japanese ") == (
        "ja",
        "Japanese",
    )


def test_explicit_language_rejects_asr_language_without_alignment() -> None:
    with pytest.raises(ValueError, match=r"can transcribe Arabic.*cannot timestamp"):
        Qwen3ASRLemonadeBackend._normalize_language("ar")


def test_auto_language_reports_unsupported_detected_alignment_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(tmp_path)
    wav = _wav(tmp_path / "arabic.wav")
    monkeypatch.setattr(
        backend,
        "_post_json",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": "language Arabic<asr_text>مرحبا بالعالم"}}]
        },
    )

    with pytest.raises(RuntimeError, match=r"detected Arabic.*cannot timestamp"):
        backend._transcribe_text(wav, language_code="auto", initial_prompt=None)


def test_old_native_backend_config_has_migration_guidance(tmp_path: Path) -> None:
    config = tmp_path / "meeting-notes.yaml"
    config.write_text(
        "setup:\n  completed: true\nruntime:\n  asr_backend: qwen3_asr\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError, match="qwen3_asr_lemonade"):
        load_config(str(config))


def test_cleanup_removes_only_obsolete_native_weights(tmp_path: Path) -> None:
    model_cache = tmp_path / "models"
    legacy = model_cache / "qwen3-asr" / "Qwen--Qwen3-ASR-1.7B-hf"
    aligner = managed_aligner_dir(model_cache)
    legacy.mkdir(parents=True)
    aligner.mkdir(parents=True)
    (legacy / "weights.bin").write_bytes(b"legacy")
    (aligner / "weights.bin").write_bytes(b"aligner")

    reclaimed = _remove_legacy_native_weights(model_cache)

    assert reclaimed == len(b"legacy")
    assert not legacy.exists()
    assert aligner.exists()
