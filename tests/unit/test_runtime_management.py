"""Tests for managed runtime/model safety and whisper.cpp device flags."""

from __future__ import annotations

import hashlib
import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from meeting_notes.asr.whisper_cpp import WhisperCppBackend
from meeting_notes.config import MeetingNotesConfig, SetupConfig, load_config, save_config
from meeting_notes.configure import (
    _diarization_recommendations,
    _latest_transcript_metadata,
)
from meeting_notes.diarization.pyannote import PyannoteDiarizationBackend
from meeting_notes.diarization.setup import (
    resolve_hf_token,
    run_diarization_setup,
)
from meeting_notes.errors import DiarizationUnavailableError
from meeting_notes.pipeline import (
    _asr_remediation,
    _resolve_whisper_threads,
    _run_diarize,
    _run_merge,
)
from meeting_notes.resources import _detect_vulkan
from meeting_notes.runtime import (
    RuntimeInstallError,
    build_commands,
    safe_extract,
    select_cpu_asset,
    verify_checksum,
)


def test_select_windows_x64_verified_asset() -> None:
    asset = select_cpu_asset(system="Windows", machine="AMD64")
    assert asset.filename == "whisper-bin-x64.zip"
    assert len(asset.sha256) == 64


def test_select_linux_arm64_asset() -> None:
    asset = select_cpu_asset(system="Linux", machine="aarch64")
    assert asset.filename == "whisper-bin-ubuntu-arm64.tar.gz"


def test_checksum_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "asset"
    artifact.write_bytes(b"verified")
    verify_checksum(artifact, hashlib.sha256(b"verified").hexdigest())
    with pytest.raises(RuntimeInstallError, match="Checksum mismatch"):
        verify_checksum(artifact, "0" * 64)


def test_safe_zip_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "bad")
    with pytest.raises(RuntimeInstallError, match="Unsafe archive"):
        safe_extract(archive, tmp_path / "out")


def test_safe_tar_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as output:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        output.addfile(info, BytesIO())
    with pytest.raises(RuntimeInstallError, match="link is not allowed"):
        safe_extract(archive, tmp_path / "out")


def test_vulkan_build_commands_are_pinned(tmp_path: Path) -> None:
    commands = build_commands(tmp_path / "source", tmp_path / "build", tmp_path / "install")
    assert "--branch" in commands[0]
    assert "v1.9.1" in commands[0]
    assert "-DGGML_VULKAN=1" in commands[1]


def test_whisper_cpu_is_strictly_no_gpu(tmp_path: Path) -> None:
    args = WhisperCppBackend()._build_args(
        audio_path=tmp_path / "audio.wav",
        model_path=tmp_path / "model.bin",
        language="ko",
        task="transcribe",
        initial_prompt=None,
        word_timestamps=False,
        threads=4,
        extra_args=["--print-progress"],
        device="cpu",
        model_variant="fp16",
        flash_attention=False,
        gpu_device=None,
    )
    assert "--no-gpu" in args
    assert "--device" not in args
    assert args[-1] == "--print-progress"


def test_whisper_vulkan_selects_configured_gpu(tmp_path: Path) -> None:
    args = WhisperCppBackend()._build_args(
        audio_path=tmp_path / "audio.wav",
        model_path=tmp_path / "model.bin",
        language="ko",
        task="transcribe",
        initial_prompt=None,
        word_timestamps=False,
        threads=0,
        extra_args=None,
        device="vulkan",
        model_variant="fp16",
        flash_attention=True,
        gpu_device="2",
    )
    index = args.index("--device")
    assert args[index + 1] == "2"
    assert "--no-gpu" not in args


def test_whisper_parses_v1_9_json_offsets() -> None:
    payload = json.dumps(
        {
            "transcription": [
                {
                    "offsets": {"from": 1500, "to": 3250},
                    "text": " hello ",
                }
            ]
        }
    )
    segments = WhisperCppBackend()._parse_output(payload)
    assert len(segments) == 1
    assert segments[0].start == 1.5
    assert segments[0].end == 3.25
    assert segments[0].text == "hello"


def test_whisper_repairs_invalid_utf8_model_output(tmp_path: Path) -> None:
    output = tmp_path / "transcription.json"
    output.write_bytes(
        b'{"transcription":[{"text":"bad ' + bytes([0xEB]) + b' text"}]}'
    )

    decoded, replacements, first_offset = WhisperCppBackend._read_json_output(output)

    assert replacements == 1
    assert first_offset == 31
    assert "\ufffd" in decoded
    parsed = WhisperCppBackend()._parse_output(decoded, fallback_to_text=False)
    assert parsed[0].text == "bad \ufffd text"


def test_whisper_invalid_json_does_not_silently_become_empty_transcript() -> None:
    with pytest.raises(RuntimeError, match="invalid JSON"):
        WhisperCppBackend()._parse_output('{"transcription": [', fallback_to_text=False)


def test_whisper_auto_threads_respect_configured_cap() -> None:
    config = MeetingNotesConfig()
    config.runtime.threads = 0
    config.runtime.max_auto_threads = 8
    config.runtime.reserve_logical_cores = 2

    with patch("meeting_notes.pipeline.os.cpu_count", return_value=32):
        assert _resolve_whisper_threads(config) == 8

    config.runtime.threads = 12
    assert _resolve_whisper_threads(config) == 12


def test_vulkan_summary_parser_is_case_insensitive() -> None:
    summary = """
    GPU0:
        deviceName = AMD Radeon(TM) 8060S Graphics
        deviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
    """
    with patch("meeting_notes.resources._run_command", return_value=(True, summary)):
        devices = _detect_vulkan()
    assert devices == [
        {
            "name": "AMD Radeon(TM) 8060S Graphics",
            "type": "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU",
        }
    ]


def test_remediation_lists_exact_runtime_model_and_doctor_commands(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = MeetingNotesConfig(
        setup=SetupConfig(completed=True),
        runtime={"device": "vulkan", "whisper_cpp_path": "whisper-cli"},
        asr={"model": "large-v3", "model_path": None},
    )
    runtimes = [
        {
            "backend": "cpu",
            "healthy": True,
            "executable_path": str(tmp_path / "whisper-cli.exe"),
        }
    ]
    with (
        patch("meeting_notes.runtime.installed_runtimes", return_value=runtimes),
        patch(
            "meeting_notes.runtime.vulkan_prerequisites",
            return_value=["cmake", "C++ compiler"],
        ),
    ):
        message = _asr_remediation(config, str(config_path))

    assert "runtime install --device vulkan" in message
    assert "runtime install --device cpu" in message
    assert "models download large-v3" in message
    assert "doctor --config" in message
    assert str(config_path.resolve()) in message


def test_diarization_recommendations_are_actionable() -> None:
    config = MeetingNotesConfig()
    config.diarization.enabled = True
    guidance = _diarization_recommendations(
        config,
        {"pyannote_installed": False, "hf_token_ready": False},
    )
    message = "\n".join(guidance)
    assert "uv sync --extra diarization" in message
    assert "meeting-notes diarization setup" in message
    assert "cannot accept them on your behalf" in message
    assert "--from diarize" in message


def test_local_diarization_model_does_not_recommend_hf_token(tmp_path: Path) -> None:
    config = MeetingNotesConfig()
    config.diarization.enabled = True
    config.diarization.model_path = str(tmp_path / "local-pipeline")
    guidance = _diarization_recommendations(
        config,
        {
            "pyannote_installed": True,
            "hf_token_ready": False,
            "local_diarization_model_ready": True,
        },
    )
    assert guidance == []


def test_hf_environment_token_takes_precedence() -> None:
    with patch.dict("os.environ", {"TEST_HF_TOKEN": "secret"}, clear=False):
        token, source = resolve_hf_token("TEST_HF_TOKEN")
    assert token == "secret"
    assert source == "environment variable TEST_HF_TOKEN"


def test_diarization_setup_reuses_managed_local_snapshot(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config = MeetingNotesConfig(setup=SetupConfig(completed=True))
    save_config(config, config_path)
    destination = tmp_path / "managed-model"
    destination.mkdir()
    (destination / "config.yaml").write_text("pipeline: test", encoding="utf-8")

    with (
        patch(
            "meeting_notes.diarization.setup.managed_diarization_dir",
            return_value=destination,
        ),
        patch("meeting_notes.diarization.setup.version", return_value="4.0.7"),
    ):
        run_diarization_setup(config_path=str(config_path))

    updated = load_config(str(config_path))
    assert updated.diarization.enabled is True
    assert updated.diarization.model_path == str(destination.resolve())


def test_enabled_unavailable_diarization_fails_instead_of_silently_skipping(
    tmp_path: Path,
) -> None:
    normalized = tmp_path / "audio" / "normalized.wav"
    normalized.parent.mkdir(parents=True)
    normalized.write_bytes(b"wave")
    config = MeetingNotesConfig()
    config.diarization.enabled = True
    manifest = {"stages": {}}

    with patch(
        "meeting_notes.diarization.pyannote.PyannoteDiarizationBackend.is_available",
        return_value=False,
    ), pytest.raises(DiarizationUnavailableError):
        _run_diarize(tmp_path, manifest, config)

    assert manifest["stages"]["diarize"]["status"] == "failed"


def test_merge_reconstructs_diarization_turns_and_assigns_speakers(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "asr" / "transcript.raw.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "metadata": {"backend": "whisper_cpp"},
                "segments": [
                    {
                        "id": "segment-000001",
                        "start": 0.0,
                        "end": 4.0,
                        "text": "hello",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    diarization_path = tmp_path / "diarization" / "diarization.json"
    diarization_path.parent.mkdir(parents=True)
    diarization_path.write_text(
        json.dumps(
            {
                "turns": [
                    {
                        "turn_id": "turn-000001",
                        "start": 0.0,
                        "end": 4.0,
                        "speaker": "SPEAKER_00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest: dict = {"stages": {}}
    config = MeetingNotesConfig(glossary={"enabled": False})

    _run_merge(tmp_path, manifest, config)

    merged = json.loads(
        (tmp_path / "transcript" / "transcript.merged.json").read_text(encoding="utf-8")
    )
    assert merged["segments"][0]["speaker"] == "SPEAKER_00"
    assert manifest["stages"]["merge"]["status"] == "completed"


def test_latest_transcript_metadata_reports_actual_backend(tmp_path: Path) -> None:
    transcript = tmp_path / "meetings" / "job" / "asr" / "transcript.raw.json"
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "metadata": {
                    "backend": "whisper_cpp",
                    "model": "large-v3-turbo",
                    "device": "cpu",
                    "language": "ko",
                }
            }
        ),
        encoding="utf-8",
    )
    config = MeetingNotesConfig(project={"data_dir": str(tmp_path)})
    metadata = _latest_transcript_metadata(config)
    assert metadata is not None
    assert metadata["backend"] == "whisper_cpp"
    assert metadata["device"] == "cpu"


def test_pyannote_uses_exclusive_diarization_when_available(tmp_path: Path) -> None:
    class Turn:
        start = 1.0
        end = 2.5

    class Output:
        exclusive_speaker_diarization = [(Turn(), "SPEAKER_00")]
        speaker_diarization = [(Turn(), "WRONG")]

    received: dict[str, object] = {}

    def fake_pipeline(audio: object, **_kwargs: object) -> Output:
        assert isinstance(audio, dict)
        received.update(audio)
        return Output()

    backend = PyannoteDiarizationBackend(use_exclusive=True)
    backend._pipeline = fake_pipeline
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"wave")
    with patch.object(
        backend,
        "_read_pcm_wave",
        return_value={"waveform": object(), "sample_rate": 16000},
    ):
        result = backend.diarize(audio)
    assert result.turns[0].speaker == "SPEAKER_00"
    assert received["sample_rate"] == 16000
