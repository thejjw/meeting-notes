"""Provision and inspect the GPU-only Lemonade Qwen3-ASR backend."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import typer
from rich.console import Console

from meeting_notes.asr.qwen3_lemonade import (
    QWEN_GGUF_CHECKPOINT,
    Qwen3ASRLemonadeBackend,
    legacy_native_asr_dir,
    managed_aligner_dir,
)
from meeting_notes.config import load_config, resolve_config_path, save_config
from meeting_notes.diarization.acceleration import (
    default_runtime_dir,
    probe_rocm,
    provision_runtime,
    runtime_environment,
    runtime_python,
)
from meeting_notes.storage import directory_size, project_cache_root

console = Console(stderr=True)

QWEN_GGUF_STORAGE_BYTES = 2_523_000_000
QWEN_ALIGNER_DOWNLOAD_BYTES = 1_835_545_960
ROCM_STAGING_BYTES = 7 * 1024**3


class QwenSetupError(RuntimeError):
    """Lemonade Qwen3-ASR setup could not complete."""


def _format_gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _download_aligner(model_id: str, destination: Path, cache_root: Path, runtime: Path) -> Path:
    if (destination / "config.json").is_file():
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "from huggingface_hub import snapshot_download; import sys; "
        "snapshot_download(repo_id=sys.argv[1],local_dir=sys.argv[2],cache_dir=sys.argv[3])"
    )
    completed = subprocess.run(
        [
            str(runtime_python(runtime)),
            "-c",
            script,
            model_id,
            str(destination),
            str(cache_root / "huggingface"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=runtime_environment(runtime),
        check=False,
    )
    if completed.returncode or not (destination / "config.json").is_file():
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise QwenSetupError(f"Forced-aligner download failed: {detail}")
    return destination.resolve()


def _backend(config: object, *, runtime: Path, aligner_path: Path) -> Qwen3ASRLemonadeBackend:
    # Keep imports local to avoid widening this setup module's public type surface.
    from meeting_notes.config import MeetingNotesConfig

    if not isinstance(config, MeetingNotesConfig):
        raise TypeError("Expected MeetingNotesConfig.")
    options = config.asr.backend_options.qwen3_asr_lemonade
    return Qwen3ASRLemonadeBackend(
        base_url=options.base_url,
        model_id=options.model_id,
        checkpoint=options.checkpoint,
        api_key_env=options.api_key_env,
        llamacpp_backend=options.llamacpp_backend,
        python_executable=str(runtime_python(runtime)),
        aligner_path=aligner_path,
        ctx_size=options.ctx_size,
        max_new_tokens=options.max_new_tokens,
        torch_compile=options.torch_compile,
        connect_timeout_seconds=options.connect_timeout_seconds,
        provisioning_timeout_seconds=options.provisioning_timeout_seconds,
        transcription_timeout_seconds=options.transcription_timeout_seconds,
        worker_timeout_seconds=options.worker_timeout_seconds,
        environment=runtime_environment(runtime),
    )


def _smoke_test(backend: Qwen3ASRLemonadeBackend) -> None:
    """Exercise chat-audio routing and load the ROCm forced aligner."""
    with tempfile.TemporaryDirectory(prefix="qwen3-smoke-") as temporary:
        audio_path = Path(temporary) / "silence.wav"
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 16_000)
        backend.transcribe(audio_path, language="ko", task="transcribe")


def _remove_legacy_native_weights(model_cache: Path) -> int:
    target = legacy_native_asr_dir(model_cache).resolve()
    allowed_root = (model_cache.resolve() / "qwen3-asr").resolve()
    if not target.is_relative_to(allowed_root) or target.name != "Qwen--Qwen3-ASR-1.7B-hf":
        raise QwenSetupError(f"Refusing unsafe legacy-model cleanup target: {target}")
    size = directory_size(target)
    if target.exists():
        shutil.rmtree(target)
    return size


def run_setup(
    *,
    config_path: str | None,
    activate: bool,
    yes: bool,
    force_runtime: bool,
) -> None:
    config = load_config(config_path)
    resolved_config = resolve_config_path(config_path)
    if resolved_config is None:
        raise QwenSetupError("No writable active configuration was found.")
    options = config.asr.backend_options.qwen3_asr_lemonade
    if options.checkpoint != QWEN_GGUF_CHECKPOINT:
        raise QwenSetupError(
            f"Only the supported 1.7B Q8 checkpoint is allowed: {QWEN_GGUF_CHECKPOINT}"
        )

    cache_root = project_cache_root(config)
    model_cache = cache_root / "models"
    aligner_path = managed_aligner_dir(model_cache, options.aligner_model_id)
    remaining_aligner = max(0, QWEN_ALIGNER_DOWNLOAD_BYTES - directory_size(aligner_path))
    runtime_path = default_runtime_dir(config)
    runtime_staging = ROCM_STAGING_BYTES if force_runtime or not runtime_path.exists() else 0
    disk_root = cache_root if cache_root.exists() else cache_root.parent
    free = shutil.disk_usage(disk_root).free

    console.print("\n[bold]Lemonade Qwen3-ASR GPU setup[/bold]\n")
    console.print(f"  GGUF checkpoint: {options.checkpoint}")
    console.print(f"  Transcription: Lemonade llama.cpp / {options.llamacpp_backend} GPU")
    console.print(f"  Forced aligner: {options.aligner_model_id} / ROCm GPU")
    console.print(
        f"  Lemonade-managed GGUF storage: approximately {_format_gib(QWEN_GGUF_STORAGE_BYTES)}"
    )
    console.print(f"  Remaining project-local aligner download: {_format_gib(remaining_aligner)}")
    console.print(f"  Project cache: {cache_root}")
    if runtime_staging:
        console.print(
            "  Shared ROCm provisioning may temporarily use approximately 7 GiB "
            "while an existing runtime is preserved."
        )
    if free < remaining_aligner + runtime_staging:
        raise QwenSetupError(
            f"Insufficient project-disk space: need approximately "
            f"{_format_gib(remaining_aligner + runtime_staging)}, have {_format_gib(free)}."
        )
    if not yes and sys.stdin.isatty() and not typer.confirm("Proceed with setup?", default=True):
        raise typer.Exit()

    probe = probe_rocm(config)
    if probe.state in {"unsupported", "prerequisites-missing"}:
        raise QwenSetupError(probe.detail)
    runtime = provision_runtime(
        config,
        force=force_runtime,
        profiles=("diarization", "qwen3_alignment"),
    )
    aligner_path = _download_aligner(options.aligner_model_id, aligner_path, cache_root, runtime)
    backend = _backend(config, runtime=runtime, aligner_path=aligner_path)
    if not backend.is_available():
        raise QwenSetupError(
            f"Lemonade Server is not reachable at {options.base_url}. Start it and retry."
        )

    info = backend.model_info()
    if not info or not info.get("downloaded"):
        last_bucket = -1

        def report_progress(event: dict[str, object]) -> None:
            nonlocal last_bucket
            raw_percent = event.get("percent")
            percent = int(float(raw_percent)) if isinstance(raw_percent, (int, float, str)) else 0
            bucket = percent // 10
            if bucket != last_bucket or event.get("event") == "complete":
                last_bucket = bucket
                console.print(f"  Lemonade model download: {percent}%")

        backend.pull_model(progress=report_progress)
    info = backend.model_info()
    if info:
        options.model_id = str(info.get("id") or options.model_id)
    backend.load_model()
    _smoke_test(backend)

    options.rocm_gpu_runtime_path = str(runtime.resolve())
    if config.diarization.rocm_gpu_runtime_path or config.diarization.device == "rocm-hybrid":
        config.diarization.rocm_gpu_runtime_path = str(runtime.resolve())
    if activate:
        config.runtime.asr_backend = "qwen3_asr_lemonade"
        config.runtime.device = "rocm"
        config.asr.model = options.model_id
        config.asr.model_path = None
    save_config(config, resolved_config)

    reclaimed = _remove_legacy_native_weights(model_cache)
    console.print("\n[green]Lemonade Qwen3-ASR and forced alignment are ready.[/green]")
    console.print(f"  Lemonade model: {options.model_id}")
    console.print(f"  Forced aligner: {aligner_path}")
    console.print(f"  Shared ROCm runtime: {runtime}")
    if reclaimed:
        console.print(f"  Removed obsolete native ASR weights: {_format_gib(reclaimed)}")
    if not activate:
        console.print("  Active ASR backend was not changed; pass --activate to select Qwen.")


def show_status(*, config_path: str | None) -> None:
    config = load_config(config_path)
    options = config.asr.backend_options.qwen3_asr_lemonade
    cache = project_cache_root(config) / "models"
    aligner_path = managed_aligner_dir(cache, options.aligner_model_id)
    runtime = (
        Path(options.rocm_gpu_runtime_path).expanduser().resolve()
        if options.rocm_gpu_runtime_path
        else default_runtime_dir(config)
    )
    probe = probe_rocm(config)
    backend = _backend(config, runtime=runtime, aligner_path=aligner_path)
    readiness = backend.check_readiness()
    console.print("\n[bold]Lemonade Qwen3-ASR status[/bold]\n")
    console.print(f"  Active: {config.runtime.asr_backend == 'qwen3_asr_lemonade'}")
    console.print(f"  Required devices: Lemonade {options.llamacpp_backend} GPU + Python ROCm GPU")
    console.print(f"  GGUF checkpoint: {options.checkpoint}")
    console.print(
        f"  Forced aligner: {'ready' if (aligner_path / 'config.json').is_file() else 'missing'}"
    )
    console.print(f"  Shared ROCm runtime: {runtime}")
    console.print(f"  ROCm probe: {probe.state} ({probe.detail})")
    console.print(
        f"  Lemonade: {'ready' if readiness.available else 'not ready'} ({readiness.detail})"
    )
