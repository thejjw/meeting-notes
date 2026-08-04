"""ASR backend registry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meeting_notes.asr.base import ASRBackend, ASRReadiness
from meeting_notes.asr.lemonade import LemonadeASRBackend
from meeting_notes.asr.whisper_cpp import WhisperCppBackend
from meeting_notes.storage import project_cache_root

if TYPE_CHECKING:
    from meeting_notes.config import MeetingNotesConfig


_registry: dict[str, type[ASRBackend]] = {
    "whisper_cpp": WhisperCppBackend,
    "lemonade": LemonadeASRBackend,
}


@dataclass
class ConfiguredASRBackend:
    """A backend plus its resolved invocation and provenance configuration."""

    backend: ASRBackend
    transcribe_kwargs: dict[str, Any]
    runtime_identity: dict[str, Any] = field(default_factory=dict)

    def check_readiness(self, *, allow_provision: bool = False) -> ASRReadiness:
        readiness = self.backend.check_readiness(
            model=str(self.transcribe_kwargs.get("model") or ""),
            expected_device=str(self.runtime_identity.get("device") or ""),
            allow_provision=allow_provision,
        )
        if (
            readiness.available
            and self.backend.name == "whisper_cpp"
            and not isinstance(self.transcribe_kwargs.get("model_path"), Path)
        ):
            return ASRReadiness(
                available=False,
                detail="configured whisper.cpp model_path is missing",
                version=readiness.version,
                device=readiness.device,
            )
        model_path = self.transcribe_kwargs.get("model_path")
        if (
            readiness.available
            and self.backend.name == "whisper_cpp"
            and isinstance(model_path, Path)
            and not model_path.is_file()
        ):
            return ASRReadiness(
                available=False,
                detail=f"configured whisper.cpp model is missing: {model_path}",
                version=readiness.version,
                device=readiness.device,
            )
        return readiness


def register_backend(name: str, backend_class: type[ASRBackend]) -> None:
    """Register an ASR backend class."""
    _registry[name] = backend_class


def get_backend(name: str, **kwargs: object) -> ASRBackend:
    """Get an ASR backend instance by name.

    Args:
        name: Backend identifier (e.g., 'whisper_cpp').
        **kwargs: Arguments passed to the backend constructor.

    Returns:
        ASRBackend instance.

    Raises:
        ValueError: If the backend name is not registered.
    """
    # Lazy-import optional backends
    if name == "openai_whisper" and name not in _registry:
        try:
            from meeting_notes.asr.openai_whisper import OpenAIWhisperBackend

            register_backend("openai_whisper", OpenAIWhisperBackend)
        except ImportError:
            pass

    if name == "faster_whisper" and name not in _registry:
        try:
            from meeting_notes.asr.faster_whisper import FasterWhisperBackend

            register_backend("faster_whisper", FasterWhisperBackend)
        except ImportError:
            pass

    if name == "whisper_cpp_docker" and name not in _registry:
        try:
            from meeting_notes.asr.whisper_cpp_docker import DockerWhisperCppBackend

            register_backend("whisper_cpp_docker", DockerWhisperCppBackend)
        except ImportError:
            pass

    if name == "lemonade" and name not in _registry:
        from meeting_notes.asr.lemonade import LemonadeASRBackend

        register_backend("lemonade", LemonadeASRBackend)

    if name not in _registry:
        raise ValueError(
            f"Unknown ASR backend: '{name}'. "
            f"Available: {', '.join(_registry.keys())}"
        )

    return _registry[name](**kwargs)  # type: ignore[call-arg]


def list_backends() -> list[str]:
    """List all registered backend names."""
    return list(_registry.keys())


def get_configured_backend(config: MeetingNotesConfig) -> ConfiguredASRBackend:
    """Construct the selected backend and resolve all backend-specific options."""
    common: dict[str, Any] = {
        "model": config.asr.model,
        "model_path": Path(config.asr.model_path) if config.asr.model_path else None,
        "language": config.asr.language,
        "task": config.asr.task,
        "initial_prompt": config.asr.initial_prompt,
        "word_timestamps": config.asr.word_timestamps,
        "threads": config.runtime.threads,
    }
    identity: dict[str, Any] = {
        "backend": config.runtime.asr_backend,
        "device": config.runtime.device,
        "model": config.asr.model,
        "model_path": config.asr.model_path,
    }

    if config.runtime.asr_backend == "whisper_cpp":
        from meeting_notes.runtime import find_manifest_for_executable

        options = config.asr.backend_options.whisper_cpp
        executable = Path(config.runtime.whisper_cpp_path).resolve()
        runtime_manifest = find_manifest_for_executable(
            executable, cache_dir=project_cache_root(config)
        )
        common.update(
            {
                "threads": _resolve_whisper_threads(config),
                "device": config.runtime.device,
                "model_variant": options.model_variant,
                "flash_attention": options.flash_attention,
                "extra_args": options.extra_args,
                "gpu_device": options.gpu_device,
            }
        )
        identity.update(
            {
                "executable": str(executable),
                "managed": runtime_manifest is not None,
                "runtime_version": (
                    runtime_manifest.get("version") if runtime_manifest else None
                ),
                "runtime_backend": (
                    runtime_manifest.get("backend") if runtime_manifest else None
                ),
                "source_revision": (
                    runtime_manifest.get("source_revision") if runtime_manifest else None
                ),
            }
        )
        backend = get_backend("whisper_cpp", executable=config.runtime.whisper_cpp_path)
    elif config.runtime.asr_backend == "lemonade":
        from meeting_notes.asr.lemonade import LemonadeASRBackend

        options = config.asr.backend_options.lemonade
        backend = LemonadeASRBackend(
            base_url=options.base_url,
            model_id=options.model_id,
            api_key_env=options.api_key_env,
            expected_device=config.runtime.device,
            connect_timeout_seconds=options.connect_timeout_seconds,
            provisioning_timeout_seconds=options.provisioning_timeout_seconds,
            transcription_timeout_seconds=options.transcription_timeout_seconds,
        )
        common["model_path"] = None
        identity.update(
            {
                "base_url": options.base_url,
                "lemonade_model_id": options.model_id,
            }
        )
    elif config.runtime.asr_backend == "faster_whisper":
        options = config.asr.backend_options.faster_whisper
        backend = get_backend("faster_whisper")
        common.update(
            {
                "device": options.device,
                "compute_type": options.compute_type,
                "batch_size": options.batch_size,
                "threads": options.cpu_threads,
            }
        )
    else:
        backend = get_backend(config.runtime.asr_backend)

    return ConfiguredASRBackend(
        backend=backend,
        transcribe_kwargs=common,
        runtime_identity=identity,
    )


def _resolve_whisper_threads(config: MeetingNotesConfig) -> int:
    if config.runtime.threads > 0:
        return config.runtime.threads
    logical_cores = os.cpu_count() or 1
    threads = max(1, logical_cores - config.runtime.reserve_logical_cores)
    if config.runtime.max_auto_threads > 0:
        threads = min(threads, config.runtime.max_auto_threads)
    return threads
