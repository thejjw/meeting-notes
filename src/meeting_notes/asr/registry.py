"""ASR backend registry."""

from __future__ import annotations

from meeting_notes.asr.base import ASRBackend
from meeting_notes.asr.whisper_cpp import WhisperCppBackend


_registry: dict[str, type[ASRBackend]] = {
    "whisper_cpp": WhisperCppBackend,
}


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

    if name not in _registry:
        raise ValueError(
            f"Unknown ASR backend: '{name}'. "
            f"Available: {', '.join(_registry.keys())}"
        )

    return _registry[name](**kwargs)  # type: ignore[call-arg]


def list_backends() -> list[str]:
    """List all registered backend names."""
    return list(_registry.keys())
