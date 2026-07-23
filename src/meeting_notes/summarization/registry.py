"""Summarizer backend registry."""

from __future__ import annotations

from meeting_notes.summarization.base import SummarizerBackend


_registry: dict[str, type[SummarizerBackend]] = {}


def register_backend(name: str, cls: type[SummarizerBackend]) -> None:
    _registry[name] = cls


def get_backend(name: str) -> SummarizerBackend:
    if name not in _registry:
        # Lazy-load
        if name == "codex_cli":
            from meeting_notes.summarization.codex_cli import CodexCliBackend
            register_backend("codex_cli", CodexCliBackend)
        elif name == "none":
            from meeting_notes.summarization.none import NoSummarizerBackend
            register_backend("none", NoSummarizerBackend)
        elif name == "local_command":
            from meeting_notes.summarization.local_command import LocalCommandBackend
            register_backend("local_command", LocalCommandBackend)
        else:
            raise ValueError(f"Unknown summarizer backend: '{name}'")

    return _registry[name]()


def list_backends() -> list[str]:
    return list(_registry.keys()) or ["codex_cli", "none", "local_command"]
