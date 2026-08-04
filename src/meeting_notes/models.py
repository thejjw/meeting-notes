"""Verified GGML model provisioning."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from meeting_notes.artifacts import MODEL_ARTIFACTS, model_url
from meeting_notes.runtime import RuntimeInstallError, download_file, sha256_file

if TYPE_CHECKING:
    from pathlib import Path


class ModelInstallError(RuntimeInstallError):
    """A model could not be downloaded or verified."""


def model_path(name: str, *, cache_dir: Path) -> Path:
    return cache_dir.resolve() / "models" / f"ggml-{name}.bin"


def model_metadata(name: str) -> dict[str, object]:
    try:
        return MODEL_ARTIFACTS[name]
    except KeyError as exc:
        raise ModelInstallError(
            f"Unknown model '{name}'. Available models: {', '.join(MODEL_ARTIFACTS)}"
        ) from exc


def verify_model(
    name: str,
    path: Path | None = None,
    *,
    cache_dir: Path | None = None,
) -> tuple[bool, str]:
    metadata = model_metadata(name)
    if path is not None:
        target = path
    elif cache_dir is not None:
        target = model_path(name, cache_dir=cache_dir)
    else:
        raise ModelInstallError("cache_dir is required when no explicit model path is provided.")
    if not target.is_file():
        return False, f"missing: {target}"
    expected_size = int(str(metadata["size"]))
    if target.stat().st_size != expected_size:
        return False, f"size mismatch: expected {expected_size}, got {target.stat().st_size}"
    actual = sha256_file(target)
    expected = str(metadata["sha256"])
    if actual != expected:
        return False, f"checksum mismatch: expected {expected}, got {actual}"
    return True, "verified"


def download_model(name: str, *, cache_dir: Path) -> Path:
    model_metadata(name)
    destination = model_path(name, cache_dir=cache_dir)
    valid, _ = verify_model(name, destination)
    if valid:
        return destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    temporary.unlink(missing_ok=True)
    try:
        download_file(model_url(name), temporary)
        valid, reason = verify_model(name, temporary)
        if not valid:
            raise ModelInstallError(f"Downloaded model verification failed: {reason}")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination.resolve()


def model_statuses(*, cache_dir: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for name, metadata in MODEL_ARTIFACTS.items():
        path = model_path(name, cache_dir=cache_dir)
        valid, detail = verify_model(name, path)
        values.append(
            {
                "name": name,
                "path": str(path),
                "size": metadata["size"],
                "installed": path.is_file(),
                "verified": valid,
                "detail": detail,
            }
        )
    return values
