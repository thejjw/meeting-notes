"""Project-local first-party cache paths and legacy cache discovery."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from meeting_notes.config import MeetingNotesConfig


class StorageMigrationError(RuntimeError):
    """Project-local cache migration could not complete safely."""


def project_cache_root(config: MeetingNotesConfig) -> Path:
    """Return the configured absolute project cache directory."""
    return Path(config.project.cache_dir).expanduser().resolve()


def legacy_user_cache_root() -> Path:
    """Return the former per-user cache, used only by explicit migration."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "meeting-notes" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "meeting-notes"


def directory_size(path: Path) -> int:
    """Return recursive file bytes while tolerating concurrent cache changes."""
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def cache_inventory(config: MeetingNotesConfig) -> dict[str, object]:
    """Return project and legacy first-party cache usage."""
    project = project_cache_root(config)
    legacy = legacy_user_cache_root().resolve()

    def section(root: Path) -> dict[str, object]:
        values = {
            name: {
                "path": str(root / name),
                "exists": (root / name).exists(),
                "bytes": directory_size(root / name),
            }
            for name in ("models", "runtimes", "diarization")
        }
        return {
            "root": str(root),
            "total_bytes": directory_size(root),
            "sections": values,
        }

    return {"project": section(project), "legacy": section(legacy)}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _prune_empty_tree(root: Path, stop: Path) -> None:
    """Remove empty descendants and roots without touching retained files."""
    if root.is_dir():
        directories = [path for path in root.rglob("*") if path.is_dir()]
        for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
            with suppress(OSError):
                directory.rmdir()
        with suppress(OSError):
            root.rmdir()
    with suppress(OSError):
        stop.rmdir()


def migrate_legacy_cache(
    config: MeetingNotesConfig,
    config_path: Path,
) -> dict[str, object]:
    """Transactionally migrate recognized Whisper assets into the project cache."""
    from meeting_notes.artifacts import MODEL_ARTIFACTS
    from meeting_notes.config import save_config
    from meeting_notes.models import verify_model
    from meeting_notes.runtime import load_manifest, validate_executable

    project = project_cache_root(config)
    legacy = legacy_user_cache_root().resolve()
    if project == legacy or _is_relative_to(project, legacy) or _is_relative_to(legacy, project):
        raise StorageMigrationError(
            "Project cache and legacy cache must be separate, non-nested directories."
        )
    project.mkdir(parents=True, exist_ok=True)

    model_sources: list[tuple[str, Path, Path]] = []
    legacy_models = legacy / "models"
    for name in MODEL_ARTIFACTS:
        source = legacy_models / f"ggml-{name}.bin"
        if source.is_file():
            valid, detail = verify_model(name, source)
            if not valid:
                raise StorageMigrationError(f"Legacy Whisper model is invalid: {detail}")
            model_sources.append((name, source, project / "models" / source.name))

    runtime_sources: list[tuple[Path, Path, dict[str, Any], Path]] = []
    legacy_runtimes = legacy / "runtimes"
    if legacy_runtimes.is_dir():
        for manifest_path in legacy_runtimes.rglob("manifest.json"):
            source = manifest_path.parent
            manifest = load_manifest(source)
            if manifest is None:
                raise StorageMigrationError(f"Legacy runtime manifest is invalid: {manifest_path}")
            executable = Path(str(manifest.get("executable_path", ""))).resolve()
            if not executable.is_file() or not _is_relative_to(executable, source):
                raise StorageMigrationError(
                    f"Legacy runtime executable is invalid or outside its runtime: {executable}"
                )
            validate_executable(executable)
            relative_executable = executable.relative_to(source.resolve())
            destination = project / "runtimes" / source.relative_to(legacy_runtimes)
            runtime_sources.append((source, destination, manifest, relative_executable))

    unknown_runtimes: list[str] = []
    if legacy_runtimes.is_dir():
        recognized_roots = [source.resolve() for source, _, _, _ in runtime_sources]
        for path in legacy_runtimes.rglob("*"):
            if not path.is_file():
                continue
            if not any(_is_relative_to(path, root) for root in recognized_roots):
                unknown_runtimes.append(str(path))

    unknown_models = []
    if legacy_models.is_dir():
        recognized = {source.resolve() for _, source, _ in model_sources}
        unknown_models = [
            str(path) for path in legacy_models.iterdir() if path.resolve() not in recognized
        ]

    installed: list[Path] = []
    migrated_models: list[str] = []
    migrated_runtimes: list[str] = []
    removable_models: list[Path] = []
    removable_runtimes: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=".cache-migrate-", dir=project) as temporary:
        staging = Path(temporary)
        staged_models: list[tuple[str, Path, Path, Path]] = []
        for name, source, destination in model_sources:
            if destination.exists():
                valid, detail = verify_model(name, destination)
                if not valid:
                    raise StorageMigrationError(
                        f"Project model destination conflicts with legacy model: {detail}"
                    )
                removable_models.append(source)
                continue
            staged = staging / "models" / source.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            valid, detail = verify_model(name, staged)
            if not valid:
                raise StorageMigrationError(f"Staged Whisper model is invalid: {detail}")
            staged_models.append((name, source, staged, destination))

        staged_runtimes: list[tuple[Path, Path, Path]] = []
        for source, destination, manifest, relative_executable in runtime_sources:
            if destination.exists():
                current = load_manifest(destination)
                executable = destination / relative_executable
                if (
                    current is None
                    or current.get("version") != manifest.get("version")
                    or current.get("backend") != manifest.get("backend")
                    or current.get("platform") != manifest.get("platform")
                    or current.get("architecture") != manifest.get("architecture")
                    or Path(str(current.get("executable_path", ""))).resolve()
                    != executable.resolve()
                    or not executable.is_file()
                ):
                    raise StorageMigrationError(
                        f"Project runtime destination conflicts with legacy runtime: {destination}"
                    )
                validate_executable(executable)
                removable_runtimes.append(source)
                continue
            staged = staging / "runtimes" / source.relative_to(legacy_runtimes)
            shutil.copytree(source, staged)
            final_executable = destination / relative_executable
            staged_manifest = dict(manifest)
            staged_manifest["executable_path"] = str(final_executable.resolve())
            (staged / "manifest.json").write_text(
                json.dumps(staged_manifest, indent=2), encoding="utf-8"
            )
            validate_executable(staged / relative_executable)
            staged_runtimes.append((source, staged, destination))

        try:
            for name, source, staged, destination in staged_models:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, destination)
                installed.append(destination)
                removable_models.append(source)
                migrated_models.append(name)
            for source, staged, destination in staged_runtimes:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, destination)
                installed.append(destination)
                removable_runtimes.append(source)
                migrated_runtimes.append(str(destination))

            old_project_cache = config.project.cache_dir
            old_model_cache = config.asr.model_cache_dir
            old_model_path = config.asr.model_path
            old_runtime_path = config.runtime.whisper_cpp_path
            config.project.cache_dir = str(project)
            config.asr.model_cache_dir = str((project / "models").resolve())
            if old_model_path:
                old_path = Path(old_model_path)
                if _is_relative_to(old_path, legacy_models):
                    config.asr.model_path = str((project / "models" / old_path.name).resolve())
            if old_runtime_path:
                old_path = Path(old_runtime_path)
                if _is_relative_to(old_path, legacy_runtimes):
                    relative = old_path.resolve().relative_to(legacy_runtimes.resolve())
                    config.runtime.whisper_cpp_path = str(
                        (project / "runtimes" / relative).resolve()
                    )
            try:
                save_config(config, config_path)
            except Exception:
                config.project.cache_dir = old_project_cache
                config.asr.model_cache_dir = old_model_cache
                config.asr.model_path = old_model_path
                config.runtime.whisper_cpp_path = old_runtime_path
                raise
        except Exception:
            for path in reversed(installed):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            raise

    removed_bytes = 0
    for source in removable_models:
        if source.is_file():
            removed_bytes += source.stat().st_size
            source.unlink()
    for source in removable_runtimes:
        if source.is_dir():
            removed_bytes += directory_size(source)
            shutil.rmtree(source)
    _prune_empty_tree(legacy_models, legacy)
    _prune_empty_tree(legacy_runtimes, legacy)
    with suppress(OSError):
        legacy.parent.rmdir()
    return {
        "project_cache": str(project),
        "legacy_cache": str(legacy),
        "migrated_models": migrated_models,
        "migrated_runtimes": migrated_runtimes,
        "removed_legacy_bytes": removed_bytes,
        "unknown_legacy_models": unknown_models,
        "unknown_legacy_runtime_files": unknown_runtimes,
        "asr_backend": config.runtime.asr_backend,
        "device": config.runtime.device,
    }
