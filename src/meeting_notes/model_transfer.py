"""Portable archives for managed Whisper and diarization models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, cast

from meeting_notes.artifacts import MODEL_ARTIFACTS
from meeting_notes.config import load_config, resolve_config_path, save_config
from meeting_notes.diarization.acceleration import model_dir as managed_diarization_model_dir
from meeting_notes.models import model_path, verify_model
from meeting_notes.runtime import sha256_file
from meeting_notes.storage import project_cache_root

TRANSFER_VERSION = 1
MANIFEST_NAME = "meeting-notes-transfer.json"


class ModelTransferError(RuntimeError):
    """A model archive could not be created or restored safely."""


def _package_version() -> str:
    try:
        return version("meeting-notes")
    except PackageNotFoundError:
        return "development"


def _active_config(config_path: str | None) -> tuple[Path, Any]:
    resolved = resolve_config_path(config_path)
    if resolved is None:
        raise ModelTransferError(
            "No active configuration found. Run 'uv run meeting-notes configure' first "
            "or pass -Config/--config."
        )
    return resolved.resolve(), load_config(str(resolved))


def _compression(value: str) -> tuple[int, int | None]:
    choices = {
        "optimal": (zipfile.ZIP_DEFLATED, 6),
        "fastest": (zipfile.ZIP_DEFLATED, 1),
        "none": (zipfile.ZIP_STORED, None),
    }
    try:
        return choices[value.lower()]
    except KeyError as error:
        raise ModelTransferError(
            "Compression level must be Optimal, Fastest, or None."
        ) from error


def _default_archive(kind: str, identity: str, fingerprint: str) -> Path:
    safe_identity = "".join(
        character if character.isalnum() or character in "-._" else "-"
        for character in identity
    )
    return Path.cwd() / f"meeting-notes-{kind}-{safe_identity}-{fingerprint[:12]}.zip"


def _archive_sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _write_sidecar(path: Path) -> Path:
    digest = sha256_file(path)
    sidecar = _archive_sidecar(path)
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(temporary, sidecar)
    return sidecar


def _verify_sidecar(path: Path) -> bool:
    sidecar = _archive_sidecar(path)
    if not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="ascii").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ModelTransferError(f"Invalid checksum sidecar: {sidecar}")
    actual = sha256_file(path)
    if actual.lower() != fields[0].lower():
        raise ModelTransferError(
            f"Archive checksum mismatch: expected {fields[0]}, got {actual}"
        )
    return True


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_archive(
    destination: Path,
    manifest: dict[str, Any],
    files: list[tuple[Path, str]],
    *,
    compression_level: str,
    force: bool,
) -> tuple[Path, Path]:
    destination = destination.resolve()
    sidecar = _archive_sidecar(destination)
    if not force and (destination.exists() or sidecar.exists()):
        raise ModelTransferError(
            f"Archive or checksum already exists: {destination}. Use -Force to replace it."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    compression, compresslevel = _compression(compression_level)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        kwargs: dict[str, Any] = {
            "mode": "w",
            "compression": compression,
            "allowZip64": True,
        }
        if compresslevel is not None:
            kwargs["compresslevel"] = compresslevel
        with zipfile.ZipFile(temporary, **kwargs) as archive:
            archive.writestr(
                MANIFEST_NAME,
                json.dumps(manifest, indent=2, ensure_ascii=False),
            )
            for source, relative in files:
                archive.write(source, relative)
        os.replace(temporary, destination)
        return destination, _write_sidecar(destination)
    finally:
        temporary.unlink(missing_ok=True)


def backup_whisper(
    *,
    config_path: str | None = None,
    model: str | None = None,
    archive_path: Path | None = None,
    compression_level: str = "optimal",
    force: bool = False,
) -> tuple[Path, Path]:
    """Archive one verified managed Whisper model."""
    _, config = _active_config(config_path)
    selected = model or config.asr.model
    if selected not in MODEL_ARTIFACTS:
        raise ModelTransferError(
            f"Unknown managed Whisper model '{selected}'. "
            f"Available: {', '.join(MODEL_ARTIFACTS)}"
        )
    source = model_path(selected, cache_dir=project_cache_root(config))
    valid, detail = verify_model(selected, source)
    if not valid:
        raise ModelTransferError(f"Whisper model is not ready for backup: {detail}")
    metadata = MODEL_ARTIFACTS[selected]
    digest = str(metadata["sha256"])
    destination = archive_path or _default_archive("whisper", selected, digest)
    relative = f"payload/{source.name}"
    manifest = {
        "version": TRANSFER_VERSION,
        "kind": "whisper",
        "created_at": datetime.now(UTC).isoformat(),
        "meeting_notes_version": _package_version(),
        "model": {
            "name": selected,
            "filename": source.name,
            "upstream_sha256": digest,
        },
        "files": [_file_record(source, relative)],
    }
    return _write_archive(
        destination,
        manifest,
        [(source, relative)],
        compression_level=compression_level,
        force=force,
    )


def _diarization_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if relative.as_posix() == ".meeting-notes-manifest.json":
            continue
        if path.is_symlink():
            raise ModelTransferError(f"Refusing to archive symbolic link: {path}")
        if path.is_file():
            files.append(path)
    if not (source / "config.yaml").is_file():
        raise ModelTransferError(f"Diarization model is missing config.yaml: {source}")
    return files


def backup_diarization(
    *,
    config_path: str | None = None,
    archive_path: Path | None = None,
    compression_level: str = "optimal",
    force: bool = False,
) -> tuple[Path, Path]:
    """Archive the configured local diarization pipeline without credentials."""
    _, config = _active_config(config_path)
    if config.diarization.backend != "pyannote":
        raise ModelTransferError("Only the pyannote diarization backend can be archived.")
    if not config.diarization.model_path:
        raise ModelTransferError(
            "diarization.model_path is not configured. Run diarization setup first."
        )
    source = Path(config.diarization.model_path).resolve()
    if not source.is_dir():
        raise ModelTransferError(f"Configured diarization model is missing: {source}")
    source_manifest: dict[str, Any] = {}
    manifest_path = source / ".meeting-notes-manifest.json"
    if manifest_path.is_file():
        try:
            source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ModelTransferError(f"Invalid diarization install manifest: {error}") from error
    repo_id = str(source_manifest.get("repo_id") or config.diarization.model)
    revision = str(source_manifest.get("revision") or "unknown")
    files = _diarization_files(source)
    archive_files = [
        (path, f"payload/{path.relative_to(source).as_posix()}") for path in files
    ]
    fingerprint = revision if revision != "unknown" else sha256_file(source / "config.yaml")
    destination = archive_path or _default_archive(
        "diarization", repo_id.rsplit("/", 1)[-1], fingerprint
    )
    manifest = {
        "version": TRANSFER_VERSION,
        "kind": "diarization",
        "created_at": datetime.now(UTC).isoformat(),
        "meeting_notes_version": _package_version(),
        "model": {
            "repo_id": repo_id,
            "revision": revision,
            "pyannote_audio": source_manifest.get("pyannote_audio"),
        },
        "files": [_file_record(path, relative) for path, relative in archive_files],
    }
    return _write_archive(
        destination,
        manifest,
        archive_files,
        compression_level=compression_level,
        force=force,
    )


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in name:
        raise ModelTransferError(f"Unsafe archive member: {name}")
    return path


def _read_manifest(archive: zipfile.ZipFile, expected_kind: str) -> dict[str, Any]:
    names: set[str] = set()
    for info in archive.infolist():
        _safe_member(info.filename)
        if info.filename in names:
            raise ModelTransferError(f"Duplicate archive member: {info.filename}")
        names.add(info.filename)
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise ModelTransferError(f"Symbolic links are not allowed: {info.filename}")
    if MANIFEST_NAME not in names:
        raise ModelTransferError(f"Archive is missing {MANIFEST_NAME}.")
    try:
        manifest = cast("dict[str, Any]", json.loads(archive.read(MANIFEST_NAME)))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ModelTransferError(f"Invalid transfer manifest: {error}") from error
    if manifest.get("version") != TRANSFER_VERSION:
        raise ModelTransferError(
            f"Unsupported transfer manifest version: {manifest.get('version')}"
        )
    if manifest.get("kind") != expected_kind:
        raise ModelTransferError(
            f"Expected a {expected_kind} archive, found {manifest.get('kind')!r}."
        )
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise ModelTransferError("Transfer manifest has no payload files.")
    records = cast("list[Any]", raw_records)
    normalized_records: list[dict[str, Any]] = []
    expected_names = {MANIFEST_NAME}
    for record in records:
        if not isinstance(record, dict):
            raise ModelTransferError("Invalid file record in transfer manifest.")
        record = cast("dict[str, Any]", record)
        relative = str(record.get("path", ""))
        safe = _safe_member(relative)
        if safe.parts[0] != "payload" or len(safe.parts) < 2:
            raise ModelTransferError(f"Payload path must be under payload/: {relative}")
        expected_names.add(relative)
        normalized_records.append(record)
    actual_files = {info.filename for info in archive.infolist() if not info.is_dir()}
    if actual_files != expected_names:
        missing = sorted(expected_names - actual_files)
        extra = sorted(actual_files - expected_names)
        raise ModelTransferError(
            f"Archive payload inventory mismatch; missing={missing}, extra={extra}"
        )
    manifest["files"] = normalized_records
    return manifest


def _extract_verified(
    archive: zipfile.ZipFile, manifest: dict[str, Any], destination: Path
) -> None:
    for record in manifest["files"]:
        relative = str(record["path"])
        target = destination.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with archive.open(relative) as source, target.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != int(record.get("size", -1)):
            raise ModelTransferError(f"Size mismatch for {relative}.")
        if digest.hexdigest() != str(record.get("sha256", "")).lower():
            raise ModelTransferError(f"Checksum mismatch for {relative}.")


def _atomic_install(
    staged: Path,
    destination: Path,
    update_config: Any,
    *,
    force: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        raise ModelTransferError(
            f"Destination already exists: {destination}. Use -Force to replace it."
        )
    backup = destination.with_name(f".{destination.name}.restore-backup-{os.getpid()}")
    if backup.exists():
        raise ModelTransferError(f"Restore backup path already exists: {backup}")
    replaced = destination.exists()
    try:
        if replaced:
            os.replace(destination, backup)
        os.replace(staged, destination)
        update_config()
    except Exception:
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if backup.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def restore_archive(
    kind: str,
    archive_path: Path,
    *,
    config_path: str | None = None,
    force: bool = False,
) -> Path:
    """Validate and transactionally restore a model archive."""
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise ModelTransferError(f"Archive does not exist: {archive_path}")
    sidecar_verified = _verify_sidecar(archive_path)
    resolved_config, config = _active_config(config_path)
    try:
        source = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ModelTransferError(f"Unsupported or corrupt ZIP archive: {error}") from error
    with source:
        manifest = _read_manifest(source, kind)
        model = manifest.get("model", {})
        if not isinstance(model, dict):
            raise ModelTransferError("Transfer manifest has invalid model metadata.")
        model = cast("dict[str, Any]", model)
        name = ""
        repo_id = ""
        if kind == "whisper":
            name = str(model.get("name", ""))
            if name not in MODEL_ARTIFACTS:
                raise ModelTransferError(
                    f"Unknown managed Whisper model '{name}'. Update meeting-notes first."
                )
            metadata = MODEL_ARTIFACTS[name]
            if (
                model.get("upstream_sha256") != metadata["sha256"]
                or model.get("filename") != f"ggml-{name}.bin"
                or len(manifest["files"]) != 1
            ):
                raise ModelTransferError("Whisper archive does not match the artifact catalog.")
            destination = project_cache_root(config) / "models" / f"ggml-{name}.bin"
        elif kind == "diarization":
            repo_id = str(model.get("repo_id", ""))
            if not repo_id or "/" not in repo_id:
                raise ModelTransferError("Diarization archive has no valid repository ID.")
            destination = managed_diarization_model_dir(config, repo_id)
        else:
            raise ModelTransferError(f"Unsupported model archive kind: {kind}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{kind}-restore-", dir=destination.parent
        ) as temporary:
            root = Path(temporary)
            _extract_verified(source, manifest, root)
            payload = root / "payload"
            if kind == "whisper":
                staged = payload / str(model["filename"])
                if not staged.is_file():
                    raise ModelTransferError("Whisper payload filename is invalid.")
                if (
                    staged.stat().st_size != int(MODEL_ARTIFACTS[name]["size"])  # type: ignore[arg-type]
                    or sha256_file(staged) != str(MODEL_ARTIFACTS[name]["sha256"])
                ):
                    raise ModelTransferError("Whisper payload failed upstream verification.")

                def update() -> None:
                    config.asr.model = name
                    config.asr.model_path = str(destination.resolve())
                    save_config(config, resolved_config)

            else:
                staged = payload
                if not (staged / "config.yaml").is_file():
                    raise ModelTransferError("Diarization payload is missing config.yaml.")
                install_manifest = {
                    "repo_id": repo_id,
                    "revision": model.get("revision", "unknown"),
                    "installed_at": datetime.now(UTC).isoformat(),
                    "authentication": "restored offline archive",
                }
                (staged / ".meeting-notes-manifest.json").write_text(
                    json.dumps(install_manifest, indent=2), encoding="utf-8"
                )

                def update() -> None:
                    config.diarization.backend = "pyannote"
                    config.diarization.enabled = True
                    config.diarization.model = repo_id
                    config.diarization.model_path = str(destination.resolve())
                    save_config(config, resolved_config)

            _atomic_install(staged, destination, update, force=force)
    if not sidecar_verified:
        print(
            f"Warning: no checksum sidecar found at {_archive_sidecar(archive_path)}; "
            "payload checksums were still verified."
        )
    return destination.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("whisper", "diarization"))
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument(
        "--compression-level",
        choices=("Optimal", "Fastest", "None"),
        default="Optimal",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "backup":
            if args.kind == "whisper":
                archive, sidecar = backup_whisper(
                    config_path=args.config,
                    model=args.model,
                    archive_path=args.archive,
                    compression_level=args.compression_level,
                    force=args.force,
                )
            else:
                if args.model:
                    raise ModelTransferError("--model is only valid for Whisper backup.")
                archive, sidecar = backup_diarization(
                    config_path=args.config,
                    archive_path=args.archive,
                    compression_level=args.compression_level,
                    force=args.force,
                )
            print(f"Archive created: {archive}")
            print(f"Checksum created: {sidecar}")
            print("Carry both files to the destination computer.")
            print(
                f"Restore with the matching *-windows.ps1 script -Action Restore "
                f'-Archive "{archive.name}"'
            )
        else:
            if args.archive is None:
                raise ModelTransferError("--archive is required for restore.")
            if args.model:
                raise ModelTransferError("--model is only valid for Whisper backup.")
            destination = restore_archive(
                args.kind,
                args.archive,
                config_path=args.config,
                force=args.force,
            )
            print(f"Model restored: {destination}")
            print("Verify with: uv run meeting-notes doctor")
        return 0
    except ModelTransferError as error:
        print(f"Model transfer failed: {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
