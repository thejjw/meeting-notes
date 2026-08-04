"""Managed whisper.cpp runtime installation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from meeting_notes.artifacts import (
    CPU_ARCHIVES,
    WHISPER_CPP_REPOSITORY,
    WHISPER_CPP_REVISION,
    WHISPER_CPP_VERSION,
)


class RuntimeInstallError(RuntimeError):
    """A managed runtime could not be installed safely."""


@dataclass(frozen=True)
class RuntimeAsset:
    filename: str
    url: str
    sha256: str
    platform: str
    architecture: str


def normalize_platform(system: str | None = None) -> str:
    value = (system or platform.system()).lower()
    if value == "windows":
        return "windows"
    if value == "linux":
        return "linux"
    raise RuntimeInstallError(f"Unsupported platform '{value}'. Windows and Linux are supported.")


def normalize_architecture(machine: str | None = None) -> str:
    value = (machine or platform.machine()).lower()
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "i386": "x86",
        "i686": "x86",
        "x86": "x86",
        "aarch64": "arm64",
        "arm64": "arm64",
    }
    if value not in aliases:
        raise RuntimeInstallError(f"Unsupported architecture '{value}'.")
    return aliases[value]


def select_cpu_asset(
    version: str = WHISPER_CPP_VERSION,
    system: str | None = None,
    machine: str | None = None,
) -> RuntimeAsset:
    """Select the verified upstream CPU release archive."""
    os_name = normalize_platform(system)
    arch = normalize_architecture(machine)
    metadata = CPU_ARCHIVES.get((os_name, arch))
    if version != WHISPER_CPP_VERSION:
        raise RuntimeInstallError(
            f"No verified CPU archive metadata for {version}. "
            f"This build supports {WHISPER_CPP_VERSION}; update the artifact catalog first."
        )
    if metadata is None:
        raise RuntimeInstallError(f"No upstream CPU archive for {os_name}/{arch}.")
    filename, digest = metadata
    return RuntimeAsset(
        filename=filename,
        url=f"https://github.com/ggml-org/whisper.cpp/releases/download/{version}/{filename}",
        sha256=digest,
        platform=os_name,
        architecture=arch,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeInstallError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}. "
            "The unverified artifact was removed."
        )


def download_file(url: str, destination: Path) -> None:
    """Stream a URL to a temporary file, deleting partial data on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeInstallError(f"Unsafe archive member: {member}") from exc
    return target


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract zip/tar without allowing traversal or links."""
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                _safe_target(destination, member.filename)
                mode = member.external_attr >> 16
                if (mode & 0o170000) == 0o120000:
                    raise RuntimeInstallError(f"Archive symlink is not allowed: {member.filename}")
            source.extractall(destination)
        return
    try:
        with tarfile.open(archive, "r:*") as source:
            for member in source.getmembers():
                _safe_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeInstallError(f"Archive link is not allowed: {member.name}")
            source.extractall(destination, filter="data")
    except tarfile.TarError as exc:
        raise RuntimeInstallError(f"Unsupported or corrupt archive: {archive}") from exc


def runtime_dir(version: str, backend: str, *, cache_dir: Path) -> Path:
    return (
        cache_dir.resolve()
        / "runtimes"
        / version
        / f"{normalize_platform()}-{normalize_architecture()}-{backend}"
    )


def _find_executable(root: Path) -> Path:
    name = "whisper-cli.exe" if os.name == "nt" else "whisper-cli"
    matches = list(root.rglob(name))
    if not matches:
        raise RuntimeInstallError(f"{name} was not found in the installed artifact.")
    return matches[0]


def validate_executable(executable: Path) -> None:
    """Confirm the installed binary starts and exposes its CLI."""
    try:
        result = subprocess.run(
            [str(executable), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeInstallError(f"Installed executable is not runnable: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise RuntimeInstallError(
            f"Installed executable failed validation (exit {result.returncode}): {detail}"
        )


def _atomic_install(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    old = destination.with_name(destination.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if destination.exists():
        os.replace(destination, old)
    try:
        os.replace(staging, destination)
    except Exception:
        if old.exists() and not destination.exists():
            os.replace(old, destination)
        raise
    if old.exists():
        shutil.rmtree(old)


def _manifest(
    *,
    version: str,
    backend: str,
    executable: Path,
    checksum: str | None = None,
    build_flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": version,
        "platform": normalize_platform(),
        "architecture": normalize_architecture(),
        "backend": backend,
        "source_revision": WHISPER_CPP_REVISION if version == WHISPER_CPP_VERSION else version,
        "checksum": checksum,
        "build_flags": build_flags or [],
        "executable_path": str(executable.resolve()),
    }


def install_cpu(version: str = WHISPER_CPP_VERSION, *, cache_dir: Path) -> Path:
    asset = select_cpu_asset(version)
    destination = runtime_dir(version, "cpu", cache_dir=cache_dir)
    existing = load_manifest(destination)
    if existing and Path(str(existing["executable_path"])).is_file():
        executable = Path(str(existing["executable_path"]))
        validate_executable(executable)
        return executable
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runtime-", dir=destination.parent) as temp:
        temp_path = Path(temp)
        archive = temp_path / asset.filename
        download_file(asset.url, archive)
        try:
            verify_checksum(archive, asset.sha256)
        except Exception:
            archive.unlink(missing_ok=True)
            raise
        staging = temp_path / "install"
        safe_extract(archive, staging)
        executable = _find_executable(staging)
        validate_executable(executable)
        relative = executable.relative_to(staging)
        manifest = _manifest(
            version=version,
            backend="cpu",
            executable=destination / relative,
            checksum=asset.sha256,
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _atomic_install(staging, destination)
    return destination / relative


def vulkan_prerequisites() -> list[str]:
    missing = [name for name in ("git", "cmake") if shutil.which(name) is None]
    if os.name == "nt":
        if not (shutil.which("cl") or shutil.which("clang-cl") or shutil.which("g++")):
            missing.append("C++ compiler (Visual Studio C++ Build Tools)")
    elif not (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")):
        missing.append("C++ compiler")
    if shutil.which("vulkaninfo") is None and not os.environ.get("VULKAN_SDK"):
        missing.append("Vulkan SDK/development tooling (vulkaninfo or VULKAN_SDK)")
    return missing


def build_commands(
    source: Path, build: Path, install: Path, version: str = WHISPER_CPP_VERSION
) -> list[list[str]]:
    """Return deterministic source checkout/build commands for tests and scripts."""
    return [
        ["git", "clone", "--branch", version, "--depth", "1", WHISPER_CPP_REPOSITORY, str(source)],
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_VULKAN=1",
            f"-DCMAKE_INSTALL_PREFIX={install}",
        ],
        ["cmake", "--build", str(build), "--config", "Release", "--parallel"],
        ["cmake", "--install", str(build), "--config", "Release"],
    ]


def install_vulkan(version: str = WHISPER_CPP_VERSION, *, cache_dir: Path) -> Path:
    missing = vulkan_prerequisites()
    if missing:
        raise RuntimeInstallError(
            "Vulkan build prerequisites missing: "
            + ", ".join(missing)
            + ". Install system developer tools, then rerun this command."
        )
    destination = runtime_dir(version, "vulkan", cache_dir=cache_dir)
    existing = load_manifest(destination)
    if existing and Path(str(existing["executable_path"])).is_file():
        executable = Path(str(existing["executable_path"]))
        validate_executable(executable)
        return executable
    destination.parent.mkdir(parents=True, exist_ok=True)
    log_path = destination.parent / f"{destination.name}-build.log"
    with tempfile.TemporaryDirectory(prefix="vulkan-build-", dir=destination.parent) as temp:
        root = Path(temp)
        source, build, staging = root / "source", root / "build", root / "install"
        commands = build_commands(source, build, staging, version)
        with log_path.open("w", encoding="utf-8") as log:
            for command in commands:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
                )
                if result.returncode:
                    raise RuntimeInstallError(
                        f"Vulkan build failed while running {' '.join(command)}. "
                        f"Build log retained at {log_path}."
                    )
        executable = _find_executable(staging)
        validate_executable(executable)
        relative = executable.relative_to(staging)
        manifest = _manifest(
            version=version,
            backend="vulkan",
            executable=destination / relative,
            build_flags=["-DGGML_VULKAN=1"],
        )
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _atomic_install(staging, destination)
    return destination / relative


def load_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return cast("dict[str, Any]", value) if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def find_manifest_for_executable(
    executable: Path, *, cache_dir: Path
) -> dict[str, Any] | None:
    resolved = executable.resolve()
    for parent in (resolved.parent, *resolved.parents):
        manifest = load_manifest(parent)
        if manifest is not None:
            return manifest
        if parent == cache_dir.resolve():
            break
    return None


def installed_runtimes(*, cache_dir: Path) -> list[dict[str, Any]]:
    root = cache_dir.resolve() / "runtimes"
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for manifest_path in root.rglob("manifest.json"):
        manifest = load_manifest(manifest_path.parent)
        if manifest:
            manifest["healthy"] = Path(str(manifest.get("executable_path", ""))).is_file()
            found.append(manifest)
    return found
