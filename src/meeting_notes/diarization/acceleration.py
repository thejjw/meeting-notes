"""Project-local managed runtime for AMD ROCm hybrid diarization."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from meeting_notes.storage import directory_size, project_cache_root

if TYPE_CHECKING:
    from meeting_notes.config import MeetingNotesConfig

ROCM_VERSION = "7.2.1"
ROCM_TORCH_VERSION = "2.9.1+rocm7.2.1"
ROCM_PYANNOTE_VERSION = "4.0.7"
ROCM_RUNTIME_NAME = "rocm-7.2.1-py312"
ROCM_DOWNLOAD_MIB = 2100
ROCM_INSTALLED_MIB = 6220
ROCM_PEAK_FREE_MIB = 9216

_BASE_URL = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
ROCM_PACKAGES = (
    (
        f"{_BASE_URL}/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
        "f68989d48df71cbfc3cb68bf705dc37c0f56e9666feddb59a1a0f5ff7539fe1c",
    ),
    (
        f"{_BASE_URL}/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
        "19e6ee67e13432b7c1e8a4077df795dcb1239546ae314700c4b4d97e5b4b8f63",
    ),
    (
        f"{_BASE_URL}/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
        "c7fe0b0731af8896093ff69e11496830d3cb6a4aed73e895c60b7cbdc200be92",
    ),
    (
        f"{_BASE_URL}/rocm-7.2.1.tar.gz",
        "9084902eaa69213a00a90784ad89e6e5fe73c702df0cc6cc3a70d777c7a6142b",
    ),
    (
        f"{_BASE_URL}/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
        "e88bf270163b48f7f27f7ea3db5ffb3be4ba107301933022bcb3c6ddedfeeabb",
    ),
    (
        f"{_BASE_URL}/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
        "62835605995c7812a3224733abf08e222afddb7a487c493521e613ca46e578ab",
    ),
)


class RocmRuntimeError(RuntimeError):
    """The managed ROCm diarization runtime could not be used."""


@dataclass(frozen=True)
class RocmProbe:
    """ROCm host/runtime readiness returned to configure and doctor."""

    state: Literal["unsupported", "prerequisites-missing", "eligible", "ready", "broken"]
    detail: str
    hip_info_path: str | None = None
    architecture: str | None = None
    device_name: str | None = None
    runtime_path: str | None = None
    torch_version: str | None = None
    hip_version: str | None = None
    pyannote_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diarization_cache_root(config: MeetingNotesConfig) -> Path:
    """Return the absolute project-local managed diarization directory."""
    return project_cache_root(config) / "diarization"


def model_dir(config: MeetingNotesConfig, repo_id: str) -> Path:
    safe_name = repo_id.replace("/", "--")
    return diarization_cache_root(config) / "models" / safe_name


def default_runtime_dir(config: MeetingNotesConfig) -> Path:
    return diarization_cache_root(config) / "runtimes" / ROCM_RUNTIME_NAME


def runtime_python(runtime: Path) -> Path:
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_environment(runtime: Path) -> dict[str, str]:
    """Build an environment that cannot leak packages from the main project venv."""
    runtime = runtime.resolve()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["VIRTUAL_ENV"] = str(runtime)
    scripts = runtime / ("Scripts" if os.name == "nt" else "bin")
    env["PATH"] = os.pathsep.join((str(scripts), env.get("PATH", "")))
    return env


def _hip_info_candidates() -> list[Path]:
    candidates: list[Path] = []
    found = shutil.which("hipInfo")
    if found:
        candidates.append(Path(found))
    if os.name == "nt":
        base = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "AMD" / "ROCm"
        if base.is_dir():
            candidates.extend(sorted(base.glob("*/bin/hipInfo.exe"), reverse=True))
    return list(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))


def _host_hip() -> tuple[Path | None, str | None, str | None]:
    for executable in _hip_info_candidates():
        try:
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode:
            continue
        architecture = None
        device_name = None
        for line in result.stdout.splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "Name" and not device_name:
                device_name = value.strip()
            if key.strip() == "gcnArchName":
                architecture = value.strip().split(":", 1)[0]
        if architecture:
            return executable, architecture, device_name
    return None, None, None


def validate_runtime(runtime: Path) -> dict[str, object]:
    """Validate PyTorch/HIP in a managed runtime and return its identity."""
    python = runtime_python(runtime)
    if not python.is_file():
        raise RocmRuntimeError(f"Managed ROCm Python is missing: {python}")
    script = (
        "import importlib.metadata as m, json, torch; "
        "ok=bool(torch.cuda.is_available() and torch.version.hip); "
        "print(json.dumps({'available':ok,'torch':torch.__version__,"
        "'hip':torch.version.hip,'pyannote':m.version('pyannote.audio'),"
        "'device':torch.cuda.get_device_name(0) if ok else None}))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=runtime_environment(runtime),
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RocmRuntimeError(f"ROCm PyTorch validation failed: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise RocmRuntimeError(f"ROCm PyTorch validation failed: {detail}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as error:
        raise RocmRuntimeError("ROCm PyTorch validation returned invalid output.") from error
    if not payload.get("available"):
        raise RocmRuntimeError("PyTorch cannot access an AMD HIP device.")
    if payload.get("torch") != ROCM_TORCH_VERSION:
        raise RocmRuntimeError(
            f"Expected torch {ROCM_TORCH_VERSION}, found {payload.get('torch')}."
        )
    if payload.get("pyannote") != ROCM_PYANNOTE_VERSION:
        raise RocmRuntimeError(
            f"Expected pyannote.audio {ROCM_PYANNOTE_VERSION}, "
            f"found {payload.get('pyannote')}."
        )
    return payload


def probe_rocm(config: MeetingNotesConfig) -> RocmProbe:
    """Probe host prerequisites and the configured/default managed runtime."""
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        return RocmProbe("unsupported", "Automated ROCm diarization supports Windows x64 only.")
    if sys.version_info[:2] != (3, 12):
        return RocmProbe("unsupported", "ROCm diarization requires Python 3.12.")

    hip_path, architecture, device_name = _host_hip()
    runtime = (
        Path(config.diarization.rocm_gpu_runtime_path).expanduser().resolve()
        if config.diarization.rocm_gpu_runtime_path
        else default_runtime_dir(config)
    )
    if runtime_python(runtime).is_file():
        try:
            identity = validate_runtime(runtime)
        except RocmRuntimeError as error:
            return RocmProbe(
                "broken",
                str(error),
                str(hip_path) if hip_path else None,
                architecture,
                device_name,
                str(runtime),
            )
        return RocmProbe(
            "ready",
            "Managed ROCm PyTorch can access the AMD GPU.",
            str(hip_path) if hip_path else None,
            architecture,
            str(identity.get("device") or device_name or "AMD GPU"),
            str(runtime),
            str(identity.get("torch")),
            str(identity.get("hip")),
            str(identity.get("pyannote")),
        )

    if not hip_path or not architecture:
        return RocmProbe(
            "prerequisites-missing",
            "AMD HIP 7.2 prerequisites were not detected; install the supported "
            "AMD Windows driver/HIP package first.",
            runtime_path=str(runtime),
        )
    return RocmProbe(
        "eligible",
        "AMD HIP is available; the project-local ROCm runtime can be provisioned.",
        str(hip_path),
        architecture,
        device_name,
        str(runtime),
    )


def provision_runtime(config: MeetingNotesConfig, *, force: bool = False) -> Path:
    """Install and validate the pinned ROCm environment atomically."""
    probe = probe_rocm(config)
    if probe.state in {"unsupported", "prerequisites-missing"}:
        raise RocmRuntimeError(probe.detail)
    destination = default_runtime_dir(config)
    if destination.exists() and not force:
        validate_runtime(destination)
        return destination.resolve()

    uv = shutil.which("uv")
    if not uv:
        raise RocmRuntimeError("uv is required to provision the managed ROCm runtime.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rocm-runtime-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "install"
        env = dict(os.environ)
        env["UV_NO_CACHE"] = "1"
        commands = [
            [uv, "venv", str(staging), "--python", "3.12"],
            [
                uv,
                "pip",
                "install",
                "--python",
                str(runtime_python(staging)),
                "--no-cache",
                *[f"{url}#sha256={digest}" for url, digest in ROCM_PACKAGES],
            ],
            [
                uv,
                "pip",
                "install",
                "--python",
                str(runtime_python(staging)),
                "--no-cache",
                f"pyannote.audio=={ROCM_PYANNOTE_VERSION}",
            ],
        ]
        for command in commands:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                check=False,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-2000:]
                raise RocmRuntimeError(f"ROCm runtime provisioning failed: {detail}")
        identity = validate_runtime(staging)
        manifest = {
            "version": 1,
            "runtime": ROCM_RUNTIME_NAME,
            "rocm": ROCM_VERSION,
            "torch": identity.get("torch"),
            "hip": identity.get("hip"),
            "device": identity.get("device"),
            "pyannote_audio": ROCM_PYANNOTE_VERSION,
            "installed_at": datetime.now(UTC).isoformat(),
        }
        (staging / ".meeting-notes-runtime.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        backup = destination.with_name(f".{destination.name}.old")
        if backup.exists():
            shutil.rmtree(backup)
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except Exception:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    validate_runtime(destination)
    return destination.resolve()


def remove_runtime(runtime: Path) -> int:
    """Remove one explicitly selected managed ROCm runtime."""
    runtime = runtime.resolve()
    size = directory_size(runtime)
    if runtime.exists():
        shutil.rmtree(runtime)
    return size
