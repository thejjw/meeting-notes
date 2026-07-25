"""Resource catalog, system probes, and requirement comparison."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import structlog

log = structlog.get_logger()


@dataclass
class CPUDetection:
    """Detected CPU information."""

    model_name: str = "unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    supports_avx: bool = False
    supports_avx2: bool = False
    supports_avx512: bool = False
    supports_neon: bool = False


@dataclass
class MemoryDetection:
    """Detected memory information."""

    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    is_unified_memory: bool = False
    notes: str = ""


@dataclass
class GPUDetection:
    """Detected GPU information."""

    available: bool = False
    backend: str = "none"
    device_name: str = ""
    driver_version: str = ""
    vram_gb: float = 0.0
    free_vram_gb: float = 0.0
    vulkan_devices: list[dict[str, Any]] = field(default_factory=list)
    rocm_architectures: list[str] = field(default_factory=list)
    cuda_available: bool = False
    notes: str = ""


@dataclass
class ToolDetection:
    """Detected external tool information."""

    ffmpeg_version: str = ""
    ffmpeg_available: bool = False
    ffprobe_version: str = ""
    ffprobe_available: bool = False
    whisper_cpp_version: str = ""
    whisper_cpp_available: bool = False
    codex_version: str = ""
    codex_available: bool = False
    git_version: str = ""
    git_available: bool = False


@dataclass
class SystemDiagnostics:
    """Complete system diagnostic report."""

    os_name: str = ""
    os_version: str = ""
    architecture: str = ""
    python_version: str = ""
    is_wsl: bool = False
    cpu: CPUDetection = field(default_factory=CPUDetection)
    memory: MemoryDetection = field(default_factory=MemoryDetection)
    gpu: GPUDetection = field(default_factory=GPUDetection)
    tools: ToolDetection = field(default_factory=ToolDetection)
    platform_label: str = "auto"


def _run_command(cmd: list[str], timeout: float = 10.0) -> tuple[bool, str]:
    """Run a command and return (success, stdout). Never raises."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return result.returncode == 0, result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False, ""


def _detect_cpu() -> CPUDetection:
    """Detect CPU model and capabilities."""
    det = CPUDetection()
    det.physical_cores = psutil.cpu_count(logical=False) or 0
    det.logical_cores = psutil.cpu_count(logical=True) or 0

    # Platform-specific CPU name
    if os.name == "nt":
        import winreg

        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            det.model_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            winreg.CloseKey(key)
        except Exception:
            det.model_name = platform.processor() or "unknown"
    else:
        # Linux: try /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        det.model_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            det.model_name = platform.processor() or "unknown"

    # Instruction set detection (simplified)
    model_lower = det.model_name.lower()
    if "ryzen" in model_lower or "epyc" in model_lower or "intel" in model_lower:
        det.supports_avx = True
        det.supports_avx2 = True
    if "512" in model_lower or "xeon" in model_lower:
        det.supports_avx512 = True

    return det


def _detect_memory() -> MemoryDetection:
    """Detect system memory."""
    mem = psutil.virtual_memory()
    det = MemoryDetection()
    det.total_ram_gb = round(mem.total / (1024**3), 1)
    det.available_ram_gb = round(mem.available / (1024**3), 1)
    return det


def _detect_wsl() -> bool:
    """Detect if running inside WSL2."""
    if os.name == "nt":
        return False
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as f:
            version = f.read().lower()
            return "microsoft" in version or "wsl" in version
    except Exception:
        return False


def _detect_vulkan() -> list[dict[str, Any]]:
    """Try to detect Vulkan devices via vulkaninfo."""
    ok, out = _run_command(["vulkaninfo", "--summary"])
    if not ok:
        return []

    devices: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("GPU"):
            if current:
                devices.append(current)
            current = {"name": "", "type": ""}
        elif "devicename" in line.lower():
            current["name"] = line.split("=", 1)[-1].strip() if "=" in line else line
        elif "devicetype" in line.lower():
            current["type"] = line.split("=", 1)[-1].strip() if "=" in line else line
    if current:
        devices.append(current)
    return devices


def _detect_rocm() -> list[str]:
    """Try to detect ROCm/HIP architectures via rocminfo."""
    ok, out = _run_command(["rocminfo"])
    if not ok:
        return []

    archs: list[str] = []
    for line in out.splitlines():
        line = line.strip().lower()
        if "gfx" in line:
            # Extract gfx architecture name
            for part in line.split():
                if part.startswith("gfx"):
                    archs.append(part)
                    break
    return list(set(archs))


def _detect_cuda() -> tuple[bool, str]:
    """Try to detect NVIDIA CUDA."""
    ok, out = _run_command(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"])
    return ok, out


def _detect_tool_version(cmd: list[str]) -> tuple[bool, str]:
    """Detect tool version from --version or -version."""
    ok, out = _run_command(cmd + ["--version"], timeout=5.0)
    if ok and out:
        return True, out.splitlines()[0]
    ok, out = _run_command(cmd + ["-version"], timeout=5.0)
    if ok and out:
        return True, out.splitlines()[0]
    return False, ""


def detect_system() -> SystemDiagnostics:
    """Run full system diagnostics."""
    diag = SystemDiagnostics()
    diag.os_name = platform.system()
    diag.os_version = platform.version()
    diag.architecture = platform.machine()
    diag.python_version = platform.python_version()
    diag.is_wsl = _detect_wsl()
    diag.platform_label = "wsl" if diag.is_wsl else diag.os_name.lower()

    diag.cpu = _detect_cpu()
    diag.memory = _detect_memory()

    # GPU detection
    vulkan_devices = _detect_vulkan()
    rocm_archs = _detect_rocm()
    cuda_ok, cuda_info = _detect_cuda()

    diag.gpu.vulkan_devices = vulkan_devices
    diag.gpu.rocm_architectures = rocm_archs
    diag.gpu.cuda_available = cuda_ok

    if cuda_ok and cuda_info:
        diag.gpu.available = True
        diag.gpu.backend = "cuda"
        parts = cuda_info.split(", ")
        if parts:
            diag.gpu.device_name = parts[0]
        if len(parts) >= 2:
            try:
                diag.gpu.vram_gb = float(parts[1].replace("MiB", "").strip()) / 1024
            except ValueError:
                pass
        if len(parts) >= 3:
            try:
                diag.gpu.free_vram_gb = float(parts[2].replace("MiB", "").strip()) / 1024
            except ValueError:
                pass
        if len(parts) >= 4:
            diag.gpu.driver_version = parts[3]
    elif vulkan_devices:
        diag.gpu.available = True
        diag.gpu.backend = "vulkan"
        diag.gpu.device_name = vulkan_devices[0].get("name", "unknown")
    elif rocm_archs:
        diag.gpu.available = True
        diag.gpu.backend = "rocm"
        diag.gpu.rocm_architectures = rocm_archs

    # Unified memory heuristic: if AMD APU with no discrete GPU
    if "ryzen" in diag.cpu.model_name.lower() and not cuda_ok:
        diag.memory.is_unified_memory = True
        diag.memory.notes = "Unified memory detected (AMD APU). RAM and VRAM are not separate pools."

    # Tool detection
    ff_ok, ff_ver = _detect_tool_version(["ffmpeg"])
    diag.tools.ffmpeg_available = ff_ok
    diag.tools.ffmpeg_version = ff_ver

    fp_ok, fp_ver = _detect_tool_version(["ffprobe"])
    diag.tools.ffprobe_available = fp_ok
    diag.tools.ffprobe_version = fp_ver

    wc_ok, wc_ver = _detect_tool_version(["whisper-cli"])
    diag.tools.whisper_cpp_available = wc_ok
    diag.tools.whisper_cpp_version = wc_ver

    cx_ok, cx_ver = _detect_tool_version(["codex"])
    diag.tools.codex_available = cx_ok
    diag.tools.codex_version = cx_ver

    g_ok, g_ver = _detect_tool_version(["git"])
    diag.tools.git_available = g_ok
    diag.tools.git_version = g_ver

    return diag


@dataclass
class ResourceEstimate:
    """Memory/resource estimate for a model configuration."""

    model_name: str
    backend: str
    disk_size_mib: int = 0
    reference_memory_mb: int = 0
    recommended_free_ram_gb: float = 0.0
    recommended_free_vram_gb: float = 0.0
    confidence: str = "estimated"
    source: str = ""
    notes: str = ""


# Official reference data for whisper models
WHISPER_CPP_RESOURCES: dict[str, ResourceEstimate] = {
    "tiny": ResourceEstimate(
        model_name="tiny",
        backend="whisper_cpp",
        disk_size_mib=75,
        reference_memory_mb=273,
        recommended_free_ram_gb=2.3,
        confidence="official_reference",
        source="https://github.com/ggerganov/whisper.cpp",
    ),
    "base": ResourceEstimate(
        model_name="base",
        backend="whisper_cpp",
        disk_size_mib=142,
        reference_memory_mb=388,
        recommended_free_ram_gb=2.4,
        confidence="official_reference",
        source="https://github.com/ggerganov/whisper.cpp",
    ),
    "small": ResourceEstimate(
        model_name="small",
        backend="whisper_cpp",
        disk_size_mib=466,
        reference_memory_mb=852,
        recommended_free_ram_gb=2.9,
        confidence="official_reference",
        source="https://github.com/ggerganov/whisper.cpp",
    ),
    "medium": ResourceEstimate(
        model_name="medium",
        backend="whisper_cpp",
        disk_size_mib=1536,
        reference_memory_mb=2150,
        recommended_free_ram_gb=4.1,
        confidence="official_reference",
        source="https://github.com/ggerganov/whisper.cpp",
    ),
    "large-v3": ResourceEstimate(
        model_name="large-v3",
        backend="whisper_cpp",
        disk_size_mib=2960,
        reference_memory_mb=3900,
        recommended_free_ram_gb=5.9,
        confidence="official_reference",
        source="https://github.com/ggerganov/whisper.cpp",
    ),
    "large-v3-turbo": ResourceEstimate(
        model_name="large-v3-turbo",
        backend="whisper_cpp",
        disk_size_mib=1536,
        reference_memory_mb=2150,
        recommended_free_ram_gb=4.1,
        confidence="estimated",
        source="Estimated from turbo model architecture",
    ),
}


OPENAI_WHISPER_RESOURCES: dict[str, ResourceEstimate] = {
    "tiny": ResourceEstimate(
        model_name="tiny",
        backend="openai_whisper",
        disk_size_mib=75,
        reference_memory_mb=1000,
        recommended_free_ram_gb=2.0,
        confidence="official_reference",
        source="https://github.com/openai/whisper",
    ),
    "base": ResourceEstimate(
        model_name="base",
        backend="openai_whisper",
        disk_size_mib=142,
        reference_memory_mb=1000,
        recommended_free_ram_gb=2.0,
        confidence="official_reference",
        source="https://github.com/openai/whisper",
    ),
    "small": ResourceEstimate(
        model_name="small",
        backend="openai_whisper",
        disk_size_mib=466,
        reference_memory_mb=2000,
        recommended_free_ram_gb=3.0,
        confidence="official_reference",
        source="https://github.com/openai/whisper",
    ),
    "medium": ResourceEstimate(
        model_name="medium",
        backend="openai_whisper",
        disk_size_mib=1536,
        reference_memory_mb=5000,
        recommended_free_ram_gb=7.0,
        confidence="official_reference",
        source="https://github.com/openai/whisper",
    ),
    "large-v3": ResourceEstimate(
        model_name="large-v3",
        backend="openai_whisper",
        disk_size_mib=2960,
        reference_memory_mb=10000,
        recommended_free_ram_gb=12.0,
        confidence="official_reference",
        source="https://github.com/openai/whisper",
    ),
}


def get_resource_estimate(model: str, backend: str = "whisper_cpp") -> ResourceEstimate | None:
    """Look up resource estimate for a model/backend combination."""
    catalog = WHISPER_CPP_RESOURCES if backend == "whisper_cpp" else OPENAI_WHISPER_RESOURCES
    return catalog.get(model)


def check_model_fit(
    estimate: ResourceEstimate, diag: SystemDiagnostics
) -> tuple[str, str]:
    """Check if a model fits on the detected machine.

    Returns (status, reason) where status is one of:
    'available', 'available_with_warning', 'not_detected', 'incompatible', 'unknown'
    """
    if not estimate:
        return "unknown", "No resource estimate available for this model."

    # Check RAM
    if diag.memory.available_ram_gb < estimate.recommended_free_ram_gb:
        if diag.memory.available_ram_gb >= estimate.reference_memory_mb / 1024:
            return (
                "available_with_warning",
                f"Minimum RAM appears sufficient ({diag.memory.available_ram_gb:.1f} GB available) "
                f"but recommended headroom ({estimate.recommended_free_ram_gb:.1f} GB) is not met.",
            )
        return (
            "not_detected",
            f"Insufficient RAM: {diag.memory.available_ram_gb:.1f} GB available, "
            f"need at least {estimate.reference_memory_mb / 1024:.1f} GB for model.",
        )

    return "available", f"Model should fit: {diag.memory.available_ram_gb:.1f} GB available, need {estimate.recommended_free_ram_gb:.1f} GB recommended."


def format_diagnostics_table(diag: SystemDiagnostics) -> str:
    """Format diagnostics as a human-readable table."""
    lines = ["Detected system", ""]

    # OS
    os_str = f"{diag.os_name} {diag.architecture}"
    if diag.is_wsl:
        os_str += " (WSL)"
    lines.append(f"  OS: {os_str}")

    # CPU
    lines.append(f"  CPU: {diag.cpu.model_name}, {diag.cpu.physical_cores} cores / {diag.cpu.logical_cores} threads")

    # RAM
    lines.append(
        f"  System RAM: {diag.memory.total_ram_gb:.1f} GiB total, "
        f"{diag.memory.available_ram_gb:.1f} GiB currently available"
    )
    if diag.memory.is_unified_memory:
        lines.append(f"  Memory type: Unified (RAM and VRAM are the same pool)")

    # GPU
    if diag.gpu.cuda_available:
        lines.append(f"  CUDA: available, {diag.gpu.device_name}, {diag.gpu.vram_gb:.1f} GB")
    elif diag.gpu.vulkan_devices:
        dev_names = ", ".join(d.get("name", "?") for d in diag.gpu.vulkan_devices)
        lines.append(f"  Vulkan: available, {dev_names}")
    else:
        lines.append(f"  Vulkan: not available in this environment")

    if diag.gpu.rocm_architectures:
        lines.append(f"  ROCm/HIP: detected architectures: {', '.join(diag.gpu.rocm_architectures)}")
    else:
        lines.append(f"  ROCm/HIP: not available in this environment")

    if not diag.gpu.cuda_available and not diag.gpu.vulkan_devices and not diag.gpu.rocm_architectures:
        lines.append(f"  CUDA: not available")

    # Tools
    lines.append("")
    lines.append("External tools")
    lines.append(f"  FFmpeg: {'available' if diag.tools.ffmpeg_available else 'not found'} {diag.tools.ffmpeg_version}")
    lines.append(f"  FFprobe: {'available' if diag.tools.ffprobe_available else 'not found'} {diag.tools.ffprobe_version}")
    lines.append(f"  whisper.cpp on PATH: {'available' if diag.tools.whisper_cpp_available else 'not found'} {diag.tools.whisper_cpp_version}")
    lines.append(f"  Codex CLI: {'available' if diag.tools.codex_available else 'not found'} {diag.tools.codex_version}")

    return "\n".join(lines)
