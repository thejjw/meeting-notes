"""Benchmark runner for comparing ASR configurations."""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import structlog
import yaml

from meeting_notes.asr.registry import get_configured_backend
from meeting_notes.audio.inspect import inspect_media
from meeting_notes.audio.normalize import normalize_audio
from meeting_notes.config import MeetingNotesConfig

log = structlog.get_logger()


@dataclass
class BenchmarkRun:
    """Result of a single benchmark run."""

    name: str
    backend: str
    model: str
    device: str
    audio_duration_seconds: float = 0.0
    transcription_seconds: float = 0.0
    model_load_seconds: float = 0.0
    real_time_factor: float = 0.0
    speed_multiple: float = 0.0
    segment_count: int = 0
    character_count: int = 0
    peak_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchmarkMatrix:
    """Benchmark configuration matrix."""

    runs: list[dict[str, Any]]


def load_benchmark_matrix(matrix_path: Path) -> BenchmarkMatrix:
    """Load benchmark matrix from YAML file."""
    data = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    return BenchmarkMatrix(runs=data.get("runs", []))


def run_single_benchmark(
    name: str,
    config_overrides: dict,
    audio_path: Path,
    model_path: Path | None = None,
) -> BenchmarkRun:
    """Run a single benchmark configuration."""
    result = BenchmarkRun(
        name=name,
        backend=config_overrides.get("runtime", {}).get("asr_backend", "whisper_cpp"),
        model=config_overrides.get("asr", {}).get("model", "medium"),
        device=config_overrides.get("runtime", {}).get("device", "cpu"),
    )

    try:
        # Inspect audio
        info = inspect_media(audio_path)
        result.audio_duration_seconds = info.duration_seconds

        # Normalize audio
        with tempfile.TemporaryDirectory(prefix="bench-") as tmp_dir:
            normalized = Path(tmp_dir) / "normalized.wav"
            normalize_audio(audio_path, normalized)

            config = MeetingNotesConfig(**config_overrides)
            if model_path is not None:
                config.asr.model_path = str(model_path)
            configured = get_configured_backend(config)
            readiness = configured.check_readiness()
            if not readiness.available:
                result.error = readiness.detail
                return result

            # Measure model load time
            load_start = time.perf_counter()

            # Run transcription
            transcribe_start = time.perf_counter()
            asr_result = configured.backend.transcribe(
                normalized,
                **configured.transcribe_kwargs,
            )
            transcribe_end = time.perf_counter()

            result.model_load_seconds = transcribe_start - load_start
            result.transcription_seconds = transcribe_end - transcribe_start
            result.segment_count = len(asr_result.segments)
            result.character_count = sum(len(s.text) for s in asr_result.segments)
            result.device = asr_result.device or result.device

            # Calculate metrics
            if result.transcription_seconds > 0:
                result.real_time_factor = (
                    result.transcription_seconds / result.audio_duration_seconds
                )
                result.speed_multiple = result.audio_duration_seconds / result.transcription_seconds

            # Peak RAM
            process = psutil.Process()
            result.peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    except Exception as e:
        result.error = str(e)
        log.error("benchmark.failed", name=name, error=str(e))

    return result


def run_benchmark_matrix(
    matrix: BenchmarkMatrix,
    audio_path: Path,
    model_dir: Path | None = None,
) -> list[BenchmarkRun]:
    """Run all configurations in the benchmark matrix."""
    results: list[BenchmarkRun] = []

    for run_config in matrix.runs:
        name = run_config.get("name", "unnamed")
        log.info("benchmark.starting", name=name)

        # Resolve model path
        model_path = None
        if model_dir:
            model_name = run_config.get("asr", {}).get("model", "medium")
            variant = (
                run_config.get("asr", {})
                .get("backend_options", {})
                .get("whisper_cpp", {})
                .get("model_variant", "fp16")
            )
            candidates = [
                model_dir / f"ggml-{model_name}-{variant}.bin",
                model_dir / f"ggml-{model_name}.bin",
            ]
            for c in candidates:
                if c.exists():
                    model_path = c
                    break

        result = run_single_benchmark(
            name=name,
            config_overrides=run_config,
            audio_path=audio_path,
            model_path=model_path,
        )
        results.append(result)

    return results


def render_benchmark_report(
    results: list[BenchmarkRun],
    output_dir: Path,
) -> dict[str, Path]:
    """Render benchmark results in JSON, CSV, and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}

    # JSON
    json_data = [
        {
            "name": r.name,
            "backend": r.backend,
            "model": r.model,
            "device": r.device,
            "audio_duration_seconds": r.audio_duration_seconds,
            "transcription_seconds": r.transcription_seconds,
            "model_load_seconds": r.model_load_seconds,
            "real_time_factor": r.real_time_factor,
            "speed_multiple": r.speed_multiple,
            "segment_count": r.segment_count,
            "character_count": r.character_count,
            "peak_ram_mb": r.peak_ram_mb,
            "error": r.error,
        }
        for r in results
    ]
    json_path = output_dir / "benchmark.json"
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    output_paths["json"] = json_path

    # CSV
    csv_lines = [
        "name,backend,model,device,audio_sec,transcribe_sec,rtf,speed_x,segments,chars,peak_ram_mb,error"
    ]
    for r in results:
        csv_lines.append(
            f"{r.name},{r.backend},{r.model},{r.device},"
            f"{r.audio_duration_seconds:.1f},{r.transcription_seconds:.1f},"
            f"{r.real_time_factor:.3f},{r.speed_multiple:.1f},"
            f"{r.segment_count},{r.character_count},{r.peak_ram_mb:.0f},"
            f"{r.error or ''}"
        )
    csv_path = output_dir / "benchmark.csv"
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    output_paths["csv"] = csv_path

    # Markdown
    md_lines = ["# Benchmark Results\n"]
    md_lines.append(
        "| Name | Backend | Model | Device | Audio | Transcribe | "
        "RTF | Speed | Segments | Peak RAM |"
    )
    md_lines.append(
        "|------|---------|-------|--------|-------|------------|"
        "-----|-------|----------|----------|"
    )
    for r in results:
        md_lines.append(
            f"| {r.name} | {r.backend} | {r.model} | {r.device} | "
            f"{r.audio_duration_seconds:.0f}s | {r.transcription_seconds:.0f}s | "
            f"{r.real_time_factor:.3f}x | {r.speed_multiple:.1f}x | "
            f"{r.segment_count} | {r.peak_ram_mb:.0f} MB |"
        )
        if r.error:
            md_lines.append(f"| **Error** | {r.error} | | | | | | | | |")

    md_path = output_dir / "benchmark.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    output_paths["markdown"] = md_path

    return output_paths
