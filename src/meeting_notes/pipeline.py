"""Pipeline orchestration — wires together all processing stages."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from meeting_notes.asr.base import ASRResult, ASRSegment
from meeting_notes.asr.registry import get_configured_backend
from meeting_notes.audio.chunk import (
    AudioChunk,
    compute_chunks,
    materialize_audio_chunks,
    save_chunks_manifest,
)
from meeting_notes.audio.inspect import inspect_media
from meeting_notes.audio.normalize import create_normalized_path, normalize_audio
from meeting_notes.config import DiarizationConfig, MeetingNotesConfig, load_config
from meeting_notes.errors import (
    ConfigurationError,
    DependencyMissingError,
    DiarizationUnavailableError,
    StageCancelledError,
)
from meeting_notes.jobs import (
    create_job_dir,
    load_manifest,
    make_job_slug,
    save_manifest,
    update_stage_status,
)
from meeting_notes.publication import (
    managed_files,
    publication_paths,
    render_transcript_variants,
    write_run_report,
)
from meeting_notes.storage import project_cache_root
from meeting_notes.timing import build_time_estimate_lines
from meeting_notes.transcript.render import render_all_formats

log = structlog.get_logger()
console = Console(stderr=True)


def _load_or_fail(config_path: str | None) -> MeetingNotesConfig:
    """Load config or print actionable error."""
    try:
        return load_config(config_path)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1) from e


def _check_tools(config: MeetingNotesConfig) -> None:
    """Verify required external tools are available."""
    from meeting_notes.subprocess_utils import run_command

    for tool_name, tool_path in [
        ("FFmpeg", config.runtime.ffmpeg_path),
        ("FFprobe", config.runtime.ffprobe_path),
    ]:
        try:
            result = run_command([tool_path, "-version"], timeout=5.0, label=f"check-{tool_name}")
            if not result.success:
                console.print(f"[yellow]Warning: {tool_name} may not be available[/yellow]")
        except RuntimeError as exc:
            console.print(f"[red]{tool_name} not found at '{tool_path}'. Install FFmpeg.[/red]")
            raise typer.Exit(1) from exc


def _active_config_path(config_path: str | None) -> Path:
    """Return the active config path for copy/paste remediation commands."""
    from meeting_notes.config import _resolve_config_path

    resolved = _resolve_config_path(config_path)
    return (resolved or Path(config_path or "meeting-notes.yaml")).resolve()


def _asr_remediation(
    config: MeetingNotesConfig,
    config_path: str | None,
    *,
    runtime_ready: bool = False,
) -> str:
    """Build exact commands that make the configured whisper.cpp backend runnable."""
    from meeting_notes.models import verify_model
    from meeting_notes.runtime import installed_runtimes, vulkan_prerequisites

    active_config = _active_config_path(config_path)
    quoted_config = f'"{active_config}"'
    lines = [
        "",
        "[bold]How to finish setup[/bold]",
        f"Active config: {active_config}",
    ]

    runtimes = installed_runtimes(cache_dir=project_cache_root(config))
    if not runtime_ready:
        matching = [
            item
            for item in runtimes
            if item.get("backend") == config.runtime.device and item.get("healthy")
        ]
        if matching:
            lines.extend(
                [
                    f"A managed {config.runtime.device} runtime is already installed but is not "
                    "selected by this config.",
                    "Run:",
                ]
            )
        elif config.runtime.device == "vulkan":
            missing = vulkan_prerequisites()
            if missing:
                lines.append("The selected Vulkan backend is missing build prerequisites:")
                lines.extend(f"  - {item}" for item in missing)
                lines.append(
                    "Install CMake and Visual Studio C++ Build Tools, then open a Developer "
                    "PowerShell and run:"
                )
            else:
                lines.append("Build and select the configured Vulkan runtime:")
        else:
            lines.append("Install and select the configured CPU runtime:")
        lines.append(
            "  uv run meeting-notes runtime install "
            f"--device {config.runtime.device} --config {quoted_config} --yes"
        )

    cpu_ready = any(item.get("backend") == "cpu" and item.get("healthy") for item in runtimes)
    if not runtime_ready and config.runtime.device == "vulkan" and cpu_ready:
        lines.extend(
            [
                "",
                "Or use the already-installed CPU runtime (this explicitly changes the config; "
                "there is no silent fallback):",
                "  uv run meeting-notes runtime install "
                f"--device cpu --config {quoted_config} --yes",
            ]
        )

    model_path = Path(config.asr.model_path) if config.asr.model_path else None
    model_ready = False
    if model_path:
        try:
            model_ready, _ = verify_model(config.asr.model, model_path)
        except RuntimeError:
            model_ready = model_path.is_file()
    if not model_ready:
        lines.extend(
            [
                "",
                f"The configured model '{config.asr.model}' is also not ready. Run:",
                "  uv run meeting-notes models download "
                f"{config.asr.model} --config {quoted_config} --yes",
            ]
        )

    lines.extend(
        [
            "",
            "Then verify everything before retrying the recording:",
            f"  uv run meeting-notes doctor --config {quoted_config}",
        ]
    )
    return "\n".join(lines)


def _check_asr_readiness(config: MeetingNotesConfig, config_path: str | None) -> None:
    """Fail before audio preparation with backend-specific remediation."""
    configured = get_configured_backend(config)
    readiness = configured.check_readiness()
    if readiness.available:
        return
    console.print(f"[red]Configured ASR backend is unavailable:[/red] {readiness.detail}")
    if config.runtime.asr_backend == "lemonade":
        options = config.asr.backend_options.lemonade
        active_config = _active_config_path(config_path)
        console.print(
            "\n[bold]How to finish setup[/bold]\n"
            f"  Configured URL: {options.base_url}\n"
            "  Start Lemonade Server manually, then verify it with:\n"
            "    lemonade status\n"
            "  If the model is not downloaded, run:\n"
            "    uv run meeting-notes models download "
            f'{config.asr.model} --config "{active_config}" --yes\n'
            "  Then verify:\n"
            f'    uv run meeting-notes doctor --config "{active_config}"'
        )
    elif config.runtime.asr_backend == "whisper_cpp":
        console.print(_asr_remediation(config, config_path, runtime_ready=False))
    raise typer.Exit(1)


def _apply_speaker_overrides(
    config: MeetingNotesConfig,
    *,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> MeetingNotesConfig:
    """Apply invocation-only speaker-count overrides to a validated config copy."""
    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        raise typer.BadParameter(
            "--num-speakers cannot be combined with --min-speakers or --max-speakers"
        )

    data = config.diarization.model_dump()
    if num_speakers is not None:
        data["num_speakers"] = num_speakers
    elif min_speakers is not None or max_speakers is not None:
        # A per-run range intentionally replaces any configured exact count.
        data["num_speakers"] = None
        if min_speakers is not None:
            data["min_speakers"] = min_speakers
        if max_speakers is not None:
            data["max_speakers"] = max_speakers

    try:
        diarization = DiarizationConfig.model_validate(data)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid speaker-count overrides: {exc}") from exc
    return config.model_copy(update={"diarization": diarization})


def _speaker_policy_description(config: DiarizationConfig) -> str:
    """Return a concise description of the effective speaker-count policy."""
    if config.num_speakers is not None:
        return f"exactly {config.num_speakers}"
    maximum = str(config.max_speakers) if config.max_speakers is not None else "unbounded"
    return f"automatic (min={config.min_speakers}, max={maximum})"


def run_pipeline(
    input_file: str,
    config_path: str | None = None,
    profile: str | None = None,
    dry_run: bool = False,
    resume: bool = True,
    from_stage: str | None = None,
    force_stage: str | None = None,
    finalize_names: bool = True,
    local_only: bool = False,
    copy_to_input: bool = False,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> None:
    """Process an audio/video file into meeting notes."""
    config = _load_or_fail(config_path)
    config = _apply_speaker_overrides(
        config,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    _check_tools(config)

    source = Path(input_file)
    if not source.exists():
        console.print(f"[red]File not found:[/red] {input_file}")
        raise typer.Exit(1)
    if not dry_run:
        _check_asr_readiness(config, config_path)

    # Create job directory
    slug = make_job_slug(source)
    data_dir = Path(config.project.data_dir)
    job_dir = create_job_dir(data_dir, slug, resume=config.project.resume)
    manifest = load_manifest(job_dir)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_started_at = datetime.now(UTC).isoformat()

    # Copy source into job
    if config.project.copy_source_into_job:
        source_in_job = job_dir / "source" / source.name
        if not source_in_job.exists():
            shutil.copy2(source, source_in_job)
        manifest["source"]["original_path"] = str(source)
        manifest["source"]["original_filename"] = source.name

    save_manifest(job_dir, manifest)

    stages = ["prepare", "transcribe"]
    if config.diarization.enabled:
        stages.append("diarize")
    stages.append("merge")
    if config.summarization.enabled:
        stages.append("summarize")
    stages.append("render")
    if config.naming.enabled and config.naming.finalize_after_summary:
        stages.append("finalize")

    if dry_run:
        _print_dry_run(config, source, job_dir, stages)
        return

    # Determine start stage and force-stage handling
    start_idx = 0
    if from_stage:
        for i, s in enumerate(stages):
            if s == from_stage:
                start_idx = i
                break

    # Force-stage: mark specific stage as needing re-run
    force_stages = set()
    if force_stage:
        force_stages.add(force_stage)
        # Also clear downstream stages from manifest
        in_force = False
        for s in reversed(stages):
            if s == force_stage:
                in_force = True
            if in_force:
                update_stage_status(manifest, s, "pending")

    total = len(stages)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, stage in enumerate(stages[start_idx:], start=start_idx):
            task = progress.add_task(f"[{i + 1}/{total}] {stage}...", total=None)
            try:
                if stage == "prepare":
                    manifest = _run_prepare(source, job_dir, manifest, config)
                    _print_time_estimate(config, manifest, stages, job_dir)
                elif stage == "transcribe":
                    manifest = _run_transcribe(job_dir, manifest, config)
                elif stage == "diarize":
                    manifest = _run_diarize(job_dir, manifest, config)
                elif stage == "merge":
                    manifest = _run_merge(job_dir, manifest, config)
                elif stage == "summarize":
                    manifest = _run_summarize(job_dir, manifest, config, local_only=local_only)
                elif stage == "render":
                    manifest = _run_render(job_dir, manifest, config)
                elif stage == "finalize":
                    manifest = _run_finalize(
                        job_dir,
                        manifest,
                        config,
                        source,
                        copy_to_input,
                        run_id=run_id,
                        started_at=run_started_at,
                    )
            except StageCancelledError as exc:
                console.print(f"\n[yellow]Stage '{stage}' cancelled.[/yellow]")
                save_manifest(job_dir, manifest)
                write_run_report(
                    job_dir / "output" / "runs" / run_id / "report.md",
                    run_id=run_id,
                    operation="pipeline",
                    status="cancelled",
                    started_at=run_started_at,
                    manifest=manifest,
                    config=config,
                    error=f"Stage '{stage}' was cancelled.",
                )
                raise typer.Exit(130) from exc
            except Exception as e:
                update_stage_status(manifest, stage, "failed", error=str(e))
                save_manifest(job_dir, manifest)
                with suppress(OSError):
                    write_run_report(
                        job_dir / "output" / "runs" / run_id / "report.md",
                        run_id=run_id,
                        operation="pipeline",
                        status="failed",
                        started_at=run_started_at,
                        manifest=manifest,
                        config=config,
                        error=e,
                    )
                console.print(f"\n[red]Stage '{stage}' failed:[/red] {e}")
                raise typer.Exit(1) from e
            progress.update(task, completed=True, description=f"[{i + 1}/{total}] {stage} done")

    save_manifest(job_dir, manifest)
    if manifest.get("finalized", {}).get("generation_id") != run_id:
        write_run_report(
            job_dir / "output" / "runs" / run_id / "report.md",
            run_id=run_id,
            operation="pipeline",
            status="success",
            started_at=run_started_at,
            manifest=manifest,
            config=config,
        )
    console.print(f"\n[green]Pipeline complete.[/green] Job: {job_dir}")
    template = job_dir.resolve() / "speakers.yaml"
    if template.exists():
        console.print(f"Speaker template: {template}")
        console.print(f'Next: uv run meeting-notes speakers apply "{job_dir.resolve()}"')


def _print_dry_run(
    config: MeetingNotesConfig,
    source: Path,
    job_dir: Path,
    stages: list[str],
) -> None:
    """Print what would be done without executing."""
    console.print("\n[bold]Dry run — resolved configuration[/bold]\n")
    console.print(f"  Backend: {config.runtime.asr_backend}")
    console.print(f"  Device: {config.runtime.device}")
    console.print(f"  Model: {config.asr.model}")
    console.print(f"  Language: {config.asr.language}")
    console.print(f"  Diarization: {'enabled' if config.diarization.enabled else 'disabled'}")
    if config.diarization.enabled:
        console.print(f"  Speaker policy: {_speaker_policy_description(config.diarization)}")
    console.print(f"  Summarization: {'enabled' if config.summarization.enabled else 'disabled'}")
    console.print(f"\n  Source: {source}")
    console.print(f"  Job dir: {job_dir}")
    console.print("\n[bold]Planned stages:[/bold]")
    for i, stage in enumerate(stages, 1):
        console.print(f"  [{i}/{len(stages)}] {stage}")


def _print_time_estimate(
    config: MeetingNotesConfig,
    manifest: dict,
    stages: list[str],
    job_dir: Path,
) -> None:
    """Print an ASR/diarization ETA based on this machine's own timing history."""
    audio_seconds = manifest.get("source", {}).get("duration_seconds")
    lines = build_time_estimate_lines(
        config,
        stages,
        audio_seconds,
        data_dir=Path(config.project.data_dir),
        exclude_job_dir=job_dir,
    )
    if not lines:
        return
    console.print(f"\n[bold]{lines[0]}[/bold]")
    for line in lines[1:]:
        console.print(line)


def _run_prepare(
    source: Path,
    job_dir: Path,
    manifest: dict,
    config: MeetingNotesConfig,
) -> dict:
    """Stage: Inspect source and normalize audio."""
    update_stage_status(manifest, "prepare", "running")
    try:
        # Inspect
        info = inspect_media(source, ffprobe_path=config.runtime.ffprobe_path)
        manifest["source"]["hash"] = info.file_hash
        manifest["source"]["duration_seconds"] = info.duration_seconds
        manifest["source"]["format"] = info.format_name

        # Normalize
        normalized_path = create_normalized_path(job_dir, source.name)
        normalize_audio(
            source,
            normalized_path,
            sample_rate=config.audio.output_sample_rate,
            channels=config.audio.output_channels,
            codec=config.audio.output_codec,
            highpass_hz=config.audio.normalize.highpass_hz,
            lowpass_hz=config.audio.normalize.lowpass_hz,
            loudnorm_enabled=config.audio.normalize.loudnorm.enabled,
            loudnorm_lufs=config.audio.normalize.loudnorm.integrated_lufs,
            loudnorm_range=config.audio.normalize.loudnorm.loudness_range,
            loudnorm_peak=config.audio.normalize.loudnorm.true_peak_db,
            extra_filters=config.audio.normalize.extra_filters,
            ffmpeg_path=config.runtime.ffmpeg_path,
        )

        update_stage_status(manifest, "prepare", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "prepare", "failed")
        raise


def _run_transcribe(job_dir: Path, manifest: dict, config: MeetingNotesConfig) -> dict:
    """Stage: Run ASR transcription."""
    update_stage_status(manifest, "transcribe", "running")
    try:
        normalized = job_dir / "audio" / "normalized.wav"
        if not normalized.exists():
            raise FileNotFoundError(f"Normalized audio not found: {normalized}")

        configured = get_configured_backend(config)
        backend = configured.backend
        runtime_identity = configured.runtime_identity
        manifest["stages"]["transcribe"]["runtime"] = runtime_identity
        log.info("asr.runtime_selected", **runtime_identity)

        readiness = configured.check_readiness()
        if not readiness.available:
            raise DependencyMissingError(readiness.detail)

        if config.runtime.asr_backend == "whisper_cpp" and config.runtime.threads <= 0:
            log.info(
                "whisper_cpp.threads_auto_selected",
                threads=configured.transcribe_kwargs["threads"],
                logical_cores=os.cpu_count() or 1,
            )

        chunks = _transcription_chunks(normalized, manifest, config)
        if len(chunks) == 1 and chunks[0].path == str(normalized):
            result = backend.transcribe(
                normalized,
                **configured.transcribe_kwargs,
            )
        else:
            materialize_audio_chunks(
                normalized,
                chunks,
                job_dir / "audio" / "chunks",
                ffmpeg_path=config.runtime.ffmpeg_path,
            )
            save_chunks_manifest(chunks, job_dir / "audio" / "chunks.json")
            chunk_results: list[tuple[AudioChunk, ASRResult]] = []
            for index, chunk in enumerate(chunks, 1):
                console.print(
                    f"  Transcribing audio chunk {index}/{len(chunks)} "
                    f"({chunk.source_start:.1f}s-{chunk.source_end:.1f}s)"
                )
                chunk_results.append(
                    (
                        chunk,
                        backend.transcribe(
                            Path(chunk.path),
                            **configured.transcribe_kwargs,
                        ),
                    )
                )
            result = _merge_asr_chunks(chunk_results)
            runtime_identity["chunk_count"] = len(chunks)
            runtime_identity["chunked"] = True
        runtime_identity["backend"] = result.backend
        runtime_identity["device"] = result.device
        if result.raw_output.get("server_version"):
            runtime_identity["server_version"] = result.raw_output["server_version"]
        manifest["stages"]["transcribe"]["runtime"] = runtime_identity

        # Use duration from manifest if available, else estimate
        if "duration_seconds" in manifest.get("source", {}):
            result.duration = manifest["source"]["duration_seconds"]
        else:
            result.duration = normalized.stat().st_size / (16000 * 2)  # 16kHz mono 16-bit

        # Render output formats
        source_filename = manifest.get("source", {}).get("original_filename", "")
        render_all_formats(
            result,
            job_dir / "asr",
            source_filename=source_filename,
            formats=config.asr.output_formats,
        )

        update_stage_status(manifest, "transcribe", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "transcribe", "failed")
        raise


def _transcription_chunks(
    normalized: Path,
    manifest: dict,
    config: MeetingNotesConfig,
) -> list[AudioChunk]:
    """Plan configured chunks plus mandatory Lemonade upload-size chunks."""
    duration = float(manifest.get("source", {}).get("duration_seconds") or 0.0)
    if duration <= 0:
        duration = normalized.stat().st_size / (
            config.audio.output_sample_rate * config.audio.output_channels * 2
        )

    chunking = config.audio.chunking
    mode = chunking.mode
    max_minutes = chunking.max_chunk_minutes
    if config.runtime.asr_backend == "lemonade":
        upload_bytes = config.asr.backend_options.lemonade.max_upload_mib * 1024 * 1024
        # Leave 10% for WAV/multipart overhead and unusual source headers.
        safe_bytes = upload_bytes * 0.9
        if normalized.stat().st_size > safe_bytes:
            bytes_per_second = normalized.stat().st_size / max(duration, 0.001)
            # compute_chunks expands interior chunks by the overlap on both
            # sides, so reserve that duration inside the upload-size budget.
            safe_chunk_seconds = safe_bytes / bytes_per_second
            overlap_reserve = max(0.0, chunking.overlap_seconds) * 2
            upload_minutes = max(
                0.001,
                safe_chunk_seconds - overlap_reserve,
            ) / 60.0
            max_minutes = min(max_minutes, upload_minutes)
            mode = "fixed"

    chunks = compute_chunks(
        duration,
        mode=mode,
        max_chunk_minutes=max_minutes,
        overlap_seconds=chunking.overlap_seconds,
        trigger_duration_minutes=chunking.trigger_duration_minutes,
    )
    if len(chunks) == 1:
        chunks[0].path = str(normalized)
    return chunks


def _merge_asr_chunks(
    chunk_results: list[tuple[AudioChunk, ASRResult]],
) -> ASRResult:
    """Merge relative chunk timestamps into one absolute ASR result."""
    if not chunk_results:
        raise RuntimeError("No ASR chunk results were produced.")

    merged: list[ASRSegment] = []
    last_index = len(chunk_results) - 1
    warnings: list[str] = []
    for chunk_index, (chunk, result) in enumerate(chunk_results):
        core_start = (
            chunk.source_start + chunk.overlap_before if chunk_index > 0 else chunk.source_start
        )
        core_end = (
            chunk.source_end - chunk.overlap_after if chunk_index < last_index else chunk.source_end
        )
        warnings.extend(result.warnings)
        for segment in result.segments:
            absolute_start = min(
                chunk.source_end,
                max(chunk.source_start, segment.start + chunk.source_start),
            )
            absolute_end = min(
                chunk.source_end,
                max(absolute_start, segment.end + chunk.source_start),
            )
            midpoint = (absolute_start + absolute_end) / 2
            if midpoint < core_start or midpoint >= core_end:
                continue
            merged.append(
                ASRSegment(
                    id="",
                    start=absolute_start,
                    end=absolute_end,
                    text=segment.text,
                    language=segment.language,
                    speaker=segment.speaker,
                    confidence=segment.confidence,
                    metrics=segment.metrics,
                    source={
                        **segment.source,
                        "chunk_id": chunk.chunk_id,
                        "chunk_source_start": chunk.source_start,
                    },
                )
            )

    merged.sort(key=lambda item: (item.start, item.end))
    for index, segment in enumerate(merged):
        segment.id = f"seg-{index:06d}"

    first = chunk_results[0][1]
    return ASRResult(
        segments=merged,
        language=first.language,
        duration=max(chunk.source_end for chunk, _ in chunk_results),
        backend=first.backend,
        model=first.model,
        device=first.device,
        raw_output={
            **first.raw_output,
            "chunk_count": len(chunk_results),
        },
        warnings=warnings,
    )


def _resolve_whisper_threads(config: MeetingNotesConfig) -> int:
    """Resolve zero/automatic thread configuration for whisper.cpp."""
    from meeting_notes.asr.registry import _resolve_whisper_threads as resolve

    return resolve(config)


def _module_available(module_name: str) -> bool:
    """Check an optional module without importing its heavy dependencies."""
    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def _print_diarization_remediation(
    config: MeetingNotesConfig,
    *,
    pyannote_installed: bool,
) -> None:
    """Print exact setup steps when optional diarization is unavailable."""
    from meeting_notes.configure import _diarization_recommendations
    from meeting_notes.diarization.setup import resolve_hf_token

    token, _ = resolve_hf_token(config.diarization.token_env)
    checks = {
        "pyannote_installed": pyannote_installed,
        "hf_token_ready": bool(token),
        "local_diarization_model_ready": bool(
            config.diarization.model_path and Path(config.diarization.model_path).exists()
        ),
    }
    console.print("[yellow]  Diarization unavailable.[/yellow]")
    for line in _diarization_recommendations(config, checks):
        console.print(line)


def _run_diarize(job_dir: Path, manifest: dict, config: MeetingNotesConfig) -> dict:
    """Stage: Run speaker diarization."""
    update_stage_status(manifest, "diarize", "running")
    try:
        # Check if diarization is enabled and available
        if not config.diarization.enabled:
            update_stage_status(manifest, "diarize", "skipped")
            return manifest

        # Load normalized audio
        normalized = job_dir / "audio" / "normalized.wav"
        if not normalized.exists():
            raise FileNotFoundError(f"Normalized audio not found: {normalized}")

        manifest["stages"]["diarize"]["runtime"] = {
            "backend": config.diarization.backend,
            "device": config.diarization.device,
            "rocm_gpu_runtime_path": config.diarization.rocm_gpu_runtime_path,
            "model": config.diarization.model,
            "num_speakers": config.diarization.num_speakers,
            "min_speakers": config.diarization.min_speakers,
            "max_speakers": config.diarization.max_speakers,
        }

        # Import and try to load pyannote
        try:
            from meeting_notes.diarization.pyannote import PyannoteDiarizationBackend

            backend = PyannoteDiarizationBackend(
                model_name=config.diarization.model,
                model_path=(
                    Path(config.diarization.model_path) if config.diarization.model_path else None
                ),
                token_env=config.diarization.token_env,
                device=config.diarization.device,
                rocm_gpu_runtime_path=(
                    Path(config.diarization.rocm_gpu_runtime_path)
                    if config.diarization.rocm_gpu_runtime_path
                    else None
                ),
                use_exclusive=config.diarization.use_exclusive_diarization,
            )

            if not backend.is_available():
                _print_diarization_remediation(
                    config,
                    pyannote_installed=_module_available("pyannote.audio"),
                )
                raise DiarizationUnavailableError(
                    "Diarization is enabled but pyannote.audio and/or "
                    f"{config.diarization.token_env} is unavailable. "
                    "Disable diarization explicitly to continue without speaker labels."
                )

            result = backend.diarize(
                normalized,
                num_speakers=config.diarization.num_speakers,
                min_speakers=config.diarization.min_speakers,
                max_speakers=config.diarization.max_speakers,
            )

            # Save diarization output
            import json

            diar_path = job_dir / "diarization" / "diarization.json"
            diar_data = {
                "turns": [
                    {
                        "turn_id": t.turn_id,
                        "start": t.start,
                        "end": t.end,
                        "speaker": t.speaker,
                        "confidence": t.confidence,
                        "source": t.source,
                    }
                    for t in result.turns
                ]
            }
            diar_path.write_text(json.dumps(diar_data, indent=2), encoding="utf-8")

        except ImportError:
            _print_diarization_remediation(config, pyannote_installed=False)
            raise DiarizationUnavailableError(
                "Diarization is enabled but pyannote.audio is not installed. "
                "Disable diarization explicitly to continue without speaker labels."
            ) from None

        update_stage_status(manifest, "diarize", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "diarize", "failed")
        raise


def _run_merge(job_dir: Path, manifest: dict, config: MeetingNotesConfig) -> dict:
    """Stage: Merge transcript and diarization."""
    update_stage_status(manifest, "merge", "running")
    try:
        import json

        # Load raw transcript
        raw_json_path = job_dir / "asr" / "transcript.raw.json"
        if not raw_json_path.exists():
            raise FileNotFoundError(f"Raw transcript not found: {raw_json_path}")

        raw_data = json.loads(raw_json_path.read_text(encoding="utf-8"))
        segments = raw_data.get("segments", [])

        # Load diarization if available
        diar_path = job_dir / "diarization" / "diarization.json"
        if diar_path.exists():
            from meeting_notes.diarization.base import DiarizationTurn

            diar_data = json.loads(diar_path.read_text(encoding="utf-8"))
            turns = [
                DiarizationTurn(
                    turn_id=t["turn_id"],
                    start=t["start"],
                    end=t["end"],
                    speaker=t["speaker"],
                    confidence=t.get("confidence"),
                    source=t.get("source", ""),
                )
                for t in diar_data.get("turns", [])
            ]

            # Convert segments to TranscriptSegment objects
            from meeting_notes.diarization.reconcile import assign_speakers, load_speaker_map
            from meeting_notes.transcript.models import TranscriptSegment

            ts_segments = [
                TranscriptSegment(
                    id=s["id"],
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    language=s.get("language"),
                    speaker=s.get("speaker"),
                    confidence=s.get("confidence"),
                    metrics=s.get("metrics", {}),
                    source=s.get("source", {}),
                )
                for s in segments
            ]

            # Assign speakers
            assign_speakers(
                ts_segments,
                turns,
                assignment_method=config.diarization.assignment_method,
                minimum_overlap_ratio=config.diarization.minimum_overlap_ratio,
                nearest_tolerance_seconds=config.diarization.nearest_tolerance_seconds,
                unknown_label=config.diarization.unknown_speaker_label,
            )

            # Apply speaker map if configured
            if config.diarization.speaker_map_path:
                speaker_map = load_speaker_map(config.diarization.speaker_map_path)
                if speaker_map:
                    from meeting_notes.diarization.reconcile import apply_speaker_map

                    apply_speaker_map(ts_segments, speaker_map)

            # Update segments with speaker info
            for s, ts in zip(segments, ts_segments, strict=True):
                s["speaker"] = ts.speaker

        # Apply glossary corrections if enabled. The job-scoped glossary
        # (<job_dir>/glossary.yaml, written by `clarify apply`) is layered
        # over the global glossary so corrections stay scoped to this
        # recording unless explicitly promoted (`meeting-notes glossary promote`).
        if config.glossary.enabled:
            from pathlib import Path as P

            from meeting_notes.transcript.glossary import (
                correct_transcript_segments,
                load_layered_glossary,
            )

            global_path = P(config.glossary.path) if config.glossary.path else None
            job_glossary_path = job_dir / "glossary.yaml"
            glossary = load_layered_glossary(global_path, job_glossary_path)
            if glossary.terms:
                segments, corrections = correct_transcript_segments(
                    segments,
                    glossary,
                    case_sensitive=config.glossary.case_sensitive,
                )
                if corrections and config.glossary.record_corrections:
                    manifest["glossary_corrections"] = [
                        {
                            "segment_id": c.segment_id,
                            "original_text": c.original_text,
                            "corrected_text": c.corrected_text,
                            "rule_canonical": c.rule_canonical,
                            "rule_alias": c.rule_alias,
                        }
                        for c in corrections
                    ]

        # Save merged transcript
        merged_path = job_dir / "transcript" / "transcript.merged.json"
        merged_data = {
            "metadata": raw_data.get("metadata", {}),
            "segments": segments,
        }
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(
            json.dumps(merged_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Speaker identification is a downstream-only workflow. Never overwrite
        # a map the user may already have edited.
        from meeting_notes.speakers import write_template

        template_path, template_warning = write_template(job_dir, automatic=True)
        if template_warning:
            console.print(
                f"[yellow]  Speaker map: {template_warning}; candidate: {template_path}[/yellow]"
            )
        elif template_path:
            manifest["speaker_template"] = {
                "version": 1,
                "path": str(template_path),
                "transcript_sha256": __import__("hashlib")
                .sha256(merged_path.read_bytes())
                .hexdigest(),
            }

        update_stage_status(manifest, "merge", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "merge", "failed")
        raise


class SummarizationUnavailable(RuntimeError):
    """Raised when the configured summarizer is disabled, rejected, or unavailable.

    The pipeline stage treats this as a skip; callers that require a summary
    (like re-summarizing after human clarification) should let it propagate.
    """


_LOCAL_SUMMARY_WARNING = (
    "> [!WARNING]\n"
    "> **Local AI — best-effort summary**\n"
    "> Generated locally with `{model}` via AMD Lemonade. Local summaries may be "
    "slower and less accurate than production-grade summarizers. Verify important "
    "details against the transcript; configure Codex or another structured summarizer "
    "when stronger correctness is required."
)


def _generate_summary_result(
    segments: list[dict],
    config: MeetingNotesConfig,
    local_only: bool,
    *,
    extra_context: str = "",
):
    """Run the configured adapter and preserve its native output contract."""
    if not config.summarization.enabled:
        raise SummarizationUnavailable("Summarization is disabled in configuration.")

    from meeting_notes.summarization.adapters import configured_adapter_options, get_adapter

    adapter_name = config.summarization.backend
    is_lemonade = adapter_name == "lemonade"
    transcript_text = (
        _format_local_summary_transcript(segments)
        if is_lemonade
        else _format_summary_transcript(segments)
    )
    configured_prompt = (
        config.summarization.lemonade.prompt_path
        if is_lemonade
        else config.summarization.prompt_path
    )
    prompt_path = Path(configured_prompt)
    prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else "Summarize this meeting transcript."
    )
    if extra_context:
        prompt = f"{extra_context}\n\n{prompt}"

    schema_path = None
    if not is_lemonade and config.summarization.output_schema_path:
        schema_path = Path(config.summarization.output_schema_path)

    if local_only and adapter_name in ("codex", "codex_cli", "opencode", "mimo", "claude"):
        raise SummarizationUnavailable(
            f"Summarization adapter '{adapter_name}' is not allowed with --local-only."
        )

    adapter = get_adapter(
        adapter_name,
        **configured_adapter_options(config.summarization),
    )
    if not adapter.is_available():
        detail = (
            f" Start Lemonade Server manually at {config.summarization.lemonade.base_url}."
            if is_lemonade
            else ""
        )
        raise SummarizationUnavailable(
            f"Summarization adapter '{adapter_name}' is not available.{detail}"
        )
    timeout = (
        config.summarization.lemonade.request_timeout_seconds
        if is_lemonade
        else config.summarization.timeout_seconds
    )
    return adapter.summarize(
        transcript_text,
        prompt=prompt,
        schema_path=schema_path,
        timeout_seconds=timeout,
        metadata={
            "language": config.summarization.language,
            "speaker_resolution": "diarized" if any(s.get("speaker") for s in segments) else "none",
        },
    )


def _summarize_transcript(
    segments: list[dict],
    config: MeetingNotesConfig,
    local_only: bool,
    *,
    extra_context: str = "",
) -> dict:
    """Run the configured summarizer adapter over transcript segments.

    Shared by the `summarize` pipeline stage and `clarifications.apply_clarifications`
    (which passes `extra_context` — confirmed human answers the model should treat
    as authoritative over the raw transcript).
    """
    result = _generate_summary_result(
        segments,
        config,
        local_only,
        extra_context=extra_context,
    )
    if result.output_format != "structured_json" or result.data is None:
        raise SummarizationUnavailable(
            "Clarification and speaker-driven re-summarization require a structured "
            "summary. Switch to Codex or another structured summarizer."
        )
    return result.data


def _run_summarize(
    job_dir: Path, manifest: dict, config: MeetingNotesConfig, local_only: bool = False
) -> dict:
    """Stage: Run summarization."""
    update_stage_status(manifest, "summarize", "running")
    try:
        import json

        from meeting_notes.summarization.adapters import summarizer_provenance

        # Load transcript text
        merged_path = job_dir / "transcript" / "transcript.merged.json"
        if not merged_path.exists():
            merged_path = job_dir / "asr" / "transcript.raw.json"

        if not merged_path.exists():
            raise FileNotFoundError("No transcript found for summarization")

        raw_data = json.loads(merged_path.read_text(encoding="utf-8"))
        segments = raw_data.get("segments", [])

        try:
            result = _generate_summary_result(segments, config, local_only)
        except SummarizationUnavailable as reason:
            console.print(f"[yellow]  {reason}[/yellow]")
            update_stage_status(manifest, "summarize", "skipped")
            return manifest

        summary_dir = job_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        provider = summarizer_provenance(config.summarization)
        if result.output_format == "markdown":
            markdown = result.markdown or ""
            model = config.summarization.lemonade.model_id
            warning = _LOCAL_SUMMARY_WARNING.format(model=model)
            (summary_dir / "summary.md").write_text(
                f"{warning}\n\n{markdown.strip()}\n",
                encoding="utf-8",
            )
            for stale in (summary_dir / "summary.json", job_dir / "output" / "summary.json"):
                stale.unlink(missing_ok=True)
            provider.update(
                {
                    "output_format": "markdown",
                    "quality_tier": "best_effort_local",
                    "metrics": result.metrics,
                }
            )
        else:
            if result.data is None:
                raise RuntimeError("Structured summarizer returned no JSON data.")
            (summary_dir / "summary.json").write_text(
                json.dumps(result.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (summary_dir / "summary.md").unlink(missing_ok=True)
            provider.update(
                {
                    "output_format": "structured_json",
                    "quality_tier": "structured",
                }
            )
        manifest["stages"]["summarize"]["provider"] = provider

        update_stage_status(manifest, "summarize", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "summarize", "failed")
        raise


def _format_summary_transcript(segments: list[dict]) -> str:
    """Format segments with stable IDs so summaries can cite evidence."""
    transcript_lines = []
    for index, segment in enumerate(segments):
        start = segment["start"]
        hours = int(start // 3600)
        minutes = int((start % 3600) // 60)
        seconds = int(start % 60)
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        segment_id = segment.get("id") or f"seg-{index:06d}"
        speaker = f" [{segment['speaker']}]" if segment.get("speaker") else ""
        transcript_lines.append(f"[{segment_id}] [{timestamp}]{speaker} {segment['text']}")
    return "\n".join(transcript_lines)


def _format_local_summary_transcript(segments: list[dict]) -> str:
    """Format a compact transcript without evidence IDs for local Markdown output."""
    lines: list[str] = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        hours = int(start // 3600)
        minutes = int((start % 3600) // 60)
        seconds = int(start % 60)
        speaker = str(segment.get("speaker") or "").strip()
        prefix = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
        if speaker:
            prefix = f"{prefix} {speaker}:"
        lines.append(f"{prefix} {str(segment.get('text') or '').strip()}")
    return "\n".join(line for line in lines if line.strip())


def _run_render(job_dir: Path, manifest: dict, config: MeetingNotesConfig) -> dict:
    """Stage: Render meeting minutes."""
    update_stage_status(manifest, "render", "running")
    try:
        import json

        from meeting_notes.minutes.render import render_minutes, save_minutes

        markdown_summary = job_dir / "summary" / "summary.md"
        summary_path = job_dir / "summary" / "summary.json"
        if markdown_summary.exists():
            minutes_path = job_dir / "output" / "minutes.md"
            minutes_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(markdown_summary, minutes_path)
            (job_dir / "output" / "summary.json").unlink(missing_ok=True)
            update_stage_status(manifest, "render", "completed")
            return manifest
        if not summary_path.exists():
            console.print("[yellow]  No summary found, skipping render[/yellow]")
            update_stage_status(manifest, "render", "skipped")
            return manifest

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # Get source info from manifest
        source_filename = manifest.get("source", {}).get("original_filename", "")

        # Render minutes
        markdown = render_minutes(
            summary,
            source_filename=source_filename,
        )

        # Save minutes
        minutes_path = job_dir / "output" / "minutes.md"
        save_minutes(markdown, minutes_path)

        # Save summary JSON export
        export_path = job_dir / "output" / "summary.json"
        export_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        update_stage_status(manifest, "render", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "render", "failed")
        raise


def _run_finalize(
    job_dir: Path,
    manifest: dict,
    config: MeetingNotesConfig,
    source: Path,
    copy_to_input: bool = False,
    *,
    run_id: str | None = None,
    started_at: str | None = None,
) -> dict:
    """Stage: Finalize recording and note filenames."""
    update_stage_status(manifest, "finalize", "running")
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    started_at = started_at or datetime.now(UTC).isoformat()
    staging = job_dir / "output" / f".finalize-{run_id}"
    generation = job_dir / "output" / "finalized" / run_id
    try:
        import json

        from meeting_notes.naming import (
            generate_filenames,
            resolve_date,
            sanitize_short_title,
        )

        markdown_summary = job_dir / "summary" / "summary.md"
        summary_path = job_dir / "summary" / "summary.json"
        if not summary_path.exists() and not markdown_summary.exists():
            console.print("[yellow]  No summary found, skipping finalize[/yellow]")
            update_stage_status(manifest, "finalize", "skipped")
            return manifest

        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            raw_title = str(summary.get("short_title", "meeting"))
        else:
            summary = {}
            markdown = markdown_summary.read_text(encoding="utf-8")
            heading = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
            raw_title = heading.group(1) if heading else source.stem
        short_title = sanitize_short_title(
            raw_title,
            max_length=config.naming.max_short_title_characters,
        )
        media_creation = manifest.get("source", {}).get("creation_time", "")
        source_mtime = source.stat().st_mtime if source.exists() else 0.0
        date, date_source = resolve_date(
            summary,
            media_creation_time=media_creation,
            source_mtime=source_mtime,
            source_order=config.naming.date_source_order,
        )
        original_ext = source.suffix if source.exists() else ".m4a"
        filenames = generate_filenames(
            date,
            short_title,
            original_ext,
            recording_template=config.naming.recording_template,
            minutes_template=config.naming.minutes_template,
            json_template=config.naming.json_export_template,
            transcript_json_template=config.naming.transcript_json_template,
            transcript_markdown_template=config.naming.transcript_markdown_template,
            transcript_srt_template=config.naming.transcript_srt_template,
            transcript_vtt_template=config.naming.transcript_vtt_template,
        )
        console.print(f"  Finalizing: {date} {short_title}")
        staging.mkdir(parents=True)
        layout = publication_paths(staging, filenames)

        transcript_path = job_dir / "transcript" / "transcript.merged.json"
        if not transcript_path.exists():
            transcript_path = job_dir / "asr" / "transcript.raw.json"
        transcript_data = json.loads(transcript_path.read_text(encoding="utf-8"))
        render_dir = staging / ".render"
        rendered = render_transcript_variants(transcript_data, render_dir)
        for key, rendered_path in rendered.items():
            target = layout[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            rendered_path.rename(target)
        shutil.rmtree(render_dir)

        source_in_job = job_dir / "source" / source.name
        if source_in_job.exists() and config.naming.recording_mode != "none":
            if config.naming.recording_mode == "in_place":
                new_path = source.parent / filenames["recording"]
                from meeting_notes.naming import resolve_collision

                new_path = resolve_collision(new_path, policy=config.naming.collision_policy)
                os.rename(str(source), str(new_path))
                console.print(f"  Renamed: {source.name} -> {new_path.name}")
                manifest["source"]["finalized_path"] = str(new_path)
            else:
                shutil.copy2(source_in_job, layout["recording"])
                console.print(f"  Copied: -> {layout['recording'].name}")

        minutes_src = job_dir / "output" / "minutes.md"
        if minutes_src.exists():
            shutil.copy2(minutes_src, layout["minutes"])
            console.print(f"  Copied: -> {layout['minutes'].name}")

        summary_src = job_dir / "output" / "summary.json"
        if summary_src.exists():
            layout["json_export"].parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_src, layout["json_export"])
            console.print(f"  Copied: -> json/{layout['json_export'].name}")

        relative_outputs = [
            str(path.relative_to(staging))
            for key, path in layout.items()
            if key != "run_report" and path.exists()
        ]
        update_stage_status(manifest, "finalize", "completed")
        write_run_report(
            layout["run_report"],
            run_id=run_id,
            operation="pipeline",
            status="success",
            started_at=started_at,
            manifest=manifest,
            config=config,
            transcript_sha256=hashlib.sha256(transcript_path.read_bytes()).hexdigest(),
            speaker_resolution=(
                "diarized"
                if any(item.get("speaker") for item in transcript_data.get("segments", []))
                else "none"
            ),
            outputs=relative_outputs,
            messages=[f"Published {len(relative_outputs)} output files."],
        )
        generation.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, generation)
        external_paths: list[str] = []

        if copy_to_input and source.exists():
            for published in (path for path in generation.rglob("*") if path.is_file()):
                relative = published.relative_to(generation)
                target = source.parent / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(published, target)
                external_paths.append(str(target))
                console.print(f"  Copied to input dir: {relative}")

        manifest["finalized"] = {
            "generation_id": run_id,
            "root": str(generation),
            "date": date,
            "date_source": date_source,
            "short_title": short_title,
            "filenames": filenames,
            "managed_paths": managed_files(generation),
            "external_paths": external_paths,
        }
        return manifest
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        update_stage_status(manifest, "finalize", "failed")
        with suppress(OSError):
            write_run_report(
                job_dir / "output" / "runs" / run_id / "report.md",
                run_id=run_id,
                operation="pipeline",
                status="failed",
                started_at=started_at,
                manifest=manifest,
                config=config,
                error=error,
            )
        raise


# --- Individual stage commands ---


def run_prepare(input_file: str, config_path: str | None = None) -> None:
    """Inspect source and normalize audio."""
    config = _load_or_fail(config_path)
    _check_tools(config)
    source = Path(input_file)
    if not source.exists():
        console.print(f"[red]File not found:[/red] {input_file}")
        raise typer.Exit(1)

    slug = make_job_slug(source)
    job_dir = create_job_dir(Path(config.project.data_dir), slug)
    manifest = load_manifest(job_dir)
    manifest = _run_prepare(source, job_dir, manifest, config)
    save_manifest(job_dir, manifest)
    console.print(f"[green]Prepare complete.[/green] Job: {job_dir}")


def run_transcribe(job_dir: str, config_path: str | None = None) -> None:
    """Run ASR transcription."""
    config = _load_or_fail(config_path)
    _check_asr_readiness(config, config_path)
    manifest = load_manifest(Path(job_dir))
    manifest = _run_transcribe(Path(job_dir), manifest, config)
    save_manifest(Path(job_dir), manifest)
    console.print("[green]Transcribe complete.[/green]")


def run_diarize(job_dir: str, config_path: str | None = None) -> None:
    """Run speaker diarization."""
    config = _load_or_fail(config_path)
    manifest = load_manifest(Path(job_dir))
    manifest = _run_diarize(Path(job_dir), manifest, config)
    save_manifest(Path(job_dir), manifest)
    console.print("[green]Diarize complete.[/green]")


def run_merge(job_dir: str, config_path: str | None = None) -> None:
    """Merge transcript and diarization."""
    config = _load_or_fail(config_path)
    manifest = load_manifest(Path(job_dir))
    manifest = _run_merge(Path(job_dir), manifest, config)
    save_manifest(Path(job_dir), manifest)
    console.print("[green]Merge complete.[/green]")
    template = Path(job_dir).resolve() / "speakers.yaml"
    if template.exists():
        console.print(f"Speaker template: {template}")
        console.print(f'Next: uv run meeting-notes speakers apply "{Path(job_dir).resolve()}"')


def run_summarize(job_dir: str, config_path: str | None = None) -> None:
    """Run summarization."""
    config = _load_or_fail(config_path)
    manifest = load_manifest(Path(job_dir))
    manifest = _run_summarize(Path(job_dir), manifest, config)
    save_manifest(Path(job_dir), manifest)
    console.print("[green]Summarize complete.[/green]")


def run_render(job_dir: str, config_path: str | None = None) -> None:
    """Render meeting minutes."""
    config = _load_or_fail(config_path)
    manifest = load_manifest(Path(job_dir))
    manifest = _run_render(Path(job_dir), manifest, config)
    save_manifest(Path(job_dir), manifest)
    console.print("[green]Render complete.[/green]")


def run_naming_preview(job_dir: str, config_path: str | None = None) -> None:
    """Preview finalized filenames."""
    from meeting_notes.naming import generate_filenames, resolve_date, sanitize_short_title

    config = _load_or_fail(config_path)
    job_path = Path(job_dir)
    manifest = load_manifest(job_path)
    summary_path = job_path / "summary" / "summary.json"

    if not summary_path.exists():
        console.print("[yellow]No summary found. Run summarize first.[/yellow]")
        return

    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    short_title = sanitize_short_title(summary.get("short_title", "meeting"))
    date, date_source = resolve_date(summary, source_order=config.naming.date_source_order)
    original_ext = Path(manifest.get("source", {}).get("original_filename", ".m4a")).suffix

    filenames = generate_filenames(
        date,
        short_title,
        original_ext,
        recording_template=config.naming.recording_template,
        minutes_template=config.naming.minutes_template,
        json_template=config.naming.json_export_template,
        transcript_json_template=config.naming.transcript_json_template,
        transcript_markdown_template=config.naming.transcript_markdown_template,
        transcript_srt_template=config.naming.transcript_srt_template,
        transcript_vtt_template=config.naming.transcript_vtt_template,
    )

    console.print("\n[bold]Filename preview[/bold]")
    console.print(f"  Date source: {date_source}")
    console.print(f"  Date: {date}")
    console.print(f"  Short title: {short_title}")
    console.print(f"  Recording: {filenames['recording']}")
    console.print(f"  Minutes: {filenames['minutes']}")
    console.print(f"  Transcript: {filenames['transcript_markdown']}")
    console.print(f"  Summary JSON: json/{filenames['json_export']}")
    console.print(f"  Transcript JSON: json/{filenames['transcript_json']}")
    console.print(f"  SRT: subtitles/{filenames['transcript_srt']}")
    console.print(f"  VTT: subtitles/{filenames['transcript_vtt']}")
    console.print("  Run report: run/report.md")


def run_naming_finalize(job_dir: str, config_path: str | None = None) -> None:
    """Finalize filenames."""
    console.print("[yellow]naming finalize: implemented in pipeline[/yellow]")


def run_benchmark(input_file: str, matrix: str, config_path: str | None = None) -> None:
    """Run benchmark comparing configurations."""
    from meeting_notes.benchmark.runner import (
        load_benchmark_matrix,
        render_benchmark_report,
        run_benchmark_matrix,
    )

    source = Path(input_file)
    if not source.exists():
        console.print(f"[red]File not found:[/red] {input_file}")
        raise typer.Exit(1)

    matrix_path = Path(matrix)
    if not matrix_path.exists():
        console.print(f"[red]Benchmark matrix not found:[/red] {matrix}")
        raise typer.Exit(1)

    bench_matrix = load_benchmark_matrix(matrix_path)
    console.print(f"[bold]Running benchmark with {len(bench_matrix.runs)} configurations[/bold]")

    results = run_benchmark_matrix(bench_matrix, source)

    # Save results
    output_dir = Path("data") / "benchmarks" / source.stem
    output_paths = render_benchmark_report(results, output_dir)

    console.print("\n[green]Benchmark complete.[/green]")
    for fmt, path in output_paths.items():
        console.print(f"  {fmt}: {path}")


def run_clean(job_dir: str, stage: str | None = None, yes: bool = False) -> None:
    """Clean job artifacts."""
    import shutil

    if not yes:
        console.print(f"[yellow]This will delete artifacts in {job_dir}[/yellow]")
        if not typer.confirm("Continue?"):
            return

    job_path = Path(job_dir)
    if stage:
        stage_dir = job_path / stage
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
            console.print(f"[green]Cleaned {stage}[/green]")
    else:
        # Clean everything except source
        for d in ["asr", "diarization", "transcript", "summary", "output", "logs"]:
            dir_path = job_path / d
            if dir_path.exists():
                shutil.rmtree(dir_path)
        console.print("[green]Cleaned all artifacts[/green]")
