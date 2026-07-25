"""Pipeline orchestration — wires together all processing stages."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from meeting_notes.audio.inspect import inspect_media
from meeting_notes.audio.normalize import create_normalized_path, normalize_audio
from meeting_notes.asr.registry import get_backend
from meeting_notes.config import MeetingNotesConfig, load_config
from meeting_notes.errors import (
    ConfigurationError,
    DependencyMissingError,
    DiarizationUnavailableError,
    StageCancelledError,
)
from meeting_notes.jobs import (
    compute_stage_fingerprint,
    create_job_dir,
    load_manifest,
    make_job_slug,
    save_manifest,
    stage_is_stale,
    update_stage_status,
)
from meeting_notes.transcript.render import render_all_formats
from meeting_notes.transcript.models import TranscriptDocument, TranscriptSegment

log = structlog.get_logger()
console = Console(stderr=True)


def _load_or_fail(config_path: str | None) -> MeetingNotesConfig:
    """Load config or print actionable error."""
    try:
        return load_config(config_path)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(1)


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
        except RuntimeError:
            console.print(f"[red]{tool_name} not found at '{tool_path}'. Install FFmpeg.[/red]")
            raise typer.Exit(1)


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

    runtimes = installed_runtimes()
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

    cpu_ready = any(
        item.get("backend") == "cpu" and item.get("healthy")
        for item in runtimes
    )
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
    """Fail before audio preparation with complete whisper.cpp remediation."""
    if config.runtime.asr_backend != "whisper_cpp":
        return
    backend = get_backend("whisper_cpp", executable=config.runtime.whisper_cpp_path)
    runtime_ready = backend.is_available()
    model_ready = False
    model_detail = "model_path is not configured"
    if config.asr.model_path:
        from meeting_notes.models import verify_model

        try:
            model_ready, model_detail = verify_model(
                config.asr.model, Path(config.asr.model_path)
            )
        except RuntimeError:
            model_ready = Path(config.asr.model_path).is_file()
            model_detail = "present" if model_ready else "missing"
    if runtime_ready and model_ready:
        return
    if not runtime_ready:
        console.print(
            f"[red]Configured whisper.cpp executable is unavailable:[/red] "
            f"{config.runtime.whisper_cpp_path}"
        )
    if not model_ready:
        console.print(
            f"[red]Configured Whisper model is unavailable:[/red] "
            f"{config.asr.model} ({model_detail})"
        )
    console.print(_asr_remediation(config, config_path, runtime_ready=runtime_ready))
    raise typer.Exit(1)


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
) -> None:
    """Process an audio/video file into meeting notes."""
    config = _load_or_fail(config_path)
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
            task = progress.add_task(f"[{i+1}/{total}] {stage}...", total=None)
            try:
                if stage == "prepare":
                    manifest = _run_prepare(source, job_dir, manifest, config)
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
                    manifest = _run_finalize(job_dir, manifest, config, source, copy_to_input)
            except StageCancelledError:
                console.print(f"\n[yellow]Stage '{stage}' cancelled.[/yellow]")
                save_manifest(job_dir, manifest)
                raise typer.Exit(130)
            except Exception as e:
                update_stage_status(manifest, stage, "failed", error=str(e))
                save_manifest(job_dir, manifest)
                console.print(f"\n[red]Stage '{stage}' failed:[/red] {e}")
                raise typer.Exit(1)
            progress.update(task, completed=True, description=f"[{i+1}/{total}] {stage} done")

    save_manifest(job_dir, manifest)
    console.print(f"\n[green]Pipeline complete.[/green] Job: {job_dir}")
    template = job_dir.resolve() / "speakers.yaml"
    if template.exists():
        console.print(f"Speaker template: {template}")
        console.print(f"Next: uv run meeting-notes speakers apply \"{job_dir.resolve()}\"")


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
    console.print(f"  Summarization: {'enabled' if config.summarization.enabled else 'disabled'}")
    console.print(f"\n  Source: {source}")
    console.print(f"  Job dir: {job_dir}")
    console.print(f"\n[bold]Planned stages:[/bold]")
    for i, stage in enumerate(stages, 1):
        console.print(f"  [{i}/{len(stages)}] {stage}")


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

        # Get ASR backend (only pass executable for whisper_cpp)
        backend_kwargs = {}
        if config.runtime.asr_backend == "whisper_cpp":
            backend_kwargs["executable"] = config.runtime.whisper_cpp_path
            from meeting_notes.runtime import find_manifest_for_executable

            executable = Path(config.runtime.whisper_cpp_path).resolve()
            runtime_manifest = find_manifest_for_executable(executable)
            runtime_identity = {
                "backend": "whisper_cpp",
                "device": config.runtime.device,
                "executable": str(executable),
                "managed": runtime_manifest is not None,
                "runtime_version": (
                    runtime_manifest.get("version") if runtime_manifest else None
                ),
                "runtime_backend": (
                    runtime_manifest.get("backend") if runtime_manifest else None
                ),
                "source_revision": (
                    runtime_manifest.get("source_revision") if runtime_manifest else None
                ),
                "model": config.asr.model,
                "model_path": config.asr.model_path,
            }
            manifest["stages"]["transcribe"]["runtime"] = runtime_identity
            log.info("asr.runtime_selected", **runtime_identity)

        backend = get_backend(config.runtime.asr_backend, **backend_kwargs)

        if not backend.is_available():
            raise DependencyMissingError(
                f"ASR backend '{config.runtime.asr_backend}' is not available."
            )

        # Resolve the configured automatic CPU thread policy. whisper-cli's
        # own default is only four threads, which underuses modern CPUs.
        threads = config.runtime.threads
        if config.runtime.asr_backend == "whisper_cpp":
            threads = _resolve_whisper_threads(config)
        if config.runtime.asr_backend == "whisper_cpp" and config.runtime.threads <= 0:
            log.info(
                "whisper_cpp.threads_auto_selected",
                threads=threads,
                logical_cores=os.cpu_count() or 1,
            )

        # Run transcription
        transcribe_kwargs = {
            "model": config.asr.model,
            "model_path": Path(config.asr.model_path) if config.asr.model_path else None,
            "language": config.asr.language,
            "task": config.asr.task,
            "initial_prompt": config.asr.initial_prompt,
            "word_timestamps": config.asr.word_timestamps,
            "threads": threads,
        }
        if config.runtime.asr_backend == "whisper_cpp":
            whisper_options = config.asr.backend_options.whisper_cpp
            transcribe_kwargs.update(
                {
                    "device": config.runtime.device,
                    "model_variant": whisper_options.model_variant,
                    "flash_attention": whisper_options.flash_attention,
                    "extra_args": whisper_options.extra_args,
                    "gpu_device": whisper_options.gpu_device,
                }
            )
        result = backend.transcribe(
            normalized,
            **transcribe_kwargs,
        )

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


def _resolve_whisper_threads(config: MeetingNotesConfig) -> int:
    """Resolve zero/automatic thread configuration for whisper.cpp."""
    if config.runtime.threads > 0:
        return config.runtime.threads

    logical_cores = os.cpu_count() or 1
    threads = max(1, logical_cores - config.runtime.reserve_logical_cores)
    if config.runtime.max_auto_threads > 0:
        threads = min(threads, config.runtime.max_auto_threads)
    return threads


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
            config.diarization.model_path
            and Path(config.diarization.model_path).exists()
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

        # Import and try to load pyannote
        try:
            from meeting_notes.diarization.pyannote import PyannoteDiarizationBackend

            backend = PyannoteDiarizationBackend(
                model_name=config.diarization.model,
                model_path=(
                    Path(config.diarization.model_path)
                    if config.diarization.model_path
                    else None
                ),
                token_env=config.diarization.token_env,
                device=config.diarization.device,
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
            for s, ts in zip(segments, ts_segments):
                s["speaker"] = ts.speaker

        # Apply glossary corrections if enabled
        if config.glossary.enabled:
            from meeting_notes.transcript.glossary import correct_transcript_segments, load_glossary
            from pathlib import Path as P

            glossary = load_glossary(P(config.glossary.path) if config.glossary.path else None)
            if glossary.terms:
                segments, corrections = correct_transcript_segments(
                    segments,
                    glossary,
                    case_sensitive=config.glossary.case_sensitive,
                )

        # Save merged transcript
        merged_path = job_dir / "transcript" / "transcript.merged.json"
        merged_data = {
            "metadata": raw_data.get("metadata", {}),
            "segments": segments,
        }
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_text(json.dumps(merged_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Speaker identification is a downstream-only workflow. Never overwrite
        # a map the user may already have edited.
        from meeting_notes.speakers import write_template
        template_path, template_warning = write_template(job_dir, automatic=True)
        if template_warning:
            console.print(f"[yellow]  Speaker map: {template_warning}; candidate: {template_path}[/yellow]")
        elif template_path:
            manifest["speaker_template"] = {
                "version": 1,
                "path": str(template_path),
                "transcript_sha256": __import__("hashlib").sha256(merged_path.read_bytes()).hexdigest(),
            }

        update_stage_status(manifest, "merge", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "merge", "failed")
        raise


def _run_summarize(job_dir: Path, manifest: dict, config: MeetingNotesConfig, local_only: bool = False) -> dict:
    """Stage: Run summarization."""
    update_stage_status(manifest, "summarize", "running")
    try:
        if not config.summarization.enabled:
            update_stage_status(manifest, "summarize", "skipped")
            return manifest

        import json
        from meeting_notes.summarization.adapters import get_adapter, detect_available_adapters

        # Load transcript text
        merged_path = job_dir / "transcript" / "transcript.merged.json"
        if not merged_path.exists():
            merged_path = job_dir / "asr" / "transcript.raw.json"

        if not merged_path.exists():
            raise FileNotFoundError("No transcript found for summarization")

        raw_data = json.loads(merged_path.read_text(encoding="utf-8"))
        segments = raw_data.get("segments", [])

        transcript_text = _format_summary_transcript(segments)

        # Load prompt
        prompt_path = Path(config.summarization.prompt_path)
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
        else:
            prompt = "Summarize this meeting transcript."

        # Load schema
        schema_path = Path(config.summarization.output_schema_path) if config.summarization.output_schema_path else None

        # Get adapter and run
        adapter_name = config.summarization.backend

        # Check local-only mode
        if local_only and adapter_name in ("codex", "codex_cli", "opencode", "mimo", "claude"):
            console.print(f"[yellow]  Summarization adapter '{adapter_name}' rejected: --local-only mode[/yellow]")
            update_stage_status(manifest, "summarize", "skipped")
            return manifest

        adapter_kwargs = {}
        if adapter_name in ("codex", "codex_cli"):
            codex_options = config.summarization.codex
            adapter_kwargs = {
                "executable": codex_options.executable,
                "model": codex_options.model,
                "reasoning_effort": codex_options.reasoning_effort,
                "ephemeral": codex_options.ephemeral,
                "skip_git_repo_check": codex_options.skip_git_repo_check,
                "ignore_user_config": codex_options.ignore_user_config,
                "ignore_rules": codex_options.ignore_rules,
                "extra_args": codex_options.extra_args,
            }
        elif adapter_name == "local_command":
            local_options = config.summarization.local_command
            adapter_kwargs = {
                "command": local_options.command,
                "environment": local_options.environment,
            }

        adapter = get_adapter(adapter_name, **adapter_kwargs)
        if not adapter.is_available():
            console.print(f"[yellow]  Summarization adapter '{adapter_name}' not available[/yellow]")
            update_stage_status(manifest, "summarize", "skipped")
            return manifest

        result = adapter.summarize(
            transcript_text,
            prompt=prompt,
            schema_path=schema_path,
            timeout_seconds=config.summarization.timeout_seconds,
        )

        # Save summary
        summary_path = job_dir / "summary" / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(result.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        transcript_lines.append(
            f"[{segment_id}] [{timestamp}]{speaker} {segment['text']}"
        )
    return "\n".join(transcript_lines)


def _run_render(job_dir: Path, manifest: dict, config: MeetingNotesConfig) -> dict:
    """Stage: Render meeting minutes."""
    update_stage_status(manifest, "render", "running")
    try:
        import json
        from meeting_notes.minutes.render import render_minutes, save_minutes
        from meeting_notes.audio.inspect import MediaInfo

        # Load summary
        summary_path = job_dir / "summary" / "summary.json"
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


def _run_finalize(job_dir: Path, manifest: dict, config: MeetingNotesConfig, source: Path, copy_to_input: bool = False) -> dict:
    """Stage: Finalize recording and note filenames."""
    update_stage_status(manifest, "finalize", "running")
    try:
        import json
        import shutil
        from meeting_notes.naming import (
            generate_filenames,
            resolve_collision,
            resolve_date,
            sanitize_short_title,
        )

        # Load summary
        summary_path = job_dir / "summary" / "summary.json"
        if not summary_path.exists():
            console.print("[yellow]  No summary found, skipping finalize[/yellow]")
            update_stage_status(manifest, "finalize", "skipped")
            return manifest

        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # Get short title from summary
        short_title = sanitize_short_title(
            summary.get("short_title", "meeting"),
            max_length=config.naming.max_short_title_characters,
        )

        # Resolve date
        media_creation = manifest.get("source", {}).get("creation_time", "")
        source_mtime = source.stat().st_mtime if source.exists() else 0.0
        date, date_source = resolve_date(
            summary,
            media_creation_time=media_creation,
            source_mtime=source_mtime,
            source_order=config.naming.date_source_order,
        )

        # Get original extension
        original_ext = source.suffix if source.exists() else ".m4a"

        # Generate filenames
        filenames = generate_filenames(
            date,
            short_title,
            original_ext,
            recording_template=config.naming.recording_template,
            minutes_template=config.naming.minutes_template,
        )

        console.print(f"  Finalizing: {date} {short_title}")

        # Copy/renamed recording
        source_in_job = job_dir / "source" / source.name
        if source_in_job.exists() and config.naming.recording_mode != "none":
            target_recording = job_dir / "output" / "finalized" / filenames["recording"]
            target_recording.parent.mkdir(parents=True, exist_ok=True)

            if config.naming.recording_mode == "in_place":
                # Rename original in its directory
                new_path = source.parent / filenames["recording"]
                new_path = resolve_collision(new_path, policy=config.naming.collision_policy)
                os.rename(str(source), str(new_path))
                console.print(f"  Renamed: {source.name} -> {new_path.name}")
                manifest["source"]["finalized_path"] = str(new_path)
            else:
                # managed_copy: copy to finalized dir
                target_recording = resolve_collision(target_recording, policy=config.naming.collision_policy)
                shutil.copy2(source_in_job, target_recording)
                console.print(f"  Copied: -> {target_recording.name}")

        # Copy minutes to finalized
        minutes_src = job_dir / "output" / "minutes.md"
        if minutes_src.exists():
            target_minutes = job_dir / "output" / "finalized" / filenames["minutes"]
            target_minutes.parent.mkdir(parents=True, exist_ok=True)
            target_minutes = resolve_collision(target_minutes, policy=config.naming.collision_policy)
            shutil.copy2(minutes_src, target_minutes)
            console.print(f"  Copied: -> {target_minutes.name}")

        # Copy summary to finalized
        summary_src = job_dir / "output" / "summary.json"
        if summary_src.exists():
            target_summary = job_dir / "output" / "finalized" / filenames["json_export"]
            target_summary.parent.mkdir(parents=True, exist_ok=True)
            target_summary = resolve_collision(target_summary, policy=config.naming.collision_policy)
            shutil.copy2(summary_src, target_summary)
            console.print(f"  Copied: -> {target_summary.name}")

        # Update manifest
        manifest["finalized"] = {
            "date": date,
            "date_source": date_source,
            "short_title": short_title,
            "filenames": filenames,
        }

        # Optionally copy finalized files next to input recording
        if copy_to_input and source.exists():
            finalized_dir = job_dir / "output" / "finalized"
            if finalized_dir.exists():
                for f in finalized_dir.iterdir():
                    if f.is_file():
                        target = source.parent / f.name
                        shutil.copy2(f, target)
                        console.print(f"  Copied to input dir: {f.name}")

        update_stage_status(manifest, "finalize", "completed")
        return manifest
    except Exception:
        update_stage_status(manifest, "finalize", "failed")
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
    console.print(f"[green]Transcribe complete.[/green]")


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
        console.print(f"Next: uv run meeting-notes speakers apply \"{Path(job_dir).resolve()}\"")


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
    from meeting_notes.naming import generate_filenames, sanitize_short_title, resolve_date

    config = _load_or_fail(config_path)
    summary_path = Path(job_dir) / "summary" / "summary.json"

    if not summary_path.exists():
        console.print("[yellow]No summary found. Run summarize first.[/yellow]")
        return

    import json
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    short_title = sanitize_short_title(summary.get("short_title", "meeting"))
    date, date_source = resolve_date(summary, source_order=config.naming.date_source_order)
    original_ext = Path(manifest.get("source", {}).get("original_filename", ".m4a")).suffix

    filenames = generate_filenames(
        date, short_title, original_ext,
        recording_template=config.naming.recording_template,
        minutes_template=config.naming.minutes_template,
    )

    console.print(f"\n[bold]Filename preview[/bold]")
    console.print(f"  Date source: {date_source}")
    console.print(f"  Date: {date}")
    console.print(f"  Short title: {short_title}")
    console.print(f"  Recording: {filenames['recording']}")
    console.print(f"  Minutes: {filenames['minutes']}")
    console.print(f"  JSON export: {filenames['json_export']}")


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

    console.print(f"\n[green]Benchmark complete.[/green]")
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
        console.print(f"[green]Cleaned all artifacts[/green]")
