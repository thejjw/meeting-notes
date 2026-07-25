"""Configuration wizard, diagnostics display, and config management commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import structlog
import typer
from rich.console import Console
from rich.table import Table

from meeting_notes.config import (
    DEFAULT_CONFIG_PATH,
    MeetingNotesConfig,
    SetupConfig,
    load_config,
    save_config,
)
from meeting_notes.errors import ConfigNotFoundError, ConfigValidationError
from meeting_notes.resources import (
    SystemDiagnostics,
    check_model_fit,
    detect_system,
    format_diagnostics_table,
    get_resource_estimate,
)

log = structlog.get_logger()
console = Console(stderr=True)


def _is_tty() -> bool:
    """Check if stdin/stdout are TTYs."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def run_configure(
    config_path: str | None = None,
    accept_defaults: bool = False,
    show_detected: bool = False,
    no_configure: bool = False,
    provision: bool = False,
    yes: bool = False,
) -> None:
    """Run the configuration wizard or create safe defaults."""
    if no_configure:
        console.print("[red]Configuration required. Run 'meeting-notes configure' or use '--accept-defaults'.[/red]")
        raise typer.Exit(1)

    if show_detected:
        diag = detect_system()
        console.print(format_diagnostics_table(diag))
        return

    if accept_defaults:
        _create_safe_defaults(config_path, provision=provision, yes=yes)
        return

    if not _is_tty():
        console.print(
            "[yellow]Non-interactive environment detected.[/yellow]\n"
            "Run: meeting-notes configure --accept-defaults\n"
            f"Or create config at: {config_path or DEFAULT_CONFIG_PATH}"
        )
        raise typer.Exit(1)

    _run_interactive_wizard(config_path)


def _create_safe_defaults(
    config_path: str | None = None, *, provision: bool = False, yes: bool = False
) -> None:
    """Create a safe CPU-only configuration without interaction."""
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    diag = detect_system()
    config = MeetingNotesConfig(
        setup=SetupConfig(
            completed=True,
            profile="safe-cpu",
            created_at=None,
        ),
    )

    # Adjust model based on available memory
    if diag.memory.available_ram_gb < 3.0:
        config.asr.model = "small"

    if provision:
        _provision_config(config, yes=yes)
    save_config(config, target)
    console.print(f"[green]Safe CPU configuration written to: {target}[/green]")
    if provision:
        console.print("Runtime and model verified. You can now run: meeting-notes process <audio-file>")
    else:
        _print_provisioning_commands(config, target)


def _run_interactive_wizard(config_path: str | None = None) -> None:
    """Run the interactive configuration wizard."""
    diag = detect_system()

    console.print("\n[bold]meeting-notes Configuration Wizard[/bold]\n")
    console.print(format_diagnostics_table(diag))
    console.print()

    # Step 1: Backend selection
    console.print("[bold]Execution options[/bold]\n")
    backend_options = _build_backend_options(diag)
    for i, opt in enumerate(backend_options, 1):
        compat = opt["compatibility"]
        compat_color = "green" if compat == "available" else ("yellow" if "warning" in compat else "red")
        console.print(f"  [{i}] {opt['name']}")
        console.print(f"      Backend: {opt['backend']}")
        console.print(f"      Model: {opt['model']}")
        console.print(f"      Estimated memory: {opt['memory']}")
        console.print(f"      Recommended RAM: {opt['recommended_ram']}")
        console.print(f"      Compatibility: [{compat_color}]{compat}[/{compat_color}]")
        if opt.get("notes"):
            console.print(f"      Notes: {opt['notes']}")
        console.print()

    default_backend_idx = 1
    choice = typer.prompt("Select an option", default=str(default_backend_idx), type=int)
    if choice < 1 or choice > len(backend_options):
        console.print("[red]Invalid choice.[/red]")
        raise typer.Exit(1)
    selected_backend = backend_options[choice - 1]

    # Step 2: Model selection
    console.print(f"\n[bold]Models compatible with {selected_backend['name']}[/bold]\n")
    model_options = _build_model_options(diag, selected_backend["runtime_device"])
    for i, opt in enumerate(model_options, 1):
        fit = opt["fit"]
        fit_color = "green" if fit == "available" else ("yellow" if "warning" in fit else "red")
        console.print(f"  [{i}] {opt['name']} {'(default)' if i == 1 else ''}")
        console.print(f"      Accuracy: {opt['accuracy']}")
        console.print(f"      Download: {opt['download']}")
        console.print(f"      Runtime memory: {opt['runtime_memory']}")
        console.print(f"      Recommended RAM: {opt['recommended_ram']}")
        console.print(f"      Fit: [{fit_color}]{fit}[/{fit_color}]")
        console.print()

    default_model_idx = 1
    choice = typer.prompt("Select a model", default=str(default_model_idx), type=int)
    if choice < 1 or choice > len(model_options):
        console.print("[red]Invalid choice.[/red]")
        raise typer.Exit(1)
    selected_model = model_options[choice - 1]

    # Step 3: Language
    console.print("\n[bold]Language settings[/bold]\n")
    lang = typer.prompt("Default language", default="ko (Korean)")
    lang_code = lang.split()[0] if lang else "ko"

    # Step 4: Diarization
    console.print("\n[bold]Speaker diarization[/bold]\n")
    enable_diarization = typer.confirm("Enable speaker diarization?", default=False)

    # Step 5: Summarization
    console.print("\n[bold]Summarization[/bold]\n")
    if diag.tools.codex_available:
        console.print("[yellow]Codex CLI detected. Note: summarization sends transcript text to OpenAI.[/yellow]")
        enable_summarization = typer.confirm("Enable Codex CLI summarization?", default=False)
    else:
        console.print("[dim]Codex CLI not detected. Summarization disabled.[/dim]")
        enable_summarization = False

    # Step 6: Naming mode
    console.print("\n[bold]File naming[/bold]\n")
    console.print("  [1] managed_copy (default) — keep original, create finalized copy in job dir")
    console.print("  [2] in_place — rename original recording after success")
    console.print("  [3] none — no file renaming")
    naming_choice = typer.prompt("Select naming mode", default="1")
    naming_modes = {"1": "managed_copy", "2": "in_place", "3": "none"}
    naming_mode = naming_modes.get(naming_choice, "managed_copy")

    # Build config
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = MeetingNotesConfig(
        setup=SetupConfig(
            completed=True,
            profile=selected_backend["profile"],
        ),
        runtime={
            "device": selected_backend["runtime_device"],
            "asr_backend": selected_backend["runtime_asr_backend"],
        },
        asr={
            "model": selected_model["name"],
            "language": lang_code,
        },
        diarization={
            "enabled": enable_diarization,
        },
        summarization={
            "enabled": enable_summarization,
            "backend": "codex" if enable_summarization else "none",
        },
        naming={
            "recording_mode": naming_mode,
        },
    )

    # Preview and confirm
    console.print(f"\n[bold]Resolved configuration[/bold]\n")
    console.print(f"  Profile: {config.setup.profile}")
    console.print(f"  Backend: {config.runtime.asr_backend}")
    console.print(f"  Device: {config.runtime.device}")
    console.print(f"  Model: {config.asr.model}")
    console.print(f"  Language: {config.asr.language}")
    console.print(f"  Diarization: {'enabled' if config.diarization.enabled else 'disabled'}")
    console.print(f"  Summarization: {'enabled' if config.summarization.enabled else 'disabled'}")
    console.print(f"  Naming mode: {config.naming.recording_mode}")
    console.print(f"  Config path: {target}")
    console.print()

    if not typer.confirm("Save this configuration?", default=True):
        console.print("[yellow]Configuration cancelled.[/yellow]")
        raise typer.Exit(0)

    provision_runtime = typer.confirm("Provision the selected whisper.cpp runtime?", default=True)
    provision_model = typer.confirm("Download and verify the selected model?", default=True)
    try:
        if provision_runtime:
            _provision_runtime(config)
        if provision_model:
            _provision_model(config, yes=False, interactive=True)
    except Exception as exc:
        console.print(f"[red]Provisioning failed: {exc}[/red]")
        console.print("[yellow]The previous configuration and installs were left unchanged.[/yellow]")
        raise typer.Exit(1)

    save_config(config, target)
    console.print(f"[green]Configuration saved to: {target}[/green]")
    if provision_runtime and provision_model:
        console.print("\nRuntime and transcription model verified.")
    else:
        _print_provisioning_commands(config, target)
    if config.diarization.enabled:
        console.print(
            "\nSpeaker diarization requires one guided browser authorization/download step:\n"
            f"  uv run meeting-notes diarization setup --config \"{target}\"\n"
            "Then verify with:\n"
            f"  uv run meeting-notes doctor --config \"{target}\""
        )
    elif provision_runtime and provision_model:
        console.print("Run: meeting-notes process <audio-file>")


def _build_backend_options(diag: SystemDiagnostics) -> list[dict]:
    """Build the list of backend options for the wizard."""
    options = []

    # Safe CPU (always available)
    options.append({
        "name": "Safe CPU (default)",
        "backend": "whisper.cpp CPU",
        "model": "medium",
        "memory": "~2.1 GB",
        "recommended_ram": ">= 4 GB",
        "compatibility": "available",
        "notes": "slowest but most portable and least driver-dependent",
        "profile": "safe-cpu",
        "runtime_device": "cpu",
        "runtime_asr_backend": "whisper_cpp",
    })

    # Vulkan
    vulkan_compat = "detected, not yet validated" if diag.gpu.vulkan_devices else "not detected"
    vulkan_notes = ""
    if not diag.gpu.vulkan_devices:
        vulkan_notes = "Vulkan devices not found; requires Vulkan-capable GPU and driver"
    options.append({
        "name": "Vulkan acceleration",
        "backend": "whisper.cpp Vulkan",
        "model": "large-v3",
        "memory": "~3.9 GB + runtime headroom",
        "recommended_ram": ">= 8 GB",
        "compatibility": vulkan_compat,
        "notes": vulkan_notes,
        "profile": "vulkan",
        "runtime_device": "vulkan",
        "runtime_asr_backend": "whisper_cpp",
    })

    # ROCm/HIP
    rocm_compat = "detected" if diag.gpu.rocm_architectures else "unavailable in this environment"
    rocm_notes = ""
    if not diag.gpu.rocm_architectures:
        rocm_notes = "rocminfo and a HIP-enabled whisper.cpp build were not found"
    options.append({
        "name": "AMD ROCm/HIP",
        "backend": "whisper.cpp HIP/ROCm",
        "model": "large-v3",
        "memory": "~3.9 GB + runtime headroom",
        "recommended_ram": ">= 8 GB",
        "compatibility": rocm_compat,
        "notes": rocm_notes,
        "profile": "amd-rocm",
        "runtime_device": "rocm",
        "runtime_asr_backend": "whisper_cpp",
    })

    # CUDA
    cuda_compat = "available" if diag.gpu.cuda_available else "not available"
    cuda_notes = "" if diag.gpu.cuda_available else "NVIDIA CUDA not detected"
    options.append({
        "name": "NVIDIA CUDA",
        "backend": "whisper.cpp CUDA or faster-whisper",
        "model": "large-v3",
        "memory": f"~{diag.gpu.vram_gb:.1f} GB VRAM detected" if diag.gpu.cuda_available else "~10 GB VRAM needed",
        "recommended_ram": ">= 12 GB",
        "compatibility": cuda_compat,
        "notes": cuda_notes,
        "profile": "nvidia-cuda",
        "runtime_device": "cuda",
        "runtime_asr_backend": "whisper_cpp",
    })

    return options


def _build_model_options(diag: SystemDiagnostics, device: str = "cpu") -> list[dict]:
    """Build model options filtered by compatibility."""
    models = [
        ("small", "lower, but faster and lighter", "466 MiB", "~852 MB", "~2.9 GB"),
        ("medium", "better than small; lower than large-v3", "1.5 GiB", "~2.1 GB", "~4.1 GB"),
        ("large-v3", "highest standard Whisper profile", "2.9 GiB", "~3.9 GB", "~5.9 GB"),
        ("large-v3-turbo", "fast, near large-v3 accuracy", "~1.5 GiB", "~2.1 GB", "~4.1 GB"),
    ]

    options = []
    for name, accuracy, download, runtime_mem, recommended in models:
        est = get_resource_estimate(name, "whisper_cpp")
        fit = "unknown"
        if est:
            fit_status, _ = check_model_fit(est, diag)
            fit = fit_status

        options.append({
            "name": name,
            "accuracy": accuracy,
            "download": f"{download} [official reference]",
            "runtime_memory": f"{runtime_mem} [official reference]",
            "recommended_ram": f"{recommended} [estimated]",
            "fit": fit,
        })

    return options


def show_config(resolved: bool = False, config_path: str | None = None) -> None:
    """Show current configuration."""
    try:
        config = load_config(config_path)
    except (ConfigNotFoundError, ConfigValidationError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if resolved:
        console.print_json(config.model_dump_json(indent=2))
    else:
        from meeting_notes.config import DEFAULT_CONFIG_PATH, _resolve_config_path

        path = _resolve_config_path(config_path)
        console.print(f"Config path: {path}")
        console.print(f"Profile: {config.setup.profile}")
        console.print_json(config.model_dump_json(indent=2))


def config_status_cmd(config_path: str | None = None) -> None:
    """Show configuration status."""
    from meeting_notes.config import DEFAULT_CONFIG_PATH, _resolve_config_path

    path = _resolve_config_path(config_path)
    if path:
        console.print(f"[green]Config found: {path}[/green]")
        try:
            config = load_config(config_path)
            console.print(f"  Profile: {config.setup.profile}")
            console.print(f"  Backend: {config.runtime.asr_backend}")
            console.print(f"  Device: {config.runtime.device}")
            console.print(f"  Model: {config.asr.model}")
        except ConfigValidationError as e:
            console.print(f"[yellow]Config invalid: {e}[/yellow]")
    else:
        console.print("[yellow]No configuration found.[/yellow]")
        console.print(f"Expected at: {DEFAULT_CONFIG_PATH}")


def config_edit_cmd(config_path: str | None = None) -> None:
    """Open configuration file in editor."""
    from meeting_notes.config import DEFAULT_CONFIG_PATH, _resolve_config_path

    path = _resolve_config_path(config_path)
    if not path:
        console.print("[yellow]No config file to edit. Run 'meeting-notes configure' first.[/yellow]")
        raise typer.Exit(1)

    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "notepad"))
    if os.name == "nt":
        os.system(f'{editor} "{path}"')
    else:
        os.execlp(editor, editor, str(path))


def config_reset_cmd(config_path: str | None = None) -> None:
    """Reset configuration to safe defaults."""
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if target.exists():
        backup = target.with_suffix(f".yaml.{target.stat().st_mtime_ns}.bak")
        backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        console.print(f"[dim]Backup saved: {backup}[/dim]")

    _create_safe_defaults(config_path)


def _configured_checks(config_path: str | None) -> dict[str, object]:
    checks: dict[str, object] = {"configured": False}
    try:
        config = load_config(config_path)
    except (ConfigNotFoundError, ConfigValidationError) as exc:
        checks["config_error"] = str(exc)
        return checks

    from meeting_notes.models import verify_model
    from meeting_notes.runtime import find_manifest_for_executable

    executable = Path(config.runtime.whisper_cpp_path)
    if not executable.is_absolute() and shutil.which(config.runtime.whisper_cpp_path):
        executable = Path(shutil.which(config.runtime.whisper_cpp_path) or executable)
    manifest = find_manifest_for_executable(executable) if executable.exists() else None
    executable_runnable = False
    if executable.is_file() and config.runtime.asr_backend == "whisper_cpp":
        try:
            check = subprocess.run(
                [str(executable), "--help"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=10,
            )
            executable_runnable = check.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            executable_runnable = False
    model = Path(config.asr.model_path) if config.asr.model_path else None
    model_valid, model_detail = (
        verify_model(config.asr.model, model) if model else (False, "model_path is not configured")
    )
    from meeting_notes.diarization.setup import resolve_hf_token

    token, token_source = resolve_hf_token(config.diarization.token_env)
    diarization_model_path = (
        Path(config.diarization.model_path) if config.diarization.model_path else None
    )
    local_diarization_model_ready = bool(
        diarization_model_path and diarization_model_path.exists()
    )
    try:
        from importlib.metadata import version

        pyannote_version = version("pyannote.audio")
        pyannote_installed = True
    except Exception:
        pyannote_installed = False
        pyannote_version = None
    latest_transcript = _latest_transcript_metadata(config)
    checks.update(
        {
            "configured": True,
            "asr_backend": config.runtime.asr_backend,
            "device": config.runtime.device,
            "executable": str(executable),
            "executable_exists": executable.is_file(),
            "executable_runnable": executable_runnable,
            "manifest": manifest,
            "manifest_backend_matches": bool(
                manifest and manifest.get("backend") == config.runtime.device
            ),
            "model_name": config.asr.model,
            "model": str(model) if model else None,
            "model_verified": model_valid,
            "model_detail": model_detail,
            "diarization_enabled": config.diarization.enabled,
            "diarization_model": config.diarization.model,
            "diarization_model_path": (
                str(diarization_model_path) if diarization_model_path else None
            ),
            "local_diarization_model_ready": local_diarization_model_ready,
            "token_env": config.diarization.token_env,
            "pyannote_installed": pyannote_installed,
            "pyannote_version": pyannote_version,
            "hf_token_ready": bool(token),
            "hf_token_source": token_source,
            "latest_transcript": latest_transcript,
        }
    )
    return checks


def _latest_transcript_metadata(config: MeetingNotesConfig) -> dict[str, object] | None:
    """Return provenance recorded in the newest rendered raw transcript."""
    meetings_dir = Path(config.project.data_dir) / "meetings"
    if not meetings_dir.exists():
        return None
    candidates = list(meetings_dir.glob("*/asr/transcript.raw.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return {
        "path": str(latest.resolve()),
        "backend": metadata.get("backend"),
        "model": metadata.get("model"),
        "device": metadata.get("device"),
        "language": metadata.get("language"),
        "duration": metadata.get("duration"),
    }


def _diarization_recommendations(
    config: MeetingNotesConfig,
    checks: dict[str, object],
) -> list[str]:
    """Return exact setup steps for the selected pyannote backend."""
    if not config.diarization.enabled:
        return []
    lines: list[str] = []
    if not checks.get("pyannote_installed"):
        lines.extend(
            [
                "  Install the optional diarization dependencies into this uv project:",
                "    uv sync --extra diarization",
            ]
        )
    if not checks.get("hf_token_ready") and not checks.get("local_diarization_model_ready"):
        lines.extend(
            [
                "  Run the guided browser login, model-consent, and download flow:",
                "    uv run meeting-notes diarization setup",
                "  Hugging Face requires you to accept gated-model conditions in your browser;",
                "  meeting-notes cannot accept them on your behalf.",
            ]
        )
    if lines:
        lines.extend(
            [
                "  Verify, then resume without retranscribing:",
                "    uv run meeting-notes doctor",
                '    uv run meeting-notes process "<audio-file>" --from diarize',
            ]
        )
    return lines


def _run_smoke_test(config_path: str | None) -> dict[str, object]:
    config = load_config(config_path)
    executable = Path(config.runtime.whisper_cpp_path)
    model = Path(config.asr.model_path or "")
    if not executable.is_file():
        return {"success": False, "detail": f"configured executable missing: {executable}"}
    if not model.is_file():
        return {"success": False, "detail": f"configured model missing: {model}"}
    ffmpeg = config.runtime.ffmpeg_path
    with tempfile.TemporaryDirectory(prefix="meeting-notes-smoke-") as temp:
        wav = Path(temp) / "silence.wav"
        generated = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono",
                "-t",
                "0.25",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(wav),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if generated.returncode:
            return {"success": False, "detail": f"FFmpeg failed: {generated.stderr.strip()}"}
        command = [
            str(executable),
            "-m",
            str(model),
            "-f",
            str(wav),
            "--no-prints",
        ]
        if config.runtime.device == "cpu":
            command.append("--no-gpu")
        else:
            command.extend(
                [
                    "--device",
                    config.asr.backend_options.whisper_cpp.gpu_device or "0",
                ]
            )
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        output = f"{result.stdout}\n{result.stderr}"
        vulkan_seen = any(
            marker in output.lower() for marker in ("vulkan", "ggml_vk", "vk_instance")
        )
        success = result.returncode == 0 and (
            config.runtime.device != "vulkan" or vulkan_seen
        )
        detail = "model loaded successfully"
        if config.runtime.device == "vulkan" and not vulkan_seen:
            detail = "whisper.cpp output did not confirm Vulkan initialization"
        elif result.returncode:
            detail = f"whisper.cpp exited {result.returncode}: {result.stderr[-500:]}"
        return {"success": success, "vulkan_initialized": vulkan_seen, "detail": detail}


def run_doctor(
    output_json: bool = False,
    config_path: str | None = None,
    smoke_test: bool = False,
) -> None:
    """Run environment diagnostics."""
    diag = detect_system()

    configured = _configured_checks(config_path)
    smoke = _run_smoke_test(config_path) if smoke_test else None
    if output_json:
        payload = _diagnostics_to_dict(diag)
        payload["configuration"] = configured
        if smoke is not None:
            payload["smoke_test"] = smoke
        console.print_json(json.dumps(payload, indent=2))
    else:
        console.print(format_diagnostics_table(diag))
        console.print()
        console.print("[bold]Configured runtime[/bold]\n")
        if not configured["configured"]:
            console.print(f"  [yellow]{configured.get('config_error')}[/yellow]")
        else:
            exe_style = "green" if configured["executable_exists"] else "red"
            model_style = "green" if configured["model_verified"] else "red"
            runnable_style = "green" if configured["executable_runnable"] else "red"
            console.print(f"  ASR backend: [bold]{configured['asr_backend']}[/bold]")
            console.print(
                f"  Configured runtime: [{runnable_style}]"
                f"{'ready' if configured['executable_runnable'] else 'not runnable'}"
                f"[/{runnable_style}]"
            )
            console.print(
                f"  Executable: [{exe_style}]{configured['executable']}[/{exe_style}]"
            )
            if configured["manifest"] is None:
                console.print("  [yellow]Manifest: user-supplied/unmanaged executable[/yellow]")
            elif not configured["manifest_backend_matches"]:
                console.print("  [red]Manifest backend does not match configured device.[/red]")
            else:
                runtime_manifest = configured["manifest"]
                assert isinstance(runtime_manifest, dict)
                console.print(
                    "  Managed runtime: "
                    f"whisper.cpp {runtime_manifest.get('version', 'unknown')} / "
                    f"{runtime_manifest.get('backend', 'unknown')} / "
                    f"{runtime_manifest.get('platform', 'unknown')}-"
                    f"{runtime_manifest.get('architecture', 'unknown')}"
                )
                console.print(
                    f"  Source revision: {runtime_manifest.get('source_revision', 'unknown')}"
                )
                if runtime_manifest.get("checksum"):
                    console.print(f"  Archive SHA-256: {runtime_manifest['checksum']}")
            console.print(
                f"  Model: [{model_style}]{configured['model_name']} — "
                f"{configured['model_detail']}[/{model_style}]"
            )
            console.print(f"  Model path: {configured['model']}")
            latest = configured.get("latest_transcript")
            if isinstance(latest, dict):
                console.print(
                    "  Latest transcript recorded: "
                    f"{latest.get('backend')} / {latest.get('model')} / "
                    f"{latest.get('device')} ({latest.get('language')})"
                )
                console.print(f"  Transcript evidence: {latest.get('path')}")
            if configured["diarization_enabled"]:
                ready = configured["pyannote_installed"] and (
                    configured["hf_token_ready"]
                    or configured["local_diarization_model_ready"]
                )
                console.print(
                    f"  Diarization: "
                    f"{'[green]ready[/green]' if ready else '[yellow]not ready[/yellow]'}"
                )
                dependency = (
                    f"installed ({configured['pyannote_version']})"
                    if configured["pyannote_installed"]
                    else "missing"
                )
                token_status = (
                    f"ready ({configured['hf_token_source']})"
                    if configured["hf_token_ready"]
                    else "missing"
                )
                console.print(f"    pyannote.audio: {dependency}")
                console.print(
                    f"    Hugging Face authentication: {token_status} "
                    "(token value is never displayed)"
                )
                if configured["diarization_model_path"]:
                    local_status = (
                        "ready" if configured["local_diarization_model_ready"] else "missing"
                    )
                    console.print(
                        f"    Local model: {local_status} "
                        f"({configured['diarization_model_path']})"
                    )
        if smoke is not None:
            style = "green" if smoke["success"] else "red"
            console.print(f"\n  Smoke test: [{style}]{smoke['detail']}[/{style}]")

        # Compatibility recommendations
        console.print("[bold]Recommendations[/bold]\n")
        if not diag.tools.ffmpeg_available:
            console.print("  [yellow]Install FFmpeg: https://ffmpeg.org/download.html[/yellow]")
        configured_runtime_ready = bool(
            configured.get("configured")
            and configured.get("asr_backend") == "whisper_cpp"
            and configured.get("executable_runnable")
        )
        if not diag.tools.whisper_cpp_available and not configured_runtime_ready:
            console.print("  [yellow]Install whisper.cpp or build from source: https://github.com/ggerganov/whisper.cpp[/yellow]")
        elif not diag.tools.whisper_cpp_available and configured_runtime_ready:
            console.print(
                "  [green]No PATH installation is needed; the configured managed "
                "whisper.cpp runtime is ready.[/green]"
            )
        if configured.get("configured"):
            config = load_config(config_path)
            for line in _diarization_recommendations(config, configured):
                console.print(line)
        if not diag.tools.codex_available:
            console.print("  [dim]Codex CLI not installed. Summarization disabled. Install: https://github.com/openai/codex[/dim]")
        if not diag.gpu.available:
            console.print("  [dim]No GPU acceleration detected. Using CPU mode.[/dim]")


def _diagnostics_to_dict(diag: SystemDiagnostics) -> dict:
    """Convert diagnostics to serializable dict."""
    return {
        "os": diag.os_name,
        "os_version": diag.os_version,
        "architecture": diag.architecture,
        "python_version": diag.python_version,
        "is_wsl": diag.is_wsl,
        "cpu": {
            "model": diag.cpu.model_name,
            "physical_cores": diag.cpu.physical_cores,
            "logical_cores": diag.cpu.logical_cores,
        },
        "memory": {
            "total_ram_gb": diag.memory.total_ram_gb,
            "available_ram_gb": diag.memory.available_ram_gb,
            "is_unified_memory": diag.memory.is_unified_memory,
        },
        "gpu": {
            "available": diag.gpu.available,
            "backend": diag.gpu.backend,
            "device_name": diag.gpu.device_name,
            "vram_gb": diag.gpu.vram_gb,
            "cuda_available": diag.gpu.cuda_available,
            "vulkan_devices": diag.gpu.vulkan_devices,
            "rocm_architectures": diag.gpu.rocm_architectures,
        },
        "tools": {
            "ffmpeg": {"available": diag.tools.ffmpeg_available, "version": diag.tools.ffmpeg_version},
            "ffprobe": {"available": diag.tools.ffprobe_available, "version": diag.tools.ffprobe_version},
            "whisper_cpp": {"available": diag.tools.whisper_cpp_available, "version": diag.tools.whisper_cpp_version},
            "codex": {"available": diag.tools.codex_available, "version": diag.tools.codex_version},
        },
    }


def run_resources_show(model: str | None = None, device: str | None = None) -> None:
    """Show resource estimates."""
    diag = detect_system()
    console.print(format_diagnostics_table(diag))
    console.print()

    models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"] if not model else [model]

    table = Table(title="Resource Estimates (whisper.cpp)")
    table.add_column("Model", style="cyan")
    table.add_column("Disk", justify="right")
    table.add_column("RAM", justify="right")
    table.add_column("Rec. Free RAM", justify="right")
    table.add_column("Fit", justify="center")

    for m in models:
        est = get_resource_estimate(m, "whisper_cpp")
        if est:
            fit_status, _ = check_model_fit(est, diag)
            fit_style = "green" if fit_status == "available" else ("yellow" if "warning" in fit_status else "red")
            table.add_row(
                m,
                f"{est.disk_size_mib} MiB",
                f"{est.reference_memory_mb} MB",
                f"{est.recommended_free_ram_gb:.1f} GB",
                f"[{fit_style}]{fit_status}[/{fit_style}]",
            )
        else:
            table.add_row(m, "?", "?", "?", "[dim]unknown[/dim]")

    console.print(table)


def run_models_list() -> None:
    """List available models."""
    console.print("[bold]Available Whisper models[/bold]\n")
    for name in ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]:
        est = get_resource_estimate(name, "whisper_cpp")
        if est:
            console.print(f"  {name:20s}  {est.disk_size_mib:>6} MiB  ~{est.reference_memory_mb} MB RAM")
        else:
            console.print(f"  {name:20s}  [dim]metadata unknown[/dim]")


def _target_config_path(config_path: str | None) -> Path:
    from meeting_notes.config import _resolve_config_path

    resolved = _resolve_config_path(config_path)
    if resolved is None:
        raise ConfigNotFoundError(
            "No configuration found. Run 'meeting-notes configure --accept-defaults' first."
        )
    return resolved


def _provision_runtime(config: MeetingNotesConfig, version: str = "v1.9.1") -> Path:
    from meeting_notes.runtime import install_cpu, install_vulkan

    if config.runtime.device not in {"cpu", "vulkan"}:
        raise RuntimeError(
            f"Managed whisper.cpp supports cpu or vulkan, not '{config.runtime.device}'."
        )
    executable = (
        install_vulkan(version) if config.runtime.device == "vulkan" else install_cpu(version)
    )
    config.runtime.whisper_cpp_path = str(executable.resolve())
    return executable


def _provision_model(
    config: MeetingNotesConfig, *, yes: bool, interactive: bool = False
) -> Path:
    from meeting_notes.artifacts import MODEL_ARTIFACTS
    from meeting_notes.models import download_model

    metadata = MODEL_ARTIFACTS.get(config.asr.model)
    if metadata is None:
        raise RuntimeError(f"No managed GGML artifact for model '{config.asr.model}'.")
    threshold = config.resources.large_download_confirmation_mib * 1024 * 1024
    is_large = int(metadata["size"]) >= threshold
    if is_large and not yes:
        if interactive and typer.confirm(
            f"{config.asr.model} is {int(metadata['size']) / 1024**3:.1f} GiB. Download it?",
            default=False,
        ):
            pass
        else:
            raise RuntimeError(
                f"{config.asr.model} is a large download; rerun with --yes to confirm."
            )
    path = download_model(config.asr.model)
    config.asr.model_path = str(path.resolve())
    config.asr.model_cache_dir = str(path.parent.resolve())
    return path


def _provision_config(config: MeetingNotesConfig, *, yes: bool) -> None:
    _provision_runtime(config)
    _provision_model(config, yes=yes)


def _print_provisioning_commands(config: MeetingNotesConfig, target: Path) -> None:
    console.print("\n[bold]Provisioning still required[/bold]")
    console.print(
        f"  meeting-notes runtime install --device {config.runtime.device} --config \"{target}\""
    )
    suffix = " --yes" if config.asr.model in {"medium", "large-v3", "large-v3-turbo"} else ""
    console.print(
        f"  meeting-notes models download {config.asr.model} --config \"{target}\"{suffix}"
    )


def run_models_status(output_json: bool = False) -> None:
    """Show model download status."""
    from meeting_notes.models import model_statuses

    statuses = model_statuses()
    if output_json:
        console.print_json(json.dumps(statuses, indent=2))
        return
    table = Table(title="Managed GGML Models")
    table.add_column("Model")
    table.add_column("Installed")
    table.add_column("Verified")
    table.add_column("Path")
    for item in statuses:
        table.add_row(
            str(item["name"]),
            "yes" if item["installed"] else "no",
            "yes" if item["verified"] else "no",
            str(item["path"]),
        )
    console.print(table)


def run_models_info(model: str, backend: str = "whisper_cpp") -> None:
    """Show model information."""
    est = get_resource_estimate(model, backend)
    if est:
        console.print(f"Model: {est.model_name}")
        console.print(f"Backend: {est.backend}")
        console.print(f"Disk: {est.disk_size_mib} MiB")
        console.print(f"Runtime memory: {est.reference_memory_mb} MB")
        console.print(f"Recommended free RAM: {est.recommended_free_ram_gb:.1f} GB")
        console.print(f"Confidence: {est.confidence}")
        console.print(f"Source: {est.source}")
    else:
        console.print(f"[yellow]No resource data for model '{model}' with backend '{backend}'[/yellow]")


def run_models_download(
    model: str,
    backend: str = "whisper_cpp",
    yes: bool = False,
    config_path: str | None = None,
) -> None:
    """Download a model."""
    if backend != "whisper_cpp":
        console.print("[red]Managed downloads currently support only --backend whisper_cpp.[/red]")
        raise typer.Exit(2)
    from meeting_notes.artifacts import MODEL_ARTIFACTS
    from meeting_notes.models import ModelInstallError, download_model

    metadata = MODEL_ARTIFACTS.get(model)
    if metadata is None:
        console.print(f"[red]Unknown model '{model}'.[/red]")
        raise typer.Exit(2)
    if int(metadata["size"]) >= 1024**3 and not yes:
        console.print(
            f"[yellow]{model} is {int(metadata['size']) / 1024**3:.1f} GiB. "
            "Rerun with --yes to confirm.[/yellow]"
        )
        raise typer.Exit(1)
    try:
        path = download_model(model)
        if config_path is not None:
            target = _target_config_path(config_path)
            config = load_config(str(target))
            config.asr.model = model
            config.asr.model_path = str(path)
            config.asr.model_cache_dir = str(path.parent)
            save_config(config, target)
        console.print(f"[green]Verified model installed: {path}[/green]")
    except (ModelInstallError, OSError) as exc:
        console.print(f"[red]Model installation failed: {exc}[/red]")
        raise typer.Exit(1)


def run_models_verify(
    model: str, backend: str = "whisper_cpp", config_path: str | None = None
) -> None:
    """Verify a downloaded model."""
    if backend != "whisper_cpp":
        console.print("[red]Managed verification supports only whisper_cpp.[/red]")
        raise typer.Exit(2)
    from meeting_notes.models import model_path, verify_model

    path = model_path(model)
    if config_path is not None:
        try:
            config = load_config(config_path)
            if config.asr.model == model and config.asr.model_path:
                path = Path(config.asr.model_path)
        except (ConfigNotFoundError, ConfigValidationError):
            pass
    valid, detail = verify_model(model, path)
    if not valid:
        console.print(f"[red]{model}: {detail}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{model}: verified ({path.resolve()})[/green]")


def run_runtime_status(output_json: bool = False) -> None:
    """Show managed runtime status."""
    from meeting_notes.runtime import installed_runtimes

    runtimes = installed_runtimes()
    if output_json:
        console.print_json(json.dumps(runtimes, indent=2))
        return
    if not runtimes:
        console.print("[yellow]No managed whisper.cpp runtimes installed.[/yellow]")
        return
    table = Table(title="Managed whisper.cpp Runtimes")
    for heading in ("Version", "Backend", "Platform", "Architecture", "Healthy", "Executable"):
        table.add_column(heading)
    for item in runtimes:
        table.add_row(
            str(item.get("version", "")),
            str(item.get("backend", "")),
            str(item.get("platform", "")),
            str(item.get("architecture", "")),
            "yes" if item.get("healthy") else "no",
            str(item.get("executable_path", "")),
        )
    console.print(table)


def run_runtime_install(
    device: str = "cpu",
    version: str = "v1.9.1",
    config_path: str | None = None,
    yes: bool = False,
) -> None:
    """Install a managed runtime and update config after success."""
    del yes  # Reserved for symmetry and future source-build confirmations.
    if device not in {"cpu", "vulkan"}:
        console.print("[red]--device must be cpu or vulkan.[/red]")
        raise typer.Exit(2)
    from meeting_notes.runtime import RuntimeInstallError, install_cpu, install_vulkan

    try:
        executable = install_vulkan(version) if device == "vulkan" else install_cpu(version)
        if config_path is not None:
            target = _target_config_path(config_path)
            config = load_config(str(target))
            config.runtime.device = device
            config.runtime.asr_backend = "whisper_cpp"
            config.runtime.whisper_cpp_path = str(executable.resolve())
            save_config(config, target)
        console.print(f"[green]Verified {device} runtime installed: {executable.resolve()}[/green]")
    except (RuntimeInstallError, OSError) as exc:
        console.print(f"[red]Runtime installation failed: {exc}[/red]")
        raise typer.Exit(1)
