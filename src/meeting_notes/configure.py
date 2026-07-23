"""Configuration wizard, diagnostics display, and config management commands."""

from __future__ import annotations

import json
import os
import sys
import webbrowser
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
        _create_safe_defaults(config_path)
        return

    if not _is_tty():
        console.print(
            "[yellow]Non-interactive environment detected.[/yellow]\n"
            "Run: meeting-notes configure --accept-defaults\n"
            f"Or create config at: {config_path or DEFAULT_CONFIG_PATH}"
        )
        raise typer.Exit(1)

    _run_interactive_wizard(config_path)


def _create_safe_defaults(config_path: str | None = None) -> None:
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

    save_config(config, target)
    console.print(f"[green]Safe CPU configuration written to: {target}[/green]")
    console.print("You can now run: meeting-notes process <audio-file>")


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
            "backend": "codex_cli" if enable_summarization else "none",
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

    save_config(config, target)
    console.print(f"[green]Configuration saved to: {target}[/green]")
    console.print("\nNext steps:")
    console.print("  meeting-notes process <audio-file>")
    console.print("  meeting-notes models download <model> --backend <backend>")


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


def run_doctor(output_json: bool = False) -> None:
    """Run environment diagnostics."""
    diag = detect_system()

    if output_json:
        console.print_json(json.dumps(_diagnostics_to_dict(diag), indent=2))
    else:
        console.print(format_diagnostics_table(diag))
        console.print()

        # Compatibility recommendations
        console.print("[bold]Recommendations[/bold]\n")
        if not diag.tools.ffmpeg_available:
            console.print("  [yellow]Install FFmpeg: https://ffmpeg.org/download.html[/yellow]")
        if not diag.tools.whisper_cpp_available:
            console.print("  [yellow]Install whisper.cpp or build from source: https://github.com/ggerganov/whisper.cpp[/yellow]")
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


def run_models_status() -> None:
    """Show model download status."""
    console.print("[dim]Model status: not yet implemented (Phase 2)[/dim]")


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


def run_models_download(model: str, backend: str = "whisper_cpp", yes: bool = False) -> None:
    """Download a model."""
    console.print(f"[dim]Model download for '{model}' ({backend}): not yet implemented (Phase 2)[/dim]")


def run_models_verify(model: str, backend: str = "whisper_cpp") -> None:
    """Verify a downloaded model."""
    console.print(f"[dim]Model verify for '{model}' ({backend}): not yet implemented (Phase 2)[/dim]")
