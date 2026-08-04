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
from meeting_notes.storage import project_cache_root

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
        console.print(
            "[red]Configuration required. Run 'meeting-notes configure' "
            "or use '--accept-defaults'.[/red]"
        )
        raise typer.Exit(1)

    if show_detected:
        diag = detect_system()
        console.print(format_diagnostics_table(diag))
        from meeting_notes.diarization.acceleration import probe_rocm

        rocm = probe_rocm(MeetingNotesConfig())
        console.print(f"\nDiarization ROCm hybrid: {rocm.state}\n  {rocm.detail}")
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
    if diag.memory.available_ram_gb < 4.1:
        config.asr.model = "small"

    if provision:
        _provision_config(config, yes=yes)
    save_config(config, target)
    console.print(f"[green]Safe CPU configuration written to: {target}[/green]")
    if provision:
        console.print(
            "Runtime and model verified. You can now run: meeting-notes process <audio-file>"
        )
    else:
        _print_provisioning_commands(config, target)


def _prompt_summarization_config(
    diag: SystemDiagnostics,
) -> tuple[str, str | None, str | None]:
    """Prompt for an installed summarizer and its requested model."""
    available: list[tuple[str, str]] = []
    if diag.tools.codex_available:
        available.append(("codex", "Codex CLI"))
    if diag.tools.claude_available:
        available.append(("claude", "Claude Code"))
    from meeting_notes.summarization.adapters import LemonadeAdapter

    lemonade_url = "http://127.0.0.1:13305"
    lemonade_ready = LemonadeAdapter(
        base_url=lemonade_url,
        connect_timeout_seconds=0.75,
    ).is_available()
    lemonade_state = "detected" if lemonade_ready else "start server manually"
    available.append(("lemonade", f"AMD Lemonade local Markdown ({lemonade_state})"))

    console.print(
        "[yellow]Summarization sends transcript text to the selected AI provider.[/yellow]"
    )
    for index, (_, label) in enumerate(available, 1):
        console.print(f"  [{index}] {label}")
    disabled_index = len(available) + 1
    console.print(f"  [{disabled_index}] Disabled (default)")
    choice = typer.prompt("Select summarization backend", default=str(disabled_index), type=int)
    if choice == disabled_index:
        return "none", None, None
    if choice < 1 or choice > len(available):
        console.print("[red]Invalid summarization backend.[/red]")
        raise typer.Exit(1)

    backend = available[choice - 1][0]
    if backend == "lemonade":
        choices = [
            (
                "Gemma-4-26B-A4B-it-MTP-GGUF",
                "Gemma 4 26B A4B MTP (recommended local model)",
            ),
            ("custom", "Custom Lemonade model ID"),
        ]
    elif backend == "codex":
        choices = [
            ("gpt-5.6-terra", "GPT-5.6 Terra (recommended: balanced cost and quality)"),
            ("gpt-5.6-sol", "GPT-5.6 Sol (flagship quality)"),
            (None, "Provider default"),
            ("custom", "Custom model ID"),
        ]
    else:
        choices = [
            ("sonnet", "Latest Sonnet (recommended: balanced cost and quality)"),
            ("opus", "Latest Opus (flagship quality)"),
            (None, "Provider default"),
            ("custom", "Custom model ID"),
        ]
    for index, (_, label) in enumerate(choices, 1):
        console.print(f"  [{index}] {label}")
    model_choice = typer.prompt("Select summarization model", default="1", type=int)
    if model_choice < 1 or model_choice > len(choices):
        console.print("[red]Invalid summarization model.[/red]")
        raise typer.Exit(1)
    model = choices[model_choice - 1][0]
    if model == "custom":
        model = typer.prompt("Model ID").strip()
        if not model:
            console.print("[red]Model ID cannot be blank.[/red]")
            raise typer.Exit(1)
    reasoning_effort: str | None = None
    if backend in {"codex", "claude"}:
        effort_choices = [
            (None, "Provider default (recommended)"),
            ("low", "Low (fastest and most economical)"),
            ("medium", "Medium (balanced)"),
            ("high", "High (more thorough)"),
            ("custom", "Custom reasoning effort"),
        ]
        provider_label = "Codex" if backend == "codex" else "Claude"
        console.print(f"\n{provider_label} reasoning effort")
        for index, (_, label) in enumerate(effort_choices, 1):
            console.print(f"  [{index}] {label}")
        effort_choice = typer.prompt("Select reasoning effort", default="1", type=int)
        if effort_choice < 1 or effort_choice > len(effort_choices):
            console.print("[red]Invalid reasoning effort.[/red]")
            raise typer.Exit(1)
        reasoning_effort = effort_choices[effort_choice - 1][0]
        if reasoning_effort == "custom":
            reasoning_effort = typer.prompt("Reasoning effort").strip()
            if not reasoning_effort:
                console.print("[red]Reasoning effort cannot be blank.[/red]")
                raise typer.Exit(1)
    return backend, model, reasoning_effort


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
        compat_color = (
            "green" if compat == "available" else ("yellow" if "warning" in compat else "red")
        )
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

    lemonade_url: str | None = None
    if selected_backend["runtime_asr_backend"] == "lemonade":
        default_url = "http://127.0.0.1:13305"
        lemonade_url = (
            typer.prompt(
                "Lemonade Server URL",
                default=default_url,
            )
            .strip()
            .rstrip("/")
        )
        from meeting_notes.asr.lemonade import LemonadeASRBackend

        lemonade = LemonadeASRBackend(
            base_url=lemonade_url,
            connect_timeout_seconds=1.5,
        )
        if lemonade.is_available():
            console.print(f"[green]Lemonade Server found at {lemonade_url}.[/green]")
        else:
            console.print(
                f"[yellow]Lemonade Server is not reachable at {lemonade_url}.[/yellow]\n"
                "Start Lemonade Server manually, then retry provisioning. "
                "You can verify it with: lemonade status"
            )

    # Step 2: Model selection
    console.print(f"\n[bold]Models compatible with {selected_backend['name']}[/bold]\n")
    if selected_backend["runtime_asr_backend"] == "lemonade":
        model_options = [
            {
                "name": "large-v3-turbo",
                "accuracy": "recommended multilingual Whisper model",
                "download": "~1.5 GiB, managed by Lemonade",
                "runtime_memory": "managed by Lemonade",
                "recommended_ram": "managed by Lemonade",
                "fit": "available through Lemonade catalogue",
            }
        ]
    else:
        model_options = _build_model_options(diag, selected_backend["runtime_device"])
    default_model_idx = _default_model_index(model_options)
    for i, opt in enumerate(model_options, 1):
        fit = opt["fit"]
        fit_color = "green" if fit == "available" else ("yellow" if "warning" in fit else "red")
        console.print(f"  [{i}] {opt['name']} {'(default)' if i == default_model_idx else ''}")
        console.print(f"      Accuracy: {opt['accuracy']}")
        console.print(f"      Download: {opt['download']}")
        console.print(f"      Runtime memory: {opt['runtime_memory']}")
        console.print(f"      Recommended RAM: {opt['recommended_ram']}")
        console.print(f"      Fit: [{fit_color}]{fit}[/{fit_color}]")
        console.print()

    choice = typer.prompt("Select a model", default=str(default_model_idx), type=int)
    if choice < 1 or choice > len(model_options):
        console.print("[red]Invalid choice.[/red]")
        raise typer.Exit(1)
    selected_model = model_options[choice - 1]

    # Step 3: Language
    console.print("\n[bold]Language settings[/bold]\n")
    console.print("  ko: Korean-dominant dialogue, including ordinary English terms")
    console.print("  en: English-dominant dialogue")
    console.print(
        "  auto: detect one dominant language per ASR chunk; not sentence-level language switching"
    )
    lang = typer.prompt("Default language", default="ko (Korean)")
    lang_code = lang.split()[0] if lang else "ko"

    # Step 4: Diarization
    console.print("\n[bold]Speaker diarization[/bold]\n")
    enable_diarization = typer.confirm("Enable speaker diarization?", default=False)
    diarization_device = "cpu"
    if enable_diarization:
        from meeting_notes.diarization.acceleration import (
            ROCM_INSTALLED_MIB,
            probe_rocm,
        )

        rocm = probe_rocm(MeetingNotesConfig())
        if rocm.state in {"eligible", "ready"}:
            console.print(
                f"[green]Optional AMD ROCm hybrid acceleration is {rocm.state}.[/green]\n"
                f"  {rocm.detail}\n"
                f"  Project-local storage: approximately "
                f"{ROCM_INSTALLED_MIB / 1024:.1f} GiB"
            )
            if typer.confirm("Opt in to GPU-accelerated speaker embeddings?", default=False):
                diarization_device = "rocm-hybrid"
        elif rocm.state == "prerequisites-missing":
            console.print(f"[dim]ROCm acceleration not ready: {rocm.detail}[/dim]")

    # Step 5: Summarization
    console.print("\n[bold]Summarization[/bold]\n")
    summarization_backend, summarization_model, reasoning_effort = _prompt_summarization_config(
        diag
    )
    enable_summarization = summarization_backend != "none"

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
    asr_config: dict[str, object] = {
        "model": selected_model["name"],
        "language": lang_code,
    }
    if lemonade_url:
        asr_config["model_path"] = None
        asr_config["backend_options"] = {
            "lemonade": {
                "base_url": lemonade_url,
                "model_id": "Whisper-Large-v3-Turbo",
            }
        }
    config = MeetingNotesConfig(
        setup=SetupConfig(
            completed=True,
            profile=selected_backend["profile"],
        ),
        runtime={
            "device": selected_backend["runtime_device"],
            "asr_backend": selected_backend["runtime_asr_backend"],
        },
        asr=asr_config,
        diarization={
            "enabled": enable_diarization,
            "device": diarization_device,
        },
        summarization={
            "enabled": enable_summarization,
            "backend": summarization_backend,
            "codex": {
                "model": summarization_model if summarization_backend == "codex" else None,
                "reasoning_effort": reasoning_effort,
            },
            "claude": {
                "model": summarization_model if summarization_backend == "claude" else None,
                "effort": reasoning_effort if summarization_backend == "claude" else None,
            },
            "lemonade": {
                "base_url": lemonade_url or "http://127.0.0.1:13305",
                "model_id": (
                    summarization_model
                    if summarization_backend == "lemonade"
                    else "Gemma-4-26B-A4B-it-MTP-GGUF"
                ),
            },
        },
        naming={
            "recording_mode": naming_mode,
        },
    )

    # Preview and confirm
    console.print("\n[bold]Resolved configuration[/bold]\n")
    console.print(f"  Profile: {config.setup.profile}")
    console.print(f"  Backend: {config.runtime.asr_backend}")
    console.print(f"  Device: {config.runtime.device}")
    console.print(f"  Model: {config.asr.model}")
    console.print(f"  Language: {config.asr.language}")
    console.print(f"  Diarization: {'enabled' if config.diarization.enabled else 'disabled'}")
    if config.diarization.enabled:
        console.print(f"  Diarization device: {config.diarization.device}")
    console.print(f"  Summarization: {'enabled' if config.summarization.enabled else 'disabled'}")
    if config.summarization.enabled:
        if config.summarization.backend == "codex":
            model = config.summarization.codex.model
        elif config.summarization.backend == "claude":
            model = config.summarization.claude.model
        else:
            model = config.summarization.lemonade.model_id
        console.print(f"  Summarization backend: {config.summarization.backend}")
        console.print(f"  Summarization model: {model or 'provider default'}")
        if config.summarization.backend in {"codex", "claude"}:
            effort = (
                config.summarization.codex.reasoning_effort
                if config.summarization.backend == "codex"
                else config.summarization.claude.effort
            )
            console.print(f"  Reasoning effort: {effort or 'provider default'}")
    console.print(f"  Naming mode: {config.naming.recording_mode}")
    console.print(f"  Config path: {target}")
    console.print()

    if not typer.confirm("Save this configuration?", default=True):
        console.print("[yellow]Configuration cancelled.[/yellow]")
        raise typer.Exit(0)

    if config.runtime.asr_backend == "lemonade":
        provision_runtime = False
        provision_model = typer.confirm(
            "Download/install and load the model through Lemonade now?",
            default=True,
        )
    else:
        provision_runtime = typer.confirm(
            "Provision the selected whisper.cpp runtime?", default=True
        )
        provision_model = typer.confirm("Download and verify the selected model?", default=True)
    provision_summarizer = False
    if config.summarization.backend == "lemonade":
        provision_summarizer = typer.confirm(
            "Download/install and load the Lemonade summarization model now?",
            default=True,
        )
    try:
        if provision_runtime:
            _provision_runtime(config)
        if provision_model:
            if config.runtime.asr_backend == "lemonade":
                _provision_lemonade_model(config, yes=False, interactive=True)
            else:
                _provision_model(config, yes=False, interactive=True)
        if provision_summarizer:
            _provision_lemonade_summarizer(config, yes=False, interactive=True)
    except Exception as exc:
        console.print(f"[red]Provisioning failed: {exc}[/red]")
        console.print(
            "[yellow]The previous configuration and installs were left unchanged.[/yellow]"
        )
        raise typer.Exit(1) from exc

    save_config(config, target)
    console.print(f"[green]Configuration saved to: {target}[/green]")
    if provision_model and (provision_runtime or config.runtime.asr_backend == "lemonade"):
        console.print("\nRuntime and transcription model verified.")
    else:
        _print_provisioning_commands(config, target)
    if config.diarization.enabled:
        acceleration = config.diarization.device
        console.print(
            "\nSpeaker diarization requires dependency setup followed by one guided step:\n"
            "  uv sync --extra diarization\n"
            f"  uv run meeting-notes diarization setup --acceleration {acceleration} "
            f'--config "{target}"\n'
            "A portable backup can be used instead of Hugging Face login:\n"
            f"  uv run meeting-notes diarization setup --acceleration {acceleration} "
            f'--model-archive "<backup.zip>" --config "{target}"\n'
            "Then verify with:\n"
            f'  uv run meeting-notes doctor --config "{target}"'
        )
    elif provision_runtime and provision_model:
        console.print("Run: meeting-notes process <audio-file>")


def _build_backend_options(diag: SystemDiagnostics) -> list[dict]:
    """Build the list of backend options for the wizard."""
    options = []

    # Safe CPU (always available)
    options.append(
        {
            "name": "Safe CPU (default)",
            "backend": "whisper.cpp CPU",
            "model": "large-v3-turbo",
            "memory": "~2.1 GB",
            "recommended_ram": ">= 4.1 GB",
            "compatibility": "available",
            "notes": "slowest but most portable and least driver-dependent",
            "profile": "safe-cpu",
            "runtime_device": "cpu",
            "runtime_asr_backend": "whisper_cpp",
        }
    )

    # AMD Lemonade (external server, opt-in)
    from meeting_notes.asr.lemonade import LemonadeASRBackend

    lemonade_url = "http://127.0.0.1:13305"
    lemonade = LemonadeASRBackend(
        base_url=lemonade_url,
        connect_timeout_seconds=0.75,
    )
    lemonade_ready = lemonade.is_available()
    options.append(
        {
            "name": "AMD Lemonade NPU (opt-in)",
            "backend": "Lemonade Server / whisper.cpp NPU",
            "model": "large-v3-turbo",
            "memory": "managed by Lemonade",
            "recommended_ram": "system dependent",
            "compatibility": "available" if lemonade_ready else "server not running",
            "notes": (
                f"detected at {lemonade_url}"
                if lemonade_ready
                else f"start Lemonade Server manually; default URL is {lemonade_url}"
            ),
            "profile": "amd-lemonade",
            "runtime_device": "npu",
            "runtime_asr_backend": "lemonade",
        }
    )

    # Vulkan
    vulkan_compat = "detected, not yet validated" if diag.gpu.vulkan_devices else "not detected"
    vulkan_notes = ""
    if not diag.gpu.vulkan_devices:
        vulkan_notes = "Vulkan devices not found; requires Vulkan-capable GPU and driver"
    options.append(
        {
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
        }
    )

    # ROCm/HIP
    rocm_compat = "detected" if diag.gpu.rocm_architectures else "unavailable in this environment"
    rocm_notes = ""
    if not diag.gpu.rocm_architectures:
        rocm_notes = "rocminfo and a HIP-enabled whisper.cpp build were not found"
    options.append(
        {
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
        }
    )

    # CUDA
    cuda_compat = "available" if diag.gpu.cuda_available else "not available"
    cuda_notes = "" if diag.gpu.cuda_available else "NVIDIA CUDA not detected"
    options.append(
        {
            "name": "NVIDIA CUDA",
            "backend": "whisper.cpp CUDA or faster-whisper",
            "model": "large-v3",
            "memory": f"~{diag.gpu.vram_gb:.1f} GB VRAM detected"
            if diag.gpu.cuda_available
            else "~10 GB VRAM needed",
            "recommended_ram": ">= 12 GB",
            "compatibility": cuda_compat,
            "notes": cuda_notes,
            "profile": "nvidia-cuda",
            "runtime_device": "cuda",
            "runtime_asr_backend": "whisper_cpp",
        }
    )

    return options


def _build_model_options(diag: SystemDiagnostics, device: str = "cpu") -> list[dict]:
    """Build model options filtered by compatibility."""
    models = [
        (
            "large-v3-turbo",
            "recommended: near large-v3 multilingual accuracy with faster transcription",
            "~1.5 GiB",
            "~2.1 GB",
            "~4.1 GB",
        ),
        (
            "small",
            "lighter, but less accurate for multilingual transcription",
            "466 MiB",
            "~852 MB",
            "~2.9 GB",
        ),
        (
            "medium",
            "translation-capable; slower and less accurate than turbo for transcription",
            "1.5 GiB",
            "~2.1 GB",
            "~4.1 GB",
        ),
        ("large-v3", "highest standard Whisper profile", "2.9 GiB", "~3.9 GB", "~5.9 GB"),
    ]

    options = []
    for name, accuracy, download, runtime_mem, recommended in models:
        est = get_resource_estimate(name, "whisper_cpp")
        fit = "unknown"
        if est:
            fit_status, _ = check_model_fit(est, diag)
            fit = fit_status

        options.append(
            {
                "name": name,
                "accuracy": accuracy,
                "download": f"{download} [official reference]",
                "runtime_memory": f"{runtime_mem} [official reference]",
                "recommended_ram": f"{recommended} [estimated]",
                "fit": fit,
            }
        )

    return options


def _default_model_index(model_options: list[dict]) -> int:
    """Choose turbo unless it cannot meet the detected minimum memory."""
    turbo = next(
        (
            index
            for index, option in enumerate(model_options, 1)
            if option["name"] == "large-v3-turbo"
        ),
        1,
    )
    small = next(
        (index for index, option in enumerate(model_options, 1) if option["name"] == "small"),
        turbo,
    )
    return small if model_options[turbo - 1]["fit"] in {"not_detected", "incompatible"} else turbo


def show_config(resolved: bool = False, config_path: str | None = None) -> None:
    """Show current configuration."""
    try:
        config = load_config(config_path)
    except (ConfigNotFoundError, ConfigValidationError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e

    if resolved:
        console.print_json(config.model_dump_json(indent=2))
    else:
        from meeting_notes.config import _resolve_config_path

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
            console.print(f"  Diarization device: {config.diarization.device}")
            checks: dict[str, object] = {}
            _add_diarization_checks(checks, config)
            rocm = checks.get("rocm_probe")
            if isinstance(rocm, dict):
                console.print(f"  Diarization ROCm: {rocm.get('state')} ({rocm.get('detail')})")
            for line in _diarization_recommendations(config, checks):
                console.print(line)
        except ConfigValidationError as e:
            console.print(f"[yellow]Config invalid: {e}[/yellow]")
    else:
        console.print("[yellow]No configuration found.[/yellow]")
        console.print(f"Expected at: {DEFAULT_CONFIG_PATH}")


def config_edit_cmd(config_path: str | None = None) -> None:
    """Open configuration file in editor."""
    from meeting_notes.config import _resolve_config_path

    path = _resolve_config_path(config_path)
    if not path:
        console.print(
            "[yellow]No config file to edit. Run 'meeting-notes configure' first.[/yellow]"
        )
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

    if config.runtime.asr_backend == "lemonade":
        from meeting_notes.asr.registry import get_configured_backend

        readiness = get_configured_backend(config).check_readiness()
        checks.update(
            {
                "configured": True,
                "asr_backend": "lemonade",
                "device": config.runtime.device,
                "runtime_ready": readiness.available,
                "runtime_detail": readiness.detail,
                "server_url": readiness.metadata.get("base_url"),
                "server_version": readiness.version,
                "model_name": config.asr.model,
                "lemonade_model_id": readiness.metadata.get("model_id"),
                "model_downloaded": readiness.metadata.get("downloaded", False),
                "model_loaded": readiness.metadata.get("loaded", False),
                "actual_device": readiness.device,
                "diarization_enabled": config.diarization.enabled,
                "latest_transcript": _latest_transcript_metadata(config),
            }
        )
        _add_diarization_checks(checks, config)
        return checks

    from meeting_notes.models import verify_model
    from meeting_notes.runtime import find_manifest_for_executable

    executable = Path(config.runtime.whisper_cpp_path)
    if not executable.is_absolute() and shutil.which(config.runtime.whisper_cpp_path):
        executable = Path(shutil.which(config.runtime.whisper_cpp_path) or executable)
    cache_dir = project_cache_root(config)
    manifest = (
        find_manifest_for_executable(executable, cache_dir=cache_dir)
        if executable.exists()
        else None
    )
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
            "latest_transcript": latest_transcript,
        }
    )
    _add_diarization_checks(checks, config)
    return checks


def _add_diarization_checks(
    checks: dict[str, object],
    config: MeetingNotesConfig,
) -> None:
    from meeting_notes.diarization.acceleration import (
        diarization_cache_root,
        probe_rocm,
    )
    from meeting_notes.diarization.setup import resolve_hf_token

    token, token_source = resolve_hf_token(config.diarization.token_env)
    model_path = Path(config.diarization.model_path) if config.diarization.model_path else None
    try:
        from importlib.metadata import version

        pyannote_version = version("pyannote.audio")
        pyannote_installed = True
    except Exception:
        pyannote_installed = False
        pyannote_version = None
    rocm = probe_rocm(config)
    checks.update(
        {
            "diarization_device": config.diarization.device,
            "diarization_model": config.diarization.model,
            "diarization_model_path": str(model_path) if model_path else None,
            "local_diarization_model_ready": bool(model_path and model_path.exists()),
            "token_env": config.diarization.token_env,
            "pyannote_installed": pyannote_installed,
            "pyannote_version": pyannote_version,
            "hf_token_ready": bool(token),
            "hf_token_source": token_source,
            "diarization_cache_root": str(diarization_cache_root(config)),
            "rocm_gpu_runtime_path": config.diarization.rocm_gpu_runtime_path,
            "rocm_probe": rocm.to_dict(),
        }
    )


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
    rocm = checks.get("rocm_probe")
    rocm_state = rocm.get("state") if isinstance(rocm, dict) else None
    if config.diarization.device == "cpu" and rocm_state in {"eligible", "ready"}:
        lines.extend(
            [
                "  Optional AMD GPU embedding acceleration is available (CPU remains default):",
                "    uv run meeting-notes diarization setup --acceleration rocm-hybrid",
            ]
        )
    elif config.diarization.device == "rocm-hybrid" and rocm_state != "ready":
        lines.extend(
            [
                "  The configured ROCm hybrid runtime is not ready:",
                "    uv run meeting-notes diarization setup --acceleration rocm-hybrid",
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
    if config.runtime.asr_backend == "lemonade":
        from meeting_notes.asr.registry import get_configured_backend

        configured = get_configured_backend(config)
        readiness = configured.check_readiness()
        if not readiness.available:
            return {"success": False, "detail": readiness.detail}
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
                return {
                    "success": False,
                    "detail": f"FFmpeg failed: {generated.stderr.strip()}",
                }
            try:
                configured.backend.transcribe(wav, **configured.transcribe_kwargs)
            except (RuntimeError, ValueError) as error:
                return {"success": False, "detail": str(error)}
            return {
                "success": True,
                "detail": f"Lemonade transcription succeeded on {config.runtime.device}",
            }
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
        success = result.returncode == 0 and (config.runtime.device != "vulkan" or vulkan_seen)
        detail = "model loaded successfully"
        if config.runtime.device == "vulkan" and not vulkan_seen:
            detail = "whisper.cpp output did not confirm Vulkan initialization"
        elif result.returncode:
            detail = f"whisper.cpp exited {result.returncode}: {result.stderr[-500:]}"
        return {"success": success, "vulkan_initialized": vulkan_seen, "detail": detail}


def _print_configured_asr(configured: dict[str, object]) -> None:
    if configured["asr_backend"] == "lemonade":
        ready = bool(configured["runtime_ready"])
        style = "green" if ready else "red"
        console.print("  ASR backend: [bold]lemonade[/bold]")
        console.print(f"  Server URL: {configured['server_url']}")
        console.print(f"  Server version: {configured['server_version'] or 'unknown'}")
        console.print(f"  Model: {configured['lemonade_model_id']}")
        console.print(f"  Downloaded: {'yes' if configured['model_downloaded'] else 'no'}")
        console.print(f"  Loaded: {'yes' if configured['model_loaded'] else 'no'}")
        console.print(f"  Device: {configured['actual_device'] or configured['device']}")
        console.print(f"  Status: [{style}]{configured['runtime_detail']}[/{style}]")
        return

    exe_style = "green" if configured["executable_exists"] else "red"
    model_style = "green" if configured["model_verified"] else "red"
    runnable_style = "green" if configured["executable_runnable"] else "red"
    console.print(f"  ASR backend: [bold]{configured['asr_backend']}[/bold]")
    console.print(
        f"  Configured runtime: [{runnable_style}]"
        f"{'ready' if configured['executable_runnable'] else 'not runnable'}"
        f"[/{runnable_style}]"
    )
    console.print(f"  Executable: [{exe_style}]{configured['executable']}[/{exe_style}]")
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
        console.print(f"  Source revision: {runtime_manifest.get('source_revision', 'unknown')}")
        if runtime_manifest.get("checksum"):
            console.print(f"  Archive SHA-256: {runtime_manifest['checksum']}")
    console.print(
        f"  Model: [{model_style}]{configured['model_name']} — "
        f"{configured['model_detail']}[/{model_style}]"
    )
    console.print(f"  Model path: {configured['model']}")


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
            _print_configured_asr(configured)
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
                    configured["hf_token_ready"] or configured["local_diarization_model_ready"]
                )
                rocm_check = configured.get("rocm_probe")
                if configured.get("diarization_device") == "rocm-hybrid":
                    ready = bool(
                        ready
                        and isinstance(rocm_check, dict)
                        and rocm_check.get("state") == "ready"
                    )
                console.print(
                    f"  Diarization: "
                    f"{'[green]ready[/green]' if ready else '[yellow]not ready[/yellow]'}"
                )
                console.print(f"    Device: {configured['diarization_device']}")
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
                        f"    Local model: {local_status} ({configured['diarization_model_path']})"
                    )
                rocm = configured.get("rocm_probe")
                if isinstance(rocm, dict):
                    console.print(f"    ROCm hybrid: {rocm.get('state')} ({rocm.get('detail')})")
                    if rocm.get("runtime_path"):
                        console.print(f"    ROCm runtime: {rocm.get('runtime_path')}")
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
        lemonade_selected = configured.get("asr_backend") == "lemonade"
        if (
            not lemonade_selected
            and not diag.tools.whisper_cpp_available
            and not configured_runtime_ready
        ):
            console.print(
                "  [yellow]Install whisper.cpp or build from source: https://github.com/ggerganov/whisper.cpp[/yellow]"
            )
        elif (
            not lemonade_selected
            and not diag.tools.whisper_cpp_available
            and configured_runtime_ready
        ):
            console.print(
                "  [green]No PATH installation is needed; the configured managed "
                "whisper.cpp runtime is ready.[/green]"
            )
        if configured.get("configured"):
            config = load_config(config_path)
            for line in _diarization_recommendations(config, configured):
                console.print(line)
        if not diag.tools.codex_available:
            console.print("  [dim]Codex CLI not installed: https://github.com/openai/codex[/dim]")
        if not diag.tools.claude_available:
            console.print("  [dim]Claude Code not installed: https://code.claude.com/docs[/dim]")
        if not diag.gpu.available and not lemonade_selected:
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
            "ffmpeg": {
                "available": diag.tools.ffmpeg_available,
                "version": diag.tools.ffmpeg_version,
            },
            "ffprobe": {
                "available": diag.tools.ffprobe_available,
                "version": diag.tools.ffprobe_version,
            },
            "whisper_cpp": {
                "available": diag.tools.whisper_cpp_available,
                "version": diag.tools.whisper_cpp_version,
            },
            "codex": {"available": diag.tools.codex_available, "version": diag.tools.codex_version},
            "claude": {
                "available": diag.tools.claude_available,
                "version": diag.tools.claude_version,
            },
        },
    }


def run_resources_show(model: str | None = None, device: str | None = None) -> None:
    """Show resource estimates."""
    diag = detect_system()
    console.print(format_diagnostics_table(diag))
    console.print()

    models = (
        ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"] if not model else [model]
    )

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
            fit_style = (
                "green"
                if fit_status == "available"
                else ("yellow" if "warning" in fit_status else "red")
            )
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
            console.print(
                f"  {name:20s}  {est.disk_size_mib:>6} MiB  ~{est.reference_memory_mb} MB RAM"
            )
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
    cache_dir = project_cache_root(config)
    executable = (
        install_vulkan(version, cache_dir=cache_dir)
        if config.runtime.device == "vulkan"
        else install_cpu(version, cache_dir=cache_dir)
    )
    config.runtime.whisper_cpp_path = str(executable.resolve())
    return executable


def _provision_model(config: MeetingNotesConfig, *, yes: bool, interactive: bool = False) -> Path:
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
    path = download_model(config.asr.model, cache_dir=project_cache_root(config))
    config.asr.model_path = str(path.resolve())
    config.asr.model_cache_dir = str(path.parent.resolve())
    return path


def _provision_lemonade_model(
    config: MeetingNotesConfig,
    *,
    yes: bool,
    interactive: bool = False,
) -> str:
    """Install and load the configured model through a running Lemonade server."""
    from meeting_notes.asr.lemonade import LemonadeASRBackend

    options = config.asr.backend_options.lemonade
    backend = LemonadeASRBackend(
        base_url=options.base_url,
        model_id=options.model_id,
        api_key_env=options.api_key_env,
        expected_device=config.runtime.device,
        connect_timeout_seconds=options.connect_timeout_seconds,
        provisioning_timeout_seconds=options.provisioning_timeout_seconds,
        transcription_timeout_seconds=options.transcription_timeout_seconds,
    )
    if not backend.is_available():
        raise RuntimeError(
            f"Lemonade Server is not reachable at {options.base_url}. "
            "Start Lemonade Server manually, verify it with 'lemonade status', then retry."
        )
    info = backend.model_info(show_all=True)
    if info is None:
        raise RuntimeError(
            f"Lemonade model '{options.model_id}' is not registered in the server catalogue."
        )
    if not info.get("downloaded"):
        size_gb = float(info.get("size") or 0.0)
        threshold_gb = config.resources.large_download_confirmation_mib / 1024
        if size_gb >= threshold_gb and not yes:
            if interactive and typer.confirm(
                f"{options.model_id} is approximately {size_gb:.1f} GiB. Download it?",
                default=False,
            ):
                pass
            else:
                raise RuntimeError(
                    f"{options.model_id} is a large download; rerun with --yes to confirm."
                )

        last_bucket = -1

        def report_progress(event: dict[str, object]) -> None:
            nonlocal last_bucket
            percent = int(float(event.get("percent") or 0))
            bucket = percent // 10
            if bucket != last_bucket or event.get("event") == "complete":
                last_bucket = bucket
                console.print(f"  Lemonade model download: {percent}%")

        backend.pull_model(progress=report_progress)
    readiness = backend.load_model()
    config.asr.model_path = None
    console.print(
        f"[green]Lemonade model ready: {options.model_id} / "
        f"{readiness.device} / server {readiness.version}[/green]"
    )
    return options.model_id


def _provision_lemonade_summarizer(
    config: MeetingNotesConfig,
    *,
    yes: bool,
    interactive: bool = False,
) -> str:
    """Download the configured local LLM through an already-running server."""
    from meeting_notes.summarization.adapters import LemonadeAdapter

    options = config.summarization.lemonade
    adapter = LemonadeAdapter(
        base_url=options.base_url,
        model_id=options.model_id,
        api_key_env=options.api_key_env,
        connect_timeout_seconds=options.connect_timeout_seconds,
        request_timeout_seconds=options.request_timeout_seconds,
        provisioning_timeout_seconds=options.provisioning_timeout_seconds,
        max_completion_tokens=options.max_completion_tokens,
    )
    if not adapter.is_available():
        raise RuntimeError(
            f"Lemonade Server is not reachable at {options.base_url}. "
            "Start Lemonade Server manually, verify it with 'lemonade status', then retry."
        )
    info = adapter.model_info()
    if info is None:
        raise RuntimeError(
            f"Lemonade model '{options.model_id}' is not registered in the server catalogue."
        )
    if not info.get("downloaded"):
        size_gb = float(info.get("size") or 0.0)
        threshold_gb = config.resources.large_download_confirmation_mib / 1024
        requires_confirmation = size_gb >= threshold_gb and not yes
        confirmed = not requires_confirmation or (
            interactive
            and typer.confirm(
                f"{options.model_id} is approximately {size_gb:.1f} GiB. Download it?",
                default=False,
            )
        )
        if not confirmed:
            raise RuntimeError(f"{options.model_id} is a large download; confirmation is required.")
        adapter.pull_model()
    adapter.ensure_model_ready()
    console.print(f"[green]Lemonade summarization model ready: {options.model_id}[/green]")
    return options.model_id


def _provision_config(config: MeetingNotesConfig, *, yes: bool) -> None:
    if config.runtime.asr_backend == "lemonade":
        _provision_lemonade_model(config, yes=yes)
    else:
        _provision_runtime(config)
        _provision_model(config, yes=yes)
    if config.summarization.enabled and config.summarization.backend == "lemonade":
        _provision_lemonade_summarizer(config, yes=yes)


def _print_provisioning_commands(config: MeetingNotesConfig, target: Path) -> None:
    console.print("\n[bold]Provisioning still required[/bold]")
    if config.runtime.asr_backend == "lemonade":
        options = config.asr.backend_options.lemonade
        console.print(f"  Start Lemonade Server manually at: {options.base_url}")
        console.print("  Verify the server: lemonade status")
        console.print(
            f"  meeting-notes models download {config.asr.model} "
            f'--backend lemonade --config "{target}" --yes'
        )
    else:
        console.print(
            f'  meeting-notes runtime install --device {config.runtime.device} --config "{target}"'
        )
        suffix = " --yes" if config.asr.model in {"medium", "large-v3", "large-v3-turbo"} else ""
        console.print(
            f'  meeting-notes models download {config.asr.model} --config "{target}"{suffix}'
        )
    if config.summarization.enabled and config.summarization.backend == "lemonade":
        options = config.summarization.lemonade
        console.print(f"  Start Lemonade Server manually at: {options.base_url}")
        console.print(f'  meeting-notes configure --config "{target}" --provision --yes')


def run_models_status(output_json: bool = False, config_path: str | None = None) -> None:
    """Show model download status."""
    from meeting_notes.models import model_statuses

    config = load_config(config_path)
    statuses = model_statuses(cache_dir=project_cache_root(config))
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
        console.print(
            f"[yellow]No resource data for model '{model}' with backend '{backend}'[/yellow]"
        )


def run_models_download(
    model: str,
    backend: str = "auto",
    yes: bool = False,
    config_path: str | None = None,
) -> None:
    """Download a model."""
    try:
        selected_config = load_config(config_path)
    except (ConfigNotFoundError, ConfigValidationError) as error:
        console.print(
            "[red]Managed model downloads require an active configuration so "
            "project.cache_dir is unambiguous.[/red]"
        )
        raise typer.Exit(2) from error
    if backend == "auto":
        backend = selected_config.runtime.asr_backend
    if backend == "lemonade":
        if model != "large-v3-turbo":
            console.print(
                "[red]The Lemonade adapter currently supports the canonical "
                "model name 'large-v3-turbo'.[/red]"
            )
            raise typer.Exit(2)
        selected_config.asr.model = model
        try:
            _provision_lemonade_model(selected_config, yes=yes, interactive=False)
            target = _target_config_path(config_path)
            save_config(selected_config, target)
            console.print(
                f"[green]Verified Lemonade model installed: "
                f"{selected_config.asr.backend_options.lemonade.model_id}[/green]"
            )
        except RuntimeError as exc:
            console.print(f"[red]Lemonade model installation failed: {exc}[/red]")
            raise typer.Exit(1) from exc
        return
    if backend != "whisper_cpp":
        console.print("[red]--backend must be auto, whisper_cpp, or lemonade.[/red]")
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
        path = download_model(model, cache_dir=project_cache_root(selected_config))
        if config_path is not None:
            target = _target_config_path(config_path)
            config = selected_config or load_config(str(target))
            config.asr.model = model
            config.asr.model_path = str(path)
            config.asr.model_cache_dir = str(path.parent)
            save_config(config, target)
        console.print(f"[green]Verified model installed: {path}[/green]")
    except (ModelInstallError, OSError) as exc:
        console.print(f"[red]Model installation failed: {exc}[/red]")
        raise typer.Exit(1) from exc


def run_models_verify(
    model: str, backend: str = "whisper_cpp", config_path: str | None = None
) -> None:
    """Verify a downloaded model."""
    if backend != "whisper_cpp":
        console.print("[red]Managed verification supports only whisper_cpp.[/red]")
        raise typer.Exit(2)
    from meeting_notes.models import model_path, verify_model

    config = load_config(config_path)
    path = model_path(model, cache_dir=project_cache_root(config))
    if config.asr.model == model and config.asr.model_path:
        path = Path(config.asr.model_path)
    valid, detail = verify_model(model, path)
    if not valid:
        console.print(f"[red]{model}: {detail}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{model}: verified ({path.resolve()})[/green]")


def run_runtime_status(output_json: bool = False, config_path: str | None = None) -> None:
    """Show managed runtime status."""
    from meeting_notes.runtime import installed_runtimes

    config = load_config(config_path)
    runtimes = installed_runtimes(cache_dir=project_cache_root(config))
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
        config = load_config(config_path)
        cache_dir = project_cache_root(config)
        executable = (
            install_vulkan(version, cache_dir=cache_dir)
            if device == "vulkan"
            else install_cpu(version, cache_dir=cache_dir)
        )
        if config_path is not None:
            target = _target_config_path(config_path)
            config.runtime.device = device
            config.runtime.asr_backend = "whisper_cpp"
            config.runtime.whisper_cpp_path = str(executable.resolve())
            save_config(config, target)
        console.print(f"[green]Verified {device} runtime installed: {executable.resolve()}[/green]")
    except (RuntimeInstallError, OSError) as exc:
        console.print(f"[red]Runtime installation failed: {exc}[/red]")
        raise typer.Exit(1) from exc


def run_cache_status(*, config_path: str | None = None, output_json: bool = False) -> None:
    """Show first-party project and legacy cache usage."""
    from meeting_notes.storage import cache_inventory

    config = load_config(config_path)
    inventory = cache_inventory(config)
    if output_json:
        console.print_json(json.dumps(inventory, indent=2))
        return
    console.print("\n[bold]meeting-notes cache status[/bold]\n")
    for label in ("project", "legacy"):
        section = inventory[label]
        if not isinstance(section, dict):
            continue
        console.print(f"  {label.title()}: {section.get('root')}")
        console.print(f"    Total: {int(section.get('total_bytes', 0)) / (1024**3):.2f} GiB")
        values = section.get("sections")
        if isinstance(values, dict):
            for name in ("models", "runtimes", "diarization"):
                value = values.get(name)
                if isinstance(value, dict):
                    console.print(
                        f"    {name}: {int(value.get('bytes', 0)) / (1024**2):.1f} MiB"
                    )


def run_cache_migrate(*, config_path: str | None = None, yes: bool = False) -> None:
    """Migrate recognized per-user Whisper assets without changing ASR selection."""
    from meeting_notes.storage import cache_inventory, migrate_legacy_cache, project_cache_root

    config = load_config(config_path)
    target = _target_config_path(config_path)
    before = cache_inventory(config)
    legacy = before.get("legacy")
    legacy_bytes = int(legacy.get("total_bytes", 0)) if isinstance(legacy, dict) else 0
    console.print("\n[bold]Project-local cache migration[/bold]")
    console.print(f"  Destination: {project_cache_root(config)}")
    console.print(f"  Legacy storage detected: {legacy_bytes / (1024**3):.2f} GiB")
    console.print(f"  Active ASR remains: {config.runtime.asr_backend}/{config.runtime.device}")
    if not yes and not typer.confirm(
        "Copy, validate, and remove recognized legacy Whisper assets?", default=True
    ):
        raise typer.Exit(1)
    result = migrate_legacy_cache(config, target)
    console.print("[green]Project-local cache migration completed.[/green]")
    console.print(f"  Project cache: {result['project_cache']}")
    console.print(
        f"  Removed legacy data: {int(result['removed_legacy_bytes']) / (1024**3):.2f} GiB"
    )
    unknown = result.get("unknown_legacy_models")
    if isinstance(unknown, list) and unknown:
        console.print("[yellow]Unrecognized legacy model files were retained:[/yellow]")
        for path in unknown:
            console.print(f"  {path}")
    unknown_runtime = result.get("unknown_legacy_runtime_files")
    if isinstance(unknown_runtime, list) and unknown_runtime:
        console.print("[yellow]Unrecognized legacy runtime files were retained:[/yellow]")
        for path in unknown_runtime:
            console.print(f"  {path}")
