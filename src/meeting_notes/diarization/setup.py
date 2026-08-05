"""Guided authentication and provisioning for gated diarization models."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import webbrowser
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer
from rich.console import Console

from meeting_notes.config import MeetingNotesConfig, _resolve_config_path, load_config, save_config
from meeting_notes.diarization.acceleration import (
    ROCM_DOWNLOAD_MIB,
    ROCM_INSTALLED_MIB,
    ROCM_PEAK_FREE_MIB,
    default_runtime_dir,
    diarization_cache_root,
    directory_size,
    model_dir,
    probe_rocm,
    provision_runtime,
    remove_runtime,
)
from meeting_notes.storage import legacy_user_cache_root, project_cache_root

console = Console(stderr=True)


class DiarizationSetupError(RuntimeError):
    """The guided diarization setup could not be completed."""


def resolve_hf_token(token_env: str = "HF_TOKEN") -> tuple[str | None, str | None]:
    """Resolve a token from the configured environment or Hugging Face login cache."""
    environment_token = os.environ.get(token_env)
    if environment_token:
        return environment_token, f"environment variable {token_env}"
    try:
        from huggingface_hub import get_token

        cached_token = get_token()
    except ImportError:
        cached_token = None
    return (cached_token, "saved Hugging Face login") if cached_token else (None, None)


def managed_diarization_dir(repo_id: str, config: MeetingNotesConfig | None = None) -> Path:
    """Return the project-local pipeline directory for a Hub repository."""
    if config is None:
        config = MeetingNotesConfig()
    return model_dir(config, repo_id)


def _valid_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.yaml").is_file()


def _copy_existing_model(
    source: Path,
    destination: Path,
    *,
    force: bool,
    remove_source: bool,
) -> None:
    """Copy and verify an existing model before optionally deleting its old copy."""
    if source.resolve() == destination.resolve():
        return
    if destination.exists() and not force:
        if _valid_model(destination):
            return
        raise DiarizationSetupError(
            f"Project-local model destination already exists: {destination}. Use --force."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".model-copy-", dir=destination.parent) as temp:
        staging = Path(temp) / "install"
        shutil.copytree(source, staging)
        if not _valid_model(staging):
            raise DiarizationSetupError(f"Existing model is missing config.yaml: {source}")
        _install_snapshot(staging, destination)
    if remove_source:
        try:
            shutil.rmtree(source)
        except OSError:
            console.print(
                f"[yellow]Project copy is ready, but the legacy copy could not be removed: "
                f"{source}[/yellow]"
            )


def _confirm_rocm_storage(config: MeetingNotesConfig, *, yes: bool) -> None:
    root = diarization_cache_root(config)
    root.parent.mkdir(parents=True, exist_ok=True)
    free_mib = shutil.disk_usage(root.parent).free / (1024**2)
    console.print("\n[bold]ROCm diarization storage[/bold]")
    console.print(f"  Destination: {root}")
    console.print(f"  Download: approximately {ROCM_DOWNLOAD_MIB / 1024:.1f} GiB")
    console.print(f"  Installed runtime: approximately {ROCM_INSTALLED_MIB / 1024:.1f} GiB")
    console.print(f"  Peak free space required: {ROCM_PEAK_FREE_MIB / 1024:.1f} GiB")
    console.print(f"  Currently free: {free_mib / 1024:.1f} GiB")
    if free_mib < ROCM_PEAK_FREE_MIB:
        raise DiarizationSetupError(
            f"Insufficient free space for ROCm provisioning: "
            f"{free_mib / 1024:.1f} GiB available, "
            f"{ROCM_PEAK_FREE_MIB / 1024:.1f} GiB required."
        )
    if not yes and not typer.confirm(
        "Proceed with the large project-local ROCm installation?", default=True
    ):
        raise typer.Exit(1)


def _install_snapshot(staging: Path, destination: Path) -> None:
    """Atomically replace a managed model directory."""
    old = destination.with_name(f"{destination.name}.old")
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


def run_diarization_setup(
    *,
    config_path: str | None = None,
    yes: bool = False,
    force: bool = False,
    acceleration: str | None = None,
    model_archive: Path | None = None,
) -> None:
    """Guide browser login/consent, download the pipeline, and update configuration."""
    config = load_config(config_path)
    if config.diarization.backend != "pyannote":
        raise DiarizationSetupError(
            "Guided setup currently supports only diarization.backend: pyannote."
        )

    selected_acceleration = acceleration or config.diarization.device
    if selected_acceleration not in {"cpu", "rocm-hybrid"}:
        raise DiarizationSetupError(
            "Diarization setup acceleration must be 'cpu' or 'rocm-hybrid'."
        )

    try:
        installed_version = version("pyannote.audio")
    except PackageNotFoundError:
        console.print(
            "[red]pyannote.audio is not installed.[/red]\n"
            "Install it, then rerun this command:\n"
            "  uv sync --extra diarization\n"
            "  uv run meeting-notes diarization setup"
        )
        raise typer.Exit(1) from None

    repo_id = config.diarization.model
    destination = managed_diarization_dir(repo_id, config)
    config_file = destination / "config.yaml"
    resolved_config_path = _resolve_config_path(config_path)
    if resolved_config_path is None:
        raise DiarizationSetupError("No writable active configuration was found.")

    console.print("\n[bold]Speaker diarization setup[/bold]\n")
    console.print(f"  Backend: pyannote.audio {installed_version}")
    console.print(f"  Model: {repo_id}")
    console.print(f"  Install path: {destination}")
    console.print(f"  Acceleration: {selected_acceleration}")

    runtime_path: Path | None = None
    if selected_acceleration == "rocm-hybrid":
        probe = probe_rocm(config)
        console.print(f"  ROCm probe: {probe.state} ({probe.detail})")
        expected_runtime = default_runtime_dir(config)
        if force or not (expected_runtime / ".meeting-notes-runtime.json").is_file():
            _confirm_rocm_storage(config, yes=yes)
        runtime_path = provision_runtime(config, force=force)

    if model_archive is not None:
        from meeting_notes.model_transfer import restore_archive

        restored = restore_archive(
            "diarization",
            model_archive,
            config_path=str(resolved_config_path),
            force=force,
        )
        config = load_config(str(resolved_config_path))
        config.diarization.device = selected_acceleration
        config.diarization.rocm_gpu_runtime_path = (
            str(runtime_path.resolve()) if runtime_path else None
        )
        config.diarization.model_path = str(restored)
        save_config(config, resolved_config_path)
        console.print("\n[green]Diarization model restored and configured.[/green]")
        console.print(f"  Local model: {restored}")
        if runtime_path:
            console.print(f"  ROCm runtime: {runtime_path}")
        return

    configured_model = (
        Path(config.diarization.model_path).expanduser().resolve()
        if config.diarization.model_path
        else None
    )
    if configured_model and _valid_model(configured_model) and configured_model != destination:
        legacy_root = (legacy_user_cache_root() / "diarization").resolve()
        try:
            is_legacy = configured_model.is_relative_to(legacy_root)
        except ValueError:
            is_legacy = False
        migrate = yes or typer.confirm(
            f"Copy the existing diarization model into the project cache at {destination}?",
            default=True,
        )
        if not migrate:
            raise typer.Exit(1)
        _copy_existing_model(
            configured_model,
            destination,
            force=force,
            remove_source=is_legacy,
        )

    if config_file.is_file() and not force:
        config.diarization.enabled = True
        config.diarization.model_path = str(destination.resolve())
        config.diarization.device = selected_acceleration
        config.diarization.rocm_gpu_runtime_path = (
            str(runtime_path.resolve()) if runtime_path else None
        )
        save_config(config, resolved_config_path)
        console.print("\n[green]The managed local pipeline is already installed.[/green]")
        console.print(f"Configuration updated: {resolved_config_path}")
        return

    try:
        from huggingface_hub import HfApi, login, snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:
        raise DiarizationSetupError(
            "huggingface_hub is unavailable. Run: uv sync --extra diarization"
        ) from None

    token, token_source = resolve_hf_token(config.diarization.token_env)
    if token is None:
        if not sys.stdin.isatty():
            raise DiarizationSetupError(
                "Hugging Face login requires an interactive terminal. "
                "Rerun without redirected input."
            )
        console.print(
            "\nHugging Face authentication is required for this gated model.\n"
            "A browser will open for a device-code login. The resulting credential is\n"
            "stored by Hugging Face; meeting-notes will not print or copy it."
        )
        try:
            login(add_to_git_credential=False, skip_if_logged_in=True)
        except Exception as error:
            raise DiarizationSetupError(
                f"Hugging Face browser login failed: {error}"
            ) from error
        token, token_source = resolve_hf_token(config.diarization.token_env)
        if token is None:
            raise DiarizationSetupError("Hugging Face login completed without a usable token.")

    console.print(f"  Authentication: {token_source}")

    def inspect_download() -> list[object]:
        result = snapshot_download(repo_id=repo_id, token=token, dry_run=True)
        return result if isinstance(result, list) else []

    try:
        files = inspect_download()
    except GatedRepoError:
        model_url = f"https://huggingface.co/{repo_id}"
        if not sys.stdin.isatty():
            raise DiarizationSetupError(
                f"Model conditions must be accepted in a browser: {model_url}"
            ) from None
        console.print(
            "\n[yellow]Your account has not accepted this model's conditions yet.[/yellow]\n"
            f"Opening: {model_url}\n"
            "Review the conditions and click the Hugging Face access/agree button."
        )
        webbrowser.open(model_url)
        if not typer.confirm("Retry after you have accepted the conditions?", default=True):
            raise typer.Exit(1) from None
        try:
            files = inspect_download()
        except GatedRepoError:
            raise DiarizationSetupError(
                "Access is still unavailable. Confirm that the browser used the same "
                f"Hugging Face account, then revisit https://huggingface.co/{repo_id}."
            ) from None
    except HfHubHTTPError as error:
        raise DiarizationSetupError(f"Hugging Face access check failed: {error}") from error

    download_bytes = sum(int(getattr(item, "file_size", 0) or 0) for item in files)
    download_gib = download_bytes / (1024**3)
    if not yes and not typer.confirm(
        f"Download {len(files)} files ({download_gib:.2f} GiB)?",
        default=True,
    ):
        raise typer.Exit(1)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="diarization-",
        dir=destination.parent,
    ) as temp_dir:
        staging = Path(temp_dir) / "install"
        console.print("\nDownloading and verifying the model snapshot...")
        try:
            snapshot_download(
                repo_id=repo_id,
                token=token,
                local_dir=staging,
            )
        except HfHubHTTPError as error:
            raise DiarizationSetupError(f"Model download failed: {error}") from error
        if not (staging / "config.yaml").is_file():
            raise DiarizationSetupError(
                "Downloaded snapshot is missing config.yaml; prior configuration was not changed."
            )
        try:
            model_info = HfApi().model_info(repo_id=repo_id, token=token)
        except HfHubHTTPError as error:
            raise DiarizationSetupError(
                f"Downloaded model provenance could not be verified: {error}"
            ) from error
        install_manifest = {
            "repo_id": repo_id,
            "revision": model_info.sha,
            "download_bytes": download_bytes,
            "installed_at": datetime.now(UTC).isoformat(),
            "authentication": token_source,
        }
        (staging / ".meeting-notes-manifest.json").write_text(
            json.dumps(install_manifest, indent=2),
            encoding="utf-8",
        )
        _install_snapshot(staging, destination)

    config.diarization.enabled = True
    config.diarization.model_path = str(destination.resolve())
    config.diarization.device = selected_acceleration
    config.diarization.rocm_gpu_runtime_path = str(runtime_path.resolve()) if runtime_path else None
    save_config(config, resolved_config_path)
    console.print("\n[green]Diarization pipeline installed and configured.[/green]")
    console.print(f"  Local model: {destination}")
    console.print(f"  Config: {resolved_config_path}")
    console.print("\nNext:")
    console.print("  uv run meeting-notes doctor")
    console.print('  uv run meeting-notes process "<audio-file>" --from diarize')


def run_diarization_status(*, config_path: str | None = None, output_json: bool = False) -> None:
    """Show project-local diarization model and accelerator state."""
    config = load_config(config_path)
    root = diarization_cache_root(config)
    model = Path(config.diarization.model_path) if config.diarization.model_path else None
    runtime = (
        Path(config.diarization.rocm_gpu_runtime_path)
        if config.diarization.rocm_gpu_runtime_path
        else default_runtime_dir(config)
    )
    probe = probe_rocm(config)
    payload = {
        "device": config.diarization.device,
        "cache_root": str(root),
        "model_path": str(model) if model else None,
        "model_ready": bool(model and _valid_model(model)),
        "model_bytes": directory_size(model) if model else 0,
        "rocm_gpu_runtime_path": str(runtime),
        "runtime_bytes": directory_size(runtime),
        "total_bytes": directory_size(root) + directory_size(runtime),
        "rocm": probe.to_dict(),
    }
    if output_json:
        console.print_json(json.dumps(payload, indent=2))
        return
    console.print("\n[bold]Speaker diarization status[/bold]\n")
    console.print(f"  Device: {payload['device']}")
    console.print(f"  Project cache: {root}")
    console.print(
        f"  Model: {'ready' if payload['model_ready'] else 'not ready'} "
        f"({payload['model_bytes'] / (1024**2):.1f} MiB)"
    )
    console.print(f"  ROCm runtime: {probe.state} ({payload['runtime_bytes'] / (1024**3):.2f} GiB)")
    console.print(f"  Total diarization storage: {payload['total_bytes'] / (1024**3):.2f} GiB")
    console.print(f"  Detail: {probe.detail}")
    if config.diarization.device == "cpu" and probe.state in {"eligible", "ready"}:
        console.print(
            "\n  Optional AMD acceleration is available. Opt in with:\n"
            "    uv run meeting-notes diarization setup --acceleration rocm-hybrid"
        )


def run_diarization_runtime_remove(*, config_path: str | None = None, yes: bool = False) -> None:
    """Remove the configured project-local ROCm runtime and select CPU."""
    config = load_config(config_path)
    resolved_config = _resolve_config_path(config_path)
    if resolved_config is None:
        raise DiarizationSetupError("No writable active configuration was found.")
    root = (project_cache_root(config) / "runtimes").resolve()
    runtime = (
        Path(config.diarization.rocm_gpu_runtime_path).resolve()
        if config.diarization.rocm_gpu_runtime_path
        else default_runtime_dir(config).resolve()
    )
    if not runtime.is_relative_to(root):
        raise DiarizationSetupError(
            f"Refusing to remove a runtime outside the project cache: {runtime}"
        )
    qwen_options = config.asr.backend_options.qwen3_asr_lemonade
    qwen_runtime = (
        Path(qwen_options.rocm_gpu_runtime_path).expanduser().resolve()
        if qwen_options.rocm_gpu_runtime_path
        else None
    )
    if qwen_runtime == runtime and (
        config.runtime.asr_backend == "qwen3_asr_lemonade" or qwen_runtime is not None
    ):
        raise DiarizationSetupError(
            "This ROCm runtime is shared with Qwen forced alignment. Remove or "
            "reconfigure the Lemonade Qwen backend before deleting it."
        )
    size = directory_size(runtime)
    if not runtime.exists():
        console.print(f"[yellow]No managed ROCm runtime exists at {runtime}.[/yellow]")
    elif not yes and not typer.confirm(
        f"Remove {size / (1024**3):.2f} GiB ROCm runtime at {runtime}?", default=False
    ):
        raise typer.Exit(1)
    else:
        reclaimed = remove_runtime(runtime)
        console.print(f"[green]Removed {reclaimed / (1024**3):.2f} GiB ROCm runtime.[/green]")
    config.diarization.device = "cpu"
    config.diarization.rocm_gpu_runtime_path = None
    save_config(config, resolved_config)
