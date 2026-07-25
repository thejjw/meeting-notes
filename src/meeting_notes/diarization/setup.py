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

from meeting_notes.config import _resolve_config_path, load_config, save_config
from meeting_notes.runtime import cache_root

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


def managed_diarization_dir(repo_id: str) -> Path:
    """Return the managed local pipeline directory for a Hub repository."""
    safe_name = repo_id.replace("/", "--")
    return cache_root() / "diarization" / safe_name


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
) -> None:
    """Guide browser login/consent, download the pipeline, and update configuration."""
    config = load_config(config_path)
    if config.diarization.backend != "pyannote":
        raise DiarizationSetupError(
            "Guided setup currently supports only diarization.backend: pyannote."
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
    destination = managed_diarization_dir(repo_id)
    config_file = destination / "config.yaml"
    resolved_config_path = _resolve_config_path(config_path)
    if resolved_config_path is None:
        raise DiarizationSetupError("No writable active configuration was found.")

    console.print("\n[bold]Speaker diarization setup[/bold]\n")
    console.print(f"  Backend: pyannote.audio {installed_version}")
    console.print(f"  Model: {repo_id}")
    console.print(f"  Install path: {destination}")

    if config_file.is_file() and not force:
        config.diarization.enabled = True
        config.diarization.model_path = str(destination.resolve())
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
    save_config(config, resolved_config_path)
    console.print("\n[green]Diarization pipeline installed and configured.[/green]")
    console.print(f"  Local model: {destination}")
    console.print(f"  Config: {resolved_config_path}")
    console.print("\nNext:")
    console.print("  uv run meeting-notes doctor")
    console.print('  uv run meeting-notes process "<audio-file>" --from diarize')
