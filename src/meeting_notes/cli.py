"""CLI entry point using Typer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from meeting_notes import __version__

app = typer.Typer(
    name="meeting-notes",
    help="Local-first Korean/English meeting notes with Whisper transcription.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"meeting-notes {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", "-V", callback=_version_callback, is_eager=True, help="Show version and exit.")
    ] = False,
) -> None:
    """meeting-notes: Local-first meeting transcription and summarization."""


# --- configure command ---


@app.command()
def configure(
    config_path: Annotated[
        Optional[str], typer.Option("--config", help="Config file path.")
    ] = None,
    accept_defaults: Annotated[
        bool, typer.Option("--accept-defaults", help="Non-interactive safe CPU defaults.")
    ] = False,
    show_detected: Annotated[
        bool, typer.Option("--show-detected", help="Show diagnostics without writing config.")
    ] = False,
    no_configure: Annotated[
        bool, typer.Option("--no-configure", help="Fail immediately instead of opening wizard.")
    ] = False,
) -> None:
    """Run the interactive configuration wizard or create safe defaults."""
    from meeting_notes.configure import run_configure

    run_configure(
        config_path=config_path,
        accept_defaults=accept_defaults,
        show_detected=show_detected,
        no_configure=no_configure,
    )


# --- config commands ---


config_app = typer.Typer(help="View and manage configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    resolved: Annotated[
        bool, typer.Option("--resolved", help="Show fully resolved configuration.")
    ] = False,
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Show current configuration."""
    from meeting_notes.configure import show_config

    show_config(resolved=resolved, config_path=config_path)


@config_app.command("status")
def config_status(
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Show configuration status."""
    from meeting_notes.configure import config_status_cmd

    config_status_cmd(config_path=config_path)


@config_app.command("edit")
def config_edit(
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Open configuration file in editor."""
    from meeting_notes.configure import config_edit_cmd

    config_edit_cmd(config_path=config_path)


@config_app.command("reset")
def config_reset(
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Reset configuration to safe defaults."""
    from meeting_notes.configure import config_reset_cmd

    config_reset_cmd(config_path=config_path)


# --- doctor command ---


@app.command()
def doctor(
    output_json: Annotated[
        bool, typer.Option("--json", help="Output as JSON.")
    ] = False,
) -> None:
    """Run environment diagnostics."""
    from meeting_notes.configure import run_doctor

    run_doctor(output_json=output_json)


# --- resources commands ---


resources_app = typer.Typer(help="Resource catalog and system memory.")
app.add_typer(resources_app, name="resources")


@resources_app.command("show")
def resources_show(
    model: Annotated[Optional[str], typer.Option("--model")] = None,
    device: Annotated[Optional[str], typer.Option("--device")] = None,
) -> None:
    """Show resource estimates and system capabilities."""
    from meeting_notes.configure import run_resources_show

    run_resources_show(model=model, device=device)


# --- models commands ---


models_app = typer.Typer(help="Model management.")
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list() -> None:
    """List available models."""
    from meeting_notes.configure import run_models_list

    run_models_list()


@models_app.command("status")
def models_status() -> None:
    """Show model download status."""
    from meeting_notes.configure import run_models_status

    run_models_status()


@models_app.command("info")
def models_info(
    model: Annotated[str, typer.Argument(help="Model name.")],
    backend: Annotated[str, typer.Option("--backend")] = "whisper_cpp",
) -> None:
    """Show model information."""
    from meeting_notes.configure import run_models_info

    run_models_info(model=model, backend=backend)


@models_app.command("download")
def models_download(
    model: Annotated[str, typer.Argument(help="Model name.")],
    backend: Annotated[str, typer.Option("--backend")] = "whisper_cpp",
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Download a model."""
    from meeting_notes.configure import run_models_download

    run_models_download(model=model, backend=backend, yes=yes)


@models_app.command("verify")
def models_verify(
    model: Annotated[str, typer.Argument(help="Model name.")],
    backend: Annotated[str, typer.Option("--backend")] = "whisper_cpp",
) -> None:
    """Verify a downloaded model."""
    from meeting_notes.configure import run_models_verify

    run_models_verify(model=model, backend=backend)


# --- process command ---


@app.command()
def process(
    input_file: Annotated[str, typer.Argument(help="Path to audio/video file.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
    profile: Annotated[Optional[str], typer.Option("--profile")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show plan without executing.")
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Resume from existing job.")
    ] = True,
    from_stage: Annotated[Optional[str], typer.Option("--from")] = None,
    force_stage: Annotated[Optional[str], typer.Option("--force-stage")] = None,
    finalize_names: Annotated[
        bool, typer.Option("--finalize-names/--no-finalize-names")
    ] = True,
    local_only: Annotated[
        bool, typer.Option("--local-only", help="Reject remote summarization adapters.")
    ] = False,
    copy_to_input: Annotated[
        bool, typer.Option("--copy-to-input", help="Copy finalized files next to input recording.")
    ] = False,
) -> None:
    """Process an audio/video file into meeting notes."""
    from meeting_notes.pipeline import run_pipeline

    run_pipeline(
        input_file=input_file,
        config_path=config_path,
        profile=profile,
        dry_run=dry_run,
        resume=resume,
        from_stage=from_stage,
        force_stage=force_stage,
        finalize_names=finalize_names,
        local_only=local_only,
        copy_to_input=copy_to_input,
    )


# --- individual stage commands ---


@app.command()
def prepare(
    input_file: Annotated[str, typer.Argument(help="Path to audio/video file.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Inspect source and normalize audio."""
    from meeting_notes.pipeline import run_prepare

    run_prepare(input_file=input_file, config_path=config_path)


@app.command()
def transcribe(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Run ASR transcription on normalized audio."""
    from meeting_notes.pipeline import run_transcribe

    run_transcribe(job_dir=job_dir, config_path=config_path)


@app.command()
def diarize(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Run speaker diarization."""
    from meeting_notes.pipeline import run_diarize

    run_diarize(job_dir=job_dir, config_path=config_path)


@app.command()
def merge(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Merge transcript and diarization."""
    from meeting_notes.pipeline import run_merge

    run_merge(job_dir=job_dir, config_path=config_path)


@app.command()
def summarize(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Run summarization."""
    from meeting_notes.pipeline import run_summarize

    run_summarize(job_dir=job_dir, config_path=config_path)


@app.command()
def render(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Render meeting minutes from summary."""
    from meeting_notes.pipeline import run_render

    run_render(job_dir=job_dir, config_path=config_path)


# --- naming commands ---


naming_app = typer.Typer(help="Filename finalization.")
app.add_typer(naming_app, name="naming")


@naming_app.command("preview")
def naming_preview(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Preview finalized filenames."""
    from meeting_notes.pipeline import run_naming_preview

    run_naming_preview(job_dir=job_dir, config_path=config_path)


@naming_app.command("finalize")
def naming_finalize(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Finalize recording and note filenames."""
    from meeting_notes.pipeline import run_naming_finalize

    run_naming_finalize(job_dir=job_dir, config_path=config_path)


# --- benchmark command ---


@app.command()
def benchmark(
    input_file: Annotated[str, typer.Argument(help="Path to audio/video file.")],
    matrix: Annotated[str, typer.Option("--matrix", help="Benchmark matrix YAML.")] = "config/benchmark-matrix.yaml",
    config_path: Annotated[Optional[str], typer.Option("--config")] = None,
) -> None:
    """Run benchmark comparing configurations."""
    from meeting_notes.pipeline import run_benchmark

    run_benchmark(input_file=input_file, matrix=matrix, config_path=config_path)


# --- clean command ---


@app.command()
def clean(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    stage: Annotated[Optional[str], typer.Option("--stage")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Clean job artifacts."""
    from meeting_notes.pipeline import run_clean

    run_clean(job_dir=job_dir, stage=stage, yes=yes)
