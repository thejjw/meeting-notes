"""CLI entry point using Typer."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """meeting-notes: Local-first meeting transcription and summarization."""


# --- configure command ---


@app.command()
def configure(
    config_path: Annotated[str | None, typer.Option("--config", help="Config file path.")] = None,
    accept_defaults: Annotated[
        bool, typer.Option("--accept-defaults", help="Non-interactive safe CPU defaults.")
    ] = False,
    show_detected: Annotated[
        bool, typer.Option("--show-detected", help="Show diagnostics without writing config.")
    ] = False,
    no_configure: Annotated[
        bool, typer.Option("--no-configure", help="Fail immediately instead of opening wizard.")
    ] = False,
    provision: Annotated[
        bool, typer.Option("--provision", help="Install the selected runtime and model.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Confirm large downloads in non-interactive mode.")
    ] = False,
) -> None:
    """Run the interactive configuration wizard or create safe defaults."""
    from meeting_notes.configure import run_configure

    run_configure(
        config_path=config_path,
        accept_defaults=accept_defaults,
        show_detected=show_detected,
        no_configure=no_configure,
        provision=provision,
        yes=yes,
    )


# --- config commands ---


config_app = typer.Typer(help="View and manage configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    resolved: Annotated[
        bool, typer.Option("--resolved", help="Show fully resolved configuration.")
    ] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Show current configuration."""
    from meeting_notes.configure import show_config

    show_config(resolved=resolved, config_path=config_path)


@config_app.command("status")
def config_status(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Show configuration status."""
    from meeting_notes.configure import config_status_cmd

    config_status_cmd(config_path=config_path)


@config_app.command("edit")
def config_edit(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Open configuration file in editor."""
    from meeting_notes.configure import config_edit_cmd

    config_edit_cmd(config_path=config_path)


@config_app.command("reset")
def config_reset(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Reset configuration to safe defaults."""
    from meeting_notes.configure import config_reset_cmd

    config_reset_cmd(config_path=config_path)


# --- doctor command ---


@app.command()
def doctor(
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    smoke_test: Annotated[
        bool, typer.Option("--smoke-test", help="Load the configured model using a generated WAV.")
    ] = False,
) -> None:
    """Run environment diagnostics."""
    from meeting_notes.configure import run_doctor

    run_doctor(output_json=output_json, config_path=config_path, smoke_test=smoke_test)


# --- resources commands ---


resources_app = typer.Typer(help="Resource catalog and system memory.")
app.add_typer(resources_app, name="resources")


@resources_app.command("show")
def resources_show(
    model: Annotated[str | None, typer.Option("--model")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
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
def models_status(
    output_json: Annotated[bool, typer.Option("--json")] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Show model download status."""
    from meeting_notes.configure import run_models_status

    run_models_status(output_json=output_json, config_path=config_path)


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
    backend: Annotated[str, typer.Option("--backend")] = "auto",
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Download a model."""
    from meeting_notes.configure import run_models_download

    run_models_download(model=model, backend=backend, yes=yes, config_path=config_path)


@models_app.command("verify")
def models_verify(
    model: Annotated[str, typer.Argument(help="Model name.")],
    backend: Annotated[str, typer.Option("--backend")] = "whisper_cpp",
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Verify a downloaded model."""
    from meeting_notes.configure import run_models_verify

    run_models_verify(model=model, backend=backend, config_path=config_path)


# --- runtime commands ---


runtime_app = typer.Typer(help="Install and inspect managed whisper.cpp runtimes.")
app.add_typer(runtime_app, name="runtime")


@runtime_app.command("status")
def runtime_status(
    output_json: Annotated[bool, typer.Option("--json")] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Show managed runtime installations."""
    from meeting_notes.configure import run_runtime_status

    run_runtime_status(output_json=output_json, config_path=config_path)


@runtime_app.command("install")
def runtime_install(
    device: Annotated[str, typer.Option("--device", help="cpu or vulkan")] = "cpu",
    version: Annotated[str, typer.Option("--version")] = "v1.9.1",
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Install a verified CPU runtime or build a Vulkan runtime."""
    from meeting_notes.configure import run_runtime_install

    run_runtime_install(device=device, version=version, config_path=config_path, yes=yes)


# --- cache commands ---


cache_app = typer.Typer(help="Inspect and migrate first-party project storage.")
app.add_typer(cache_app, name="cache")


@cache_app.command("status")
def cache_status(
    output_json: Annotated[bool, typer.Option("--json")] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Show project-local and legacy meeting-notes cache usage."""
    from meeting_notes.configure import run_cache_status

    run_cache_status(config_path=config_path, output_json=output_json)


@cache_app.command("migrate")
def cache_migrate(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Migrate recognized legacy Whisper assets into the project cache."""
    from meeting_notes.configure import run_cache_migrate

    try:
        run_cache_migrate(config_path=config_path, yes=yes)
    except RuntimeError as error:
        console.print(f"[red]Cache migration failed:[/red] {error}")
        raise typer.Exit(1) from error


# --- diarization commands ---


diarization_app = typer.Typer(help="Set up and inspect speaker diarization.")
app.add_typer(diarization_app, name="diarization")


@diarization_app.command("setup")
def diarization_setup(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    acceleration: Annotated[
        str | None,
        typer.Option("--acceleration", help="cpu or rocm-hybrid; defaults to config."),
    ] = None,
    model_archive: Annotated[
        str | None,
        typer.Option("--model-archive", help="Verified portable diarization ZIP."),
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Confirm large storage and downloads.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Redownload an existing managed pipeline.")
    ] = False,
) -> None:
    """Provision the diarization model and optional project-local ROCm runtime."""
    from meeting_notes.diarization.setup import run_diarization_setup

    try:
        run_diarization_setup(
            config_path=config_path,
            yes=yes,
            force=force,
            acceleration=acceleration,
            model_archive=Path(model_archive) if model_archive else None,
        )
    except RuntimeError as error:
        console.print(f"[red]Diarization setup failed:[/red] {error}")
        raise typer.Exit(1) from error


@diarization_app.command("status")
def diarization_status(
    output_json: Annotated[bool, typer.Option("--json")] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Show model, project-local storage, and ROCm readiness."""
    from meeting_notes.diarization.setup import run_diarization_status

    run_diarization_status(config_path=config_path, output_json=output_json)


@diarization_app.command("remove-runtime")
def diarization_remove_runtime(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Remove the project-local ROCm runtime and select CPU diarization."""
    from meeting_notes.diarization.setup import run_diarization_runtime_remove

    try:
        run_diarization_runtime_remove(config_path=config_path, yes=yes)
    except RuntimeError as error:
        console.print(f"[red]Diarization runtime removal failed:[/red] {error}")
        raise typer.Exit(1) from error


# --- process command ---


@app.command()
def process(
    input_file: Annotated[str, typer.Argument(help="Path to audio/video file.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show plan without executing.")
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Resume from existing job.")
    ] = True,
    from_stage: Annotated[str | None, typer.Option("--from")] = None,
    force_stage: Annotated[str | None, typer.Option("--force-stage")] = None,
    finalize_names: Annotated[bool, typer.Option("--finalize-names/--no-finalize-names")] = True,
    local_only: Annotated[
        bool, typer.Option("--local-only", help="Reject remote summarization adapters.")
    ] = False,
    copy_to_input: Annotated[
        bool, typer.Option("--copy-to-input", help="Copy finalized files next to input recording.")
    ] = False,
    num_speakers: Annotated[
        int | None,
        typer.Option("--num-speakers", min=1, help="Known exact number of speakers."),
    ] = None,
    min_speakers: Annotated[
        int | None,
        typer.Option("--min-speakers", min=1, help="Minimum number of speakers for this run."),
    ] = None,
    max_speakers: Annotated[
        int | None,
        typer.Option("--max-speakers", min=1, help="Maximum number of speakers for this run."),
    ] = None,
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
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


# --- individual stage commands ---


@app.command()
def prepare(
    input_file: Annotated[str, typer.Argument(help="Path to audio/video file.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Inspect source and normalize audio."""
    from meeting_notes.pipeline import run_prepare

    run_prepare(input_file=input_file, config_path=config_path)


@app.command()
def transcribe(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Run ASR transcription on normalized audio."""
    from meeting_notes.pipeline import run_transcribe

    run_transcribe(job_dir=job_dir, config_path=config_path)


@app.command()
def diarize(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Run speaker diarization."""
    from meeting_notes.pipeline import run_diarize

    run_diarize(job_dir=job_dir, config_path=config_path)


@app.command()
def merge(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Merge transcript and diarization."""
    from meeting_notes.pipeline import run_merge

    run_merge(job_dir=job_dir, config_path=config_path)


@app.command()
def summarize(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Run summarization."""
    from meeting_notes.pipeline import run_summarize

    run_summarize(job_dir=job_dir, config_path=config_path)


@app.command()
def render(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
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
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Preview finalized filenames."""
    from meeting_notes.pipeline import run_naming_preview

    run_naming_preview(job_dir=job_dir, config_path=config_path)


@naming_app.command("finalize")
def naming_finalize(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Finalize recording and note filenames."""
    from meeting_notes.pipeline import run_naming_finalize

    run_naming_finalize(job_dir=job_dir, config_path=config_path)


# --- speaker identification commands ---

speakers_app = typer.Typer(help="Identify speakers and regenerate downstream outputs.")
app.add_typer(speakers_app, name="speakers")


@speakers_app.command("template")
def speakers_template(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    output: Annotated[str | None, typer.Option("--output", help="Template output path.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Regenerate and back up the old template.")
    ] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Create an editable speaker identification template."""
    del config_path  # Reserved for consistent command configuration.
    from meeting_notes.speakers import SpeakerMapError, command_template

    try:
        command_template(job_dir, output, force)
    except SpeakerMapError as error:
        console.print(f"[red]Speaker template failed:[/red] {error}")
        raise typer.Exit(1) from error


@speakers_app.command("apply")
def speakers_apply(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    map_path: Annotated[str | None, typer.Option("--map", help="Speaker map path.")] = None,
    cleanup: Annotated[
        bool, typer.Option("--cleanup", help="Remove tracked superseded publications.")
    ] = False,
    cleanup_all: Annotated[
        bool, typer.Option("--cleanup-all", help="Also remove reproducible upstream artifacts.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm cleanup.")] = False,
    local_only: Annotated[
        bool, typer.Option("--local-only", help="Reject remote summarizers.")
    ] = False,
    without_diarization: Annotated[
        bool,
        typer.Option(
            "--without-diarization",
            help="Remove speaker attribution and regenerate downstream outputs.",
        ),
    ] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Apply speaker names without rerunning transcription or diarization."""
    if cleanup and cleanup_all:
        console.print("[red]--cleanup and --cleanup-all are mutually exclusive.[/red]")
        raise typer.Exit(2)
    from meeting_notes.speakers import SpeakerMapError, command_apply

    try:
        command_apply(
            job_dir,
            map_path,
            config_path,
            cleanup,
            cleanup_all,
            yes,
            local_only,
            without_diarization,
        )
    except SpeakerMapError as error:
        console.print(f"[red]Speaker apply failed:[/red] {error}")
        raise typer.Exit(1) from error


clarify_app = typer.Typer(help="Review AI clarification questions and apply human answers.")
app.add_typer(clarify_app, name="clarify")


@clarify_app.command("template")
def clarify_template(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    output: Annotated[str | None, typer.Option("--output", help="Template output path.")] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Regenerate and back up the old template.")
    ] = False,
) -> None:
    """Create an editable clarifications.yaml with the AI's open questions."""
    from meeting_notes.clarifications import ClarificationError, command_template

    try:
        command_template(job_dir, output, force)
    except ClarificationError as error:
        console.print(f"[red]Clarifications template failed:[/red] {error}")
        raise typer.Exit(1) from error


@clarify_app.command("apply")
def clarify_apply(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    answers: Annotated[
        str | None, typer.Option("--answers", help="Clarifications file path.")
    ] = None,
    local_only: Annotated[
        bool, typer.Option("--local-only", help="Reject remote summarizers.")
    ] = False,
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Apply answered clarifications: correct the job glossary and transcript,
    then re-summarize with the confirmed answers as context and publish a new
    generation."""
    from meeting_notes.clarifications import ClarificationError, command_apply

    try:
        command_apply(job_dir, answers, config_path, local_only)
    except ClarificationError as error:
        console.print(f"[red]Clarify apply failed:[/red] {error}")
        raise typer.Exit(1) from error


glossary_app = typer.Typer(help="Manage the global and per-job glossaries.")
app.add_typer(glossary_app, name="glossary")


@glossary_app.command("promote")
def glossary_promote(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Promote a job's glossary terms into the global glossary."""
    from meeting_notes.clarifications import command_promote

    command_promote(job_dir, config_path)


summarizers_app = typer.Typer(help="Inspect and test summarization adapters.")
app.add_typer(summarizers_app, name="summarizers")


@summarizers_app.command("test")
def summarizers_test(
    config_path: Annotated[str | None, typer.Option("--config")] = None,
    output_json: Annotated[bool, typer.Option("--json", help="Output JSON.")] = False,
) -> None:
    """Run a small schema-validated request without publishing files."""
    from meeting_notes.summarizer_probe import run_summarizer_test

    try:
        run_summarizer_test(config_path=config_path, output_json=output_json)
    except Exception as error:
        console.print(f"[red]Summarizer test failed:[/red] {error}")
        raise typer.Exit(1) from error


# --- benchmark command ---


@app.command()
def benchmark(
    input_file: Annotated[str, typer.Argument(help="Path to audio/video file.")],
    matrix: Annotated[
        str, typer.Option("--matrix", help="Benchmark matrix YAML.")
    ] = "config/benchmark-matrix.yaml",
    config_path: Annotated[str | None, typer.Option("--config")] = None,
) -> None:
    """Run benchmark comparing configurations."""
    from meeting_notes.pipeline import run_benchmark

    run_benchmark(input_file=input_file, matrix=matrix, config_path=config_path)


# --- clean command ---


@app.command()
def clean(
    job_dir: Annotated[str, typer.Argument(help="Job directory path.")],
    stage: Annotated[str | None, typer.Option("--stage")] = None,
    final_only: Annotated[
        bool,
        typer.Option(
            "--final-only",
            help="Keep only the final recording, meeting notes, and transcript.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview --final-only cleanup without changing files."),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Clean job artifacts."""
    from meeting_notes.cleanup import CleanupError, run_final_only_cleanup
    from meeting_notes.pipeline import run_clean

    if final_only and stage:
        console.print("[red]--final-only and --stage are mutually exclusive.[/red]")
        raise typer.Exit(2)
    if dry_run and not final_only:
        console.print("[red]--dry-run requires --final-only.[/red]")
        raise typer.Exit(2)
    if final_only:
        try:
            run_final_only_cleanup(job_dir, yes=yes, dry_run=dry_run)
        except CleanupError as error:
            console.print(f"[red]{error}[/red]")
            raise typer.Exit(1) from error
        return
    run_clean(job_dir=job_dir, stage=stage, yes=yes)
