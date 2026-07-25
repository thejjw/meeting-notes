"""User clarification review: sidecar Q&A file, transcript correction, and re-publication."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from meeting_notes.config import MeetingNotesConfig, load_config
from meeting_notes.jobs import atomic_write_text, file_sha256, load_manifest, save_manifest
from meeting_notes.minutes.render import render_minutes
from meeting_notes.naming import generate_filenames, resolve_date, sanitize_short_title
from meeting_notes.publication import (
    managed_files,
    publication_paths,
    render_transcript_variants,
    write_run_report,
)
from meeting_notes.speakers import _publish_recording
from meeting_notes.transcript.glossary import (
    add_term_to_glossary,
    correct_transcript_segments,
    load_glossary,
    load_layered_glossary,
    merge_glossaries,
    save_glossary,
)

console = Console(stderr=True)
TEMPLATE_VERSION = 1

# Categories whose confirmed answer becomes a glossary term. Answers to
# missing_info questions (owners, dates) are never terminology.
_GLOSSARY_CATEGORIES = {"asr_correction", "term_clarification"}


class ClarificationError(ValueError):
    """An invalid, stale, or unanswered clarifications file."""


def _summary(job_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = job_dir / "summary" / "summary.json"
    if not path.is_file():
        raise ClarificationError(f"Summary not found: {path}. Run summarize first.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ClarificationError(f"Cannot read summary: {error}") from error
    return path, data


def _transcript(job_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = job_dir / "transcript" / "transcript.merged.json"
    if not path.is_file():
        raise ClarificationError(f"Merged transcript not found: {path}. Run merge first.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ClarificationError(f"Cannot read merged transcript: {error}") from error
    if not isinstance(data.get("segments"), list):
        raise ClarificationError(f"Merged transcript has no segments: {path}")
    return path, data


def _item_id(index: int) -> str:
    return f"clarif-{index:03d}"


def _read_existing(path: Path) -> tuple[dict[str, str], list[str]]:
    """Read answers and general comments already typed into a sidecar file."""
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ClarificationError(f"Malformed clarifications file {path}: {error}") from error
    if not isinstance(value, dict):
        return {}, []
    items = value.get("clarifications", {})
    answers: dict[str, str] = {}
    if isinstance(items, dict):
        for item_id, entry in items.items():
            if isinstance(entry, dict):
                answer = entry.get("answer", "")
                if answer:
                    answers[str(item_id)] = str(answer).strip()
    raw_comments = value.get("comments", [])
    comments: list[str] = []
    if isinstance(raw_comments, list):
        comments = [str(c).strip() for c in raw_comments if str(c).strip()]
    return answers, comments


def build_template(
    job_dir: Path,
    preserved_answers: dict[str, str] | None = None,
    preserved_comments: list[str] | None = None,
) -> dict[str, Any]:
    """Build a clarifications template dict from the current summary's open questions."""
    _, summary = _summary(job_dir)
    transcript_path, _ = _transcript(job_dir)
    preserved_answers = preserved_answers or {}
    clarifications = summary.get("user_clarifications") or []
    items: dict[str, Any] = {}
    for index, item in enumerate(clarifications):
        item_id = _item_id(index)
        items[item_id] = {
            "category": item.get("category"),
            "question": item.get("question"),
            "heard_text": item.get("heard_text"),
            "suggested_correction": item.get("suggested_correction"),
            "evidence": item.get("evidence", []),
            "answer": preserved_answers.get(item_id) or item.get("user_answer") or "",
        }
    return {
        "version": TEMPLATE_VERSION,
        "job_id": job_dir.name,
        "transcript_sha256": file_sha256(transcript_path),
        "clarifications": items,
        # General notes not tied to a specific flagged item -- steers the
        # re-summarization LLM without touching deterministic glossary
        # substitution. Duplicate the entry to add more.
        "comments": list(preserved_comments) if preserved_comments else [""],
    }


def write_template(
    job_dir: Path,
    output: Path | None = None,
    *,
    force: bool = False,
) -> tuple[Path | None, str | None]:
    """Write (or refresh) the editable clarifications sidecar file.

    Returns (path, warning). Never overwrites answers or comments a user may
    have already typed unless the existing file is stale for the current
    transcript.
    """
    output = output or job_dir / "clarifications.yaml"
    _, summary = _summary(job_dir)
    if not summary.get("user_clarifications"):
        return None, None

    if output.exists() and not force:
        current = yaml.safe_load(output.read_text(encoding="utf-8")) or {}
        fingerprint = current.get("transcript_sha256") if isinstance(current, dict) else None
        expected = file_sha256(job_dir / "transcript" / "transcript.merged.json")
        if fingerprint != expected:
            candidate = output.with_name(f"{output.stem}.candidate{output.suffix}")
            atomic_write_text(
                candidate,
                yaml.safe_dump(build_template(job_dir), allow_unicode=True, sort_keys=False),
            )
            return (
                candidate,
                f"existing template is stale ({fingerprint or 'no fingerprint'} != {expected})",
            )
        return output, None

    answers: dict[str, str] = {}
    comments: list[str] = []
    if output.exists():
        answers, comments = _read_existing(output)
        backup = output.with_name(
            f"{output.name}.bak-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
        shutil.copy2(output, backup)
    template = build_template(job_dir, answers, comments)
    atomic_write_text(output, yaml.safe_dump(template, allow_unicode=True, sort_keys=False))
    return output, None


def load_answers(
    job_dir: Path, path: Path
) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, Any], str]:
    """Load and validate an answered clarifications file.

    Returns (items_by_id, comments, summary, transcript_sha256).
    """
    transcript_path, _ = _transcript(job_dir)
    _, summary = _summary(job_dir)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ClarificationError(f"Malformed clarifications file {path}: {error}") from error
    if not isinstance(document, dict):
        raise ClarificationError(f"Clarifications file must be a YAML mapping: {path}")
    if document.get("version") != TEMPLATE_VERSION:
        raise ClarificationError(
            f"Unsupported clarifications template version {document.get('version')!r}; "
            "run `meeting-notes clarify template --force JOB_DIR`."
        )
    if document.get("job_id") != job_dir.name:
        raise ClarificationError(
            f"Clarifications file belongs to job {document.get('job_id')!r}, not {job_dir.name!r}."
        )
    actual_hash = file_sha256(transcript_path)
    if document.get("transcript_sha256") != actual_hash:
        raise ClarificationError(
            "Clarifications file is stale for the current transcript; run "
            "`meeting-notes clarify template --force JOB_DIR`, review it, then apply again."
        )
    items = document.get("clarifications", {})
    if not isinstance(items, dict):
        raise ClarificationError("The 'clarifications' value must be a mapping.")
    raw_comments = document.get("comments", [])
    if not isinstance(raw_comments, list):
        raise ClarificationError("The 'comments' value must be a list.")
    comments = [str(c).strip() for c in raw_comments if str(c).strip()]
    return items, comments, summary, actual_hash


def _apply_report_stages() -> dict[str, dict[str, str]]:
    return {
        "prepare": {"status": "skipped", "message": "reused existing job"},
        "transcribe": {"status": "skipped", "message": "reused merged transcript"},
        "diarize": {"status": "skipped", "message": "reused existing speaker labels"},
        "merge": {"status": "completed", "message": "job glossary corrections re-applied"},
        "summarize": {"status": "completed", "message": "re-run with confirmed answers as context"},
        "render": {"status": "completed"},
        "finalize": {"status": "completed"},
    }


def apply_clarifications(
    job_dir: Path,
    answers_path: Path | None,
    config: MeetingNotesConfig,
    *,
    local_only: bool = False,
) -> dict[str, Any]:
    """Apply answered clarifications: correct the job glossary and transcript,
    re-summarize with the confirmed answers as authoritative context, and
    publish the result as a new generation.
    """
    answers_path = answers_path or job_dir / "clarifications.yaml"
    if not answers_path.is_file():
        raise ClarificationError(
            f"Clarifications file not found: {answers_path}. Run "
            "`meeting-notes clarify template JOB_DIR` first."
        )
    items, comments, summary, transcript_hash = load_answers(job_dir, answers_path)

    answered = {
        item_id: entry
        for item_id, entry in items.items()
        if isinstance(entry, dict) and str(entry.get("answer") or "").strip()
    }
    if not answered and not comments:
        raise ClarificationError("No answers or comments found in clarifications file.")

    clarifications = list(summary.get("user_clarifications") or [])
    context_lines: list[str] = []
    glossary_updates: list[tuple[str, str]] = []
    applied_count = 0

    for index, clarification in enumerate(clarifications):
        entry = answered.get(_item_id(index))
        if not entry:
            continue
        answer = str(entry.get("answer")).strip()
        clarification["user_answer"] = answer
        clarification["resolved"] = True
        applied_count += 1

        category = clarification.get("category")
        heard_text = clarification.get("heard_text")
        question = clarification.get("question", "")
        context_lines.append(f"- Q: {question}\n  A: {answer}")
        if category in _GLOSSARY_CATEGORIES and heard_text:
            glossary_updates.append((answer, heard_text))

    if answered and applied_count == 0:
        raise ClarificationError(
            "Answers did not match any pending clarification. The clarifications "
            "file may be for a different summary; run `clarify template --force`."
        )

    summary["user_clarifications"] = clarifications

    job_glossary_path = job_dir / "glossary.yaml"
    for canonical, alias in glossary_updates:
        add_term_to_glossary(job_glossary_path, canonical=canonical, alias=alias)

    transcript_path, transcript_data = _transcript(job_dir)
    global_glossary_path = Path(config.glossary.path) if config.glossary.path else None
    glossary = load_layered_glossary(global_glossary_path, job_glossary_path)
    corrected_segments, corrections = correct_transcript_segments(
        transcript_data.get("segments", []),
        glossary,
        case_sensitive=config.glossary.case_sensitive,
    )
    transcript_data["segments"] = corrected_segments
    atomic_write_text(
        transcript_path,
        json.dumps(transcript_data, indent=2, ensure_ascii=False),
    )

    summary_path = job_dir / "summary" / "summary.json"
    atomic_write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False))

    manifest = load_manifest(job_dir)
    old_generations = manifest.get("clarification_publications", {}).get("generations", [])
    generation_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    started_at = datetime.now(UTC).isoformat()
    staging = job_dir / "output" / f".clarify-{generation_id}"
    generation = job_dir / "output" / "finalized" / generation_id
    staging.mkdir(parents=True)
    try:
        render_dir = staging / ".render"
        transcript_paths = render_transcript_variants(transcript_data, render_dir)

        context_sections: list[str] = []
        if context_lines:
            context_sections.append(
                "The following corrections were confirmed by a human reviewer after "
                "reviewing the transcript and are authoritative over the raw transcript "
                "wording, including for terms that were not literally substituted:\n"
                + "\n".join(context_lines)
            )
        if comments:
            context_sections.append(
                "Additional reviewer notes (general guidance, not tied to a specific "
                "flagged item):\n" + "\n".join(f"- {c}" for c in comments)
            )
        extra_context = "\n\n".join(context_sections)
        from meeting_notes.pipeline import _summarize_transcript

        updated_summary = _summarize_transcript(
            corrected_segments,
            config,
            local_only,
            extra_context=extra_context,
        )
        updated_summary["user_clarifications"] = summary["user_clarifications"]

        (staging / "summary.json").write_text(
            json.dumps(updated_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (staging / "minutes.md").write_text(
            render_minutes(
                updated_summary,
                source_filename=manifest.get("source", {}).get("original_filename", ""),
            ),
            encoding="utf-8",
        )
        title = sanitize_short_title(
            updated_summary.get("short_title", "meeting"),
            max_length=config.naming.max_short_title_characters,
        )
        original = str(manifest.get("source", {}).get("original_filename") or "recording.m4a")
        source_path = Path(manifest.get("source", {}).get("original_path") or original)
        date, date_source = resolve_date(
            updated_summary,
            media_creation_time=manifest.get("source", {}).get("creation_time", ""),
            source_mtime=source_path.stat().st_mtime if source_path.exists() else 0,
            source_order=config.naming.date_source_order,
        )
        names = generate_filenames(
            date,
            title,
            Path(original).suffix,
            recording_template=config.naming.recording_template,
            minutes_template=config.naming.minutes_template,
            json_template=config.naming.json_export_template,
            transcript_json_template=config.naming.transcript_json_template,
            transcript_markdown_template=config.naming.transcript_markdown_template,
            transcript_srt_template=config.naming.transcript_srt_template,
            transcript_vtt_template=config.naming.transcript_vtt_template,
        )
        layout = publication_paths(staging, names)
        for key, source in transcript_paths.items():
            target = layout[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        shutil.rmtree(render_dir)
        (staging / "minutes.md").rename(layout["minutes"])
        layout["json_export"].parent.mkdir(parents=True, exist_ok=True)
        (staging / "summary.json").rename(layout["json_export"])
        _publish_recording(job_dir, layout["recording"], manifest)
        relative_outputs = [
            str(path.relative_to(staging))
            for key, path in layout.items()
            if key != "run_report" and path.exists()
        ]
        write_run_report(
            layout["run_report"],
            run_id=generation_id,
            operation="clarify apply",
            status="success",
            started_at=started_at,
            manifest=manifest,
            config=config,
            transcript_sha256=file_sha256(transcript_path),
            outputs=relative_outputs,
            messages=[
                f"Applied {applied_count} clarification answer(s).",
                f"Added {len(glossary_updates)} term(s) to {job_glossary_path.name}.",
                f"Corrected {len(corrections)} transcript occurrence(s) via glossary substitution.",
                f"Included {len(comments)} reviewer note(s).",
            ],
            stages=_apply_report_stages(),
            asr_activity="not run (reused transcript)",
            diarization_activity="not run (reused existing speaker labels)",
        )
        generation.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, generation)
    except Exception as error:
        shutil.rmtree(staging, ignore_errors=True)
        with suppress(OSError):
            write_run_report(
                job_dir / "output" / "runs" / generation_id / "report.md",
                run_id=generation_id,
                operation="clarify apply",
                status="failed",
                started_at=started_at,
                manifest=manifest,
                config=config,
                error=error,
                stages={},
            )
        raise

    final_layout = publication_paths(generation, names)
    managed = managed_files(generation)
    canonical_sources = {
        job_dir / "transcript" / "transcript.merged.md": final_layout["transcript_markdown"],
        job_dir / "transcript" / "transcript.merged.srt": final_layout["transcript_srt"],
        job_dir / "transcript" / "transcript.merged.vtt": final_layout["transcript_vtt"],
        job_dir / "summary" / "summary.json": final_layout["json_export"],
        job_dir / "output" / "minutes.md": final_layout["minutes"],
        job_dir / "output" / "summary.json": final_layout["json_export"],
    }
    for target, source in canonical_sources.items():
        atomic_write_text(target, source.read_text(encoding="utf-8"))

    generations = []
    for item in old_generations:
        copied = dict(item)
        copied["state"] = "superseded"
        generations.append(copied)
    generation_record = {
        "id": generation_id,
        "state": "active",
        "created_at": datetime.now(UTC).isoformat(),
        "transcript_sha256": transcript_hash,
        "applied_count": applied_count,
        "glossary_terms_added": len(glossary_updates),
        "comment_count": len(comments),
        "managed_paths": managed,
        "external_paths": [],
        "date_source": date_source,
    }
    generations.append(generation_record)
    manifest["clarification_publications"] = {
        "active_generation": generation_id,
        "generations": generations,
    }
    save_manifest(job_dir, manifest)

    note_suffix = f" and {len(comments)} note(s)" if comments else ""
    console.print(
        f"[green]Applied {applied_count} clarification answer(s){note_suffix}; "
        f"published generation {generation_id}.[/green]"
    )
    return generation_record


def load_meeting_config(config_path: str | None) -> MeetingNotesConfig:
    return load_config(config_path)


def command_template(job_dir: str, output: str | None, force: bool) -> None:
    path, warning = write_template(
        Path(job_dir).resolve(), Path(output).resolve() if output else None, force=force
    )
    if warning:
        console.print(f"[yellow]{warning}; wrote replacement candidate: {path}[/yellow]")
    elif path:
        console.print(str(path))
    else:
        console.print("[yellow]No open clarifications found for this job.[/yellow]")


def command_apply(
    job_dir: str,
    answers_path: str | None,
    config_path: str | None,
    local_only: bool,
) -> None:
    path = Path(job_dir).resolve()
    answers = Path(answers_path).resolve() if answers_path else None
    apply_clarifications(
        path,
        answers,
        load_meeting_config(config_path),
        local_only=local_only,
    )


def command_promote(job_dir: str, config_path: str | None) -> None:
    """Promote a job's glossary terms into the global glossary."""
    job_path = Path(job_dir).resolve()
    job_glossary_path = job_path / "glossary.yaml"
    if not job_glossary_path.exists():
        console.print(f"[yellow]No job glossary found: {job_glossary_path}[/yellow]")
        return
    config = load_meeting_config(config_path)
    global_path = Path(config.glossary.path)
    base = load_glossary(global_path if global_path.exists() else None)
    overlay = load_glossary(job_glossary_path)
    before = {t.canonical for t in base.terms}
    merged = merge_glossaries(base, overlay)
    added = [t.canonical for t in merged.terms if t.canonical not in before]
    save_glossary(merged, global_path)
    if added:
        console.print(f"[green]Promoted {len(added)} term(s) to {global_path}:[/green]")
        for canonical in added:
            console.print(f"  - {canonical}")
    else:
        console.print(f"[green]Global glossary already up to date: {global_path}[/green]")
