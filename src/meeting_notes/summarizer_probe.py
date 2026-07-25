"""Low-cost configured summarizer integration probe."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from meeting_notes.config import load_config
from meeting_notes.summarization.adapters import (
    configured_adapter_options,
    get_adapter,
    summarizer_provenance,
)

_PROBE_TRANSCRIPT = (
    "[seg-000000] [00:00:00] 프로젝트 상태를 검토했다.\n"
    "[seg-000001] [00:00:10] 다음 주에 테스트 결과를 다시 확인하기로 했다."
)


def run_summarizer_test(
    *,
    config_path: str | None = None,
    output_json: bool = False,
) -> dict[str, Any]:
    """Invoke and validate the configured summarizer without publishing files."""
    config = load_config(config_path)
    summary = config.summarization
    adapter = get_adapter(summary.backend, **configured_adapter_options(summary))
    if not adapter.is_available():
        raise RuntimeError(f"Summarization adapter '{summary.backend}' is not available")

    prompt_path = Path(summary.prompt_path)
    prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else "Summarize this meeting transcript."
    )
    schema_path = Path(summary.output_schema_path) if summary.output_schema_path else None
    started = time.monotonic()
    result = adapter.summarize(
        _PROBE_TRANSCRIPT,
        prompt=prompt,
        schema_path=schema_path,
        timeout_seconds=summary.timeout_seconds,
        metadata={
            "language": summary.language,
            "speaker_resolution": "none",
            "probe": True,
        },
    )
    payload: dict[str, Any] = {
        "success": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "provider": summarizer_provenance(summary),
        "title": result.data.get("title"),
        "short_title": result.data.get("short_title"),
    }
    console = Console()
    if output_json:
        sys.stdout.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    else:
        console.print(
            f"[green]Summarizer test passed[/green] in {payload['elapsed_seconds']}s "
            f"using {summary.backend}"
        )
        if payload["provider"].get("launcher"):
            console.print(f"  Launcher: {payload['provider']['launcher']}")
        console.print(f"  Title: {payload['title'] or '(none)'}")
    return payload
