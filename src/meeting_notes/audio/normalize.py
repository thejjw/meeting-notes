"""FFmpeg-based audio normalization for transcription input."""

from __future__ import annotations

import tempfile
from pathlib import Path

import structlog

from meeting_notes.subprocess_utils import run_command

log = structlog.get_logger()


def normalize_audio(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
    highpass_hz: int | None = None,
    lowpass_hz: int | None = None,
    loudnorm_enabled: bool = False,
    loudnorm_lufs: float = -20.0,
    loudnorm_range: float = 11.0,
    loudnorm_peak: float = -2.0,
    extra_filters: list[str] | None = None,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Normalize audio to mono 16kHz PCM WAV using FFmpeg.

    Args:
        input_path: Source audio/video file.
        output_path: Destination WAV file.
        sample_rate: Output sample rate (default 16000).
        channels: Output channels (default 1 = mono).
        codec: Output codec (default pcm_s16le).
        highpass_hz: Optional high-pass filter frequency.
        lowpass_hz: Optional low-pass filter frequency.
        loudnorm_enabled: Enable EBU R128 loudness normalization.
        loudnorm_lufs: Target integrated loudness in LUFS.
        loudnorm_range: Target loudness range.
        loudnorm_peak: Target true peak in dB.
        extra_filters: Additional FFmpeg filter expressions.
        ffmpeg_path: Path to ffmpeg executable.

    Returns:
        Path to the normalized output file.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build filter chain
    filters: list[str] = []

    if highpass_hz:
        filters.append(f"highpass=f={highpass_hz}")
    if lowpass_hz:
        filters.append(f"lowpass=f={lowpass_hz}")
    if extra_filters:
        filters.extend(extra_filters)

    if loudnorm_enabled:
        filters.append(
            f"loudnorm=I={loudnorm_lufs}:LRA={loudnorm_range}:TP={loudnorm_peak}"
        )

    # Build FFmpeg command
    args = [
        ffmpeg_path,
        "-y",  # Overwrite output
        "-i", str(input_path),
        "-vn",  # No video
        "-acodec", codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
    ]

    if filters:
        args.extend(["-af", ",".join(filters)])

    args.append(str(output_path))

    log.info(
        "audio.normalize",
        input=str(input_path),
        output=str(output_path),
        sample_rate=sample_rate,
        channels=channels,
        filters=filters if filters else None,
    )

    result = run_command(args, timeout=300.0, label="ffmpeg-normalize")
    if not result.success:
        raise RuntimeError(
            f"FFmpeg normalization failed (exit {result.returncode}):\n"
            f"  stderr: {result.stderr[:500]}"
        )

    log.info(
        "audio.normalized",
        output=str(output_path),
        output_size=output_path.stat().st_size if output_path.exists() else 0,
    )

    return output_path


def create_normalized_path(job_dir: Path, original_filename: str) -> Path:
    """Create the standard normalized audio path within a job directory.

    Args:
        job_dir: Job directory path.
        original_filename: Original source filename (for reference).

    Returns:
        Path to audio/normalized.wav within the job directory.
    """
    return job_dir / "audio" / "normalized.wav"
