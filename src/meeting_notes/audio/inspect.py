"""FFprobe-based audio/video metadata inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from meeting_notes.subprocess_utils import run_command

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()


@dataclass
class StreamInfo:
    """Information about a single stream."""

    index: int
    codec_type: str  # "audio", "video", "subtitle"
    codec_name: str = ""
    sample_rate: str = ""
    channels: int = 0
    channel_layout: str = ""
    bits_per_sample: int = 0
    bit_rate: str = ""
    width: int = 0
    height: int = 0
    duration: str = ""

    @property
    def sample_rate_int(self) -> int:
        try:
            return int(self.sample_rate)
        except (ValueError, TypeError):
            return 0


@dataclass
class MediaInfo:
    """Complete media file metadata from FFprobe."""

    file_path: str
    file_size_bytes: int = 0
    file_hash: str = ""
    format_name: str = ""
    format_long_name: str = ""
    duration_seconds: float = 0.0
    bit_rate: str = ""
    nb_streams: int = 0
    streams: list[StreamInfo] = field(default_factory=list)
    creation_time: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    raw_output: dict[str, Any] = field(default_factory=dict)

    @property
    def has_audio(self) -> bool:
        return any(s.codec_type == "audio" for s in self.streams)

    @property
    def has_video(self) -> bool:
        return any(s.codec_type == "video" for s in self.streams)

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.codec_type == "audio"]

    @property
    def primary_audio(self) -> StreamInfo | None:
        audio = self.audio_streams
        return audio[0] if audio else None

    @property
    def duration_timestamp(self) -> str:
        """Format duration as HH:MM:SS."""
        total = int(self.duration_seconds)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"


def inspect_media(
    file_path: Path,
    ffprobe_path: str = "ffprobe",
) -> MediaInfo:
    """Run FFprobe to extract media metadata.

    Args:
        file_path: Path to the media file.
        ffprobe_path: Path to ffprobe executable.

    Returns:
        MediaInfo with all extracted metadata.

    Raises:
        RuntimeError: If FFprobe fails or file doesn't exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    info = MediaInfo(
        file_path=str(file_path),
        file_size_bytes=file_path.stat().st_size,
    )

    # Compute file hash
    import hashlib

    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    info.file_hash = h.hexdigest()[:16]

    # Run FFprobe with JSON output
    args = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    result = run_command(args, timeout=60.0, label="ffprobe")
    if not result.success:
        raise RuntimeError(f"FFprobe failed: {result.stderr[:500]}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"FFprobe returned invalid JSON: {e}") from e

    info.raw_output = data

    # Parse format
    fmt = data.get("format", {})
    info.format_name = fmt.get("format_name", "")
    info.format_long_name = fmt.get("format_long_name", "")
    info.duration_seconds = float(fmt.get("duration", 0))
    info.bit_rate = fmt.get("bit_rate", "")
    info.nb_streams = int(fmt.get("nb_streams", 0))

    # Parse creation time from tags
    tags = fmt.get("tags", {})
    info.creation_time = tags.get("creation_time", "")
    info.tags = {k: str(v) for k, v in tags.items()}

    # Parse streams
    for stream_data in data.get("streams", []):
        stream = StreamInfo(
            index=stream_data.get("index", 0),
            codec_type=stream_data.get("codec_type", ""),
            codec_name=stream_data.get("codec_name", ""),
            sample_rate=stream_data.get("sample_rate", ""),
            channels=stream_data.get("channels", 0),
            channel_layout=stream_data.get("channel_layout", ""),
            bits_per_sample=stream_data.get("bits_per_sample", 0),
            bit_rate=stream_data.get("bit_rate", ""),
            width=stream_data.get("width", 0),
            height=stream_data.get("height", 0),
            duration=stream_data.get("duration", ""),
        )
        info.streams.append(stream)

    log.info(
        "media.inspected",
        path=str(file_path),
        format=info.format_name,
        duration=info.duration_timestamp,
        streams=info.nb_streams,
        has_audio=info.has_audio,
        has_video=info.has_video,
    )

    return info
