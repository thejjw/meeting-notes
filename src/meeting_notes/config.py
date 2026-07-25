"""Configuration models, discovery, and persistence."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# OS-specific default config locations


def _default_config_path() -> Path:
    """Return the OS-appropriate default config path."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "meeting-notes" / "config.yaml"


DEFAULT_CONFIG_PATH = _default_config_path()
PROJECT_CONFIG_NAME = "meeting-notes.yaml"
CONFIG_ENV_VAR = "MEETING_NOTES_CONFIG"


class SetupConfig(BaseModel):
    """Tracks whether first-run configuration has completed."""

    completed: bool = False
    profile: str = "safe-cpu"
    created_at: str | None = None
    created_by_version: str | None = None
    resource_catalog_version: int = 1


class ProjectConfig(BaseModel):
    """Project-level paths and behavior."""

    data_dir: str = "./data"
    cache_dir: str = "./cache"
    temp_dir: str | None = None
    copy_source_into_job: bool = True
    keep_intermediates: bool = True
    resume: bool = True
    overwrite: bool = False
    log_level: str = "INFO"


class WhisperCppBackendOptions(BaseModel):
    """whisper.cpp-specific backend options."""

    model_variant: str = "fp16"
    hip_arch: str | None = None
    gpu_device: str | None = None
    flash_attention: bool = True
    extra_args: list[str] = Field(default_factory=list)


class OpenAIWhisperBackendOptions(BaseModel):
    """openai-whisper-specific backend options."""

    torch_device: str = "auto"
    fp16: bool = True
    extra_options: dict[str, Any] = Field(default_factory=dict)


class FasterWhisperBackendOptions(BaseModel):
    """faster-whisper-specific backend options."""

    device: str = "auto"
    compute_type: str = "auto"
    cpu_threads: int = 0
    batch_size: int = 1
    extra_options: dict[str, Any] = Field(default_factory=dict)


class ASRBackendOptions(BaseModel):
    """Backend-specific ASR options."""

    whisper_cpp: WhisperCppBackendOptions = Field(default_factory=WhisperCppBackendOptions)
    openai_whisper: OpenAIWhisperBackendOptions = Field(default_factory=OpenAIWhisperBackendOptions)
    faster_whisper: FasterWhisperBackendOptions = Field(default_factory=FasterWhisperBackendOptions)


class RuntimeConfig(BaseModel):
    """Runtime environment and backend selection."""

    platform: str = "auto"
    device: str = "cpu"
    asr_backend: str = "whisper_cpp"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    whisper_cpp_path: str = "whisper-cli"
    python_executable: str | None = None
    threads: int = 0
    max_auto_threads: int = 8
    reserve_logical_cores: int = 2
    workers: int = 1
    process_priority: str = "normal"
    allow_fallback: bool = False
    environment_overrides: dict[str, str] = Field(default_factory=dict)


class ResourcesConfig(BaseModel):
    """Resource catalog and estimation settings."""

    catalog_path: str = "./config/resource-catalog.yaml"
    prefer_local_measurements: bool = True
    cpu_system_headroom_gb: float = 2.0
    gpu_headroom_ratio: float = 0.35
    large_download_confirmation_mib: int = 1024
    warn_below_recommended: bool = True
    allow_unknown_estimates: bool = True


class LoudnormConfig(BaseModel):
    """FFmpeg loudnorm filter settings."""

    enabled: bool = False
    integrated_lufs: float = -20.0
    loudness_range: float = 11.0
    true_peak_db: float = -2.0


class AudioNormalizeConfig(BaseModel):
    """Audio normalization settings."""

    enabled: bool = True
    highpass_hz: int | None = None
    lowpass_hz: int | None = None
    loudnorm: LoudnormConfig = Field(default_factory=LoudnormConfig)
    extra_filters: list[str] = Field(default_factory=list)


class AudioChunkingConfig(BaseModel):
    """Audio chunking settings."""

    mode: str = "auto"
    trigger_duration_minutes: float = 45.0
    max_chunk_minutes: float = 20.0
    overlap_seconds: float = 2.0
    split_on_silence: bool = True
    minimum_silence_seconds: float = 0.5


class AudioConfig(BaseModel):
    """Audio processing settings."""

    output_sample_rate: int = 16000
    output_channels: int = 1
    output_codec: str = "pcm_s16le"
    preserve_original: bool = True
    normalize: AudioNormalizeConfig = Field(default_factory=AudioNormalizeConfig)
    chunking: AudioChunkingConfig = Field(default_factory=AudioChunkingConfig)


class VADConfig(BaseModel):
    """Voice Activity Detection settings."""

    enabled: bool = True
    backend: str = "backend_native"
    model_path: str | None = None
    threshold: float = 0.50
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    speech_pad_ms: int = 200
    max_speech_seconds: float = 900.0


class ASRConfig(BaseModel):
    """ASR (speech recognition) settings."""

    model: str = "medium"
    model_path: str | None = None
    model_cache_dir: str = "./cache/models"
    language: str = "ko"
    task: str = "transcribe"
    initial_prompt: str | None = None
    glossary_path: str = "./config/glossary.yaml"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    condition_on_previous_text: bool = True
    word_timestamps: bool = False
    suppress_blank: bool = True
    no_speech_threshold: float = 0.60
    logprob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    output_formats: list[str] = Field(default_factory=lambda: ["json", "md", "srt", "vtt"])
    backend_options: ASRBackendOptions = Field(default_factory=ASRBackendOptions)


class DiarizationConfig(BaseModel):
    """Speaker diarization settings."""

    enabled: bool = True
    backend: str = "pyannote"
    model: str = "pyannote/speaker-diarization-community-1"
    model_path: str | None = None
    token_env: str = "HF_TOKEN"
    device: str = "auto"
    num_speakers: int | None = None
    min_speakers: int = 2
    max_speakers: int = 8
    use_exclusive_diarization: bool = True
    assignment_method: str = "maximum_overlap"
    minimum_overlap_ratio: float = 0.15
    nearest_tolerance_seconds: float = 1.0
    unknown_speaker_label: str = "UNKNOWN"
    speaker_map_path: str | None = None
    write_rttm: bool = True


class GlossaryConfig(BaseModel):
    """Glossary and terminology preservation settings."""

    enabled: bool = True
    path: str = "./config/glossary.yaml"
    use_asr_initial_prompt: bool = True
    conservative_post_replace: bool = True
    case_sensitive: bool = False
    record_corrections: bool = True


class CodexConfig(BaseModel):
    """Codex CLI-specific summarization options."""

    executable: str = "codex"
    model: str | None = None
    reasoning_effort: str | None = None
    ephemeral: bool = True
    skip_git_repo_check: bool = True
    ignore_user_config: bool = False
    ignore_rules: bool = False
    extra_args: list[str] = Field(default_factory=list)


class ClaudeConfig(BaseModel):
    """Claude Code CLI-specific summarization options."""

    executable: str = "claude"
    model: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    launcher_execution: Literal["direct", "powershell", "posix_shell"] = "direct"
    launcher_command: str | None = None

    @model_validator(mode="after")
    def validate_launcher(self) -> ClaudeConfig:
        if self.launcher_execution == "direct" and self.launcher_command:
            raise ValueError("claude.launcher_command requires a shell launcher_execution")
        if self.launcher_execution != "direct" and not self.launcher_command:
            raise ValueError("claude.launcher_command is required for shell launchers")
        return self


class LocalCommandConfig(BaseModel):
    """Local command summarizer options."""

    protocol: Literal["request_json_v1", "transcript_stdin_v0"] = "request_json_v1"
    execution: Literal["direct", "powershell", "posix_shell"] = "direct"
    command: list[str] = Field(default_factory=list)
    script: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_execution(self) -> LocalCommandConfig:
        if self.execution == "direct":
            if self.script:
                raise ValueError("local_command.script is only valid for shell execution")
        else:
            if self.command:
                raise ValueError("local_command.command and shell execution are mutually exclusive")
            if not self.script:
                raise ValueError("local_command.script is required for shell execution")
        return self


class SummarizationConfig(BaseModel):
    """Summarization settings."""

    enabled: bool = True
    backend: str = "codex"
    source_transcript: str = "merged"
    prompt_path: str = "./prompts/meeting-summary.md"
    chunk_prompt_path: str = "./prompts/meeting-chunk-summary.md"
    output_schema_path: str = "./schemas/meeting-summary.schema.json"
    chunk_schema_path: str = "./schemas/meeting-chunk-summary.schema.json"
    language: str = "ko"
    max_single_pass_characters: int = 120000
    chunk_target_characters: int = 60000
    chunk_overlap_segments: int = 2
    retries: int = 2
    timeout_seconds: int = 1800
    codex: CodexConfig = Field(default_factory=CodexConfig)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    local_command: LocalCommandConfig = Field(default_factory=LocalCommandConfig)


class NamingConfig(BaseModel):
    """Post-summary filename finalization settings."""

    enabled: bool = True
    recording_mode: str = "managed_copy"
    finalize_after_summary: bool = True
    allow_without_summary: bool = False
    date_source_order: list[str] = Field(
        default_factory=lambda: [
            "summary_meeting_date",
            "media_creation_time",
            "source_mtime",
            "processing_date",
        ]
    )
    recording_template: str = "{date}_{short_title}{extension}"
    minutes_template: str = "{date}_{short_title}_meeting-notes.md"
    json_export_template: str = "{date}_{short_title}_meeting-notes.json"
    transcript_json_template: str = "{date}_{short_title}_transcript.json"
    transcript_markdown_template: str = "{date}_{short_title}_transcript.md"
    transcript_srt_template: str = "{date}_{short_title}_transcript.srt"
    transcript_vtt_template: str = "{date}_{short_title}_transcript.vtt"
    preserve_unicode: bool = True
    whitespace_replacement: str = "-"
    max_short_title_characters: int = 48
    max_filename_characters: int = 180
    collision_policy: str = "increment"
    copy_method: str = "auto"
    preview_before_apply: bool = True


class RenderConfig(BaseModel):
    """Meeting minutes rendering settings."""

    title_template: str = "{date} {title}"
    timestamp_format: str = "HH:MM:SS"
    include_executive_summary: bool = True
    include_participants: bool = True
    include_agenda: bool = True
    include_decisions: bool = True
    include_action_items: bool = True
    include_open_questions: bool = True
    include_risks: bool = True
    include_evidence_links: bool = True
    include_full_transcript_link: bool = True
    unknown_value_text: str = "미정"
    output_markdown: bool = True


class BenchmarkConfig(BaseModel):
    """Benchmark settings."""

    warmup_runs: int = 0
    repeat_runs: int = 1
    collect_peak_memory: bool = True
    collect_gpu_metrics: bool = True
    reference_transcript: str | None = None
    language_for_scoring: str = "ko"


class MeetingNotesConfig(BaseModel):
    """Top-level configuration for meeting-notes."""

    version: int = 1
    setup: SetupConfig = Field(default_factory=SetupConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    glossary: GlossaryConfig = Field(default_factory=GlossaryConfig)
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    naming: NamingConfig = Field(default_factory=NamingConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


def _resolve_config_path(explicit: str | None = None) -> Path | None:
    """Resolve config file path through the discovery order.

    Returns None when no configuration is found.
    """
    # 1. Explicit --config path
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    # 2. Environment variable
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        p = Path(env_path)
        return p if p.exists() else None

    # 3. Project-local config
    project_path = Path(PROJECT_CONFIG_NAME)
    if project_path.exists():
        return project_path

    # 4. OS-specific user config
    os_path = _default_config_path()
    if os_path.exists():
        return os_path

    return None


def load_config(explicit_path: str | None = None) -> MeetingNotesConfig:
    """Load and validate configuration from the discovery order.

    Raises ConfigNotFoundError when no valid configuration exists.
    """
    from meeting_notes.errors import ConfigNotFoundError, ConfigValidationError

    path = _resolve_config_path(explicit_path)
    if path is None:
        raise ConfigNotFoundError(
            f"No valid configuration found. Run 'meeting-notes configure' to create one, "
            f"or use '--accept-defaults' for a safe CPU-only setup."
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigValidationError(f"Config file {path} is not a valid YAML mapping.")
        config = MeetingNotesConfig(**raw)
    except Exception as e:
        raise ConfigValidationError(f"Failed to load config from {path}: {e}") from e

    if not config.setup.completed:
        raise ConfigNotFoundError(
            f"Configuration at {path} has not been completed. "
            f"Run 'meeting-notes configure' to finish setup."
        )

    return config


def save_config(config: MeetingNotesConfig, path: Path) -> None:
    """Atomically save configuration to disk.

    Writes to a temp file, validates, then renames for atomicity.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    content = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Atomic write: temp file + rename
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix="config.")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        # os.replace is atomic on the same filesystem and replaces an existing
        # destination on both Windows and POSIX.
        os.replace(tmp_path, path)
    except Exception:
        os.close(fd) if not os.get_inheritable(fd) else None  # type: ignore[arg-type]
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
