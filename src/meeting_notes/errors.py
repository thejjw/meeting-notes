"""Domain-specific exceptions for meeting-notes."""

from __future__ import annotations


class MeetingNotesError(Exception):
    """Base exception for all meeting-notes errors."""


class ConfigurationError(MeetingNotesError):
    """Invalid, missing, or incomplete configuration."""


class ConfigNotFoundError(ConfigurationError):
    """No valid configuration found."""


class ConfigValidationError(ConfigurationError):
    """Configuration exists but fails validation."""


class DependencyMissingError(MeetingNotesError):
    """A required external tool or library is not installed."""


class UnsupportedBackendError(MeetingNotesError):
    """Selected backend/device combination is not supported."""


class ModelMissingError(MeetingNotesError):
    """Requested model is not downloaded or path is invalid."""


class ModelChecksumError(MeetingNotesError):
    """Downloaded model checksum does not match expected value."""


class AudioToolError(MeetingNotesError):
    """FFmpeg or FFprobe execution failed."""


class ASRError(MeetingNotesError):
    """ASR backend execution failed."""


class ASRTimeoutError(ASRError):
    """ASR process exceeded timeout."""


class OutOfMemoryError(MeetingNotesError):
    """Insufficient memory for the requested operation."""


class DeviceInitError(MeetingNotesError):
    """GPU or accelerator device failed to initialize."""


class DiarizationError(MeetingNotesError):
    """Diarization stage failed."""


class DiarizationUnavailableError(DiarizationError):
    """Diarization is not available (missing dependency, token, or model)."""


class SummarizerError(MeetingNotesError):
    """Summarization stage failed."""


class SummarizerAuthError(SummarizerError):
    """Summarizer authentication or quota failure."""


class SummarizerSchemaError(SummarizerError):
    """Summarizer output does not match expected JSON schema."""


class EvidenceValidationError(SummarizerError):
    """Generated summary references non-existent transcript segments."""


class StageCancelledError(MeetingNotesError):
    """Pipeline stage was cancelled by user."""


class FinalizationError(MeetingNotesError):
    """Filename finalization failed."""


class CollisionError(FinalizationError):
    """Filename collision could not be resolved."""


class JobError(MeetingNotesError):
    """Job directory or manifest operation failed."""
