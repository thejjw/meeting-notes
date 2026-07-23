# Changelog

All notable changes to meeting-notes will be documented in this file.

## [0.1.0] - 2026-07-22

### Added

#### Core Pipeline
- Audio inspection via FFprobe with metadata extraction
- FFmpeg audio normalization (mono, 16kHz, PCM WAV)
- ASR backend system with whisper.cpp, openai-whisper, faster-whisper adapters
- Transcript output in JSON, Markdown, SRT, and WebVTT formats
- Chunk transcript merging with overlap deduplication
- Glossary system for English term preservation
- Speaker diarization via pyannote.audio (optional)
- Speaker reconciliation with maximum-overlap assignment
- Deterministic meeting minutes rendering from summary JSON
- Filename finalization with date/topic naming and collision handling

#### Configuration
- Pydantic-based config model with discovery order
- Interactive first-run wizard with system diagnostics
- Resource catalog with whisper model memory estimates
- Profile system (safe-cpu, vulkan, amd-rocm, nvidia-cuda, balanced, accuracy)

#### Summarization
- Interchangeable summarizer adapter system
- Built-in adapters: Codex CLI, OpenCode, Mimo Code, Claude Code
- Generic local command adapter for custom AI tools
- Long transcript chunking for hierarchical summarization

#### Docker
- Dockerfile for whisper.cpp CPU build
- Docker wrapper for containerized transcription
- docker-compose.yml for easy management

#### CLI Commands
- `meeting-notes configure` - Configuration wizard
- `meeting-notes process` - Full pipeline
- `meeting-notes prepare` - Audio inspection/normalization
- `meeting-notes transcribe` - ASR transcription
- `meeting-notes diarize` - Speaker diarization
- `meeting-notes merge` - Transcript/diarization merge
- `meeting-notes summarize` - Summarization
- `meeting-notes render` - Minutes rendering
- `meeting-notes doctor` - Environment diagnostics
- `meeting-notes models` - Model management
- `meeting-notes config` - Configuration management
- `meeting-notes resources` - Resource estimates
- `meeting-notes benchmark` - Configuration comparison
- `meeting-notes naming` - Filename finalization
- `meeting-notes clean` - Artifact cleanup

#### Testing
- 160 unit and integration tests
- Tests cover: config, jobs, ASR, audio, transcript, summarization, minutes, naming, benchmarks, VAD

### Tested With
- Korean m4a files: 4.4MB (4:39), 28.3MB (30:17), 33.7MB (36:08)
- openai-whisper with large-v3-turbo model on CPU
- Docker whisper.cpp CPU build
