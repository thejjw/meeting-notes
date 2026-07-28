# Changelog

All notable changes to meeting-notes will be documented in this file.

## [Unreleased]

### Added
- `meeting-notes clean JOB_DIR --final-only` - preview and transactionally reduce a
  completed job to its finalized recording, Markdown meeting notes, and Markdown
  transcript; supports `--dry-run` and non-interactive `--yes`.
- `meeting-notes clarify template` / `clarify apply` - review AI-generated clarification
  questions (ASR mishearings, missing owners/dates, ambiguous terms) via an editable
  `clarifications.yaml` sidecar, then apply confirmed answers: correct the per-job
  glossary and transcript, re-summarize with the answers as authoritative context, and
  publish a new output generation.
- `meeting-notes glossary promote` - explicitly promote a job's glossary terms into the
  global glossary.
- Per-job glossary scoping (`<job_dir>/glossary.yaml`), layered over the global glossary
  at merge time, so corrections from one recording don't silently affect unrelated
  meetings.
- `user_clarifications` schema/prompt/render fields: `heard_text`, `user_answer`,
  `resolved`.
- `process` now prints an ASR/diarization time estimate before those stages run,
  derived from this machine's own timing history (`transcribe`/`diarize` stage
  durations recorded per completed job, matched by backend/device/model). Shows
  "no timing history yet" until a matching job has completed once.

### Changed
- Glossary matching (`transcript/glossary.py`) now supports a layered
  global-then-per-job lookup via `load_layered_glossary`/`merge_glossaries`.
- Diarization stage now records its backend/device/model identity in the
  manifest (`stages.diarize.runtime`), mirroring what transcription already
  recorded, so both stages can be used for time estimation.

### Removed
- `meeting-notes feedback` - replaced by `clarify template`/`clarify apply`. The
  previous command read from paths that don't exist in the real job layout
  (`<job>/minutes/meeting-notes.md`, `job_manifest.json`) and stored the misheard/
  corrected term pair backwards, so glossary corrections were silently inert.

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
