# meeting-notes

Local-first Korean/English meeting notes with Whisper transcription.

## What It Does

Takes a recorded meeting audio/video file and produces:
1. **Transcript** — timestamped Korean/English text with segment IDs
2. **Meeting Notes** — structured Markdown with summary, decisions, action items
3. **Summary JSON** — machine-readable meeting data
4. **Finalized Recording** — copy with date/topic filename

## Quick Start

```bash
# Install
uv sync --group dev

# One-time setup (downloads whisper.cpp binary automatically)
uv run meeting-notes configure --accept-defaults

# Process a recording
uv run meeting-notes process "C:\Recordings\meeting.m4a"
```

That's it. The pipeline runs: inspect audio → normalize → transcribe → summarize → render minutes → finalize filenames.

### First Run

On first run, the setup wizard downloads the whisper.cpp binary for your platform and configures CPU-only transcription. No Docker or compilation needed.

## Output Files

After processing, output files appear in **two places**:

### 1. Job Directory (internal)

```
data/meetings/YYYY-MM-DD-<topic>-<hash>/
├── source/original.m4a          # Copy of recording
├── audio/normalized.wav         # Processed audio
├── asr/transcript.raw.json      # Full timestamped transcript
├── asr/transcript.srt           # Subtitle file
├── asr/transcript.vtt           # WebVTT subtitle
├── transcript/transcript.merged.json  # Merged transcript
├── summary/summary.json         # AI-generated summary
├── output/minutes.md            # Meeting minutes
└── output/finalized/            # ← FINAL OUTPUT FILES HERE
    ├── 2026-07-22_topic-name.m4a
    ├── 2026-07-22_topic-name_meeting-notes.md
    └── 2026-07-22_topic-name_meeting-notes.json
```

### 2. Input Directory (finalized copies)

The finalized files are also copied next to the original recording:

```
C:\Recordings\
├── meeting.m4a                              # Original (unchanged)
├── 2026-07-22_topic-name.m4a                # Finalized recording copy
├── 2026-07-22_topic-name_meeting-notes.md   # ← THE MEETING MINUTES
└── 2026-07-22_topic-name_meeting-notes.json # ← STRUCTURED SUMMARY
```

### What Each File Contains

| File | Description |
|------|-------------|
| `*_meeting-notes.md` | **Main output** — formatted meeting minutes with summary, discussion topics, decisions, action items, evidence links |
| `*_meeting-notes.json` | Structured JSON with all meeting data for programmatic use |
| `*.m4a` | Finalized recording copy with date/topic filename |

### Example Output

```markdown
# 2026-07-22 에이전트 설치 및 시범 운영 일정 협의

- 원본 녹음: `recording.m4a`
- 참석자: 홍길동, 김철수, 이영희

## 핵심 요약
- 에이전트 설치 초기에는 현장 참여 지원이 필요하다
- 8월 시범 운영, 9월 본 운영 계획

## 논의 사항
### 설치 초기 지원과 전담자 필요성 (00:00 ~ 01:34)
- 에이전트 사용법 가이드 제공이 필요
- 원격 접속 불가로 현장 방문 필요

## 확정 사항
1. 현장 대응 담당자는 이영희로 배정 (provisional)

## 후속 조치
| 담당자 | 작업 | 기한 | 근거 |
|---|---|---|---|
| 미정 | 이영희를 현장 대응 담당자로 배정 | 미정 | [seg-000051](...) |
```

## Commands

| Command | What It Does |
|---------|--------------|
| `meeting-notes configure` | Interactive setup wizard |
| `meeting-notes process FILE` | **Main command** — full pipeline |
| `meeting-notes process FILE --dry-run` | Show plan without running |
| `meeting-notes process FILE --from summarize` | Resume from specific stage |
| `meeting-notes process FILE --force-stage transcribe` | Re-run a specific stage |
| `meeting-notes doctor` | Check environment and tools |
| `meeting-notes config show` | Show current configuration |
| `meeting-notes models list` | List available Whisper models |
| `meeting-notes resources show` | Show memory estimates |

## Configuration

```bash
# First run — interactive wizard
uv run meeting-notes configure

# Quick setup — safe CPU defaults
uv run meeting-notes configure --accept-defaults

# Edit config manually
# Windows: %APPDATA%\meeting-notes\config.yaml
# Linux:   ~/.config/meeting-notes/config.yaml
```

### Config Summary

```yaml
runtime:
  device: cpu                    # cpu, vulkan, rocm, cuda
  asr_backend: whisper_cpp       # whisper_cpp, openai_whisper, faster_whisper

asr:
  model: large-v3-turbo          # tiny, base, small, medium, large-v3, large-v3-turbo
  language: ko                   # ko, en, auto

diarization:
  enabled: false                 # requires pyannote + HF token

summarization:
  enabled: true
  backend: codex                 # codex, opencode, mimo, claude, local_command
```

## ASR Backends

| Backend | Install | Notes |
|---------|---------|-------|
| `whisper_cpp` | Pre-built binary (auto-downloaded) | **Default**, works out of the box |
| `openai_whisper` | `uv pip install openai-whisper` | Python/PyTorch, opt-in |
| `faster_whisper` | `uv pip install faster-whisper` | CPU/NVIDIA CUDA, opt-in |

The default `whisper_cpp` backend uses a pre-built Windows binary from [whisper.cpp releases](https://github.com/ggml-org/whisper.cpp/releases). No Docker or compilation needed for the common case.

## Summarization Backends

| Backend | Config | Notes |
|---------|--------|-------|
| `codex` | `backend: codex` | OpenAI Codex CLI |
| `opencode` | `backend: opencode` | OpenCode CLI |
| `mimo` | `backend: mimo` | Mimo Code CLI |
| `claude` | `backend: claude` | Claude Code CLI |
| `local_command` | `backend: local_command` | Any custom CLI |

Set `summarization.enabled: false` to skip summarization (transcript-only mode).

## whisper.cpp Setup

**Default: Pre-built binary** (recommended for most users)

The first run of `meeting-notes configure --accept-defaults` downloads the appropriate whisper.cpp binary for your platform. No compilation needed.

**Alternative: Build from source**

If you need a custom build (e.g., for specific GPU support):

```bash
# Using Docker (builds inside container, copies binary out)
docker build -t whisper-builder docker/whisper-cpp
docker run --rm -v "$(pwd)/bin:/out" whisper-builder

# Or native build scripts
scripts/build-whisper-cpp.sh    # Linux/macOS
scripts/build-whisper-cpp.ps1   # Windows
```

**Alternative: openai-whisper (Python)**

```bash
uv pip install openai-whisper
# Then set asr_backend: openai_whisper in config
```

## License

zlib-acknowledgement — see [LICENSE](LICENSE)
