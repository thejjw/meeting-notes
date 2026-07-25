# meeting-notes

Local-first Korean/English meeting notes with Whisper transcription.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) must be
  installed and available on `PATH`. Verify it with `uv --version`. The project
  uses `uv` to install and run its required Python 3.12 environment.
- FFmpeg and FFprobe must be available. The configuration wizard and
  `meeting-notes doctor` report whether they were detected.
- The model-transfer scripts described below target Windows PowerShell. They do
  not require administrator privileges.

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

# One-time CPU setup (verified runtime + model)
uv run meeting-notes configure --accept-defaults --provision --yes

# Process a recording
uv run meeting-notes process "C:\Recordings\meeting.m4a"
```

That's it. The pipeline runs: inspect audio → normalize → transcribe → summarize → render minutes → finalize filenames.

`configure --accept-defaults` alone only writes configuration. Add `--provision` to
install assets. Large models require `--yes` in non-interactive use.

## Output Files

Each successful publication receives a generation directory. Its root stays
human-facing while machine-readable exports and operational metadata live in
named subdirectories:

```text
data/meetings/YYYY-MM-DD-<topic>-<hash>/output/finalized/<generation>/
├── 2026-07-22_topic-name.m4a
├── 2026-07-22_topic-name_meeting-notes.md
├── 2026-07-22_topic-name_transcript.md
├── json/
│   ├── 2026-07-22_topic-name_meeting-notes.json
│   └── 2026-07-22_topic-name_transcript.json
├── subtitles/
│   ├── 2026-07-22_topic-name_transcript.srt
│   └── 2026-07-22_topic-name_transcript.vtt
└── run/
    └── report.md
```

`run/report.md` records concise provenance and stage results. It deliberately
excludes credentials, environment values, prompts, transcript content, raw
subprocess output, and stack traces. Failed attempts retain only a compact
report under `output/runs/<run-id>/report.md`.

With `process --copy-to-input`, the same relative layout is copied beside the
original recording. Without that flag, the input directory is not modified.

| File | Description |
|------|-------------|
| `*_meeting-notes.md` | Main human-readable meeting minutes |
| `*_transcript.md` | Timestamped human-readable transcript |
| `json/*.json` | Structured summary and transcript |
| `subtitles/*` | SRT and WebVTT captions |
| `run/report.md` | Sanitized run provenance and concise status |
| `*.m4a`, `*.wav`, etc. | Byte-preserving finalized recording copy |

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

# Config only — safe CPU defaults
uv run meeting-notes configure --accept-defaults

# Config plus verified runtime and model
uv run meeting-notes configure --accept-defaults --provision --yes

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
  codex:
    model: gpt-5.6-terra         # null inherits the Codex CLI default
    reasoning_effort: null       # null inherits the Codex CLI default
  claude:
    model: sonnet                # floating alias; null inherits Claude settings
    effort: null                 # null inherits the Claude Code default
```

## ASR Backends

| Backend | Install | Notes |
|---------|---------|-------|
| `whisper_cpp` | Managed CPU binary or Vulkan source build | **Default** |
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

### Codex and Claude model selection

`meeting-notes` passes configured models through each provider's native
`--model` option. A null value omits that option and lets the provider CLI,
user settings, and built-in defaults choose.

For routine transcript summarization, the recommended economical presets are:

```yaml
summarization:
  backend: codex
  codex:
    model: gpt-5.6-terra
    reasoning_effort: null
```

or:

```yaml
summarization:
  backend: claude
  claude:
    model: sonnet
    effort: null
```

Use `gpt-5.6-sol` or `opus` for quality-first runs. Claude's `sonnet` and
`opus` names are provider-supported floating aliases. Codex model names should
use documented full IDs such as `gpt-5.6-terra`; bare `terra` is not a
documented Codex alias. Set Codex `reasoning_effort` explicitly when stable
behavior matters, or leave it null to inherit the provider default.
When Codex or Claude is selected, the interactive configuration wizard offers
provider default, low, medium, high, and a custom reasoning-effort value.

### Custom summarization agents

Claude-compatible PowerShell functions can be used without enabling an
implicit shell for every command. For example, an artificial `claude-alt`
function can configure a custom Claude Code endpoint:

```powershell
function claude-alt {
    $savedUrl = $env:ANTHROPIC_BASE_URL
    $savedToken = $env:ANTHROPIC_AUTH_TOKEN
    $savedModel = $env:ANTHROPIC_DEFAULT_SONNET_MODEL
    try {
        $env:ANTHROPIC_BASE_URL = "https://example.invalid/anthropic"
        $env:ANTHROPIC_AUTH_TOKEN = Get-MySecret "CLAUDE_ALT_TOKEN"
        $env:ANTHROPIC_DEFAULT_SONNET_MODEL = "custom-balanced-model"
        claude @args
    }
    finally {
        $env:ANTHROPIC_BASE_URL = $savedUrl
        $env:ANTHROPIC_AUTH_TOKEN = $savedToken
        $env:ANTHROPIC_DEFAULT_SONNET_MODEL = $savedModel
    }
}
```

Configure it as an explicit Claude launcher:

```yaml
summarization:
  backend: claude
  claude:
    executable: claude
    model: null
    launcher_execution: powershell
    launcher_command: claude-alt
```

`model: null` means meeting-notes supplies no `--model`; the function,
endpoint, or Claude settings choose the effective model. The function must be
available in the PowerShell profile loaded by `powershell.exe`. POSIX functions
can use `launcher_execution: posix_shell`.

For a non-Claude agent, use `local_command`. Its default
`request_json_v1` protocol writes one JSON object to stdin containing
`protocol_version`, `task`, `prompt`, `transcript`, `schema`, and `metadata`.
The command must write only the schema-conforming summary object to stdout:

```yaml
summarization:
  backend: local_command
  local_command:
    protocol: request_json_v1
    execution: direct
    command: ["my-meeting-agent", "--json"]
    environment:
      MY_AGENT_TOKEN: ${MY_AGENT_TOKEN}
```

Use `execution: powershell` with `script: my-agent-function` or
`execution: posix_shell` with a trusted shell expression. Existing integrations
that expect only transcript text on stdin can select
`protocol: transcript_stdin_v0`.

Test an adapter using a small request that does not publish meeting files:

```powershell
uv run meeting-notes summarizers test
```

## Data Footprint and Cleanup

meeting-notes keeps processing data local, but a complete run can accumulate
recording copies, normalized audio, model files, transcripts, and several
publication generations.

### Project and job data

Paths such as `data_dir`, `cache_dir`, and `model_cache_dir` are taken from the
active configuration. Relative paths are resolved from the directory where the
command is run. With the example defaults, the workspace contains:

| Path | Contents |
|------|----------|
| `./data/meetings/<job>/source/` | Optional copy of the original recording |
| `./data/meetings/<job>/audio/` | Normalized audio and chunks |
| `./data/meetings/<job>/asr/` | Raw ASR JSON, Markdown, and subtitles |
| `./data/meetings/<job>/diarization/` | Speaker turns and diarization artifacts |
| `./data/meetings/<job>/transcript/` | Anonymous and currently named transcripts |
| `./data/meetings/<job>/summary/` | Current structured summary |
| `./data/meetings/<job>/output/` | Current minutes, publication generations, and compact run reports |
| `./data/meetings/<job>/logs/` | Retained tool or build logs when produced |
| `./cache/` and `./cache/models/` | Project-configured caches and model files |

The job root also retains `manifest.json`, `speakers.yaml`, and speaker-template
backups. The manifest contains paths, timestamps, checksums, configuration
fingerprints, and publication provenance, but not API tokens.

Useful cleanup choices are:

```powershell
# Regenerate successfully, then remove superseded files recorded in the manifest.
uv run meeting-notes speakers apply JOB_DIR --map JOB_DIR/speakers.yaml --cleanup --yes

# After successful regeneration, also remove reproducible source copies,
# normalized audio, raw ASR, diarization artifacts, and logs.
uv run meeting-notes speakers apply JOB_DIR --map JOB_DIR/speakers.yaml --cleanup-all --yes

# Remove most derived data, including finalized output, while retaining source/.
uv run meeting-notes clean JOB_DIR --yes
```

For speakerless jobs, replace `--map ...` with `--without-diarization`. Review
the job directory before cleanup. Deleting an entire individual job directory
is the complete scrub for that meeting; never delete the shared `data/`
directory unless every contained job is intentionally disposable.

### Per-user application data

| Platform | Location | Contents |
|----------|----------|----------|
| Windows | `%APPDATA%\meeting-notes\config.yaml` | User configuration |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/meeting-notes/config.yaml` | User configuration |
| Windows | `%LOCALAPPDATA%\meeting-notes\cache\` | Managed whisper.cpp runtimes, downloads, build logs, and managed diarization models |
| Linux | `${XDG_CACHE_HOME:-~/.cache}/meeting-notes/` | The same managed cache |
| All | `~/.cache/silero-vad/` | Silero VAD model when that backend downloads it |

An explicit `--config`, `MEETING_NOTES_CONFIG`, or project-local
`meeting-notes.yaml` may select a different configuration file. Inspect
`meeting-notes config status` and `meeting-notes config show --resolved` before
scrubbing.

Managed cache directories can be removed after meeting-notes processes have
stopped; required runtimes and models will need to be downloaded or rebuilt
again. Removing the active configuration does not remove jobs or caches, and
removing jobs does not remove the configuration or models.

### Provider and development data

Hugging Face login credentials and its download cache are managed by
`huggingface_hub`, normally under `${HF_HOME:-~/.cache/huggingface}`. Claude,
Codex, custom launchers, OS credential managers, Git, `uv`, and Python tooling
may maintain their own configuration, authentication, caches, virtual
environments, and history outside the paths above. meeting-notes does not copy
those credentials into jobs or run reports. Use each provider's logout or
credential-management command when credentials must also be scrubbed.

Development commands may additionally create `.venv/`, `.pytest_cache/`,
`.ruff_cache/`, and tool-specific caches in or outside the repository. These
are not meeting records and may be removed when no development process is
using them.

## Speaker Diarization

Diarization is optional and uses the local
[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
pipeline. Install the optional dependencies, then run the guided setup:

```powershell
uv sync --extra diarization
uv run meeting-notes diarization setup
uv run meeting-notes doctor
```

The setup command uses Hugging Face's browser device-login flow, opens the
Community-1 conditions page when approval is still needed, downloads the model
to the managed cache, and writes its local path to configuration. Users must
personally accept gated-model conditions in the browser; the application cannot
accept them on their behalf. No manual `HF_TOKEN` environment variable is
required.

After `doctor` reports diarization as ready, resume an existing transcription
without rerunning ASR:

```powershell
uv run meeting-notes process "<audio-file>" --from diarize
```

An already-downloaded pyannote pipeline can instead be selected with
`diarization.model_path`; in that mode the application does not require
`HF_TOKEN`. Obtaining the official Community-1 files initially still requires
accepting the model's conditions.

## Transferring Models Between Windows Computers

The Windows transfer scripts create portable ZIP64 archives for large or gated
models. Archives contain model files and sanitized provenance only. They never
contain Hugging Face credentials, meeting data, user configuration, FFmpeg,
Python environments, or whisper.cpp runtime executables.

On the source computer, back up the configured Whisper model and diarization
pipeline:

```powershell
.\scripts\transfer-whisper-model-windows.ps1 -Action Backup
.\scripts\transfer-diarization-model-windows.ps1 -Action Backup
```

Use `-Model large-v3-turbo` to select a different installed managed Whisper
model, `-Archive C:\Backups\model.zip` to choose the output, or
`-CompressionLevel Fastest` for a faster but larger archive. Each backup writes
both a `.zip` and adjacent `.zip.sha256` file. Carry both files to the new
computer.

On the destination computer, install `uv`, clone the project, create or copy a
valid meeting-notes configuration, and restore:

```powershell
uv sync --extra diarization
.\scripts\transfer-whisper-model-windows.ps1 -Action Restore -Archive D:\Transfer\meeting-notes-whisper-model.zip
.\scripts\transfer-diarization-model-windows.ps1 -Action Restore -Archive D:\Transfer\meeting-notes-diarization-model.zip
uv run meeting-notes doctor
```

Pass `-Config PATH` when the configuration is not at the normal per-user
location. Restore refuses to replace an installed destination unless `-Force`
is supplied. The Whisper model archive does not include whisper.cpp itself; use
`meeting-notes runtime install` or the configuration wizard to provision the
appropriate CPU or GPU runtime separately.

Checksums detect transfer corruption but are not signatures. Only restore
archives obtained from a trusted source. Hugging Face gated-model conditions
and all other model licenses continue to apply when archives are copied.

## whisper.cpp Setup

CPU installs use SHA-256-verified upstream release archives. Vulkan installs are
native source builds because upstream does not publish Windows/Linux Vulkan binaries.
Both paths pin whisper.cpp `v1.9.1`.

```bash
uv run meeting-notes runtime install --device cpu --config meeting-notes.yaml
uv run meeting-notes runtime install --device vulkan --config meeting-notes.yaml
uv run meeting-notes models download large-v3 --yes --config meeting-notes.yaml
uv run meeting-notes doctor --config meeting-notes.yaml
uv run meeting-notes doctor --config meeting-notes.yaml --smoke-test
```

Windows Vulkan requires Git, CMake, Visual Studio C++ Build Tools, and Vulkan SDK
tooling. Linux requires Git, CMake, a C++ toolchain, and Vulkan development packages.
The app diagnoses these prerequisites but does not install system-wide developer tools.

Managed assets live under `%LOCALAPPDATA%\meeting-notes\cache` on Windows and
`${XDG_CACHE_HOME:-~/.cache}/meeting-notes` on Linux. Failed downloads do not replace
verified assets; Vulkan build logs are retained beside the runtime directory.

**Build scripts and Docker**

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
