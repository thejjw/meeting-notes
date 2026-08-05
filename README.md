# meeting-notes

Local-first Korean/English meeting transcription and notes.

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

Before transcription and diarization start, `process` prints an estimated wall-clock
time for each, based on this machine's own history of prior runs with the same
backend/device/model (median real-time factor from completed jobs in `data/meetings/`).
The first run with a given configuration has no history to draw on, so it prints
"no timing history yet" instead of a guess; every completed run afterward improves
the estimate for next time.

## Complete Reviewed Workflow

The quick-start command produces usable automated output. For a reviewed result,
use the following workflow to identify speakers, resolve the AI's open questions,
and retain only the final human-facing files.

### 1. Process the recording

This example uses a Korean filename with spaces:

```powershell
$recording = 'C:\Users\user\Downloads\음성 19800101_1.m4a'
uv run meeting-notes process "$recording"
```

At completion, the command prints a line like:

```text
Pipeline complete. Job: data\meetings\2026-07-28--807b1c9b1dac39f9
```

Copy the exact printed path into `$job`, resolving it from the directory where
you ran `meeting-notes` when it is relative:

```powershell
$job = 'C:\path\to\meeting-notes\data\meetings\2026-07-28--807b1c9b1dac39f9'
```

Do not guess the job directory from the recording name. The directory name also
depends on recording metadata and a content-derived hash.

### 2. Correct speaker names

Open `$job\speakers.yaml`, identify each speaker from the included timestamps and
example utterances, and fill in the `name` values. Then apply the mapping:

```powershell
uv run meeting-notes speakers apply "$job" --map "$job\speakers.yaml"
```

Review the newly published `*_meeting-notes.md` and `*_transcript.md` under
`$job\output\finalized\<generation>\`. If a name is wrong, edit the YAML and
apply it again. See [Correct Speaker Names](#correct-speaker-names) for the YAML
format and diarization requirements.

### 3. Resolve the initial clarifications

Create the first answer sheet:

```powershell
uv run meeting-notes clarify template "$job"
```

Open `$job\clarifications.yaml`, fill in the relevant `answer:` fields or add
general guidance under `comments:`, and publish the corrections:

```powershell
uv run meeting-notes clarify apply "$job"
```

Review the newly published meeting notes and transcript again.

### 4. Repeat clarification review until clear

An apply can correct the transcript and produce a new summary with new or
remaining questions. Regenerate the answer sheet against the current transcript:

```powershell
uv run meeting-notes clarify template "$job" --force
```

`--force` backs up the previous sidecar and preserves existing answers and
comments. If the command reports `No open clarifications found for this job.`,
the loop is complete. Otherwise, review `$job\clarifications.yaml`, fill in the
new or remaining answers, and apply again:

```powershell
uv run meeting-notes clarify apply "$job"
```

Repeat the regenerate, review, and apply cycle until there are no open questions
and the latest meeting notes and transcript pass human review. See
[User Feedback & Terminology Refinement](#user-feedback--terminology-refinement)
for the clarification fields and correction behavior.

### 5. Retain only the final files

Final-only cleanup is irreversible because it removes the manifest, editable
sidecars, JSON, subtitles, and all regeneration artifacts. Preview the exact
selection first:

```powershell
uv run meeting-notes clean "$job" --final-only --dry-run
```

After verifying that the preview retains the correct recording, meeting notes,
and transcript, perform the cleanup:

```powershell
uv run meeting-notes clean "$job" --final-only --yes
```

The job directory now contains exactly those three files. See
[Data Footprint and Cleanup](#data-footprint-and-cleanup) for other cleanup
choices.

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

### High-Level Workflow
| Command | Description |
|---------|-------------|
| `meeting-notes configure` | Interactive setup wizard |
| `meeting-notes process FILE` | **Main command** — full pipeline execution |
| `meeting-notes process FILE --dry-run` | Show execution plan without running |
| `meeting-notes process FILE --from STAGE` | Resume pipeline execution from a specific stage |
| `meeting-notes process FILE --force-stage STAGE` | Re-run a specific stage |
| `meeting-notes process FILE --num-speakers N` | Use a known exact speaker count for this run |
| `meeting-notes process FILE --min-speakers N --max-speakers N` | Bound automatic speaker counting for this run |
| `meeting-notes doctor` | Check environment, dependencies, and tool integrity |

### Job Review & Refinement
| Command | Description |
|---------|-------------|
| `meeting-notes speakers template JOB_DIR` | Create or locate editable `speakers.yaml` map |
| `meeting-notes speakers apply JOB_DIR` | Apply speaker names, re-summarize, and publish |
| `meeting-notes clarify template JOB_DIR` | Create editable `clarifications.yaml` from open AI questions |
| `meeting-notes clarify apply JOB_DIR` | Apply answers: correct glossary & transcript, re-summarize, publish |
| `meeting-notes glossary promote JOB_DIR` | Promote job glossary terms into global `config/glossary.yaml` |
| `meeting-notes clean JOB_DIR --final-only` | Transactionally retain only final recording, notes, and transcript |

### Management & Diagnostics
| Command | Description |
|---------|-------------|
| `meeting-notes config [show/status/edit/reset]` | Display, inspect, edit, or reset application configuration |
| `meeting-notes models [list/status/info/download/verify]` | List, check, download, or verify Whisper model integrity |
| `meeting-notes runtime [status/install]` | Check runtime status or install whisper.cpp CPU/Vulkan binaries |
| `meeting-notes cache [status/migrate]` | Inspect project storage or transactionally migrate legacy Whisper assets |
| `meeting-notes asr setup/status` | Provision or inspect GPU Qwen3-ASR plus project-local alignment |
| `meeting-notes diarization setup` | Provision Community-1 from Hugging Face or a portable backup, with optional ROCm acceleration |
| `meeting-notes diarization status` | Show model/runtime readiness and total project-local storage |
| `meeting-notes diarization remove-runtime` | Remove the project-local ROCm runtime and return diarization to CPU |
| `meeting-notes naming [preview/finalize]` | Preview or finalize output file naming |
| `meeting-notes resources show` | Display memory and resource allocation estimates |
| `meeting-notes summarizers test` | Test summarizer adapter configuration with a test prompt |

## Correct Speaker Names

Speaker names can be corrected when diarization is enabled and produced stable
speaker IDs such as `SPEAKER_00`. The automated pipeline normally creates
`JOB_DIR\speakers.yaml` after merging the diarization results. This command
creates or locates it explicitly:

```powershell
uv run meeting-notes speakers template "$job"
```

If the command reports that no stable diarization speaker labels were found,
there are no speaker IDs to map. Configure diarization and rerun from that stage,
as described in [Speaker Diarization](#speaker-diarization), or publish without
speaker attribution:

```powershell
uv run meeting-notes speakers apply "$job" --without-diarization
```

For a diarized job, each entry in `speakers.yaml` includes timestamps, segment
IDs, example utterances, and speaking-time statistics. Use that evidence to
identify the person, then edit only the `name` value:

```yaml
speakers:
  SPEAKER_00:
    name: '김민수'
    segment_count: 58
    total_seconds: 134.5
    examples:
      - timestamp: 00:38:20
        segment_id: seg-001053
        text: 이 연동에 대한 스펙을 잡아놓고 있는데
  SPEAKER_01:
    name: ''
    segment_count: 12
    total_seconds: 22.2
    examples:
      - timestamp: 00:00:23
        segment_id: seg-000007
        text: 설계서도 한번 주신 거 있잖아요.
```

Leave an unknown speaker's name empty. Do not change the `SPEAKER_*` keys,
transcript fingerprint, examples, segment counts, or timestamps. Save the file
as UTF-8, then apply it:

```powershell
uv run meeting-notes speakers apply "$job" --map "$job\speakers.yaml"
```

This does not rerun transcription or diarization. It reuses the merged transcript,
renders named transcript variants, re-summarizes with the mapped participants,
and publishes a new generation under `output\finalized\`. Previous generations
remain available for comparison. To correct another name, edit the same sidecar,
apply again, and review the newest generation before starting clarification
review.

## User Feedback & Terminology Refinement

Automatic Speech Recognition (ASR) may mishear technical terms or leave action item assignees unspecified. Generated meeting notes include a `## 사용자 확인 및 정정` section listing the AI's open questions. Answers are entered in a separate `clarifications.yaml` sidecar file, not in the notes themselves — the notes are regenerated on every apply, so anything typed directly into them would be overwritten.

### Workflow

1. **Create the answer sheet**:
   ```bash
   uv run meeting-notes clarify template JOB_DIR
   ```
   This writes `JOB_DIR/clarifications.yaml` with one entry per open question, plus a `comments:` section, e.g.:
   ```yaml
   clarifications:
     clarif-000:
       category: asr_correction
       question: '"아르고 시디"로 전사됨. "ArgoCD"가 맞나요?'
       heard_text: 아르고 시디
       suggested_correction: ArgoCD
       evidence: [seg-000012]
       answer: ''
     clarif-001:
       category: missing_info
       question: "OAuth 전환 작업" 담당자가 미정입니다.
       evidence: [seg-000045]
       answer: ''
   comments:
     - ''
   ```

2. **Fill in the `answer:` fields** and apply:
   ```bash
   uv run meeting-notes clarify apply JOB_DIR
   ```

   For general guidance that isn't tied to one flagged item — a hint, a preference, a "when unsure, assume X" — add free-text lines under `comments:` instead (duplicate the entry to add more). Unlike an `answer:`, a comment is never used for exact-match glossary substitution; it's passed to the re-summarization model as steering context only. Comments are preserved across `clarify template --force` regenerations, same as answers.

3. **Review the new publication and repeat until clear**:
   ```bash
   uv run meeting-notes clarify template JOB_DIR --force
   ```
   Applying answers may correct the transcript and create a new summary with new
   or remaining questions. The next template must be regenerated against that
   current transcript. `--force` backs up the previous `clarifications.yaml` and
   preserves its existing answers and comments. If there are still open
   questions, review the regenerated sidecar and apply it again:
   ```bash
   uv run meeting-notes clarify apply JOB_DIR
   ```
   Stop when `clarify template --force` reports that no open clarifications were
   found and the latest meeting notes and transcript pass human review.

### What Happens:
- **Job-scoped glossary**: `asr_correction`/`term_clarification` answers (e.g. `아르고 시디` -> `ArgoCD`) are saved to `JOB_DIR/glossary.yaml`, scoped to this recording only — not the shared global glossary. `missing_info` answers (owners, dates) are never added to any glossary. `comments:` entries are never added to any glossary.
- **Transcript correction**: the job glossary, layered over the global one, is re-applied to `transcript/transcript.merged.json` and its exported `.md`/`.srt`/`.vtt` variants. Only literal (non-inflected) occurrences of the misheard text are substituted; Korean particle-suffixed occurrences may remain in the raw transcript even after this step.
- **Re-summarization**: the AI summarizer is re-run with the confirmed answers, plus any `comments:`, passed as authoritative/steering context, so the regenerated summary reflects correct terminology even where the literal substitution above didn't apply.
- **New generation published**: results land in a new `output/finalized/<generation>/` directory; the previous generation is marked superseded in `manifest.json`, never overwritten in place.
- **Global promotion (optional)**: once a term proves useful across meetings, promote it explicitly:
  ```bash
  uv run meeting-notes glossary promote JOB_DIR
  ```
  This is the only way terms move from a job glossary into `config/glossary.yaml`, where they'll apply to every future recording.

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
  device: cpu                    # cpu baseline; npu/vulkan for Lemonade Whisper
  asr_backend: whisper_cpp       # also lemonade, qwen3_asr_lemonade, faster_whisper

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

### Language configuration

`asr.language` is shared by every transcription backend. Each backend receives
one dominant-language hint for each transcription chunk; there is no supported
two-language value such as `ko,en` or `[ko, en]`. For the Korean/English
technical meetings this project primarily targets:

```yaml
asr:
  language: ko
```

Use `ko` when the conversation is Korean-dominant but includes English product
names, technical terms, or ordinary code-switching. Both multilingual Whisper
and Qwen can still emit English text; `ko` does not mean that every recognized
token must be Korean. Use `en` for English-dominant dialogue.

Use `auto` when the dominant language is genuinely unknown:

```yaml
asr:
  language: auto
```

`auto` is not sentence- or speaker-level bilingual detection. The active engine
detects one dominant language independently for each ASR chunk. It is useful
when the language is unknown or when substantial Korean-heavy and English-heavy
sections fall into different chunks. It is usually not better for an ordinary
Korean meeting containing English terminology, and rapid sentence-by-sentence
switching remains a model limitation. In that common case, select the dominant
language explicitly (`ko` or `en`). Qwen also passes the one detected language
to its forced aligner, so mixed-language timestamps can be imperfect even when
the transcript contains both languages.

The managed `tiny`, `base`, `small`, `medium`, `large-v3`, and
`large-v3-turbo` files are multilingual models rather than English-only `.en`
variants. Other supported examples include `zh` (Chinese), `ja` (Japanese),
`de` (German), `fr` (French), `es` (Spanish), `pt` (Portuguese), and `vi`
(Vietnamese). See OpenAI's
[authoritative language-code list](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
for the complete set. Other languages are available on a best-effort basis;
project maintenance and fixtures primarily exercise Korean, English, and
mixed Korean/English technical dialogue, and Whisper quality varies by
language.

### Whisper model choice

`large-v3-turbo` is the recommended default for transcription. OpenAI describes
it as substantially faster than the large model with minimal accuracy loss,
while its whisper.cpp disk and memory footprint is approximately the same as
`medium`. OpenAI's evaluation also shows that multilingual recognition
generally improves as model size increases:
[Whisper README](https://github.com/openai/whisper/blob/main/README.md) and
[Whisper paper](https://cdn.openai.com/papers/whisper.pdf).

- Use `large-v3-turbo` for the normal Korean/English transcription workflow.
- Use `small` when download size or available memory is constrained, accepting
  a larger multilingual accuracy tradeoff.
- Use `large-v3` when quality matters more than memory and runtime.
- Use `medium` or `large-v3` for `task: translate`. Turbo is optimized for
  transcription and does not translate non-English speech into English.

## ASR Backends

| Backend | Install | Notes |
|---------|---------|-------|
| `whisper_cpp` | Managed CPU binary or Vulkan source build | **Default** |
| `lemonade` | Running AMD Lemonade Server | Opt-in Vulkan or NPU acceleration |
| `qwen3_asr_lemonade` | Lemonade Vulkan plus managed ROCm aligner | Experimental AMD GPU, opt-in |
| `openai_whisper` | `uv pip install openai-whisper` | Python/PyTorch, opt-in |
| `faster_whisper` | `uv pip install faster-whisper` | CPU/NVIDIA CUDA, opt-in |

The default `whisper_cpp` backend uses a pre-built Windows binary from [whisper.cpp releases](https://github.com/ggml-org/whisper.cpp/releases). No Docker or compilation needed for the common case.

### AMD Lemonade Whisper acceleration

Run `meeting-notes configure` and select **AMD Lemonade Vulkan** or
**AMD Lemonade NPU** to opt in. Vulkan is the recommended AMD GPU route when it
is available; NPU remains useful when GPU resources need to stay free. The wizard
supplies the standard server URL, `http://127.0.0.1:13305`; press Enter unless
your server uses a custom address.

Start Lemonade Server yourself before configuration or processing:

```powershell
lemonade status
lemonade backends install whispercpp:vulkan  # or whispercpp:npu
uv run meeting-notes configure
```

Once the server is reachable, the wizard can download, install, and load
`Whisper-Large-v3-Turbo` through Lemonade. Install the corresponding
`whispercpp:vulkan` or `whispercpp:npu` backend in Lemonade first. Large model
downloads require explicit confirmation. Meeting-notes does not start or stop
Lemonade, and it fails clearly rather than silently switching accelerators if
the server, model, or selected backend is unavailable.

Lemonade reports Vulkan as the generic device `gpu`; meeting-notes additionally
checks `recipe_options.whispercpp_backend` so that Vulkan and any other GPU
backend cannot be confused. ROCm is intentionally not offered for Lemonade
transcription. The portable slow baseline remains the default local
`whisper_cpp` backend with `runtime.device: cpu`.

On this project's Ryzen AI Max+ 395 validation machine, a repeated 240-second
Korean clip took about 13.2 seconds with Vulkan (18.2x real time) and 30.3 seconds
with NPU (7.9x real time). Treat those as machine-specific speed measurements,
not an accuracy guarantee; the generated transcripts differed and should be
compared on representative recordings.

Useful verification and provisioning commands:

```powershell
uv run meeting-notes doctor --smoke-test
uv run meeting-notes models download large-v3-turbo --backend lemonade --yes
```

The Lemonade adapter uses timestamped `verbose_json` responses, so subtitles,
speaker alignment, and evidence timestamps continue to work.

Long normalized WAV files are split before upload when they exceed the
`asr.backend_options.lemonade.max_upload_mib` server ceiling (100 MiB by
default). Meeting-notes reserves 10% for WAV headers, multipart overhead, and
chunk overlap, giving the default an effective audio budget of about 90 MiB.
Responses are shifted back to absolute recording time, overlap is removed at
chunk boundaries, and the pipeline writes one continuous transcript. Values
above 100 MiB require a Lemonade Server with a correspondingly increased
request limit and may otherwise produce HTTP 413 errors.

### Qwen3-ASR 1.7B GPU alternative

`qwen3_asr_lemonade` is an opt-in AMD GPU backend. Lemonade runs the fixed
`unslothai/Qwen3-ASR-1.7B-GGUF:Q8_0` checkpoint through llama.cpp Vulkan. The
project-managed ROCm Python environment then runs
`Qwen/Qwen3-ForcedAligner-0.6B-hf` to produce timestamps for speaker assignment
and subtitles. Qwen does not perform diarization; pyannote remains the following
pipeline stage.

There is no CPU mode or silent fallback. Install AMD HIP and start Lemonade Server,
then run:

```powershell
uv run meeting-notes asr setup --backend qwen3_asr_lemonade
uv run meeting-notes asr status
```

Setup reuses a matching Lemonade Model Manager download or pulls it automatically,
loads it with `llamacpp_backend: vulkan`, provisions the aligner under the project
cache, and exercises both components. It reports approximately 2.35 GiB of
Lemonade-owned GGUF storage, 1.72 GiB of project-local aligner storage, and any
temporary ROCm staging requirement before proceeding. Confirmation defaults to
yes. Add `--activate` to select Qwen; otherwise Whisper/NPU remains active.

The removed native `qwen3_asr` backend is not accepted as an alias. Successful
setup removes only its obsolete project-local 1.7B Transformers weights, reclaiming
approximately 3.81 GiB, while retaining the forced aligner and shared ROCm runtime.

Vulkan is the only supported Lemonade backend for Qwen transcription because it
matches Lemonade's AMD preference and passed the project's local 30-second and
four-minute ASR checks. The forced aligner and optional pyannote acceleration
still use the project-local Python ROCm runtime. `runtime.device: rocm` therefore
identifies that mandatory Python runtime; it does not select a Lemonade ROCm
backend.

Qwen language behavior is independent of the configured Korean default. Set an
explicit dominant language using either its code or English name, for example:

```yaml
asr:
  language: ja  # Japanese
```

Use `language: auto` to let Qwen detect one language independently for every ASR
chunk. This is not sentence-level or speaker-level language detection, so rapid
code-switching within a chunk remains a model limitation. Qwen3-ASR recognizes
30 languages, but its forced aligner currently timestamps only `zh`, `en`, `yue`,
`fr`, `de`, `it`, `ja`, `ko`, `pt`, `ru`, and `es`. Meeting-notes accepts those
11 explicit languages because timestamps are required for subtitles and speaker
assignment. In auto mode, an unsupported detected language produces an actionable
error and suggests using Whisper instead of silently returning untimestamped text.

To benchmark Whisper/NPU and Qwen/GPU on the first ten minutes of a recording:

```powershell
uv run meeting-notes benchmark "meeting.m4a" `
  --matrix config/qwen3-asr-benchmark.example.yaml `
  --start-seconds 0 --duration-seconds 600
```

The benchmark records cold load and end-to-end timing, RTF, memory data, and a
separate transcript artifact for every run. Qwen alignment chunks are capped at
four minutes, below the forced aligner's five-minute limit. Aligned words, rather
than whole segments, are assigned at overlap boundaries so chunk merging does not
repeat speech.

The current Ryzen AI Max+ 395 / Radeon 8060S environment produced these results:

| Backend/device | Time | RTF | Speed | Observed peak memory |
|---|---:|---:|---:|---:|
| Lemonade Whisper/NPU | 65.3 s | 0.109 | 9.18x real time | server memory not included |
| Qwen first-ever GPU run | 101.4 s / 240 s | 0.422 | 2.37x real time | includes ROCm kernel warm-up |
| Qwen subsequent fresh worker | 29.8 s / 240 s | 0.124 | 8.04x real time | 5.1 GiB alignment allocation |

The first forced-alignment run spent most of its time warming ROCm kernels. A later
fresh worker transcribed in 15.3 seconds and reported 6.2 seconds of alignment,
with process startup and synchronization accounting for the remainder. The sample
suggested fewer Whisper-style repetitions, but it has no reference transcript, so
compare retained artifacts before drawing quality conclusions.

## Summarization Backends

| Backend | Config | Notes |
|---------|--------|-------|
| `codex` | `backend: codex` | OpenAI Codex CLI |
| `opencode` | `backend: opencode` | OpenCode CLI |
| `mimo` | `backend: mimo` | Mimo Code CLI |
| `claude` | `backend: claude` | Claude Code CLI |
| `local_command` | `backend: local_command` | Any custom CLI |
| `lemonade` | `backend: lemonade` | Opt-in local best-effort Markdown |

Set `summarization.enabled: false` to skip summarization (transcript-only mode).

### AMD Lemonade local summarization

The setup wizard offers Lemonade summarization independently from the ASR
backend. It defaults to `http://127.0.0.1:13305` and
`Gemma-4-26B-A4B-it-MTP-GGUF`. Start Lemonade Server manually; when it is
reachable, provisioning may download and load the selected model after the
normal large-download confirmation.

```yaml
summarization:
  enabled: true
  backend: lemonade
  lemonade:
    base_url: http://127.0.0.1:13305
    model_id: Gemma-4-26B-A4B-it-MTP-GGUF
    prompt_path: ./prompts/meeting-summary-local.md
    request_timeout_seconds: 7200
    max_completion_tokens: null
```

Lemonade uses its own concise prompt and returns relaxed Markdown rather than
the production JSON schema. It does not request segment quotations or evidence
IDs. Every local summary is marked **Local AI — best-effort summary** because
local generation may be slower and less correct than Codex or another
production-grade summarizer. Clarification regeneration and speaker-driven
summary republishing require structured JSON and are unavailable for this
output format.

With `max_completion_tokens: null`, meeting-notes omits `max_tokens` and lets
the model stop naturally. The model's total context window still includes the
prompt, transcript, reasoning, and visible answer. Set a positive integer only
when an explicit operational cap is wanted; the request timeout remains the
runaway-generation safeguard.

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

The `claude` backend invokes `claude -p --output-format json --permission-mode dontAsk`,
so the summarization call has no filesystem/Bash tool access beyond read-only
commands — it only reads the transcript from stdin and returns JSON. It does
**not** pass `--bare`, so your normal `~/.claude` OAuth login, hooks, and MCP
servers still apply; if you rely on `claude login` rather than
`ANTHROPIC_API_KEY`, this keeps working unchanged.

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

Paths such as `data_dir` and `cache_dir` are taken from the active configuration.
`project.cache_dir` is the authority for all meeting-notes-managed models and
runtimes. Relative paths are resolved from the directory where the command is run;
an absolute cache path is recommended when the config file is outside the project.
With the example defaults, the workspace contains:

| Path | Contents |
|------|----------|
| `./data/meetings/<job>/source/` | Optional copy of the original recording |
| `./data/meetings/<job>/audio/` | Normalized audio and chunks |
| `./data/meetings/<job>/asr/` | Raw ASR JSON, Markdown, and subtitles |
| `./data/meetings/<job>/diarization/` | Speaker turns and diarization artifacts |
| `./data/meetings/<job>/transcript/` | Anonymous and currently named transcripts |
| `./data/meetings/<job>/summary/` | Current structured JSON or local Markdown summary |
| `./data/meetings/<job>/output/` | Current minutes, publication generations, and compact run reports |
| `./data/meetings/<job>/logs/` | Retained tool or build logs when produced |
| `./cache/models/` | Managed Whisper and Qwen forced-aligner model files |
| `./cache/runtimes/` | Managed whisper.cpp and shared ROCm Python runtimes/build logs |
| `./cache/diarization/models/` | Project-local Community-1 model snapshots |

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

# Preview a safe final cleanup that keeps exactly the published recording,
# Markdown meeting notes, and Markdown transcript in the job root.
uv run meeting-notes clean JOB_DIR --final-only --dry-run

# Perform that final cleanup after showing the same preview and asking for confirmation.
uv run meeting-notes clean JOB_DIR --final-only

# Remove most derived data, including finalized output, while retaining source/.
uv run meeting-notes clean JOB_DIR --yes
```

Run final-only cleanup only after completing the
[speaker and clarification review workflow](#complete-reviewed-workflow).
For speakerless jobs, replace `--map ...` with `--without-diarization`. Review
the job directory before cleanup. `clean --final-only` selects the newest active
pipeline, speaker-name, or clarification publication, verifies the three retained
files in a staging directory, and then replaces the job transactionally. Use
`--yes` to skip its confirmation in automation. It intentionally removes the
manifest, speaker map, JSON, subtitles, and regeneration artifacts, so later
corrections require processing the retained recording again.

Deleting an entire individual job directory is the complete scrub for that
meeting; never delete the shared `data/` directory unless every contained job
is intentionally disposable.

### Per-user application data

| Platform | Location | Contents |
|----------|----------|----------|
| Windows | `%APPDATA%\meeting-notes\config.yaml` | User configuration |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/meeting-notes/config.yaml` | User configuration |
| All | `~/.cache/silero-vad/` | Silero VAD model when that backend downloads it |

An explicit `--config`, `MEETING_NOTES_CONFIG`, or project-local
`meeting-notes.yaml` may select a different configuration file. Inspect
`meeting-notes config status` and `meeting-notes config show --resolved` before
scrubbing.

First-party managed cache directories live under `project.cache_dir` and can be
removed after meeting-notes processes have stopped; required runtimes and models
will need to be downloaded or rebuilt again. Older releases used the OS user cache.
Inspect and migrate those recognized assets with:

```powershell
uv run meeting-notes cache status
uv run meeting-notes cache migrate
```

Migration verifies models and runtimes in project staging, updates legacy absolute
references, and removes recognized legacy copies only after configuration is saved.
Unrecognized files are retained and reported. Removing the active configuration
does not remove jobs or caches, and removing jobs does not remove the configuration
or models.

### Provider and development data

Qwen model payloads and their Hugging Face download metadata are explicitly kept
under `project.cache_dir`. Hugging Face login credentials remain provider-managed
and may live outside the project. Claude,
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
under `./cache/diarization/models/`, and writes its absolute path to configuration.
Users must
personally accept gated-model conditions in the browser; the application cannot
accept them on their behalf. No manual `HF_TOKEN` environment variable is
required.

CPU is the only default diarization device. The removed `auto` value is invalid;
existing configurations must change it to `cpu`, `rocm-hybrid`, or `cuda`.
`configure`, `config status`, and `doctor` probe for optional native-Windows AMD
acceleration and print an opt-in command when the prerequisites are available.

### Optional AMD ROCm hybrid acceleration

On supported Windows 11 AMD hardware, first install the driver/HIP prerequisites
from AMD and install the normal diarization extra. Then explicitly provision the
project-local runtime:

```powershell
uv sync --extra diarization
uv run meeting-notes diarization setup --acceleration rocm-hybrid
uv run meeting-notes diarization status
```

Setup displays the destination, available disk space, approximately 2.1 GiB of
downloads, approximately 6.1 GiB installed size, and a 9 GiB peak free-space
requirement before continuing. The confirmation defaults to yes; use `--yes` for
non-interactive provisioning. Failed staging installs are removed. The accelerated
worker deliberately keeps segmentation and clustering on CPU and moves only
speaker embeddings to the AMD GPU in FP32. It does not silently fall back to CPU.
The shared managed environment is stored under `./cache/runtimes/`, is pinned to
pyannote.audio 4.0.7 and AMD's Windows PyTorch 2.9.1+rocm7.2.1 packages, and can
also contains the Qwen forced-alignment Transformers profile. Readiness checks validate each
installed profile independently.

AMD's supported Windows PyTorch instructions and prerequisites are maintained at
<https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/windows/install-pytorch.html>.
Automated provisioning in meeting-notes currently supports native Windows only.

To reclaim the runtime storage while keeping the model, run the command below.
Removal is refused while Lemonade Qwen alignment is configured to share the runtime:

```powershell
uv run meeting-notes diarization remove-runtime
```

This changes `diarization.device` to `cpu` and clears
`diarization.rocm_gpu_runtime_path`.

### Offline setup from a model backup

A portable backup satisfies the gated-model requirement without logging in on the
destination computer:

```powershell
uv run meeting-notes diarization setup `
  --model-archive "D:\Transfer\meeting-notes-diarization-community-1.zip" `
  --acceleration rocm-hybrid
```

The archive sidecar is checked when present and every payload file is always
verified against the checksum inventory stored in the archive. Runtime packages and
credentials are never restored from a model backup. An already-restored valid
model is also detected and reused by a later setup command.

After `doctor` reports diarization as ready, resume an existing transcription
without rerunning ASR:

```powershell
uv run meeting-notes process "<audio-file>" --from diarize
```

Community-1 detects the number of speakers automatically. The default configuration
keeps a minimum of two speakers but does not impose a maximum. When attendance is
known, an exact count generally gives the pipeline a better constraint:

```powershell
uv run meeting-notes process "<audio-file>" --num-speakers 10
```

For variable attendance, provide either or both bounds for that invocation. These
options override the loaded configuration in memory and do not rewrite it:

```powershell
uv run meeting-notes process "<audio-file>" --min-speakers 4 --max-speakers 10
```

`--num-speakers` cannot be combined with `--min-speakers` or `--max-speakers`.
The resolved speaker policy is shown by `--dry-run` and recorded in the diarization
stage runtime metadata.

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

Alternatively, pass the diarization archive directly to `meeting-notes
diarization setup --model-archive ...`; this restores the model into the project
cache and provisions the selected CPU or ROCm execution path in one flow.

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

Managed assets live under the active configuration's `project.cache_dir` on every
platform. Failed downloads do not replace verified assets; Vulkan build logs are
retained beside the project-local runtime directory. The former per-user cache is
read only by `meeting-notes cache status/migrate` for one-way legacy migration.

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
