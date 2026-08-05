# Performance measurements

These figures describe one machine, one Korean meeting recording, and the
software versions listed below. They are useful for choosing among the project's
implemented engines, but they are not universal hardware claims or accuracy
benchmarks.

## Validation machine

Measured on 2026-08-05:

| Component | Value |
|---|---|
| OS | Windows 11 Pro, build 10.0.26200 |
| CPU/APU | AMD Ryzen AI Max+ 395, 16 cores / 32 logical processors |
| GPU | AMD Radeon 8060S, driver 32.0.31021.6002 |
| Memory | 128 GB class unified memory, 125.6 GiB visible to Windows |
| Lemonade | Server 11.5.1 |
| Lemonade Whisper backend | `whispercpp` 1.8.4 |
| Local whisper.cpp | 1.9.1 CPU runtime, 8 threads |
| Python AMD stack | PyTorch 2.9.1+rocm7.2.1, HIP 7.2 |

Windows reports only a small fixed `AdapterRAM` value for this unified-memory
GPU, so that field is not a meaningful VRAM limit.

## Controlled 240-second ASR comparison

The test used the first 240 seconds of the retained normalized Korean meeting
clip: mono, 16 kHz PCM WAV, `language: ko`. All Whisper rows used
`large-v3-turbo`. Local CPU execution included whisper.cpp process startup and
model loading. Lemonade Server was already running; Vulkan was warm, while the
NPU timing included switching the loaded Whisper recipe from Vulkan to NPU.
Qwen included its server request, alignment worker startup, and forced alignment.

| Route | End-to-end time | RTF | Processing speed | Speedup vs CPU Whisper |
|---|---:|---:|---:|---:|
| Local whisper.cpp CPU | 93.18 s | 0.388 | 2.58x real time | 1.00x |
| Lemonade Whisper NPU | 35.34 s | 0.147 | 6.79x real time | 2.64x |
| Lemonade Whisper Vulkan | 13.06 s | 0.054 | 18.38x real time | 7.14x |
| Qwen3-ASR 1.7B Q8, Vulkan + ROCm aligner | 29.03 s | 0.121 | 8.27x real time | 3.21x |

For the current warm Qwen run, internal metrics reported 8.89 seconds for Vulkan
transcription, 6.34 seconds for ROCm alignment, and 2.52 seconds for aligner
loading. The rest of its 29.03-second end-to-end time was orchestration, worker
startup, model preparation, and synchronization overhead. The alignment worker
reported about 7.3 GiB peak process RAM and 5.1 GiB peak GPU allocation. An
earlier cold run took 54.79 seconds, so model and server warm state matters.

The observed Whisper speedups are runtime speedups only. The outputs differed
substantially in segmentation and character count even though the same Whisper
model name and language were requested:

| Route | Segments | Output characters |
|---|---:|---:|
| Local Whisper CPU | 224 | 2,300 |
| Lemonade Whisper NPU | 57 | 744 |
| Lemonade Whisper Vulkan | 49 | 807 |
| Qwen3-ASR | 19 | 973 |

There is no human reference transcript for this clip, so these measurements do
not establish which output is more accurate. Qwen also uses a different model
and forced-alignment segmentation, making its row an operational throughput
comparison rather than a like-for-like model benchmark.

## Longer-recording evidence

The retained benchmark runner also processed 3,125.7 seconds (52.1 minutes) of
the same source recording:

| Route | End-to-end time | RTF | Processing speed |
|---|---:|---:|---:|
| Lemonade Whisper NPU | 358.4 s | 0.115 | 8.72x real time |
| Qwen3-ASR Vulkan + ROCm aligner | 467.9 s | 0.150 | 6.68x real time |

The corresponding production job manifest recorded 341.4 seconds for its
chunked Lemonade Whisper/NPU transcription stage. Differences between benchmark
and production timing include chunking, warm state, and runner overhead.

## Controlled 240-second diarization comparison

Both Community-1 routes used the configured automatic-count policy
(`min_speakers: 2`, no maximum). They produced the same 56 turns and detected
the same two speaker labels.

| Route | End-to-end time | RTF | Processing speed | Speedup vs CPU |
|---|---:|---:|---:|---:|
| pyannote CPU | 72.09 s | 0.300 | 3.33x real time | 1.00x |
| pyannote ROCm hybrid, first validated use | 22.94 s | 0.096 | 10.46x real time | 3.14x |
| pyannote ROCm hybrid, validation cached | 16.61 s | 0.069 | 14.45x real time | 4.34x |

The ROCm hybrid keeps segmentation on CPU and moves embedding inference to the
AMD GPU. Its first row includes 6.32 seconds spent validating the shared runtime
and importing the complete pyannote profile. A backend instance caches that
successful result, so subsequent calls use the second timing.

The combined Qwen/diarization environment initially exposed an incompatibility:
Transformers 5.14.1 imported optional distributed symbols that AMD's native-
Windows PyTorch omits. The project now installs one narrowly scoped compatibility
shim before either worker imports Transformers or pyannote. Runtime validation
also imports each requested profile, so this class of failure is reported as a
broken runtime during status/setup instead of at inference time. The measurements
above are from the normal application adapter after that fix, not a patched test
worker.

Two older CPU production jobs provide additional context:

| Audio duration | Diarization time | RTF | Processing speed |
|---:|---:|---:|---:|
| 2,766.7 s | 675.8 s | 0.244 | 4.09x real time |
| 3,125.7 s | 556.5 s | 0.178 | 5.62x real time |

Those jobs predate the explicit `cpu` device label and recorded the former
`auto` value, which resolved to CPU.

## Summarization

No cross-backend summarization speed ranking is published. Codex, Claude, and
other provider-backed adapters depend on remote model choice, service load, and
reasoning settings. Lemonade uses a different local model and a reduced Markdown
contract, so elapsed time alone would not represent equivalent output quality or
features.

## Reproducing the benchmark

The Qwen/NPU benchmark matrix can retain JSON, CSV, Markdown, and transcript
artifacts for a chosen recording:

```powershell
uv run meeting-notes benchmark "meeting.m4a" `
  --matrix config/qwen3-asr-benchmark.example.yaml `
  --start-seconds 0 --duration-seconds 240
```

Run backends on the same normalized clip, record whether each model/server is
cold or warm, and compare transcript artifacts before interpreting speed as a
quality-neutral improvement. Generated benchmark data is intentionally ignored
by Git because it can contain private meeting content. Reproducing the full
four-route table also requires the verified local CPU model/runtime and the
Lemonade Vulkan backend; run those against the same extracted clip rather than
comparing results from unrelated recordings.
