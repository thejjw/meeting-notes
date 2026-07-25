"""whisper.cpp ASR backend adapter."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path

import structlog

from meeting_notes.asr.base import ASRBackend, ASRResult, ASRSegment
from meeting_notes.subprocess_utils import run_command, run_command_streaming

log = structlog.get_logger()


class WhisperCppBackend(ASRBackend):
    """whisper.cpp ASR backend using the whisper-cli executable."""

    def __init__(self, executable: str = "whisper-cli") -> None:
        self._executable = executable
        self._version: str | None = None

    @property
    def name(self) -> str:
        return "whisper_cpp"

    def is_available(self) -> bool:
        """Check if whisper-cli is installed and runnable."""
        try:
            result = run_command(
                [self._executable, "--help"],
                timeout=5.0,
                label="whisper-cli-check",
            )
            return result.returncode == 0
        except RuntimeError:
            return False

    def get_version(self) -> str:
        """Get whisper-cli version string."""
        if self._version:
            return self._version
        try:
            result = run_command(
                [self._executable, "--version"],
                timeout=5.0,
                label="whisper-cli-version",
            )
            self._version = result.stdout.strip() or "unknown"
        except RuntimeError:
            self._version = "not installed"
        return self._version

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str = "medium",
        model_path: Path | None = None,
        language: str = "ko",
        task: str = "transcribe",
        initial_prompt: str | None = None,
        word_timestamps: bool = False,
        threads: int = 0,
        extra_args: list[str] | None = None,
        device: str = "cpu",
        model_variant: str = "fp16",
        flash_attention: bool = True,
        gpu_device: str | None = None,
    ) -> ASRResult:
        """Run whisper.cpp transcription.

        Supports CPU, Vulkan, ROCm/HIP, and CUDA depending on the build.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Resolve model path
        resolved_model_path = self._resolve_model_path(model, model_path)

        # Build command
        args = self._build_args(
            audio_path=audio_path,
            model_path=resolved_model_path,
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            threads=threads,
            extra_args=extra_args,
            device=device,
            model_variant=model_variant,
            flash_attention=flash_attention,
            gpu_device=gpu_device,
        )

        log.info(
            "whisper_cpp.transcribe",
            audio=str(audio_path),
            model=model,
            device=device,
            language=language,
            args_display=" ".join(args[:10]) + "...",
        )

        progress_seen = -1

        def report_progress(stream: str, line: str) -> None:
            nonlocal progress_seen
            match = re.search(r"progress\s*=\s*(\d+)%", line, re.IGNORECASE)
            if match:
                percent = int(match.group(1))
                if percent != progress_seen:
                    progress_seen = percent
                    log.info("whisper_cpp.progress", percent=percent)
            elif "error" in line.lower():
                log.warning("whisper_cpp.output", stream=stream, message=line[:500])

        # -oj writes JSON to a file; it does not emit JSON on stdout. Manage
        # this directory explicitly so valuable output can be retained when
        # decoding or parsing fails after a long transcription.
        temp_dir = Path(tempfile.mkdtemp(prefix="meeting-notes-whisper-"))
        retain_raw_output = False
        try:
            output_prefix = temp_dir / "transcription"
            args.extend(["-of", str(output_prefix)])
            if "--print-progress" not in args and "-pp" not in args:
                args.append("--print-progress")

            result = run_command_streaming(
                args,
                on_output=report_progress,
                timeout=7200.0,
                label="whisper-cpp-transcribe",
            )

            if not result.success:
                log.error(
                    "whisper_cpp.failed",
                    returncode=result.returncode,
                    stderr=result.stderr[:1000],
                )
                raise RuntimeError(
                    f"whisper.cpp transcription failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:500]}"
                )

            json_path = output_prefix.with_suffix(".json")
            if not json_path.exists():
                raise RuntimeError(
                    "whisper.cpp completed but did not create its JSON output. "
                    f"Expected: {json_path}"
                )
            json_output, invalid_utf8, first_invalid_offset = self._read_json_output(json_path)
            if invalid_utf8:
                retain_raw_output = True
                log.warning(
                    "whisper_cpp.invalid_utf8_repaired",
                    replacement_sequences=invalid_utf8,
                    first_byte_offset=first_invalid_offset,
                    raw_json=str(json_path),
                )
            segments = self._parse_output(json_output, fallback_to_text=False)
        except Exception:
            retain_raw_output = True
            log.error("whisper_cpp.raw_output_retained", path=str(temp_dir))
            raise
        finally:
            if not retain_raw_output:
                shutil.rmtree(temp_dir, ignore_errors=True)

        raw = {"stdout": result.stdout[:5000], "stderr": result.stderr[:2000]}

        # Extract language if auto-detected
        detected_lang = language
        lang_match = re.search(r"Detected language:\s*(\w+)", result.stderr)
        if lang_match:
            detected_lang = lang_match.group(1)

        return ASRResult(
            segments=segments,
            language=detected_lang,
            backend=self.name,
            model=model,
            device=device,
            raw_output=raw,
        )

    def _resolve_model_path(self, model: str, explicit_path: Path | None) -> Path:
        """Resolve model file path from name or explicit path."""
        if explicit_path:
            if not explicit_path.exists():
                raise FileNotFoundError(f"Model file not found: {explicit_path}")
            return explicit_path

        # Try common GGML model naming patterns
        candidates = [
            Path(f"ggml-{model}.bin"),
            Path(f"models/ggml-{model}.bin"),
            Path(f"cache/models/ggml-{model}.bin"),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Return expected name even if not found (will fail with clear error)
        return Path(f"ggml-{model}.bin")

    def _build_args(
        self,
        audio_path: Path,
        model_path: Path,
        language: str,
        task: str,
        initial_prompt: str | None,
        word_timestamps: bool,
        threads: int,
        extra_args: list[str] | None,
        device: str,
        model_variant: str,
        flash_attention: bool,
        gpu_device: str | None,
    ) -> list[str]:
        """Build whisper-cli argument list."""
        args = [
            self._executable,
            "-m", str(model_path),
            "-f", str(audio_path),
            "--language", language,
            "-oj",  # Output JSON format
        ]

        # Add translate flag if task is translation (no --task flag in this version)
        if task == "translate":
            args.append("--translate")

        if threads > 0:
            args.extend(["-t", str(threads)])

        if initial_prompt:
            args.extend(["--prompt", initial_prompt])

        if word_timestamps:
            args.append("--word-timestamps")

        if flash_attention:
            args.append("--flash-attn")

        # Device-specific flags
        if device == "cpu":
            args.append("--no-gpu")
        elif device in {"vulkan", "cuda", "rocm"}:
            args.extend(["--device", gpu_device or "0"])
        else:
            raise ValueError(f"Unsupported whisper.cpp device: {device}")

        if extra_args:
            args.extend(extra_args)

        return args

    @staticmethod
    def _read_json_output(path: Path) -> tuple[str, int, int | None]:
        """Decode whisper.cpp JSON, repairing rare malformed model byte sequences."""
        raw = path.read_bytes()
        try:
            return raw.decode("utf-8"), 0, None
        except UnicodeDecodeError as error:
            repaired = raw.decode("utf-8", errors="replace")
            return repaired, repaired.count("\ufffd"), error.start

    def _parse_output(
        self,
        stdout: str,
        *,
        fallback_to_text: bool = True,
    ) -> list[ASRSegment]:
        """Parse whisper.cpp JSON output into segments."""
        segments: list[ASRSegment] = []

        # whisper-cli -oj outputs JSON with a "transcription" array
        try:
            data = json.loads(stdout)
            transcription = data.get("transcription", [])

            for i, seg in enumerate(transcription):
                offsets = seg.get("offsets", {})
                start_ms = offsets.get("from", seg.get("t0", 0))
                end_ms = offsets.get("to", seg.get("t1", 0))
                segment = ASRSegment(
                    id=f"seg-{i:06d}",
                    start=start_ms / 1000.0,
                    end=end_ms / 1000.0,
                    text=seg.get("text", "").strip(),
                    language=seg.get("language"),
                    confidence=1.0 - seg.get("no_speech_prob", 0)
                    if "no_speech_prob" in seg
                    else None,
                    metrics={
                        "avg_logprob": seg.get("avg_logprob"),
                        "no_speech_prob": seg.get("no_speech_prob"),
                        "compression_ratio": seg.get("compression_ratio"),
                    },
                    source={
                        "backend": "whisper_cpp",
                        "raw_segment_index": i,
                    },
                )
                if segment.text:
                    segments.append(segment)

        except json.JSONDecodeError as error:
            if not fallback_to_text:
                raise RuntimeError(
                    "whisper.cpp produced invalid JSON; raw output was retained for recovery"
                ) from error
            # Fall back to line-by-line parsing if JSON fails
            log.warning("whisper_cpp.json_parse_failed", falling_back_to_text=True)
            segments = self._parse_text_output(stdout)

        return segments

    def _parse_text_output(self, stdout: str) -> list[ASRSegment]:
        """Parse whisper.cpp plain text output as fallback.

        Format: [HH:MM:SS.mmm --> HH:MM:SS.mmm]  text
        """
        segments: list[ASRSegment] = []
        pattern = re.compile(
            r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)"
        )

        for i, line in enumerate(stdout.splitlines()):
            match = pattern.match(line.strip())
            if match:
                start = self._timestamp_to_seconds(match.group(1))
                end = self._timestamp_to_seconds(match.group(2))
                text = match.group(3).strip()
                if text:
                    segments.append(
                        ASRSegment(
                            id=f"seg-{i:06d}",
                            start=start,
                            end=end,
                            text=text,
                            source={"backend": "whisper_cpp", "raw_segment_index": i},
                        )
                    )

        return segments

    @staticmethod
    def _timestamp_to_seconds(ts: str) -> float:
        """Convert HH:MM:SS.mmm to seconds."""
        parts = ts.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
