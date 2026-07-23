"""whisper.cpp ASR backend adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import structlog

from meeting_notes.asr.base import ASRBackend, ASRResult, ASRSegment
from meeting_notes.subprocess_utils import run_command

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
        )

        log.info(
            "whisper_cpp.transcribe",
            audio=str(audio_path),
            model=model,
            device=device,
            language=language,
            args_display=" ".join(args[:10]) + "...",
        )

        # Execute
        result = run_command(
            args,
            timeout=3600.0,  # 1 hour timeout for long recordings
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

        # Parse output
        segments = self._parse_output(result.stdout)
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

        if flash_attention and device == "cpu":
            args.append("--flash-attn")

        # Device-specific flags
        if device == "vulkan":
            args.extend(["--gpu-device", "0"])
        elif device == "cuda":
            args.extend(["--gpu-device", "0"])
        elif device == "rocm":
            args.extend(["--gpu-device", "0"])

        if extra_args:
            args.extend(extra_args)

        return args

    def _parse_output(self, stdout: str) -> list[ASRSegment]:
        """Parse whisper.cpp JSON output into segments."""
        segments: list[ASRSegment] = []

        # whisper-cli -oj outputs JSON with a "transcription" array
        try:
            data = json.loads(stdout)
            transcription = data.get("transcription", [])

            for i, seg in enumerate(transcription):
                segment = ASRSegment(
                    id=f"seg-{i:06d}",
                    start=seg.get("t0", 0) / 1000.0,  # ms to seconds
                    end=seg.get("t1", 0) / 1000.0,
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

        except json.JSONDecodeError:
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
