"""Docker-based whisper.cpp backend wrapper."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import structlog

from meeting_notes.asr.base import ASRBackend, ASRResult, ASRSegment
from meeting_notes.subprocess_utils import run_command

log = structlog.get_logger()

# Default Docker image name
DEFAULT_IMAGE = "meeting-notes-whisper-cpp"


class DockerWhisperCppBackend(ASRBackend):
    """wh.cpp ASR backend running inside Docker.

    Builds and runs the whisper.cpp Docker image for isolated,
    reproducible transcription without requiring local compilation.
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        dockerfile_dir: str | None = None,
    ) -> None:
        self._image = image
        self._dockerfile_dir = dockerfile_dir

    @property
    def name(self) -> str:
        return "whisper_cpp_docker"

    def is_available(self) -> bool:
        """Check if Docker is available and image exists."""
        try:
            result = run_command(
                ["docker", "image", "inspect", self._image],
                timeout=10.0,
                label="docker-check",
            )
            return result.returncode == 0
        except RuntimeError:
            return False

    def build_image(self, dockerfile_dir: Path | None = None) -> None:
        """Build the whisper.cpp Docker image."""
        build_dir = dockerfile_dir or Path(__file__).parent.parent.parent / "docker" / "whisper-cpp"

        if not build_dir.exists():
            raise FileNotFoundError(f"Dockerfile directory not found: {build_dir}")

        log.info("docker.building", image=self._image, dir=str(build_dir))

        result = run_command(
            ["docker", "build", "-t", self._image, str(build_dir)],
            timeout=600.0,  # 10 minutes for build
            label="docker-build",
        )

        if not result.returncode == 0:
            raise RuntimeError(f"Docker build failed:\n{result.stderr[:1000]}")

        log.info("docker.built", image=self._image)

    def get_version(self) -> str:
        return f"docker:{self._image}"

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
    ) -> ASRResult:
        """Run transcription inside Docker container."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Resolve model path
        resolved_model = self._resolve_model_path(model, model_path)

        with tempfile.TemporaryDirectory(prefix="whisper-docker-") as tmp_dir:
            tmp = Path(tmp_dir)

            # Copy audio to temp dir for Docker mount
            audio_copy = tmp / audio_path.name
            import shutil
            shutil.copy2(audio_path, audio_copy)

            # Copy model if local
            model_copy = None
            if resolved_model.exists():
                model_copy = tmp / resolved_model.name
                shutil.copy2(resolved_model, model_copy)
            else:
                # Model will be inside the container
                model_copy = tmp / f"ggml-{model}.bin"

            # Build Docker command
            docker_args = [
                "docker", "run", "--rm",
                "-v", f"{tmp}:/data",
            ]

            whisper_args = [
                "-m", f"/data/{model_copy.name}",
                "-f", f"/data/{audio_copy.name}",
                "--language", language,
                "--task", task,
                "-oj",  # JSON output
            ]

            if threads > 0:
                whisper_args.extend(["-t", str(threads)])

            if initial_prompt:
                whisper_args.extend(["--prompt", initial_prompt])

            if word_timestamps:
                whisper_args.append("--word-timestamps")

            if extra_args:
                whisper_args.extend(extra_args)

            cmd = docker_args + [self._image] + whisper_args

            log.info(
                "docker.transcribe",
                audio=str(audio_path),
                model=model,
                language=language,
            )

            result = run_command(
                cmd,
                timeout=3600.0,
                label="docker-whisper-transcribe",
            )

            if not result.returncode == 0:
                raise RuntimeError(
                    f"Docker whisper.cpp failed (exit {result.returncode}):\n"
                    f"  stderr: {result.stderr[:500]}"
                )

            # Parse output
            segments = self._parse_output(result.stdout)

            return ASRResult(
                segments=segments,
                language=language,
                backend=self.name,
                model=model,
                device="docker",
            )

    def _resolve_model_path(self, model: str, explicit_path: Path | None) -> Path:
        """Resolve model file path."""
        if explicit_path:
            return explicit_path

        candidates = [
            Path(f"ggml-{model}.bin"),
            Path(f"models/ggml-{model}.bin"),
            Path.home() / ".cache" / "whisper" / f"ggml-{model}.bin",
        ]

        for c in candidates:
            if c.exists():
                return c

        return Path(f"ggml-{model}.bin")

    def _parse_output(self, stdout: str) -> list[ASRSegment]:
        """Parse whisper.cpp JSON output."""
        segments = []

        try:
            data = json.loads(stdout)
            for i, seg in enumerate(data.get("transcription", [])):
                segment = ASRSegment(
                    id=f"seg-{i:06d}",
                    start=seg.get("t0", 0) / 1000.0,
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
                        "backend": "whisper_cpp_docker",
                        "raw_segment_index": i,
                    },
                )
                if segment.text:
                    segments.append(segment)
        except json.JSONDecodeError:
            log.warning("docker.json_parse_failed")

        return segments
