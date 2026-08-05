"""pyannote.audio diarization backend (optional dependency)."""

from __future__ import annotations

import json
import subprocess
import warnings
import wave
from pathlib import Path

import structlog

from meeting_notes.diarization.acceleration import (
    runtime_environment,
    runtime_python,
    validate_runtime,
)
from meeting_notes.diarization.base import (
    DiarizationBackend,
    DiarizationResult,
    DiarizationTurn,
)
from meeting_notes.diarization.setup import resolve_hf_token

log = structlog.get_logger()


class PyannoteDiarizationBackend(DiarizationBackend):
    """pyannote.audio speaker diarization backend.

    Requires: pip install pyannote.audio (or meeting-notes[diarization])
    Requires: Hugging Face token (HF_TOKEN env var) and model acceptance.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-community-1",
        model_path: Path | None = None,
        token_env: str = "HF_TOKEN",
        device: str = "cpu",
        rocm_gpu_runtime_path: Path | None = None,
        use_exclusive: bool = True,
    ) -> None:
        self._model_name = model_name
        self._model_path = model_path
        self._token_env = token_env
        self._device = device
        self._rocm_gpu_runtime_path = rocm_gpu_runtime_path
        self._use_exclusive = use_exclusive
        self._pipeline = None
        self._runtime_validated = False

    def _validate_rocm_runtime(self) -> None:
        if self._runtime_validated:
            return
        if not self._rocm_gpu_runtime_path:
            raise RuntimeError("rocm_gpu_runtime_path is not configured.")
        validate_runtime(self._rocm_gpu_runtime_path)
        self._runtime_validated = True

    @property
    def name(self) -> str:
        return "pyannote"

    @property
    def model_source(self) -> str:
        """Configured local pipeline path or remote model identifier."""
        return str(self._model_path) if self._model_path else self._model_name

    def is_available(self) -> bool:
        """Check if pyannote.audio is installed and HF token is set."""
        try:
            if self._device == "rocm-hybrid":
                if not self._model_path or not (self._model_path / "config.yaml").is_file():
                    return False
                if not self._rocm_gpu_runtime_path:
                    return False
                self._validate_rocm_runtime()
                return True
            from importlib.metadata import version

            version("pyannote.audio")

            local_model_ready = bool(self._model_path and self._model_path.exists())
            token, _ = resolve_hf_token(self._token_env)
            if not local_model_ready and not token:
                log.warning(
                    "pyannote.token_missing",
                    env_var=self._token_env,
                )
                return False
            return True
        except Exception:
            return False

    def _load_pipeline(self) -> None:
        """Lazy-load the pyannote pipeline."""
        if self._pipeline is not None:
            return
        if self._device == "rocm-hybrid":
            raise RuntimeError("ROCm hybrid diarization must run through the managed worker.")

        with warnings.catch_warnings():
            # The application supplies decoded waveform tensors, so pyannote's
            # optional TorchCodec decoder is not used.
            warnings.filterwarnings(
                "ignore",
                message=r"(?s)\s*torchcodec is not installed correctly.*",
                category=UserWarning,
                module=r"pyannote\.audio\.core\.io",
            )
            from pyannote.audio import Pipeline

        local_model_ready = bool(self._model_path and self._model_path.exists())
        token, _ = resolve_hf_token(self._token_env)
        if not local_model_ready and not token:
            raise RuntimeError(
                f"No local diarization model found and Hugging Face token is missing. "
                f"Set diarization.model_path or the {self._token_env} environment variable."
            )

        log.info(
            "pyannote.loading",
            model=self.model_source,
            device=self._device,
        )

        if local_model_ready:
            self._pipeline = Pipeline.from_pretrained(str(self._model_path))
        else:
            self._pipeline = Pipeline.from_pretrained(self._model_name, token=token)

        import torch

        self._pipeline.to(torch.device(self._device))

    def _diarize_rocm(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None,
        min_speakers: int,
        max_speakers: int | None,
    ) -> DiarizationResult:
        """Run Community-1 in the isolated CPU-segmentation/GPU-embedding worker."""
        if not self._model_path or not (self._model_path / "config.yaml").is_file():
            raise RuntimeError("ROCm hybrid diarization requires a configured local model.")
        if not self._rocm_gpu_runtime_path:
            raise RuntimeError("rocm_gpu_runtime_path is not configured.")
        self._validate_rocm_runtime()
        request = {
            "audio_path": str(audio_path.resolve()),
            "model_path": str(self._model_path.resolve()),
            "use_exclusive": self._use_exclusive,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        worker = Path(__file__).with_name("worker.py")
        result = subprocess.run(
            [str(runtime_python(self._rocm_gpu_runtime_path)), str(worker)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=runtime_environment(self._rocm_gpu_runtime_path),
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise RuntimeError(f"ROCm hybrid diarization failed: {detail}")
        try:
            payload = json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            raise RuntimeError("ROCm hybrid worker returned invalid JSON.") from error
        turns = [DiarizationTurn(**turn) for turn in payload.get("turns", [])]
        return DiarizationResult(
            turns=turns,
            backend="pyannote",
            model=str(payload.get("model", self.model_source)),
            device="rocm-hybrid",
            speakers=[str(value) for value in payload.get("speakers", [])],
        )

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int | None = None,
    ) -> DiarizationResult:
        """Run pyannote speaker diarization."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._device == "rocm-hybrid":
            return self._diarize_rocm(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

        self._load_pipeline()
        assert self._pipeline is not None

        # Build parameters
        params: dict = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers

        log.info(
            "pyannote.diarizing",
            audio=str(audio_path),
            params=params,
        )

        # Run diarization
        with warnings.catch_warnings():
            # Pyannote can produce a harmless degrees-of-freedom warning for
            # very short internal windows. It does not invalidate the completed
            # diarization result.
            warnings.filterwarnings(
                "ignore",
                message=r"std\(\): degrees of freedom is <= 0\..*",
                category=UserWarning,
            )
            diarization = self._pipeline(self._read_pcm_wave(audio_path), **params)
        annotation = diarization
        exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
        standard = getattr(diarization, "speaker_diarization", None)
        if self._use_exclusive and exclusive is not None:
            annotation = exclusive
        elif standard is not None:
            annotation = standard

        # Convert to our format
        turns: list[DiarizationTurn] = []
        speakers = set()

        if hasattr(annotation, "itertracks"):
            tracks = (
                (turn, speaker)
                for turn, _, speaker in annotation.itertracks(yield_label=True)
            )
        else:
            tracks = iter(annotation)

        for turn_idx, (turn, speaker) in enumerate(tracks):
            turns.append(
                DiarizationTurn(
                    turn_id=f"turn-{turn_idx:06d}",
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    source=self.model_source,
                )
            )
            speakers.add(speaker)

        log.info(
            "pyannote.complete",
            turns=len(turns),
            speakers=len(speakers),
        )

        return DiarizationResult(
            turns=turns,
            backend=self.name,
            model=self.model_source,
            device=self._device,
            speakers=sorted(speakers),
        )

    @staticmethod
    def _read_pcm_wave(audio_path: Path) -> dict[str, object]:
        """Load the normalized PCM WAV without pyannote/TorchCodec decoding."""
        import torch

        with wave.open(str(audio_path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            compression = stream.getcomptype()
            frames = stream.readframes(stream.getnframes())

        if sample_width != 2 or compression != "NONE":
            raise RuntimeError(
                "Diarization expects an uncompressed 16-bit PCM WAV. "
                f"Got sample_width={sample_width}, compression={compression}."
            )
        samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).clone()
        waveform = samples.reshape(-1, channels).transpose(0, 1).to(torch.float32)
        waveform /= 32768.0
        return {"waveform": waveform, "sample_rate": sample_rate}
