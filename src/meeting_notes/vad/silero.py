"""Silero VAD backend using ONNX runtime."""

from __future__ import annotations

import structlog

from meeting_notes.vad.base import VADBackend, VADSegment

log = structlog.get_logger()


class SileroVADBackend(VADBackend):
    """Silero VAD backend using ONNX runtime.

    Requires: pip install onnxruntime (or use PyTorch backend)
    Model: silero_vad.onnx (auto-downloaded on first use)
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path
        self._model = None

    @property
    def name(self) -> str:
        return "silero"

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self) -> None:
        """Load Silero VAD model."""
        if self._model is not None:
            return

        try:
            import onnxruntime as ort

            model_path = self._model_path
            if not model_path:
                # Try to find or download the model
                import urllib.request
                from pathlib import Path

                cache_dir = Path.home() / ".cache" / "silero-vad"
                cache_dir.mkdir(parents=True, exist_ok=True)
                model_path = str(cache_dir / "silero_vad.onnx")

                if not Path(model_path).exists():
                    url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
                    log.info("silero.downloading_model", url=url)
                    urllib.request.urlretrieve(url, model_path)

            self._model = ort.InferenceSession(model_path)
            log.info("silero.model_loaded", path=model_path)

        except ImportError:
            raise RuntimeError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )

    def detect(
        self,
        audio_path: str | Path,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        speech_pad_ms: int = 200,
    ) -> list[VADSegment]:
        """Detect speech segments using Silero VAD."""
        import wave

        import numpy as np

        self._load_model()

        # Read audio file
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)

        # Convert to float32 numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Simple resampling
            ratio = 16000 / sample_rate
            new_length = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_length),
                np.arange(len(audio)),
                audio,
            )

        # Process in chunks
        chunk_size = 512  # Silero expects 512-sample chunks
        speech_probs = []

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            # Get speech probability
            input_name = self._model.get_inputs()[0].name
            prob = self._model.run(None, {input_name: chunk.reshape(1, -1)})[0]
            speech_probs.append(float(prob[0]))

        # Convert probabilities to segments
        segments: list[VADSegment] = []
        min_speech_samples = int(min_speech_ms * 16000 / 1000)
        min_silence_samples = int(min_silence_ms * 16000 / 1000)
        speech_pad_samples = int(speech_pad_ms * 16000 / 1000)

        in_speech = False
        speech_start = 0
        silence_count = 0

        for i, prob in enumerate(speech_probs):
            sample_idx = i * chunk_size

            if prob >= threshold:
                if not in_speech:
                    speech_start = max(0, sample_idx - speech_pad_samples)
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += chunk_size
                    if silence_count >= min_silence_samples:
                        speech_end = sample_idx + speech_pad_samples
                        speech_duration = speech_end - speech_start
                        if speech_duration >= min_speech_samples:
                            segments.append(
                                VADSegment(
                                    start=speech_start / 16000.0,
                                    end=speech_end / 16000.0,
                                    confidence=threshold,
                                )
                            )
                        in_speech = False
                        silence_count = 0

        # Handle final speech segment
        if in_speech:
            speech_end = len(audio)
            speech_duration = speech_end - speech_start
            if speech_duration >= min_speech_samples:
                segments.append(
                    VADSegment(
                        start=speech_start / 16000.0,
                        end=speech_end / 16000.0,
                        confidence=threshold,
                    )
                )

        log.info("silero.detected", segments=len(segments))
        return segments
