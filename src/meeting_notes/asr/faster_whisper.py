"""faster-whisper ASR backend adapter (optional dependency)."""

from __future__ import annotations

from pathlib import Path

from meeting_notes.asr.base import ASRBackend, ASRResult


class FasterWhisperBackend(ASRBackend):
    """faster-whisper ASR backend using the CTranslate2 library.

    Requires: pip install faster-whisper (or meeting-notes[faster-whisper])
    Supports: CPU and NVIDIA CUDA. Does NOT support AMD ROCm.
    """

    @property
    def name(self) -> str:
        return "faster_whisper"

    def is_available(self) -> bool:
        try:
            from faster_whisper import WhisperModel  # noqa: F401

            return True
        except ImportError:
            return False

    def get_version(self) -> str:
        try:
            import faster_whisper

            return getattr(faster_whisper, "__version__", "unknown")
        except ImportError:
            return "not installed"

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
        device: str = "auto",
        compute_type: str = "auto",
        batch_size: int = 1,
    ) -> ASRResult:
        from faster_whisper import WhisperModel

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Determine device and compute type
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        # Load model
        model_path_str = str(model_path) if model_path else model
        model_obj = WhisperModel(
            model_path_str,
            device=device,
            compute_type=compute_type,
        )

        # Transcribe
        segments_gen, info = model_obj.transcribe(
            str(audio_path),
            language=language if language != "auto" else None,
            task=task,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            beam_size=5,
            vad_filter=False,
        )

        # Convert to our format
        from meeting_notes.asr.base import ASRSegment

        segments = []
        for i, seg in enumerate(segments_gen):
            segments.append(
                ASRSegment(
                    id=f"seg-{i:06d}",
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    language=getattr(info, "language", language),
                    confidence=seg.avg_logprob if hasattr(seg, "avg_logprob") else None,
                    metrics={
                        "avg_logprob": getattr(seg, "avg_logprob", None),
                        "no_speech_prob": getattr(seg, "no_speech_prob", None),
                        "compression_ratio": getattr(seg, "compression_ratio", None),
                    },
                    source={
                        "backend": "faster_whisper",
                        "model": model,
                        "raw_segment_index": i,
                    },
                )
            )

        return ASRResult(
            segments=segments,
            language=getattr(info, "language", language),
            backend=self.name,
            model=model,
            device=device,
        )
