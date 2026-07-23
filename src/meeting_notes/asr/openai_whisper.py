"""openai-whisper ASR backend adapter (optional dependency)."""

from __future__ import annotations

from pathlib import Path

from meeting_notes.asr.base import ASRBackend, ASRResult


class OpenAIWhisperBackend(ASRBackend):
    """openai-whisper ASR backend using the Python library.

    Requires: pip install openai-whisper (or meeting-notes[whisper-openai])
    """

    def __init__(self) -> None:
        self._model = None

    @property
    def name(self) -> str:
        return "openai_whisper"

    def is_available(self) -> bool:
        try:
            import whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def get_version(self) -> str:
        try:
            import whisper

            return whisper.__version__
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
    ) -> ASRResult:
        import whisper

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Determine device
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = device == "cuda"

        # Load model
        model_obj = whisper.load_model(model, device=device)

        # Transcribe
        result = model_obj.transcribe(
            str(audio_path),
            language=language if language != "auto" else None,
            task=task,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            fp16=fp16,
        )

        # Convert to our format
        from meeting_notes.asr.base import ASRSegment

        segments = []
        for i, seg in enumerate(result.get("segments", [])):
            segments.append(
                ASRSegment(
                    id=f"seg-{i:06d}",
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    language=result.get("language"),
                    metrics={
                        "avg_logprob": seg.get("avg_logprob"),
                        "no_speech_prob": seg.get("no_speech_prob"),
                        "compression_ratio": seg.get("compression_ratio"),
                    },
                    source={
                        "backend": "openai_whisper",
                        "model": model,
                        "raw_segment_index": i,
                    },
                )
            )

        return ASRResult(
            segments=segments,
            language=result.get("language", language),
            backend=self.name,
            model=model,
            device=device,
        )
