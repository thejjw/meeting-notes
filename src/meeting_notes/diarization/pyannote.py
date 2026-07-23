"""pyannote.audio diarization backend (optional dependency)."""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from meeting_notes.diarization.base import DiarizationBackend, DiarizationResult, DiarizationTurn

log = structlog.get_logger()


class PyannoteDiarizationBackend(DiarizationBackend):
    """pyannote.audio speaker diarization backend.

    Requires: pip install pyannote.audio (or meeting-notes[diarization])
    Requires: Hugging Face token (HF_TOKEN env var) and model acceptance.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-community-1",
        token_env: str = "HF_TOKEN",
        device: str = "auto",
        use_exclusive: bool = True,
    ) -> None:
        self._model_name = model_name
        self._token_env = token_env
        self._device = device
        self._use_exclusive = use_exclusive
        self._pipeline = None

    @property
    def name(self) -> str:
        return "pyannote"

    def is_available(self) -> bool:
        """Check if pyannote.audio is installed and HF token is set."""
        try:
            import pyannote.audio  # noqa: F401

            token = os.environ.get(self._token_env, "")
            if not token:
                log.warning(
                    "pyannote.token_missing",
                    env_var=self._token_env,
                )
                return False
            return True
        except ImportError:
            return False

    def _load_pipeline(self) -> None:
        """Lazy-load the pyannote pipeline."""
        if self._pipeline is not None:
            return

        from pyannote.audio import Pipeline

        token = os.environ.get(self._token_env, "")
        if not token:
            raise RuntimeError(
                f"Hugging Face token not found. Set {self._token_env} environment variable."
            )

        log.info(
            "pyannote.loading",
            model=self._model_name,
            device=self._device,
        )

        self._pipeline = Pipeline.from_pretrained(
            self._model_name,
            use_auth_token=token,
        )

        if self._device != "auto":
            self._pipeline.to(self._device)

    def diarize(
        self,
        audio_path: Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int = 2,
        max_speakers: int = 8,
    ) -> DiarizationResult:
        """Run pyannote speaker diarization."""
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_pipeline()
        assert self._pipeline is not None

        # Build parameters
        params: dict = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        log.info(
            "pyannote.diarizing",
            audio=str(audio_path),
            params=params,
        )

        # Run diarization
        diarization = self._pipeline(str(audio_path), **params)

        # Convert to our format
        turns: list[DiarizationTurn] = []
        speakers = set()

        for turn_idx, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            turns.append(
                DiarizationTurn(
                    turn_id=f"turn-{turn_idx:06d}",
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    source=self._model_name,
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
            model=self._model_name,
            device=self._device,
            speakers=sorted(speakers),
        )
