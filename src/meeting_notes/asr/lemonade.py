"""AMD Lemonade OpenAI-compatible ASR backend."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from meeting_notes.asr.base import ASRBackend, ASRReadiness, ASRResult, ASRSegment

log = structlog.get_logger()

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class LemonadeASRBackend(ASRBackend):
    """Transcribe normalized WAV files through a running Lemonade server."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:13305",
        model_id: str = "Whisper-Large-v3-Turbo",
        api_key_env: str = "LEMONADE_API_KEY",
        expected_device: str = "npu",
        connect_timeout_seconds: float = 5.0,
        provisioning_timeout_seconds: float = 3600.0,
        transcription_timeout_seconds: float = 7200.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.expected_device = expected_device
        self.connect_timeout_seconds = connect_timeout_seconds
        self.provisioning_timeout_seconds = provisioning_timeout_seconds
        self.transcription_timeout_seconds = transcription_timeout_seconds
        self._version = ""

    @property
    def name(self) -> str:
        return "lemonade"

    def _headers(self) -> dict[str, str]:
        token = os.environ.get(self.api_key_env) if self.api_key_env else None
        return {"Authorization": f"Bearer {token}"} if token else {}

    def _timeout(self, total: float) -> httpx.Timeout:
        return httpx.Timeout(total, connect=self.connect_timeout_seconds)

    def _get_json(self, path: str, *, timeout: float | None = None) -> dict[str, Any]:
        try:
            response = httpx.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
                timeout=self._timeout(timeout or self.connect_timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"Lemonade request failed at {self.base_url}{path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"Lemonade returned a non-object response for {path}.")
        return payload

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        try:
            response = httpx.post(
                f"{self.base_url}{path}",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout(timeout),
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"Lemonade request failed at {self.base_url}{path}: {error}"
            ) from error
        if not isinstance(body, dict):
            raise RuntimeError(f"Lemonade returned a non-object response for {path}.")
        if body.get("status") == "error":
            raise RuntimeError(str(body.get("message") or "Lemonade operation failed."))
        return body

    def is_available(self) -> bool:
        try:
            response = httpx.get(
                f"{self.base_url}/live",
                headers=self._headers(),
                timeout=self._timeout(self.connect_timeout_seconds),
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def get_version(self) -> str:
        if self._version:
            return self._version
        try:
            self._version = str(self._get_json("/v1/health").get("version") or "unknown")
        except RuntimeError:
            self._version = "unavailable"
        return self._version

    def list_models(self, *, show_all: bool = False) -> list[dict[str, Any]]:
        payload = self._get_json(f"/v1/models{'?show_all=true' if show_all else ''}")
        data = payload.get("data", [])
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def model_info(self, *, show_all: bool = True) -> dict[str, Any] | None:
        return next(
            (
                item
                for item in self.list_models(show_all=show_all)
                if item.get("id") == self.model_id
            ),
            None,
        )

    def _loaded_model(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        health = self._get_json("/v1/health")
        self._version = str(health.get("version") or "unknown")
        loaded = health.get("all_models_loaded", [])
        if not isinstance(loaded, list):
            loaded = []
        model = next(
            (
                item
                for item in loaded
                if isinstance(item, dict) and item.get("model_name") == self.model_id
            ),
            None,
        )
        return model, health

    def check_readiness(
        self,
        *,
        model: str = "",
        expected_device: str = "",
        allow_provision: bool = False,
    ) -> ASRReadiness:
        del model
        required_device = expected_device or self.expected_device
        if not self.is_available():
            return ASRReadiness(
                available=False,
                detail=(
                    f"Lemonade Server is not reachable at {self.base_url}. "
                    "Start Lemonade Server manually, then retry."
                ),
                device=required_device,
                metadata={"base_url": self.base_url, "model_id": self.model_id},
            )
        try:
            info = self.model_info(show_all=True)
            if info is None:
                return ASRReadiness(
                    available=False,
                    detail=(
                        f"Lemonade model '{self.model_id}' is not registered "
                        "in the server catalogue."
                    ),
                    version=self.get_version(),
                    device=required_device,
                    metadata={"base_url": self.base_url, "model_id": self.model_id},
                )
            labels = info.get("labels")
            if not isinstance(labels, list) or "transcription" not in labels:
                return ASRReadiness(
                    available=False,
                    detail=f"Lemonade model '{self.model_id}' is not a transcription model.",
                    version=self.get_version(),
                    device=required_device,
                    metadata={"base_url": self.base_url, "model_id": self.model_id},
                )
            downloaded = bool(info.get("downloaded"))
            if not downloaded and not allow_provision:
                return ASRReadiness(
                    available=False,
                    detail=f"Lemonade model '{self.model_id}' is registered but not downloaded.",
                    version=self.get_version(),
                    device=required_device,
                    metadata={
                        "base_url": self.base_url,
                        "model_id": self.model_id,
                        "downloaded": False,
                        "size_gb": info.get("size"),
                    },
                )
            loaded, health = self._loaded_model()
            if loaded is None:
                return ASRReadiness(
                    available=downloaded,
                    detail=(
                        f"Lemonade model '{self.model_id}' is downloaded and will be loaded "
                        "before transcription."
                    ),
                    version=str(health.get("version") or "unknown"),
                    device=required_device,
                    metadata={
                        "base_url": self.base_url,
                        "model_id": self.model_id,
                        "downloaded": downloaded,
                        "loaded": False,
                        "size_gb": info.get("size"),
                    },
                )
            actual_device = str(loaded.get("device") or "")
            ready = (
                bool(loaded.get("backend_alive", True))
                and str(loaded.get("status")) in {"ready", "in_use"}
                and (not required_device or required_device in actual_device.split())
            )
            detail = (
                f"Lemonade model '{self.model_id}' is ready on {actual_device or 'unknown device'}."
                if ready
                else (
                    f"Lemonade model '{self.model_id}' is loaded on "
                    f"{actual_device or 'an unknown device'} with status "
                    f"'{loaded.get('status')}', expected ready on {required_device}."
                )
            )
            return ASRReadiness(
                available=ready,
                detail=detail,
                version=str(health.get("version") or "unknown"),
                device=actual_device,
                metadata={
                    "base_url": self.base_url,
                    "model_id": self.model_id,
                    "downloaded": downloaded,
                    "loaded": True,
                    "status": loaded.get("status"),
                    "size_gb": info.get("size"),
                },
            )
        except RuntimeError as error:
            return ASRReadiness(
                available=False,
                detail=str(error),
                version=self.get_version(),
                device=required_device,
                metadata={"base_url": self.base_url, "model_id": self.model_id},
            )

    def pull_model(self, *, progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        """Download/install the registered model, optionally reporting SSE progress."""
        info = self.model_info(show_all=True)
        if info is None:
            raise RuntimeError(f"Lemonade model '{self.model_id}' is not registered.")
        if info.get("downloaded"):
            return
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/v1/pull",
                json={"model_name": self.model_id, "stream": True},
                headers=self._headers(),
                timeout=self._timeout(self.provisioning_timeout_seconds),
            ) as response:
                response.raise_for_status()
                event = ""
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event = line.partition(":")[2].strip()
                    elif line.startswith("data:"):
                        import json

                        data = json.loads(line.partition(":")[2].strip())
                        if isinstance(data, dict):
                            data["event"] = event
                            if progress:
                                progress(data)
                            if event == "error":
                                raise RuntimeError(str(data.get("error") or "download failed"))
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"Failed to download Lemonade model '{self.model_id}': {error}"
            ) from error
        refreshed = self.model_info(show_all=True)
        if not refreshed or not refreshed.get("downloaded"):
            raise RuntimeError(f"Lemonade did not report '{self.model_id}' as downloaded.")

    def load_model(self) -> ASRReadiness:
        """Load the installed model on the required accelerator and verify it."""
        info = self.model_info(show_all=True)
        if not info or not info.get("downloaded"):
            raise RuntimeError(f"Lemonade model '{self.model_id}' is not downloaded.")
        current = self.check_readiness(expected_device=self.expected_device)
        if current.available and current.metadata.get("loaded"):
            return current
        self._post_json(
            "/v1/load",
            {
                "model_name": self.model_id,
                "whispercpp_backend": self.expected_device,
                "save_options": False,
            },
            timeout=self.provisioning_timeout_seconds,
        )
        deadline = time.monotonic() + min(self.provisioning_timeout_seconds, 120.0)
        readiness = self.check_readiness(expected_device=self.expected_device)
        while time.monotonic() < deadline:
            if readiness.available and readiness.metadata.get("loaded"):
                return readiness
            if (
                readiness.metadata.get("loaded")
                and self.expected_device not in readiness.device.split()
            ):
                break
            time.sleep(0.5)
            readiness = self.check_readiness(expected_device=self.expected_device)
        raise RuntimeError(readiness.detail)

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
        del model_path, word_timestamps, threads
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.lower() != ".wav":
            raise ValueError("Lemonade transcription requires a normalized WAV file.")
        if task != "transcribe":
            raise ValueError("Lemonade ASR currently supports transcription only.")
        if initial_prompt:
            raise ValueError("Lemonade ASR does not currently support an initial prompt.")
        if extra_args:
            raise ValueError("Lemonade ASR does not accept arbitrary backend arguments.")

        readiness = self.load_model()
        fields = {
            "model": self.model_id,
            "response_format": "verbose_json",
        }
        if language and language != "auto":
            fields["language"] = language
        try:
            with audio_path.open("rb") as audio:
                response = httpx.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    data=fields,
                    files={"file": (audio_path.name, audio, "audio/wav")},
                    headers=self._headers(),
                    timeout=self._timeout(self.transcription_timeout_seconds),
                )
                response.raise_for_status()
                payload = response.json()
        except (OSError, httpx.HTTPError, ValueError) as error:
            raise RuntimeError(f"Lemonade transcription failed: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("Lemonade returned a non-object transcription response.")
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            raise RuntimeError(
                "Lemonade did not return timestamped segments; "
                "response_format=verbose_json is required."
            )

        detected = str(payload.get("detected_language") or payload.get("language") or language)
        language_map = {"korean": "ko", "english": "en"}
        normalized_language = language_map.get(detected.lower(), detected.lower())
        segments: list[ASRSegment] = []
        for index, raw in enumerate(raw_segments):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            no_speech = raw.get("no_speech_prob")
            segments.append(
                ASRSegment(
                    id=f"seg-{index:06d}",
                    start=float(raw.get("start") or 0.0),
                    end=float(raw.get("end") or 0.0),
                    text=text,
                    language=normalized_language,
                    confidence=(1.0 - float(no_speech)) if no_speech is not None else None,
                    metrics={
                        "avg_logprob": _optional_float(raw.get("avg_logprob")),
                        "no_speech_prob": _optional_float(no_speech),
                        "compression_ratio": _optional_float(raw.get("compression_ratio")),
                    },
                    source={
                        "backend": self.name,
                        "raw_segment_index": index,
                        "lemonade_model_id": self.model_id,
                    },
                )
            )
        if not segments and str(payload.get("text") or "").strip():
            raise RuntimeError(
                "Lemonade returned transcript text but no usable timestamped segments."
            )

        return ASRResult(
            segments=segments,
            language=normalized_language,
            duration=float(payload.get("duration") or 0.0),
            backend=self.name,
            model=model,
            device=readiness.device or self.expected_device,
            raw_output={
                "server_version": readiness.version,
                "base_url": self.base_url,
                "model_id": self.model_id,
                "task": payload.get("task"),
                "detected_language_probability": payload.get(
                    "detected_language_probability"
                ),
            },
        )


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
