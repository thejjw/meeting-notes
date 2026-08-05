"""GPU-only Qwen3-ASR GGUF backend served by AMD Lemonade."""

# JSON payloads returned by Lemonade are runtime-validated before use.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import httpx

from meeting_notes.asr.base import ASRBackend, ASRReadiness, ASRResult, ASRSegment

if TYPE_CHECKING:
    from collections.abc import Callable

QWEN_GGUF_CHECKPOINT = "unslothai/Qwen3-ASR-1.7B-GGUF:Q8_0"
QWEN_GGUF_MODEL_ID = "Qwen3-ASR-1.7B-GGUF-Q8_0"
QWEN_ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

# Qwen3-ASR recognizes these languages. The separate forced aligner is the
# narrower boundary for this backend because meeting-notes requires timestamps.
QWEN_ASR_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
QWEN_ALIGNER_LANGUAGE_CODES = frozenset(
    {"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}
)
_LANGUAGE_CODES = {name.casefold(): code for code, name in QWEN_ASR_LANGUAGES.items()}
_QWEN_OUTPUT = re.compile(r"^language\s+([^<]+)<asr_text>(.*)$", re.DOTALL)
_SENTENCE_END = re.compile("[.!?\u3002\uff01\uff1f]$")
_PUNCTUATION = re.compile("\\s+([,.;:!?\uff0c\u3002\uff1b\uff1a\uff01\uff1f])")


def managed_aligner_dir(cache_dir: Path, model_id: str = QWEN_ALIGNER_MODEL_ID) -> Path:
    """Return the stable project-local directory for the forced aligner."""
    return cache_dir.resolve() / "qwen3-asr" / model_id.replace("/", "--")


def legacy_native_asr_dir(cache_dir: Path) -> Path:
    """Return the exact obsolete native-ASR directory eligible for cleanup."""
    return cache_dir.resolve() / "qwen3-asr" / "Qwen--Qwen3-ASR-1.7B-hf"


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def _segments_from_words(
    words: list[dict[str, Any]], language: str, *, backend: str = "qwen3_asr_lemonade"
) -> list[ASRSegment]:
    segments: list[ASRSegment] = []
    group: list[dict[str, Any]] = []

    def flush() -> None:
        if not group:
            return
        text = _PUNCTUATION.sub(r"\1", " ".join(str(item["text"]).strip() for item in group))
        segments.append(
            ASRSegment(
                id=f"seg-{len(segments):06d}",
                start=float(group[0]["start"]),
                end=float(group[-1]["end"]),
                text=text.strip(),
                language=language or None,
                source={
                    "backend": backend,
                    "aligned_words": len(group),
                    "aligned_word_timestamps": [dict(item) for item in group],
                },
            )
        )
        group.clear()

    previous_end: float | None = None
    for word in words:
        start = float(word["start"])
        end = float(word["end"])
        if end < start or (previous_end is not None and start < previous_end - 0.05):
            raise RuntimeError("Qwen forced aligner returned non-monotonic timestamps.")
        if group and previous_end is not None and start - previous_end >= 0.8:
            flush()
        group.append(word)
        if (
            _SENTENCE_END.search(str(word["text"]).strip())
            or end - float(group[0]["start"]) >= 15.0
        ):
            flush()
        previous_end = end
    flush()
    return segments


class Qwen3ASRLemonadeBackend(ASRBackend):
    """Transcribe with Lemonade/llama.cpp GPU and align with native ROCm."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:13305",
        model_id: str = QWEN_GGUF_MODEL_ID,
        checkpoint: str = QWEN_GGUF_CHECKPOINT,
        api_key_env: str = "LEMONADE_API_KEY",
        llamacpp_backend: Literal["vulkan", "rocm"] = "vulkan",
        python_executable: str,
        aligner_path: Path,
        ctx_size: int = 8192,
        max_new_tokens: int = 4096,
        torch_compile: bool = False,
        connect_timeout_seconds: float = 5.0,
        provisioning_timeout_seconds: float = 3600.0,
        transcription_timeout_seconds: float = 7200.0,
        worker_timeout_seconds: float = 7200.0,
        environment: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.api_key_env = api_key_env
        self.llamacpp_backend = llamacpp_backend
        self.python_executable = python_executable
        self.aligner_path = aligner_path
        self.ctx_size = ctx_size
        self.max_new_tokens = max_new_tokens
        self.torch_compile = torch_compile
        self.connect_timeout_seconds = connect_timeout_seconds
        self.provisioning_timeout_seconds = provisioning_timeout_seconds
        self.transcription_timeout_seconds = transcription_timeout_seconds
        self.worker_timeout_seconds = worker_timeout_seconds
        self.environment = environment
        self._version = ""
        self._resolved_model_id = model_id

    @property
    def name(self) -> str:
        return "qwen3_asr_lemonade"

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

    def _post_json(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
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
            detail = ""
            if isinstance(error, httpx.HTTPStatusError):
                detail = f" ({error.response.text[-1000:]})"
            raise RuntimeError(
                f"Lemonade request failed at {self.base_url}{path}: {error}{detail}"
            ) from error
        if not isinstance(body, dict):
            raise RuntimeError(f"Lemonade returned a non-object response for {path}.")
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
        if not self._version:
            try:
                self._version = str(self._get_json("/v1/health").get("version") or "unknown")
            except RuntimeError:
                self._version = "unavailable"
        return self._version

    def list_models(self, *, show_all: bool = True) -> list[dict[str, Any]]:
        suffix = "?show_all=true" if show_all else ""
        data = self._get_json(f"/v1/models{suffix}").get("data", [])
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def model_info(self) -> dict[str, Any] | None:
        models = self.list_models(show_all=True)
        for item in models:
            if item.get("checkpoint") == self.checkpoint:
                self._resolved_model_id = str(item.get("id") or self.model_id)
                return item
        for item in models:
            if item.get("id") == self.model_id and item.get("checkpoint") in {
                None,
                self.checkpoint,
            }:
                self._resolved_model_id = str(item.get("id") or self.model_id)
                return item
        return None

    def variant_info(self) -> dict[str, Any]:
        """Inspect Lemonade's remote GGUF variant metadata for storage reporting."""
        return self._get_json(
            f"/api/v1/pull/variants?checkpoint={quote(self.checkpoint, safe='')}",
            timeout=60.0,
        )

    def pull_model(self, *, progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        """Pull the exact remote checkpoint, registering it when necessary."""
        info = self.model_info()
        if info and info.get("downloaded"):
            return
        model_name = str(info.get("id")) if info else self.checkpoint
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/v1/pull",
                json={"model_name": model_name, "stream": True},
                headers=self._headers(),
                timeout=self._timeout(self.provisioning_timeout_seconds),
            ) as response:
                response.raise_for_status()
                event = ""
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event = line.partition(":")[2].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line.partition(":")[2].strip())
                        if isinstance(data, dict):
                            data["event"] = event
                            if progress:
                                progress(data)
                            if event == "error":
                                raise RuntimeError(str(data.get("error") or "download failed"))
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"Failed to pull Lemonade checkpoint '{self.checkpoint}': {error}"
            ) from error
        refreshed = self.model_info()
        if not refreshed or not refreshed.get("downloaded"):
            raise RuntimeError(f"Lemonade did not report '{self.checkpoint}' as downloaded.")

    def _loaded_model(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        health = self._get_json("/v1/health")
        self._version = str(health.get("version") or "unknown")
        loaded = health.get("all_models_loaded", [])
        candidates = loaded if isinstance(loaded, list) else []
        model = next(
            (
                item
                for item in candidates
                if isinstance(item, dict)
                and (
                    item.get("model_name") == self._resolved_model_id
                    or item.get("checkpoint") == self.checkpoint
                )
            ),
            None,
        )
        return model, health

    def _is_expected_backend_loaded(self, loaded: dict[str, Any] | None) -> bool:
        if not loaded:
            return False
        options = loaded.get("recipe_options")
        backend = options.get("llamacpp_backend") if isinstance(options, dict) else None
        return (
            bool(loaded.get("backend_alive", True))
            and str(loaded.get("status")) in {"ready", "in_use"}
            and str(loaded.get("device")) == "gpu"
            and backend == self.llamacpp_backend
        )

    @property
    def accelerator_device(self) -> str:
        """Describe the two accelerators used by the timestamped ASR path."""
        return "rocm" if self.llamacpp_backend == "rocm" else f"{self.llamacpp_backend}+rocm"

    def check_readiness(
        self,
        *,
        model: str = "",
        expected_device: str = "",
        allow_provision: bool = False,
    ) -> ASRReadiness:
        del model, expected_device
        metadata: dict[str, Any] = {
            "base_url": self.base_url,
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "llamacpp_backend": self.llamacpp_backend,
        }
        if not Path(self.python_executable).is_file():
            return ASRReadiness(
                False, f"Qwen alignment runtime is missing: {self.python_executable}"
            )
        if not (self.aligner_path / "config.json").is_file():
            return ASRReadiness(False, f"Qwen forced aligner is missing: {self.aligner_path}")
        if not self.is_available():
            return ASRReadiness(
                False,
                f"Lemonade Server is not reachable at {self.base_url}.",
                device=self.accelerator_device,
                metadata=metadata,
            )
        try:
            info = self.model_info()
            if info is None:
                return ASRReadiness(
                    allow_provision,
                    (
                        f"Lemonade checkpoint '{self.checkpoint}' can be provisioned."
                        if allow_provision
                        else f"Lemonade checkpoint '{self.checkpoint}' is not registered."
                    ),
                    self.get_version(),
                    self.accelerator_device,
                    metadata,
                )
            metadata.update(
                {
                    "model_id": self._resolved_model_id,
                    "downloaded": bool(info.get("downloaded")),
                    "size_gb": info.get("size"),
                }
            )
            if not info.get("downloaded"):
                return ASRReadiness(
                    allow_provision,
                    f"Lemonade checkpoint '{self.checkpoint}' is not downloaded.",
                    self.get_version(),
                    self.accelerator_device,
                    metadata,
                )
            loaded, health = self._loaded_model()
            metadata["loaded"] = loaded is not None
            if loaded is None:
                return ASRReadiness(
                    True,
                    "Qwen3-ASR GGUF is downloaded and will be loaded with "
                    f"llama.cpp {self.llamacpp_backend} when used.",
                    str(health.get("version") or "unknown"),
                    self.accelerator_device,
                    metadata,
                )
            metadata["recipe_options"] = loaded.get("recipe_options")
            if not self._is_expected_backend_loaded(loaded):
                return ASRReadiness(
                    False,
                    "Qwen3-ASR is loaded by Lemonade, but not with llama.cpp "
                    f"{self.llamacpp_backend}.",
                    str(health.get("version") or "unknown"),
                    str(loaded.get("device") or "unknown"),
                    metadata,
                )
            return ASRReadiness(
                True,
                "Qwen3-ASR GGUF is ready through Lemonade llama.cpp "
                f"{self.llamacpp_backend}; forced alignment uses ROCm.",
                str(health.get("version") or "unknown"),
                self.accelerator_device,
                metadata,
            )
        except RuntimeError as error:
            return ASRReadiness(
                False, str(error), self.get_version(), self.accelerator_device, metadata
            )

    def load_model(self) -> ASRReadiness:
        info = self.model_info()
        if not info or not info.get("downloaded"):
            raise RuntimeError(f"Lemonade checkpoint '{self.checkpoint}' is not downloaded.")
        loaded, _health = self._loaded_model()
        if self._is_expected_backend_loaded(loaded):
            return self.check_readiness()
        self._post_json(
            "/v1/load",
            {
                "model_name": self._resolved_model_id,
                "llamacpp_backend": self.llamacpp_backend,
                "ctx_size": self.ctx_size,
                "save_options": False,
            },
            timeout=self.provisioning_timeout_seconds,
        )
        deadline = time.monotonic() + min(self.provisioning_timeout_seconds, 180.0)
        while time.monotonic() < deadline:
            loaded, _health = self._loaded_model()
            if self._is_expected_backend_loaded(loaded):
                return self.check_readiness()
            time.sleep(0.5)
        raise RuntimeError(
            f"Lemonade did not load Qwen3-ASR with llama.cpp {self.llamacpp_backend}."
        )

    @staticmethod
    def _normalize_language(language: str) -> tuple[str | None, str | None]:
        requested = language.strip().casefold()
        if requested in {"", "auto"}:
            return None, None
        code = requested if requested in QWEN_ASR_LANGUAGES else _LANGUAGE_CODES.get(requested)
        if code is None:
            supported = ", ".join(sorted(QWEN_ALIGNER_LANGUAGE_CODES))
            raise ValueError(
                f"Unknown Qwen language '{language}'. Timestamped meeting output supports "
                f"auto or one of: {supported}."
            )
        if code not in QWEN_ALIGNER_LANGUAGE_CODES:
            raise ValueError(
                f"Qwen3-ASR can transcribe {QWEN_ASR_LANGUAGES[code]}, but its forced "
                "aligner cannot timestamp that language. This meeting backend requires "
                "timestamps for subtitles and speaker assignment."
            )
        return code, QWEN_ASR_LANGUAGES[code]

    def _transcribe_text(
        self, audio_path: Path, *, language_code: str, initial_prompt: str | None
    ) -> tuple[str, str, dict[str, Any], float]:
        _requested_code, language = self._normalize_language(language_code)
        content: list[dict[str, Any]] = [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
                    "format": "wav",
                },
            }
        ]
        if initial_prompt:
            content.append(
                {
                    "type": "text",
                    "text": f"Context for accurate transcription: {initial_prompt}",
                }
            )
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        payload: dict[str, Any] = {
            "model": self._resolved_model_id,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_new_tokens,
        }
        if language:
            messages.append({"role": "assistant", "content": f"language {language}<asr_text>"})
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        started = time.perf_counter()
        response = self._post_json(
            "/v1/chat/completions", payload, timeout=self.transcription_timeout_seconds
        )
        elapsed = time.perf_counter() - started
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("Lemonade Qwen response did not contain a completion choice.")
        message = choices[0].get("message")
        output = str(message.get("content") or "") if isinstance(message, dict) else ""
        match = _QWEN_OUTPUT.fullmatch(output.strip())
        if not match:
            raise RuntimeError("Lemonade Qwen response did not match the Qwen ASR output format.")
        output_language = match.group(1).strip()
        output_code = _LANGUAGE_CODES.get(output_language.casefold())
        if output_code is None:
            raise RuntimeError(f"Qwen returned unknown language label '{output_language}'.")
        if output_code not in QWEN_ALIGNER_LANGUAGE_CODES:
            raise RuntimeError(
                f"Qwen detected {output_language}, but its forced aligner cannot timestamp "
                "that language. Set asr.language to a supported language when the dominant "
                "language is known, or use a Whisper backend for this recording."
            )
        return match.group(2).strip(), output_code, response, elapsed

    def _align(
        self,
        audio_paths: list[Path],
        transcripts: list[str],
        language_codes: list[str],
    ) -> dict[str, Any]:
        worker = Path(__file__).with_name("qwen3_aligner_worker.py")
        with tempfile.TemporaryDirectory(prefix="qwen3-align-") as temporary:
            root = Path(temporary)
            request_path = root / "request.json"
            result_path = root / "result.json"
            request_path.write_text(
                json.dumps(
                    {
                        "audio_paths": [str(path.resolve()) for path in audio_paths],
                        "transcripts": transcripts,
                        "languages": [QWEN_ASR_LANGUAGES[code] for code in language_codes],
                        "aligner_path": str(self.aligner_path.resolve()),
                        "torch_compile": self.torch_compile,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [self.python_executable, str(worker), str(request_path), str(result_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self.environment,
                timeout=self.worker_timeout_seconds,
                check=False,
            )
            payload: dict[str, Any] = (
                json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
            )
            if completed.returncode or payload.get("error"):
                detail = payload.get("error") or completed.stderr or completed.stdout
                raise RuntimeError(f"Qwen forced aligner failed: {str(detail).strip()[-2000:]}")
            return payload

    def transcribe(self, audio_path: Path, **kwargs: Any) -> ASRResult:
        return self.transcribe_batch([audio_path], **kwargs)[0]

    def transcribe_batch(self, audio_paths: list[Path], **kwargs: Any) -> list[ASRResult]:
        if kwargs.get("task", "transcribe") != "transcribe":
            raise ValueError("Qwen3-ASR supports transcription only.")
        for path in audio_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Audio file not found: {path}")
            if path.suffix.lower() != ".wav":
                raise ValueError("Lemonade Qwen transcription requires normalized WAV files.")
        readiness = self.load_model()
        configured_language = str(kwargs.get("language") or "auto")
        initial_prompt = str(kwargs.get("initial_prompt") or "") or None

        transcripts: list[str] = []
        language_codes: list[str] = []
        asr_payloads: list[dict[str, Any]] = []
        asr_seconds: list[float] = []
        for path in audio_paths:
            transcript, language_code, payload, elapsed = self._transcribe_text(
                path,
                language_code=configured_language,
                initial_prompt=initial_prompt,
            )
            transcripts.append(transcript)
            language_codes.append(language_code)
            asr_payloads.append(payload)
            asr_seconds.append(elapsed)

        aligned = self._align(audio_paths, transcripts, language_codes)
        outputs = aligned.get("results")
        if not isinstance(outputs, list) or len(outputs) != len(audio_paths):
            raise RuntimeError("Qwen forced aligner returned an unexpected result count.")
        shared_metrics = aligned.get("metrics")
        if not isinstance(shared_metrics, dict):
            shared_metrics = {}

        results: list[ASRResult] = []
        for index, (path, transcript, language_code, output) in enumerate(
            zip(audio_paths, transcripts, language_codes, outputs, strict=True)
        ):
            if not isinstance(output, dict):
                raise RuntimeError("Qwen forced aligner returned an invalid result.")
            words = output.get("words")
            word_items = words if isinstance(words, list) else []
            segments = _segments_from_words(word_items, language_code, backend=self.name)
            if transcript and not segments:
                raise RuntimeError("Qwen forced aligner returned no timestamps for non-empty text.")
            usage = asr_payloads[index].get("usage")
            metrics = {
                **shared_metrics,
                "transcribe_seconds": sum(asr_seconds),
                "alignment_seconds": sum(
                    float(item.get("alignment_seconds") or 0.0)
                    for item in outputs
                    if isinstance(item, dict)
                ),
                "chunk_transcribe_seconds": asr_seconds[index],
                "chunk_alignment_seconds": output.get("alignment_seconds"),
                "input_count": len(outputs),
                "token_usage": usage if isinstance(usage, dict) else {},
            }
            results.append(
                ASRResult(
                    segments=segments,
                    language=language_code,
                    duration=_audio_duration(path),
                    backend=self.name,
                    model=self._resolved_model_id,
                    device=self.accelerator_device,
                    raw_output={
                        "metrics": metrics,
                        "transcript": transcript,
                        "server_version": readiness.version,
                        "base_url": self.base_url,
                        "checkpoint": self.checkpoint,
                        "lemonade_model_id": self._resolved_model_id,
                        "llamacpp_backend": self.llamacpp_backend,
                    },
                )
            )
        return results
