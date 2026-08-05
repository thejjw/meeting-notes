"""Isolated Qwen3 forced-alignment worker for the managed ROCm runtime.

This file intentionally imports no meeting_notes modules so the project-managed
native-Windows ROCm environment can execute it directly.
"""

# Qwen's processor methods are dynamic custom APIs supplied by Transformers.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false, reportMissingTypeStubs=false, reportPrivateImportUsage=false

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any


def _memory_sampler(stop: threading.Event, peak: list[int]) -> None:
    try:
        import psutil
    except ImportError:
        return
    process = psutil.Process()
    while not stop.wait(0.05):
        peak[0] = max(peak[0], process.memory_info().rss)


def _install_windows_rocm_transformers_compatibility(torch: Any) -> None:
    """Hide distributed APIs absent from AMD's native-Windows torch build."""
    if torch.distributed.is_available():
        return

    fsdp = types.ModuleType("transformers.distributed.fsdp")
    fsdp.is_fsdp_managed_module = lambda _module: False
    fsdp.is_fsdp_enabled = lambda: False
    fsdp.get_fsdp_ckpt_kwargs = lambda: {}
    fsdp.update_fsdp_plugin_peft = lambda *_args, **_kwargs: None
    sys.modules[fsdp.__name__] = fsdp

    sharding = types.ModuleType("transformers.distributed.sharding_utils")

    class DtensorShardOperation:
        pass

    def distributed_tensors_unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("Distributed tensors are unavailable in this PyTorch build.")

    sharding.DtensorShardOperation = DtensorShardOperation
    sharding._dtensor_from_local_like = distributed_tensors_unavailable
    sys.modules[sharding.__name__] = sharding


def _load_audio(path: str) -> Any:
    import soundfile

    samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=False)
    if sample_rate != 16_000:
        raise ValueError(f"Qwen alignment input must be 16 kHz; got {sample_rate} Hz.")
    if samples.ndim != 1:
        raise ValueError("Qwen alignment input must be mono audio.")
    return samples


def _run(request: dict[str, Any]) -> dict[str, Any]:
    import torch

    _install_windows_rocm_transformers_compatibility(torch)
    from transformers import AutoModelForTokenClassification, AutoProcessor

    if not torch.distributed.is_available():
        import transformers.core_model_loading as core_model_loading

        class UnavailableDTensor:
            pass

        core_model_loading.DTensor = UnavailableDTensor

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("ROCm PyTorch cannot access an AMD GPU.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dtype = torch.bfloat16

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(request["aligner_path"], local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        request["aligner_path"],
        local_files_only=True,
        dtype=dtype,
        device_map="cuda:0",
    )
    if request.get("torch_compile"):
        model.compile()
    load_seconds = time.perf_counter() - load_started

    audio_paths = list(request["audio_paths"])
    transcripts = list(request["transcripts"])
    languages = list(request["languages"])
    if not (len(audio_paths) == len(transcripts) == len(languages)):
        raise ValueError("Alignment request arrays must have the same length.")

    results: list[dict[str, Any]] = []
    for audio_path, transcript, language in zip(
        audio_paths, transcripts, languages, strict=True
    ):
        audio = _load_audio(str(audio_path))
        align_started = time.perf_counter()
        words: list[dict[str, Any]] = []
        transcript = str(transcript).strip()
        if transcript:
            inputs, word_lists = processor.prepare_forced_aligner_inputs(
                audio=audio,
                transcript=transcript,
                language=str(language),
            )
            inputs = inputs.to(model.device, model.dtype)
            with torch.inference_mode():
                outputs = model(**inputs)
            decoded = processor.decode_forced_alignment(
                logits=outputs.logits,
                input_ids=inputs["input_ids"],
                word_lists=word_lists,
                timestamp_token_id=model.config.timestamp_token_id,
            )[0]
            words = [
                {
                    "text": str(item["text"]),
                    "start": float(item["start_time"]),
                    "end": float(item["end_time"]),
                }
                for item in decoded
            ]
        results.append(
            {
                "words": words,
                "alignment_seconds": time.perf_counter() - align_started,
            }
        )

    return {
        "results": results,
        "metrics": {
            "aligner_load_seconds": load_seconds,
            "peak_ram_bytes": 0,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device_name": torch.cuda.get_device_name(0),
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: qwen3_aligner_worker.py REQUEST.json RESULT.json", file=sys.stderr)
        return 2
    request_path, result_path = map(Path, sys.argv[1:])
    stop = threading.Event()
    peak = [0]
    sampler = threading.Thread(target=_memory_sampler, args=(stop, peak), daemon=True)
    sampler.start()
    try:
        payload = _run(json.loads(request_path.read_text(encoding="utf-8")))
        payload["metrics"]["peak_ram_bytes"] = peak[0]
        result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as error:
        result_path.write_text(
            json.dumps(
                {
                    "error": f"{type(error).__name__}: {error}\n"
                    f"{traceback.format_exc()[-4000:]}"
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raise
    finally:
        stop.set()
        sampler.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
