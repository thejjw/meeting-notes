"""Standalone JSON worker for the managed ROCm diarization environment."""

from __future__ import annotations

import json
import sys
import warnings
import wave
from pathlib import Path
from typing import Any

# Executing this file directly puts its directory first on sys.path, where the
# sibling pyannote.py would shadow the third-party pyannote package.
_WORKER_DIRECTORY = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == _WORKER_DIRECTORY:
    sys.path.pop(0)


def _read_pcm_wave(audio_path: Path) -> dict[str, object]:
    import torch

    with wave.open(str(audio_path), "rb") as stream:
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        sample_rate = stream.getframerate()
        compression = stream.getcomptype()
        frames = stream.readframes(stream.getnframes())
    if sample_width != 2 or compression != "NONE":
        raise RuntimeError(
            "Diarization expects an uncompressed 16-bit PCM WAV; "
            f"got sample_width={sample_width}, compression={compression}."
        )
    samples = torch.frombuffer(bytearray(frames), dtype=torch.int16).clone()
    waveform = samples.reshape(-1, channels).transpose(0, 1).to(torch.float32)
    waveform /= 32768.0
    return {"waveform": waveform, "sample_rate": sample_rate}


def run(request: dict[str, Any]) -> dict[str, Any]:
    """Execute one hybrid diarization request."""
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(?s)\s*torchcodec is not installed correctly.*",
            category=UserWarning,
            module=r"pyannote\.audio\.core\.io",
        )
        from pyannote.audio import Pipeline

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("The managed PyTorch runtime cannot access an AMD HIP device.")

    model_path = Path(str(request["model_path"]))
    audio_path = Path(str(request["audio_path"]))
    if not (model_path / "config.yaml").is_file():
        raise RuntimeError(f"Local diarization model is invalid: {model_path}")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = Pipeline.from_pretrained(str(model_path))
    inferences = getattr(pipeline, "_inferences", {})
    embedding = inferences.get("_embedding")
    segmentation = inferences.get("_segmentation")
    if embedding is None or segmentation is None:
        raise RuntimeError(
            "The pinned Community-1 pipeline no longer exposes the expected inference stages."
        )
    embedding.to(torch.device("cuda"))
    if str(getattr(segmentation, "device", "cpu")) != "cpu":
        raise RuntimeError("ROCm hybrid safety check failed: segmentation is not on CPU.")

    params: dict[str, int] = {}
    num_speakers = request.get("num_speakers")
    if num_speakers is not None:
        params["num_speakers"] = int(num_speakers)
    else:
        params["min_speakers"] = int(request.get("min_speakers", 2))
        max_speakers = request.get("max_speakers")
        if max_speakers is not None:
            params["max_speakers"] = int(max_speakers)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"std\(\): degrees of freedom is <= 0\..*",
            category=UserWarning,
        )
        output = pipeline(_read_pcm_wave(audio_path), **params)
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    standard = getattr(output, "speaker_diarization", None)
    annotation = (
        exclusive if request.get("use_exclusive", True) and exclusive is not None else standard
    )
    if annotation is None:
        annotation = output

    turns: list[dict[str, object]] = []
    speakers: set[str] = set()
    if hasattr(annotation, "itertracks"):
        tracks = ((turn, speaker) for turn, _, speaker in annotation.itertracks(yield_label=True))
    else:
        tracks = iter(annotation)
    for index, (turn, speaker) in enumerate(tracks):
        label = str(speaker)
        turns.append(
            {
                "turn_id": f"turn-{index:06d}",
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": label,
                "source": str(model_path),
            }
        )
        speakers.add(label)
    return {
        "turns": turns,
        "speakers": sorted(speakers),
        "backend": "pyannote",
        "model": str(model_path),
        "device": "rocm-hybrid",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise TypeError("Worker request must be a JSON object.")
        print(json.dumps(run(request), ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"ROCm diarization worker failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
