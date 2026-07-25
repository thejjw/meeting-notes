"""Pinned, verified external artifacts used by meeting-notes."""

from __future__ import annotations

WHISPER_CPP_VERSION = "v1.9.1"
WHISPER_CPP_REVISION = "f049fff95a089aa9969deb009cdd4892b3e74916"
WHISPER_CPP_REPOSITORY = "https://github.com/ggml-org/whisper.cpp.git"

CPU_ARCHIVES: dict[tuple[str, str], tuple[str, str]] = {
    ("windows", "x86_64"): (
        "whisper-bin-x64.zip",
        "7d8be46ecd31828e1eb7a2ecdd0d6b314feafd82163038ab6092594b0a063539",
    ),
    ("windows", "x86"): (
        "whisper-bin-Win32.zip",
        "be1ea26c9665f1165a2f3afb64f24476c09ba7da479c844bf33ef2870d47c954",
    ),
    ("linux", "x86_64"): (
        "whisper-bin-ubuntu-x64.tar.gz",
        "f3bf3b4369a99b54665b0f19b88483b30de27f25963b0414235dea03198515c5",
    ),
    ("linux", "arm64"): (
        "whisper-bin-ubuntu-arm64.tar.gz",
        "e0b66cd551ff6f2a28fabe3c6e89691eea037bb76833493abb9a71ca788994b3",
    ),
}

# SHA-256 values are the official LFS/Xet object digests published by
# https://huggingface.co/ggerganov/whisper.cpp.
MODEL_ARTIFACTS: dict[str, dict[str, object]] = {
    "tiny": {
        "sha256": "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        "size": 77_713_815,
    },
    "base": {
        "sha256": "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        "size": 147_964_211,
    },
    "small": {
        "sha256": "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        "size": 487_601_967,
    },
    "medium": {
        "sha256": "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
        "size": 1_533_763_059,
    },
    "large-v3": {
        "sha256": "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        "size": 3_095_033_483,
    },
    "large-v3-turbo": {
        "sha256": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        "size": 1_624_555_275,
    },
}


def model_url(name: str) -> str:
    """Return the official GGML model download URL."""
    return f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"

