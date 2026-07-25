#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-cpu}"
VERSION="${WHISPER_CPP_VERSION:-v1.9.1}"
case "$BACKEND" in cpu|vulkan) ;; *) echo "Usage: $0 [cpu|vulkan]" >&2; exit 2;; esac
for command in cmake git c++; do
    command -v "$command" >/dev/null || { echo "$command is required" >&2; exit 1; }
done
if [[ "$BACKEND" == vulkan ]] && ! command -v vulkaninfo >/dev/null && [[ -z "${VULKAN_SDK:-}" ]]; then
    echo "Vulkan development packages and tooling are required" >&2
    exit 1
fi

CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}/meeting-notes"
ARCH="$(uname -m)"; [[ "$ARCH" == aarch64 ]] && ARCH=arm64
INSTALL="$CACHE_BASE/runtimes/$VERSION/linux-$ARCH-$BACKEND"
WORK="$(mktemp -d)"
trap 'rm -rf -- "$WORK"' EXIT
git clone --branch "$VERSION" --depth 1 https://github.com/ggml-org/whisper.cpp.git "$WORK/source"
FLAGS=(-S "$WORK/source" -B "$WORK/build" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$INSTALL")
[[ "$BACKEND" == vulkan ]] && FLAGS+=(-DGGML_VULKAN=1)
cmake "${FLAGS[@]}"
cmake --build "$WORK/build" --config Release --parallel
cmake --install "$WORK/build" --config Release
"$INSTALL/bin/whisper-cli" --help >/dev/null
echo "Installed $VERSION $BACKEND runtime at $INSTALL"
