#!/bin/bash
# Build whisper.cpp for CPU on Linux/macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.whisper-build"

echo "=== Building whisper.cpp for CPU ==="

# Check dependencies
for cmd in cmake g++ git; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd not found. Please install it."
        exit 1
    fi
done

# Clone or update whisper.cpp
if [ -d "$BUILD_DIR/whisper.cpp" ]; then
    echo "Updating whisper.cpp..."
    cd "$BUILD_DIR/whisper.cpp"
    git pull
else
    echo "Cloning whisper.cpp..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git
fi

cd "$BUILD_DIR/whisper.cpp"

# Build
echo "Building..."
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PROJECT_DIR/.whisper-install"

cmake --build build --config Release -j$(nproc)

# Install locally
cmake --install build

echo ""
echo "=== Build complete ==="
echo "whisper-cli installed to: $PROJECT_DIR/.whisper-install/bin/whisper-cli"
echo ""
echo "Add to PATH:"
echo "  export PATH=\"$PROJECT_DIR/.whisper-install/bin:\$PATH\""
