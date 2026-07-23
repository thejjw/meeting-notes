# Multi-stage build: compile whisper.cpp, then copy binary out
#
# Usage:
#   docker build -t whisper-builder -f Dockerfile .
#   docker run --rm -v $(pwd)/../../bin:/out whisper-builder
#
# This copies whisper-cli and libraries to ./bin/ on the host.
# After that, use the native binary directly — no Docker needed.

FROM ubuntu:22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone whisper.cpp
RUN git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git

WORKDIR /build/whisper.cpp

# Build with CPU support
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    && cmake --build build --config Release -j$(nproc)

# Output stage — just the binaries
FROM ubuntu:22.04 AS output

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=builder /build/whisper.cpp/build/bin/lib*.so* /usr/local/lib/
RUN ldconfig

# Default: copy binaries to /out volume mount
RUN mkdir -p /out
VOLUME ["/out"]

CMD ["sh", "-c", "cp /usr/local/bin/whisper-cli /out/ && cp /usr/local/lib/lib*.so* /out/ && echo 'Binaries copied to /out/'"]
