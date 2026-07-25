FROM ubuntu:24.04 AS builder
ARG DEBIAN_FRONTEND=noninteractive
ARG WHISPER_CPP_VERSION=v1.9.1
ARG WHISPER_BACKEND=cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake git libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN git clone --branch "${WHISPER_CPP_VERSION}" --depth 1 https://github.com/ggml-org/whisper.cpp.git source
RUN test "$WHISPER_BACKEND" = cpu || test "$WHISPER_BACKEND" = vulkan
RUN if [ "$WHISPER_BACKEND" = vulkan ]; then flag=-DGGML_VULKAN=1; else flag=-DGGML_VULKAN=0; fi; \
    cmake -S source -B output -DCMAKE_BUILD_TYPE=Release "$flag" -DCMAKE_INSTALL_PREFIX=/opt/whisper \
    && cmake --build output --config Release --parallel \
    && cmake --install output --config Release

FROM ubuntu:24.04
ARG DEBIAN_FRONTEND=noninteractive
ARG WHISPER_BACKEND=cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/whisper /opt/whisper
ENV LD_LIBRARY_PATH="/opt/whisper/lib"
RUN /opt/whisper/bin/whisper-cli --help >/dev/null
ENV PATH="/opt/whisper/bin:${PATH}"
ENTRYPOINT ["whisper-cli"]
