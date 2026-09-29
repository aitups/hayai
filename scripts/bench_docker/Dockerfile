# syntax=docker/dockerfile:1
#
# Linux CPU-only bench image: hayai (nightly) + llama.cpp (master), same GGUFs.
#
# Goal: measure tok/s and RAM on a real Linux kernel so the streaming path uses
# io_uring (not the Windows FileIo fallback), and prove the disk is not the
# bottleneck. No GPU: the comparison is disk + RAM + CPU only.
#
# Build context is the repo root (see scripts/bench_docker.sh); `.dockerignore`
# drops models/ and target/, so the context stays small.
#
#   docker build -f scripts/bench_docker/Dockerfile -t hayai-bench-linux .

# ── hayai builder ────────────────────────────────────────────────────────────
FROM rust:1.85-bookworm AS hayai-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# The pinned toolchain in rust-toolchain.toml is Windows-GNU; force Linux nightly.
RUN rustup toolchain install nightly --profile minimal --component rustfmt --component clippy
ENV RUSTUP_TOOLCHAIN=nightly
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
RUN cargo build --release --workspace

# ── llama.cpp builder (CPU) ──────────────────────────────────────────────────
FROM debian:bookworm AS llama-builder
ARG LLAMA_REF=master
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "$LLAMA_REF" https://github.com/ggml-org/llama.cpp /llama.cpp
WORKDIR /llama.cpp
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)"

# ── runtime ──────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 time \
    && rm -rf /var/lib/apt/lists/*

COPY --from=hayai-builder /build/target/release/hayai-cli /usr/local/bin/hayai-cli
COPY --from=llama-builder /llama.cpp/build/bin/llama-bench /usr/local/bin/llama-bench
COPY --from=llama-builder /llama.cpp/build/bin/llama-cli   /usr/local/bin/llama-cli
# llama.cpp ships shared libs (libllama.so, libggml*.so); copy them next to the bins.
COPY --from=llama-builder /llama.cpp/build/bin/*.so* /usr/local/lib/
ENV LD_LIBRARY_PATH=/usr/local/lib

COPY scripts/bench_docker/compare_inside.sh /usr/local/bin/compare_inside.sh
COPY scripts/bench_docker/models.txt /usr/local/share/bench_models.txt
RUN chmod +x /usr/local/bin/compare_inside.sh

# GGUF directory is mounted read-only at run time.
ENV MODELS_DIR=/models
WORKDIR /work
ENTRYPOINT ["/usr/local/bin/compare_inside.sh"]
