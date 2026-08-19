# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Builder: stable Rust base + nightly toolchain (std::simd) on Debian bookworm
# ─────────────────────────────────────────────────────────────────────────────
FROM rust:1.85-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# rust-toolchain.toml pins the Windows GNU nightly (dev host); install and force
# the Linux nightly so the build does NOT cross-compile to Windows.
RUN rustup toolchain install nightly --profile minimal --component rustfmt --component clippy
ENV RUSTUP_TOOLCHAIN=nightly

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/

RUN cargo build --release --workspace

# ─────────────────────────────────────────────────────────────────────────────
# Runtime: Debian bookworm-slim
#   - ca-certificates: HTTPS (HuggingFace downloads via rustls)
#   - curl: healthcheck
#   - ocl-icd-libopencl1: generic OpenCL ICD loader (vendor ICDs can be added
#     at run time, e.g. NVIDIA container toolkit / Intel / AMD runtimes).
#     Without an ICD Hayai runs CPU-only.
# ─────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/hayai-server /usr/local/bin/hayai-server
COPY --from=builder /build/target/release/hayai-cli /usr/local/bin/hayai-cli

# Non-root user (uid 1000). The models volume must be writable by this uid for
# `--hf` downloads (e.g. `chown 1000:1000 models` on Linux or a named volume).
RUN useradd --system --uid 1000 --create-home hayai \
    && mkdir -p /hayai/models \
    && chown -R hayai:hayai /hayai
WORKDIR /hayai
USER hayai

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

# All hayai-server options are configurable via HAYAI_* env vars (docker-compose
# or `docker run -e`). The defaults below listen on 0.0.0.0 so the container is
# reachable from outside. Explicit CLI args (e.g. `docker run image --port 8081`)
# still take precedence over env vars.
ENV HAYAI_HOST=0.0.0.0 \
    HAYAI_PORT=8080 \
    HAYAI_MODELS_DIR=/hayai/models

ENTRYPOINT ["hayai-server"]
