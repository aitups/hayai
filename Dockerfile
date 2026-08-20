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
#   - clinfo: OpenCL platform/device debug utility (run `docker exec ... clinfo -l`)
#   - ocl-icd-libopencl1: generic OpenCL ICD loader (libOpenCL.so.1).
#
# OpenCL uses the ICD (Installable Client Driver) model: the loader only
# reports a platform when a vendor driver is REGISTERED through a *.icd file in
# /etc/OpenCL/vendors/ that points to the vendor's OpenCL runtime library
# (e.g. libnvidia-opencl.so.1 for NVIDIA). We ship the registration file here;
# the vendor library itself must be made available at run time:
#   - native Linux + nvidia-container-toolkit mounts libnvidia-opencl.so.1
#     into the container automatically (the bare filename in nvidia.icd is then
#     resolved through the loader search path), or
#   - mount the NVIDIA driver userspace manually (see docker-compose.gpu.yml).
#
# IMPORTANT: passing the CUDA interface to a container (Docker Desktop/WSL2
# `--gpus all`) does NOT provide OpenCL — CUDA and OpenCL are separate APIs
# with separate userspace libraries. Without a registered vendor ICD the
# loader returns CL_PLATFORM_NOT_FOUND_KHR and Hayai runs CPU-only.
# ─────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        clinfo \
        ocl-icd-libopencl1 \
    && mkdir -p /etc/OpenCL/vendors \
    && printf 'libnvidia-opencl.so.1\n' > /etc/OpenCL/vendors/nvidia.icd \
    && rm -rf /var/lib/apt/lists/*

# Where a vendor OpenCL runtime may be injected at run time:
#   - /usr/lib/wsl/lib   -> NVIDIA driver userspace mounted from the WSL2 host
#                           (Docker Desktop on Windows, if the driver ships the
#                           OpenCL ICD).
#   - /usr/local/nvidia  -> nvidia-container-toolkit injection point.
# LD_LIBRARY_PATH is additive to the standard loader search path, so pointing
# at non-existent directories is harmless.
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/nvidia/lib64

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
