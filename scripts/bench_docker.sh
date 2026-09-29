#!/usr/bin/env bash
# Build the Linux bench image and run hayai vs llama.cpp on the mounted models.
#
#   scripts/bench_docker.sh
#   STAGE=1 scripts/bench_docker.sh                      # copy models to a Linux volume first
#   MODELS="Qwen2.5-7B-Instruct-Q4_K_M.gguf" N=64 scripts/bench_docker.sh
#
# Requires Docker. On Docker Desktop for Windows run this from WSL or Git Bash.
#
# io_uring is blocked by Docker's default seccomp profile (io_uring_setup -> EPERM),
# so every run passes --security-opt seccomp=unconfined. The mounted models are read
# through the host filesystem: on Windows that is WSL2 drvfs (~200 MB/s), which is far
# below a real SSD — set STAGE=1 to copy the GGUFs onto the Linux volume (ext4) so the
# I/O numbers reflect the disk instead of the mount.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT/models}"
IMAGE="${IMAGE:-hayai-bench-linux}"
N="${N:-128}"
PP="${PP:-512}"
OUT_DIR="${OUT_DIR:-$ROOT/bench_results}"
SECOPT=(--security-opt seccomp=unconfined)
mkdir -p "$OUT_DIR"

docker build -f "$ROOT/scripts/bench_docker/Dockerfile" -t "$IMAGE" "$ROOT"

if [ "${STAGE:-0}" = "1" ]; then
  VOLUME="${VOLUME:-hayai-bench-models}"
  docker volume create "$VOLUME" >/dev/null
  echo "staging models into volume '$VOLUME' (native Linux fs)..."
  docker run --rm --entrypoint bash -v "$(cd "$MODELS_DIR" && pwd):/src:ro" -v "$VOLUME:/models" \
    "$IMAGE" -c 'set -e; while read -r f; do [ -z "$f" ] && continue; [ -f "/models/$f" ] || cp "/src/$f" /models/; done < /usr/local/share/bench_models.txt; echo "staged"'
  MOUNT="$VOLUME:/models"
else
  MOUNT="$(cd "$MODELS_DIR" && pwd):/models:ro"
fi

run_args=(--rm "${SECOPT[@]}" -v "$MOUNT" -e "N=$N" -e "PP=$PP")
[ -n "${THREADS:-}" ] && run_args+=(-e "THREADS=$THREADS")
[ -n "${MODELS:-}" ] && run_args+=(-e "MODELS=$MODELS")
[ -n "${MEMORY_STRATEGY:-}" ] && run_args+=(-e "MEMORY_STRATEGY=$MEMORY_STRATEGY")

OUT="$OUT_DIR/bench_linux_$(date +%Y%m%d_%H%M%S).md"
docker run "${run_args[@]}" "$IMAGE" | tee "$OUT"
echo
echo "saved: $OUT"
