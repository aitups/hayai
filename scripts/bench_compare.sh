#!/usr/bin/env bash
# Multi-metric Hayai vs llama.cpp compare (tok/s + e2e + memory).
set -euo pipefail

MODEL="${1:?usage: $0 <model.gguf> [prompt] [max_tokens]}"
PROMPT="${2:-Explain streaming weight inference in two sentences.}"
N="${3:-128}"
DEVICE="${DEVICE:-auto}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GGUF_BYTES=$(wc -c <"$MODEL" | tr -d ' ')
GGUF_GIB=$(awk -v b="$GGUF_BYTES" 'BEGIN{printf "%.3f", b/1024/1024/1024}')

echo "=== hayai plan ==="
cargo run -q -p hayai-cli -- plan --model "$MODEL" || {
  echo "BLOCKED: unknown_op — see plan output"
  exit 2
}

echo "=== hayai bench-generate ==="
RSS0=$(grep VmRSS /proc/self/status 2>/dev/null | awk '{print $2}' || echo 0)
START=$(date +%s.%N)
cargo run -q -p hayai-cli --release -- bench-generate \
  --model "$MODEL" --prompt "$PROMPT" --max-tokens "$N" --device "$DEVICE" \
  | tee /tmp/hayai_bench_out.txt
END=$(date +%s.%N)
E2E=$(awk -v s="$START" -v e="$END" 'BEGIN{printf "%.3f", e-s}')

echo "=== summary (Hayai) ==="
echo "gguf_gib=$GGUF_GIB e2e_wall_s=$E2E"
grep -E 'Decode tok/s|Prefill|RSS|Hayai-owned|Budget|Wall time|Overlap' /tmp/hayai_bench_out.txt || true

if command -v llama-bench >/dev/null 2>&1; then
  echo "=== llama-bench ==="
  llama-bench -m "$MODEL" -p 512 -n "$N" || true
elif command -v llama-cli >/dev/null 2>&1; then
  echo "=== llama-cli e2e (time -v) ==="
  /usr/bin/time -v llama-cli -m "$MODEL" -p "$PROMPT" -n "$N" -c 4096 --temp 0 2>&1 | tee /tmp/llama_bench_out.txt || true
else
  echo "llama.cpp binaries not in PATH — install llama-bench/llama-cli to complete compare"
fi

echo "Done. Paste rows into implementation_plan.md"
