#!/usr/bin/env bash
# Runs inside the bench image (scripts/bench_docker/Dockerfile).
#
# For each GGUF in MODELS_DIR: measures prefill and decode tok/s plus peak RSS
# for hayai (streaming, CPU) and llama.cpp (CPU), then prints a markdown table.
#
#   docker run --rm -v /path/to/models:/models:ro hayai-bench-linux
#
# Tunables (env): MODELS_DIR, N (decode tokens), PP (llama-bench prefill tokens),
# THREADS, MEMORY_STRATEGY, MODELS (space-separated basenames to override).
set -uo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
N="${N:-128}"
PP="${PP:-512}"
THREADS="${THREADS:-$(nproc)}"
MEMORY_STRATEGY="${MEMORY_STRATEGY:-minimal}"

# Fixed ~450-token prompt (raw, no chat template) so prefill is comparable.
DEFAULT_PROMPT='Streaming weight inference stores the model on disk and pages each layer weights through a small ping-pong buffer as the forward pass advances, instead of mapping the entire file into host memory. Every token re-reads the same bytes, so the design trades disk bandwidth for resident RAM: the working set stays bounded no matter how large the model is, at the cost of reading the weights again for each generated token. The critical questions are therefore how fast the storage can deliver a layer and how well that I/O overlaps with the attention and feed-forward compute. A well tuned engine keeps the read engine busy while the matrix multiplies run on the CPU or a co-processor, so wall time approaches the maximum of the two rather than their sum. This report compares such an engine against a conventional loader on the same four models, using the same quantization files, and reports both throughput and the peak resident set size measured by the operating system.'
PROMPT="${PROMPT:-$DEFAULT_PROMPT}"

MODELS_FILE="${MODELS_FILE:-/usr/local/share/bench_models.txt}"
DEFAULT_MODELS=(
  "SmolLM2-135M-Instruct-Q4_K_M.gguf"
  "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
  "OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
  "Qwen_Qwen3.5-4B-Q4_K_M.gguf"
)
if [ -n "${MODELS:-}" ]; then
  read -r -a SELECTED <<<"$MODELS"
elif [ -f "$MODELS_FILE" ]; then
  mapfile -t SELECTED < <(grep -v '^[[:space:]]*$' "$MODELS_FILE")
else
  SELECTED=("${DEFAULT_MODELS[@]}")
fi

export RAYON_NUM_THREADS="$THREADS"

echo "host: $(uname -srm)  kernels=$(nproc)  threads=$THREADS  n=$N  pp=$PP  strategy=$MEMORY_STRATEGY"
echo "llama.cpp: $(llama-bench --version 2>/dev/null | head -1)"
probe="$MODELS_DIR/${SELECTED[0]:-}"
if [ -f "$probe" ]; then
  backend=$(hayai-cli bench-generate --model "$probe" --prompt hi --max-tokens 1 --device cpu --raw 2>&1 | grep -oE 'WeightIo=[a-z_]+' | head -1)
  echo "hayai WeightIo: ${backend:-unknown}  (io_uring expected on Linux; Docker needs --security-opt seccomp=unconfined)"
fi
echo
echo "| model | gguf_gib | engine | prefill_tps | decode_tps | peak_rss_mib | hayai_owned_mib |"
echo "|---|---:|---|---:|---:|---:|---:|"

decode_tps() { awk -F': *' '/Decode tok\/s/{print $2; exit}' "$1"; }
prefill_dec() { awk -F': *' '/Prefill\+dec/{print $2; exit}' "$1"; }
prompt_toks() { awk -F': *' '/Prompt tokens/{print $2; exit}' "$1"; }
owned_peak() { grep -oP 'Hayai-owned:\s+peak \K[0-9.]+' "$1" | head -1; }
time_rss_mib() { awk '/Maximum resident set size/{printf "%d", $NF/1024}' "$1"; }

for f in "${SELECTED[@]}"; do
  m="$MODELS_DIR/$f"
  if [ ! -f "$m" ]; then
    echo "| $f | - | MISSING | - | - | - | - |"
    continue
  fi
  gz=$(awk -v b="$(stat -c %s "$m")" 'BEGIN{printf "%.2f", b/1073741824}')

  # ── hayai: prefill (1 token) then decode (N tokens) ────────────────────────
  /usr/bin/time -v hayai-cli bench-generate --model "$m" --prompt "$PROMPT" \
      --max-tokens 1 --device cpu --raw --memory-strategy "$MEMORY_STRATEGY" \
      >/tmp/h_pre.txt 2>/tmp/h_pre.time
  h_pre=$(prefill_dec /tmp/h_pre.txt)
  h_pt=$(prompt_toks /tmp/h_pre.txt)

  /usr/bin/time -v hayai-cli bench-generate --model "$m" --prompt "$PROMPT" \
      --max-tokens "$N" --device cpu --raw --memory-strategy "$MEMORY_STRATEGY" \
      >/tmp/h_dec.txt 2>/tmp/h_dec.time
  h_dec=$(decode_tps /tmp/h_dec.txt)
  h_owned=$(owned_peak /tmp/h_dec.txt)
  h_rss=$(time_rss_mib /tmp/h_dec.time)

  printf '| %s | %s | hayai(%s) | %s | %s | %s | %s |\n' \
    "$f" "$gz" "$MEMORY_STRATEGY" "${h_pre:-?}" "${h_dec:-?}" "${h_rss:-?}" "${h_owned:-?}"
  echo "    hayai prompt_tokens=$h_pt  (prefill tps = Prefill+dec at N=1)"

  # ── llama.cpp: prefill/decode tok/s + RSS (llama-bench under /usr/bin/time) ─
  pp=""; tg=""; l_rss=""
  if command -v llama-bench >/dev/null 2>&1; then
    /usr/bin/time -v llama-bench -m "$m" -p "$PP" -n "$N" -t "$THREADS" \
      >/tmp/l.txt 2>&1
    pp=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^pp[0-9]+$/){for(j=i+1;j<=NF;j++) if($j ~ /^[0-9]+(\.[0-9]+)?$/){print $j; exit}}}' /tmp/l.txt)
    tg=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^tg[0-9]+$/){for(j=i+1;j<=NF;j++) if($j ~ /^[0-9]+(\.[0-9]+)?$/){print $j; exit}}}' /tmp/l.txt)
    l_rss=$(time_rss_mib /tmp/l.txt)
  fi

  printf '| %s | %s | llama.cpp | %s | %s | %s | - |\n' \
    "$f" "$gz" "${pp:-?}" "${tg:-?}" "${l_rss:-?}"
done

echo
echo "_prefill_tps: hayai = \`Prefill+dec\` at max-tokens=1; llama.cpp = llama-bench pp$PP._"
echo "_decode_tps: hayai = \`Decode tok/s\`; llama.cpp = llama-bench tg$N._"
echo "_peak_rss_mib: /usr/bin/time -v \`Maximum resident set size\`._"
