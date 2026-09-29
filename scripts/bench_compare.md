# Hayai vs llama.cpp comparative harness

## Metrics (required)

| Family | Fields |
| --- | --- |
| Throughput | `prefill_tps`, `decode_tps` |
| Memory | `hayai_owned_peak`, `os_rss_peak`, `gguf_gib`, ratios vs GGUF |
| End-to-end | `e2e_s` (load→N tokens), `ttft_s` / prefill wall |

## Models (product suite)

Dev smoke (SmolLM) is **not** the comparative. Use:

1. Qwen3.5-4B Q4_K_M GGUF (text-only)
2. Gemma 4 E4B Q4_K_M GGUF
3. HRM-Text-1B Q5_K_M / Q6_K GGUF

Same file for Hayai and llama.cpp. First run `hayai plan --model <gguf>` — fails only on **unknown layer op**.

## Protocol

- Prompt ≈ 256–512 tokens after chat template (document the string)
- `N=128` new tokens, greedy / temp=0
- Matching context length (`-c` in llama.cpp)
- Prefer Linux/WSL for `io_uring` + `/proc` RSS

## Commands

```bash
# Plan (metadata → ops → HW strategy)
cargo run -p hayai-cli -- plan --model models/foo.gguf

# Hayai bench
cargo run -p hayai-cli --release -- bench-generate --model models/foo.gguf \
  --prompt "..." --max-tokens 128 --device auto

# llama.cpp (example)
llama-bench -m models/foo.gguf -p 512 -n 128
/usr/bin/time -v llama-cli -m models/foo.gguf -p "..." -n 128 -c 4096
```

Or run `scripts/bench_compare.sh` / `scripts/bench_compare.ps1`.

## Linux container (io_uring) — hayai vs llama.cpp

Same four GGUFs for both engines, one per execution path, so the Linux `io_uring`
read path is compared against llama.cpp on equal footing (proves the disk is not
the limiter):

1. `SmolLM2-135M-Instruct-Q4_K_M.gguf` — Dense small (resident)
2. `Qwen2.5-7B-Instruct-Q4_K_M.gguf` — Dense large (streaming, I/O-bound)
3. `OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf` — MoE (sparse)
4. `Qwen_Qwen3.5-4B-Q4_K_M.gguf` — hybrid / DeltaNet

CPU-only image: builds hayai (nightly) and llama.cpp (`master`, CPU) inside
Debian bookworm, then runs both with `models/` mounted read-only. Reports
prefill/decode tok/s and peak RSS (`/usr/bin/time -v`).

```bash
# Linux / WSL / Git Bash — builds the image, runs the table, saves to bench_results/
scripts/bench_docker.sh

# PowerShell (Docker Desktop)
scripts\bench_docker.ps1

# One model, custom sizes
MODELS="Qwen2.5-7B-Instruct-Q4_K_M.gguf" N=64 PP=256 scripts/bench_docker.sh
```

Env knobs: `MODELS_DIR`, `MODELS` (space-separated basenames), `N` (decode
tokens), `PP` (llama-bench prefill tokens), `THREADS`, `MEMORY_STRATEGY`
(default `minimal` so hayai streams instead of becoming resident),
`LLAMA_REF` (build arg). Fixtures: `scripts/bench_docker/Dockerfile`,
`scripts/bench_docker/compare_inside.sh`.

