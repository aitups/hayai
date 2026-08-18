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
