# Hayai

**Low-memory LLM weight-streaming inference engine** built in **Rust** with an **OpenCL 3.0** backend.

Hayai runs LLMs on hardware with severely constrained memory (edge devices, APUs, consumer iGPUs, small-RAM servers). Instead of relying on the operating system's reactive memory mapping (`mmap`) of a whole model, it **streams compressed weights deterministically from disk** through a double-buffered (ping-pong) pipeline, overlapping I/O with compute and orchestrating CPU and GPU **heterogeneously**:

- **GPU (OpenCL 3.0)** executes the feed-forward networks (FFN), computing directly on packed quantized weights (LUT MatMul / GEMV).
- **CPU (Rust portable SIMD)** runs attention, RoPE, the KV cache and state management.
- **Bounded memory budget:** only a small layer window of weights plus the KV cache and activations are resident at any time.

The engine is **architecture-agnostic**: it reads a GGUF's metadata and builds a streaming `ExecPlan`, classifying tensors into registered layer ops. It only refuses to run on truly novel layer types (`unknown layer op`).

> **Status:** active development. Product target is **Linux**; **Windows is a development host only**.

---

## Repository layout

| Crate | Role |
| --- | --- |
| `hayai-core` | Orchestration, streaming inference, exec-plan, memory budget and metrics |
| `hayai-io` | Async disk I/O (`io_uring` on Linux, tokio fallback), ping-pong buffers, weight streaming |
| `hayai-opencl` | OpenCL engine, device discovery, SVM / pinned-memory transfer paths, heterogeneous scratch |
| `hayai-kernels` | OpenCL C kernel sources (embedded at compile time) |
| `hayai-cpu` | SIMD attention, Q4 LUT MatMul, KV cache |
| `hayai-model` | GGUF parsing, quantized GEMV formats, tokenizer, sampler |
| `hayai-cli` | Command-line interface |
| `hayai-api` | OpenAI-compatible HTTP server (`hayai-server`) |

---

## Requirements

- **Rust (nightly)** — the repo pins the toolchain in `rust-toolchain.toml`. The engine uses the unstable portable SIMD API (`std::simd`).
- **OpenCL 3.0 runtime** — vendor drivers with an ICD loader (`opencl.dll` on Windows, `libOpenCL.so` on Linux). No OpenCL SDK is needed to build: the OpenCL loader is resolved dynamically at runtime.
- **Linux** (product target): a kernel with `io_uring` support is used when available; the engine transparently falls back to tokio-based I/O.
- **Windows** (dev host only): the pinned toolchain is `nightly-x86_64-pc-windows-gnu`, which requires a **MinGW-w64** environment (e.g. MSYS2's `mingw64`). The machine-specific linker configuration lives in `.cargo/config.toml` (gitignored).
- **Optional:** `llama-bench` / `llama-cli` (llama.cpp) for side-by-side benchmark comparisons.

---

## Build

```bash
# Windows dev host (MSYS2 MinGW-w64 gcc must be on PATH)
rustup toolchain install nightly-x86_64-pc-windows-gnu
cargo build --release

# Linux
rustup toolchain install nightly
RUSTUP_TOOLCHAIN=nightly cargo build --release
```

`rust-toolchain.toml` pins the Windows GNU nightly, so on Linux either use `RUSTUP_TOOLCHAIN=nightly` as above or override per-directory:

```bash
rustup override set nightly
cargo build --release
```

Run tests:

```bash
cargo test -p hayai-core -p hayai-model -p hayai-opencl -p hayai-cpu
```

---

## Quick start

Build, download a small test model, and generate text:

```bash
# 1. Build
cargo build --release

# 2. Inspect available hardware (OpenCL APUs/GPUs + CPU SIMD)
cargo run --release -p hayai-cli -- info

# 3. Download SmolLM2-135M-Instruct (Q4_K_M) into models/
cargo run --release -p hayai-cli -- fetch-model

# 4. Dry-run the streaming plan (metadata → layer ops)
cargo run --release -p hayai-cli -- plan --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf

# 5. Generate text (ChatML-wrapped prompt; add --raw to send it as-is)
cargo run --release -p hayai-cli -- generate \
  --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  --prompt "The capital of France is" \
  --max-tokens 64 \
  --device auto
```

> Model files are not part of the repository (`models/` is gitignored); download them into `models/` or pass `--model /any/path/model.gguf`.

---

## CLI commands

| Command | Description |
| --- | --- |
| `info` | Discover OpenCL devices, transfer paths (SVM / pinned), CPU SIMD backend |
| `bench` | MatMul micro-benchmark (SmolLM-135M FFN layer, Q4) — OpenCL when available, else CPU |
| `validate` | Validate Q4 LUT MatMul against an FP32 reference (CPU vs OpenCL) |
| `bench-io` | Streaming I/O pipeline benchmark: disk → ping-pong → compute/upload (synthetic model) |
| `bench-hetero` | Heterogeneous pipeline benchmark: CPU attention ∥ GPU FFN |
| `fetch-model` | Download the SmolLM-135M Instruct GGUF into `models/` |
| `generate` | End-to-end text generation from a GGUF |
| `bench-generate` | Benchmark packed-Q generation: tok/s, owned memory MiB, budget check |
| `plan` | Parse GGUF metadata → streaming `ExecPlan` (fails only on unknown layer ops) |

### Common options

- `--device auto|cpu|<substring>` — `auto` picks any OpenCL GPU and falls back to CPU; `cpu` forces CPU-only; any other string selects the first OpenCL device whose name contains it (falls back to CPU if not found).
- `--memory-strategy auto|minimal|<cap_mb>` — memory window for the weight stream:
  - `auto` (default): detect free VRAM/RAM and pick the largest safe layer chunk (`k_chunk`). If the whole model fits, it becomes **resident** (all layers preloaded into SVM/VRAM once → zero disk I/O and zero per-token DMA).
  - `minimal`: strict 2-slot ping-pong (lowest footprint).
  - `2048` / `2048mb`: cap the resident window (e.g. macro-chunking with `k_chunk` blocks).
- `generate`: `--model`, `--prompt`, `--max-tokens`, `--device`, `--sinks`, `--window`, `--sample greedy|temperature|top_p`, `--temperature`, `--top_p`, `--seed`, `--raw`, `--dev_mmap` (dev-only mmap path that violates the streaming design).
- `bench-generate`: same model/prompt/tokens/device/sinks/window options plus `--raw`.

Full option reference is available with `--help`:

```bash
cargo run --release -p hayai-cli -- generate --help
```

---

## Server (OpenAI-compatible API)

`hayai-server` exposes the engine through an OpenAI-compatible HTTP API with
token-by-token SSE streaming.

```bash
# Serve a single model
hayai-server --model models/SmolLM2-135M-Instruct-Q4_K_M.gguf --port 8080 --device cpu

# Auto-scan models/ + download a model from HuggingFace (default *Q4_K_M.gguf)
hayai-server --models-dir models --hf bartowski/SmolLM2-135M-Instruct-GGUF --port 8080

# Exact HF file + resident weights + custom chat template
hayai-server --hf Qwen/Qwen2.5-7B-Instruct-GGUF --hf-file qwen2.5-7b-instruct-q4_k_m.gguf \
  --memory-strategy auto --chat-template @my_template.jinja
```

### Endpoints

| Endpoint | Description |
| --- | --- |
| `GET /healthz` | Health check |
| `GET /v1/models` | List registered models (auto-scan of `--models-dir` + `--model` + `--hf`) |
| `POST /v1/completions` | Legacy completion (`prompt`, `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `stream`) |
| `POST /v1/chat/completions` | Chat completion (`messages`, same options) — renders the model's chat template |

Both generation endpoints support `"stream": true` → SSE events with
`choices[].delta.content` per token and a final `data: [DONE]`.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"SmolLM2-135M-Instruct-Q4_K_M","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32,"stream":true}'
```

### Chat templates

The chat template is read from the model's GGUF metadata
(`tokenizer.chat_template`, Jinja) and rendered with `minijinja`. If the GGUF
has none, a ChatML fallback is assumed. `--chat-template <string|@file>` overrides
it for every model.

### Notes

- Requests for the same model are serialized on the shared OpenCL orchestrator;
  each request keeps its own KV context. Default `--memory-strategy minimal`
  avoids per-request resident preloads on multi-request servers.
- `n > 1`, `logprobs` and `/v1/embeddings` are not implemented yet.

## Supported models

`ExecPlan` is metadata-driven, so any GGUF whose tensors map to registered layer ops should build a streaming plan. Families exercised in the current suite:

| Family | Architecture / notes |
| --- | --- |
| Llama-family | e.g. SmolLM2 (default smoke model) |
| Gemma 4 E4B | shared-KV, SWA/global attention, PLE, GELU softcap |
| HRM-Text | recurrent H/L stacks (`hrm_text`) |
| Qwen 3.5 | hybrid DeltaNet (SSM/linear attention) + full attention + MTP `nextn` draft head |

**Quantization formats:** F32/F16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, the K-family (Q2_K–Q6_K, Q8_K), and IQ2/IQ3/IQ4. **IQ1/TQ** types are recognized but raise a hard unsupported-type error.

---

## Environment variables

| Variable | Effect |
| --- | --- |
| `RUST_LOG` | `tracing` filter (default `info`), e.g. `RUST_LOG=debug` |
| `HAYAI_FORCE_TOKIO_IO` | Force tokio-based I/O instead of `io_uring` on Linux |
| `HAYAI_O_DIRECT` | Open the GGUF with `O_DIRECT` (Linux, bypass page cache; buffers are page-aligned) |
| `HAYAI_MAX_LAYERS` | Cap the number of model layers processed (debug) |
| `HAYAI_SKIP_DN_ATTN` / `HAYAI_SKIP_FA_ATTN` | Skip DeltaNet / full-attention residuals (ablation) |
| `HAYAI_DUMP_TOP` / `HAYAI_DUMP_AT_POS` / `HAYAI_DUMP_LAYER_RMS` | Dump top-k logits / layer RMS norms (debug) |
| `HAYAI_DN_*` | DeltaNet internal ablations (`HAYAI_CONV_FLIP`, `HAYAI_DN_NO_CONV`, `HAYAI_DN_NO_L2`, `HAYAI_DN_SWAP_AB`, `HAYAI_DN_BETA_RAW`, `HAYAI_DN_A_RAW`, `HAYAI_DN_KMAP`) |

---

## Benchmarks

- `scripts/bench_suite.ps1` — full suite: hayai vs llama.cpp (llama-bench/llama-cli) over SmolLM2, HRM-Text, Qwen3.5 and Gemma4; writes a timestamped report into `bench_results/` (gitignored).
- `scripts/bench_compare.sh` — Linux equivalent, along with `bench_compare.ps1` and the protocol notes in `scripts/bench_compare.md`.

```powershell
powershell -File scripts/bench_suite.ps1 -SmolTokens 32 -SuiteTokens 8
```

---

## License

MIT OR Apache-2.0

