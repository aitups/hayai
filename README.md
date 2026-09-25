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
| `calibrate` | Measure host/disk/device bandwidth and print 75–80% utilization targets (see `CALIBRATION.md`) |
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
(`tokenizer.chat_template`, Jinja) and rendered with `minijinja` — used by the API
server, the CLI `generate`/`bench-generate` (when not `--raw`), and the session API.
If the GGUF has none, a ChatML fallback is assumed. `--chat-template <string|@file>`
overrides it for every model.

Compatibility shims so transformers templates work in `minijinja`:
`.get('key')` → `['key']`, `.startswith(...)`/`.endswith(...)` → filters. If the
rendered prompt references a special-token marker (e.g. `<|user|>`) that is not in
the vocab (a GGUF conversion dropped added tokens), hayai logs a clear warning — the
prompt would otherwise be BPE-split and degraded. Base models without any chat
markers get a warning to use `--raw`.

### Notes

- Requests for the same model are serialized on the shared OpenCL orchestrator;
  each request keeps its own KV context. Default `--memory-strategy minimal`
  avoids per-request resident preloads on multi-request servers.
- `n > 1`, `logprobs` and `/v1/embeddings` are not implemented yet.

## Docker

A precompiled image is provided (`aitups/hayai`, Debian bookworm, multi-arch
amd64/arm64). It ships `hayai-server` and `hayai-cli`, runs as a non-root user
(uid 1000) and includes the generic OpenCL ICD loader (`ocl-icd-libopencl1`)
plus an NVIDIA ICD registration file (`/etc/OpenCL/vendors/nvidia.icd`).
Without a vendor OpenCL runtime available at run time the engine falls back to
CPU-only.

```bash
# Build locally
docker build -t aitups/hayai:latest .

# Run with your models directory mounted (reads work even if read-only; make it
# writable by uid 1000 for --hf downloads: `sudo chown 1000:1000 models`)
docker run -d -p 8080:8080 -v "$PWD/models:/hayai/models" aitups/hayai:latest

# Or use compose (CPU-only unless the GPU override is applied)
docker compose up -d
```

### Docker + GPU (OpenCL)

OpenCL uses the **ICD (Installable Client Driver)** model. The loader in the
image (`libOpenCL.so.1`) only reports a platform when a vendor driver is
registered through a `*.icd` file in `/etc/OpenCL/vendors/` that points to the
vendor's OpenCL runtime library (`libnvidia-opencl.so.1` for NVIDIA). The image
ships the registration file; the **library itself must be provided at run
time**.

> **CUDA ≠ OpenCL.** Passing the GPU to a container with `--gpus all` (Docker
> Desktop/WSL2 or `nvidia-container-toolkit`) injects the **CUDA driver
> libraries** (`libcuda.so.1`, `libnvidia-ml.so.1`) but **not** the OpenCL ICD.
> If you see `Failed to query OpenCL platforms: CL_PLATFORM_NOT_FOUND_KHR`
> (or Hayai falls back to CPU-only), the OpenCL runtime library is missing from
> the container — this is exactly the symptom of "only CUDA was passed".
> Debug with `docker exec -it <container> clinfo -l`.

**Native Linux** (recommended): install `nvidia-container-toolkit`. It mounts
the driver userspace — including `libnvidia-opencl.so.1` — into GPU containers,
and the image's registration file makes it visible:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

**Docker Desktop / WSL2 (Windows):** the GPU-PV passthrough only provides the
CUDA interface, and NVIDIA WSL drivers (at least up to 560.x) do **not** ship
`libnvidia-opencl.so.1` in `/usr/lib/wsl/lib`, so NVIDIA OpenCL is currently
**unavailable** inside WSL2 containers; Hayai will run CPU-only. To get GPU
acceleration with Hayai on Windows, use native Linux (or a Linux VM with GPU
passthrough) with `nvidia-container-toolkit`. If a future NVIDIA driver ships
the WSL OpenCL ICD, mount it with `-v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro`
(the image's `LD_LIBRARY_PATH` already includes that path).

All `hayai-server` options are configurable via `HAYAI_*` env vars (see
`docker-compose.yml`): `HAYAI_HOST`, `HAYAI_PORT`, `HAYAI_MODELS_DIR`,
`HAYAI_MODEL` (comma-separated), `HAYAI_HF`, `HAYAI_HF_FILE`, `HAYAI_DEVICE`,
`HAYAI_MEMORY_STRATEGY`, `HAYAI_SINKS`, `HAYAI_WINDOW`, `HAYAI_CHAT_TEMPLATE`,
`HAYAI_LOG`. Explicit CLI args still take precedence. `HAYAI_DEVICE=auto`
picks any OpenCL GPU and falls back to CPU-only if none is available.

> **io_uring & Docker:** Hayai's streaming I/O path uses `io_uring` on Linux.
> Docker's **default seccomp profile** (not WSL2, not the container, not the
> kernel) blocks the `io_uring_setup` syscall with `EPERM`, so Hayai logs
> `io_uring WeightIo open failed (Operation not permitted (os error 1))` and
> falls back to buffered File I/O (`WeightIo=file`) — which is fully functional.
> The `docker-compose.yml` ships with `security_opt: [seccomp=unconfined]` so
> the container uses `io_uring` out of the box; with plain `docker run` add
> `--security-opt seccomp=unconfined`. Verified behavior: with the flag the
> logs show `WeightIo=io_uring` and `Opened GGUF catalog v3 (io_uring)`.



## Supported models

`ExecPlan` is metadata-driven: tensors are classified into a **registered layer-op catalog** (`LayerOpKind` + per-op HW binding), so any GGUF whose tensors map to registered ops builds a streaming plan. Failure is only `unknown layer op` for truly novel tensor roles — never the architecture name. Families exercised:

| Family | Architecture / notes |
| --- | --- |
| Llama-family | e.g. SmolLM2 (default smoke model) |
| Gemma 4 | 12B (`post_ffw_norm`, `layer_output_scale`, per-head Q/K norms) and E4B (shared-KV, SWA/global, PLE, GELU softcap) |
| HRM-Text | recurrent H/L stacks (`hrm_text`) |
| Qwen 3.5 | hybrid DeltaNet (SSM/linear attention) + full attention + MTP `nextn` draft head |
| MoE | router (`ffn_gate_inp`) + per-expert (`ffn_exp.E.*`) and fused-3D (`ffn_*_exps.*`, incl. fused `ffn_gate_up_exps`) experts; **sparse streaming**: only the top-k experts are read per token (~`top_k/n_expert` of the FFN disk bandwidth). Validated on OLMoE-1B-7B (64 experts / 8 active) |
| MLA | DeepSeek-V2/V3, Kimi: non-absorbed (`attn_kv_b`) and absorbed (`attn_k_b`/`attn_v_b`) paths, YaRN `attn_factor`/`kq_scale` matching `llama.cpp`. Validated on DeepSeek-V2-Lite (position-0 logits) |
| Classic transformer | GPT-2 (learned absolute positions, LayerNorm, erf-GELU), BLOOM/MPT (ALiBi), ungated FFN, **parallel residual + single shared norm** (Phi-2/GPT-J/Falcon-7B style), partial RoPE; RoPE is opt-in per model. Validated on GPT-2/BLOOM/Phi-2 (greedy identical to `llama.cpp`) and Falcon-7B (fused MQA QKV, greedy identical) |
| StableLM-2 | `stablelm`: pre-norm sequential blocks, partial RoPE (`rope.dimension_count`), q/k/v biases, gated FFN. Validated on `stablelm-2-1_6b.q4_k_m` (top-5 pos-0 logits identical to `llama.cpp`) |
| Phi | Phi-3-mini 4k/128k (RMSNorm, fused `attn_qkv`, fused gate+up stored as `ffn_up` with `nrows = 2·ffn_length`, silu; **LongRoPE** `rope_factors_{short,long}` + `attn_factor` for the 128k variant). Validated on the official `Phi-3-mini-4k-instruct-q4` (greedy identical to `llama.cpp`) and `Phi-3-mini-128k` (top-1 logits + greedy identical) |
| GLM-4 | `chatglm` (GLM-4-9B): fused `attn_qkv` + bias, fused gate+up as `ffn_up`, GQA, partial RoPE (64/128), RMSNorm. Validated on `glm-4-9b-chat.Q2_K` (prefill top-k identical to `llama.cpp`) |
| Granite | IBM Granite (RMSNorm, GQA, tied embeddings) with architecture scalars (`embedding_multiplier`, `residual_multiplier`, `logits_scaling`, `attention_multiplier`) and a per-layer `head_count_kv` array. Validated on `granite-4.0-1b-Q4_K_M` (greedy identical to `llama.cpp`) |
| Encoder-decoder | T5/FLAN-T5 (`t5`): bidirectional encoder + causal decoder with **cross-attention** and learned **relative position bias** (no RoPE/absolute positions, no `1/√d`), gated `gelu_new` FFN, SentencePiece **unigram** tokenizer. Validated vs HF `flan-t5-small` (CLI + server) |
| BART | `bart`: pre-LayerNorm-free **post-norm** blocks, learned positions (offset 2), `1/√d` scaling, non-gated `gelu` FFN, cross-attention, tied `lm_head` + `final_logits_bias`; byte-level BPE. Validated vs HF `facebook/bart-base` (top-8 logits + greedy). Convert with `scripts/bart_to_gguf.py` (llama.cpp no longer ships BART) |

**Quantization formats:** F32/F16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, the K-family (Q2_K–Q6_K), and IQ2/IQ3/IQ4 (BF16 has CPU GEMV only). **Q8_K/Q8_1** are recognized but size-only; **IQ1/TQ** raise a hard unsupported-type error.

---

## Environment variables

| Variable | Effect |
| --- | --- |
| `RUST_LOG` | `tracing` filter (default `info`), e.g. `RUST_LOG=debug` |
| `HAYAI_FORCE_TOKIO_IO` | Force tokio-based I/O instead of `io_uring` on Linux |
| `HAYAI_O_DIRECT` | Open the GGUF with `O_DIRECT` (Linux, bypass page cache; buffers are page-aligned) |
| `HAYAI_MAX_LAYERS` | Cap the number of model layers processed (debug) |
| `HAYAI_MOE_CACHE_MB` | MoE expert LRU cache budget in MiB (default 512; `0` disables) — hot experts stay resident, cutting disk I/O per token |
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

