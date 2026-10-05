# AGENTS.md

Hayai: Rust workspace for a low-memory LLM weight-streaming inference engine with an
OpenCL 3.0 backend. CPU runs attention/RoPE/KV; GPU runs the FFN; weights stream
deterministically from disk (ping-pong buffers), never via OS `mmap`.

## Toolchain / build

- **Nightly is required**, not optional: `hayai-cpu` and `hayai-model` use
  `#![feature(portable_simd)]`. `rust-toolchain.toml` pins
  `nightly-x86_64-pc-windows-gnu` (with `rustfmt`+`clippy`).
- On Linux the pin is a Windows toolchain, so override: `RUSTUP_TOOLCHAIN=nightly cargo build --release`
  (or `rustup override set nightly` once).
- Windows dev host: MinGW-w64 must be on `PATH`. `.cargo/config.toml` is **gitignored**
  and machine-specific (hardcoded `gcc.exe`/`ar.exe` paths). Put **both**
  `C:\msys64\mingw64\bin` and `C:\msys64\usr\bin` on `PATH`: with only the former, even
  dependency build scripts fail to link with `collect2.exe: error: ld returned 53`.
  `scripts/build_windows.bat` prepends both.
- `cargo build --release` is the normal build (README examples all use `--release`).
- Hardware calibration: `cargo run --release -p hayai-cli -- calibrate --model <gguf>`
  measures host/disk/device bandwidth and prints the 75–80% utilization tok/s targets;
  protocol and per-host baselines live in `CALIBRATION.md` (tracked).
- Tests: `cargo test -p hayai-core -p hayai-model -p hayai-opencl -p hayai-cpu -p hayai-api -p hayai-io`.
  `hayai-opencl` tests auto-skip when no OpenCL device is present. There is no
  workspace-level test alias; use `-p <crate> <testname>` for a single test.
- ARM64 product target (GB10-class): the lib crates cross-check cleanly with
  `cargo check --target aarch64-unknown-linux-gnu -p hayai-core -p hayai-io -p hayai-opencl -p hayai-cpu -p hayai-model -p hayai-kernels`.
- No `rustfmt.toml`/`clippy.toml`, no `[workspace.lints]`. CI is
  `.github/workflows/ci.yml` (Linux, `RUSTUP_TOOLCHAIN=nightly`): fmt + clippy + tests,
  plus an advisory Miri job for `hayai-model`.

## Layout / entrypoints

- Bins: `hayai-cli` (package `hayai-cli`) and `hayai-server` (package `hayai-api`, `src/main.rs`).
- `hayai-core`: orchestration, streaming inference, memory budget, and the architecture
  dispatch. `hayai-io`: disk I/O (`io_uring` on Linux, tokio fallback). `hayai-opencl`:
  device/engine/SVM. `hayai-kernels`: `.cl` sources embedded via `include_str!`.
  `hayai-cpu`: SIMD attention/KV. `hayai-model`: GGUF, quant formats, tokenizer, sampler.
- Sampler (`hayai-model/src/sampling.rs`): greedy/temperature/top-p/top-k/min-p, plus
  `Penalties` (repetition/presence/frequency + `logit_bias`); NaN/±inf logits are
  sanitized and seed 0 is reseeded so sampling is never degenerate. CLI
  `--sample {greedy|temperature|top_p|top_k|top_k_top_p|min_p}` + `--top-k/--min-p/
  --repetition-penalty/--presence-penalty/--frequency-penalty`; the server request
  accepts `top_k/min_p/presence_penalty/frequency_penalty/logit_bias`.
- Tokenizer (`hayai-model/src/tokenizer.rs`): GPT-2 (byte-level BPE) pre-tokenization
  follows the HF `pattern` (contractions, ` ?\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+`, whitespace);
  SPM (`spm`) keeps `▁`/`<0xNN>` byte fallback. `encode→decode` is byte-exact. `encode`
  does not duplicate a BOS already emitted by the template, and the special-token
  heuristic covers `<|name|>`, `<|name>` and `<name|>` (Gemma 4 `<|turn>`/`<turn|>`/
  `<|channel>`/`<channel|>`). Generation stops on every id in `eos_ids`: the GGUF
  `eos_token_id` (scalar or array) plus known end-of-turn markers (`<turn|>`,
  `<end_of_turn>`, `<|eot_id|>`, ...), and stop tokens are excluded from the text.
  All Jinja chat rendering (CLI + server) goes through
  `Tokenizer::render_chat_template[_override]` with `raise_exception`/`strftime_now`
  shims and a `tools` context; never duplicate it.
- Quants (`hayai-model/src/{quant,gguf}.rs`): CPU dequant + GEMV cover F32/F16/BF16/F64,
  I8/I16/I32/I64, Q4_0/1, Q5_0/1, Q8_0/1, Q8_K and the K/IQ families. OpenCL kernels
  exist for F32/F16/BF16 + Q4_0/1/Q5_0/1/Q8_0/Q8_1/Q*_K/IQ* (BF16/Q8_1/Q8_K GEMV added
  in F2.6b; parity-tested). F64/I* have no OpenCL kernel (CPU `execute` fallback only).
- GGUF (`hayai-model/src/gguf{,_stream}.rs`): versions 1–4 accepted (same layout; v1
  was previously rejected). Split GGUFs (`split.count` > 1, `<base>-NNNNN-of-MMMMM.gguf`)
  are merged into one catalog via `ShardedIo` (virtual offsets), so all read paths work
  unchanged. CLI `--logit-bias "id:bias,id:bias"`.
- Grammar (`hayai-model/src/grammar.rs`): llama.cpp-style GBNF parser + NFA with a
  return stack, so right-recursive grammars (JSON) work; left recursion fails closed.
  `StreamingGenerator::set_grammar` masks logits per step (CLI `--grammar <file>`; the
  server request also accepts a `grammar` GBNF string). Head-interleaved fused QKV is
  opt-in via GGUF metadata `hayai.attn_qkv_interleave_repeats` (= number of heads);
  default is concat `[q|k|v]`.
- Tool calling (`hayai-api`): chat requests accept `tools`/`tool_choice` and messages
  carry `tool_calls`/`tool_call_id`/`name`; all are passed through to the Jinja template
  (`Tokenizer::render_chat_template_values`). `parse_tool_calls` converts model-emitted
  `<tool_call>{...}</tool_call>` blocks into OpenAI `tool_calls` (`finish_reason =
  "tool_calls"`) in non-streaming responses (streaming emits raw text).
- Global sparse FFN ("Vía B", `hayai-model/src/cppn.rs`): a CPPN genome + threshold
  (`saor.sparse`/`saor.genome`/`saor.tau`) decodes the live FFN connections per layer;
  `StreamingGenerator::build_global_sparse_overrides` prunes the dense weights to CSR
  and `generate_with_override` runs them (CLI `--sparse-global`, Dense models only).
- NextN/MTP draft head (`hayai-core/src/mtp.rs`): Qwen3.5's trailing `blk.{N}` is
  excluded from the main trunk and executed on demand via `forward_mtp`. The head is
  numerically correct (`mtp-probe` draft agreement ~80–92%); it needs (a) the target's
  post-final-norm hidden per token (`last_hidden_nextn`, saved by every forward path)
  and (b) its own KV primed over the prompt (`token[i]`, `h[i-1]` at position i), as in
  llama.cpp. `spec_decode.rs` drives greedy speculative decoding: draft `k` tokens with
  the MTP head, verify them in one batched `prefill_hybrid_all` forward, commit the
  accepted prefix and roll back KV + DeltaNet state on rejection (CLI `--spec-drafts k`,
  Hybrid only). Output is identical to greedy; ~1.5–2× faster on the loaded dev host.
- Architecture is metadata-driven, not name-driven: GGUF tensors map to `LayerOpKind`
  in `crates/hayai-core/src/exec_plan.rs`; failure is only `UnknownLayerOp`. Per-family
  execution lives in `crates/hayai-core/src/{infer,gemma_infer,hrm_infer,hybrid_infer,moe_infer,mamba_infer}.rs`.
  **Execution is op-driven, never `general.architecture`-name-driven**: `ModelKind::from_catalog`
  scans tensors/ops, `load_config` falls back to tensor shapes, and an unknown family name
  with known ops runs unchanged (tests `unknown_architecture_loads_from_ops_not_name`,
  `mamba_detected_from_ops_not_name`). Mamba-1 selective scan (`ssm_in`/`ssm_x`/`ssm_d`/
  `ssm_out`/`ssm_a`/`ssm_dt`/`ssm_conv1d`) is a first-class op (`LayerOpKind::Mamba`,
  `mamba_infer.rs`), distinct from Qwen3.5 DeltaNet (`ssm_alpha`/`ssm_beta`/`ssm_norm`).
- Adding a model family = register a `LayerOpKind` + HW binding in `exec_plan.rs`, then an
  `*_infer.rs`. Per `layer_cfg.rs`, **never bake model-size constants** — head counts, SSM
  ranks, etc. must be resolved from tensor shapes so every size of a family works.
  Step-by-step guide + checklist: `docs/adding-a-model-family.md`; the complete op→tensor
  catalog (F2.11) is `docs/op-catalog.md`. The classifier is op-by-op with **safe aliases
  only** and no `other` bucket: a real op that is not executed fails loudly
  (`gate_up`/`gateup` fused FFN, `wpe`/`position_embd` learned positions), and a compute
  `.proj.weight` is no longer swallowed by `Aux`.
- **RoPE is opt-in per model**: `AttentionConfig::use_rope` is `false` for models
  whose positions come from learned embeddings (GPT-2) or ALiBi (BLOOM/MPT); Falcon
  is split by `falcon.tensor_data_layout` — the new multiquery architecture
  (`"jploski"`, falcon-7B/40B) uses RoPE, only the old arch (falcon-rw-1b) ALiBi.
  RoPE scaling covers linear/YaRN and **LongRoPE** (Phi-3-128k: `rope_factors_{short,
  long}.weight` selected by sequence length + `attn_factor` mscale, passed as per-dim
  `freq_factors`). A `rope_freqs.weight` tensor (Gemma4 proportional RoPE, or Llama-3's
  NTK-by-parts scaling baked by the converter) is also passed as `freq_factors` in the
  Dense path. The tokenizer defaults `add_bos=true` for the `llama-bpe`/`llama3`
  pre-tokenizers when `tokenizer.ggml.add_bos_token` is absent (matching llama.cpp,
  whose default is true). applying RoPE to them corrupts attention. Validation against a local `llama.cpp`
  (`llama-server`) is done at **fixed positions** (compare top-k logits, not greedy
  text) — llama.cpp quantizes GEMV activations (Q8_K) and its `/completion` adds a
  sampler chain, so text diverges even when the forward matches. MLA
  (DeepSeek-V2/V3, Kimi) is implemented in `moe_infer::mla_attention` and matches
  `llama-context.cpp`'s YaRN `attn_factor`/`kq_scale`. Both variants are supported:
  the non-absorbed Lite path (`attn_kv_b`, per-head `[k_nope|v]` decompression) and the
  absorbed path (`attn_k_b`/`attn_v_b`, split-3D `wk_b`/`wv_b` per head with a single MQA
  latent KV cache — `MlaMeta::absorbed` selects it; `view_of_head` slices each head).
- **Parallel residual + single shared norm** (Phi-2/GPT-J/PaLM): detected when block 0
  has `attn_norm` + `ffn_up` but no `ffn_norm` (`detect_parallel_residual`); then
  `forward_inner` computes `x = x + attn(ln(x)) + ffn(ln(x))` (both from the same
  normed input) and forces the `simple_dense` per-token path. Phi-2 also uses a fused
  `attn_qkv` (already split) and **partial RoPE** (`rope.dimension_count=32` of 80).
  Validated vs `llama-server` on `phi-2.Q4_K_M` (20-token greedy identical; top-4
  pos-0 logits identical).
- **Architecture scalars** (Granite): `{arch}.embedding_scale` (×embedding),
  `residual_scale` (×each sublayer output before the residual add),
  `logit_scale` (÷final logits) and `attention.scale` → `AttentionConfig::scale_override`
  are read in `open`/`build_attn_config` and applied in the Dense paths
  (`embed_row`, the residual adds, `apply_logit_scale`). Granite's
  `attention.head_count_kv` is a **per-layer array**; `load_config` takes the first
  element for the generic Dense path. Validated on `granite-4.0-1b-Q4_K_M` (greedy
  identical to `llama.cpp`).
- **Cohere2** (Command-R7B): alternating SWA/global layers (`is_swa(i) = i % 4 != 3`,
  via GGUF debug log). SWA layers apply RoPE + a `sliding_window` cache; global layers
  apply **NoPE** (no RoPE) + full attention — handled by `StreamingGenerator::layer_apply_rope`
  (per-layer `apply_rope` arg to `attention_decode_step_ex`) and per-layer KV windows.
  `logit_scale` is a **multiplier** here (Granite divides). Single shared norm parallel
  residual (`attn_norm` only). Validated on `c4ai-command-r7b-12-2024-Q2_K` (tokenization
  + top-1 logits identical to `llama.cpp`).
- Fused 3D experts (`ffn_gate_up_exps`, per-expert rows `[gate|up]`) are split into
  `ExpertGate`/`ExpertUp` slices at plan time (`build_exec_plan`); only `is_expert_op`
  3-D tensors feed the fused path, so per-head MLA `attn_k_b`/`attn_v_b` are never
  mistaken for experts.
- Encoder-decoder (T5/BART) is a distinct architecture **class**, run by
  `hayai-core/src/encoder_decoder_infer.rs` (`T5Model`), not the decoder-only
  `StreamingGenerator`. `enc.blk.N.*` and `dec.blk.N.*` become separate plan units
  (`dec` offset by `DEC_BLOCK_OFFSET`). T5 has **no RoPE / no absolute positions /
  no `1/√d` scaling**: attention adds a learned relative position bias (`attn_rel_b`,
  HF `_relative_position_bucket`, bidirectional for encoder+cross, causal for decoder
  self-attn); cross-attention has no bias; norms are T5 RMS; FFN is gated `gelu_new`
  (`gated-gelu`; not stored in the GGUF, assumed). The T5 tokenizer is SentencePiece
  **unigram** — `Tokenizer::unigram_encode` runs Viterbi over `tokenizer.ggml.scores`
  for vocabs with scores and **no merges** (T5, plus some non-standard conversions),
  with `<0xNN>` byte fallback. Validated vs HF `fln-t5-small`
  (ids + top-5 logits); tests auto-skip without the model. The CLI `generate` and
  `hayai-server` auto-detect enc-dec models (`is_encoder_decoder`) and use this path
  (`T5Session`/`EncDecSession`); the decoder-only `StreamingGenerator` rejects them
  explicitly. **BART** (`general.architecture = "bart"`) shares the class but differs:
  **post-norm** blocks (`x = ln(x + sublayer(x))`, no final norm), **learned positions**
  (`enc/dec.pos_embd`, offset 2), standard `1/√d` scaling, non-gated `gelu` FFN,
  attention/FFN biases, tied `lm_head` + `output.bias` (`final_logits_bias`). Validated
  vs HF `facebook/bart-base` (top-8 logits + greedy identical). llama.cpp no longer
  ships BART, so its GGUF is produced by `scripts/bart_to_gguf.py` from safetensors.
- Fused `attn_qkv.weight` is split by output rows (concat `[q|k|v]`) into q/k/v at pack
  load (see `load_layer_pack_into_fused`); a head-interleaved layout is opt-in via the
  `hayai.attn_qkv_interleave_repeats` GGUF metadata. Phi-3 and GLM-4 (`chatglm`) store
  their fused **gate+up** under the name `ffn_up` (`nrows = 2·ffn_length`, no `ffn_gate`);
  `GgufCatalog::ffn_gate_up_source` detects that shape and splits `[gate|up]` like the
  `ffn_gate_up` form (validated on `Phi-3-mini-4k-instruct-q4` and `glm-4-9b-chat`, both
  matching `llama.cpp`). Generic `Conv` tensors execute as a
  depthwise causal conv1d + activation residual after attention (`apply_depthwise_conv`;
  activation from `hayai.conv_activation` = `silu`(default)/`gelu`/`none`), with the conv
  weights cached and no per-token plan scan. A conv whose `channels != hidden` is applied
  to a matching buffer by the family path; the Dense residual path hard-fails instead of
  silently skipping it. SSM/DeltaNet convs (`ssm_conv1d`, `mamba`) route to `deltanet.rs`.
- Attention q/k/v/o biases (`attn_{q,k,v,output}.bias`, `{q,k,v,o}_proj.bias`) are
  preloaded once per model into `StreamingGenerator::attn_bias` and added after the
  q/k/v/o GEMV in every Dense attention path (Qwen2/2.5 `attention_bias=true`). FFN/
  output biases are **not** implemented and hard-fail in `classify_tensor` (never silent).

## Gotchas
- Vision/multimodal (`mmproj`/`clip` GGUFs): the **encoders + injection are implemented**
  (`hayai-model/src/vision.rs`: `ClipEmbedder` for `gemma4uv` vision and `gemma4ua` audio;
  `image`/`symphonia` decoding; `StreamingGenerator::prefill_media`/`generate_media` with
  `MediaInput`; CLI `--mmproj --image --audio`, `<__media__>` markers). **Validated**:
  image content is distinguished (gemma-4-12B: red vs blue images → "red" / "dark blue",
  matching `llama-server --mmproj`), and audio embeddings reach the model (greedy output
  differs for no-audio / tone / white-noise; a pure sine is not "heard" as speech).
  No llama.cpp audio reference in the WinGet build (`llama-mtmd-cli` crashes, the server's
  `input_audio` is unsupported). Gemma4UV is not a ViT: im2col → LayerNorm(`patch_norm.1`)
  → `patch_embd`+bias → LayerNorm(`patch_norm.2`) → 2D `position_embd` (x/y tables) →
  LayerNorm(`patch_norm.3`) → RMSNorm → `mm.input_projection`. Audio (`gemma4ua`) is
  encoder-free: 640-sample frames → RMSNorm → `mm.a.input_projection`.
- Gemma4 global/SWA layers differ **per layer** (global/SWA use different head_dim /
  n_kv; the pack loader treats a missing `attn_v` as an empty V (k_eq_v)). `GemmaMeta`
  now resolves per-layer `head_dim`/`n_heads`/`n_kv` from each block's tensors
  (`attn_q_norm`/`attn_k_norm` dims, `attn_k.nrows`, `attn_output.ncols`) and reads
  `attention.head_count_kv` if it is an array. `gemma-4-E4B` and `gemma-4-12B` both run
  coherently (`gemma-4-12B`: "The capital of France is Paris."). The 12B's former
  `<|channel>thought` loop was **not** in `gemma_infer.rs` but in the tokenizer (control
  tokens adjacent to text were split; plus a duplicated BOS) — see the Tokenizer bullet.
  For future architecture debugging, compare logits at one position against a clean
  llama.cpp reference (`llama-server` `/completion` with `logprobs`/`top_logprobs` is
  unambiguous; `llama-perplexity --save-all-logits` row alignment is easy to get wrong).
- The PRD's non-negotiable is: no reactive OS `mmap` as the inference path. `--dev_mmap`
  exists only for debugging and violates the design; don't build features on it.
- OpenCL is dynamically loaded (no SDK at build time). **OpenCL ≠ CUDA**: passing `--gpus all`
  in Docker injects CUDA libs but not the OpenCL ICD, so Hayai silently falls back to CPU
  (`CL_PLATFORM_NOT_FOUND_KHR`). See README "Docker + GPU".
- `models/`, `bench_results/`, `results_*.txt`, `target_q4gpu/`, and the
  `implementation_plan*.md` / Spanish design docs are gitignored. `models/` is empty on
  checkout — use `cargo run --release -p hayai-cli -- fetch-model`.
- `scripts/gen_iq{2,3}_rs.py` and `scripts/patch_iq{2,3}_cl.py` regenerate the IQ grid
  tables/kernels from a **hardcoded local `ggml-common.h` dump**; they are one-shot
  machine-specific helpers, not a reproducible codegen step.
- `scripts/bench_suite.ps1` (Windows) / `bench_compare.{sh,ps1}` (Linux) run side-by-side
  vs llama.cpp; protocol notes in `scripts/bench_compare.md`. Not part of `cargo test`.
  `scripts/bench_docker.{sh,ps1}` build a CPU-only Debian image (hayai nightly + llama.cpp
  master) and compare both engines on the 4-path suite (SmolLM2/Qwen2.5-7B/OLMoE/Qwen3.5-4B)
  with `models/` mounted read-only — prefill/decode tok/s + peak RSS on real Linux
  `io_uring`.
- `hayai-server` flags added for safety: `--max-concurrency` (default 4, bounds concurrent
  generations) and `--api-key` / `HAYAI_API_KEY` (when set, `/v1/*` requires
  `Authorization: Bearer`). Request body is capped at 8 MiB and `/v1/*` rejects `n>1`,
  `logprobs` and `max_tokens > 32768` with 400.
- `--memory-strategy auto` (AutoFit) caps CPU-only resident windows at 1 GiB so large
  models stream instead of being copied into host RAM; the "owned memory" budget counts
  **two** ping-pong slots (`2 * k_chunk * layer`) plus resident output/embedding.
- Per-family streaming: Dense and hybrid/gemma prefill are **layer-major** (each layer
  read once per prompt). Dense prefill can additionally batch the FFN across the prompt
  tokens (`HAYAI_PREFILL_BATCH=1`), reading each weight once per layer via
  `execute_quant_gemv_batched` (dense FFN; CSR blocks fall back per-token). MoE caches the non-expert (attn+router+shared) tensors in a
  bounded RAM cache (`HAYAI_MOE_NONEXPERT_MB`, default 1024) and streams only the top-k
  experts (`HAYAI_MOE_CACHE_MB`, default 512). Gemma PLE's global `per_layer_model_proj`
  is loaded once; per-layer PLE weights are prefetched one layer ahead. The MoE CPU path
  evaluates the top-k experts in parallel (rayon). **Expert-pack prefetch is not
  implemented**: prefetching the next layer's likely experts (previous-token selection,
  then warming the LRU cache) was prototyped and reverted — for OLMoE the selection is not
  predictable the next token so it ~doubled I/O (11506 → 23862 MiB, 1.74 → 1.30 tok/s),
  and the 512 MiB cache (128 packs vs 1024 experts) thrashes. Needs a better predictor
  (global frequency / grouped routing) and a native Linux host to measure.
- **Hardware-aware planner** (`hayai-core/src/planner.rs`): the engine no longer hardcodes
  "attention→CPU, FFN→GPU". At session start `prepare_session` calibrates the host + every
  device (host RAM, disk, per-device effective *and resident* GEMV bandwidth, per-op launch
  overhead, DMA link — new `OpenClEngine::bench_*`) and runs a cost-driven list scheduler
  that assigns **every placeable op** to a concrete `ComputeTarget::Cpu | Device(i)`
  (`ExecPlan.placement`). `orchestrator::execute_op` honors it; `execute_op_bound`/
  `submit_op_bound` run an op from the layer's device mirror (no per-op host upload) and
  let independent ops (attention Q/K/V) run concurrently across devices. Ops are placed by
  measured bytes/bandwidth, not by name, so a novel architecture is planned by the same
  code. `HAYAI_FFN_MIN_GPU_PARAMS` (the old `auto→CPU` patch) is gone.
- **OpenCL attention kernel** (`hayai-kernels/kernels/attn_decode.cl`, `hayai_attn_decode`)
  + `DeviceKvCache` (FP32, per layer): single-query online-softmax attention, GQA-aware.
  `OpenClEngine::kv_append`/`attn_decode`; parity test `attn_decode_matches_reference`
  (auto-skips without a device). Wired into the hybrid full-attention path behind the
  opt-in `HAYAI_ATTN_GPU=1` (the default keeps the host INT8 path bit-identical). Known
  gap: the host step also applies RoPE, so the opt-in GPU path still needs the host RoPE
  before the kernel — it is experimental until validated on the T4.
- **OpenCL DeltaNet kernel** (`hayai-kernels/kernels/deltanet_step.cl`,
  `hayai_deltanet_step`) + `DeviceDeltanetState`: one work-item per (value head, value
  dim) doing the in-place recurrent state decay/read/delta-write/output, llama.cpp order
  with the `%`-tile KV-head mapping. `OpenClEngine::deltanet_step`; parity test
  `deltanet_step_matches_reference` over two steps (state carries). **MoE routing needs
  no new kernel**: the router is a GEMV (already `GpuAsync`/placeable) and the top-k +
  softmax run on the host (tiny, expert_count-sized).
- **io_uring** is the production Linux `WeightIo`; the batched scatter path now has a
  correctness test (`io_uring_read_many_at_scatter_is_correct`, Linux only).
- **N-slot scratch + prefetch pipeline** (`StreamingScratch.n_slots`, `HAYAI_PREFETCH_DEPTH`
  default 2): slots are a `Vec` addressed by `idx % n_slots` (macro-chunk:
  `(layer/block_k) % n_slots`), and the Dense streaming loop keeps up to `n_slots - 1`
  layer reads in flight. Depth > 2 is enabled only with no device mirror (the CPU path is
  validated at depth 3); the GPU mirror DMA path is validated at depth 2 and N-slot
  mirrors still need work.
- `PRD.md`, `implementation_plan.md`, and `pr_soporte_gguf_disperso_v3.md` are the design
  sources of truth.
