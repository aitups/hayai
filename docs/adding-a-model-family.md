# Adding a model family

Hayai is **metadata-driven, not name-driven**: a GGUF is turned into an `ExecPlan`
by mapping every tensor name to a `LayerOpKind` (`crates/hayai-core/src/exec_plan.rs`),
and each family runs through an `*_infer.rs`. Adding a family is therefore a
repeatable, bounded change — this is the checklist.

Read first: `AGENTS.md` (layout + gotchas) and `PRD.md` (streaming non-negotiables).

## 0. Golden rules

- **Never bake model-size constants.** Head counts, SSM ranks, `d_inner`, block counts
  must be resolved from tensor shapes (see `layer_cfg.rs`) so every size of a family
  works. `layer_cfg.rs` is the canonical place that does shape-derived resolution.
- **No silent drops.** A tensor that affects compute and is not executed must
  **fail loudly** in `classify_tensor_impl` (this bug class made Qwen2.5 produce
  garbage: its attention biases were classified `Aux` and dropped). If you can't
  execute a tensor, return `Err`.
- **Preload once, stream the rest.** Small per-layer tensors (norms, biases,
  conv kernels, `z_l_init`, `rope_freqs`) are loaded once into `StreamingGenerator`
  fields; only the large weight matrices stream from disk.
- **Reuse shared runners.** `stage_pack`/`begin_ffn_dma`/`ffn_apply`,
  `full_attn_apply`, `run_deltanet_block`, `run_ffn_block` already implement
  DMA/SVM, ping-pong and sparse CSR. Call them instead of reimplementing.

## 1. Classify tensors → `LayerOpKind`

In `crates/hayai-core/src/exec_plan.rs`:

1. Add a variant to `enum LayerOpKind` with a doc comment naming the tensors.
2. Add an arm in `classify_tensor_impl` **in the right phase order** (earlier
   matches win): biases/draft heads before generic name substrings. Prefer a
   phase comment (`// ── Phase N: … ──`).
3. Add the HW binding in `op_binding`: `CPU_GEMV` for CPU ops, `GPU_ASYNC` for
   FFN-style GEMV, or the appropriate binding. The `match` is exhaustive, so a
   missing arm is a compile error.
4. Preload-related tensors that are genuinely passive (RoPE tables, scales) stay
   `Aux`; **anything else must be executed or return `Err`**.

Add a classifier test next to the existing `classifies_*` tests.

## 2. Detect the family → `ModelKind`

In `stream_infer.rs` (`ModelKind::from_catalog` / `model_kind`): families are
detected from **ops/tensors present**, never from `general.architecture` (that is
only a fallback). Dispatch in `prefill` / `decode_step`:

```rust
match self.model_kind() {
    ModelKind::Dense  => self.forward_inner(...),
    ModelKind::Hybrid => crate::hybrid_infer::forward_hybrid(...),
    ModelKind::Gemma  => crate::gemma_infer::forward_gemma(...),
    ModelKind::MoE    => crate::moe_infer::forward_moe(...),
    ModelKind::MyFam  => crate::myfamily_infer::forward_myfam(...),
}
```

Put per-family execution in `crates/hayai-core/src/myfamily_infer.rs` as
`impl StreamingGenerator { … }` (or free functions taking `&mut StreamingGenerator`).

## 3. Per-layer shapes → `layer_cfg.rs`

If the family has non-LLaMA attention (GQA with unusual head dims, SSM ranks,
different KV kinds), resolve its config from that block's tensors and size the KV
cache in `build_hybrid_kv_caches` (full-attention/linear-attention/NextN kinds).
Do **not** add a `4B`-shaped fallback.

## 4. Config parsing (only if new metadata)

If the family needs new metadata keys, parse them in `load_config`
(`stream_infer.rs`) into `ModelConfig` (add a field, keep `Default`-safe) using
`cat.meta_*`. Unknown/missing keys must degrade to a safe default or fail loudly,
never panic.

## 5. Streaming FFN + device paths

Reuse the standard layer-pack flow:

```rust
let (pack, layout) = self.stage_pack(orch, scratch, slot, layer)?;
self.begin_ffn_dma(orch, scratch, slot, &layout)?;
full_attn_apply(self, layer, pos, eps, &pack, x)?;          // or family attention
self.finish_ffn_unmap(orch, scratch, slot)?;
ffn_apply(self, orch, Some(scratch), layer, eps, &pack, Some(&layout), x, ov)?;
```

`ffn_apply`/`run_ffn_block` already handle dense GEMV, sparse CSR and FFN
overrides. For prefill, prefer a **layer-major** loop (stage each layer once, loop
the prompt tokens) — see `prefill_hybrid_all`.

## 6. Tests

1. `classify_tensor_impl` unit test for every new tensor name.
2. A synthetic-GGUF test that opens the model and checks dispatch / KV sizing
   (`write_minimal_gguf`; see `fused_qkv_open`, `global_sparse_overrides_build_from_genome`).
3. If the family has a runnable model in `models/`, a greedy smoke test with a
   known continuation (e.g. `1 2 3 4 5` → `6 7 8 9`).
4. Run the full gate:
   `cargo test -p hayai-core -p hayai-model -p hayai-opencl -p hayai-cpu -p hayai-api -p hayai-io`
   and `cargo clippy --workspace --all-targets`, plus the aarch64 cross-check
   (`cargo check --target aarch64-unknown-linux-gnu -p hayai-core -p hayai-io -p hayai-opencl -p hayai-cpu -p hayai-model -p hayai-kernels`).

## 7. Checklist

- [ ] `LayerOpKind` variant + classifier arm (correct phase) + `op_binding` (exhaustive)
- [ ] Classifier unit test for each tensor
- [ ] `ModelKind` detection + dispatch in `prefill`/`decode_step`
- [ ] `*_infer.rs` using shared runners; no baked constants
- [ ] KV/config resolution from tensor shapes (`layer_cfg.rs`)
- [ ] Small tensors preloaded once; large weights streamed
- [ ] Unknown compute tensors fail loudly (never silent drop)
- [ ] Synthetic open test + model smoke test + full test/clippy/aarch64 gate
- [ ] Document any new metadata key here and in `AGENTS.md`
