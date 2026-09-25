# Op catalog (F2.11)

Every GGUF tensor maps to exactly one [`LayerOpKind`](../crates/hayai-core/src/exec_plan.rs).
The classifier is **op-by-op**: a tensor's *role* decides the op, never
`general.architecture`. The catalog below is the complete set of roles Hayai knows
today, the tensor names/aliases each recognizes, its hardware binding, and whether
execution is implemented.

Rules:

- **No `other` bucket.** An unknown role returns `Err` (loud), never a silent `Aux`.
- **Safe aliases only.** An alias is added only when it unambiguously denotes one op
  (e.g. `out_proj` → attention output is guarded so `linear_attn.out_proj` stays
  DeltaNet).
- **New op ⇒ fail until implemented.** A real op that is not executed yet returns
  `Err` with a specific message (never misclassified as an existing op).

## Catalog

| `LayerOpKind` | Tensor names / aliases | Binding | Executor |
|---|---|---|---|
| `TokenEmbed` | `token_embd`, `embed_tokens`, `tok_embeddings`, `word_embeddings`, `embed_in`, `wte` | HOST_ROW | host row read |
| `TokenEmbedNorm` | `token_embd_norm`, `word_embeddings_layernorm`, `embed_norm` (BLOOM) | CPU_NORM | input LayerNorm |
| `PositionEmbed` | `position_embd`, `wpe`, `pos_embd` — learned absolute (GPT-2) | HOST_ROW | added at input |
| `OutputNorm` | `output_norm`, `model.norm`, `norm.weight`, `final_layer_norm`, `final_layernorm`, `ln_f` | CPU_NORM | dense/gemma/… |
| `OutputProj` | `output.weight`, `lm_head`, `embed_out` | GPU_ASYNC | LM head (tied→`token_embd`) |
| `AttnNorm` | `attn_norm`, `input_layernorm`, `attention_norm`, `norm_1`, `ln_1` | CPU_NORM | all attention paths |
| `AttnQ` / `AttnK` / `AttnV` | `attn_q`/`q_proj`/`.wq.`; `attn_k`/`k_proj`/`.wk.`; `attn_v`/`v_proj`/`.wv.` | CPU_GEMV | all attention paths |
| `AttnO` | `attn_output`, `o_proj`, `attn_out`, `.wo.`, `out_proj`\*, `self_attention.dense`, `attention.dense` | CPU_GEMV | all attention paths |
| `AttnGate` | `attn_gate`, `inp_gate` (no PLE) | CPU_GEMV | hybrid/Qwen3.5 |
| `AttnQkv` | `attn_qkv`, `qkv_proj`, `wqkv`, `query_key_value`, `c_attn` | CPU_GEMV | split `[q\|k\|v]` at load |
| `MlaQa`/`MlaQaNorm`/`MlaQb` | `attn_q_a[_norm]`, `attn_q_b` (DeepSeek-V2/V3, Kimi) | CPU_GEMV/NORM | MLA forward (full path) |
| `MlaKvA`/`MlaKvANorm` | `attn_kv_a_mqa`, `attn_kv_a_norm` | CPU_GEMV/NORM | MLA forward |
| `MlaKb`/`MlaVb`/`MlaKvB` | `attn_k_b`, `attn_v_b`, `attn_kv_b` (fused lite) | CPU_GEMV | `MlaKvB` (DeepSeek-V2-Lite) implemented |
| `AttnQNorm` / `AttnKNorm` | `attn_q_norm` / `q_norm`; `attn_k_norm` / `k_norm` | CPU_NORM | gemma4 |
| `PostAttnNorm` | `post_attention_norm` | CPU_NORM | gemma4 |
| `FfnNorm` | `ffn_norm`, `post_attention_layernorm`, `norm_2`, `ln_2` | CPU_NORM | all FFN paths |
| `FfnGate` | `ffn_gate`, `gate_proj` | GPU_ASYNC | dense/hybrid/gemma |
| `FfnUp` | `ffn_up`, `up_proj`, `dense_h_to_4h` | GPU_ASYNC | dense/hybrid/gemma |
| `FfnDown` | `ffn_down`, `down_proj`, `dense_4h_to_h` | GPU_ASYNC | dense/hybrid/gemma |
| `PostFfnNorm` | `post_ffw_norm`, `post_mlp_norm` | CPU_NORM | gemma4 |
| `LayerOutputScale` | `layer_output_scale` | DISCARD (preloaded scalar) | gemma4 |
| `Router` | `router`, `gate_inp`, `block_sparse_moe.gate`, `moe.gate`, `mlp.gate.weight` | CPU_GEMV | MoE |
| `RouterBias` | `ffn_gate_inp.bias` — `e_score_correction_bias` (DeepSeek V3) | DISCARD (preloaded) | choice-score bias |
| `RouterScale` | `ffn_gate_inp.scale` | DISCARD (preloaded) | router input scale |
| `ExpertScale` | `ffn_*_exps.scale` (length `n_expert`) | DISCARD (preloaded) | per-expert scale |
| `ExpertGate`/`ExpertUp`/`ExpertDown` | `ffn_exp.E.ffn_*`, `experts.E.{w1,w2,w3,gate/up/down_proj}` | GPU_ASYNC | MoE (fused 3D slices) |
| `SharedExpert` | `shared_expert`, `ffn_shexp` | GPU_ASYNC | MoE |
| `DeltaNet` | `delta`, `linear_attn`, `ssm*`, `conv1d`, `in_proj_qkv(z)`, `in_proj_ba`, `shortconv.out_proj` | CPU_GEMV | hybrid (Qwen3.5) |
| `Mamba` | `ssm_x`, `ssm_in`, `.ssm_d` | CPU_GEMV | mamba_infer |
| `NextN` | `.nextn.`, `nextn_`, `.mtp.`, `mtp_` | CPU_GEMV | mtp/spec_decode |
| `PleEmbed`/`PleModelProj`/`PleProjNorm`/`PleGate`/`PleProj`/`PlePostNorm` | `per_layer_*`, `blk.N.{inp_gate,proj,post_norm}` (when `embedding_length_per_layer_input > 0`) | mixed | gemma4 PLE |
| `Recurrence` | `h_cycle`, `l_cycle`, `recurrent`, `z_h`, `z_l`, `h_layers`, `l_layers` | CPU_GEMV | HRM |
| `Conv` | `conv*` (depthwise causal, activation from `hayai.conv_activation`) | CPU_GEMV | dense residual |
| `FfnDagAdjacency`/`FfnDagWeights` | `ffn_dag_adjacency`, `ffn_dag_weights` | GPU_ASYNC | sparse DAG |
| `Aux` | `*.bias`/`*_bias`, `*.scale`/`*_scale`, `*rope*`, `*cos*`, `*sin*`, `*inv_freq*`, `*alibi*` | DISCARD | passive (preloaded/ignored) |

\* `out_proj` is attention output **unless** the name also matches
`linear_attn`/`ssm`/`shortconv`/`delta`/`mamba`, in which case it is `DeltaNet`.

## Known ops that fail until implemented (loud `Err`)

| Tensor | Reason |
|---|---|
| MLA full/absorbed (`is_mla` split 3D `wk_b`/`wv_b`) | Not implemented (non-absorbed `attn_kv_b` Lite path is) |
| `relative_attention_bias` / `cross_attn` / `.encoder.` / `.decoder.` | T5/BART are encoder-decoder — a separate architecture class, not yet implemented |
| `ffn_gate_up_exps` (fused experts) | Fused per-expert gate+up; needs a 3D row split |
| `attn_sink`, `shear` | Compute-affecting, unimplemented |
| anything unrecognized | Register a `LayerOpKind` + classifier arm + `op_binding` (see `docs/adding-a-model-family.md`) |

## Implemented classic-transformer Dense ops (Fase 1)

- **LayerNorm** vs RMSNorm: chosen from `{arch}.attention.layer_norm_epsilon`
  (LayerNorm) vs `layer_norm_rms_epsilon` (RMSNorm); norm biases (`*_norm.bias`) are
  preloaded and applied.
- **Ungated FFN** (`up → gelu → down`, no `ffn_gate`): GPT-2 / BLOOM / OPT / Falcon.
- **Fused gate+up** (`ffn_gate_up` / HF `gate_up_proj`, Phi-3): `FfnGateUp`, rows
  concat `[gate | up]` split at pack load.
- **FFN / QKV / output biases**: preloaded once and applied by the Dense path
  (fused `attn_qkv.bias` is split `[q|k|v]`).
- **Learned positions** (`position_embd`) and **input-embedding LayerNorm**
  (`token_embd_norm`, BLOOM).
- **ALiBi** (BLOOM/Falcon/MPT/Starcoder): `-slope[h]·(q_pos - k_pos)` added to scores
  in `BoundedKvCache::attend`; absolute key positions from `attention_slot_positions`.
- **RoPE scaling** (linear / YaRN): `{arch}.rope.scaling.{type,factor,beta_*,
  original_context_length,attn_factor}` → `AttentionConfig.rope` /
  `apply_rope_partial_factors_scaled`.
- **`AttentionConfig::use_rope`**: `false` for learned-position (GPT-2) / ALiBi
  (BLOOM/Falcon/MPT) models, which must not rotate Q/K.
- Validated against `llama.cpp` at fixed positions (bit-identical top-k logits):
  GPT-2 (F16/Q4_K_M/Q2_K, no RoPE), tiny BLOOM (ALiBi), OLMoE (MoE + RoPE),
  SmolLM2 (F16, GQA + RoPE, pos 0/1/2/5).

## MLA (DeepSeek-V2/V3, Kimi)

`mla_attention` (`crates/hayai-core/src/moe_infer.rs`) implements the non-absorbed
`deepseek2` path (Lite + legacy fused `attn_kv_b`): `attn_q` (lite) or `q_a`→`q_b`,
`attn_kv_a_mqa` → latent `c_kv`(+`k_pe`), `attn_kv_a_norm`, fused `attn_kv_b` →
per-head `[k_nope | v]`, YaRN RoPE on the trailing `qk_rope` dims (offset
`qk_nope`), MQA-style compressed cache. Validated: `attn_factor`/`kq_scale` match
`llama-context.cpp`/`deepseek2.cpp`; a clean-room reference reproduces `q_pe`/`attn`
bit-for-bit; DeepSeek-V2-Lite logits at position 0 (after BOS) are identical to
`llama.cpp`.

## MoE routing (modern)

`route_experts(scores, correction_bias, meta)` (`crates/hayai-core/src/moe_infer.rs`)
implements llama.cpp `build_moe_ffn`:

- **Grouped top-k**: `{arch}.expert_group_count` / `expert_group_used_count`; the
  group score is the sum of its top-2 experts (DeepSeek V3).
- **Gating**: `{arch}.expert_gating_func` 0 = softmax (Mixtral/Qwen-MoE) or 1 =
  sigmoid (DeepSeek V3 / GLM).
- **`e_score_correction_bias`**: `ffn_gate_inp.bias` shifts the *selection* scores
  but not the weights.
- **`norm_topk_prob`** (`expert_weights_norm`) and **`routed_scaling_factor`**
  (`expert_weights_scale`).

## Adding an op

1. Variant in `LayerOpKind` (doc comment naming the tensors).
2. Arm in `classify_tensor_impl` in the right phase order (earlier wins).
3. `op_binding` arm (exhaustive `match` → a missing arm is a compile error).
4. Executor: reuse `ffn_apply`/`full_attn_apply`/`run_deltanet_block`, or an
   `*_infer.rs` for a new family.
5. Classifier test + (if runnable) a model smoke test; run the full gate.
