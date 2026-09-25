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
| `ExpertGateUp` | `ffn_gate_up_exps` (fused `[gate\|up]` per expert) | GPU_ASYNC | split into `ExpertGate`/`ExpertUp` at plan time |
| `SharedExpert` | `shared_expert`, `ffn_shexp` | GPU_ASYNC | MoE |
| `DeltaNet` | `delta`, `linear_attn`, `ssm*`, `conv1d`, `in_proj_qkv(z)`, `in_proj_ba`, `shortconv.out_proj` | CPU_GEMV | hybrid (Qwen3.5) |
| `Mamba` | `ssm_x`, `ssm_in`, `.ssm_d` | CPU_GEMV | mamba_infer |
| `NextN` | `.nextn.`, `nextn_`, `.mtp.`, `mtp_` | CPU_GEMV | mtp/spec_decode |
| `PleEmbed`/`PleModelProj`/`PleProjNorm`/`PleGate`/`PleProj`/`PlePostNorm` | `per_layer_*`, `blk.N.{inp_gate,proj,post_norm}` (when `embedding_length_per_layer_input > 0`) | mixed | gemma4 PLE |
| `Recurrence` | `h_cycle`, `l_cycle`, `recurrent`, `z_h`, `z_l`, `h_layers`, `l_layers` | CPU_GEMV | HRM |
| `CrossAttnQ/K/V/O` | `dec.blk.N.cross_attn_{q,k,v,o}` (T5/BART) | CPU_GEMV | `encoder_decoder_infer` |
| `CrossAttnNorm` | `dec.blk.N.cross_attn_norm` | DISCARD (preloaded) | encoder-decoder |
| `EncOutputNorm`/`DecOutputNorm` | `enc.output_norm` / `dec.output_norm` | DISCARD (preloaded) | encoder-decoder |
| `RelPosBias` | `attn_rel_b` / `relative_attention_bias` (T5) | DISCARD (preloaded) | relative position bias |
| `Conv` | `conv*` (depthwise causal, activation from `hayai.conv_activation`) | CPU_GEMV | dense residual |
| `FfnDagAdjacency`/`FfnDagWeights` | `ffn_dag_adjacency`, `ffn_dag_weights` | GPU_ASYNC | sparse DAG |
| `Aux` | `*.bias`/`*_bias`, `*.scale`/`*_scale`, `*rope*`, `*cos*`, `*sin*`, `*inv_freq*`, `*alibi*` | DISCARD | passive (preloaded/ignored) |

\* `out_proj` is attention output **unless** the name also matches
`linear_attn`/`ssm`/`shortconv`/`delta`/`mamba`, in which case it is `DeltaNet`.

## Known ops that fail until implemented (loud `Err`)

| Tensor | Reason |
|---|---|
| `attn_sink`, `shear` | Compute-affecting, unimplemented |
| `cohere2` per-layer SWA + NoPE | Command-R7B: `load_swa_pattern(4)` — SWA layers use RoPE + a 4096 sliding window, global layers use **NoPE** (no RoPE) + full attention, plus a multiplicative `logit_scale`. Parallel residual is already handled. |
| anything unrecognized | Register a `LayerOpKind` + classifier arm + `op_binding` (see `docs/adding-a-model-family.md`) |

## Encoder-decoder (T5 / BART) — `encoder_decoder_infer`

T5 is a separate architecture **class** (two stacks), handled by
`crates/hayai-core/src/encoder_decoder_infer.rs` (`T5Model`), not the decoder-only
`StreamingGenerator`. Tensors still stream layer-by-layer via the shared
`ExecPlan` + GGUF slice loader. Details:

- `enc.blk.N.*` (bidirectional self-attn) and `dec.blk.N.*` (causal self-attn +
  cross-attn) map to distinct plan units (`dec` offset by `DEC_BLOCK_OFFSET`).
- **No RoPE / no absolute positions / no `1/√d` score scaling**: attention adds a
  learned **relative position bias** (`attn_rel_b`, shared from block 0) binned by
  the HF `_relative_position_bucket` function (bidirectional for encoder + cross,
  causal for decoder self-attn). Cross-attention has no bias.
- Norms are T5 `T5LayerNorm` (RMS, weight only); the FFN is **gated with
  `gelu_new`** (`gated-gelu`; the GGUF does not record `feed_forward_proj`, so that
  is the assumed default). Output uses a separate `output.weight` (not tied) and is
  **not** scaled by `d_model^-0.5` when the embeddings are untied.
- Tokenizer: `tokenizer.ggml.model = "t5"` uses SentencePiece **unigram** (no
  merges); `Tokenizer` now runs Viterbi over `tokenizer.ggml.scores` for these
  vocabularies (`unigram_encode`), with a leading space (`add_dummy_prefix`) and a
  trailing EOS.
- **Validated** vs HuggingFace `google/flan-t5-small`: encoder ids and greedy ids
  match exactly (`translate English to German: The house is wonderful.` →
  `Das Haus ist schön.`); first-step top-5 logits match (`[644,316,37,660,1122]`,
  Δ ≈ F16 rounding). Tests: `t5_first_logits_match_hf`, `t5_greedy_matches_hf`
  (auto-skip without `models/flan-t5-small.F16.gguf`).
- **BART** (`general.architecture = "bart"`, `BartModel`): **post-norm** blocks
  (`x = ln(x + sublayer(x))`, no final norm), **learned positions** (`enc/dec.pos_embd`,
  offset 2) + `layernorm_embedding`, standard `1/√d` scaling, non-gated `gelu` FFN,
  attention/FFN biases, tied `lm_head` (`token_embd`) + `output.bias`
  (`final_logits_bias`). Validated vs HF `facebook/bart-base` (top-8 logits + greedy
  identical). llama.cpp no longer ships BART, so the GGUF is built by
  `scripts/bart_to_gguf.py` from safetensors.

## Implemented classic-transformer Dense ops (Fase 1)

- **LayerNorm** vs RMSNorm: chosen from `{arch}.attention.layer_norm_epsilon`
  (LayerNorm) vs `layer_norm_rms_epsilon` (RMSNorm); norm biases (`*_norm.bias`) are
  preloaded and applied.
- **Ungated FFN** (`up → gelu → down`, no `ffn_gate`): GPT-2 / BLOOM / OPT / Falcon.
- **Fused gate+up** (`ffn_gate_up` / HF `gate_up_proj`, Phi-3): `FfnGateUp`, rows
  concat `[gate | up]` split at pack load. Phi-3's converter stores the same fused
  tensor under the name **`ffn_up`** (`nrows = 2·ffn_length`, no `ffn_gate`); the pack
  loader detects that shape (`ffn_gate_up_source`) and splits it identically.
- **FFN / QKV / output biases**: preloaded once and applied by the Dense path
  (fused `attn_qkv.bias` is split `[q|k|v]`).
- **Learned positions** (`position_embd`) and **input-embedding LayerNorm**
  (`token_embd_norm`, BLOOM).
- **ALiBi** (BLOOM/Falcon/MPT/Starcoder): `-slope[h]·(q_pos - k_pos)` added to scores
  in `BoundedKvCache::attend`; absolute key positions from `attention_slot_positions`.
  Falcon is split by `falcon.tensor_data_layout`: the new multiquery architecture
  (`"jploski"`, falcon-7B/40B) uses **RoPE**, only the old arch (falcon-rw-1b) ALiBi.
  MPT clamps the bias to `{arch}.attention.max_alibi_bias` (`alibi_max_bias`).
- **RoPE scaling** (linear / YaRN / **LongRoPE**): `{arch}.rope.scaling.{type,factor,
  beta_*,original_context_length,attn_factor}` → `AttentionConfig.rope` /
  `apply_rope_partial_factors_scaled`. LongRoPE (Phi-3-128k) is detected by the
  `rope_factors_{short,long}.weight` tensors: `short`/`long` are selected by sequence
  length (`> original_context_length`) and passed as per-dim `freq_factors`, with
  `attn_factor` as the cos/sin mscale. A `rope_freqs.weight` tensor (Gemma4
  proportional RoPE, or the Llama-3 NTK-by-parts scaling baked by the converter) is
  likewise passed as per-dim `freq_factors` in the Dense path.
- **Architecture scalars** (Granite): `{arch}.embedding_scale`, `residual_scale`,
  `logit_scale` and `attention.scale` (→ `AttentionConfig::scale_override`) applied in
  the Dense paths; `head_count_kv` may be a per-layer array (first element used).
- **`AttentionConfig::use_rope`**: `false` for learned-position (GPT-2) / ALiBi
  (BLOOM/Falcon/MPT) models, which must not rotate Q/K.
- **Parallel residual + single shared norm** (Phi-2/GPT-J/PaLM): when block 0 has
  `attn_norm` + `ffn_up` but no `ffn_norm` (`detect_parallel_residual`), the per-token
  Dense path computes `x = x + attn(ln(x)) + ffn(ln(x))` from the same normed input.
- Validated against `llama.cpp` at fixed positions (bit-identical top-k logits):
  GPT-2 (F16/Q4_K_M/Q2_K, no RoPE), tiny BLOOM (ALiBi), OLMoE (MoE + RoPE),
  SmolLM2 (F16, GQA + RoPE, pos 0/1/2/5), Phi-2 (Q4_K_M, parallel residual + partial
  RoPE, 20-token greedy identical), Phi-3-mini (fused `ffn_up` gate+up, greedy
  identical), Granite-4.0-1b (architecture scalars + array `head_count_kv`, greedy
  identical), Qwen2.5-7B (attention biases, top-4 pos-0/1 identical),
  StableLM-2 (partial RoPE + qkv biases, top-5 pos-0 identical), Falcon-7B (fused MQA
  QKV + shared-norm parallel residual, greedy identical), GLM-4/`chatglm` (fused QKV
  + bias, fused `ffn_up` gate+up, partial RoPE, prefill top-k identical), MPT-7B (ALiBi
  + `max_alibi_bias` clamp + fused QKV, greedy identical), MiniCPM5-2B (`llama` arch,
  top-1 identical).

## MLA (DeepSeek-V2/V3, Kimi)

`mla_attention` (`crates/hayai-core/src/moe_infer.rs`) implements both `deepseek2`
variants:
- **Non-absorbed** (Lite + legacy fused `attn_kv_b`): `attn_q` (lite) or `q_a`→`q_b`,
  `attn_kv_a_mqa` → latent `c_kv`(+`k_pe`), `attn_kv_a_norm`, fused `attn_kv_b` →
  per-head `[k_nope | v]`, per-head KV cache.
- **Absorbed** (`attn_k_b`/`attn_v_b`, full DeepSeek-V2/V3 + Kimi): `W_kb` absorbs
  `q_nope` per head (`q_absorbed = W_kb^T q_nope`), a single MQA latent KV cache
  (`[c_kv | k_pe]`), and `W_vb` maps the attention output back to `v_head_dim`;
  `MlaMeta::absorbed` selects it and `view_of_head` slices the packed 3D per-head tensors.

Both use YaRN RoPE on the trailing `qk_rope` dims (offset `qk_nope`). Validated:
`attn_factor`/`kq_scale` match `llama-context.cpp`/`deepseek2.cpp`; a clean-room
reference reproduces `q_pe`/`attn` bit-for-bit; DeepSeek-V2-Lite (non-absorbed) logits
at position 0 (after BOS) are identical to `llama.cpp`; the absorbed `W_kb^T q` algebra
is unit-tested (`absorbed_wkb_gemv_is_transpose_product`).

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
