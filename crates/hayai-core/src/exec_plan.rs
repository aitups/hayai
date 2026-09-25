//! Metadata-driven execution plan: classify GGUF tensors into known layer ops.
//!
//! Hayai is architecture-agnostic: any GGUF whose tensors map to registered ops
//! gets a streaming plan. Failure is only `UnknownLayerOp` for truly novel layers.
//!
//! This module is the **layer-op catalog**: every tensor role a GGUF may contain
//! maps to a [`LayerOpKind`], each op carries a hardware binding ([`op_binding`]),
//! and tensors are grouped into streaming units. Nothing falls into a catch-all
//! "known" bucket — unregistered roles hard-fail.

use hayai_model::GgufCatalog;
use std::collections::{BTreeMap, BTreeSet};

/// Known computational layer / tensor roles (the extensible op registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LayerOpKind {
    /// `token_embd.weight` — embedding lookup (host row read).
    TokenEmbed,
    /// `token_embd_norm.weight/bias` — LayerNorm applied to the input embeddings
    /// (BLOOM `word_embeddings_layernorm`), before the first block.
    TokenEmbedNorm,
    /// `position_embd.weight` / `wpe` — learned absolute position embeddings (GPT-2).
    PositionEmbed,
    /// `output_norm.weight` — final norm before the LM head.
    OutputNorm,
    /// `output.weight` / `lm_head.weight` — vocabulary projection.
    OutputProj,
    /// `attn_norm.weight` / `input_layernorm.weight` — attention input norm.
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    /// Gated attention (`attn_gate`, or `inp_gate` on non-PLE models).
    AttnGate,
    /// MLA (DeepSeek-V2/V3, Kimi): `attn_q_a` (down-proj to q LoRA rank).
    MlaQa,
    /// MLA q LoRA norm (`attn_q_a_norm`).
    MlaQaNorm,
    /// MLA `attn_q_b` (up-proj from q LoRA rank to heads).
    MlaQb,
    /// MLA `attn_kv_a_mqa` (down-proj to `kv_lora_rank + qk_rope`).
    MlaKvA,
    /// MLA kv LoRA norm (`attn_kv_a_norm`).
    MlaKvANorm,
    /// MLA `attn_k_b` (absorbed K up-proj; also `attn_kv_b` legacy fused).
    MlaKb,
    /// MLA `attn_v_b` (absorbed V up-proj).
    MlaVb,
    /// MLA legacy fused `attn_kv_b` (unsplit `k_b`+`v_b`).
    MlaKvB,
    /// Encoder-decoder (T5/BART): encoder stack final norm (`enc.output_norm`).
    EncOutputNorm,
    /// Encoder-decoder (T5/BART): decoder stack final norm (`dec.output_norm`).
    DecOutputNorm,
    /// Cross-attention q/k/v/o projections (`dec.blk.N.cross_attn_*`).
    CrossAttnQ,
    CrossAttnK,
    CrossAttnV,
    CrossAttnO,
    /// Cross-attention input norm (`dec.blk.N.cross_attn_norm`).
    CrossAttnNorm,
    /// Relative position bias table (`attn_rel_b`, `[n_buckets, n_heads]`).
    RelPosBias,
    /// Fused QKV projection (`attn_qkv.weight` / `qkv_proj.weight`).
    AttnQkv,
    /// `attn_q_norm.weight` — per-head query norm (Gemma4).
    AttnQNorm,
    /// `attn_k_norm.weight` — per-head key norm (Gemma4).
    AttnKNorm,
    /// `post_attention_norm.weight` — attn-output norm before residual/FFN.
    PostAttnNorm,
    /// `ffn_norm.weight` — FFN input norm.
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    /// Fused gate+up (`ffn_gate_up.weight` / HF `gate_up_proj`): rows concat
    /// `[gate | up]`, split by rows at pack load.
    FfnGateUp,
    /// FFN sparse DAG (GGUF disperso de `saor`): bit-tensor de adyacencia
    /// (`ffn_dag_adjacency`, I8 bytes, LSB-first).
    FfnDagAdjacency,
    /// FFN sparse DAG (GGUF disperso de `saor`): pesos activos (`ffn_dag_weights`,
    /// F32, orden i-mayor, solo conexiones vivas).
    FfnDagWeights,
    /// `post_ffw_norm.weight` / `post_mlp_norm.weight` — norm after FFN (Gemma4 12B).
    PostFfnNorm,
    /// `layer_output_scale.weight` — per-block output scale (Gemma4 12B).
    LayerOutputScale,
    /// MoE router — `ffn_gate_inp.weight` / HF `block_sparse_moe.gate.weight`.
    Router,
    /// MoE router selection bias (`ffn_gate_inp.bias`, DeepSeek V3
    /// `e_score_correction_bias`): preloaded, applied to the choice scores.
    RouterBias,
    /// MoE router input scale (`ffn_gate_inp.scale`): preloaded (passive here).
    RouterScale,
    /// Per-expert scale (`ffn_*_exps.scale`, length `n_expert`): preloaded.
    ExpertScale,
    /// Per-expert FFN matrices — `ffn_exp.E.ffn_gate/up/down` / HF `experts.E.w1/w2/w3`.
    ExpertGate,
    ExpertUp,
    ExpertDown,
    /// Fused per-expert gate+up 3D tensor (`ffn_gate_up_exps.weight`): split by rows
    /// into `ExpertGate`/`ExpertUp` at plan time.
    ExpertGateUp,
    /// Shared expert — `ffn_shexp.*` / `shared_expert.*` (DeepSeek-style).
    SharedExpert,
    /// Gated DeltaNet / SSM / linear-attention family (Qwen3.5 hybrid, etc.).
    DeltaNet,
    /// Mamba-1 selective-scan SSM (`ssm_in`/`ssm_x`/`ssm_d`, full selective scan).
    Mamba,
    /// Multi-token prediction / next-n draft head (Qwen3.5 `blk.*.nextn.*`).
    NextN,
    /// Gemma4 per-layer embeddings (PLE).
    PleEmbed,
    PleModelProj,
    PleProjNorm,
    /// PLE block gate (`blk.N.inp_gate.weight` when per-layer dims > 0).
    PleGate,
    /// PLE block projection (`blk.N.proj.weight`).
    PleProj,
    /// PLE block post-norm (`blk.N.post_norm.weight` / `per_layer_final_norm`).
    PlePostNorm,
    /// HRM / recurrent outer-loop state tensors.
    Recurrence,
    /// Conv / positional extras still recognized.
    Conv,
    /// Genuinely passive aux (rope freqs, biases, scales, ambiguous `.proj.weight`).
    Aux,
}

/// Hardware class an op is bound to (the executor consumes this, not prose).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpDevice {
    /// CPU SIMD / small GEMV (attention, norms, router).
    Cpu,
    /// Async GEMV on the OpenCL pool (FFN, experts, output head).
    GpuAsync,
    /// Row-oriented read from host/disk (embeddings).
    HostRowRead,
    /// Passive tensor: cataloged but not computed in the hot path.
    Discard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKernel {
    Gemv,
    GemvAsync,
    RmsNorm,
    Row,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpBinding {
    pub device: OpDevice,
    pub kernel: OpKernel,
}

impl OpBinding {
    pub const CPU_GEMV: Self = Self {
        device: OpDevice::Cpu,
        kernel: OpKernel::Gemv,
    };
    pub const CPU_NORM: Self = Self {
        device: OpDevice::Cpu,
        kernel: OpKernel::RmsNorm,
    };
    pub const GPU_ASYNC: Self = Self {
        device: OpDevice::GpuAsync,
        kernel: OpKernel::GemvAsync,
    };
    pub const HOST_ROW: Self = Self {
        device: OpDevice::HostRowRead,
        kernel: OpKernel::Row,
    };
    pub const DISCARD: Self = Self {
        device: OpDevice::Discard,
        kernel: OpKernel::None,
    };
}

impl std::fmt::Display for OpDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            OpDevice::Cpu => "CPU",
            OpDevice::GpuAsync => "GPU-async",
            OpDevice::HostRowRead => "HOST-row",
            OpDevice::Discard => "discard",
        };
        write!(f, "{s}")
    }
}

/// Hardware binding per op (data, not prose): FFN/experts → GPU pool async,
/// embeddings → host row reads, attention/norms/router → CPU (PRD).
pub fn op_binding(kind: LayerOpKind) -> OpBinding {
    use LayerOpKind::*;
    match kind {
        TokenEmbed | PleEmbed => OpBinding::HOST_ROW,
        PositionEmbed => OpBinding::HOST_ROW,
        OutputNorm | AttnNorm | AttnQNorm | AttnKNorm | PostAttnNorm | FfnNorm | PostFfnNorm
        | PleProjNorm | PlePostNorm | TokenEmbedNorm | MlaQaNorm | MlaKvANorm => OpBinding::CPU_NORM,
        AttnQ | AttnK | AttnV | AttnO | AttnGate | AttnQkv | PleGate | PleProj | Router
        | DeltaNet | Mamba | NextN | Recurrence | Conv | MlaQa | MlaQb | MlaKvA | MlaKb
        | MlaVb | MlaKvB | CrossAttnQ | CrossAttnK | CrossAttnV | CrossAttnO => OpBinding::CPU_GEMV,
        FfnGate | FfnUp | FfnDown | FfnGateUp | ExpertGate | ExpertUp | ExpertDown
        | ExpertGateUp | SharedExpert | OutputProj | PleModelProj => OpBinding::GPU_ASYNC,
        FfnDagAdjacency | FfnDagWeights => OpBinding::GPU_ASYNC,
        LayerOutputScale | Aux | RouterBias | RouterScale | ExpertScale | EncOutputNorm
        | DecOutputNorm | CrossAttnNorm | RelPosBias => OpBinding::DISCARD,
    }
}

/// A tensor that could not be classified — only valid arch-block reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLayerOp {
    pub tensor_name: String,
    pub hint: String,
}

/// Self-describing tensor inside a streaming unit: name, op role, and enough GGUF
/// info to load it into a scratch slot and re-derive `QuantMatrix` views.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub op: LayerOpKind,
    pub nbytes: usize,
    pub ggml_type: hayai_model::GgmlType,
    pub ncols: usize,
    pub nrows: usize,
    /// Byte offset of this tensor within its streaming unit's buffer (dst layout).
    pub offset: usize,
    /// In-tensor byte offset for sliced reads (fused MoE experts); 0 for whole tensors.
    pub src_off: usize,
    /// Tensor dimensionality (2D matmul vs 3D fused-expert).
    pub ndims: usize,
}

/// MoE metadata from GGUF keys (`{arch}.expert_count` /
/// `{arch}.attention.expert_used_count`), with llama.cpp/HF fallbacks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeMeta {
    pub expert_count: usize,
    pub top_k: usize,
    /// Grouped routing (`{arch}.expert_group_count` / `expert_group_used_count`);
    /// 0 = no grouping (Mixtral/Qwen-MoE).
    pub n_group: usize,
    pub topk_group: usize,
    /// `{arch}.expert_gating_func`: 0 = softmax, 1 = sigmoid (DeepSeek V3/GLM).
    pub gating: u32,
    /// `{arch}.expert_weights_norm` (`norm_topk_prob`).
    pub norm_topk_prob: bool,
    /// `{arch}.expert_weights_scale` (`routed_scaling_factor`).
    pub routed_scaling_factor: f32,
}

impl Default for MoeMeta {
    fn default() -> Self {
        Self {
            expert_count: 0,
            top_k: 1,
            n_group: 0,
            topk_group: 0,
            gating: 0,
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
        }
    }
}

/// One MoE expert's tensor group. Offsets are relative to the expert pack start.
#[derive(Debug, Clone)]
pub struct ExpertUnit {
    pub expert_id: usize,
    /// True when the expert is a byte-slice of a fused 3D tensor (`ffn_*_exps.weight`).
    pub fused: bool,
    pub tensors: Vec<TensorRef>,
}

#[derive(Debug, Clone)]
pub struct StreamingUnit {
    /// Logical block id (e.g. layer index) when detectable.
    pub block_id: Option<usize>,
    pub tensors: Vec<TensorRef>,
    /// Streaming window bytes (sparse for MoE: attn/router + top_k × expert).
    pub total_bytes: usize,
    /// Full block bytes (all experts) — used as the resident stride / preload size.
    pub full_bytes: usize,
    /// MoE: per-expert tensor groups (empty for dense blocks).
    pub experts: Vec<ExpertUnit>,
    /// MoE: bytes of non-expert tensors (attn + router + shared expert).
    pub non_expert_bytes: usize,
    /// MoE: largest single expert pack (bytes).
    pub max_expert_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct ExecPlan {
    pub architecture: String,
    pub units: Vec<StreamingUnit>,
    /// Sparse streaming window (max unit bytes).
    pub max_unit_bytes: usize,
    /// Full resident block bytes (all experts for MoE) — resident stride / preload.
    pub max_full_block_bytes: usize,
    pub known_ops: BTreeSet<LayerOpKind>,
    /// MoE metadata (router/expert layout) when the model is sparse.
    pub moe: Option<MoeMeta>,
    /// Per-op hardware binding (executor consumes this).
    pub op_bindings: BTreeMap<LayerOpKind, OpBinding>,
    pub hw_notes: Vec<String>,
}

/// Build a streaming execution plan from GGUF metadata + HW hints.
pub fn build_exec_plan(
    catalog: &GgufCatalog,
    opencl_devices: usize,
    supports_svm: bool,
) -> Result<ExecPlan, UnknownLayerOp> {
    let architecture = catalog
        .meta_str("general.architecture")
        .unwrap_or("unknown")
        .to_string();

    let mut unknowns = Vec::new();
    let mut classified = Vec::new();
    for t in &catalog.tensors {
        match classify_tensor(catalog, &t.name) {
            Ok(op) => {
                let nbytes = hayai_model::tensor_nbytes(t).unwrap_or(0);
                classified.push(TensorRef {
                    name: t.name.clone(),
                    op,
                    nbytes,
                    ggml_type: t.ggml_type,
                    ncols: t.ncols(),
                    nrows: t.nrows(),
                    offset: 0,
                    src_off: 0,
                    ndims: t.dims.len(),
                });
            }
            Err(hint) => unknowns.push(UnknownLayerOp {
                tensor_name: t.name.clone(),
                hint,
            }),
        }
    }
    if let Some(u) = unknowns.into_iter().next() {
        return Err(u);
    }

    let mut known_ops = BTreeSet::new();
    for c in &classified {
        known_ops.insert(c.op);
    }

    let op_bindings = known_ops.iter().map(|&op| (op, op_binding(op))).collect();

    // Group by blk.N / layers.N when present; otherwise one unit per tensor role cluster.
    // Tensor offsets are assigned in catalog order so units are self-describing:
    // `load_tensors_into` can stream them straight into a scratch slot.
    let moe = detect_moe_meta(catalog);

    let mut units: Vec<StreamingUnit> = Vec::new();
    let mut by_block: std::collections::BTreeMap<Option<usize>, Vec<TensorRef>> =
        std::collections::BTreeMap::new();
    for c in classified {
        let bid = parse_block_id(&c.name);
        by_block.entry(bid).or_default().push(c);
    }
    for (block_id, mut tensors) in by_block {
        // Base offset layout: catalog order (executor re-packs experts on demand).
        let mut off = 0usize;
        for t in tensors.iter_mut() {
            t.offset = off;
            off += t.nbytes;
        }
        let mut full_bytes = off; // resident preload needs the whole block
        // MoE: split expert tensors into per-expert packs; the streaming window is
        // attn/router + top_k × largest expert — sparse disk streaming, not the
        // whole block.
        let mut experts: Vec<ExpertUnit> = Vec::new();
        let mut non_expert_bytes = off;
        let mut max_expert_bytes = 0usize;
        if moe.is_some() && tensors.iter().any(|t| is_expert_op(t.op)) {
            let mm = moe.expect("guarded above");
            let n_exp = mm.expert_count.max(1);
            let mut by_exp: std::collections::BTreeMap<usize, Vec<TensorRef>> =
                std::collections::BTreeMap::new();
            let mut fused: Vec<(TensorRef, usize)> = Vec::new(); // (tensor, slice_bytes)
            for t in tensors.iter() {
                if let Some(eid) = parse_expert_id(&t.name) {
                    by_exp.entry(eid).or_default().push(t.clone());
                } else if is_expert_op(t.op) && t.ndims >= 3 {
                    fused.push((t.clone(), t.nbytes / n_exp));
                }
            }
            non_expert_bytes = tensors
                .iter()
                .filter(|t| !is_expert_op(t.op))
                .map(|t| t.nbytes)
                .sum();
            if fused.is_empty() {
                // Per-expert tensors (`ffn_exp.E.*` / HF `experts.E.*`).
                for (eid, mut ets) in by_exp {
                    let mut eoff = 0usize;
                    for t in ets.iter_mut() {
                        t.offset = eoff;
                        eoff += t.nbytes;
                    }
                    max_expert_bytes = max_expert_bytes.max(eoff);
                    experts.push(ExpertUnit {
                        expert_id: eid,
                        fused: false,
                        tensors: ets,
                    });
                }
            } else {
                // Fused 3D experts (`ffn_*_exps.weight`, OLMoE-style): slice each
                // expert's byte range; pack gate@0 / up@slice / down@2*slice.
                fused.sort_by_key(|(t, _)| fused_role_rank(t.op));
                for e in 0..n_exp {
                    let mut ets: Vec<TensorRef> = Vec::new();
                    for (t, slice) in &fused {
                        if t.op == LayerOpKind::ExpertGateUp {
                            // Rows concat [gate | up]: split into two views.
                            let half = *slice / 2;
                            let rows_half = (t.nrows / 2).max(1);
                            for (op, off) in
                                [(LayerOpKind::ExpertGate, 0usize), (LayerOpKind::ExpertUp, half)]
                            {
                                ets.push(TensorRef {
                                    name: t.name.clone(),
                                    op,
                                    nbytes: half,
                                    ggml_type: t.ggml_type,
                                    ncols: t.ncols,
                                    nrows: rows_half,
                                    offset: 0,
                                    src_off: e * slice + off,
                                    ndims: 2,
                                });
                            }
                        } else {
                            ets.push(TensorRef {
                                name: t.name.clone(),
                                op: t.op,
                                nbytes: *slice,
                                ggml_type: t.ggml_type,
                                ncols: t.ncols,
                                nrows: t.nrows,
                                offset: 0,
                                src_off: e * slice,
                                ndims: 2,
                            });
                        }
                    }
                    let mut eoff = 0usize;
                    for ts in ets.iter_mut() {
                        ts.offset = eoff;
                        eoff += ts.nbytes;
                    }
                    max_expert_bytes = max_expert_bytes.max(eoff);
                    experts.push(ExpertUnit {
                        expert_id: e,
                        fused: true,
                        tensors: ets,
                    });
                }
            }
            experts.sort_by_key(|e| e.expert_id);
            let top_k = mm.top_k.min(mm.expert_count).max(1);
            full_bytes = non_expert_bytes + mm.expert_count * max_expert_bytes;
            off = non_expert_bytes + top_k * max_expert_bytes;
        }
        units.push(StreamingUnit {
            block_id,
            tensors,
            total_bytes: off,
            full_bytes,
            experts,
            non_expert_bytes,
            max_expert_bytes,
        });
    }
    units.sort_by_key(|u| u.block_id.unwrap_or(usize::MAX));

    // Scratch window = max transformer block, not embed/lm_head (those stream separately).
    let max_unit_bytes = units
        .iter()
        .filter(|u| u.block_id.is_some())
        .map(|u| u.total_bytes)
        .max()
        .or_else(|| units.iter().map(|u| u.total_bytes).max())
        .unwrap_or(0);
    let max_full_block_bytes = units
        .iter()
        .filter(|u| u.block_id.is_some())
        .map(|u| u.full_bytes)
        .max()
        .unwrap_or(max_unit_bytes);

    let mut hw_notes = Vec::new();
    if opencl_devices == 0 {
        hw_notes.push("CPU-only FFN (empty OpenCL pool)".into());
    } else {
        hw_notes.push(format!(
            "OpenCL pool size={opencl_devices}; Attn/RoPE/KV→CPU; large GEMV→GPU"
        ));
        if supports_svm {
            hw_notes.push("SVM host base preferred for APU zero-copy".into());
        } else {
            hw_notes.push("Pinned host + FFN-slice DMA to dGPU mirrors".into());
        }
    }
    hw_notes.push(format!(
        "Ping-pong scratch ≥ 2 × max_streaming_unit ({} KiB)",
        max_unit_bytes / 1024
    ));

    Ok(ExecPlan {
        architecture,
        units,
        max_unit_bytes,
        max_full_block_bytes,
        known_ops,
        moe,
        op_bindings,
        hw_notes,
    })
}

fn parse_block_id(name: &str) -> Option<usize> {
    // Disjoint block-id base for the decoder stack of encoder-decoder models.
    pub(crate) const DEC_BLOCK_OFFSET: usize = 100_000;
    // Encoder-decoder (T5/BART): `enc.blk.N.*` / `dec.blk.N.*`. The decoder stack is
    // offset into a disjoint id range so encoder/decoder units never collide.
    for prefix in ["enc.blk.", "encoder.blk."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = num.parse() {
                return Some(v);
            }
        }
    }
    for prefix in ["dec.blk.", "decoder.blk."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = num.parse::<usize>() {
                return Some(DEC_BLOCK_OFFSET + v);
            }
        }
    }
    // blk.12.attn_q.weight / model.layers.3.mlp.down_proj.weight / etc.
    for prefix in ["blk.", "model.layers.", "layers."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = num.parse() {
                return Some(v);
            }
        }
    }
    // HRM stacks: often h_layers.N / l_layers.N
    for prefix in ["h_layers.", "l_layers.", "H.", "L."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = num.parse() {
                return Some(v);
            }
        }
    }
    None
}

fn per_layer_dim(cat: &GgufCatalog) -> usize {
    cat.meta_str("general.architecture")
        .and_then(|arch| cat.meta_u32(&format!("{arch}.embedding_length_per_layer_input")))
        .unwrap_or(0) as usize
}

/// Classify one GGUF tensor into a known op. Metadata-dependent decisions
/// (e.g. Gemma4 `inp_gate` = PLE gate only when per-layer embeddings exist) use
/// the catalog; unknown roles hard-fail (the only arch-block reason).
pub fn classify_tensor(cat: &GgufCatalog, name: &str) -> Result<LayerOpKind, String> {
    let has_ple = per_layer_dim(cat) > 0;
    classify_tensor_impl(name, has_ple)
}

/// Pure-name classifier (unit tests). `has_ple` toggles Gemma4 PLE tensor roles.
pub fn classify_tensor_impl(name: &str, has_ple: bool) -> Result<LayerOpKind, String> {
    use LayerOpKind::*;
    let n = name.to_ascii_lowercase();

    // ── Bias tensors ─────────────────────────────────────────────────────────
    // Compute biases are preloaded once per model and applied by the execution
    // paths (attention q/k/v/qkv/o, FFN gate/up/down, output). Norm biases
    // (`*_norm.bias`) fall through to their norm op below (LayerNorm).
    if n.contains("bias") {
        let is_compute_bias = n.ends_with("attn_q.bias")
            || n.ends_with("attn_k.bias")
            || n.ends_with("attn_v.bias")
            || n.ends_with("attn_output.bias")
            || n.ends_with("attn_qkv.bias")
            || n.ends_with("q_proj.bias")
            || n.ends_with("k_proj.bias")
            || n.ends_with("v_proj.bias")
            || n.ends_with("o_proj.bias")
            || n.ends_with("qkv_proj.bias")
            || n.ends_with("ffn_gate.bias")
            || n.ends_with("ffn_up.bias")
            || n.ends_with("ffn_down.bias")
            || n.ends_with("gate_proj.bias")
            || n.ends_with("up_proj.bias")
            || n.ends_with("down_proj.bias")
            || n.ends_with("output.bias")
            || n.ends_with("lm_head.bias")
            || n.ends_with("token_embd.bias");
        if is_compute_bias {
            return Ok(Aux);
        }
    }

    // ── Phase 0: draft heads (before any attn/ffn substring). ────────────────
    if n.contains(".nextn.") || n.contains("nextn_") || n.contains(".mtp.") || n.contains("mtp_") {
        return Ok(NextN);
    }

    // ── Phase 1: Gemma4 per-layer embeddings (before generic token_embd). ────
    if n.contains("per_layer_token_embd") {
        return Ok(if has_ple { PleEmbed } else { TokenEmbed });
    }
    if n.contains("per_layer_model_proj") {
        return Ok(PleModelProj);
    }
    if n.contains("per_layer_proj_norm") {
        return Ok(PleProjNorm);
    }
    if n.contains("per_layer_final_norm") {
        return Ok(PlePostNorm);
    }

    // ── Phase 1b: encoder-decoder (T5/BART) — before generic attn/norm/output. ─
    if n.contains("enc.output_norm") || n.contains("encoder.final_layer_norm") {
        return Ok(EncOutputNorm);
    }
    if n.contains("dec.output_norm") || n.contains("decoder.final_layer_norm") {
        return Ok(DecOutputNorm);
    }
    if n.contains("cross_attn_norm") || n.contains("cross_attention_norm") {
        return Ok(CrossAttnNorm);
    }
    if n.contains("cross_attn_q") || n.contains("enc_dec_attn_q") {
        return Ok(CrossAttnQ);
    }
    if n.contains("cross_attn_k") || n.contains("enc_dec_attn_k") {
        return Ok(CrossAttnK);
    }
    if n.contains("cross_attn_v") || n.contains("enc_dec_attn_v") {
        return Ok(CrossAttnV);
    }
    if n.contains("cross_attn_o") || n.contains("enc_dec_attn_o") {
        return Ok(CrossAttnO);
    }
    if n.contains("attn_rel_b") || n.contains("relative_attention_bias") || n.contains("rel_attn_bias")
    {
        return Ok(RelPosBias);
    }

    // ── Phase 2: global embeddings / output head. ────────────────────────────
    // Input-embedding LayerNorm (BLOOM `token_embd_norm`/`word_embeddings_layernorm`).
    if n.contains("token_embd_norm")
        || n.contains("word_embeddings_layernorm")
        || n.contains("embed_norm")
    {
        return Ok(TokenEmbedNorm);
    }
    if n.contains("token_embd")
        || n.ends_with("embed_tokens.weight")
        || n.contains("tok_embeddings")
        || n.contains("word_embeddings")
        || n.contains("embed_in")
        || n.ends_with("wte.weight")
    {
        return Ok(TokenEmbed);
    }
    if n.contains("output_norm")
        || n.ends_with("model.norm.weight")
        || n == "norm.weight"
        || n.contains("final_layer_norm")
        || n.contains("final_layernorm")
        || n.ends_with("ln_f.weight")
    {
        return Ok(OutputNorm);
    }
    if n == "output.weight" || n.ends_with("lm_head.weight") || n.ends_with("embed_out.weight") {
        return Ok(OutputProj);
    }
    // Learned absolute position embeddings (GPT-2 `position_embd` / `wpe`): added to
    // the input embeddings by row `position`.
    if n.ends_with("wpe.weight")
        || n.contains("position_embd")
        || n.contains("position_embedding")
        || n.ends_with("pos_embd.weight")
    {
        return Ok(PositionEmbed);
    }

    // ── Phase 3: MoE router — before dense FFN (`ffn_gate_inp` ⊃ `ffn_gate`). ─
    // Router bias/scale and per-expert scales are preloaded out-of-band (they are
    // not expert matrices) — classify them before the generic router/expert arms.
    if n.ends_with("ffn_gate_inp.bias") {
        return Ok(RouterBias);
    }
    if n.ends_with("ffn_gate_inp.scale") {
        return Ok(RouterScale);
    }
    if n.ends_with("_exps.scale") || n.ends_with("exps.scale") {
        return Ok(ExpertScale);
    }
    if n.contains("router")
        || n.contains("gate_inp")
        || n.contains("block_sparse_moe.gate")
        || n.contains("moe.gate")
        || n.contains("mlp.gate.weight")
    {
        return Ok(Router);
    }

    // ── Phase 4: MoE experts / shared expert — before dense FFN. ─────────────
    if n.contains("shared_expert") || n.contains("ffn_shexp") || n.contains("shexp") {
        return Ok(SharedExpert);
    }
    if n.contains("ffn_exp") || n.contains("experts.") || n.contains("exps") {
        if n.contains("gate_up") || n.contains("gateup") {
            return Ok(ExpertGateUp);
        }
        return Ok(expert_kind(&n));
    }

    // ── Phase 4b: MLA (DeepSeek-V2/V3, Kimi) — before generic attn/norm names. ─
    if n.contains("attn_q_a_norm") {
        return Ok(MlaQaNorm);
    }
    if n.contains("attn_kv_a_norm") {
        return Ok(MlaKvANorm);
    }
    if n.contains("attn_q_a") || n.contains("wq_a.weight") {
        return Ok(MlaQa);
    }
    if n.contains("attn_q_b") || n.contains("wq_b.weight") {
        return Ok(MlaQb);
    }
    if n.contains("attn_kv_a_mqa") {
        return Ok(MlaKvA);
    }
    if n.contains("attn_kv_b") {
        return Ok(MlaKvB);
    }
    if n.contains("attn_k_b") {
        return Ok(MlaKb);
    }
    if n.contains("attn_v_b") {
        return Ok(MlaVb);
    }

    // ── Phase 5: norms — before projections (`attn_q_norm` ⊃ `attn_q`). ───────
    if n.contains("post_attention_norm")
        || n.contains("post_attention_layernorm")
        || n.contains("ffn_norm")
        || n.ends_with("norm_2.weight")
        || n.ends_with("ln_2.weight")
    {
        return Ok(FfnNorm);
    }
    if n.contains("post_ffw_norm") || n.contains("post_mlp_norm") {
        return Ok(PostFfnNorm);
    }
    if n.contains("post_norm") {
        return Ok(if has_ple { PlePostNorm } else { PostFfnNorm });
    }
    if n.contains("attn_norm")
        || n.contains("input_layernorm")
        || n.contains("attention_norm")
        || n.ends_with("norm_1.weight")
        || n.ends_with("ln_1.weight")
    {
        return Ok(AttnNorm);
    }
    if n.contains("q_norm") {
        return Ok(AttnQNorm);
    }
    if n.contains("k_norm") {
        return Ok(AttnKNorm);
    }

    // ── Phase 5c: FFN disperso (DAG irregular, GGUF de saor) — antes del FFN denso. ─
    if n.contains("ffn_dag_adjacency") {
        return Ok(FfnDagAdjacency);
    }
    if n.contains("ffn_dag_weights") {
        return Ok(FfnDagWeights);
    }

    // ── Phase 6: dense FFN. ──────────────────────────────────────────────────
    // Fused gate+up (`gate_up_proj` / `ffn_gate_up`): split into gate/up by rows
    // at pack load (`load_layer_pack_into`).
    if n.contains("gate_up") || n.contains("gateup") {
        return Ok(FfnGateUp);
    }
    if n.contains("ffn_gate") || n.contains("gate_proj") {
        return Ok(FfnGate);
    }
    if n.contains("ffn_up") || n.contains("up_proj") || n.contains("dense_h_to_4h") {
        return Ok(FfnUp);
    }
    if n.contains("ffn_down") || n.contains("down_proj") || n.contains("dense_4h_to_h") {
        return Ok(FfnDown);
    }

    // ── Phase 7: attention projections. ──────────────────────────────────────
    if n.contains("attn_qkv")
        || n.contains("qkv_proj")
        || n.contains("wqkv")
        || n.contains("query_key_value")
        || n.contains("c_attn")
    {
        return Ok(AttnQkv);
    }
    if n.contains("attn_q") || n.contains("q_proj") || n.contains(".wq.") {
        return Ok(AttnQ);
    }
    if n.contains("attn_k") || n.contains("k_proj") || n.contains(".wk.") {
        return Ok(AttnK);
    }
    if n.contains("attn_v") || n.contains("v_proj") || n.contains(".wv.") {
        return Ok(AttnV);
    }
    // `out_proj` is an attention-output alias (MPT `attn.out_proj`) — but linear
    // attention (`linear_attn.out_proj`) belongs to DeltaNet, so guard it.
    let delta_like = n.contains("linear_attn")
        || n.contains("ssm")
        || n.contains("shortconv")
        || n.contains("delta")
        || n.contains("mamba");
    if n.contains("attn_output")
        || n.contains("o_proj")
        || n.contains("attn_out")
        || n.contains("attn_o.")
        || n.contains(".wo.")
        || n.contains("self_attention.dense")
        || n.contains("attention.dense")
        || (n.contains("out_proj") && !delta_like)
    {
        return Ok(AttnO);
    }
    if n.contains("inp_gate") {
        return Ok(if has_ple { PleGate } else { AttnGate });
    }
    if n.contains("attn_gate") || n.contains("attn.gate") {
        return Ok(AttnGate);
    }

    // ── Phase 8: per-block output scale (before generic `scale`). ─────────────
    if n.contains("layer_output_scale") {
        return Ok(LayerOutputScale);
    }

    // ── Phase 9: SSM / DeltaNet / linear attention. ──────────────────────────
    // Mamba-1 is distinguished by its x_proj / D-skip / in-proj tensors (Qwen3.5
    // DeltaNet blocks use `ssm_alpha`/`ssm_beta`/`ssm_norm`/`in_proj_qkvz` instead).
    if n.contains("ssm_x") || n.contains("ssm_in") || n.ends_with(".ssm_d") {
        return Ok(Mamba);
    }
    if n.contains("delta")
        || n.contains("linear_attn")
        || n.contains("mamba")
        || n.contains("ssm")
        || n.contains("conv1d")
        || n.contains("in_proj_qkv")
        || n.contains("in_proj_ba")
        || (n.contains("out_proj") && n.contains("shortconv"))
    {
        return Ok(DeltaNet);
    }

    // ── Phase 10: PLE in-block projections (Gemma4 with per-layer dims). ──────
    if has_ple && n.contains(".proj.weight") {
        return Ok(PleProj);
    }

    // ── Phase 11: recurrence / HRM outer-loop tensors. ────────────────────────
    if n.contains("h_cycle")
        || n.contains("l_cycle")
        || n.contains("recurrent")
        || n.starts_with("h.")
        || n.starts_with("l.")
        || n.contains("z_h")
        || n.contains("z_l")
        || n.contains("h_layers")
        || n.contains("l_layers")
    {
        return Ok(Recurrence);
    }

    // ── Phase 12: conv / positional extras. ───────────────────────────────────
    if n.contains("conv") {
        return Ok(Conv);
    }

    // ── Phase 13: genuinely passive aux. ──────────────────────────────────────
    // Compute-affecting tensors we do NOT execute must fail loudly: silently
    // discarding them produces wrong logits with no diagnostic.
    if n.contains("attn_sink") || n.contains("shear") {
        return Err(format!(
            "compute-affecting tensor not implemented in the streaming path: {name}"
        ));
    }
    if n.ends_with(".bias")
        || n.ends_with("_bias")
        || n.ends_with(".scale")
        || n.ends_with("_scale")
        || n.contains("rope")
        || n.contains("cos")
        || n.contains("sin")
        || n.contains("inv_freq")
        || n.contains("alibi")
    {
        return Ok(Aux);
    }

    // ── Phase 14: hard failures (the only arch-block reasons). ────────────────
    if n.contains("mmproj") || n.contains("vision") || n.contains("clip") {
        return Err(format!(
            "vision/multimodal tensor not in text streaming registry yet: {name}"
        ));
    }

    Err(format!(
        "unrecognized tensor role (register a LayerOpKind or fix naming): {name}"
    ))
}

/// Map an expert-family tensor name to its gate/up/down role.
fn expert_kind(n: &str) -> LayerOpKind {
    if n.contains("ffn_down") || n.contains("down_proj") || n.contains(".w2.") {
        LayerOpKind::ExpertDown
    } else if n.contains("ffn_up") || n.contains("up_proj") || n.contains(".w3.") {
        LayerOpKind::ExpertUp
    } else {
        LayerOpKind::ExpertGate
    }
}

/// Parse the expert id from `blk.N.ffn_exp.E.*` / HF `experts.E.*` tensor names.
pub(crate) fn parse_expert_id(name: &str) -> Option<usize> {
    for marker in ["ffn_exp.", "experts."] {
        if let Some(idx) = name.find(marker) {
            let rest = &name[idx + marker.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() {
                return digits.parse().ok();
            }
        }
    }
    None
}

pub(crate) fn is_expert_op(op: LayerOpKind) -> bool {
    matches!(
        op,
        LayerOpKind::ExpertGate
            | LayerOpKind::ExpertUp
            | LayerOpKind::ExpertDown
            | LayerOpKind::ExpertGateUp
    )
}

/// Pack order for fused expert slices: gate@0, up@slice, down@2*slice.
fn fused_role_rank(op: LayerOpKind) -> usize {
    match op {
        LayerOpKind::ExpertGate | LayerOpKind::ExpertGateUp => 0,
        LayerOpKind::ExpertUp => 1,
        LayerOpKind::ExpertDown => 2,
        _ => 3,
    }
}

/// Detect MoE metadata from llama.cpp / HF GGUF keys.
pub fn detect_moe_meta(cat: &GgufCatalog) -> Option<MoeMeta> {
    let arch = cat.meta_str("general.architecture").unwrap_or("");
    let expert_count = cat
        .meta_u32(&format!("{arch}.expert_count"))
        .or_else(|| cat.meta_u32("llama.expert_count"))
        .map(|v| v as usize);
    let top_k = cat
        .meta_u32(&format!("{arch}.attention.expert_used_count"))
        .or_else(|| cat.meta_u32(&format!("{arch}.expert_used_count")))
        .or_else(|| cat.meta_u32("llama.attention.expert_used_count"))
        .map(|v| v as usize);
    match (expert_count, top_k) {
        (Some(n), Some(k)) if n > 0 && k > 0 => {
            let n_group = cat
                .meta_u32(&format!("{arch}.expert_group_count"))
                .or_else(|| cat.meta_u32("llama.expert_group_count"))
                .unwrap_or(0) as usize;
            let topk_group = cat
                .meta_u32(&format!("{arch}.expert_group_used_count"))
                .or_else(|| cat.meta_u32("llama.expert_group_used_count"))
                .unwrap_or(0) as usize;
            let gating = cat
                .meta_u32(&format!("{arch}.expert_gating_func"))
                .or_else(|| cat.meta_u32("llama.expert_gating_func"))
                .unwrap_or(0);
            let norm_topk_prob = cat
                .meta_bool(&format!("{arch}.expert_weights_norm"))
                .or_else(|| cat.meta_bool("llama.expert_weights_norm"))
                .unwrap_or(false);
            let routed_scaling_factor = cat
                .meta_f32(&format!("{arch}.expert_weights_scale"))
                .or_else(|| cat.meta_f32("llama.expert_weights_scale"))
                .unwrap_or(1.0);
            Some(MoeMeta {
                expert_count: n,
                top_k: k,
                n_group,
                topk_group,
                gating,
                norm_topk_prob,
                routed_scaling_factor,
            })
        }
        _ => None,
    }
}

impl ExecPlan {
    pub fn format_report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("architecture: {}\n", self.architecture));
        s.push_str(&format!(
            "streaming_units: {} | max_unit: {} KiB\n",
            self.units.len(),
            self.max_unit_bytes / 1024
        ));
        s.push_str("known_ops: ");
        for (i, op) in self.known_ops.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{op:?}"));
        }
        s.push('\n');
        for note in &self.hw_notes {
            s.push_str(&format!("hw: {note}\n"));
        }
        s.push_str("op_bindings:\n");
        for (op, b) in &self.op_bindings {
            s.push_str(&format!("  {op:?} → {} ({:?})\n", b.device, b.kernel));
        }
        if let Some(moe) = self.moe {
            s.push_str(&format!(
                "moe: experts={} top_k={} (sparse: ~{:.1}% of FFN bytes/token)\n",
                moe.expert_count,
                moe.top_k,
                100.0 * moe.top_k as f64 / moe.expert_count.max(1) as f64
            ));
        }
        for u in self.units.iter().take(8) {
            s.push_str(&format!(
                "  unit block={:?} tensors={} bytes={}\n",
                u.block_id,
                u.tensors.len(),
                u.total_bytes
            ));
        }
        if self.units.len() > 8 {
            s.push_str(&format!("  ... {} more units\n", self.units.len() - 8));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cls(name: &str) -> Result<LayerOpKind, String> {
        classify_tensor_impl(name, false)
    }

    fn cls_ple(name: &str) -> Result<LayerOpKind, String> {
        classify_tensor_impl(name, true)
    }

    #[test]
    fn classifies_llama_tensors() {
        assert_eq!(cls("blk.0.attn_q.weight").unwrap(), LayerOpKind::AttnQ);
        assert_eq!(cls("blk.3.ffn_down.weight").unwrap(), LayerOpKind::FfnDown);
        assert_eq!(cls("token_embd.weight").unwrap(), LayerOpKind::TokenEmbed);
        // Fused QKV under either GGUF (`attn_qkv`) or HF (`qkv_proj`) naming.
        assert_eq!(
            cls("model.layers.0.self_attn.qkv_proj.weight").unwrap(),
            LayerOpKind::AttnQkv
        );
        // MPT `attn.Wqkv.weight` is concat [q|k|v] — same op.
        assert_eq!(cls("blk.0.attn.Wqkv.weight").unwrap(), LayerOpKind::AttnQkv);
    }

    #[test]
    fn attention_biases_are_aux_unhandled_biases_fail() {
        // Qwen2/2.5: q/k/v biases are preloaded and applied out-of-band.
        assert_eq!(cls("blk.0.attn_q.bias").unwrap(), LayerOpKind::Aux);
        assert_eq!(cls("blk.0.attn_k.bias").unwrap(), LayerOpKind::Aux);
        assert_eq!(cls("blk.0.attn_v.bias").unwrap(), LayerOpKind::Aux);
        assert_eq!(cls("blk.0.attn_output.bias").unwrap(), LayerOpKind::Aux);
        // QKV / FFN / output biases are preloaded and applied by the Dense path.
        assert_eq!(cls("blk.0.attn_qkv.bias").unwrap(), LayerOpKind::Aux);
        assert_eq!(cls("blk.0.ffn_gate.bias").unwrap(), LayerOpKind::Aux);
        assert_eq!(cls("blk.0.ffn_up.bias").unwrap(), LayerOpKind::Aux);
        assert_eq!(cls("blk.0.ffn_down.bias").unwrap(), LayerOpKind::Aux);
        assert_eq!(cls("output.bias").unwrap(), LayerOpKind::Aux);
        // Norm biases map to their norm op (LayerNorm), not `Aux`.
        assert_eq!(cls("blk.0.attn_norm.bias").unwrap(), LayerOpKind::AttnNorm);
        assert_eq!(cls("blk.0.ffn_norm.bias").unwrap(), LayerOpKind::FfnNorm);
        assert_eq!(cls("output_norm.bias").unwrap(), LayerOpKind::OutputNorm);
    }

    #[test]
    fn unknown_tensor_fails_loudly_never_silent() {
        // Framework invariant (docs/adding-a-model-family.md): an unrecognized tensor
        // role must be an error, never a silently-ignored `Aux`.
        assert!(cls("blk.0.mystery_tensor.weight").is_err());
        assert!(cls("blk.0.wibble.weight").is_err());
    }

    #[test]
    fn classifies_mamba_ops() {
        // Mamba-1 selective-scan tensors are their own op...
        assert_eq!(cls("blk.0.ssm_in.weight").unwrap(), LayerOpKind::Mamba);
        assert_eq!(cls("blk.0.ssm_x.weight").unwrap(), LayerOpKind::Mamba);
        assert_eq!(cls("blk.0.ssm_d").unwrap(), LayerOpKind::Mamba);
        // ...while Qwen3.5 DeltaNet blocks stay `DeltaNet` (shared `ssm_*` names).
        assert_eq!(cls("blk.0.ssm_alpha.weight").unwrap(), LayerOpKind::DeltaNet);
        assert_eq!(cls("blk.0.ssm_conv1d.weight").unwrap(), LayerOpKind::DeltaNet);
        assert_eq!(cls("blk.0.ssm_dt.weight").unwrap(), LayerOpKind::DeltaNet);
    }

    #[test]
    fn classifies_qwen35_hybrid() {
        assert_eq!(cls("blk.0.ssm_out.weight").unwrap(), LayerOpKind::DeltaNet);
        assert_eq!(cls("blk.0.attn_qkv.weight").unwrap(), LayerOpKind::AttnQkv);
        assert_eq!(
            cls("blk.0.post_attention_norm.weight").unwrap(),
            LayerOpKind::FfnNorm
        );
        assert_eq!(cls("blk.0.attn_gate.weight").unwrap(), LayerOpKind::AttnGate);
        assert_eq!(
            cls("blk.32.nextn.eh_proj.weight").unwrap(),
            LayerOpKind::NextN
        );
    }

    /// Gemma4-12B `blk.0` real tensor set: every role mapped, none masked as Aux.
    #[test]
    fn classifies_gemma4_12b_block0() {
        for (name, expected) in [
            ("blk.0.attn_k.weight", LayerOpKind::AttnK),
            ("blk.0.attn_k_norm.weight", LayerOpKind::AttnKNorm),
            ("blk.0.attn_norm.weight", LayerOpKind::AttnNorm),
            ("blk.0.attn_output.weight", LayerOpKind::AttnO),
            ("blk.0.attn_q.weight", LayerOpKind::AttnQ),
            ("blk.0.attn_q_norm.weight", LayerOpKind::AttnQNorm),
            ("blk.0.attn_v.weight", LayerOpKind::AttnV),
            ("blk.0.ffn_down.weight", LayerOpKind::FfnDown),
            ("blk.0.ffn_gate.weight", LayerOpKind::FfnGate),
            ("blk.0.ffn_norm.weight", LayerOpKind::FfnNorm),
            ("blk.0.ffn_up.weight", LayerOpKind::FfnUp),
            ("blk.0.layer_output_scale.weight", LayerOpKind::LayerOutputScale),
            ("blk.0.post_attention_norm.weight", LayerOpKind::FfnNorm),
            ("blk.0.post_ffw_norm.weight", LayerOpKind::PostFfnNorm),
        ] {
            assert_eq!(cls(name).unwrap(), expected, "tensor {name}");
        }
    }

    /// Gemma4 E4B PLE tensors (per-layer embeddings active).
    #[test]
    fn classifies_gemma4_ple_when_metadata_enables_it() {
        assert_eq!(cls_ple("blk.0.inp_gate.weight").unwrap(), LayerOpKind::PleGate);
        assert_eq!(cls_ple("blk.0.proj.weight").unwrap(), LayerOpKind::PleProj);
        assert_eq!(
            cls_ple("blk.0.post_norm.weight").unwrap(),
            LayerOpKind::PlePostNorm
        );
        assert_eq!(
            cls_ple("per_layer_token_embd.weight").unwrap(),
            LayerOpKind::PleEmbed
        );
        assert_eq!(
            cls_ple("per_layer_model_proj.weight").unwrap(),
            LayerOpKind::PleModelProj
        );
        assert_eq!(
            cls_ple("per_layer_proj_norm.weight").unwrap(),
            LayerOpKind::PleProjNorm
        );
        // Without PLE metadata, `inp_gate` is an attention gate.
        assert_eq!(cls("blk.0.inp_gate.weight").unwrap(), LayerOpKind::AttnGate);
    }

    /// MoE naming families: llama.cpp GGUF (`blk.N.ffn_exp.E.*`) and HF transformers.
    #[test]
    fn classifies_moe_router_and_experts() {
        assert_eq!(
            cls("blk.0.ffn_gate_inp.weight").unwrap(),
            LayerOpKind::Router
        );
        // Router bias / scales are preloaded out-of-band, not expert matrices.
        assert_eq!(
            cls("blk.0.ffn_gate_inp.bias").unwrap(),
            LayerOpKind::RouterBias
        );
        assert_eq!(
            cls("blk.0.ffn_gate_inp.scale").unwrap(),
            LayerOpKind::RouterScale
        );
        assert_eq!(
            cls("blk.0.ffn_down_exps.scale").unwrap(),
            LayerOpKind::ExpertScale
        );
        assert_eq!(
            cls("blk.0.ffn_exp.0.ffn_gate.weight").unwrap(),
            LayerOpKind::ExpertGate
        );
        assert_eq!(
            cls("blk.0.ffn_exp.0.ffn_up.weight").unwrap(),
            LayerOpKind::ExpertUp
        );
        assert_eq!(
            cls("blk.0.ffn_exp.3.ffn_down.weight").unwrap(),
            LayerOpKind::ExpertDown
        );
        assert_eq!(
            cls("blk.0.ffn_shexp.ffn_gate.weight").unwrap(),
            LayerOpKind::SharedExpert
        );

        assert_eq!(
            cls("model.layers.0.block_sparse_moe.gate.weight").unwrap(),
            LayerOpKind::Router
        );
        assert_eq!(
            cls("model.layers.0.block_sparse_moe.experts.0.w1.weight").unwrap(),
            LayerOpKind::ExpertGate
        );
        assert_eq!(
            cls("model.layers.0.block_sparse_moe.experts.0.w2.weight").unwrap(),
            LayerOpKind::ExpertDown
        );
        assert_eq!(
            cls("model.layers.0.block_sparse_moe.experts.0.w3.weight").unwrap(),
            LayerOpKind::ExpertUp
        );
        // Qwen3 HF naming: router = mlp.gate, experts = gate/up/down_proj.
        assert_eq!(cls("model.layers.0.mlp.gate.weight").unwrap(), LayerOpKind::Router);
        assert_eq!(
            cls("model.layers.0.mlp.experts.0.gate_proj.weight").unwrap(),
            LayerOpKind::ExpertGate
        );
        assert_eq!(
            cls("model.layers.0.mlp.experts.0.up_proj.weight").unwrap(),
            LayerOpKind::ExpertUp
        );
        assert_eq!(
            cls("model.layers.0.mlp.experts.0.down_proj.weight").unwrap(),
            LayerOpKind::ExpertDown
        );
        assert_eq!(
            cls("model.layers.0.mlp.shared_expert.gate_proj.weight").unwrap(),
            LayerOpKind::SharedExpert
        );
    }

    /// Dense `gate_proj` must NOT be mistaken for a MoE router.
    #[test]
    fn dense_mlp_gate_proj_is_not_router() {
        assert_eq!(
            cls("model.layers.0.mlp.gate_proj.weight").unwrap(),
            LayerOpKind::FfnGate
        );
        assert_eq!(cls("blk.0.ffn_gate.weight").unwrap(), LayerOpKind::FfnGate);
    }

    /// FFN disperso (DAG irregular) de `saor`: los dos tensores del bloque se
    /// clasifican (sin prefijo `blk.N.`) y no caen en denso FFN ni en aux.
    #[test]
    fn classifies_saor_sparse_dag_tensors() {
        assert_eq!(
            cls("ffn_dag_adjacency").unwrap(),
            LayerOpKind::FfnDagAdjacency
        );
        assert_eq!(cls("ffn_dag_weights").unwrap(), LayerOpKind::FfnDagWeights);
        // Un GGUF de saor no trae `general.architecture`; la clasificación es por
        // nombre, no por arquitectura.
        assert_eq!(
            cls("ffn_dag_weights").unwrap(),
            LayerOpKind::FfnDagWeights
        );
    }

    /// MLA (DeepSeek-V2/V3, Kimi): latent projections + q/kv LoRA norms.
    #[test]
    fn classifies_mla_ops() {
        for (name, expected) in [
            ("blk.0.attn_q_a.weight", LayerOpKind::MlaQa),
            ("blk.0.attn_q_a_norm.weight", LayerOpKind::MlaQaNorm),
            ("blk.0.attn_q_b.weight", LayerOpKind::MlaQb),
            ("blk.0.attn_kv_a_mqa.weight", LayerOpKind::MlaKvA),
            ("blk.0.attn_kv_a_norm.weight", LayerOpKind::MlaKvANorm),
            ("blk.0.attn_k_b.weight", LayerOpKind::MlaKb),
            ("blk.0.attn_v_b.weight", LayerOpKind::MlaVb),
            ("blk.0.attn_kv_b.weight", LayerOpKind::MlaKvB),
        ] {
            assert_eq!(cls(name).unwrap(), expected, "tensor {name}");
        }
        // MLA norms must not be confused with the gemma per-head q/k norms.
        assert_eq!(
            cls("blk.0.attn_q_norm.weight").unwrap(),
            LayerOpKind::AttnQNorm
        );
    }

    #[test]
    fn unknown_vision_blocked() {
        assert!(cls("mmproj.weight").is_err());
        assert!(cls("blk.0.unknown_op.weight").is_err());
    }

    /// F2.11: safe aliases for existing families (Falcon / GPT-NeoX / Pythia /
    /// MPT / GPT-2 / Qwen3-Next) map to already-implemented ops.
    #[test]
    fn classifies_safe_aliases_for_existing_families() {
        for (name, expected) in [
            ("transformer.h.0.self_attention.query_key_value.weight", LayerOpKind::AttnQkv),
            ("gpt_neox.layers.0.attention.query_key_value.weight", LayerOpKind::AttnQkv),
            ("blk.0.attn.Wqkv.weight", LayerOpKind::AttnQkv),
            ("transformer.h.0.self_attention.dense.weight", LayerOpKind::AttnO),
            ("blk.0.attn.out_proj.weight", LayerOpKind::AttnO),
            ("transformer.h.0.mlp.dense_h_to_4h.weight", LayerOpKind::FfnUp),
            ("transformer.h.0.mlp.dense_4h_to_h.weight", LayerOpKind::FfnDown),
            ("transformer.ln_f.weight", LayerOpKind::OutputNorm),
            ("gpt_neox.final_layer_norm.weight", LayerOpKind::OutputNorm),
            ("transformer.word_embeddings.weight", LayerOpKind::TokenEmbed),
            ("gpt_neox.embed_in.weight", LayerOpKind::TokenEmbed),
            ("embed_out.weight", LayerOpKind::OutputProj),
            ("blk.0.norm_1.weight", LayerOpKind::AttnNorm),
            ("blk.0.norm_2.weight", LayerOpKind::FfnNorm),
            ("blk.0.ln_1.weight", LayerOpKind::AttnNorm),
            ("blk.0.ln_2.weight", LayerOpKind::FfnNorm),
            ("blk.0.in_proj_ba.weight", LayerOpKind::DeltaNet),
            // `linear_attn.out_proj` is DeltaNet, not attention output.
            ("model.layers.0.linear_attn.out_proj.weight", LayerOpKind::DeltaNet),
        ] {
            assert_eq!(cls(name).unwrap(), expected, "tensor {name}");
        }
    }

    /// F2.11: ops that are real but not implemented must fail loudly, never be
    /// silently treated as an existing op or swallowed by `Aux`.
    #[test]
    fn unimplemented_ops_fail_loudly() {
        // Fused gate+up is split by rows at pack load (Phi-3 `gate_up_proj`).
        assert_eq!(
            cls("blk.0.ffn_gate_up.weight").unwrap(),
            LayerOpKind::FfnGateUp
        );
        assert_eq!(
            cls("model.layers.0.mlp.gate_up_proj.weight").unwrap(),
            LayerOpKind::FfnGateUp
        );
        // Fused expert gate+up (split by rows at plan time).
        assert_eq!(
            cls("blk.0.ffn_gate_up_exps.weight").unwrap(),
            LayerOpKind::ExpertGateUp
        );
        // Learned absolute position embeddings (GPT-2 `wpe`) are a cataloged op.
        assert_eq!(
            cls("transformer.wpe.weight").unwrap(),
            LayerOpKind::PositionEmbed
        );
        assert_eq!(
            cls("blk.0.position_embd.weight").unwrap(),
            LayerOpKind::PositionEmbed
        );
        // Input-embedding LayerNorm (BLOOM) is its own op, not `TokenEmbed`.
        assert_eq!(
            cls("token_embd_norm.weight").unwrap(),
            LayerOpKind::TokenEmbedNorm
        );
        assert_eq!(
            cls("token_embd_norm.bias").unwrap(),
            LayerOpKind::TokenEmbedNorm
        );
        // A compute `.proj.weight` is no longer silently `Aux`.
        assert!(cls("blk.0.mystery.proj.weight").is_err());
    }

    #[test]
    fn op_bindings_follow_prd() {
        assert_eq!(op_binding(LayerOpKind::FfnDown), OpBinding::GPU_ASYNC);
        assert_eq!(op_binding(LayerOpKind::ExpertGate), OpBinding::GPU_ASYNC);
        assert_eq!(op_binding(LayerOpKind::TokenEmbed), OpBinding::HOST_ROW);
        assert_eq!(op_binding(LayerOpKind::Router), OpBinding::CPU_GEMV);
        assert_eq!(op_binding(LayerOpKind::AttnQ), OpBinding::CPU_GEMV);
        assert_eq!(op_binding(LayerOpKind::FfnNorm), OpBinding::CPU_NORM);
        // Sparse DAG → GPU async SpMM.
        assert_eq!(op_binding(LayerOpKind::FfnDagAdjacency), OpBinding::GPU_ASYNC);
        assert_eq!(op_binding(LayerOpKind::FfnDagWeights), OpBinding::GPU_ASYNC);
    }

    /// Track M: a synthetic MoE GGUF must split into per-expert units with a
    /// **sparse** streaming window (attn/router + top_k × expert) — not the whole
    /// block, which would defeat disk streaming on edge devices.
    #[test]
    fn moe_plan_splits_experts_with_sparse_window() {
        use hayai_model::{gguf::write_minimal_gguf, GgufCatalog, MetadataValue};
        use std::env::temp_dir;

        let path = temp_dir().join("hayai_plan_moe.gguf");
        let h: u64 = 8; // hidden
        let ff: u64 = 16; // expert intermediate
        let vocab: u64 = 16;
        let ne: u64 = 4; // experts
        let mut tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("token_embd.weight", vec![h, vocab], vec![0.1f32; (h * vocab) as usize]),
            ("output_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("output.weight", vec![vocab, h], vec![0.1f32; (vocab * h) as usize]),
            ("blk.0.attn_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("blk.0.attn_q.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_k.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_v.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_output.weight", vec![2 * h, h], vec![0.1f32; (2 * h * h) as usize]),
            ("blk.0.ffn_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("blk.0.ffn_gate_inp.weight", vec![h, ne], vec![0.1f32; (h * ne) as usize]),
        ];
        let mut names: Vec<String> = Vec::new();
        for e in 0..ne {
            names.push(format!("blk.0.ffn_exp.{e}.ffn_gate.weight"));
            names.push(format!("blk.0.ffn_exp.{e}.ffn_up.weight"));
            names.push(format!("blk.0.ffn_exp.{e}.ffn_down.weight"));
        }
        let mut ti = 0usize;
        for _ in 0..ne {
            tensors.push((names[ti].as_str(), vec![h, ff], vec![0.1f32; (h * ff) as usize]));
            ti += 1;
            tensors.push((names[ti].as_str(), vec![h, ff], vec![0.1f32; (h * ff) as usize]));
            ti += 1;
            tensors.push((names[ti].as_str(), vec![ff, h], vec![0.1f32; (ff * h) as usize]));
            ti += 1;
        }
        write_minimal_gguf(
            &path,
            &[
                (
                    "general.architecture",
                    MetadataValue::String("qwen3moe".into()),
                ),
                ("qwen3moe.expert_count", MetadataValue::U32(4)),
                (
                    "qwen3moe.attention.expert_used_count",
                    MetadataValue::U32(2),
                ),
            ],
            &tensors,
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        let plan = build_exec_plan(&cat, 0, false).unwrap();
        let _ = std::fs::remove_file(&path);

        let moe = plan.moe.expect("moe meta");
        assert_eq!(moe.expert_count, 4);
        assert_eq!(moe.top_k, 2);
        let unit = plan.units.iter().find(|u| u.block_id == Some(0)).unwrap();
        assert_eq!(unit.experts.len(), 4);
        assert_eq!(unit.experts[0].expert_id, 0);
        assert_eq!(unit.experts[0].tensors.len(), 3); // gate + up + down
        assert!(unit.non_expert_bytes > 0);
        assert!(unit.max_expert_bytes > 0);

        // Sparse window: attn/router + top_k × expert, strictly less than the full
        // block (which would include all 4 experts).
        let full_block: usize = unit
            .tensors
            .iter()
            .map(|t| t.nbytes)
            .sum();
        let sparse_window = unit.non_expert_bytes + 2 * unit.max_expert_bytes;
        assert_eq!(unit.total_bytes, sparse_window);
        assert_eq!(plan.max_unit_bytes, sparse_window);
        assert!(sparse_window < full_block);
    }

    /// Track M: fused 3D experts (`ffn_*_exps.weight`, OLMoE-style) must become
    /// per-expert byte-slices (`src_off`) with a sparse streaming window too.
    #[test]
    fn moe_plan_splits_fused_experts() {
        use hayai_model::{gguf::write_minimal_gguf, GgufCatalog, MetadataValue};
        use std::env::temp_dir;

        let path = temp_dir().join("hayai_plan_moe_fused.gguf");
        let h: u64 = 8;
        let ff: u64 = 4;
        let vocab: u64 = 16;
        let ne: u64 = 2;
        let tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("token_embd.weight", vec![h, vocab], vec![0.1f32; (h * vocab) as usize]),
            ("output_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("output.weight", vec![vocab, h], vec![0.1f32; (vocab * h) as usize]),
            ("blk.0.attn_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("blk.0.attn_q.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_k.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_v.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_output.weight", vec![2 * h, h], vec![0.1f32; (2 * h * h) as usize]),
            ("blk.0.ffn_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("blk.0.ffn_gate_inp.weight", vec![h, ne], vec![0.1f32; (h * ne) as usize]),
            // Fused 3D expert tensors: [n_embd, ffn, n_expert].
            (
                "blk.0.ffn_gate_exps.weight",
                vec![h, ff, ne],
                vec![0.1f32; (h * ff * ne) as usize],
            ),
            (
                "blk.0.ffn_up_exps.weight",
                vec![h, ff, ne],
                vec![0.1f32; (h * ff * ne) as usize],
            ),
            (
                "blk.0.ffn_down_exps.weight",
                vec![ff, h, ne],
                vec![0.1f32; (ff * h * ne) as usize],
            ),
        ];
        write_minimal_gguf(
            &path,
            &[
                (
                    "general.architecture",
                    MetadataValue::String("olmoe".into()),
                ),
                ("olmoe.expert_count", MetadataValue::U32(2)),
                (
                    "olmoe.attention.expert_used_count",
                    MetadataValue::U32(1),
                ),
            ],
            &tensors,
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        let plan = build_exec_plan(&cat, 0, false).unwrap();
        let _ = std::fs::remove_file(&path);

        let unit = plan.units.iter().find(|u| u.block_id == Some(0)).unwrap();
        assert_eq!(unit.experts.len(), 2);
        assert!(unit.experts[0].fused);
        // Gate/up/down slices, gate@0, up@slice, down@2*slice.
        assert_eq!(unit.experts[0].tensors.len(), 3);
        assert_eq!(unit.experts[0].tensors[0].op, LayerOpKind::ExpertGate);
        assert_eq!(unit.experts[0].tensors[0].offset, 0);
        let slice = unit.experts[0].tensors[0].nbytes;
        assert_eq!(unit.experts[1].tensors[0].src_off, slice); // expert 1 starts at slice
        assert_eq!(unit.experts[0].tensors[1].offset, slice);
        // Sparse window: attn/router + top_k × (3 slices) < full block.
        let full_block: usize = unit.tensors.iter().map(|t| t.nbytes).sum::<usize>()
            + unit.experts.len() * 3 * slice;
        let sparse_window = unit.non_expert_bytes + unit.max_expert_bytes;
        assert_eq!(unit.total_bytes, sparse_window);
        assert!(sparse_window < full_block);
    }

    /// Fused expert gate+up (`ffn_gate_up_exps`) splits into per-expert gate/up
    /// views (rows concat `[gate | up]`) alongside the fused down experts.
    #[test]
    fn moe_plan_splits_fused_gate_up_experts() {
        use hayai_model::{gguf::write_minimal_gguf, GgufCatalog, MetadataValue};
        use std::env::temp_dir;

        let path = temp_dir().join("hayai_plan_moe_fused_gu.gguf");
        let h: u64 = 8;
        let ff: u64 = 4;
        let vocab: u64 = 16;
        let ne: u64 = 2;
        let tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("token_embd.weight", vec![h, vocab], vec![0.1f32; (h * vocab) as usize]),
            ("output_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("output.weight", vec![vocab, h], vec![0.1f32; (vocab * h) as usize]),
            ("blk.0.attn_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("blk.0.attn_q.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_k.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_v.weight", vec![h, 2 * h], vec![0.1f32; (h * 2 * h) as usize]),
            ("blk.0.attn_output.weight", vec![2 * h, h], vec![0.1f32; (2 * h * h) as usize]),
            ("blk.0.ffn_norm.weight", vec![h], vec![1.0f32; h as usize]),
            ("blk.0.ffn_gate_inp.weight", vec![h, ne], vec![0.1f32; (h * ne) as usize]),
            // Fused gate+up: [n_embd, 2*ffn, n_expert].
            (
                "blk.0.ffn_gate_up_exps.weight",
                vec![h, 2 * ff, ne],
                vec![0.1f32; (h * 2 * ff * ne) as usize],
            ),
            (
                "blk.0.ffn_down_exps.weight",
                vec![ff, h, ne],
                vec![0.1f32; (ff * h * ne) as usize],
            ),
        ];
        write_minimal_gguf(
            &path,
            &[
                ("general.architecture", MetadataValue::String("olmoe".into())),
                ("olmoe.expert_count", MetadataValue::U32(2)),
                ("olmoe.attention.expert_used_count", MetadataValue::U32(1)),
            ],
            &tensors,
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        let plan = build_exec_plan(&cat, 0, false).unwrap();
        let _ = std::fs::remove_file(&path);

        let unit = plan.units.iter().find(|u| u.block_id == Some(0)).unwrap();
        assert_eq!(unit.experts.len(), 2);
        // gate, up (from the fused tensor) + down.
        assert_eq!(unit.experts[0].tensors.len(), 3);
        let gate = &unit.experts[0].tensors[0];
        let up = &unit.experts[0].tensors[1];
        let down = &unit.experts[0].tensors[2];
        assert_eq!(gate.op, LayerOpKind::ExpertGate);
        assert_eq!(up.op, LayerOpKind::ExpertUp);
        assert_eq!(down.op, LayerOpKind::ExpertDown);
        assert_eq!(gate.nrows, ff as usize);
        assert_eq!(up.nrows, ff as usize);
        assert_eq!(gate.ncols, h as usize);
        assert_eq!(gate.src_off, 0);
        assert_eq!(up.src_off, gate.nbytes); // up starts after gate rows
        assert_eq!(down.src_off, 0);
    }

    /// Un GGUF disperso de `saor` (2 tensores sin prefijo `blk.N.`) abre y se
    /// planifica sin `UnknownLayerOp`, agrupándose en un único `StreamingUnit`
    /// con `block_id = None` y binding GPU-async para el SpMM.
    #[test]
    fn sparse_dag_gguf_plans_as_unit() {
        use hayai_model::{gguf::write_minimal_gguf, GgufCatalog, MetadataValue};

        let path = std::env::temp_dir().join("hayai_plan_sparse_dag.gguf");
        let tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("ffn_dag_adjacency", vec![4], vec![0.0f32; 4]),
            ("ffn_dag_weights", vec![6], vec![0.1f32; 6]),
        ];
        write_minimal_gguf(
            &path,
            &[
                ("saor.d_in", MetadataValue::U32(8)),
                ("saor.d_out", MetadataValue::U32(4)),
            ],
            &tensors,
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        let plan = build_exec_plan(&cat, 0, false).unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(plan.known_ops.contains(&LayerOpKind::FfnDagAdjacency));
        assert!(plan.known_ops.contains(&LayerOpKind::FfnDagWeights));
        let unit = plan
            .units
            .iter()
            .find(|u| u.block_id.is_none())
            .expect("unidad sin blk.N.");
        assert_eq!(unit.tensors.len(), 2);
        assert_eq!(
            plan.op_bindings[&LayerOpKind::FfnDagWeights].device,
            OpDevice::GpuAsync
        );
    }
}
