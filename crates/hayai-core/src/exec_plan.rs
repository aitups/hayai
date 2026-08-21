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
    /// `post_ffw_norm.weight` / `post_mlp_norm.weight` — norm after FFN (Gemma4 12B).
    PostFfnNorm,
    /// `layer_output_scale.weight` — per-block output scale (Gemma4 12B).
    LayerOutputScale,
    /// MoE router — `ffn_gate_inp.weight` / HF `block_sparse_moe.gate.weight`.
    Router,
    /// Per-expert FFN matrices — `ffn_exp.E.ffn_gate/up/down` / HF `experts.E.w1/w2/w3`.
    ExpertGate,
    ExpertUp,
    ExpertDown,
    /// Shared expert — `ffn_shexp.*` / `shared_expert.*` (DeepSeek-style).
    SharedExpert,
    /// Gated DeltaNet / SSM / linear-attention family (Qwen3.5 hybrid, etc.).
    DeltaNet,
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
        OutputNorm | AttnNorm | AttnQNorm | AttnKNorm | PostAttnNorm | FfnNorm | PostFfnNorm
        | PleProjNorm | PlePostNorm => OpBinding::CPU_NORM,
        AttnQ | AttnK | AttnV | AttnO | AttnGate | AttnQkv | PleGate | PleProj | Router
        | DeltaNet | NextN | Recurrence | Conv => OpBinding::CPU_GEMV,
        FfnGate | FfnUp | FfnDown | ExpertGate | ExpertUp | ExpertDown | SharedExpert
        | OutputProj | PleModelProj => OpBinding::GPU_ASYNC,
        LayerOutputScale | Aux => OpBinding::DISCARD,
    }
}

/// A tensor that could not be classified — only valid arch-block reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLayerOp {
    pub tensor_name: String,
    pub hint: String,
}

#[derive(Debug, Clone)]
pub struct ClassifiedTensor {
    pub name: String,
    pub op: LayerOpKind,
    pub nbytes: usize,
}

#[derive(Debug, Clone)]
pub struct StreamingUnit {
    /// Logical block id (e.g. layer index) when detectable.
    pub block_id: Option<usize>,
    pub tensors: Vec<ClassifiedTensor>,
    pub total_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct ExecPlan {
    pub architecture: String,
    pub units: Vec<StreamingUnit>,
    pub max_unit_bytes: usize,
    pub known_ops: BTreeSet<LayerOpKind>,
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
                classified.push(ClassifiedTensor {
                    name: t.name.clone(),
                    op,
                    nbytes,
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
    let mut units: Vec<StreamingUnit> = Vec::new();
    let mut by_block: std::collections::BTreeMap<Option<usize>, Vec<ClassifiedTensor>> =
        std::collections::BTreeMap::new();
    for c in classified {
        let bid = parse_block_id(&c.name);
        by_block.entry(bid).or_default().push(c);
    }
    for (block_id, tensors) in by_block {
        let total_bytes = tensors.iter().map(|t| t.nbytes).sum();
        units.push(StreamingUnit {
            block_id,
            tensors,
            total_bytes,
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
        known_ops,
        op_bindings,
        hw_notes,
    })
}

fn parse_block_id(name: &str) -> Option<usize> {
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

    // ── Phase 2: global embeddings / output head. ────────────────────────────
    if n.contains("token_embd") || n.ends_with("embed_tokens.weight") || n.contains("tok_embeddings")
    {
        return Ok(TokenEmbed);
    }
    if n.contains("output_norm") || n.ends_with("model.norm.weight") || n == "norm.weight" {
        return Ok(OutputNorm);
    }
    if n == "output.weight" || n.ends_with("lm_head.weight") {
        return Ok(OutputProj);
    }

    // ── Phase 3: MoE router — before dense FFN (`ffn_gate_inp` ⊃ `ffn_gate`). ─
    if n.contains("router")
        || n.contains("gate_inp")
        || n.contains("block_sparse_moe.gate")
        || n.contains("moe.gate")
        || n.contains("mlp.gate.weight")
    {
        return Ok(Router);
    }

    // ── Phase 4: MoE experts / shared expert — before dense FFN. ─────────────
    if n.contains("shared_expert") || n.contains("ffn_shexp") {
        return Ok(SharedExpert);
    }
    if n.contains("ffn_exp") || n.contains("experts.") {
        return Ok(expert_kind(&n));
    }

    // ── Phase 5: norms — before projections (`attn_q_norm` ⊃ `attn_q`). ───────
    if n.contains("post_attention_norm")
        || n.contains("post_attention_layernorm")
        || n.contains("ffn_norm")
    {
        return Ok(FfnNorm);
    }
    if n.contains("post_ffw_norm") || n.contains("post_mlp_norm") {
        return Ok(PostFfnNorm);
    }
    if n.contains("post_norm") {
        return Ok(if has_ple { PlePostNorm } else { PostFfnNorm });
    }
    if n.contains("attn_norm") || n.contains("input_layernorm") || n.contains("attention_norm") {
        return Ok(AttnNorm);
    }
    if n.contains("q_norm") {
        return Ok(AttnQNorm);
    }
    if n.contains("k_norm") {
        return Ok(AttnKNorm);
    }

    // ── Phase 6: dense FFN. ──────────────────────────────────────────────────
    if n.contains("ffn_gate") || n.contains("gate_proj") {
        return Ok(FfnGate);
    }
    if n.contains("ffn_up") || n.contains("up_proj") {
        return Ok(FfnUp);
    }
    if n.contains("ffn_down") || n.contains("down_proj") {
        return Ok(FfnDown);
    }

    // ── Phase 7: attention projections. ──────────────────────────────────────
    if n.contains("attn_qkv") || n.contains("qkv_proj") {
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
    if n.contains("attn_output") || n.contains("o_proj") || n.contains("attn_out") || n.contains(".wo.")
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
    if n.contains("delta")
        || n.contains("linear_attn")
        || n.contains("mamba")
        || n.contains("ssm")
        || n.contains("conv1d")
        || n.contains("in_proj_qkv")
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
    if n.contains("bias")
        || n.contains("scale")
        || n.contains("rope")
        || n.contains("cos")
        || n.contains("sin")
        || n.contains("inv_freq")
        || n.contains("alibi")
        || n.contains("shear")
        || n.contains("attn_sink")
        || n.ends_with(".proj.weight")
        || n.contains(".proj.bias")
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

    #[test]
    fn unknown_vision_blocked() {
        assert!(cls("mmproj.weight").is_err());
        assert!(cls("blk.0.unknown_op.weight").is_err());
    }

    #[test]
    fn op_bindings_follow_prd() {
        assert_eq!(op_binding(LayerOpKind::FfnDown), OpBinding::GPU_ASYNC);
        assert_eq!(op_binding(LayerOpKind::ExpertGate), OpBinding::GPU_ASYNC);
        assert_eq!(op_binding(LayerOpKind::TokenEmbed), OpBinding::HOST_ROW);
        assert_eq!(op_binding(LayerOpKind::Router), OpBinding::CPU_GEMV);
        assert_eq!(op_binding(LayerOpKind::AttnQ), OpBinding::CPU_GEMV);
        assert_eq!(op_binding(LayerOpKind::FfnNorm), OpBinding::CPU_NORM);
    }
}
