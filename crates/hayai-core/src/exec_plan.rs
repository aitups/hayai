//! Metadata-driven execution plan: classify GGUF tensors into known layer ops.
//!
//! Hayai is architecture-agnostic: any GGUF whose tensors map to registered ops
//! gets a streaming plan. Failure is only `UnknownLayerOp` for truly novel layers.

use hayai_model::GgufCatalog;
use std::collections::BTreeSet;

/// Known computational layer / tensor roles (extensible registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LayerOpKind {
    TokenEmbed,
    OutputNorm,
    OutputProj,
    AttnNorm,
    AttnQ,
    AttnK,
    AttnV,
    AttnO,
    AttnGate,
    FfnNorm,
    FfnGate,
    FfnUp,
    FfnDown,
    /// Gated DeltaNet / SSM / linear-attention family (Qwen3.5 hybrid, etc.).
    DeltaNet,
    /// Multi-token prediction / next-n draft head (Qwen3.5 `blk.*.nextn.*`).
    NextN,
    /// Fused QKV projection (`attn_qkv`).
    AttnQkv,
    /// MoE router or expert weights.
    MoE,
    /// HRM / recurrent outer-loop state tensors.
    Recurrence,
    /// Conv / positional extras still recognized.
    Conv,
    /// Bias / scale / misc known aux.
    Aux,
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
        match classify_tensor(&t.name) {
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

fn classify_tensor(name: &str) -> Result<LayerOpKind, String> {
    let n = name.to_ascii_lowercase();

    // Global / non-block
    if n.contains("token_embd")
        || n.ends_with("embed_tokens.weight")
        || n.contains("tok_embeddings")
        || n.contains("per_layer_token_embd")
    {
        return Ok(LayerOpKind::TokenEmbed);
    }
    if n.contains("output_norm") || n.ends_with("model.norm.weight") || n == "norm.weight" {
        return Ok(LayerOpKind::OutputNorm);
    }
    if n == "output.weight" || n.ends_with("lm_head.weight") {
        return Ok(LayerOpKind::OutputProj);
    }
    // Gemma4 per-layer embedding projection
    if n.contains("per_layer_model_proj") || n.contains("per_layer_proj_norm") {
        return Ok(LayerOpKind::Aux);
    }

    // Qwen3.5 MTP / next-n head (before attn_* substring matches).
    if n.contains(".nextn.") || n.contains("nextn_") || n.contains("mtp_") {
        return Ok(LayerOpKind::NextN);
    }

    // Post-attn / FFN norms before `attn_norm` substring (avoids `post_attention_norm` → AttnNorm).
    if n.contains("ffn_norm")
        || n.contains("post_attention_layernorm")
        || n.contains("post_attention_norm")
        || n.contains("post_ffw_norm")
    {
        return Ok(LayerOpKind::FfnNorm);
    }

    // Attention family
    if n.contains("attn_norm") || n.contains("input_layernorm") || n.contains("attention_norm") {
        return Ok(LayerOpKind::AttnNorm);
    }
    if n.contains("attn_qkv") || n.contains("qkv_proj") {
        return Ok(LayerOpKind::AttnQkv);
    }
    if n.contains("attn_q") || n.contains("q_proj") || n.contains(".wq.") {
        return Ok(LayerOpKind::AttnQ);
    }
    if n.contains("attn_k") || n.contains("k_proj") || n.contains(".wk.") {
        return Ok(LayerOpKind::AttnK);
    }
    if n.contains("attn_v") || n.contains("v_proj") || n.contains(".wv.") {
        return Ok(LayerOpKind::AttnV);
    }
    if n.contains("attn_output") || n.contains("o_proj") || n.contains("attn_out") || n.contains(".wo.")
    {
        return Ok(LayerOpKind::AttnO);
    }
    if n.contains("attn_gate") || n.contains("attn.gate") || n.contains("inp_gate") {
        return Ok(LayerOpKind::AttnGate);
    }

    // FFN / MLP
    if n.contains("ffn_gate") || n.contains("gate_proj") {
        return Ok(LayerOpKind::FfnGate);
    }
    if n.contains("ffn_up") || n.contains("up_proj") {
        return Ok(LayerOpKind::FfnUp);
    }
    if n.contains("ffn_down") || n.contains("down_proj") {
        return Ok(LayerOpKind::FfnDown);
    }

    // Hybrid / SSM / DeltaNet (known op class)
    if n.contains("delta")
        || n.contains("linear_attn")
        || n.contains("mamba")
        || n.contains("ssm")
        || n.contains("conv1d")
        || n.contains("in_proj_qkv")
        || (n.contains("out_proj") && n.contains("shortconv"))
    {
        return Ok(LayerOpKind::DeltaNet);
    }
    if n.contains("conv") {
        return Ok(LayerOpKind::Conv);
    }

    // MoE
    if n.contains("expert") || n.contains("router") || n.contains("gate.weight") && n.contains("moe")
    {
        return Ok(LayerOpKind::MoE);
    }

    // Recurrence / HRM
    if n.contains("h_cycle")
        || n.contains("l_cycle")
        || n.contains("recurrent")
        || n.starts_with("h.")
        || n.starts_with("l.")
        || n.contains("z_h")
        || n.contains("z_l")
    {
        return Ok(LayerOpKind::Recurrence);
    }

    // Rope freqs, biases, scales, shexp, ALiBi — aux
    if n.contains("bias")
        || n.contains("scale")
        || n.contains("rope")
        || n.contains("cos")
        || n.contains("sin")
        || n.contains("inv_freq")
        || n.contains("alibi")
        || n.contains("shear")
        || n.contains("attn_sink")
        || n.contains("gate_inp")
        || n.contains("ffn_gate_inp")
        || n.contains("ffn_exp")
        || n.contains("exps")
        || n.contains("layer_output_scale")
        || n.contains("post_norm")
        || n.ends_with(".proj.weight")
        || n.contains(".proj.bias")
    {
        return Ok(LayerOpKind::Aux);
    }
    // Gemma / Qwen naming variants still in known Attn/FFN families
    if n.contains("q_norm") || n.contains("k_norm") {
        return Ok(LayerOpKind::Aux);
    }
    if n.contains("shared_expert") || n.contains("ffn_shexp") {
        return Ok(LayerOpKind::MoE);
    }

    // mmproj / vision — treat as unknown for text streaming v1 if clearly vision
    if n.contains("mmproj") || n.contains("vision") || n.contains("clip") {
        return Err(format!(
            "vision/multimodal tensor not in text streaming registry yet: {name}"
        ));
    }

    Err(format!(
        "unrecognized tensor role (register a LayerOpKind or fix naming): {name}"
    ))
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

    #[test]
    fn classifies_llama_tensors() {
        assert_eq!(
            classify_tensor("blk.0.attn_q.weight").unwrap(),
            LayerOpKind::AttnQ
        );
        assert_eq!(
            classify_tensor("blk.3.ffn_down.weight").unwrap(),
            LayerOpKind::FfnDown
        );
        assert_eq!(
            classify_tensor("token_embd.weight").unwrap(),
            LayerOpKind::TokenEmbed
        );
    }

    #[test]
    fn unknown_vision_blocked() {
        assert!(classify_tensor("mmproj.weight").is_err());
    }

    #[test]
    fn classifies_qwen35_hybrid() {
        assert_eq!(
            classify_tensor("blk.0.ssm_out.weight").unwrap(),
            LayerOpKind::DeltaNet
        );
        assert_eq!(
            classify_tensor("blk.0.attn_qkv.weight").unwrap(),
            LayerOpKind::AttnQkv
        );
        assert_eq!(
            classify_tensor("blk.0.post_attention_norm.weight").unwrap(),
            LayerOpKind::FfnNorm
        );
        assert_eq!(
            classify_tensor("blk.32.nextn.eh_proj.weight").unwrap(),
            LayerOpKind::NextN
        );
    }
}
