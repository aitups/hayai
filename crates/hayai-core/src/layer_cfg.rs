//! Per-layer attention / hybrid config resolved from tensor shapes (+ arch meta cross-check).
//!
//! Do **not** bake model-size constants (4B head counts, SSM ranks, etc.). A Qwen3.5-9B/27B
//! GGUF must work from the same path as 4B.

use crate::deltanet::{is_deltanet_layer, is_nextn_layer};
use hayai_cpu::{AttentionConfig, LayerKvCache};
use hayai_model::{GgufCatalog, GgufError, ModelConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerKind {
    FullAttn,
    DeltaNet,
    NextN,
}

/// Resolve kind from resident tensors (preferred over `full_attention_interval` heuristics).
pub fn hybrid_layer_kind(cat: &GgufCatalog, layer: usize) -> HybridLayerKind {
    if is_nextn_layer(cat, layer) {
        HybridLayerKind::NextN
    } else if is_deltanet_layer(cat, layer) {
        HybridLayerKind::DeltaNet
    } else {
        HybridLayerKind::FullAttn
    }
}

/// Full-attention dims for one physical block, derived from that block's tensors.
pub fn resolve_full_attn_cfg(
    cat: &GgufCatalog,
    layer: usize,
    config: &ModelConfig,
) -> Result<AttentionConfig, GgufError> {
    let q_norm = cat
        .tensor(&format!("blk.{layer}.attn_q_norm.weight"))
        .map(|t| t.ncols().max(t.nrows()))
        .ok();
    let k_norm = cat
        .tensor(&format!("blk.{layer}.attn_k_norm.weight"))
        .map(|t| t.ncols().max(t.nrows()))
        .ok();
    let head_dim = q_norm
        .or(k_norm)
        .or_else(|| {
            cat.meta_u32(&format!("{}.attention.key_length", config.architecture))
                .map(|v| v as usize)
        })
        .filter(|&d| d > 0)
        .ok_or_else(|| {
            GgufError::Msg(format!(
                "blk.{layer}: cannot resolve head_dim (missing attn_q_norm / key_length)"
            ))
        })?;

    let wq = cat.tensor(&format!("blk.{layer}.attn_q.weight"))?;
    let wk = cat.tensor(&format!("blk.{layer}.attn_k.weight"))?;
    let wo = cat.tensor(&format!("blk.{layer}.attn_output.weight"))?;

    // GGUF: [ncols=ne0=in, nrows=ne1=out]
    let q_out = wq.nrows();
    let k_out = wk.nrows();
    let o_in = wo.ncols();

    // Fused Q∥gate: q_out == 2 * n_heads * head_dim (typically == 2 * o_in).
    let n_heads = if q_out == o_in * 2 && o_in % head_dim == 0 {
        o_in / head_dim
    } else if q_out % (2 * head_dim) == 0 && q_out / (2 * head_dim) > 0 {
        q_out / (2 * head_dim)
    } else if q_out % head_dim == 0 {
        q_out / head_dim
    } else {
        return Err(GgufError::Msg(format!(
            "blk.{layer}: attn_q out={q_out} not divisible by head_dim={head_dim}"
        )));
    };
    if k_out % head_dim != 0 {
        return Err(GgufError::Msg(format!(
            "blk.{layer}: attn_k out={k_out} not divisible by head_dim={head_dim}"
        )));
    }
    let n_kv = k_out / head_dim;
    if n_heads == 0 || n_kv == 0 || n_heads % n_kv != 0 {
        return Err(GgufError::Msg(format!(
            "blk.{layer}: invalid GQA n_heads={n_heads} n_kv={n_kv}"
        )));
    }

    let mut cfg = AttentionConfig {
        num_heads: n_heads,
        num_kv_heads: n_kv,
        head_dim,
        rope_theta: config.rope_theta,
        rope_dim: head_dim,
        scale_override: None,
    };
    if let Some(rd) = cat
        .meta_u32(&format!("{}.rope.dimension_count", config.architecture))
        .or_else(|| cat.meta_u32("llama.rope.dimension_count"))
    {
        let rd = rd as usize;
        if rd > 0 && rd <= head_dim {
            cfg.rope_dim = rd;
        }
    }
    Ok(cfg)
}

/// KV caches sized per full-attn layer; DeltaNet/NextN get unused 1×1 placeholders.
pub fn build_hybrid_kv_caches(
    cat: &GgufCatalog,
    config: &ModelConfig,
    sink: usize,
    window: usize,
) -> Result<Vec<LayerKvCache>, GgufError> {
    let mut out = Vec::with_capacity(config.num_layers);
    for i in 0..config.num_layers {
        match hybrid_layer_kind(cat, i) {
            HybridLayerKind::FullAttn => {
                let cfg = resolve_full_attn_cfg(cat, i, config)?;
                out.push(LayerKvCache::new(cfg.num_kv_heads, cfg.head_dim, sink, window));
            }
            HybridLayerKind::DeltaNet | HybridLayerKind::NextN => {
                out.push(LayerKvCache::new(1, 1, sink, window));
            }
        }
    }
    Ok(out)
}
