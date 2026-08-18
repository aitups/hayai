//! Gemma4 text-only forward: SWA/global heads, shared KV, per-layer embeddings, GELU FFN.

use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{
    ffn_begin_gate_up_scratch, ffn_finish_gelu_scratch, StreamInferError, StreamingGenerator,
};
use hayai_cpu::{attention_decode_step_ex, rms_norm, AttentionConfig, LayerKvCache};
use hayai_model::GgufCatalog;
use hayai_opencl::StreamingScratch;
use std::time::Instant;

pub(crate) fn is_gemma4(gen: &StreamingGenerator) -> bool {
    gen.config.architecture.contains("gemma")
}

/// Build per-layer KV caches with SWA vs global head dims / windows.
pub(crate) fn build_layer_kv_caches(
    cat: &GgufCatalog,
    config: &hayai_model::ModelConfig,
    sink: usize,
    window: usize,
) -> Result<Vec<LayerKvCache>, StreamInferError> {
    let meta = GemmaMeta::from_catalog(cat, config)?;
    Ok((0..config.num_layers)
        .map(|i| {
            let cfg = meta.layer_attn(i);
            let win = if meta.is_swa(i) {
                meta.swa_window.min(window).max(1)
            } else {
                window.max(1)
            };
            LayerKvCache::new(cfg.num_kv_heads, cfg.head_dim, sink, win)
        })
        .collect())
}

struct GemmaMeta {
    n_heads: usize,
    n_kv: usize,
    head_full: usize,
    head_swa: usize,
    rope_full: f32,
    rope_swa: f32,
    swa_window: usize,
    n_kv_from_start: usize,
    per_layer_dim: usize,
    softcap: f32,
    is_swa: Vec<bool>,
}

impl GemmaMeta {
    fn from_catalog(
        cat: &GgufCatalog,
        config: &hayai_model::ModelConfig,
    ) -> Result<Self, StreamInferError> {
        let arch = &config.architecture;
        let n_heads = config.num_attention_heads;
        let n_kv = config.num_key_value_heads;
        let head_full = cat
            .meta_u32(&format!("{arch}.attention.key_length"))
            .unwrap_or(512) as usize;
        let head_swa = cat
            .meta_u32(&format!("{arch}.attention.key_length_swa"))
            .unwrap_or(256) as usize;
        let rope_full = cat
            .meta_f32(&format!("{arch}.rope.freq_base"))
            .unwrap_or(1_000_000.0);
        let rope_swa = cat
            .meta_f32(&format!("{arch}.rope.freq_base_swa"))
            .unwrap_or(10_000.0);
        let swa_window = cat
            .meta_u32(&format!("{arch}.attention.sliding_window"))
            .unwrap_or(512) as usize;
        let shared = cat
            .meta_u32(&format!("{arch}.attention.shared_kv_layers"))
            .unwrap_or(0) as usize;
        let n_kv_from_start = if shared > 0 && shared < config.num_layers {
            config.num_layers - shared
        } else {
            config.num_layers
        };
        let per_layer_dim = cat
            .meta_u32(&format!("{arch}.embedding_length_per_layer_input"))
            .unwrap_or(0) as usize;
        let softcap = cat
            .meta_f32(&format!("{arch}.final_logit_softcapping"))
            .unwrap_or(0.0);
        let is_swa = cat
            .meta_bool_array(&format!("{arch}.attention.sliding_window_pattern"))
            .unwrap_or_else(|| {
                // Default: SWA except every 6th layer (False = global).
                (0..config.num_layers)
                    .map(|i| (i + 1) % 6 != 0)
                    .collect()
            });
        Ok(Self {
            n_heads,
            n_kv,
            head_full,
            head_swa,
            rope_full,
            rope_swa,
            swa_window,
            n_kv_from_start,
            per_layer_dim,
            softcap,
            is_swa,
        })
    }

    fn is_swa(&self, layer: usize) -> bool {
        self.is_swa.get(layer).copied().unwrap_or(true)
    }

    fn has_kv(&self, layer: usize) -> bool {
        layer < self.n_kv_from_start
    }

    fn layer_attn(&self, layer: usize) -> AttentionConfig {
        let swa = self.is_swa(layer);
        let head_dim = if swa { self.head_swa } else { self.head_full };
        let rope = if swa { self.rope_swa } else { self.rope_full };
        AttentionConfig {
            num_heads: self.n_heads,
            num_kv_heads: self.n_kv,
            head_dim,
            rope_theta: rope,
            rope_dim: head_dim,
            scale_override: Some(1.0), // Gemma4: no 1/sqrt(d)
        }
    }

    /// Shared-KV layers reuse the latest prior layer with the same SWA/global type.
    fn kv_source(&self, layer: usize) -> usize {
        if self.has_kv(layer) {
            return layer;
        }
        let want = self.is_swa(layer);
        for j in (0..self.n_kv_from_start.min(layer)).rev() {
            if self.is_swa(j) == want {
                return j;
            }
        }
        layer.saturating_sub(1).min(self.n_kv_from_start.saturating_sub(1))
    }
}

pub(crate) fn prefill_gemma(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    tokens: &[u32],
    scratch: &mut StreamingScratch,
) -> Result<Vec<f32>, StreamInferError> {
    let mut logits = Vec::new();
    for &tok in tokens {
        logits = forward_gemma(gen, orch, tok, scratch)?;
    }
    Ok(logits)
}

pub(crate) fn forward_gemma(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    token: u32,
    scratch: &mut StreamingScratch,
) -> Result<Vec<f32>, StreamInferError> {
    let h = gen.config.hidden_size;
    let eps = gen.config.rms_norm_eps;
    let n_layers = gen.config.num_layers;
    let pos = gen.position;
    let meta = GemmaMeta::from_catalog(&gen.catalog, &gen.config)?;

    let mut x = vec![0.0f32; h];
    gen.embed_row("token_embd.weight", token, h, &mut x)?;
    // Gemma: scale token embeddings by sqrt(n_embd).
    let emb_scale = (h as f32).sqrt();
    for v in x.iter_mut() {
        *v *= emb_scale;
    }

    // Global-attn proportional RoPE factors (GGUF `rope_freqs.weight`); SWA uses None.
    let rope_freqs = gen.catalog.dequant_f32("rope_freqs.weight").ok();

    // Per-layer inputs: tok_ple + project(h) then scale.
    let ple = if meta.per_layer_dim > 0 {
        Some(build_per_layer_inputs(gen, orch, token, &x, &meta)?)
    } else {
        None
    };

    for layer in 0..n_layers {
        let slot = layer % 2;
        if gen
            .catalog
            .tensor(&format!("blk.{layer}.attn_q.weight"))
            .is_err()
        {
            continue;
        }
        let (pack, layout) = gen.stage_pack(orch, scratch, slot, layer)?;
        gen.begin_ffn_dma(orch, scratch, slot, &layout)?;

        let t_attn = Instant::now();
        let mut xn = x.clone();
        rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);

        let layer_cfg = meta.layer_attn(layer);
        let q_dim = layer_cfg.hidden_size();
        let kv_dim = layer_cfg.kv_dim();
        let mut q = vec![0.0f32; pack.wq.nrows.max(q_dim)];
        q.truncate(pack.wq.nrows);
        pack.wq.gemv(&xn, &mut q)?;
        q.resize(q_dim, 0.0);
        if let Ok(qn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_q_norm.weight")) {
            apply_head_rmsnorm(&mut q, &qn, layer_cfg.num_heads, layer_cfg.head_dim);
        }

        let write_kv = meta.has_kv(layer);
        let kv_i = meta.kv_source(layer);
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        if write_kv {
            let mut k_raw = vec![0.0f32; pack.wk.nrows];
            pack.wk.gemv(&xn, &mut k_raw)?;
            k_raw.resize(kv_dim, 0.0);
            let k_pre_norm = k_raw.clone();
            if let Ok(kn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_k_norm.weight")) {
                apply_head_rmsnorm(&mut k_raw, &kn, layer_cfg.num_kv_heads, layer_cfg.head_dim);
            }
            k = k_raw;
            if pack.wv.nrows > 0 {
                let mut v_raw = vec![0.0f32; pack.wv.nrows];
                pack.wv.gemv(&xn, &mut v_raw)?;
                v_raw.resize(kv_dim, 0.0);
                // Gemma4: per-head RMSNorm V without learned weight (llama ggml_rms_norm
                // after reshape to [head_dim, n_kv, tokens]).
                rms_norm_plain_heads(
                    &mut v_raw,
                    layer_cfg.num_kv_heads,
                    layer_cfg.head_dim,
                    eps,
                );
                v = v_raw;
            } else {
                // k_eq_v: V = K before k_norm, then plain per-head RMSNorm.
                let mut v_raw = k_pre_norm;
                rms_norm_plain_heads(
                    &mut v_raw,
                    layer_cfg.num_kv_heads,
                    layer_cfg.head_dim,
                    eps,
                );
                v = v_raw;
            }
        }

        let factors = if meta.is_swa(layer) {
            None
        } else {
            rope_freqs.as_deref()
        };
        let mut attn_out = vec![0.0f32; q_dim];
        attention_decode_step_ex(
            &layer_cfg,
            &mut gen.kv[kv_i],
            &mut q,
            &mut k,
            &v,
            pos,
            &mut attn_out,
            write_kv,
            factors,
        );

        let mut attn_proj = vec![0.0f32; h];
        if pack.wo.ncols == attn_out.len() {
            pack.wo.gemv(&attn_out, &mut attn_proj)?;
        } else {
            return Err(StreamInferError::Msg(format!(
                "gemma wo ncols {} != attn {}",
                pack.wo.ncols,
                attn_out.len()
            )));
        }
        // post_attention_norm BEFORE residual (llama.cpp gemma4).
        if let Ok(pn) = gen
            .catalog
            .dequant_f32(&format!("blk.{layer}.post_attention_norm.weight"))
        {
            rms_norm(&mut attn_proj, &pn, eps);
        }
        for i in 0..h {
            x[i] += attn_proj[i];
        }
        gen.attn_secs += t_attn.elapsed().as_secs_f64();

        // FFN: norm → GELU-gated → post_ffw_norm → residual
        let attn_residual = x.clone();
        let mut xn = x.clone();
        rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
        gen.ws_gate.fill(0.0);
        gen.ws_up.fill(0.0);
        gen.ws_down.fill(0.0);
        gen.finish_ffn_unmap(orch, scratch, slot)?;
        let t_ffn = Instant::now();
        let inflight = ffn_begin_gate_up_scratch(
            orch,
            &pack.gate,
            &pack.up,
            &xn,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.used_dgpu,
            &mut gen.used_apu,
            Some(scratch),
            layer,
            Some(&layout),
        )?;
        ffn_finish_gelu_scratch(
            orch,
            inflight,
            &pack.down,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.ws_down,
            &mut gen.used_dgpu,
            Some(scratch),
            layer,
            Some(&layout),
        )?;
        gen.ffn_secs += t_ffn.elapsed().as_secs_f64();
        let mut down = gen.ws_down.clone();
        if let Ok(pn) = gen
            .catalog
            .dequant_f32(&format!("blk.{layer}.post_ffw_norm.weight"))
        {
            rms_norm(&mut down, &pn, eps);
        }
        for i in 0..h {
            x[i] = attn_residual[i] + down[i];
        }

        // Per-layer embedding residual (after FFN).
        if let Some(ref ple_all) = ple {
            apply_per_layer_emb(gen, orch, layer, &meta, ple_all, &mut x)?;
        }

        if let Ok(scale) = gen
            .catalog
            .dequant_f32(&format!("blk.{layer}.layer_output_scale.weight"))
        {
            let s = scale.first().copied().unwrap_or(1.0);
            for v in x.iter_mut() {
                *v *= s;
            }
        }
        let _ = pack;
    }

    let mut xn = x;
    rms_norm(&mut xn, &gen.output_norm, eps);
    let vocab = gen.config.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    if gen.has_output_weight {
        if let Some(ow) = &gen.resident_output {
            orch.execute_quant_gemv(ow, &xn, &mut logits)?;
        } else {
            let ow = gen.catalog.load_quant_matrix("output.weight")?;
            orch.execute_quant_gemv(&ow, &xn, &mut logits)?;
            gen.io_bytes += ow.nbytes() as u64;
        }
    } else {
        if let Some(emb) = &gen.resident_embed {
            orch.execute_quant_gemv(emb, &xn, &mut logits)?;
        } else {
            let emb = gen.catalog.load_quant_matrix("token_embd.weight")?;
            orch.execute_quant_gemv(&emb, &xn, &mut logits)?;
            gen.io_bytes += emb.nbytes() as u64;
        }
    }
    if meta.softcap > 0.0 {
        let inv = 1.0 / meta.softcap;
        for l in logits.iter_mut() {
            *l = meta.softcap * (*l * inv).tanh();
        }
    }
    gen.position += 1;
    Ok(logits)
}

/// `[n_layers * per_layer_dim]` projected per-layer inputs for the current token.
fn build_per_layer_inputs(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    token: u32,
    h_emb: &[f32],
    meta: &GemmaMeta,
) -> Result<Vec<f32>, StreamInferError> {
    let n_layers = gen.config.num_layers;
    let d = meta.per_layer_dim;
    let total = n_layers * d;
    let mut tok_ple = vec![0.0f32; total];
    // per_layer_token_embd: [vocab, n_layers * d]
    gen.embed_row("per_layer_token_embd.weight", token, total, &mut tok_ple)?;
    let scale = (d as f32).sqrt();
    for v in tok_ple.iter_mut() {
        *v *= scale;
    }

    let proj = gen
        .catalog
        .load_quant_matrix("per_layer_model_proj.weight")?;
    // proj: [n_embd → n_layers * d]
    let mut h_proj = vec![0.0f32; total];
    orch.execute_quant_gemv(&proj, h_emb, &mut h_proj)?;
    let proj_scale = 1.0 / (gen.config.hidden_size as f32).sqrt();
    for v in h_proj.iter_mut() {
        *v *= proj_scale;
    }
    if let Ok(pn) = gen.catalog.dequant_f32("per_layer_proj_norm.weight") {
        // Norm is over the per_layer_dim axis for each layer slice.
        for layer in 0..n_layers {
            let base = layer * d;
            let mut ms = 0.0f32;
            for i in 0..d {
                ms += h_proj[base + i] * h_proj[base + i];
            }
            let inv = 1.0 / (ms / d as f32 + gen.config.rms_norm_eps).sqrt();
            for i in 0..d {
                let w = pn.get(i).copied().unwrap_or(1.0);
                h_proj[base + i] *= inv * w;
            }
        }
    }
    let input_scale = 1.0 / 2.0f32.sqrt();
    for i in 0..total {
        tok_ple[i] = (tok_ple[i] + h_proj[i]) * input_scale;
    }
    gen.io_bytes += proj.nbytes() as u64;
    Ok(tok_ple)
}

fn apply_per_layer_emb(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    layer: usize,
    meta: &GemmaMeta,
    ple_all: &[f32],
    x: &mut [f32],
) -> Result<(), StreamInferError> {
    let d = meta.per_layer_dim;
    let h = x.len();
    let base = layer * d;
    if base + d > ple_all.len() {
        return Ok(());
    }
    let pe_in = x.to_vec();
    let gate = gen
        .catalog
        .load_quant_matrix(&format!("blk.{layer}.inp_gate.weight"))?;
    // inp_gate: [n_embd → d]
    let mut g = vec![0.0f32; d];
    orch.execute_quant_gemv(&gate, x, &mut g)?;
    for v in g.iter_mut() {
        // GELU
        let t = *v;
        let t3 = t * t * t;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (t + 0.044715 * t3);
        *v = 0.5 * t * (1.0 + inner.tanh());
    }
    let ple = &ple_all[base..base + d];
    for i in 0..d {
        g[i] *= ple[i];
    }
    let proj = gen
        .catalog
        .load_quant_matrix(&format!("blk.{layer}.proj.weight"))?;
    // proj: [d → n_embd]
    let mut delta = vec![0.0f32; h];
    orch.execute_quant_gemv(&proj, &g, &mut delta)?;
    if let Ok(pn) = gen
        .catalog
        .dequant_f32(&format!("blk.{layer}.post_norm.weight"))
    {
        rms_norm(&mut delta, &pn, gen.config.rms_norm_eps);
    }
    for i in 0..h {
        x[i] = pe_in[i] + delta[i];
    }
    gen.io_bytes += (gate.nbytes() + proj.nbytes()) as u64;
    Ok(())
}

fn apply_head_rmsnorm(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut ms = 0.0f32;
        for i in 0..head_dim {
            ms += x[base + i] * x[base + i];
        }
        let inv = 1.0 / (ms / head_dim as f32 + 1e-6).sqrt();
        for i in 0..head_dim {
            let w = weight.get(i).copied().unwrap_or(1.0);
            x[base + i] *= inv * w;
        }
    }
}

fn rms_norm_plain_heads(x: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut ms = 0.0f32;
        for i in 0..head_dim {
            ms += x[base + i] * x[base + i];
        }
        let inv = 1.0 / (ms / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            x[base + i] *= inv;
        }
    }
}
