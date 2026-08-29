//! Plan-driven hybrid forward (Qwen3.5): full-attn blocks + DeltaNet/SSM blocks.
//!
//! Layer routing and dims are resolved **per physical block** (tensor presence / shapes),
//! not from a single model-size hardcode.

use crate::deltanet::{is_deltanet_layer, DeltaNetLayerWeights, DeltaNetState};
use crate::infer::FfnOverride;
use crate::layer_cfg::{hybrid_layer_kind, resolve_full_attn_cfg, HybridLayerKind};
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{
    ffn_begin_gate_up_scratch, ffn_finish_scratch, StreamInferError, StreamingGenerator,
};
use hayai_cpu::{attention_decode_step, rms_norm};
use hayai_opencl::StreamingScratch;
use std::time::Instant;

pub(crate) fn prefill_hybrid(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    tokens: &[u32],
    scratch: &mut StreamingScratch,
) -> Result<Vec<f32>, StreamInferError> {
    let mut logits = Vec::new();
    for &tok in tokens {
        logits = forward_hybrid(gen, orch, tok, scratch, None)?;
    }
    Ok(logits)
}

pub(crate) fn forward_hybrid(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    token: u32,
    scratch: &mut StreamingScratch,
    override_ffn: Option<&[FfnOverride]>,
) -> Result<Vec<f32>, StreamInferError> {
    let h = gen.config.hidden_size;
    let eps = gen.config.rms_norm_eps;
    let n_layers = gen.config.num_layers;
    let pos = gen.position;

    let mut x = vec![0.0f32; h];
    gen.embed_row("token_embd.weight", token, h, &mut x)?;
    if std::env::var("HAYAI_DUMP_LAYER_RMS").ok().as_deref() == Some("1") && pos == 0 {
        let mut ms = 0.0f32;
        for &v in x.iter() {
            ms += v * v;
        }
        eprintln!(
            "HAYAI_LAYER_RMS tok0 embed: {:.4}",
            (ms / x.len() as f32).sqrt()
        );
    }

    // Lazy DeltaNet weight/state cache on the generator via owned_mem side channel —
    // store in thread-local for this session would be cleaner; use Vec on first call.
    ensure_deltanet_cache(gen)?;

    // Ablation (keep FFN always):
    //   HAYAI_SKIP_DN_ATTN=1  — DeltaNet residual = identity (FFN still runs)
    //   HAYAI_SKIP_FA_ATTN=1  — full-attn residual = identity (FFN still runs)
    let skip_dn = std::env::var("HAYAI_SKIP_DN_ATTN").ok().as_deref() == Some("1");
    let skip_fa = std::env::var("HAYAI_SKIP_FA_ATTN").ok().as_deref() == Some("1");
    let dump_rms = std::env::var("HAYAI_DUMP_LAYER_RMS").ok().as_deref() == Some("1");
    let max_layers = std::env::var("HAYAI_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(n_layers)
        .min(n_layers);
    for layer in 0..max_layers {
        let ov = override_ffn.and_then(|o| o.get(layer));
        match hybrid_layer_kind(&gen.catalog, layer) {
            HybridLayerKind::NextN => continue,
            HybridLayerKind::DeltaNet => {
                if skip_dn {
                    run_ffn_only_block(gen, orch, scratch, layer, eps, &mut x, ov)?;
                } else {
                    run_deltanet_block(gen, orch, scratch, layer, eps, &mut x, ov)?;
                }
            }
            HybridLayerKind::FullAttn => {
                if skip_fa {
                    run_ffn_only_block(gen, orch, scratch, layer, eps, &mut x, ov)?;
                } else {
                    run_full_attn_block(gen, orch, scratch, layer, pos, eps, &mut x, ov)?;
                }
            }
        }
        if dump_rms && pos == 0 {
            let mut ms = 0.0f32;
            for &v in x.iter() {
                ms += v * v;
            }
            let rms = (ms / x.len() as f32).sqrt();
            let kind = match hybrid_layer_kind(&gen.catalog, layer) {
                HybridLayerKind::DeltaNet => "DN",
                HybridLayerKind::FullAttn => "FA",
                HybridLayerKind::NextN => "N",
            };
            eprintln!("HAYAI_LAYER_RMS tok0 L{layer}({kind}): {rms:.4}");
        }
    }

    let mut xn = x;
    rms_norm(&mut xn, &gen.output_norm, eps);
    let vocab = gen.config.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    if gen.has_output_weight {
        if let Some(ow) = &gen.resident_output {
            orch.execute_quant_gemv(ow, &xn, &mut logits)?;
        } else {
            let t0 = Instant::now();
            let ow = gen.catalog.load_quant_matrix("output.weight")?;
            gen.io_secs += t0.elapsed().as_secs_f64();
            gen.io_bytes += ow.nbytes() as u64;
            orch.execute_quant_gemv(&ow, &xn, &mut logits)?;
        }
    } else {
        if let Some(emb) = &gen.resident_embed {
            orch.execute_quant_gemv(emb, &xn, &mut logits)?;
        } else {
            let t0 = Instant::now();
            let emb = gen.catalog.load_quant_matrix("token_embd.weight")?;
            gen.io_secs += t0.elapsed().as_secs_f64();
            gen.io_bytes += emb.nbytes() as u64;
            orch.execute_quant_gemv(&emb, &xn, &mut logits)?;
        }
    }
    gen.position += 1;
    if std::env::var("HAYAI_DUMP_TOP").ok().as_deref() == Some("1") {
        let at = std::env::var("HAYAI_DUMP_AT_POS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        if gen.position == at {
            dump_top_logits(&logits, 8);
        }
    }
    Ok(logits)
}

fn dump_top_logits(logits: &[f32], k: usize) {
    let mut pairs: Vec<(f32, usize)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprint!("HAYAI_DUMP_TOP:");
    for &(v, i) in pairs.iter().take(k) {
        eprint!(" {i}:{v}");
    }
    eprintln!();
}

fn ensure_deltanet_cache(gen: &mut StreamingGenerator) -> Result<(), StreamInferError> {
    if gen.deltanet_weights.is_some() {
        return Ok(());
    }
    let n = gen.config.num_layers;
    let mut weights = Vec::with_capacity(n);
    let mut states = Vec::with_capacity(n);
    for i in 0..n {
        if is_deltanet_layer(&gen.catalog, i) {
            let w = DeltaNetLayerWeights::load(&mut gen.catalog, i)?;
            let st = DeltaNetState::new(w.conv_k, w.conv_dim, w.n_v_heads, w.head_k, w.head_v);
            weights.push(Some(w));
            states.push(Some(st));
        } else {
            weights.push(None);
            states.push(None);
        }
    }
    gen.deltanet_weights = Some(weights);
    gen.deltanet_states = Some(states);
    Ok(())
}

/// FFN-only residual block (ablation helper): `x += FFN(rms_norm(x))`.
fn run_ffn_only_block(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    scratch: &mut StreamingScratch,
    layer: usize,
    eps: f32,
    x: &mut [f32],
    ov: Option<&FfnOverride>,
) -> Result<(), StreamInferError> {
    let h = x.len();
    let t_io = Instant::now();
    let gate = gen
        .catalog
        .load_quant_matrix(&format!("blk.{layer}.ffn_gate.weight"))?;
    let up = gen
        .catalog
        .load_quant_matrix(&format!("blk.{layer}.ffn_up.weight"))?;
    let down = gen
        .catalog
        .load_quant_matrix(&format!("blk.{layer}.ffn_down.weight"))?;
    gen.io_secs += t_io.elapsed().as_secs_f64();
    gen.io_bytes += (gate.nbytes() + up.nbytes() + down.nbytes()) as u64;

    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
    gen.ws_gate.fill(0.0);
    gen.ws_up.fill(0.0);
    gen.ws_down.fill(0.0);
    let t_ffn = Instant::now();
    let has_ov = ov.map(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some()).unwrap_or(false);
    if has_ov {
        let pack = hayai_model::LayerWeightPack {
            wq: gate.clone(),
            wk: gate.clone(),
            wv: gate.clone(),
            wo: gate.clone(),
            gate,
            up,
            down,
            attn_gate: None,
            gate_csr: None,
            up_csr: None,
            down_csr: None,
        };
        gen.run_ffn_block(orch, &pack, &xn, ov)?;
    } else {
        let inflight = ffn_begin_gate_up_scratch(
            orch,
            &gate,
            &up,
            &xn,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.used_dgpu,
            &mut gen.used_apu,
            None,
            0,
            None,
        )?;
        ffn_finish_scratch(
            orch,
            inflight,
            &down,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.ws_down,
            &mut gen.used_dgpu,
            None,
            0,
            None,
        )?;
    }
    gen.ffn_secs += t_ffn.elapsed().as_secs_f64();
    for i in 0..h {
        x[i] += gen.ws_down[i];
    }
    let _ = scratch;
    Ok(())
}

fn run_deltanet_block(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    scratch: &mut StreamingScratch,
    layer: usize,
    eps: f32,
    x: &mut [f32],
    ov: Option<&FfnOverride>,
) -> Result<(), StreamInferError> {
    let h = x.len();
    // Norm + DeltaNet residual
    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);

    {
        let weights = gen
            .deltanet_weights
            .as_ref()
            .and_then(|w| w[layer].as_ref())
            .ok_or_else(|| StreamInferError::Msg("missing DeltaNet weights".into()))?;
        // Split borrow: state mutably, weights immutably via raw indices.
        let _ = weights;
    }
    let t0 = Instant::now();
    {
        let (w_ptr, s_ptr) = {
            let w = gen.deltanet_weights.as_mut().unwrap();
            let s = gen.deltanet_states.as_mut().unwrap();
            (
                w[layer].as_ref().unwrap() as *const DeltaNetLayerWeights,
                s[layer].as_mut().unwrap() as *mut DeltaNetState,
            )
        };
        unsafe {
            (*w_ptr).decode_step(&xn, &mut *s_ptr, x)?;
        }
    }
    if std::env::var("HAYAI_DUMP_LAYER_RMS").ok().as_deref() == Some("1") && gen.position == 0 {
        let mut ms = 0.0f32;
        for &v in x.iter() {
            ms += v * v;
        }
        eprintln!(
            "HAYAI_LAYER_RMS tok0 L{layer} afterDN: {:.4}",
            (ms / x.len() as f32).sqrt()
        );
    }
    gen.attn_secs += t0.elapsed().as_secs_f64();

    // FFN via WeightIo (owned matrices — DeltaNet layers have no attn_q pack).
    // Los bloques dispersos embebidos (D16) cargan con CSR.
    let t_io = Instant::now();
    let (gate, up, down, gate_csr, up_csr, down_csr) =
        gen.catalog.load_ffn_matrices(layer)?;
    gen.io_secs += t_io.elapsed().as_secs_f64();
    gen.io_bytes += (gate.nbytes() + up.nbytes() + down.nbytes()) as u64;

    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
    gen.ws_gate.fill(0.0);
    gen.ws_up.fill(0.0);
    gen.ws_down.fill(0.0);
    let t_ffn = Instant::now();
    if gate_csr.is_some() || up_csr.is_some() || down_csr.is_some()
        || ov.map(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some()).unwrap_or(false)
    {
        // FFN disperso embebido (D16) u override de evolución: CSR.
        let pack = hayai_model::LayerWeightPack {
            wq: gate.clone(),
            wk: gate.clone(),
            wv: gate.clone(),
            wo: gate.clone(),
            gate,
            up,
            down,
            attn_gate: None,
            gate_csr,
            up_csr,
            down_csr,
        };
        gen.run_ffn_block(orch, &pack, &xn, ov)?;
    } else {
        let inflight = ffn_begin_gate_up_scratch(
            orch,
            &gate,
            &up,
            &xn,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.used_dgpu,
            &mut gen.used_apu,
            None,
            0,
            None,
        )?;
        ffn_finish_scratch(
            orch,
            inflight,
            &down,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.ws_down,
            &mut gen.used_dgpu,
            None,
            0,
            None,
        )?;
    }
    gen.ffn_secs += t_ffn.elapsed().as_secs_f64();
    for i in 0..h {
        x[i] += gen.ws_down[i];
    }
    let _ = scratch;
    Ok(())
}

fn run_full_attn_block(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    scratch: &mut StreamingScratch,
    layer: usize,
    pos: usize,
    eps: f32,
    x: &mut [f32],
    ov: Option<&FfnOverride>,
) -> Result<(), StreamInferError> {
    let h = x.len();
    let slot = layer % 2;
    let (pack, layout) = gen.stage_pack(orch, scratch, slot, layer)?;
    gen.begin_ffn_dma(orch, scratch, slot, &layout)?;

    let t_attn = Instant::now();
    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);
    // Qwen3.5 full-attn: `attn_q` is fused Q∥gate interleaved per head
    // (llama.cpp view stride = 2 * head_dim), not [Q|Gate] halves.
    let wo_in = pack.wo.ncols;
    let mut q_full = vec![0.0f32; pack.wq.nrows];
    let mut k = vec![0.0f32; pack.wk.nrows];
    let mut v = vec![0.0f32; pack.wv.nrows];
    pack.wq.gemv(&xn, &mut q_full)?;
    pack.wk.gemv(&xn, &mut k)?;
    pack.wv.gemv(&xn, &mut v)?;

    // Per-layer dims from this block's tensors (9B/27B may differ from 4B meta defaults).
    let layer_cfg = resolve_full_attn_cfg(&gen.catalog, layer, &gen.config)?;
    let n_heads = layer_cfg.num_heads;
    let n_kv = layer_cfg.num_kv_heads;
    let head_dim = layer_cfg.head_dim;
    // Fused Q∥gate is interleaved per head: [Q0|G0|Q1|G1|…] (llama.cpp / HF Qwen3.5).
    let (mut q, fused_gate) = if q_full.len() == n_heads * head_dim * 2 {
        deinterleave_qg(&q_full, n_heads, head_dim)
    } else if q_full.len() == wo_in * 2 {
        let (qh, gh) = q_full.split_at(wo_in);
        (qh.to_vec(), Some(gh.to_vec()))
    } else {
        (q_full, None)
    };

    let kv_dim = n_kv * head_dim;
    let q_dim = n_heads * head_dim;
    q.resize(q_dim, 0.0);
    k.resize(kv_dim, 0.0);
    v.resize(kv_dim, 0.0);
    // Q/K RMSNorm before RoPE (llama.cpp order).
    if let Ok(qn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_q_norm.weight")) {
        apply_head_rmsnorm(&mut q, &qn, n_heads, head_dim, eps);
    }
    if let Ok(kn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_k_norm.weight")) {
        apply_head_rmsnorm(&mut k, &kn, n_kv, head_dim, eps);
    }
    if gen.kv[layer].heads.len() != n_kv || gen.kv[layer].heads[0].dim != head_dim {
        return Err(StreamInferError::Msg(format!(
            "blk.{layer}: KV cache heads/dim mismatch (cache {}×{}, layer {n_kv}×{head_dim})",
            gen.kv[layer].heads.len(),
            gen.kv[layer].heads.first().map(|h| h.dim).unwrap_or(0)
        )));
    }
    let mut attn_out = vec![0.0f32; q_dim];
    attention_decode_step(
        &layer_cfg,
        &mut gen.kv[layer],
        &mut q,
        &mut k,
        &v,
        pos,
        &mut attn_out,
    );
    if let Some(ref g) = fused_gate {
        for i in 0..attn_out.len().min(g.len()) {
            attn_out[i] *= 1.0 / (1.0 + (-g[i]).exp());
        }
    } else if let Some(ref gate_w) = pack.attn_gate {
        let mut gate = vec![0.0f32; gate_w.nrows];
        gate_w.gemv(&xn, &mut gate)?;
        for i in 0..attn_out.len().min(gate.len()) {
            attn_out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
        }
    }
    attn_out.resize(wo_in, 0.0);
    let mut attn_proj = vec![0.0f32; h];
    pack.wo.gemv(&attn_out, &mut attn_proj)?;
    for i in 0..h {
        x[i] += attn_proj[i];
    }
    gen.attn_secs += t_attn.elapsed().as_secs_f64();

    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
    gen.ws_gate.fill(0.0);
    gen.ws_up.fill(0.0);
    gen.ws_down.fill(0.0);
    gen.finish_ffn_unmap(orch, scratch, slot)?;
    let t_ffn = Instant::now();
    let sparse_ffn =
        pack.gate_csr.is_some() || pack.up_csr.is_some() || pack.down_csr.is_some();
    let has_ov = ov.map(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some()).unwrap_or(false);
    if sparse_ffn || has_ov {
        // FFN disperso embebido (D16) u override de evolución: CSR.
        gen.run_ffn_block(orch, &pack, &xn, ov)?;
    } else {
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
        ffn_finish_scratch(
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
    }
    gen.ffn_secs += t_ffn.elapsed().as_secs_f64();
    for i in 0..h {
        x[i] += gen.ws_down[i];
    }
    let _ = pack;
    Ok(())
}

fn apply_head_rmsnorm(
    x: &mut [f32],
    weight: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut ms = 0.0f32;
        for i in 0..head_dim {
            ms += x[base + i] * x[base + i];
        }
        let inv = 1.0 / (ms / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            let w = weight.get(i).copied().unwrap_or(1.0);
            x[base + i] *= inv * w;
        }
    }
}

/// Split fused Qwen3.5 Q-projection: `[Q0|G0|Q1|G1|…]` → `(Q, gate)`.
fn deinterleave_qg(q_full: &[f32], n_heads: usize, head_dim: usize) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut q = vec![0.0f32; n_heads * head_dim];
    let mut g = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let src = h * 2 * head_dim;
        let dst = h * head_dim;
        q[dst..dst + head_dim].copy_from_slice(&q_full[src..src + head_dim]);
        g[dst..dst + head_dim].copy_from_slice(&q_full[src + head_dim..src + 2 * head_dim]);
    }
    (q, Some(g))
}
