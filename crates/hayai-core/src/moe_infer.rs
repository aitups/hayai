//! MoE forward: sparse expert streaming from disk.
//!
//! Per block per token:
//! 1. Stage attn + router (+ shared expert) into the layer slot.
//! 2. Attention (CPU per binding) → residual.
//! 3. Router GEMV (CPU, small: hidden → n_expert) → top-k.
//! 4. Load **only** the selected experts into the slot (colibri-style sparse disk
//!    streaming: ~top_k/expert_count of the dense FFN bandwidth per token).
//! 5. Per-expert gate/up/down via `execute_op` (GpuAsync binding → OpenCL pool),
//!    SiLU gating, softmax routing weights, accumulated residual.

use crate::exec_plan::{is_expert_op, op_binding, ExpertUnit, LayerOpKind, TensorRef};
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{StreamInferError, StreamingGenerator};
use hayai_cpu::{attention_decode_step, rms_norm};
use hayai_model::QuantMatrix;
use hayai_opencl::StreamingScratch;
use std::collections::{HashMap, VecDeque};
use std::thread::JoinHandle;
use std::time::Instant;

pub(crate) fn prefill_moe(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    prompt_ids: &[u32],
    scratch: &mut StreamingScratch,
) -> Result<Vec<f32>, StreamInferError> {
    let mut last = Vec::new();
    for &tok in prompt_ids {
        last = forward_moe(gen, orch, tok, scratch)?;
    }
    Ok(last)
}

pub(crate) fn forward_moe(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    token: u32,
    scratch: &mut StreamingScratch,
) -> Result<Vec<f32>, StreamInferError> {
    let h = gen.config.hidden_size;
    let eps = gen.config.rms_norm_eps;
    let n_layers = gen.config.num_layers;
    let pos = gen.position;
    let plan = gen
        .exec_plan
        .clone()
        .ok_or_else(|| StreamInferError::Msg("MoE forward without ExecPlan".into()))?;
    let mm = plan
        .moe
        .ok_or_else(|| StreamInferError::Msg("MoE forward without MoeMeta".into()))?;

    let mut x = vec![0.0f32; h];
    gen.embed_row("token_embd.weight", token, h, &mut x)?;

    // Colibri-style pipeline: the next layer's non-expert tensors are prefetched
    // into the OTHER ping-pong slot while this layer's attention+FFN compute runs
    // (spawned before attention, joined after the expert compute). Same shape as
    // the dense `forward_staged` prefetch.
    let mut prefetch: Option<JoinHandle<Result<usize, StreamInferError>>> = None;
    let mut prefetched_ready = false;

    for layer in 0..n_layers {
        let unit = plan
            .units
            .iter()
            .find(|u| u.block_id == Some(layer))
            .cloned()
            .ok_or_else(|| StreamInferError::Msg(format!("plan missing blk.{layer}")))?;

        // Compact non-expert (attn + router + shared expert) offsets. Expert tensors
        // (per-expert `ffn_exp.E` or fused `ffn_*_exps`) are excluded — they stream
        // on demand via `unit.experts`.
        let mut off = 0usize;
        let mut loaded: Vec<(TensorRef, usize)> = Vec::new();
        for t in unit
            .tensors
            .iter()
            .filter(|t| !is_expert_op(t.op))
        {
            loaded.push((t.clone(), off));
            off += t.nbytes;
        }

        // Stage non-expert tensors into the layer slot (resident: preloaded once;
        // streaming: prefetched by the previous layer or loaded here).
        scratch.prepare_host_write(&orch.pool, layer)?;
        let non_expert_prefetched = prefetched_ready;
        prefetched_ready = false;
        if !scratch.resident && !non_expert_prefetched {
            let t_io = Instant::now();
            {
                let specs: Vec<(&str, usize, usize, usize)> = loaded
                    .iter()
                    .map(|(t, o)| (t.name.as_str(), 0, *o, t.nbytes))
                    .collect();
                let dst = scratch.host_slot_mut(layer);
                gen.catalog.load_tensors_into(&specs, dst)?;
            }
            gen.io_secs += t_io.elapsed().as_secs_f64();
            gen.io_bytes += off as u64;
        }

        // ── Prefetch the NEXT layer's non-expert into the other ping-pong slot ──
        // Spawned BEFORE attention so the I/O overlaps with this layer's compute.
        // Streaming (block_k<=1) only: in macro-chunk mode each layer lives at
        // `(layer % block_k) × stride` inside its block slot, so the simple
        // `(layer+1) % 2` ping-pong target is invalid.
        if !scratch.resident && scratch.block_k <= 1 && layer + 1 < n_layers {
            let next_unit = plan
                .units
                .iter()
                .find(|u| u.block_id == Some(layer + 1))
                .cloned();
            if let Some(next_unit) = next_unit {
                let mut specs: Vec<(String, usize, usize, usize)> = Vec::new();
                let mut noff = 0usize;
                for t in next_unit
                    .tensors
                    .iter()
                    .filter(|t| !is_expert_op(t.op))
                {
                    specs.push((t.name.clone(), 0, noff, t.nbytes));
                    noff += t.nbytes;
                }
                let ptr = gen.prepare_prefetch_slot(orch, scratch, (layer + 1) % 2)?;
                let mut cat = gen.catalog.fork_reader()?;
                prefetch = Some(std::thread::spawn(move || {
                    let refs: Vec<(&str, usize, usize, usize)> = specs
                        .iter()
                        .map(|(n, a, b, c)| (n.as_str(), *a, *b, *c))
                        .collect();
                    let dst = unsafe { ptr.as_mut_slice() };
                    cat.load_tensors_into(&refs, dst)
                        .map_err(StreamInferError::from)?;
                    Ok(noff)
                }));
            }
        }

        let t_attn = Instant::now();
        // ── Attention (CPU per binding) ─────────────────────────────────────────
        {
            let base = scratch.host_slot(layer);
            let mut xn = x.clone();
            rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);
            let cfg = &gen.attn_cfg;
            let q_dim = cfg.hidden_size();
            let kv_dim = cfg.kv_dim();
            let wq = view_of(&base, &loaded, LayerOpKind::AttnQ)?;
            let wk = view_of(&base, &loaded, LayerOpKind::AttnK)?;
            let wv = view_of(&base, &loaded, LayerOpKind::AttnV)?;
            let wo = view_of(&base, &loaded, LayerOpKind::AttnO)?;
            let mut q = vec![0.0f32; q_dim];
            wq.gemv(&xn, &mut q)?;
            // Per-head Q/K RMSNorm (Gemma/OLMoE-style) when present.
            if let Some(qn) = &gen.layer_norms[layer].attn_q_norm {
                crate::gemma_infer::apply_head_rmsnorm(&mut q, qn, cfg.num_heads, cfg.head_dim);
            }
            let mut k = vec![0.0f32; kv_dim];
            wk.gemv(&xn, &mut k)?;
            if let Some(kn) = &gen.layer_norms[layer].attn_k_norm {
                crate::gemma_infer::apply_head_rmsnorm(&mut k, kn, cfg.num_kv_heads, cfg.head_dim);
            }
            let mut v = vec![0.0f32; kv_dim];
            wv.gemv(&xn, &mut v)?;
            let mut attn_out = vec![0.0f32; q_dim];
            attention_decode_step(cfg, &mut gen.kv[layer], &mut q, &mut k, &v, pos, &mut attn_out);
            let mut attn_proj = vec![0.0f32; h];
            wo.gemv(&attn_out, &mut attn_proj)?;
            for i in 0..h {
                x[i] += attn_proj[i];
            }
        }
        gen.attn_secs += t_attn.elapsed().as_secs_f64();

        // ── Router → top-k (CPU; small hidden → n_expert GEMV) ──────────────────
        let mut xn = x.clone();
        rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
        let (top, weights) = {
            let base = scratch.host_slot(layer);
            let router = view_of(&base, &loaded, LayerOpKind::Router)?;
            let mut scores = vec![0.0f32; mm.expert_count];
            orch.execute_op(
                LayerOpKind::Router,
                op_binding(LayerOpKind::Router),
                &router,
                &xn,
                &mut scores,
            )?;
            let top = top_k_indices(&scores, mm.top_k);
            let weights = softmax_weights(&scores, &top);
            (top, weights)
        };

        // ── Sparse expert streaming (colibri): read ONLY the experts missing from
        // the LRU cache; hot experts are already resident in host RAM. ───────────
        // Resident mode: all experts are preloaded, so nothing is read here.
        let mut miss_meta: Vec<(usize, usize, usize, usize)> = Vec::new(); // (i,eid,base_off,pack_len)
        if !scratch.resident {
            let mut miss_specs: Vec<(&str, usize, usize, usize)> =
                Vec::with_capacity(top.len() * 3);
            let mut miss_bytes = 0usize;
            {
                let dst = scratch.host_slot_mut(layer);
                for (i, eid) in top.iter().enumerate() {
                    let expert = unit
                        .experts
                        .iter()
                        .find(|e| e.expert_id == *eid)
                        .ok_or_else(|| {
                            StreamInferError::Msg(format!("top-k expert {eid} not in plan"))
                        })?;
                    let base_off = unit.non_expert_bytes + i * unit.max_expert_bytes;
                    let pack_len = expert_pack_len(expert);
                    if let Some(bytes) = gen.moe_cache.get(layer, *eid) {
                        // Cache hit: memcpy into the slot, zero disk I/O.
                        dst[base_off..base_off + pack_len].copy_from_slice(bytes);
                        continue;
                    }
                    for t in &expert.tensors {
                        miss_specs
                            .push((t.name.as_str(), t.src_off, base_off + t.offset, t.nbytes));
                    }
                    miss_bytes += pack_len;
                    miss_meta.push((i, *eid, base_off, pack_len));
                }
            }
            if !miss_specs.is_empty() {
                let t_io = Instant::now();
                let dst = scratch.host_slot_mut(layer);
                gen.catalog.load_tensors_into(&miss_specs, dst)?;
                gen.io_secs += t_io.elapsed().as_secs_f64();
                gen.io_bytes += miss_bytes as u64;
            }
            // Populate the LRU cache with the newly loaded expert packs.
            if !miss_meta.is_empty() {
                let base = scratch.host_slot(layer);
                for (_i, eid, base_off, pack_len) in miss_meta.iter() {
                    let bytes = &base[*base_off..*base_off + pack_len];
                    gen.moe_cache.insert(layer, *eid, bytes);
                }
            }
        }

        // ── Per-expert FFN (GpuAsync binding → OpenCL pool) ─────────────────────
        let t_ffn = Instant::now();
        {
            let base = scratch.host_slot(layer);
            let mut acc = vec![0.0f32; h];
            // Pre-resolve the selected experts + their fixed (streaming/resident) offsets.
            let mut expert_meta: Vec<(usize, usize, usize, usize)> = Vec::new(); // (i,eid,base_off,ff)
            for (i, eid) in top.iter().enumerate() {
                let expert = unit
                    .experts
                    .iter()
                    .find(|e| e.expert_id == *eid)
                    .ok_or_else(|| {
                        StreamInferError::Msg(format!("top-k expert {eid} not in plan"))
                    })?;
                let base_off = if scratch.resident {
                    // Resident: expert eid sits at a fixed slot (all preloaded).
                    unit.non_expert_bytes + eid * unit.max_expert_bytes
                } else {
                    // Streaming: top-k experts packed into slots 0..top_k.
                    unit.non_expert_bytes + i * unit.max_expert_bytes
                };
                let gate = expert_view(&base, expert, base_off, LayerOpKind::ExpertGate)?;
                let up = expert_view(&base, expert, base_off, LayerOpKind::ExpertUp)?;
                expert_meta.push((i, *eid, base_off, gate.nrows.max(up.nrows)));
            }
            if orch.pool.is_empty() {
                // CPU: synchronous per expert.
                for &(i, eid, base_off, ff) in &expert_meta {
                    let expert = unit
                        .experts
                        .iter()
                        .find(|e| e.expert_id == eid)
                        .ok_or_else(|| {
                            StreamInferError::Msg(format!("top-k expert {eid} not in plan"))
                        })?;
                    let gate = expert_view(&base, expert, base_off, LayerOpKind::ExpertGate)?;
                    let up = expert_view(&base, expert, base_off, LayerOpKind::ExpertUp)?;
                    let down = expert_view(&base, expert, base_off, LayerOpKind::ExpertDown)?;
                    let mut g = vec![0.0f32; ff];
                    let mut u = vec![0.0f32; ff];
                    let mut d = vec![0.0f32; h];
                    orch.execute_op(
                        LayerOpKind::ExpertGate,
                        op_binding(LayerOpKind::ExpertGate),
                        &gate,
                        &xn,
                        &mut g,
                    )?;
                    orch.execute_op(
                        LayerOpKind::ExpertUp,
                        op_binding(LayerOpKind::ExpertUp),
                        &up,
                        &xn,
                        &mut u,
                    )?;
                    for j in 0..ff {
                        g[j] = (g[j] / (1.0 + (-g[j]).exp())) * u[j];
                    }
                    orch.execute_op(
                        LayerOpKind::ExpertDown,
                        op_binding(LayerOpKind::ExpertDown),
                        &down,
                        &g,
                        &mut d,
                    )?;
                    let w = weights[i];
                    for j in 0..h {
                        acc[j] += w * d[j];
                    }
                }
            } else {
                // GPU: async across experts — enqueue all gate∥up, wait, SiLU, down.
                gen.used_dgpu = true;
                if orch.pool.len() >= 2 {
                    gen.used_apu = true;
                }
                let mut gate_pending = Vec::with_capacity(expert_meta.len());
                let mut up_pending = Vec::with_capacity(expert_meta.len());
                for &(_i, eid, base_off, _ff) in &expert_meta {
                    let expert = unit
                        .experts
                        .iter()
                        .find(|e| e.expert_id == eid)
                        .ok_or_else(|| {
                            StreamInferError::Msg(format!("top-k expert {eid} not in plan"))
                        })?;
                    let gate = expert_view(&base, expert, base_off, LayerOpKind::ExpertGate)?;
                    let up = expert_view(&base, expert, base_off, LayerOpKind::ExpertUp)?;
                    gate_pending.push(crate::orchestrator::begin_gemv_engine(
                        orch.pool.for_role(0),
                        &gate,
                        &xn,
                    )?);
                    up_pending.push(crate::orchestrator::begin_gemv_engine(
                        orch.pool.for_role(1),
                        &up,
                        &xn,
                    )?);
                }
                let mut gs = Vec::with_capacity(expert_meta.len());
                let mut us = Vec::with_capacity(expert_meta.len());
                for (gp, up) in gate_pending.into_iter().zip(up_pending) {
                    gs.push(gp.wait()?);
                    us.push(up.wait()?);
                }
                let mut down_pending = Vec::with_capacity(expert_meta.len());
                for (k, &(_i, eid, base_off, ff)) in expert_meta.iter().enumerate() {
                    let g = &mut gs[k];
                    for j in 0..ff {
                        g[j] = (g[j] / (1.0 + (-g[j]).exp())) * us[k][j];
                    }
                    let expert = unit
                        .experts
                        .iter()
                        .find(|e| e.expert_id == eid)
                        .ok_or_else(|| {
                            StreamInferError::Msg(format!("top-k expert {eid} not in plan"))
                        })?;
                    let down = expert_view(&base, expert, base_off, LayerOpKind::ExpertDown)?;
                    down_pending.push(crate::orchestrator::begin_gemv_engine(
                        orch.pool.for_role(0),
                        &down,
                        &g[..],
                    )?);
                }
                let downs: Vec<Vec<f32>> = down_pending
                    .into_iter()
                    .map(|p| p.wait())
                    .collect::<Result<Vec<_>, _>>()?;
                for (k, &(i, _eid, _b, _ff)) in expert_meta.iter().enumerate() {
                    let d = &downs[k];
                    let w = weights[i];
                    for j in 0..h {
                        acc[j] += w * d[j];
                    }
                }
            }
            // Shared expert (DeepSeek-style) always active, weight 1.
            if let (Some(sg), Some(su), Some(sd)) = (
                shared_view(&base, &loaded, "gate")?,
                shared_view(&base, &loaded, "up")?,
                shared_view(&base, &loaded, "down")?,
            ) {
                let ff = sg.nrows.max(su.nrows);
                let mut g = vec![0.0f32; ff];
                let mut u = vec![0.0f32; ff];
                let mut d = vec![0.0f32; h];
                orch.execute_op(
                    LayerOpKind::SharedExpert,
                    op_binding(LayerOpKind::SharedExpert),
                    &sg,
                    &xn,
                    &mut g,
                )?;
                orch.execute_op(
                    LayerOpKind::SharedExpert,
                    op_binding(LayerOpKind::SharedExpert),
                    &su,
                    &xn,
                    &mut u,
                )?;
                for j in 0..ff {
                    g[j] = (g[j] / (1.0 + (-g[j]).exp())) * u[j];
                }
                orch.execute_op(
                    LayerOpKind::SharedExpert,
                    op_binding(LayerOpKind::SharedExpert),
                    &sd,
                    &g,
                    &mut d,
                )?;
                for j in 0..h {
                    acc[j] += d[j];
                }
            }
            for j in 0..h {
                x[j] += acc[j];
            }
        }
        gen.ffn_secs += t_ffn.elapsed().as_secs_f64();

        // ── Join the prefetch → next layer's non-expert is ready in its slot ──
        if let Some(handle) = prefetch.take() {
            let t_join = Instant::now();
            match handle.join() {
                Ok(Ok(bytes)) => {
                    gen.overlap_secs += t_join.elapsed().as_secs_f64();
                    gen.prefetch_hits += 1;
                    gen.io_bytes += bytes as u64;
                    prefetched_ready = true;
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(StreamInferError::Msg("MoE prefetch join panicked".into()))
                }
            }
        }
    }
    if !orch.pool.is_empty() {
        gen.used_dgpu = true;
        if orch.pool.len() >= 2 {
            gen.used_apu = true;
        }
    }

    // ── Output head ─────────────────────────────────────────────────────────────
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
    } else if let Some(emb) = &gen.resident_embed {
        orch.execute_quant_gemv(emb, &xn, &mut logits)?;
    } else {
        let emb = gen.catalog.load_quant_matrix("token_embd.weight")?;
        orch.execute_quant_gemv(&emb, &xn, &mut logits)?;
        gen.io_bytes += emb.nbytes() as u64;
    }
    gen.position += 1;
    if std::env::var("HAYAI_DUMP_TOP").ok().as_deref() == Some("1") {
        let at = std::env::var("HAYAI_DUMP_AT_POS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
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

/// View over a non-expert tensor of the block (compact offset).
fn view_of<'a>(
    base: &'a [u8],
    loaded: &[(TensorRef, usize)],
    op: LayerOpKind,
) -> Result<QuantMatrix, StreamInferError> {
    let (t, off) = loaded
        .iter()
        .find(|(t, _)| t.op == op)
        .ok_or_else(|| StreamInferError::Msg(format!("MoE block missing {op:?} tensor")))?;
    Ok(QuantMatrix::view(
        t.name.clone(),
        t.ncols,
        t.nrows,
        t.ggml_type,
        &base[*off..*off + t.nbytes],
    ))
}

/// View over one matrix of a loaded expert pack.
fn expert_view<'a>(
    base: &'a [u8],
    expert: &ExpertUnit,
    base_off: usize,
    op: LayerOpKind,
) -> Result<QuantMatrix, StreamInferError> {
    let t = expert
        .tensors
        .iter()
        .find(|t| t.op == op)
        .ok_or_else(|| {
            StreamInferError::Msg(format!("expert {} missing {op:?}", expert.expert_id))
        })?;
    Ok(QuantMatrix::view(
        t.name.clone(),
        t.ncols,
        t.nrows,
        t.ggml_type,
        &base[base_off + t.offset..base_off + t.offset + t.nbytes],
    ))
}

/// View over a shared-expert matrix (`ffn_shexp.*` / `shared_expert.*`).
fn shared_view<'a>(
    base: &'a [u8],
    loaded: &[(TensorRef, usize)],
    role: &str,
) -> Result<Option<QuantMatrix>, StreamInferError> {
    for (t, off) in loaded
        .iter()
        .filter(|(t, _)| t.op == LayerOpKind::SharedExpert)
    {
        let n = t.name.to_ascii_lowercase();
        let is_role = match role {
            "gate" => n.contains("ffn_gate") || n.contains("gate_proj") || n.contains(".w1."),
            "up" => n.contains("ffn_up") || n.contains("up_proj") || n.contains(".w3."),
            "down" => n.contains("ffn_down") || n.contains("down_proj") || n.contains(".w2."),
            _ => false,
        };
        if is_role {
            return Ok(Some(QuantMatrix::view(
                t.name.clone(),
                t.ncols,
                t.nrows,
                t.ggml_type,
                &base[*off..*off + t.nbytes],
            )));
        }
    }
    Ok(None)
}

/// Indices of the `k` highest-scoring experts (descending).
fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len()).max(1);
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

/// Softmax routing weights over the selected top-k experts.
fn softmax_weights(scores: &[f32], top: &[usize]) -> Vec<f32> {
    let max = top
        .iter()
        .map(|&i| scores[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut w: Vec<f32> = top.iter().map(|&i| (scores[i] - max).exp()).collect();
    let sum: f32 = w.iter().sum();
    if sum <= 0.0 {
        let n = w.len() as f32;
        for v in w.iter_mut() {
            *v = 1.0 / n;
        }
    } else {
        for v in w.iter_mut() {
            *v /= sum;
        }
    }
    w
}

/// Total bytes of one expert pack (gate + up + down).
fn expert_pack_len(expert: &ExpertUnit) -> usize {
    expert.tensors.iter().map(|t| t.nbytes).sum()
}

/// LRU cache of recently-used MoE expert packs (colibri-style). Hot experts stay
/// resident in host RAM so repeat top-k selections skip disk reads — the dominant
/// win of "stream experts on demand". Capacity in packs; `0` disables caching.
pub(crate) struct ExpertCache {
    capacity: usize,
    slots: Vec<Option<(usize, usize, Vec<u8>)>>,
    map: HashMap<(usize, usize), usize>,
    lru: VecDeque<usize>,
    pub(crate) hits: usize,
    pub(crate) misses: usize,
}

impl ExpertCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity,
            slots: vec![None; capacity],
            map: HashMap::new(),
            lru: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Cache lookup for `(layer, expert_id)`. Moves the slot to the LRU front.
    pub(crate) fn get(&mut self, layer: usize, expert_id: usize) -> Option<&[u8]> {
        if self.capacity == 0 {
            return None;
        }
        let slot = *self.map.get(&(layer, expert_id))?;
        if let Some(pos) = self.lru.iter().position(|&s| s == slot) {
            self.lru.remove(pos);
        }
        self.lru.push_front(slot);
        self.hits += 1;
        self.slots[slot].as_ref().map(|c| c.2.as_slice())
    }

    /// Store `bytes` for `(layer, expert_id)`, evicting the least-recently-used
    /// slot when at capacity.
    pub(crate) fn insert(&mut self, layer: usize, expert_id: usize, bytes: &[u8]) {
        if self.capacity == 0 {
            return;
        }
        if self.map.contains_key(&(layer, expert_id)) {
            return;
        }
        self.misses += 1;
        let slot = if let Some(free) = self.slots.iter().position(|s| s.is_none()) {
            free
        } else if let Some(victim) = self.lru.pop_back() {
            if let Some(entry) = &self.slots[victim] {
                self.map.remove(&(entry.0, entry.1));
            }
            victim
        } else {
            return;
        };
        self.slots[slot] = Some((layer, expert_id, bytes.to_vec()));
        self.map.insert((layer, expert_id), slot);
        self.lru.push_front(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_picks_highest_scores() {
        let scores = [0.1, 0.9, 0.5, -0.2, 0.7];
        assert_eq!(top_k_indices(&scores, 2), vec![1, 4]);
        assert_eq!(top_k_indices(&scores, 10), vec![1, 4, 2, 0, 3]);
        assert_eq!(top_k_indices(&scores, 0), vec![1]);
    }

    #[test]
    fn softmax_weights_sum_to_one() {
        let scores = [0.1, 0.9, 0.5, -0.2, 0.7];
        let top = top_k_indices(&scores, 2);
        let w = softmax_weights(&scores, &top);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // The best expert dominates the weight.
        assert!(w[0] > w[1]);
    }

    #[test]
    fn expert_cache_hits_and_evicts_lru() {
        let mut c = ExpertCache::new(2);
        c.insert(0, 1, b"pack-a");
        c.insert(0, 2, b"pack-b");
        // Hit expert 1 (moves it to the front).
        assert_eq!(c.get(0, 1), Some(&b"pack-a"[..]));
        // Insert a third → evicts expert 2 (LRU at the back).
        c.insert(0, 3, b"pack-c");
        assert!(c.get(0, 2).is_none());
        assert_eq!(c.get(0, 1), Some(&b"pack-a"[..]));
        assert_eq!(c.get(0, 3), Some(&b"pack-c"[..]));
        assert!(c.hits >= 3);
        assert_eq!(c.misses, 3);
        // Disabled cache never hits.
        let mut off = ExpertCache::new(0);
        assert!(off.get(0, 1).is_none());
        off.insert(0, 1, b"x");
        assert!(off.get(0, 1).is_none());
    }
}

