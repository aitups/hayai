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

use crate::exec_plan::{is_expert_op, op_binding, ExpertUnit, LayerOpKind, MoeMeta, TensorRef};
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{PrefetchJob, PrefetchWorker, StreamInferError, StreamingGenerator};
use hayai_cpu::{apply_rope_partial_factors_scaled, attention_decode_step, rms_norm, softmax};
use hayai_model::QuantMatrix;
use hayai_opencl::StreamingScratch;
use std::collections::{HashMap, VecDeque};
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
    let mut prefetch_bytes: Option<usize> = None;
    let mut prefetched_ready = false;

    for layer in 0..n_layers {
        let unit = plan
            .units
            .iter()
            .find(|u| u.block_id == Some(layer))
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
        // streaming: served from the bounded non-expert RAM cache, from the
        // prefetched slot, or read from disk on the first token of this layer).
        scratch.prepare_host_write(&orch.pool, layer)?;
        let non_expert_prefetched = prefetched_ready;
        prefetched_ready = false;
        if !scratch.resident {
            let cached_hit = gen
                .moe_non_expert
                .as_ref()
                .and_then(|c| c.get(layer))
                .and_then(|c| c.as_deref())
                .is_some();
            if cached_hit {
                // RAM hit: memcpy the compact non-expert block, zero disk I/O.
                let cache = gen.moe_non_expert.as_ref().expect("cache");
                let buf = cache[layer].as_deref().expect("cached");
                scratch.host_slot_mut(layer)[..off].copy_from_slice(buf);
            } else {
                if !non_expert_prefetched {
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
                // Cache the freshly staged block if it fits the budget.
                if gen.moe_non_expert_bytes + off <= gen.moe_non_expert_cap {
                    let bytes = scratch.host_slot(layer)[..off].to_vec().into_boxed_slice();
                    let cache = gen
                        .moe_non_expert
                        .get_or_insert_with(|| vec![None; n_layers]);
                    if cache[layer].is_none() {
                        cache[layer] = Some(bytes);
                        gen.moe_non_expert_bytes += off;
                    }
                }
            }
        }

        // ── Prefetch the NEXT layer's non-expert into the other ping-pong slot ──
        // Spawned BEFORE attention so the I/O overlaps with this layer's compute.
        // Skipped when the next layer is already in the RAM cache.
        // Streaming (block_k<=1) only: in macro-chunk mode each layer lives at
        // `(layer % block_k) × stride` inside its block slot, so the simple
        // `(layer+1) % 2` ping-pong target is invalid.
        let next_cached = gen
            .moe_non_expert
            .as_ref()
            .and_then(|c| c.get(layer + 1))
            .map(|c| c.is_some())
            .unwrap_or(false);
        if !scratch.resident && scratch.block_k <= 1 && layer + 1 < n_layers && !next_cached {
            let next_unit = plan
                .units
                .iter()
                .find(|u| u.block_id == Some(layer + 1));
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
                let slot_ptr = gen.prepare_prefetch_slot(orch, scratch, (layer + 1) % 2)?;
                if gen.prefetch_worker.is_none() {
                    let cat = gen.catalog.fork_reader()?;
                    gen.prefetch_worker = Some(PrefetchWorker::new(cat));
                }
                gen.prefetch_worker.as_ref().unwrap().submit(PrefetchJob {
                    read: Box::new(move |cat| {
                        let refs: Vec<(&str, usize, usize, usize)> = specs
                            .iter()
                            .map(|(n, a, b, c)| (n.as_str(), *a, *b, *c))
                            .collect();
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(slot_ptr.addr as *mut u8, slot_ptr.len)
                        };
                        cat.load_tensors_into(&refs, dst)
                    }),
                })?;
                prefetch_bytes = Some(noff);
            }
        }

        let t_attn = Instant::now();
        // ── Attention (CPU per binding) ─────────────────────────────────────────
        if gen.mla.is_some()
            && view_of(scratch.host_slot(layer), &loaded, LayerOpKind::MlaKvA).is_ok()
        {
            mla_attention(gen, layer, scratch.host_slot(layer), &loaded, pos, &mut x)?;
        } else {
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
            // Q/K RMSNorm: per-head (`weight.len()==head_dim`, Gemma) or over the whole
            // projection (`weight.len()==q_dim`, OLMoE).
            if let Some(qn) = &gen.layer_norms[layer].attn_q_norm {
                apply_qk_norm(&mut q, qn, cfg.num_heads, cfg.head_dim, eps);
            }
            let mut k = vec![0.0f32; kv_dim];
            wk.gemv(&xn, &mut k)?;
            if let Some(kn) = &gen.layer_norms[layer].attn_k_norm {
                apply_qk_norm(&mut k, kn, cfg.num_kv_heads, cfg.head_dim, eps);
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
        // Optional depthwise causal short-conv residual (generic `Conv` op).
        gen.apply_conv(layer, &mut x)?;

        // ── Router → top-k (CPU; small hidden → n_expert GEMV) ──────────────────
        let mut xn = x.clone();
        rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
        // DeepSeek leading dense blocks have no router: dense SwiGLU FFN.
        let base_ffn = scratch.host_slot(layer);
        let dense_lead = view_of(base_ffn, &loaded, LayerOpKind::Router).is_err()
            && view_of(base_ffn, &loaded, LayerOpKind::FfnGate).is_ok();
        let t_ffn = Instant::now();
        if dense_lead {
            let gate = view_of(base_ffn, &loaded, LayerOpKind::FfnGate)?;
            let up = view_of(base_ffn, &loaded, LayerOpKind::FfnUp)?;
            let down = view_of(base_ffn, &loaded, LayerOpKind::FfnDown)?;
            let ff = gate.nrows;
            let mut g = vec![0.0f32; ff];
            let mut u = vec![0.0f32; ff];
            let mut d = vec![0.0f32; h];
            gate.gemv(&xn, &mut g)?;
            up.gemv(&xn, &mut u)?;
            for j in 0..ff {
                g[j] = (g[j] / (1.0 + (-g[j]).exp())) * u[j];
            }
            down.gemv(&g, &mut d)?;
            for j in 0..h {
                x[j] += d[j];
            }
        } else {
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
            let bias = gen
                .moe_router_bias
                .as_ref()
                .and_then(|v| v.get(layer))
                .and_then(|b| b.as_deref());
            let (top, weights) = route_experts(&scores, bias, &mm);
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
                // CPU: experts are independent → evaluate them in parallel (rayon).
                use rayon::prelude::*;
                let partials: Vec<(f32, Vec<f32>)> = expert_meta
                    .par_iter()
                    .map(
                        |&(i, eid, base_off, ff)| -> Result<(f32, Vec<f32>), StreamInferError> {
                            let expert = unit
                                .experts
                                .iter()
                                .find(|e| e.expert_id == eid)
                                .ok_or_else(|| {
                                    StreamInferError::Msg(format!(
                                        "top-k expert {eid} not in plan"
                                    ))
                                })?;
                            let gate =
                                expert_view(&base, expert, base_off, LayerOpKind::ExpertGate)?;
                            let up = expert_view(&base, expert, base_off, LayerOpKind::ExpertUp)?;
                            let down =
                                expert_view(&base, expert, base_off, LayerOpKind::ExpertDown)?;
                            let mut g = vec![0.0f32; ff];
                            let mut u = vec![0.0f32; ff];
                            let mut d = vec![0.0f32; h];
                            gate.gemv(&xn, &mut g)?;
                            up.gemv(&xn, &mut u)?;
                            for j in 0..ff {
                                g[j] = (g[j] / (1.0 + (-g[j]).exp())) * u[j];
                            }
                            down.gemv(&g, &mut d)?;
                            Ok((weights[i], d))
                        },
                    )
                    .collect::<Result<Vec<_>, _>>()?;
                for (w, d) in partials {
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
                        orch
                            .pool
                            .for_role(0)
                            .ok_or_else(|| StreamInferError::Msg("empty GPU pool".into()))?,
                        &gate,
                        &xn,
                    )?);
                    up_pending.push(crate::orchestrator::begin_gemv_engine(
                        orch
                            .pool
                            .for_role(1)
                            .ok_or_else(|| StreamInferError::Msg("empty GPU pool".into()))?,
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
                        orch
                            .pool
                            .for_role(0)
                            .ok_or_else(|| StreamInferError::Msg("empty GPU pool".into()))?,
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
        }
        gen.ffn_secs += t_ffn.elapsed().as_secs_f64();

        // ── Join the prefetch → next layer's non-expert is ready in its slot ──
        if let Some(bytes) = prefetch_bytes.take() {
            let t_join = Instant::now();
            gen.prefetch_worker
                .as_ref()
                .ok_or_else(|| StreamInferError::Msg("prefetch worker missing".into()))?
                .wait()?;
            gen.overlap_secs += t_join.elapsed().as_secs_f64();
            gen.prefetch_hits += 1;
            gen.io_bytes += bytes as u64;
            prefetched_ready = true;
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
/// Q/K RMSNorm: flat over the whole projection when the weight matches `x`
/// (OLMoE `attn_q_norm` has dim `n_embd`), else per-head (Gemma).
fn apply_qk_norm(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    if weight.len() == x.len() && weight.len() != head_dim {
        rms_norm(x, weight, eps);
    } else {
        crate::gemma_infer::apply_head_rmsnorm(x, weight, n_heads, head_dim);
    }
}

/// MLA (non-absorbed) attention for one layer. `attn_q` (lite) or `q_a`→`q_b`,
/// `attn_kv_a_mqa` → latent, `attn_kv_b` → per-head `[k_nope | v]`, RoPE on the
/// trailing `qk_rope` dims, against the compressed K/V cache. DeepSeek-V2-Lite.
fn mla_attention(
    gen: &mut StreamingGenerator,
    layer: usize,
    base: &[u8],
    loaded: &[(TensorRef, usize)],
    pos: usize,
    x: &mut [f32],
) -> Result<(), StreamInferError> {
    let m = gen
        .mla
        .ok_or_else(|| StreamInferError::Msg("MLA layer without MlaMeta".into()))?;
    let h = gen.config.hidden_size;
    let eps = gen.config.rms_norm_eps;
    let theta = gen.attn_cfg.rope_theta;
    let qk_head = m.qk_head();

    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);

    // Q: lite models use `attn_q`; full MLA uses `attn_q_a` → norm → `attn_q_b`.
    let mut q = vec![0.0f32; m.n_heads * qk_head];
    if let Ok(wq) = view_of(base, loaded, LayerOpKind::AttnQ) {
        wq.gemv(&xn, &mut q)?;
    } else {
        let wq_a = view_of(base, loaded, LayerOpKind::MlaQa)?;
        let mut q_a = vec![0.0f32; wq_a.nrows];
        wq_a.gemv(&xn, &mut q_a)?;
        if let Some(qn) = gen.mla_q_a_norm.as_ref().and_then(|v| v[layer].as_ref()) {
            rms_norm(&mut q_a, qn, eps);
        }
        let wq_b = view_of(base, loaded, LayerOpKind::MlaQb)?;
        q.resize(wq_b.nrows, 0.0);
        wq_b.gemv(&q_a, &mut q)?;
    }
    for hd in 0..m.n_heads {
        let s = hd * qk_head + m.qk_nope;
        apply_rope_partial_factors_scaled(
            &mut q[s..s + m.qk_rope],
            pos,
            m.qk_rope,
            m.qk_rope,
            theta,
            None,
            gen.attn_cfg.rope,
        );
    }

    // KV latent: `[kv_lora_rank | k_pe]`.
    let wkv_a = view_of(base, loaded, LayerOpKind::MlaKvA)?;
    let mut kv_pe = vec![0.0f32; m.kv_lora_rank + m.qk_rope];
    wkv_a.gemv(&xn, &mut kv_pe)?;
    let (c_kv, k_pe) = kv_pe.split_at_mut(m.kv_lora_rank);
    apply_rope_partial_factors_scaled(
        k_pe,
        pos,
        m.qk_rope,
        m.qk_rope,
        theta,
        None,
        gen.attn_cfg.rope,
    );
    let k_pe_snapshot = k_pe.to_vec();
    if let Some(kn) = gen.mla_kv_a_norm.as_ref().and_then(|v| v[layer].as_ref()) {
        rms_norm(c_kv, kn, eps);
    }

    // DeepSeek pre-scales `kq_scale` so YaRN's `mscale²` is applied once here
    // (the RoPE itself scales q/k by `attn_factor_org`).
    let rope = gen.attn_cfg.rope;
    let inv = if rope.freq_scale > 0.0 {
        1.0 / rope.freq_scale
    } else {
        1.0
    };
    let attn_factor_org = rope.attn_factor * (1.0 + 0.1 * inv.ln());
    let mscale = if rope.ext_factor != 0.0 {
        attn_factor_org * (1.0 + 0.1 * rope.yarn_log_mul * inv.ln())
    } else {
        attn_factor_org
    };
    let scale = mscale * mscale / (qk_head as f32).sqrt();
    let mut attn = vec![0.0f32; m.n_heads * m.v_head_dim];
    // `k_nope`-equivalent shown by the dump (per-head decompressed, or the latent).
    let k_nope_dump: Vec<f32>;
    {
        let caches = gen
            .mla_kv
            .as_mut()
            .ok_or_else(|| StreamInferError::Msg("MLA without cache".into()))?;
        let cache = &mut caches[layer];
        if m.absorbed {
            // Absorbed path (`attn_k_b`/`attn_v_b`): one MQA cache head in latent
            // space; `q_nope` is absorbed per head via `W_kb`, output via `W_vb`.
            let mut k_full = Vec::with_capacity(m.kv_lora_rank + m.qk_rope);
            k_full.extend_from_slice(c_kv);
            k_full.extend_from_slice(&k_pe_snapshot);
            cache.k.heads[0].append(&k_full, &k_full);
            cache.v.heads[0].append(c_kv, c_kv);
            let slots = cache.k.heads[0].attention_slots();
            for hd in 0..m.n_heads {
                let qn = &q[hd * qk_head..hd * qk_head + m.qk_nope];
                let wkb = view_of_head(base, loaded, LayerOpKind::MlaKb, hd, m.n_heads)?;
                let mut qcur = vec![0.0f32; m.kv_lora_rank];
                wkb.gemv(qn, &mut qcur)?;
                qcur.extend_from_slice(&q[hd * qk_head + m.qk_nope..(hd + 1) * qk_head]);
                let mut scores: Vec<f32> = slots
                    .iter()
                    .map(|&s| cache.k.heads[0].score_key(s, &qcur) * scale)
                    .collect();
                softmax(&mut scores);
                let mut acc = vec![0.0f32; m.kv_lora_rank];
                for (i, &s) in slots.iter().enumerate() {
                    cache.v.heads[0].accumulate_value(s, scores[i], &mut acc);
                }
                let wvb = view_of_head(base, loaded, LayerOpKind::MlaVb, hd, m.n_heads)?;
                let out = &mut attn[hd * m.v_head_dim..(hd + 1) * m.v_head_dim];
                wvb.gemv(&acc, out)?;
            }
            k_nope_dump = c_kv.to_vec();
        } else {
            // Fused decompression: `attn_kv_b` → per head `[k_nope | v]`.
            let wkv_b = view_of(base, loaded, LayerOpKind::MlaKvB)?;
            let ph = m.qk_nope + m.v_head_dim;
            let mut kv = vec![0.0f32; m.n_heads * ph];
            wkv_b.gemv(c_kv, &mut kv)?;
            k_nope_dump = kv[..3.min(kv.len())].to_vec();
            for hd in 0..m.n_heads {
                let mut k = vec![0.0f32; qk_head];
                k[..m.qk_nope].copy_from_slice(&kv[hd * ph..hd * ph + m.qk_nope]);
                k[m.qk_nope..].copy_from_slice(&k_pe_snapshot);
                let v = kv[hd * ph + m.qk_nope..hd * ph + ph].to_vec();
                // K cache stores the full `[k_nope | k_pe]`; its value slot is unused.
                cache.k.heads[hd].append(&k, &k);
                cache.v.heads[hd].append(&v, &v);
                let qh = &q[hd * qk_head..(hd + 1) * qk_head];
                let slots = cache.k.heads[hd].attention_slots();
                let mut scores: Vec<f32> = slots
                    .iter()
                    .map(|&s| cache.k.heads[hd].score_key(s, qh) * scale)
                    .collect();
                softmax(&mut scores);
                let mut acc = vec![0.0f32; m.v_head_dim];
                for (i, &s) in slots.iter().enumerate() {
                    cache.v.heads[hd].accumulate_value(s, scores[i], &mut acc);
                }
                attn[hd * m.v_head_dim..(hd + 1) * m.v_head_dim].copy_from_slice(&acc);
            }
        }
    }

    let wo = view_of(base, loaded, LayerOpKind::AttnO)?;
    let mut proj = vec![0.0f32; h];
    wo.gemv(&attn, &mut proj)?;
    for i in 0..h {
        x[i] += proj[i];
    }
    if std::env::var("HAYAI_MLA_DUMP").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!(
            "MLA_DUMP layer=0 pos={} scale={:.5} c_kv[0..3]={:?} k_pe[0..3]={:?} q[0..3]={:?} q_pe[0..3]={:?} k_nope[0..3]={:?} attn[0..3]={:?}",
            pos,
            scale,
            &c_kv[..3.min(c_kv.len())],
            &k_pe_snapshot[..3.min(k_pe_snapshot.len())],
            &q[..3],
            &q[m.qk_nope..m.qk_nope + 3],
            &k_nope_dump[..3.min(k_nope_dump.len())],
            &attn[..3]
        );
    }
    Ok(())
}

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

/// View over one head of a packed 3D per-head tensor (`attn_k_b`/`attn_v_b`:
/// `[qk_nope, kv_lora, n_head]` / `[kv_lora, v_head_dim, n_head]`).
fn view_of_head(
    base: &[u8],
    loaded: &[(TensorRef, usize)],
    op: LayerOpKind,
    head: usize,
    n_head: usize,
) -> Result<QuantMatrix, StreamInferError> {
    let (t, off) = loaded
        .iter()
        .find(|(t, _)| t.op == op)
        .ok_or_else(|| StreamInferError::Msg(format!("MLA block missing {op:?} tensor")))?;
    let head_bytes = t.nbytes / n_head.max(1);
    let start = *off + head * head_bytes;
    Ok(QuantMatrix::view(
        t.name.clone(),
        t.ncols,
        t.nrows,
        t.ggml_type,
        &base[start..start + head_bytes],
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

/// MoE routing (llama.cpp `build_moe_ffn`): grouped top-k selection with an optional
/// `e_score_correction_bias` (applied to the *choice* scores only), softmax or
/// sigmoid gating, optional top-k normalization and a routed scaling factor.
/// Returns `(expert_ids, weights)`.
pub(crate) fn route_experts(
    scores: &[f32],
    correction_bias: Option<&[f32]>,
    meta: &MoeMeta,
) -> (Vec<usize>, Vec<f32>) {
    let n = scores.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // `e_score_correction_bias` shifts the selection scores but not the weights.
    let choice: Vec<f32> = match correction_bias {
        Some(b) => scores.iter().zip(b.iter()).map(|(s, b)| s + b).collect(),
        None => scores.to_vec(),
    };
    let grouped = meta.n_group > 0 && meta.topk_group > 0 && n % meta.n_group == 0;
    let top: Vec<usize> = if grouped {
        // Group score = sum of the top-2 experts in the group; keep `topk_group`.
        let gs = n / meta.n_group;
        let mut group_scores: Vec<(usize, f32)> = (0..meta.n_group)
            .map(|g| {
                let mut vals: Vec<f32> = (0..gs).map(|j| choice[g * gs + j]).collect();
                vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                (g, vals.iter().take(2).sum())
            })
            .collect();
        group_scores
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cand: Vec<usize> = Vec::with_capacity(meta.topk_group * gs);
        for (g, _) in group_scores.into_iter().take(meta.topk_group) {
            cand.extend((0..gs).map(|j| g * gs + j));
        }
        let k = meta.top_k.max(1).min(cand.len().max(1));
        cand.sort_by(|&a, &b| choice[b].partial_cmp(&choice[a]).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(k);
        cand
    } else {
        top_k_indices(&choice, meta.top_k)
    };
    // Weights: sigmoid per expert, or softmax over **all** experts then take the
    // selected values (llama.cpp `build_moe_ffn`). `norm_topk_prob` renormalizes.
    let mut weights: Vec<f32> = if meta.gating == 1 {
        // Sigmoid gating (DeepSeek V3 / GLM): raw scores through the sigmoid.
        top.iter()
            .map(|&e| 1.0 / (1.0 + (-scores[e]).exp()))
            .collect()
    } else {
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        top.iter().map(|&e| exps[e] * inv).collect()
    };
    if meta.norm_topk_prob {
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
    }
    if (meta.routed_scaling_factor - 1.0).abs() > 1e-9 {
        for w in weights.iter_mut() {
            *w *= meta.routed_scaling_factor;
        }
    }
    (top, weights)
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
    fn absorbed_wkb_gemv_is_transpose_product() {
        // Absorbed MLA computes `q_absorbed = W_kb^T @ q_nope`. `W_kb` is stored
        // `[qk_nope, kv_lora]` (ne0 = qk_nope contiguous), and our row-major GEMV
        // over `ncols=qk_nope, nrows=kv_lora` yields exactly `sum_nope W[nope, l]*q`.
        use hayai_model::GgmlType;
        let qk_nope = 3usize;
        let kv_lora = 2usize;
        let mut data = Vec::new();
        for latent in 0..kv_lora {
            for nope in 0..qk_nope {
                data.extend_from_slice(&((nope as f32) + 10.0 * latent as f32).to_le_bytes());
            }
        }
        let w = QuantMatrix::view("wk_b", qk_nope, kv_lora, GgmlType::F32, &data);
        let q = [1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; kv_lora];
        w.gemv(&q, &mut out).unwrap();
        assert!((out[0] - (0.0 * 1.0 + 1.0 * 2.0 + 2.0 * 3.0)).abs() < 1e-6);
        assert!((out[1] - (10.0 * 1.0 + 11.0 * 2.0 + 12.0 * 3.0)).abs() < 1e-6);
    }

    #[test]
    fn route_experts_softmax_topk_uses_full_softmax() {
        let meta = MoeMeta {
            expert_count: 5,
            top_k: 2,
            ..Default::default()
        };
        let scores = [0.1f32, 0.9, 0.3, 0.7, 0.2];
        let (top, w) = route_experts(&scores, None, &meta);
        assert_eq!(top, vec![1, 3]);
        // Softmax over all 5 experts, then take the top-2 values (no renormalization).
        let max = 0.9f32;
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        assert!((w[0] - exps[1] / sum).abs() < 1e-5);
        assert!((w[1] - exps[3] / sum).abs() < 1e-5);
        assert!(w.iter().sum::<f32>() < 1.0);
    }

    #[test]
    fn route_experts_norm_topk_prob_renormalizes() {
        let meta = MoeMeta {
            expert_count: 5,
            top_k: 2,
            norm_topk_prob: true,
            ..Default::default()
        };
        let scores = [0.1f32, 0.9, 0.3, 0.7, 0.2];
        let (_, w) = route_experts(&scores, None, &meta);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn route_experts_correction_bias_selects_but_weights_raw_scores() {
        let meta = MoeMeta {
            expert_count: 4,
            top_k: 1,
            ..Default::default()
        };
        let scores = [1.0f32, 0.5, 0.0, -1.0];
        let bias = [0.0f32, 10.0, 0.0, 0.0];
        let (top, w) = route_experts(&scores, Some(&bias), &meta);
        assert_eq!(top, vec![1]); // bias made expert 1 win the selection
        // Weight is the full-softmax value of the raw score of expert 1.
        let exps: Vec<f32> = scores.iter().map(|s| s.exp()).collect();
        let sum: f32 = exps.iter().sum();
        assert!((w[0] - exps[1] / sum).abs() < 1e-5);
    }

    #[test]
    fn route_experts_grouped_selects_within_kept_groups() {
        // 4 experts, 2 groups of 2, keep 1 group, pick 1 expert.
        let meta = MoeMeta {
            expert_count: 4,
            top_k: 1,
            n_group: 2,
            topk_group: 1,
            gating: 1,
            ..Default::default()
        };
        let scores = [0.6f32, 0.5, 0.9, 0.1];
        // group0 top-2 sum = 1.1 > group1 = 1.0 → group0 wins → expert 0.
        let (top, _) = route_experts(&scores, None, &meta);
        assert_eq!(top, vec![0]);
    }

    #[test]
    fn route_experts_sigmoid_and_scaling() {
        let meta = MoeMeta {
            expert_count: 3,
            top_k: 2,
            gating: 1, // sigmoid
            routed_scaling_factor: 2.0,
            ..Default::default()
        };
        let scores = [0.0f32, 1.0, -1.0];
        let (top, w) = route_experts(&scores, None, &meta);
        assert_eq!(top, vec![1, 0]);
        assert!((w[0] - 2.0 * 0.731_058_6).abs() < 1e-4); // sigmoid(1)*2
        assert!((w[1] - 1.0).abs() < 1e-4); // sigmoid(0)*2
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


