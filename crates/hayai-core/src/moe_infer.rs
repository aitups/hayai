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

        // Stage non-expert tensors into the layer slot (resident: preloaded once).
        scratch.prepare_host_write(&orch.pool, layer)?;
        if !scratch.resident {
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

        // ── Sparse expert streaming: read only the selected experts (colibri) ──
        // Resident mode: all experts are preloaded, so nothing is read here.
        if !scratch.resident {
            let t_io = Instant::now();
            {
                let mut specs: Vec<(&str, usize, usize, usize)> =
                    Vec::with_capacity(top.len() * 3);
                for (i, eid) in top.iter().enumerate() {
                    let expert = unit
                        .experts
                        .iter()
                        .find(|e| e.expert_id == *eid)
                        .ok_or_else(|| {
                            StreamInferError::Msg(format!("top-k expert {eid} not in plan"))
                        })?;
                    let base_off = unit.non_expert_bytes + i * unit.max_expert_bytes;
                    for t in &expert.tensors {
                        specs.push((t.name.as_str(), t.src_off, base_off + t.offset, t.nbytes));
                    }
                }
                let dst = scratch.host_slot_mut(layer);
                gen.catalog.load_tensors_into(&specs, dst)?;
            }
            gen.io_secs += t_io.elapsed().as_secs_f64();
            gen.io_bytes += (top.len() * unit.max_expert_bytes) as u64;
        }

        // ── Per-expert FFN (GpuAsync binding → OpenCL pool) ─────────────────────
        let t_ffn = Instant::now();
        {
            let base = scratch.host_slot(layer);
            let mut acc = vec![0.0f32; h];
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
                let down = expert_view(&base, expert, base_off, LayerOpKind::ExpertDown)?;
                let ff = gate.nrows.max(up.nrows);
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
}

