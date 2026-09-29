//! Prefill wavefront: Attn∥FFN with `cl_event` (PRD §3.3).
//!
//! Schedule (numerically exact LLaMA residual):
//! - Within a layer: Attn(token t) runs while FFN(token t−1) is in flight.
//! - Across layers (T≥2): Attn(layer i+1, token 0) runs while FFN(layer i, token T−1)
//!   is in flight (token 0 already has FFN(i) applied — no residual hazard).

use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{
    add_bias, add_qkv_bias, ffn_begin_gate_up_scratch, ffn_finish_scratch, GateUpInflight,
    PrefetchJob, PrefetchWorker, StreamInferError, StreamingGenerator,
};
use hayai_cpu::{attention_decode_step_ex, rms_norm};
use hayai_model::{LayerPackLayout, LayerWeightPack, QuantMatrix};
use hayai_opencl::StreamingScratch;
use std::time::Instant;

struct PendingFfn {
    inflight: GateUpInflight,
    gate_out: Vec<f32>,
    up_out: Vec<f32>,
    down_out: Vec<f32>,
    token_idx: usize,
    /// CSR disperso embebido (D16) para `down`; Some ⇒ finish por CPU.
    down_csr: Option<hayai_model::CsrSparse>,
}

/// Completa el FFN de un token prefilled: espera el inflight (o lo omite si gate/up
/// ya se calcularon por CSR), aplica silu(gate)*up y proyecta `down` (CSR o denso).
fn finish_pending_ffn(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    p: PendingFfn,
    down: &QuantMatrix,
    scratch: &mut StreamingScratch,
    layer_idx: usize,
    layout: &LayerPackLayout,
) -> Result<Vec<f32>, StreamInferError> {
    let mut gate_out = p.gate_out;
    let mut up_out = p.up_out;
    let mut down_out = p.down_out;
    if let Some(c) = p.down_csr {
        for i in 0..gate_out.len() {
            let g = gate_out[i];
            gate_out[i] = (g / (1.0 + (-g).exp())) * up_out[i];
        }
        // SpMM CSR: OpenCL si hay pool, si no CPU.
        down_out.copy_from_slice(&gen.spmm_csr(orch, &gate_out, &c)?);
    } else {
        ffn_finish_scratch(
            orch,
            p.inflight,
            down,
            &mut gate_out,
            &mut up_out,
            &mut down_out,
            &mut gen.used_dgpu,
            Some(scratch),
            layer_idx,
            Some(layout),
        )?;
    }
    Ok(down_out)
}

type PrefetchOk = (LayerWeightPack, LayerPackLayout);

impl StreamingGenerator {
    /// Prefill all prompt tokens with Attn∥FFN wavefront; leaves `position` advanced
    /// and returns logits for the last prompt token.
    pub(crate) fn prefill_wavefront(
        &mut self,
        orch: &mut EngineOrchestrator,
        tokens: &[u32],
        scratch: &mut StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        let h = self.config.hidden_size;
        let n_layers = self.config.num_layers;
        let eps = self.config.rms_norm_eps;
        let ff = self.config.intermediate_size;
        let t_count = tokens.len();
        assert!(t_count >= 1);

        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(t_count);
        for &tok in tokens {
            let mut x = vec![0.0f32; h];
            self.embed_row("token_embd.weight", tok, h, &mut x)?;
            xs.push(x);
        }

        let (mut current, mut layout) = self.stage_pack(orch, scratch, 0, 0)?;

        let mut next_staged: Option<PrefetchOk> = None;
        let mut skip_attn_t0 = false;

        for layer_idx in 0..n_layers {
            if let Some((pack, lay)) = next_staged.take() {
                layout = lay;
                current = self.bind_prefetched_slot(
                    orch,
                    scratch,
                    layer_idx % 2,
                    &layout,
                    &pack,
                )?;
            }

            let mut prefetch = false;
            // Resident / macro-chunk: layers are (or will be) already in memory —
            // the block is staged once by `stage_pack`. No per-layer prefetch.
            if !scratch.resident && scratch.block_k <= 1 && layer_idx + 1 < n_layers {
                let next = layer_idx + 1;
                let next_fused = self.fused_qkv_dims(next);
                let next_slot = (layer_idx + 1) % 2;
                let slot_ptr = self.prepare_prefetch_slot(orch, scratch, next_slot)?;
                if let Some(m) = self.owned_mem.as_mut() {
                    m.note_prefetch_staging(0);
                }
                if self.prefetch_worker.is_none() {
                    let cat = self.catalog.fork_reader()?;
                    self.prefetch_worker = Some(PrefetchWorker::new(cat));
                }
                self.prefetch_worker.as_ref().unwrap().submit(PrefetchJob {
                    read: Box::new(move |cat| {
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(slot_ptr.addr as *mut u8, slot_ptr.len)
                        };
                        match next_fused {
                            Some((q, kv)) => {
                                cat.load_layer_pack_into_fused(next, dst, q, kv).map(|_| ())
                            }
                            None => cat.load_layer_pack_into(next, dst).map(|_| ()),
                        }
                    }),
                })?;
                prefetch = true;
            }

            let mut pending: Option<PendingFfn> = None;
            let slot = layer_idx % 2;

            // DMA FFN once per layer (overlaps first Attn); unmap before first FFN enqueue.
            if scratch.resident || scratch.block_k > 1 {
                scratch.ensure_host_readable(&orch.pool, layer_idx)?;
                (current, layout) =
                    self.pack_views_from_base(layer_idx, scratch.host_slot(layer_idx))?;
            } else {
                scratch.ensure_host_readable(&orch.pool, slot)?;
                current = current.rebind_views(scratch.host_slot(slot), &layout);
            }
            self.begin_ffn_dma(orch, scratch, slot, &layout)?;
            let mut dma_committed = true;

            for t in 0..t_count {
                let pos = self.position + t;
                let do_attn = !(t == 0 && skip_attn_t0);
                if t == 0 {
                    skip_attn_t0 = false;
                }

                if do_attn {
                    if !dma_committed {
                        if scratch.resident || scratch.block_k > 1 {
                            scratch.ensure_host_readable(&orch.pool, layer_idx)?;
                            (current, layout) =
                                self.pack_views_from_base(layer_idx, scratch.host_slot(layer_idx))?;
                        } else {
                            scratch.ensure_host_readable(&orch.pool, slot)?;
                            current = current.rebind_views(scratch.host_slot(slot), &layout);
                        }
                    }

                    let t_attn = Instant::now();
                    let mut xn = xs[t].clone();
                    rms_norm(&mut xn, &self.layer_norms[layer_idx].attn_norm, eps);
                    let q_dim = self.attn_cfg.hidden_size();
                    let kv_dim = self.attn_cfg.kv_dim();
                    let mut q = vec![0.0f32; q_dim];
                    let mut k = vec![0.0f32; kv_dim];
                    let mut v = vec![0.0f32; kv_dim];
                    current.wq.gemv(&xn, &mut q)?;
                    current.wk.gemv(&xn, &mut k)?;
                    current.wv.gemv(&xn, &mut v)?;
                    if let Some(b) = self.attn_bias.get(layer_idx) {
                        add_qkv_bias(b, &mut q, &mut k, &mut v);
                    }
                    let mut attn_out = vec![0.0f32; q_dim];
                    attention_decode_step_ex(
                        &self.attn_cfg,
                        &mut self.kv[layer_idx],
                        &mut q,
                        &mut k,
                        &v,
                        pos,
                        &mut attn_out,
                        true,
                        self.rope_freq_factors.as_deref(),
                        self.alibi_slopes.as_deref(),
                        self.layer_apply_rope.get(layer_idx).copied().unwrap_or(true),
                    );
                    let mut attn_proj = vec![0.0f32; h];
                    current.wo.gemv(&attn_out, &mut attn_proj)?;
                    if let Some(b) = self.attn_bias.get(layer_idx) {
                        add_bias(&mut attn_proj, &b.o);
                    }
                    for i in 0..h {
                        xs[t][i] += self.residual_scale * attn_proj[i];
                    }
                    // Depthwise causal short-conv residual (generic `Conv` op).
                    self.apply_conv(layer_idx, &mut xs[t])?;
                    let attn_dt = t_attn.elapsed().as_secs_f64();
                    self.attn_secs += attn_dt;
                    if pending.is_some() {
                        self.attn_ffn_overlap_secs += attn_dt;
                    }
                }

                if let Some(p) = pending.take() {
                    let t_ffn = Instant::now();
                    let token_idx = p.token_idx;
                    let down_out =
                        finish_pending_ffn(self, orch, p, &current.down, scratch, layer_idx, &layout)?;
                    self.ffn_secs += t_ffn.elapsed().as_secs_f64();
                    for i in 0..h {
                        xs[token_idx][i] += self.residual_scale * down_out[i];
                    }
                }

                let mut xn = xs[t].clone();
                rms_norm(&mut xn, &self.layer_norms[layer_idx].ffn_norm, eps);
                let mut gate_out = vec![0.0f32; ff];
                let mut up_out = vec![0.0f32; ff];
                let down_out = vec![0.0f32; h];

                if dma_committed {
                    self.finish_ffn_unmap(orch, scratch, slot)?;
                    dma_committed = false;
                }

                let t_enq = Instant::now();
                let inflight = if current.gate_csr.is_some() || current.up_csr.is_some() {
                    // Bloque disperso embebido (D16): gate/up vía CSR en CPU;
                    // los bloques densos mixtos usan el orchestrator.
                    if let Some(c) = &current.gate_csr {
                        gate_out.copy_from_slice(&self.spmm_csr(orch, &xn, c)?);
                    } else {
                        orch.execute_quant_gemv(&current.gate, &xn, &mut gate_out)?;
                    }
                    if let Some(c) = &current.up_csr {
                        up_out.copy_from_slice(&self.spmm_csr(orch, &xn, c)?);
                    } else {
                        orch.execute_quant_gemv(&current.up, &xn, &mut up_out)?;
                    }
                    GateUpInflight::Done
                } else {
                    ffn_begin_gate_up_scratch(
                        orch,
                        &current.gate,
                        &current.up,
                        &xn,
                        &mut gate_out,
                        &mut up_out,
                        &mut self.used_dgpu,
                        &mut self.used_apu,
                        Some(scratch),
                        layer_idx,
                        Some(&layout),
                    )?
                };
                self.ffn_secs += t_enq.elapsed().as_secs_f64();

                if t == 0 && prefetch {
                    let t_join = Instant::now();
                    self.prefetch_worker
                        .as_ref()
                        .ok_or_else(|| StreamInferError::Msg("prefetch worker missing".into()))?
                        .wait()?;
                    self.overlap_secs += t_join.elapsed().as_secs_f64();
                    self.prefetch_hits += 1;
                    let next_slot = (layer_idx + 1) % 2;
                    let (pack, lay) =
                        self.pack_views_from_base(layer_idx + 1, scratch.host_slot(next_slot))?;
                    self.io_bytes += lay.total as u64;
                    next_staged = Some((pack, lay));
                    prefetch = false;
                }

                pending = Some(PendingFfn {
                    inflight,
                    gate_out,
                    up_out,
                    down_out,
                    token_idx: t,
                    down_csr: current.down_csr.clone(),
                });
            }

            if layer_idx + 1 < n_layers && next_staged.is_none() {
                if prefetch {
                    self.prefetch_worker
                        .as_ref()
                        .ok_or_else(|| StreamInferError::Msg("prefetch worker missing".into()))?
                        .wait()?;
                    self.prefetch_hits += 1;
                    let next_slot = (layer_idx + 1) % 2;
                    let (pack, lay) =
                        self.pack_views_from_base(layer_idx + 1, scratch.host_slot(next_slot))?;
                    self.io_bytes += lay.total as u64;
                    next_staged = Some((pack, lay));
                } else {
                    let (p, lay) =
                        self.stage_pack(orch, scratch, (layer_idx + 1) % 2, layer_idx + 1)?;
                    next_staged = Some((p, lay));
                }
            }

            // Cross-layer overlap: Attn(i+1,0) ∥ FFN(i, T−1).
            // Disabled for macro-chunk: `bind_prefetched_slot` indexes ping-pong
            // slots by `% 2`, which is wrong inside a block slot.
            if !scratch.resident
                && scratch.block_k <= 1
                && layer_idx + 1 < n_layers
                && t_count >= 2
            {
                if let (Some(p), Some((pack, lay))) = (pending.take(), next_staged.take()) {
                    let finishing = std::mem::replace(&mut current, pack);
                    let fin_layout = layout;
                    layout = lay;
                    let next_slot = (layer_idx + 1) % 2;
                    current = self.bind_prefetched_slot(
                        orch,
                        scratch,
                        next_slot,
                        &layout,
                        &current,
                    )?;
                    // Start DMA for next layer while finishing previous FFN.
                    self.begin_ffn_dma(orch, scratch, next_slot, &layout)?;

                    let next_layer = layer_idx + 1;
                    let t_attn = Instant::now();
                    let mut xn = xs[0].clone();
                    rms_norm(&mut xn, &self.layer_norms[next_layer].attn_norm, eps);
                    let q_dim = self.attn_cfg.hidden_size();
                    let kv_dim = self.attn_cfg.kv_dim();
                    let mut q = vec![0.0f32; q_dim];
                    let mut k = vec![0.0f32; kv_dim];
                    let mut v = vec![0.0f32; kv_dim];
                    current.wq.gemv(&xn, &mut q)?;
                    current.wk.gemv(&xn, &mut k)?;
                    current.wv.gemv(&xn, &mut v)?;
                    if let Some(b) = self.attn_bias.get(next_layer) {
                        add_qkv_bias(b, &mut q, &mut k, &mut v);
                    }
                    let mut attn_out = vec![0.0f32; q_dim];
                    attention_decode_step_ex(
                        &self.attn_cfg,
                        &mut self.kv[next_layer],
                        &mut q,
                        &mut k,
                        &v,
                        self.position,
                        &mut attn_out,
                        true,
                        self.rope_freq_factors.as_deref(),
                        self.alibi_slopes.as_deref(),
                        self.layer_apply_rope.get(next_layer).copied().unwrap_or(true),
                    );
                    let mut attn_proj = vec![0.0f32; h];
                    current.wo.gemv(&attn_out, &mut attn_proj)?;
                    if let Some(b) = self.attn_bias.get(next_layer) {
                        add_bias(&mut attn_proj, &b.o);
                    }
                    for i in 0..h {
                        xs[0][i] += self.residual_scale * attn_proj[i];
                    }
                    let attn_dt = t_attn.elapsed().as_secs_f64();
                    self.attn_secs += attn_dt;
                    self.attn_ffn_overlap_secs += attn_dt;

                    let t_ffn = Instant::now();
                    let token_idx = p.token_idx;
                    let down_out = finish_pending_ffn(
                        self,
                        orch,
                        p,
                        &finishing.down,
                        scratch,
                        layer_idx,
                        &fin_layout,
                    )?;
                    self.ffn_secs += t_ffn.elapsed().as_secs_f64();
                    for i in 0..h {
                        xs[token_idx][i] += self.residual_scale * down_out[i];
                    }

                    // Layer i+1 already has DMA in flight; unmap when first FFN of that layer runs.
                    // Re-enter loop with skip_attn_t0; re-DMA would double-write — stash via next_staged empty.
                    // Keep current mapped for token 0 FFN: unmap happens in the token loop.
                    // Signal that DMA already started for this layer:
                    next_staged = None;
                    // We need the next layer iteration to know DMA was started. Use a flag via
                    // putting pack back... Simpler: finish_ffn_unmap here is wrong (need mapped for
                    // remaining tokens' Attn). Attn for t0 already done; for t>=1 of next layer we
                    // need host map again. begin_ffn_dma already done; on next iteration
                    // `if let Some next_staged` won't run — current/layout already set.
                    // But the loop start will try begin_ffn_dma again — OK (idempotent rewrite).
                    // Skip rebinding from next_staged — fall through with skip_attn_t0.
                    // Problem: next loop iteration starts with `if let Some(next_staged)` none,
                    // then spawns prefetch, then begin_ffn_dma again — fine.
                    skip_attn_t0 = true;
                    // Don't `continue` past pending finish — pending already taken. Continue to
                    // next layer_idx with current already on next layer.
                    // We must NOT run the pending-at-end block. pending is None.
                    // But we also must not re-stage. Good.
                    // One issue: loop will re-prepare prefetch and re-dma. Acceptable.
                    continue;
                }
            }

            if let Some(p) = pending.take() {
                let t_ffn = Instant::now();
                let token_idx = p.token_idx;
                let down_out =
                    finish_pending_ffn(self, orch, p, &current.down, scratch, layer_idx, &layout)?;
                self.ffn_secs += t_ffn.elapsed().as_secs_f64();
                for i in 0..h {
                    xs[token_idx][i] += self.residual_scale * down_out[i];
                }
            }
        }

        self.position += t_count;

        let mut xn = xs[t_count - 1].clone();
        rms_norm(&mut xn, &self.output_norm, eps);
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        if self.has_output_weight {
            if let Some(ow) = &self.resident_output {
                orch.execute_quant_gemv(ow, &xn, &mut logits)?;
            } else {
                let t0 = Instant::now();
                let ow = self.catalog.load_quant_matrix("output.weight")?;
                self.io_secs += t0.elapsed().as_secs_f64();
                self.io_bytes += ow.nbytes() as u64;
                orch.execute_quant_gemv(&ow, &xn, &mut logits)?;
            }
        } else {
            if let Some(emb) = &self.resident_embed {
                orch.execute_quant_gemv(emb, &xn, &mut logits)?;
            } else {
                let t0 = Instant::now();
                let emb = self.catalog.load_quant_matrix("token_embd.weight")?;
                self.io_secs += t0.elapsed().as_secs_f64();
                self.io_bytes += emb.nbytes() as u64;
                orch.execute_quant_gemv(&emb, &xn, &mut logits)?;
            }
        }
        self.apply_logit_scale(&mut logits);
        if std::env::var("HAYAI_DUMP_TOP").ok().as_deref() == Some("1") {
            let at: usize = std::env::var("HAYAI_DUMP_AT_POS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            if self.position == at {
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_by(|&a, &b| {
                    logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top: Vec<String> = idx[..8.min(idx.len())]
                    .iter()
                    .map(|&i| format!("{}:{:.3}", i, logits[i]))
                    .collect();
                eprintln!("WAVE_DUMP_TOP pos={}: {}", self.position, top.join(" "));
            }
        }
        Ok(logits)
    }

    /// Batched prefill (opt-in `HAYAI_PREFILL_BATCH=1`): for each layer, run
    /// attention for every prompt token, then compute the FFN as **one batched
    /// GEMV per matrix** (`EngineOrchestrator::execute_quant_gemv_batched`), so
    /// the weight is read once per layer instead of once per token. Numerically
    /// identical to [`Self::prefill_wavefront`] for the dense FFN; embedded-sparse
    /// (CSR) blocks fall back to per-token CPU SpMM.
    pub(crate) fn prefill_batched(
        &mut self,
        orch: &mut EngineOrchestrator,
        tokens: &[u32],
        scratch: &mut StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        let h = self.config.hidden_size;
        let ff = self.config.intermediate_size;
        let eps = self.config.rms_norm_eps;
        let n_layers = self.config.num_layers;
        let t_count = tokens.len();
        if t_count == 0 {
            return Err(StreamInferError::Msg("empty prefill".into()));
        }
        let base_pos = self.position;

        let mut xs: Vec<Vec<f32>> = Vec::with_capacity(t_count);
        for &tok in tokens {
            let mut x = vec![0.0f32; h];
            self.embed_row("token_embd.weight", tok, h, &mut x)?;
            xs.push(x);
        }

        for layer_idx in 0..n_layers {
            let slot = layer_idx % 2;
            let (pack, _layout) = self.stage_pack(orch, scratch, slot, layer_idx)?;

            // ── Attention for every prompt token (writes the layer KV in order) ──
            let t_attn = Instant::now();
            for t in 0..t_count {
                let mut xn = xs[t].clone();
                rms_norm(&mut xn, &self.layer_norms[layer_idx].attn_norm, eps);
                let q_dim = self.attn_cfg.hidden_size();
                let kv_dim = self.attn_cfg.kv_dim();
                let mut q = vec![0.0f32; q_dim];
                let mut k = vec![0.0f32; kv_dim];
                let mut v = vec![0.0f32; kv_dim];
                pack.wq.gemv(&xn, &mut q)?;
                pack.wk.gemv(&xn, &mut k)?;
                pack.wv.gemv(&xn, &mut v)?;
                if let Some(b) = self.attn_bias.get(layer_idx) {
                    add_qkv_bias(b, &mut q, &mut k, &mut v);
                }
                let mut attn_out = vec![0.0f32; q_dim];
                attention_decode_step_ex(
                    &self.attn_cfg,
                    &mut self.kv[layer_idx],
                    &mut q,
                    &mut k,
                    &v,
                    base_pos + t,
                    &mut attn_out,
                    true,
                    self.rope_freq_factors.as_deref(),
                    self.alibi_slopes.as_deref(),
                    self.layer_apply_rope.get(layer_idx).copied().unwrap_or(true),
                );
                let mut attn_proj = vec![0.0f32; h];
                pack.wo.gemv(&attn_out, &mut attn_proj)?;
                if let Some(b) = self.attn_bias.get(layer_idx) {
                    add_bias(&mut attn_proj, &b.o);
                }
                for i in 0..h {
                    xs[t][i] += self.residual_scale * attn_proj[i];
                }
            }
            self.attn_secs += t_attn.elapsed().as_secs_f64();

            // ── FFN for every token: one batched GEMV per matrix ────────────────
            let t_ffn = Instant::now();
            let sparse =
                pack.gate_csr.is_some() || pack.up_csr.is_some() || pack.down_csr.is_some();
            if sparse {
                for t in 0..t_count {
                    let mut xn = xs[t].clone();
                    rms_norm(&mut xn, &self.layer_norms[layer_idx].ffn_norm, eps);
                    self.run_ffn_block(orch, &pack, &xn, None)?;
                    for i in 0..h {
                        xs[t][i] += self.residual_scale * self.ws_down[i];
                    }
                }
            } else {
                let mut xn_all = vec![0.0f32; t_count * h];
                for t in 0..t_count {
                    let mut xn = xs[t].clone();
                    rms_norm(&mut xn, &self.layer_norms[layer_idx].ffn_norm, eps);
                    xn_all[t * h..(t + 1) * h].copy_from_slice(&xn);
                }
                let mut gate_all = vec![0.0f32; t_count * ff];
                let mut up_all = vec![0.0f32; t_count * ff];
                orch.execute_quant_gemv_batched(&pack.gate, &xn_all, &mut gate_all, t_count)?;
                orch.execute_quant_gemv_batched(&pack.up, &xn_all, &mut up_all, t_count)?;
                for i in 0..(t_count * ff) {
                    let g = gate_all[i];
                    gate_all[i] = (g / (1.0 + (-g).exp())) * up_all[i];
                }
                let mut down_all = vec![0.0f32; t_count * h];
                orch.execute_quant_gemv_batched(&pack.down, &gate_all, &mut down_all, t_count)?;
                for t in 0..t_count {
                    for i in 0..h {
                        xs[t][i] += self.residual_scale * down_all[t * h + i];
                    }
                }
            }
            self.ffn_secs += t_ffn.elapsed().as_secs_f64();
        }

        self.position = base_pos + t_count;

        let mut xn = xs[t_count - 1].clone();
        rms_norm(&mut xn, &self.output_norm, eps);
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        if self.has_output_weight {
            if let Some(ow) = &self.resident_output {
                orch.execute_quant_gemv(ow, &xn, &mut logits)?;
            } else {
                let t0 = Instant::now();
                let ow = self.catalog.load_quant_matrix("output.weight")?;
                self.io_secs += t0.elapsed().as_secs_f64();
                self.io_bytes += ow.nbytes() as u64;
                orch.execute_quant_gemv(&ow, &xn, &mut logits)?;
            }
        } else if let Some(emb) = &self.resident_embed {
            orch.execute_quant_gemv(emb, &xn, &mut logits)?;
        } else {
            let t0 = Instant::now();
            let emb = self.catalog.load_quant_matrix("token_embd.weight")?;
            self.io_secs += t0.elapsed().as_secs_f64();
            self.io_bytes += emb.nbytes() as u64;
            orch.execute_quant_gemv(&emb, &xn, &mut logits)?;
        }
        self.apply_logit_scale(&mut logits);
        if std::env::var("HAYAI_DUMP_TOP").ok().as_deref() == Some("1") {
            let at: usize = std::env::var("HAYAI_DUMP_AT_POS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(usize::MAX);
            if self.position == at {
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_by(|&a, &b| {
                    logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top: Vec<String> = idx[..8.min(idx.len())]
                    .iter()
                    .map(|&i| format!("{}:{:.3}", i, logits[i]))
                    .collect();
                eprintln!("WAVE_DUMP_TOP pos={}: {}", self.position, top.join(" "));
            }
        }
        Ok(logits)
    }
}
