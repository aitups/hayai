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
use hayai_cpu::{attention_decode_step, rms_norm, LayerKvCache};
use hayai_model::{GgmlType, QuantMatrix};
use hayai_opencl::StreamingScratch;
use std::sync::Arc;
use std::time::Instant;

/// FFN matrices of one hybrid layer (dense or embedded sparse CSR), preloaded by
/// the prefetch thread so a DeltaNet/FFN-only block does not stall on disk.
type FfnMats = (
    QuantMatrix,
    QuantMatrix,
    QuantMatrix,
    Option<hayai_model::CsrSparse>,
    Option<hayai_model::CsrSparse>,
    Option<hayai_model::CsrSparse>,
);

pub(crate) fn prefill_hybrid(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    tokens: &[u32],
    scratch: &mut StreamingScratch,
) -> Result<Vec<f32>, StreamInferError> {
    let mut all = prefill_hybrid_all(gen, orch, tokens, scratch)?;
    all.0.pop().ok_or_else(|| StreamInferError::Msg("empty prefill".into()))
}

/// Layer-major hybrid prefill returning **per-token** logits and post-norm hidden
/// states (`t_h_nextn`). Used by speculative decoding to verify a draft batch and
/// to prime the MTP head.
pub(crate) fn prefill_hybrid_all(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    tokens: &[u32],
    scratch: &mut StreamingScratch,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), StreamInferError> {
    let h = gen.config.hidden_size;
    let eps = gen.config.rms_norm_eps;
    let n_layers = gen.config.num_layers;
    let t_count = tokens.len();
    if t_count == 0 {
        return Err(StreamInferError::Msg("empty prefill".into()));
    }
    ensure_deltanet_cache(gen)?;

    let skip_dn = std::env::var("HAYAI_SKIP_DN_ATTN").ok().as_deref() == Some("1");
    let skip_fa = std::env::var("HAYAI_SKIP_FA_ATTN").ok().as_deref() == Some("1");
    let max_layers = std::env::var("HAYAI_MAX_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(n_layers)
        .min(n_layers);

    // Embed every prompt token once.
    let mut xs: Vec<Vec<f32>> = Vec::with_capacity(t_count);
    for &tok in tokens {
        let mut x = vec![0.0f32; h];
        gen.embed_row("token_embd.weight", tok, h, &mut x)?;
        xs.push(x);
    }
    let base_pos = gen.position;

    // Layer-major prefill: each layer's weights are read **once** for the whole
    // prompt (vs. once per token in the old token-major loop), and the token loop
    // stays in cache. Full-attn layers run the prompt through their per-layer KV
    // cache in order; DeltaNet layers advance their recurrent state per token.
    for layer in 0..max_layers {
        match hybrid_layer_kind(&gen.catalog, layer) {
            HybridLayerKind::NextN => {
                static WARN_NEXTN: std::sync::Once = std::sync::Once::new();
                WARN_NEXTN.call_once(|| {
                    tracing::warn!(
                        "blk.{layer}: next-n/MTP layer skipped (not implemented in the streaming path)"
                    );
                });
                continue;
            }
            HybridLayerKind::FullAttn if !skip_fa => {
                // Stage the layer pack into the ping-pong scratch once (host-mapped
                // views, FFN slices DMA'd to device mirrors) and reuse it for every
                // prompt token instead of host-uploading the FFN per token.
                let slot = layer % 2;
                let (pack, layout) = gen.stage_pack(orch, scratch, slot, layer)?;
                gen.begin_ffn_dma(orch, scratch, slot, &layout)?;
                for (t, x) in xs.iter_mut().enumerate() {
                    full_attn_apply(gen, orch, layer, base_pos + t, eps, &pack, x)?;
                    ffn_apply(
                        gen,
                        orch,
                        Some(scratch),
                        layer,
                        eps,
                        &pack,
                        Some(&layout),
                        x,
                        None,
                    )?;
                }
            }
            HybridLayerKind::DeltaNet if !skip_dn => {
                // DeltaNet attention weights are cached resident
                // (`ensure_deltanet_cache`); only the FFN is read, once.
                let t0 = Instant::now();
                let ffn = gen.catalog.load_ffn_matrices(layer)?;
                gen.io_secs += t0.elapsed().as_secs_f64();
                gen.io_bytes += (ffn.0.nbytes() + ffn.1.nbytes() + ffn.2.nbytes()) as u64;
                for x in xs.iter_mut() {
                    run_deltanet_block(gen, orch, scratch, layer, eps, x, None, Some(&ffn))?;
                }
            }
            _ => {
                // FFN-only (ablation: attention skipped on this layer kind).
                let t0 = Instant::now();
                let ffn = gen.catalog.load_ffn_matrices(layer)?;
                gen.io_secs += t0.elapsed().as_secs_f64();
                gen.io_bytes += (ffn.0.nbytes() + ffn.1.nbytes() + ffn.2.nbytes()) as u64;
                for x in xs.iter_mut() {
                    run_ffn_only_block(gen, orch, scratch, layer, eps, x, None, Some(&ffn))?;
                }
            }
        }
    }

    gen.position = base_pos + t_count;
    let mut all_logits = Vec::with_capacity(t_count);
    let mut all_hidden = Vec::with_capacity(t_count);
    for x in xs.iter_mut() {
        let lg = output_logits(gen, orch, x, eps)?;
        all_hidden.push(gen.last_hidden_nextn.clone());
        all_logits.push(lg);
    }
    Ok((all_logits, all_hidden))
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
    let mut pre: Option<(usize, std::thread::JoinHandle<Result<FfnMats, hayai_model::GgufError>>)> =
        None;
    for layer in 0..max_layers {
        let ov = override_ffn.and_then(|o| o.get(layer));
        // Holds the joined prefetch for this layer so `preloaded` can borrow it.
        let pre_holder: Option<FfnMats>;

        // Join the prefetch targeting this layer (started one/more layers ago, so
        // it overlapped the intermediate compute). A prefetch for a later layer is
        // put back untouched.
        let preloaded: Option<&FfnMats> = match pre.take() {
            Some((pl, handle)) if pl == layer => {
                let mats = handle
                    .join()
                    .map_err(|_| StreamInferError::Msg("ffn prefetch thread panicked".into()))??;
                gen.io_bytes += (mats.0.nbytes() + mats.1.nbytes() + mats.2.nbytes()) as u64;
                pre_holder = Some(mats);
                pre_holder.as_ref()
            }
            other => {
                pre = other;
                None
            }
        };

        // Prefetch the next layer that loads its FFN from disk (DeltaNet, or a
        // FullAttn layer whose attention is skipped). FullAttn layers stage the
        // whole pack through the ping-pong scratch and need no prefetch here.
        if pre.is_none() {
            let next = (layer + 1..max_layers)
                .find(|&l| needs_ffn_prefetch(&gen.catalog, l, skip_fa));
            if let Some(next) = next {
                let mut cat = gen.catalog.fork_reader()?;
                pre = Some((
                    next,
                    std::thread::spawn(move || cat.load_ffn_matrices(next)),
                ));
            }
        }

        match hybrid_layer_kind(&gen.catalog, layer) {
            HybridLayerKind::NextN => {
                // MTP / next-n draft head is not executed yet; report it once
                // instead of silently changing the graph.
                static WARN_NEXTN: std::sync::Once = std::sync::Once::new();
                WARN_NEXTN.call_once(|| {
                    tracing::warn!(
                        "blk.{layer}: next-n/MTP layer skipped (not implemented in the streaming path)"
                    );
                });
                continue;
            }
            HybridLayerKind::DeltaNet => {
                if skip_dn {
                    run_ffn_only_block(gen, orch, scratch, layer, eps, &mut x, ov, preloaded)?;
                } else {
                    run_deltanet_block(gen, orch, scratch, layer, eps, &mut x, ov, preloaded)?;
                }
            }
            HybridLayerKind::FullAttn => {
                if skip_fa {
                    run_ffn_only_block(gen, orch, scratch, layer, eps, &mut x, ov, preloaded)?;
                } else {
                    // FullAttn stages via scratch; a pending prefetch is for a
                    // later layer and was retained above.
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

    let logits = output_logits(gen, orch, &mut x, eps)?;
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

/// Final RMSNorm + LM head projection for one hidden state.
fn output_logits(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    x: &mut [f32],
    eps: f32,
) -> Result<Vec<f32>, StreamInferError> {
    rms_norm(x, &gen.output_norm, eps);
    // `t_h_nextn`: post-final-norm hidden consumed by the NextN/MTP draft head.
    gen.last_hidden_nextn.clear();
    gen.last_hidden_nextn.extend_from_slice(x);
    let vocab = gen.config.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    if gen.has_output_weight {
        if let Some(ow) = &gen.resident_output {
            orch.execute_quant_gemv(ow, x, &mut logits)?;
        } else {
            let t0 = Instant::now();
            let ow = gen.catalog.load_quant_matrix("output.weight")?;
            gen.io_secs += t0.elapsed().as_secs_f64();
            gen.io_bytes += ow.nbytes() as u64;
            orch.execute_quant_gemv(&ow, x, &mut logits)?;
        }
    } else if let Some(emb) = &gen.resident_embed {
        orch.execute_quant_gemv(emb, x, &mut logits)?;
    } else {
        let t0 = Instant::now();
        let emb = gen.catalog.load_quant_matrix("token_embd.weight")?;
        gen.io_secs += t0.elapsed().as_secs_f64();
        gen.io_bytes += emb.nbytes() as u64;
        orch.execute_quant_gemv(&emb, x, &mut logits)?;
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

pub(crate) fn ensure_deltanet_cache(gen: &mut StreamingGenerator) -> Result<(), StreamInferError> {
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

/// Whether a layer's FFN is loaded outside the full-pack scratch staging
/// (`run_deltanet_block` / `run_ffn_only_block`) and can therefore use the
/// prefetched FFN.
fn needs_ffn_prefetch(cat: &hayai_model::GgufCatalog, layer: usize, skip_fa: bool) -> bool {
    match hybrid_layer_kind(cat, layer) {
        HybridLayerKind::NextN => false,
        HybridLayerKind::DeltaNet => true,
        HybridLayerKind::FullAttn => skip_fa,
    }
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
    preloaded: Option<&FfnMats>,
) -> Result<(), StreamInferError> {
    let h = x.len();
    // Use the prefetched FFN when available (overlapped with the previous layer);
    // otherwise read it now into a function-local that stays alive for the call.
    let loaded: Option<FfnMats> = if preloaded.is_none() {
        let t_io = Instant::now();
        let m = gen.catalog.load_ffn_matrices(layer)?;
        gen.io_secs += t_io.elapsed().as_secs_f64();
        gen.io_bytes += (m.0.nbytes() + m.1.nbytes() + m.2.nbytes()) as u64;
        Some(m)
    } else {
        None
    };
    let m: &FfnMats = preloaded.or(loaded.as_ref()).expect("ffn present");

    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
    gen.ws_gate.fill(0.0);
    gen.ws_up.fill(0.0);
    gen.ws_down.fill(0.0);
    let t_ffn = Instant::now();
    let sparse = m.3.is_some() || m.4.is_some() || m.5.is_some();
    let has_ov = ov.map(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some()).unwrap_or(false);
    if sparse || has_ov {
        let pack = hayai_model::LayerWeightPack {
            wq: m.0.clone(),
            wk: m.0.clone(),
            wv: m.0.clone(),
            wo: m.0.clone(),
            gate: m.0.clone(),
            up: m.1.clone(),
            down: m.2.clone(),
            attn_gate: None,
            gate_csr: m.3.clone(),
            up_csr: m.4.clone(),
            down_csr: m.5.clone(),
        };
        gen.run_ffn_block(orch, &pack, &xn, ov)?;
    } else {
        let inflight = ffn_begin_gate_up_scratch(
            orch,
            &m.0,
            &m.1,
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
            &m.2,
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
    preloaded: Option<&FfnMats>,
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

    // FFN via WeightIo (owned matrices - DeltaNet layers have no attn_q pack).
    // Los bloques dispersos embebidos (D16) cargan con CSR.
    // Use the prefetched FFN when available (overlapped with the previous layer);
    // otherwise read it once into a function-local that outlives all uses.
    let loaded: Option<FfnMats> = if preloaded.is_none() {
        let t_io = Instant::now();
        let m = gen.catalog.load_ffn_matrices(layer)?;
        gen.io_secs += t_io.elapsed().as_secs_f64();
        gen.io_bytes += (m.0.nbytes() + m.1.nbytes() + m.2.nbytes()) as u64;
        Some(m)
    } else {
        None
    };
    let m: &FfnMats = preloaded.or(loaded.as_ref()).expect("ffn present");

    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
    if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!("PROD_XN L0: {:?}", &xn[0..8]);
    }

    gen.ws_gate.fill(0.0);
    gen.ws_up.fill(0.0);
    gen.ws_down.fill(0.0);
    let t_ffn = Instant::now();
    if m.3.is_some() || m.4.is_some() || m.5.is_some()
        || ov.map(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some()).unwrap_or(false)
    {
        // FFN disperso embebido (D16) u override de evolución: CSR.
        let pack = hayai_model::LayerWeightPack {
            wq: m.0.clone(),
            wk: m.0.clone(),
            wv: m.0.clone(),
            wo: m.0.clone(),
            gate: m.0.clone(),
            up: m.1.clone(),
            down: m.2.clone(),
            attn_gate: None,
            gate_csr: m.3.clone(),
            up_csr: m.4.clone(),
            down_csr: m.5.clone(),
        };
        gen.run_ffn_block(orch, &pack, &xn, ov)?;
    } else {
        let inflight = ffn_begin_gate_up_scratch(
            orch,
            &m.0,
            &m.1,
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
            &m.2,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.ws_down,
            &mut gen.used_dgpu,
            None,
            0,
            None,
        )?;
        if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
            eprintln!("PROD_GATE_ACT L0: {:?}", &gen.ws_gate[0..8]);
            eprintln!("PROD_DOWN L0: {:?}", &gen.ws_down[0..8]);
        }
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
    let slot = layer % 2;
    let (pack, layout) = gen.stage_pack(orch, scratch, slot, layer)?;
    gen.begin_ffn_dma(orch, scratch, slot, &layout)?;
    full_attn_apply(gen, orch, layer, pos, eps, &pack, x)?;
    gen.finish_ffn_unmap(orch, scratch, slot)?;
    ffn_apply(
        gen,
        orch,
        Some(scratch),
        layer,
        eps,
        &pack,
        Some(&layout),
        x,
        ov,
    )
}

/// Attention residual for one full-attention token: `x += Wo·attn(Q,K,V)`.
pub(crate) fn full_attn_apply(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    layer: usize,
    pos: usize,
    eps: f32,
    pack: &hayai_model::LayerWeightPack,
    x: &mut [f32],
) -> Result<(), StreamInferError> {
    use crate::exec_plan::{op_binding, LayerOpKind};
    let h = x.len();
    let t_attn = Instant::now();
    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);
    // Qwen3.5 full-attn: `attn_q` is fused Q∥gate interleaved per head
    // (llama.cpp view stride = 2 * head_dim), not [Q|Gate] halves.
    let wo_in = pack.wo.ncols;
    let mut q_full = vec![0.0f32; pack.wq.nrows];
    let mut k = vec![0.0f32; pack.wk.nrows];
    let mut v = vec![0.0f32; pack.wv.nrows];
    // Attention GEMVs go through the orchestrator so the planner places them where
    // they are fastest (they are not pinned to the CPU).
    orch.execute_op(LayerOpKind::AttnQ, op_binding(LayerOpKind::AttnQ), &pack.wq, &xn, &mut q_full)?;
    orch.execute_op(LayerOpKind::AttnK, op_binding(LayerOpKind::AttnK), &pack.wk, &xn, &mut k)?;
    orch.execute_op(LayerOpKind::AttnV, op_binding(LayerOpKind::AttnV), &pack.wv, &xn, &mut v)?;

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
    orch.execute_op(
        LayerOpKind::AttnO,
        op_binding(LayerOpKind::AttnO),
        &pack.wo,
        &attn_out,
        &mut attn_proj,
    )?;
    for i in 0..h {
        x[i] += attn_proj[i];
    }
    gen.attn_secs += t_attn.elapsed().as_secs_f64();
    // Optional depthwise causal short-conv residual (generic `Conv` op).
    gen.apply_conv(layer, x)?;
    Ok(())
}

/// FFN residual for one token. `scratch` selects the staged/DMA device path;
/// `None` runs the FFN from host-owned bytes (used by layer-major prefill).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ffn_apply(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    scratch: Option<&mut StreamingScratch>,
    layer: usize,
    eps: f32,
    pack: &hayai_model::LayerWeightPack,
    layout: Option<&hayai_model::LayerPackLayout>,
    x: &mut [f32],
    ov: Option<&FfnOverride>,
) -> Result<(), StreamInferError> {
    let h = x.len();
    let mut xn = x.to_vec();
    rms_norm(&mut xn, &gen.layer_norms[layer].ffn_norm, eps);
    gen.ws_gate.fill(0.0);
    gen.ws_up.fill(0.0);
    gen.ws_down.fill(0.0);
    let t_ffn = Instant::now();
    let sparse_ffn = pack.gate_csr.is_some() || pack.up_csr.is_some() || pack.down_csr.is_some();
    let has_ov = ov.map(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some()).unwrap_or(false);
    if sparse_ffn || has_ov {
        // FFN disperso embebido (D16) u override de evolución: CSR.
        gen.run_ffn_block(orch, pack, &xn, ov)?;
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
            scratch.as_deref(),
            layer,
            layout,
        )?;
        ffn_finish_scratch(
            orch,
            inflight,
            &pack.down,
            &mut gen.ws_gate,
            &mut gen.ws_up,
            &mut gen.ws_down,
            &mut gen.used_dgpu,
            scratch.as_deref(),
            layer,
            layout,
        )?;
    }
    gen.ffn_secs += t_ffn.elapsed().as_secs_f64();
    for i in 0..h {
        x[i] += gen.ws_down[i];
    }
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
            let v = x[base + i];
            ms += v * v;
        }
        let inv = 1.0 / (ms / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            x[base + i] *= weight[i] * inv;
        }
    }
}

/// Forward **batcheado** híbrido (Fase 2): procesa N candidatos por capa con
/// override de FFN por (candidato, capa) y estado recurrente (KV + DeltaNet) por
/// candidato — agnóstico a la arquitectura híbrida (qwen35/qwen27). Reutiliza
/// `run_deltanet_block` / `run_full_attn_block` intercambiando el estado del
/// generador por el del candidato (el bloque validado se ejecuta sin cambios).
#[allow(dead_code)]
pub(crate) fn forward_batched_hybrid(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    token: u32,
    pos: usize,
    scratch: &mut StreamingScratch,
    kv: &mut [Vec<LayerKvCache>],
    dn: &mut [Vec<Option<DeltaNetState>>],
    n_candidates: usize,
    mut get_override: impl FnMut(usize, usize) -> FfnOverride,
) -> Result<Vec<Vec<f32>>, StreamInferError> {
    let h = gen.config.hidden_size;
    let eps = gen.config.rms_norm_eps;
    let n_layers = gen.config.num_layers;
    let n = n_candidates;
    if kv.len() < n || dn.len() < n {
        return Err(StreamInferError::Msg(format!(
            "forward_batched_hybrid: kv/dn tienen {} filas, se necesitan {n}",
            kv.len().min(dn.len())
        )));
    }
    ensure_deltanet_cache(gen)?;

    let mut x: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut emb = vec![0.0f32; h];
        gen.embed_row("token_embd.weight", token, h, &mut emb)?;
        x.push(emb);
    }

    for layer in 0..n_layers {
        for c in 0..n {
            let ov = get_override(c, layer);
            // Mueve el estado DeltaNet del candidato al generador (y el del
            // generador a dn[c]); idem para el KV.
            std::mem::swap(&mut gen.kv, &mut kv[c]);
            let mut moved: Option<Vec<Option<DeltaNetState>>> = None;
            if let Some(v) = gen.deltanet_states.as_mut() {
                moved = Some(std::mem::take(v));
            }
            if let (Some(m), Some(cand)) = (moved.as_mut(), dn.get_mut(c)) {
                std::mem::swap(m, cand);
            }
            if let Some(v) = gen.deltanet_states.as_mut() {
                *v = moved.take().unwrap();
            }
            match hybrid_layer_kind(&gen.catalog, layer) {
                HybridLayerKind::DeltaNet => {
                    run_deltanet_block(gen, orch, scratch, layer, eps, &mut x[c], Some(&ov), None)?;
                }
                HybridLayerKind::FullAttn => {
                    run_full_attn_block(gen, orch, scratch, layer, pos, eps, &mut x[c], Some(&ov))?;
                }
                HybridLayerKind::NextN => {}
            }
            // Restaura el estado del generador (swap inverso).
            let mut moved: Option<Vec<Option<DeltaNetState>>> = None;
            if let Some(v) = gen.deltanet_states.as_mut() {
                moved = Some(std::mem::take(v));
            }
            if let (Some(m), Some(cand)) = (moved.as_mut(), dn.get_mut(c)) {
                std::mem::swap(m, cand);
            }
            if let Some(v) = gen.deltanet_states.as_mut() {
                *v = moved.take().unwrap();
            }
            std::mem::swap(&mut gen.kv, &mut kv[c]);
        }
    }

    // lm_head por candidato.
    let vocab = gen.config.vocab_size;
    let mut out = Vec::with_capacity(n);
    for c in 0..n {
        let mut xn = x[c].clone();
        rms_norm(&mut xn, &gen.output_norm, eps);
        let mut logits = vec![0.0f32; vocab];
        if gen.has_output_weight {
            if let Some(ow) = &gen.resident_output {
                orch.execute_quant_gemv(ow, &xn, &mut logits)?;
            } else {
                let ow = gen.catalog.load_quant_matrix("output.weight")?;
                orch.execute_quant_gemv(&ow, &xn, &mut logits)?;
            }
        } else {
            if let Some(emb) = &gen.resident_embed {
                orch.execute_quant_gemv(emb, &xn, &mut logits)?;
            } else {
                let emb = gen.catalog.load_quant_matrix("token_embd.weight")?;
                orch.execute_quant_gemv(&emb, &xn, &mut logits)?;
            }
        }
        out.push(logits);
    }
    Ok(out)
}


/// Forward **batcheado** híbrido (Fase 2, GEMM batcheado): N candidatos por capa,
/// con los pesos de cada capa cargados UNA vez por (capa, token) y las proyecciones
/// de atención + FFN **batcheadas** sobre N (criterios C1/C4). Estado recurrente
/// (KV + DeltaNet) por candidato. Agnóstico a la arquitectura híbrida.
pub(crate) fn forward_batched_hybrid_gemm(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    token: u32,
    pos: usize,
    _scratch: &mut StreamingScratch,
    kv: &mut [Vec<LayerKvCache>],
    dn: &mut [Vec<Option<DeltaNetState>>],
    n_candidates: usize,
    mut get_override: impl FnMut(usize, usize) -> FfnOverride,
) -> Result<Vec<Vec<f32>>, StreamInferError> {
    let h = gen.config.hidden_size;
    let n_layers = gen.config.num_layers;
    let eps = gen.config.rms_norm_eps;
    let n = n_candidates;
    if kv.len() < n || dn.len() < n {
        return Err(StreamInferError::Msg(format!(
            "forward_batched_hybrid_gemm: kv/dn tienen {} filas, se necesitan {n}",
            kv.len().min(dn.len())
        )));
    }
    ensure_deltanet_cache(gen)?;
    let mut x: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut emb = vec![0.0f32; h];
        gen.embed_row("token_embd.weight", token, h, &mut emb)?;
        x.push(emb);
    }
    for layer in 0..n_layers {
        let kind = hybrid_layer_kind(&gen.catalog, layer);
        if kind == HybridLayerKind::NextN {
            continue;
        }
        let ov: Vec<FfnOverride> = (0..n).map(|c| get_override(c, layer)).collect();
        match kind {
            HybridLayerKind::DeltaNet => {
                // SSM recurrente por (candidato, token) — estado propio.
                for c in 0..n {
                    let w = gen
                        .deltanet_weights
                        .as_ref()
                        .and_then(|v| v[layer].as_ref())
                        .ok_or_else(|| StreamInferError::Msg("deltanet weights ausentes".into()))?;
                    let s = dn[c][layer]
                        .as_mut()
                        .ok_or_else(|| StreamInferError::Msg("deltanet state ausente".into()))?;
                    let mut xn = x[c].clone();
                    rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);
                    w.decode_step(&xn, s, &mut x[c])?;
                }
            }
            HybridLayerKind::FullAttn => {
                // Atención batcheada: proyecciones GEMM sobre N + núcleo por candidato.
                let pack = gen.load_pack(layer)?;
                let layer_cfg = resolve_full_attn_cfg(&gen.catalog, layer, &gen.config)?;
                let n_heads = layer_cfg.num_heads;
                let n_kv = layer_cfg.num_kv_heads;
                let head_dim = layer_cfg.head_dim;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv * head_dim;
                let wo_in = pack.wo.ncols;
                let wq_rows = pack.wq.nrows;
                let mut x_flat = vec![0.0f32; n * h];
                for c in 0..n {
                    x_flat[c * h..(c + 1) * h].copy_from_slice(&x[c]);
                    rms_norm(&mut x_flat[c * h..(c + 1) * h], &gen.layer_norms[layer].attn_norm, eps);
                }
                let mut q_full = vec![0.0f32; n * wq_rows];
                let mut k_flat = vec![0.0f32; n * kv_dim];
                let mut v_flat = vec![0.0f32; n * kv_dim];
                orch.execute_quant_gemv_batched(&pack.wq, &x_flat, &mut q_full, n)?;
                orch.execute_quant_gemv_batched(&pack.wk, &x_flat, &mut k_flat, n)?;
                orch.execute_quant_gemv_batched(&pack.wv, &x_flat, &mut v_flat, n)?;
                let mut attn_out_flat = vec![0.0f32; n * wo_in];
                for c in 0..n {
                    let (mut q, fused_gate) = if wq_rows == n_heads * head_dim * 2 {
                        deinterleave_qg(
                            &q_full[c * (n_heads * head_dim * 2)..(c + 1) * (n_heads * head_dim * 2)],
                            n_heads,
                            head_dim,
                        )
                    } else if wq_rows == wo_in * 2 {
                        let (qh, gh) =
                            q_full[c * (wo_in * 2)..(c + 1) * (wo_in * 2)].split_at(wo_in);
                        (qh.to_vec(), Some(gh.to_vec()))
                    } else {
                        (q_full[c * wq_rows..(c + 1) * wq_rows].to_vec(), None)
                    };
                    q.resize(q_dim, 0.0);
                    if let Ok(qn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_q_norm.weight")) {
                        apply_head_rmsnorm(&mut q, &qn, n_heads, head_dim, eps);
                    }
                    if let Ok(kn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_k_norm.weight")) {
                        apply_head_rmsnorm(&mut k_flat[c * kv_dim..(c + 1) * kv_dim], &kn, n_kv, head_dim, eps);
                    }
                    let need_rebuild = kv[c][layer].heads.len() != n_kv
                        || kv[c][layer].heads.first().map(|hd| hd.dim) != Some(head_dim);
                    if need_rebuild {
                        kv[c][layer] = LayerKvCache::new(n_kv, head_dim, 4, 128);
                    }
                    attention_decode_step(
                        &layer_cfg,
                        &mut kv[c][layer],
                        &mut q,
                        &mut k_flat[c * kv_dim..(c + 1) * kv_dim],
                        &v_flat[c * kv_dim..(c + 1) * kv_dim],
                        pos,
                        &mut attn_out_flat[c * wo_in..(c + 1) * wo_in],
                    );
                    if let Some(g) = fused_gate {
                        for i in 0..q_dim {
                            attn_out_flat[c * wo_in + i] *= 1.0 / (1.0 + (-g[i]).exp());
                        }
                    } else if let Some(ref gate_w) = pack.attn_gate {
                        let mut gate = vec![0.0f32; gate_w.nrows];
                        gate_w.gemv(&x_flat[c * h..(c + 1) * h], &mut gate)?;
                        for i in 0..q_dim.min(gate.len()) {
                            attn_out_flat[c * wo_in + i] *= 1.0 / (1.0 + (-gate[i]).exp());
                        }
                    }
                }
                let mut attn_proj_flat = vec![0.0f32; n * h];
                orch.execute_quant_gemv_batched(&pack.wo, &attn_out_flat, &mut attn_proj_flat, n)?;
                for c in 0..n {
                    for i in 0..h {
                        x[c][i] += attn_proj_flat[c * h + i];
                    }
                }
            }
            HybridLayerKind::NextN => {}
        }
        // FFN batcheado (pesos cargados UNA vez por capa).
        batch_ffn_hybrid(gen, orch, layer, &ov, &mut x, n, eps)?;
    }

    // lm_head batcheado.
    let vocab = gen.config.vocab_size;
    let mut xn_flat = vec![0.0f32; n * h];
    for c in 0..n {
        xn_flat[c * h..(c + 1) * h].copy_from_slice(&x[c]);
        rms_norm(&mut xn_flat[c * h..(c + 1) * h], &gen.output_norm, eps);
    }
    let mut logits_flat = vec![0.0f32; n * vocab];
    if gen.has_output_weight {
        if let Some(ow) = &gen.resident_output {
            orch.execute_quant_gemv_batched(ow, &xn_flat, &mut logits_flat, n)?;
        } else {
            let ow = gen.catalog.load_quant_matrix("output.weight")?;
            orch.execute_quant_gemv_batched(&ow, &xn_flat, &mut logits_flat, n)?;
        }
    } else {
        if let Some(emb) = &gen.resident_embed {
            orch.execute_quant_gemv_batched(emb, &xn_flat, &mut logits_flat, n)?;
        } else {
            let emb = gen.catalog.load_quant_matrix("token_embd.weight")?;
            orch.execute_quant_gemv_batched(&emb, &xn_flat, &mut logits_flat, n)?;
        }
    }
    let mut out = Vec::with_capacity(n);
    for c in 0..n {
        out.push(logits_flat[c * vocab..(c + 1) * vocab].to_vec());
    }
    Ok(out)
}

/// FFN batcheado de la capa `layer` para N candidatos: carga gate/up/down UNA vez,
/// GEMM `[N×M]` + override (CSR de Vía B) por candidato esparso.
fn batch_ffn_hybrid(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    layer: usize,
    ov: &[FfnOverride],
    x: &mut [Vec<f32>],
    n: usize,
    eps: f32,
) -> Result<(), StreamInferError> {
    let h = gen.config.hidden_size;
    let ff = gen.config.intermediate_size;
    let (gate, up, down, gcsr, ucsr, dcsr) = gen.catalog.load_ffn_matrices(layer)?;
    let mut x_flat = vec![0.0f32; n * h];
    for c in 0..n {
        x_flat[c * h..(c + 1) * h].copy_from_slice(&x[c]);
        rms_norm(&mut x_flat[c * h..(c + 1) * h], &gen.layer_norms[layer].ffn_norm, eps);
    }
    let mut gate_flat = vec![0.0f32; n * ff];
    let mut up_flat = vec![0.0f32; n * ff];
    let mut down_flat = vec![0.0f32; n * h];
    orch.execute_quant_gemv_batched(&gate, &x_flat, &mut gate_flat, n)?;
    orch.execute_quant_gemv_batched(&up, &x_flat, &mut up_flat, n)?;
    for c in 0..n {
        let cs = ov[c].gate.as_ref().or(gcsr.as_ref());
        if let Some(cs) = cs {
            let out = gen.spmm_csr(orch, &x_flat[c * h..(c + 1) * h], cs)?;
            gate_flat[c * ff..(c + 1) * ff].copy_from_slice(&out);
        }
        let cs = ov[c].up.as_ref().or(ucsr.as_ref());
        if let Some(cs) = cs {
            let out = gen.spmm_csr(orch, &x_flat[c * h..(c + 1) * h], cs)?;
            up_flat[c * ff..(c + 1) * ff].copy_from_slice(&out);
        }
    }
    for i in 0..n * ff {
        let g = gate_flat[i];
        gate_flat[i] = (g / (1.0 + (-g).exp())) * up_flat[i];
    }
    orch.execute_quant_gemv_batched(&down, &gate_flat, &mut down_flat, n)?;
    for c in 0..n {
        let cs = ov[c].down.as_ref().or(dcsr.as_ref());
        if let Some(cs) = cs {
            let out = gen.spmm_csr(orch, &gate_flat[c * ff..(c + 1) * ff], cs)?;
            down_flat[c * h..(c + 1) * h].copy_from_slice(&out);
        }
        for i in 0..h {
            x[c][i] += down_flat[c * h + i];
        }
    }
    Ok(())
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

/// Forward **por capas** híbrido (Fase 2, layer-major): procesa TODA la secuencia
/// de tokens por capa en una sola llamada. Los pesos de cada capa se cargan UNA
/// vez por generación, el override se construye UNA vez por (candidato, capa) y el
/// FFN + las proyecciones de atención se batchean sobre `[N×n_pos]` (criterios
/// C1/C4). Estado recurrente (KV + DeltaNet) por candidato.
pub(crate) fn forward_batched_hybrid_seq(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    tokens: &[u32],
    kv: &mut [Vec<LayerKvCache>],
    dn: &mut [Vec<Option<DeltaNetState>>],
    n_candidates: usize,
    mut get_override: impl FnMut(
        usize,
        usize,
        &Option<Arc<Vec<f32>>>,
        &Option<Arc<Vec<f32>>>,
        &Option<Arc<Vec<f32>>>,
    ) -> FfnOverride,
) -> Result<Vec<Vec<f32>>, StreamInferError> {
    let h = gen.config.hidden_size;
    let n_layers = gen.config.num_layers;
    let n_pos = tokens.len();
    let n = n_candidates;
    let eps = gen.config.rms_norm_eps;
    if kv.len() < n || dn.len() < n || n_pos == 0 {
        return Err(StreamInferError::Msg(format!(
            "forward_batched_hybrid_seq: kv/dn/n_pos inconsistentes (n={n}, n_pos={n_pos})"
        )));
    }
    ensure_deltanet_cache(gen)?;
    let mut x: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; n_pos * h]).collect();
    for c in 0..n {
        for t in 0..n_pos {
            let mut emb = vec![0.0f32; h];
            gen.embed_row("token_embd.weight", tokens[t], h, &mut emb)?;
            x[c][t * h..(t + 1) * h].copy_from_slice(&emb);
        }
    }
    for layer in 0..n_layers {
        let kind = hybrid_layer_kind(&gen.catalog, layer);
        if kind == HybridLayerKind::NextN {
            continue;
        }
        // Pesos compartidos por capa (Fase 2, C1/C4): si hay GPU y Q4_K, el kernel
        // dequantiza en GPU (sin 23 GB F32/gen); si no, dequant F32 una vez por capa.
        let ffn = gen.catalog.load_ffn_matrices(layer)?;
        let has_gpu = orch.opencl_engine().is_some();
        // Dequant Q4_K en GPU opt-in (HAYAI_SPMM_Q4=1); F32 compartido por defecto.
        let can_q4 = std::env::var("HAYAI_SPMM_Q4").ok().as_deref() == Some("1")
            && ffn.0.ggml_type == GgmlType::Q4_K;
        let gate_w = if has_gpu && can_q4 {
            None
        } else {
            gen.catalog
                .dequant_f32(&format!("blk.{layer}.ffn_gate.weight"))
                .ok()
                .map(Arc::new)
        };
        let up_w = if has_gpu && can_q4 {
            None
        } else {
            gen.catalog
                .dequant_f32(&format!("blk.{layer}.ffn_up.weight"))
                .ok()
                .map(Arc::new)
        };
        let down_w = if has_gpu && can_q4 {
            None
        } else {
            gen.catalog
                .dequant_f32(&format!("blk.{layer}.ffn_down.weight"))
                .ok()
                .map(Arc::new)
        };
        let gate_q4 = if has_gpu && can_q4 {
            Some(ffn.0.raw_bytes())
        } else {
            None
        };
        let up_q4 = if has_gpu && can_q4 {
            Some(ffn.1.raw_bytes())
        } else {
            None
        };
        let down_q4 = if has_gpu && can_q4 {
            Some(ffn.2.raw_bytes())
        } else {
            None
        };

        let ov: Vec<FfnOverride> = (0..n)
            .map(|c| get_override(c, layer, &gate_w, &up_w, &down_w))
            .collect();
        match kind {
            HybridLayerKind::DeltaNet => {
                // SSM recurrente por (candidato, token) — estado propio.
                for c in 0..n {
                    for t in 0..n_pos {
                        let w = gen
                            .deltanet_weights
                            .as_ref()
                            .and_then(|v| v[layer].as_ref())
                            .ok_or_else(|| StreamInferError::Msg("deltanet weights ausentes".into()))?;
                        let s = dn[c][layer]
                            .as_mut()
                            .ok_or_else(|| StreamInferError::Msg("deltanet state ausente".into()))?;
                        let mut xn = x[c][t * h..(t + 1) * h].to_vec();
                        rms_norm(&mut xn, &gen.layer_norms[layer].attn_norm, eps);
                        w.decode_step(&xn, s, &mut x[c][t * h..(t + 1) * h])?;
                    }
                }
                if std::env::var("HAYAI_DUMP_LAYER_RMS").ok().as_deref() == Some("1") {
                    let mut ms = 0.0f32;
                    for i in 0..h {
                        ms += x[0][i] * x[0][i];
                    }
                    eprintln!(
                        "BATCH_LAYER_RMS tok0 L{layer} afterDN: {:.4}",
                        (ms / h as f32).sqrt()
                    );
                }
            }
            HybridLayerKind::FullAttn => {
                // Atención batcheada sobre N×n_pos: proyecciones GEMM + núcleo por
                // (candidato, token) con KV acumulada.
                let pack = gen.load_pack(layer)?;
                let layer_cfg = resolve_full_attn_cfg(&gen.catalog, layer, &gen.config)?;
                let n_heads = layer_cfg.num_heads;
                let n_kv = layer_cfg.num_kv_heads;
                let head_dim = layer_cfg.head_dim;
                let q_dim = n_heads * head_dim;
                let kv_dim = n_kv * head_dim;
                let wo_in = pack.wo.ncols;
                let wq_rows = pack.wq.nrows;
                let batch = n * n_pos;
                let mut x_flat = vec![0.0f32; batch * h];
                for c in 0..n {
                    for t in 0..n_pos {
                        x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h]
                            .copy_from_slice(&x[c][t * h..(t + 1) * h]);
                        rms_norm(
                            &mut x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h],
                            &gen.layer_norms[layer].attn_norm,
                            eps,
                        );
                    }
                }
                let mut q_full = vec![0.0f32; batch * wq_rows];
                let mut k_flat = vec![0.0f32; batch * kv_dim];
                let mut v_flat = vec![0.0f32; batch * kv_dim];
                orch.execute_quant_gemv_batched(&pack.wq, &x_flat, &mut q_full, batch)?;
                orch.execute_quant_gemv_batched(&pack.wk, &x_flat, &mut k_flat, batch)?;
                orch.execute_quant_gemv_batched(&pack.wv, &x_flat, &mut v_flat, batch)?;
                let mut attn_out_flat = vec![0.0f32; batch * wo_in];
                for c in 0..n {
                    for t in 0..n_pos {
                        let idx = c * n_pos + t;
                        let (mut q, fused_gate) = if wq_rows == n_heads * head_dim * 2 {
                            deinterleave_qg(
                                &q_full[idx * (n_heads * head_dim * 2)..(idx + 1) * (n_heads * head_dim * 2)],
                                n_heads,
                                head_dim,
                            )
                        } else if wq_rows == wo_in * 2 {
                            let (qh, gh) = q_full[idx * (wo_in * 2)..(idx + 1) * (wo_in * 2)].split_at(wo_in);
                            (qh.to_vec(), Some(gh.to_vec()))
                        } else {
                            (q_full[idx * wq_rows..(idx + 1) * wq_rows].to_vec(), None)
                        };
                        q.resize(q_dim, 0.0);
                        if let Ok(qn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_q_norm.weight")) {
                            apply_head_rmsnorm(&mut q, &qn, n_heads, head_dim, eps);
                        }
                        if let Ok(kn) = gen.catalog.dequant_f32(&format!("blk.{layer}.attn_k_norm.weight")) {
                            apply_head_rmsnorm(&mut k_flat[idx * kv_dim..(idx + 1) * kv_dim], &kn, n_kv, head_dim, eps);
                        }
                        let need_rebuild = kv[c][layer].heads.len() != n_kv
                            || kv[c][layer].heads.first().map(|hd| hd.dim) != Some(head_dim);
                        if need_rebuild {
                            kv[c][layer] = LayerKvCache::new(n_kv, head_dim, 4, 128);
                        }
                        attention_decode_step(
                            &layer_cfg,
                            &mut kv[c][layer],
                            &mut q,
                            &mut k_flat[idx * kv_dim..(idx + 1) * kv_dim],
                            &v_flat[idx * kv_dim..(idx + 1) * kv_dim],
                            t,
                            &mut attn_out_flat[idx * wo_in..(idx + 1) * wo_in],
                        );
                        if let Some(g) = fused_gate {
                            for i in 0..q_dim {
                                attn_out_flat[idx * wo_in + i] *= 1.0 / (1.0 + (-g[i]).exp());
                            }
                        } else if let Some(ref gate_w) = pack.attn_gate {
                            let mut gate = vec![0.0f32; gate_w.nrows];
                            gate_w.gemv(&x_flat[idx * h..(idx + 1) * h], &mut gate)?;
                            for i in 0..q_dim.min(gate.len()) {
                                attn_out_flat[idx * wo_in + i] *= 1.0 / (1.0 + (-gate[i]).exp());
                            }
                        }
                    }
                }
                let mut attn_proj_flat = vec![0.0f32; batch * h];
                orch.execute_quant_gemv_batched(&pack.wo, &attn_out_flat, &mut attn_proj_flat, batch)?;
                for c in 0..n {
                    for t in 0..n_pos {
                        for i in 0..h {
                            x[c][t * h + i] += attn_proj_flat[(c * n_pos + t) * h + i];
                        }
                    }
                }
            }
            HybridLayerKind::NextN => {}
        }
        // FFN batcheado sobre N×n_pos (override construido UNA vez por (cand, capa)).
        batch_ffn_hybrid_seq(
            gen, orch, layer, &ov, &mut x, n, n_pos, eps, gate_w.as_ref(), up_w.as_ref(),
            down_w.as_ref(), gate_q4.as_deref(), up_q4.as_deref(), down_q4.as_deref(), ffn,
        )?;
        if std::env::var("HAYAI_DUMP_LAYER_RMS").ok().as_deref() == Some("1") {
            let kind = match kind {
                HybridLayerKind::DeltaNet => "DN",
                HybridLayerKind::FullAttn => "FA",
                HybridLayerKind::NextN => "N",
            };
            let mut ms = 0.0f32;
            for i in 0..h {
                ms += x[0][i] * x[0][i];
            }
            eprintln!(
                "BATCH_LAYER_RMS tok0 L{layer}({kind}): {:.4}",
                (ms / h as f32).sqrt()
            );
        }
    }

    // lm_head batcheado sobre N×n_pos.
    let vocab = gen.config.vocab_size;
    let batch = n * n_pos;
    let mut xn_flat = vec![0.0f32; batch * h];
    for c in 0..n {
        for t in 0..n_pos {
            xn_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h]
                .copy_from_slice(&x[c][t * h..(t + 1) * h]);
            rms_norm(
                &mut xn_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h],
                &gen.output_norm,
                eps,
            );
        }
    }
    let mut logits_flat = vec![0.0f32; batch * vocab];
    if gen.has_output_weight {
        if let Some(ow) = &gen.resident_output {
            orch.execute_quant_gemv_batched(ow, &xn_flat, &mut logits_flat, batch)?;
        } else {
            let ow = gen.catalog.load_quant_matrix("output.weight")?;
            orch.execute_quant_gemv_batched(&ow, &xn_flat, &mut logits_flat, batch)?;
        }
    } else {
        if let Some(emb) = &gen.resident_embed {
            orch.execute_quant_gemv_batched(emb, &xn_flat, &mut logits_flat, batch)?;
        } else {
            let emb = gen.catalog.load_quant_matrix("token_embd.weight")?;
            orch.execute_quant_gemv_batched(&emb, &xn_flat, &mut logits_flat, batch)?;
        }
    }
    let mut out = Vec::with_capacity(n);
    for c in 0..n {
        out.push(logits_flat[c * n_pos * vocab..(c + 1) * n_pos * vocab].to_vec());
    }
    Ok(out)
}

/// FFN batcheado por capa sobre N×n_pos (pesos cargados UNA vez por el caller;
/// override CSR por candidato esparso construido UNA vez por (candidato, capa);
/// SparseAdj batcheado en GPU desde Q4 con dequant en-kernel o F32 compartido —
/// Fase 2, C1/C4).
fn batch_ffn_hybrid_seq(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    layer: usize,
    ov: &[FfnOverride],
    x: &mut [Vec<f32>],
    n: usize,
    n_pos: usize,
    eps: f32,
    gate_w: Option<&Arc<Vec<f32>>>,
    up_w: Option<&Arc<Vec<f32>>>,
    down_w: Option<&Arc<Vec<f32>>>,
    gate_q4: Option<&[u8]>,
    up_q4: Option<&[u8]>,
    down_q4: Option<&[u8]>,
    ffn: (
        QuantMatrix,
        QuantMatrix,
        QuantMatrix,
        Option<hayai_model::CsrSparse>,
        Option<hayai_model::CsrSparse>,
        Option<hayai_model::CsrSparse>,
    ),
) -> Result<(), StreamInferError> {
    let h = gen.config.hidden_size;
    let ff = gen.config.intermediate_size;
    let (gate, up, down, gcsr, ucsr, dcsr) = ffn;
    let batch = n * n_pos;
    let mut x_flat = vec![0.0f32; batch * h];
    for c in 0..n {
        for t in 0..n_pos {
            x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h]
                .copy_from_slice(&x[c][t * h..(t + 1) * h]);
            rms_norm(
                &mut x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h],
                &gen.layer_norms[layer].ffn_norm,
                eps,
            );
        }
    }
    if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!("BATCH_XN L0: {:?}", &x_flat[0..8]);
    }
    let mut gate_flat = vec![0.0f32; batch * ff];
    let mut up_flat = vec![0.0f32; batch * ff];
    let mut down_flat = vec![0.0f32; batch * h];
    let has_any_adj = ov
        .iter()
        .any(|o| o.gate_adj.is_some() || o.up_adj.is_some() || o.down_adj.is_some());
    orch.execute_quant_gemv_batched(&gate, &x_flat, &mut gate_flat, batch)?;
    if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!("BATCH_GATE L0 pre: {:?}", &gate_flat[0..8]);
    }
    let dense_gate = if std::env::var("HAYAI_DEBUG_OVERRIDE").ok().as_deref() == Some("1") {
        Some(gate_flat.clone())
    } else {
        None
    };
    orch.execute_quant_gemv_batched(&up, &x_flat, &mut up_flat, batch)?;
    if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!("BATCH_UP L0 pre: {:?}", &up_flat[0..8]);
    }
    if has_any_adj {
        gen.apply_sparse_adj_block(
            orch, ov, |o| o.gate_adj.as_ref(), &x_flat, n, n_pos, h, ff,
            gate_w, gate_q4, &mut gate_flat,
        )?;
        gen.apply_sparse_adj_block(
            orch, ov, |o| o.up_adj.as_ref(), &x_flat, n, n_pos, h, ff,
            up_w, up_q4, &mut up_flat,
        )?;
        if let Some(dg) = dense_gate {
            let mut max = 0.0f32;
            let mut mean = 0.0f32;
            let mut mnorm = 0.0f32;
            for i in 0..gate_flat.len() {
                let d = (gate_flat[i] - dg[i]).abs();
                max = max.max(d);
                mean += d;
                mnorm += dg[i].abs();
            }
            let cnt = gate_flat.len() as f32;
            eprintln!(
                "[override dbg] layer {layer} gate: max|ov-dense|={max:.4} mean|ov-dense|={:.4} mean|dense|={:.4}",
                mean / cnt,
                mnorm / cnt
            );
        }
    }
    for c in 0..n {
        let cs = ov[c].gate.as_ref().or(gcsr.as_ref());
        if let Some(cs) = cs {
            let out = gen.spmm_csr(orch, &x_flat[c * n_pos * h..(c + 1) * n_pos * h], cs)?;
            gate_flat[c * n_pos * ff..(c + 1) * n_pos * ff].copy_from_slice(&out);
        }
        let cs = ov[c].up.as_ref().or(ucsr.as_ref());
        if let Some(cs) = cs {
            let out = gen.spmm_csr(orch, &x_flat[c * n_pos * h..(c + 1) * n_pos * h], cs)?;
            up_flat[c * n_pos * ff..(c + 1) * n_pos * ff].copy_from_slice(&out);
        }
    }
    for i in 0..batch * ff {
        let g = gate_flat[i];
        gate_flat[i] = (g / (1.0 + (-g).exp())) * up_flat[i];
    }
    if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!("BATCH_GATE_ACT L0: {:?}", &gate_flat[0..8]);
    }
    orch.execute_quant_gemv_batched(&down, &gate_flat, &mut down_flat, batch)?;
    if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!("BATCH_DOWN L0: {:?}", &down_flat[0..8]);
    }
    if has_any_adj {
        gen.apply_sparse_adj_block(
            orch, ov, |o| o.down_adj.as_ref(), &gate_flat, n, n_pos, ff, h,
            down_w, down_q4, &mut down_flat,
        )?;
    }
    for c in 0..n {
        let cs = ov[c].down.as_ref().or(dcsr.as_ref());
        if let Some(cs) = cs {
            let out = gen.spmm_csr(orch, &gate_flat[c * n_pos * ff..(c + 1) * n_pos * ff], cs)?;
            down_flat[c * n_pos * h..(c + 1) * n_pos * h].copy_from_slice(&out);
        }
        for t in 0..n_pos {
            for i in 0..h {
                x[c][t * h + i] += down_flat[(c * n_pos + t) * h + i];
            }
        }
    }
    Ok(())
}


