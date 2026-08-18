//! PRD generate: deterministic layer streaming + Attn∥FFN + dGPU∥APU.
//!
//! Pipeline per layer (decode):
//! 1. Prefetch layer N+1 into ping-pong (second `fork_reader` handle).
//! 2. CPU Attention on pack N.
//! 3. Enqueue FFN gate/up async (`cl_event`); while in flight, join prefetch (I/O∥FFN).
//! 4. SiLU + down; residual. Next iteration's Attn starts as soon as FFN finishes
//!    (macro-pipeline: GPU FFN(N) overlaps CPU/I/O prep for N+1; Attn(N+1) follows
//!    immediately — residual deps prevent Attn(N+1) before FFN(N) completes).

use hayai_cpu::{attention_decode_step, rms_norm, AttentionConfig, LayerKvCache};
use hayai_model::{
    sample, GgmlType, GgufCatalog, GgufError, LayerPackLayout, LayerWeightPack, ModelConfig,
    QuantMatrix, SamplerConfig, Tokenizer,
};
use hayai_opencl::PendingGemv;
use std::path::PathBuf;
use std::thread::{self, JoinHandle};
use std::time::Instant;
use tracing::{debug, info};

use crate::adaptive_window::{compute_window_plan, MemoryStrategy, WindowPlan};
use crate::infer::GenerateStats;
use crate::orchestrator::EngineOrchestrator;

/// Raw host-slot pointer for prefetch threads (other ping-pong slot only).
/// Stored as `usize` so the handle is `Send` without relying on raw-pointer auto traits.
pub(crate) struct PrefetchSlotPtr {
    addr: usize,
    len: usize,
}

impl PrefetchSlotPtr {
    pub(crate) fn new(ptr: *mut u8, len: usize) -> Self {
        Self {
            addr: ptr as usize,
            len,
        }
    }

    /// # Safety
    /// Caller guarantees the slot remains mapped/alive for the duration of the borrow.
    pub(crate) unsafe fn as_mut_slice(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.addr as *mut u8, self.len) }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StreamInferError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Orchestrator(#[from] crate::orchestrator::OrchestratorError),
    #[error(transparent)]
    OpenCl(#[from] hayai_opencl::OpenClError),
    #[error("{0}")]
    Msg(String),
}

pub(crate) struct LayerNorms {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
}

pub struct StreamingGenerator {
    pub catalog: GgufCatalog,
    pub path: PathBuf,
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    pub attn_cfg: AttentionConfig,
    pub kv: Vec<LayerKvCache>,
    pub(crate) layer_norms: Vec<LayerNorms>,
    pub(crate) output_norm: Vec<f32>,
    pub(crate) has_output_weight: bool,
    pub sampler: SamplerConfig,
    pub rng: u64,
    pub position: usize,
    pub io_bytes: u64,
    pub attn_secs: f64,
    pub ffn_secs: f64,
    pub io_secs: f64,
    /// SVM map / host-slot prepare (honest desglose; previously hidden in wall).
    pub map_secs: f64,
    /// dGPU WriteBufferRect FFN-slice DMA (honest desglose).
    pub dma_secs: f64,
    /// Wall time spent joining prefetch / doing CPU work while FFN events were in flight.
    pub overlap_secs: f64,
    /// Seconds of Attn that ran while a previous token's FFN was still in flight (prefill).
    pub attn_ffn_overlap_secs: f64,
    pub used_apu: bool,
    pub used_dgpu: bool,
    pub prefetch_hits: usize,
    pub io_backend: hayai_io::IoBackend,
    pub transfer_path: Option<hayai_opencl::TransferPath>,
    pub wall_compute_secs: f64,
    /// Enforceable streaming budget (2×layer + KV + acts + meta slack).
    pub memory_budget: Option<crate::metrics::StreamingMemoryBudget>,
    /// Runtime Hayai-owned bytes (weights windows + KV + acts + staging).
    pub owned_mem: Option<crate::metrics::HayaiOwnedMemory>,
    kv_sinks: usize,
    kv_window: usize,
    /// Activation ping-pong A/B (design §1): `act_sel` is the live residual stream.
    act_pp: [Vec<f32>; 2],
    act_sel: usize,
    pub(crate) ws_gate: Vec<f32>,
    pub(crate) ws_up: Vec<f32>,
    pub(crate) ws_down: Vec<f32>,
    /// Host/SVM slot capacity used by prefetch threads.
    pub(crate) layer_scratch_cap: usize,
    /// HRM frozen low-cycle init state (`hrm.z_l_init`), length = hidden.
    pub(crate) z_l_init: Option<Vec<f32>>,
    /// Adaptive memory strategy for the resident/macro-chunk window.
    pub memory_strategy: MemoryStrategy,
    /// Last computed adaptive window plan (k_chunk / resident / window_bytes).
    pub window_plan: Option<WindowPlan>,
    /// Resident mode: cached final projection matrix (`output.weight`) for zero-I/O
    /// decode. `None` in streaming mode.
    pub(crate) resident_output: Option<QuantMatrix>,
    /// Resident mode: cached embedding matrix (`token_embd.weight`) for zero-I/O
    /// decode. `None` in streaming mode.
    pub(crate) resident_embed: Option<QuantMatrix>,
    /// Stored ExecPlan used to drive the forward graph (Ola 2).
    pub exec_plan: Option<crate::exec_plan::ExecPlan>,
    /// Qwen3.5 DeltaNet weights (None entries for full-attn layers).
    pub(crate) deltanet_weights: Option<Vec<Option<crate::deltanet::DeltaNetLayerWeights>>>,
    pub(crate) deltanet_states: Option<Vec<Option<crate::deltanet::DeltaNetState>>>,
}

impl StreamingGenerator {
    pub fn open(
        path: impl AsRef<std::path::Path>,
        tokenizer: Tokenizer,
        sink: usize,
        window: usize,
        sampler: SamplerConfig,
        seed: u64,
    ) -> Result<Self, StreamInferError> {
        let path = path.as_ref().to_path_buf();
        let mut catalog = GgufCatalog::open(&path)?;
        let config = load_config(&catalog)?;
        let attn_cfg = build_attn_config(&catalog, &config)?;

        // HRM / some hybrids use parameterless RMSNorm (no weight tensors).
        let ones = |n: usize| vec![1.0f32; n];
        let h = config.hidden_size;
        let mut layer_norms = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let attn_norm = catalog
                .dequant_f32(&format!("blk.{i}.attn_norm.weight"))
                .or_else(|_| catalog.dequant_f32(&format!("blk.{i}.attention_norm.weight")))
                .unwrap_or_else(|_| ones(h));
            let ffn_norm = catalog
                .dequant_f32(&format!("blk.{i}.ffn_norm.weight"))
                .or_else(|_| catalog.dequant_f32(&format!("blk.{i}.post_attention_norm.weight")))
                .unwrap_or_else(|_| ones(h));
            layer_norms.push(LayerNorms { attn_norm, ffn_norm });
        }
        let output_norm = catalog
            .dequant_f32("output_norm.weight")
            .unwrap_or_else(|_| ones(h));
        let has_output_weight = catalog.tensor("output.weight").is_ok();

        let kv_slots = config
            .hrm
            .as_ref()
            .map(|h| h.kv_slots())
            .unwrap_or(config.num_layers);
        let kv = if config.architecture.contains("gemma") {
            crate::gemma_infer::build_layer_kv_caches(&catalog, &config, sink, window)?
        } else if config.architecture.contains("qwen35")
            || config.architecture.contains("qwen3next")
        {
            // Per full-attn layer dims from tensors (not a single 4B-shaped AttentionConfig).
            crate::layer_cfg::build_hybrid_kv_caches(&catalog, &config, sink, window)?
        } else {
            (0..kv_slots)
                .map(|_| LayerKvCache::new(attn_cfg.num_kv_heads, attn_cfg.head_dim, sink, window))
                .collect()
        };

        let z_l_init = if config.hrm.is_some() {
            catalog.dequant_f32("hrm.z_l_init").ok().or_else(|| {
                catalog
                    .dequant_f32("z_l_init")
                    .ok()
            })
        } else {
            None
        };

        info!(
            "StreamingGenerator: arch={} physical_layers={} kv_slots={} — WeightIo={} ping-pong + async FFN (no weight mmap)",
            config.architecture,
            config.num_layers,
            kv_slots,
            catalog.io_backend().as_str()
        );
        if let Some(ref hrm) = config.hrm {
            info!(
                "HRM: H_cycles={} L_cycles={} layers/stack={} embed_scale={:.4} z_l_init={}",
                hrm.h_cycles,
                hrm.l_cycles,
                hrm.layers_per_stack,
                hrm.embedding_scale,
                z_l_init.is_some()
            );
        }

        let io_backend = catalog.io_backend();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        Ok(Self {
            catalog,
            path,
            config,
            tokenizer,
            attn_cfg,
            kv,
            layer_norms,
            output_norm,
            has_output_weight,
            sampler,
            rng: seed,
            position: 0,
            io_bytes: 0,
            attn_secs: 0.0,
            ffn_secs: 0.0,
            io_secs: 0.0,
            map_secs: 0.0,
            dma_secs: 0.0,
            overlap_secs: 0.0,
            attn_ffn_overlap_secs: 0.0,
            used_apu: false,
            used_dgpu: false,
            prefetch_hits: 0,
            io_backend,
            transfer_path: None,
            wall_compute_secs: 0.0,
            memory_budget: None,
            owned_mem: None,
            kv_sinks: sink,
            kv_window: window,
            act_pp: [vec![0.0; hidden], vec![0.0; hidden]],
            act_sel: 0,
            ws_gate: vec![0.0; intermediate],
            ws_up: vec![0.0; intermediate],
            ws_down: vec![0.0; hidden],
            layer_scratch_cap: 0,
            z_l_init,
            exec_plan: None,
            deltanet_weights: None,
            deltanet_states: None,
            memory_strategy: MemoryStrategy::AutoFit,
            window_plan: None,
            resident_output: None,
            resident_embed: None,
        })
    }

    /// Override the memory strategy before the first `generate()` call.
    pub fn set_memory_strategy(&mut self, strategy: MemoryStrategy) {
        self.memory_strategy = strategy;
    }

    fn act(&self) -> &[f32] {
        &self.act_pp[self.act_sel]
    }

    fn act_mut(&mut self) -> &mut [f32] {
        &mut self.act_pp[self.act_sel]
    }

    /// Design §1: copy live residual into the other buffer and switch (A↔B).
    fn act_swap_prepare(&mut self) {
        let src = self.act_sel;
        let dst = 1 - src;
        // Split so both halves can be borrowed without overlapping `&mut` / `&`.
        if src == 0 {
            let (a, b) = self.act_pp.split_at_mut(1);
            b[0].copy_from_slice(&a[0]);
        } else {
            let (a, b) = self.act_pp.split_at_mut(1);
            a[0].copy_from_slice(&b[0]);
        }
        self.act_sel = dst;
    }

    pub(crate) fn load_pack(&mut self, layer: usize) -> Result<LayerWeightPack, StreamInferError> {
        let t0 = Instant::now();
        let pack = self.catalog.load_layer_pack(layer)?;
        self.io_secs += t0.elapsed().as_secs_f64();
        self.io_bytes += pack.nbytes() as u64;
        Ok(pack)
    }

    /// Embedding row read. Resident mode serves it from the cached embedding matrix
    /// (zero disk I/O per token); streaming mode reads the row on demand.
    pub(crate) fn embed_row(
        &mut self,
        embd_name: &str,
        token: u32,
        h: usize,
        dst: &mut [f32],
    ) -> Result<(), StreamInferError> {
        if embd_name == "token_embd.weight" {
            if let Some(emb) = &self.resident_embed {
                emb.embed_row(token, dst)?;
                self.io_bytes += (h * 2) as u64;
                return Ok(());
            }
        }
        let t0 = Instant::now();
        self.catalog.read_embed_row(embd_name, token, h, dst)?;
        self.io_secs += t0.elapsed().as_secs_f64();
        self.io_bytes += (h * 2) as u64;
        Ok(())
    }

    /// Disk → host/SVM slot as views (host-mapped). DMA overlaps Attn; unmap before FFN.
    ///
    /// Resident mode: no disk read — layers already live in device memory; this
    /// only maps the SVM base (coarse) and rebuilds views over this layer's slice.
    pub(crate) fn stage_pack(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        slot: usize,
        layer: usize,
    ) -> Result<(LayerWeightPack, LayerPackLayout), StreamInferError> {
        if scratch.resident {
            let t_map = Instant::now();
            scratch.prepare_host_write(&orch.pool, layer)?;
            self.map_secs += t_map.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let base = scratch.host_slot(layer);
            let (pack, layout) = self.catalog.layer_pack_views_from_base(layer, base)?;
            self.io_secs += t0.elapsed().as_secs_f64();
            self.io_bytes += layout.total as u64;
            return Ok((pack, layout));
        }
        let t_map = Instant::now();
        scratch.prepare_host_write(&orch.pool, slot)?;
        self.map_secs += t_map.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let (pack, layout) = self
            .catalog
            .load_layer_pack_into(layer, scratch.host_slot_mut(slot))?;
        self.io_secs += t0.elapsed().as_secs_f64();
        self.io_bytes += layout.total as u64;
        self.owned_mem
            .as_mut()
            .map(|m| m.note_scratch(scratch.slot_capacity() as u64 * 2));
        Ok((pack, layout))
    }

    /// DMA FFN slices while host stays mapped (overlaps Attn). Does **not** unmap.
    pub(crate) fn begin_ffn_dma(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        slot: usize,
        layout: &LayerPackLayout,
    ) -> Result<(), StreamInferError> {
        if scratch.resident {
            // Layers were DMA'd into the dGPU mirrors once at preload.
            return Ok(());
        }
        let t0 = Instant::now();
        scratch.dma_ffn_keep_mapped(&orch.pool, slot, layout)?;
        self.dma_secs += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// Unmap SVM after Attn so device FFN can run (DMA already issued).
    pub(crate) fn finish_ffn_unmap(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        slot: usize,
    ) -> Result<(), StreamInferError> {
        if scratch.resident {
            // Unmap the whole resident base so device kernels can read the SVM.
            let t0 = Instant::now();
            scratch.unmap_host_for_device(&orch.pool, slot)?;
            self.map_secs += t0.elapsed().as_secs_f64();
            return Ok(());
        }
        let t0 = Instant::now();
        scratch.unmap_host_for_device(&orch.pool, slot)?;
        self.map_secs += t0.elapsed().as_secs_f64();
        Ok(())
    }

    /// Combined DMA+unmap (legacy / single-call sites). Prefer begin_ffn_dma ∥ Attn + finish.
    #[allow(dead_code)]
    pub(crate) fn commit_ffn_dma(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        slot: usize,
        layout: &LayerPackLayout,
    ) -> Result<(), StreamInferError> {
        self.begin_ffn_dma(orch, scratch, slot, layout)?;
        self.finish_ffn_unmap(orch, scratch, slot)
    }

    /// Prefetch already wrote into the ping-pong slot — rebind views only (no heap copy).
    pub(crate) fn bind_prefetched_slot(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        slot: usize,
        layout: &LayerPackLayout,
        pack: &LayerWeightPack,
    ) -> Result<LayerWeightPack, StreamInferError> {
        if scratch.resident {
            // Resident: the pack already points into the correct resident layer
            // slice (built by `stage_pack`). Rebind nothing — `host_slot(idx)` here
            // would index a ping-pong slot (`% 2`) which is wrong for residents.
            return Ok(pack.clone());
        }
        let t_map = Instant::now();
        scratch.ensure_host_readable(&orch.pool, slot)?;
        self.map_secs += t_map.elapsed().as_secs_f64();
        let rebound = pack.rebind_views(scratch.host_slot(slot), layout);
        if let Some(m) = self.owned_mem.as_mut() {
            m.note_prefetch_staging(0);
            m.note_scratch(scratch.slot_capacity() as u64 * 2);
        }
        Ok(rebound)
    }

    /// Compat: copy blob into slot then bind (tests / non-scratch paths).
    #[allow(dead_code)]
    pub(crate) fn ingest_prefetched(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        slot: usize,
        blob: &[u8],
        layout: &LayerPackLayout,
        pack: &LayerWeightPack,
    ) -> Result<LayerWeightPack, StreamInferError> {
        if blob.is_empty() {
            return self.bind_prefetched_slot(orch, scratch, slot, layout, pack);
        }
        let t_map = Instant::now();
        scratch.prepare_host_write(&orch.pool, slot)?;
        self.map_secs += t_map.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let n = blob.len().min(scratch.host_slot(slot).len());
        scratch.host_slot_mut(slot)[..n].copy_from_slice(&blob[..n]);
        self.io_secs += t0.elapsed().as_secs_f64();
        self.bind_prefetched_slot(orch, scratch, slot, layout, pack)
    }

    /// Map next slot and return a Send raw pointer for direct prefetch I/O.
    pub(crate) fn prepare_prefetch_slot(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        slot: usize,
    ) -> Result<PrefetchSlotPtr, StreamInferError> {
        let t_map = Instant::now();
        scratch.prepare_host_write(&orch.pool, slot)?;
        self.map_secs += t_map.elapsed().as_secs_f64();
        let (ptr, len) = scratch.host_slot_ptr_mut(slot);
        Ok(PrefetchSlotPtr::new(ptr, len))
    }

    fn forward_inner(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
        mut scratch: Option<&mut hayai_opencl::StreamingScratch>,
    ) -> Result<Vec<f32>, StreamInferError> {
        let h = self.config.hidden_size;
        self.act_sel = 0;
        self.act_pp[0].fill(0.0);
        let mut embed_buf = vec![0.0f32; h];
        self.embed_row("token_embd.weight", token, h, &mut embed_buf)?;
        self.act_pp[0].copy_from_slice(&embed_buf);

        let pos = self.position;
        let n_layers = self.config.num_layers;
        let eps = self.config.rms_norm_eps;

        type PrefetchOk = (LayerWeightPack, LayerPackLayout);
        let mut current: LayerWeightPack;
        let mut layout: LayerPackLayout;
        if let Some(sc) = scratch.as_mut() {
            let (p, lay) = self.stage_pack(orch, sc, 0, 0)?;
            current = p;
            layout = lay;
        } else {
            current = self.load_pack(0)?;
            layout = current.layout();
        }
        let mut prefetch: Option<JoinHandle<Result<PrefetchOk, GgufError>>> = None;

        for layer_idx in 0..n_layers {
            self.act_swap_prepare();

            let slot = layer_idx % 2;
            // Attn needs host-mapped views; start FFN DMA before Attn (overlap).
            if let Some(sc) = scratch.as_mut() {
                if sc.resident {
                    // Resident: rebind views over this layer's slice of the resident
                    // base — no disk read, no DMA (all layers preloaded once).
                    let t_map = Instant::now();
                    sc.ensure_host_readable(&orch.pool, slot)?;
                    self.map_secs += t_map.elapsed().as_secs_f64();
                    let base = sc.host_slot(layer_idx);
                    (current, layout) =
                        self.catalog.layer_pack_views_from_base(layer_idx, base)?;
                    self.io_bytes += layout.total as u64;
                } else {
                    let t_map = Instant::now();
                    sc.ensure_host_readable(&orch.pool, slot)?;
                    self.map_secs += t_map.elapsed().as_secs_f64();
                    current = current.rebind_views(sc.host_slot(slot), &layout);
                    self.begin_ffn_dma(orch, sc, slot, &layout)?;
                }
            }

            // Prefetch N+1 **directly** into the other ping-pong slot (no heap staging).
            // Resident mode skips prefetch: every layer is already in device memory.
            if !scratch.as_ref().map(|s| s.resident).unwrap_or(false)
                && layer_idx + 1 < n_layers
                && prefetch.is_none()
            {
                let mut cat = self.catalog.fork_reader()?;
                let next = layer_idx + 1;
                if let Some(sc) = scratch.as_mut() {
                    let next_slot = (layer_idx + 1) % 2;
                    let slot_ptr = self.prepare_prefetch_slot(orch, sc, next_slot)?;
                    if let Some(m) = self.owned_mem.as_mut() {
                        m.note_prefetch_staging(0);
                    }
                    prefetch = Some(thread::spawn(move || {
                        let dst = unsafe { slot_ptr.as_mut_slice() };
                        cat.load_layer_pack_into(next, dst)
                    }));
                } else {
                    let cap = self.layer_scratch_cap.max(layout.total).max(1);
                    prefetch = Some(thread::spawn(move || {
                        let mut blob = vec![0u8; cap];
                        let (pack, lay) = cat.load_layer_pack_into(next, &mut blob)?;
                        Ok((pack, lay))
                    }));
                }
            }

            // --- CPU Attention (∥ DMA already enqueued) ---
            let t_attn = Instant::now();
            let mut xn = self.act().to_vec();
            rms_norm(&mut xn, &self.layer_norms[layer_idx].attn_norm, eps);

            let q_dim = self.attn_cfg.hidden_size();
            let kv_dim = self.attn_cfg.kv_dim();
            let mut q = vec![0.0f32; q_dim];
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            current.wq.gemv(&xn, &mut q)?;
            current.wk.gemv(&xn, &mut k)?;
            current.wv.gemv(&xn, &mut v)?;

            let mut attn_out = vec![0.0f32; q_dim];
            attention_decode_step(
                &self.attn_cfg,
                &mut self.kv[layer_idx],
                &mut q,
                &mut k,
                &v,
                pos,
                &mut attn_out,
            );
            if let Some(ref gate_w) = current.attn_gate {
                let mut gate = vec![0.0f32; q_dim];
                gate_w.gemv(&xn, &mut gate)?;
                for i in 0..q_dim {
                    attn_out[i] *= 1.0 / (1.0 + (-gate[i]).exp());
                }
            }
            let mut attn_proj = vec![0.0f32; h];
            current.wo.gemv(&attn_out, &mut attn_proj)?;
            {
                let x = self.act_mut();
                for i in 0..h {
                    x[i] += attn_proj[i];
                }
            }
            self.attn_secs += t_attn.elapsed().as_secs_f64();

            // --- FFN: unmap after Attn, then enqueue async ---
            let mut xn = self.act().to_vec();
            rms_norm(&mut xn, &self.layer_norms[layer_idx].ffn_norm, eps);

            self.ws_gate.fill(0.0);
            self.ws_up.fill(0.0);
            self.ws_down.fill(0.0);

            if let Some(sc) = scratch.as_mut() {
                self.finish_ffn_unmap(orch, sc, slot)?;
            }

            let t_ffn_enq = Instant::now();
            let inflight = ffn_begin_gate_up_scratch(
                orch,
                &current.gate,
                &current.up,
                &xn,
                &mut self.ws_gate,
                &mut self.ws_up,
                &mut self.used_dgpu,
                &mut self.used_apu,
                scratch.as_deref(),
                layer_idx,
                Some(&layout),
            )?;
            self.ffn_secs += t_ffn_enq.elapsed().as_secs_f64();

            let mut next_staged: Option<PrefetchOk> = None;
            if let Some(handle) = prefetch.take() {
                let t_join = Instant::now();
                match handle.join() {
                    Ok(Ok((pack, lay))) => {
                        let dt = t_join.elapsed().as_secs_f64();
                        self.overlap_secs += dt;
                        self.attn_ffn_overlap_secs += dt;
                        self.io_bytes += lay.total as u64;
                        self.prefetch_hits += 1;
                        next_staged = Some((pack, lay));
                    }
                    Ok(Err(e)) => return Err(e.into()),
                    Err(_) => return Err(StreamInferError::Msg("prefetch join panicked".into())),
                }
            }

            let t_ffn_fin = Instant::now();
            ffn_finish_scratch(
                orch,
                inflight,
                &current.down,
                &mut self.ws_gate,
                &mut self.ws_up,
                &mut self.ws_down,
                &mut self.used_dgpu,
                scratch.as_deref(),
                layer_idx,
                Some(&layout),
            )?;
            self.ffn_secs += t_ffn_fin.elapsed().as_secs_f64();

            {
                let sel = self.act_sel;
                for i in 0..h {
                    self.act_pp[sel][i] += self.ws_down[i];
                }
            }
            if let Some((pack, lay)) = next_staged {
                layout = lay;
                if let Some(sc) = scratch.as_mut() {
                    current = self.bind_prefetched_slot(
                        orch,
                        sc,
                        (layer_idx + 1) % 2,
                        &layout,
                        &pack,
                    )?;
                } else {
                    current = pack;
                }
            } else if layer_idx + 1 < n_layers {
                if let Some(sc) = scratch.as_mut() {
                    let (p, lay) = self.stage_pack(orch, sc, (layer_idx + 1) % 2, layer_idx + 1)?;
                    current = p;
                    layout = lay;
                } else {
                    current = self.load_pack(layer_idx + 1)?;
                    layout = current.layout();
                }
            }
        }

        let mut xn = self.act().to_vec();
        rms_norm(&mut xn, &self.output_norm, eps);
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        if self.has_output_weight {
            if let Some(ow) = &self.resident_output {
                // Resident: cached final projection (zero disk I/O).
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

        self.position += 1;
        Ok(logits)
    }

    pub fn generate(
        &mut self,
        orch: &mut EngineOrchestrator,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerateStats, StreamInferError> {
        let prompt_ids = self.tokenizer.encode(prompt, self.tokenizer.add_bos);
        if prompt_ids.is_empty() {
            return Err(StreamInferError::Msg("empty prompt tokenization".into()));
        }
        debug!("stream prompt tokens: {}", prompt_ids.len());

        // Metadata-driven plan: fail only on unknown layer ops (not arch name).
        {
            let n_gpu = orch.pool.len();
            let svm = orch
                .opencl_engine()
                .map(|e| e.device_info.supports_svm)
                .unwrap_or(false);
            match crate::exec_plan::build_exec_plan(&self.catalog, n_gpu, svm) {
                Ok(plan) => {
                    info!(
                        "ExecPlan arch={} units={} max_unit={}KiB ops={:?}",
                        plan.architecture,
                        plan.units.len(),
                        plan.max_unit_bytes / 1024,
                        plan.known_ops.iter().take(12).collect::<Vec<_>>()
                    );
                    for note in &plan.hw_notes {
                        debug!("ExecPlan HW: {note}");
                    }
                    self.exec_plan = Some(plan);
                }
                Err(u) => {
                    return Err(StreamInferError::Msg(format!(
                        "arch-blocked unknown_op tensor={} ({})",
                        u.tensor_name, u.hint
                    )));
                }
            }
        }

        // ── Adaptive Memory Window ──────────────────────────────────────────────────
        // Expert Base+Offset: SVM host on APU + dGPU DMA mirrors + parametric offsets.
        let layer_bytes = self.catalog.max_layer_pack_nbytes()?.max(1);
        self.layer_scratch_cap = layer_bytes;
        let win = compute_window_plan(
            &orch.pool,
            layer_bytes,
            self.config.num_layers,
            self.memory_strategy,
        );
        self.window_plan = Some(win);
        let resident = win.resident && win.k_chunk >= self.config.num_layers;
        info!(
            "AdaptiveWindow: k_chunk={} resident={} window={}  (strategy={:?})",
            win.k_chunk, win.resident,
            crate::metrics::format_bytes(win.window_bytes),
            self.memory_strategy,
        );

        let (xfer, mut scratch) = if resident {
            hayai_opencl::StreamingScratch::allocate_resident_for_pool(
                &orch.pool,
                layer_bytes,
                self.config.num_layers,
            )?
        } else {
            hayai_opencl::StreamingScratch::allocate_for_pool(&orch.pool, layer_bytes)?
        };
        self.transfer_path = Some(xfer);
        let budget = crate::metrics::StreamingMemoryBudget::estimate(
            &self.config,
            layer_bytes,
            win.k_chunk,
            self.kv_sinks,
            self.kv_window,
        );
        self.memory_budget = Some(budget);
        let act_bytes = (self.act_pp[0].len() * 2
            + self.ws_gate.len()
            + self.ws_up.len()
            + self.ws_down.len())
            * 4;
        let mut owned = crate::metrics::HayaiOwnedMemory::default();
        if resident {
            owned.note_scratch(layer_bytes as u64 * self.config.num_layers as u64);
        } else {
            owned.note_scratch(layer_bytes as u64 * 2);
        }
        owned.note_kv(budget.kv_bytes);
        owned.note_activations(act_bytes as u64);
        owned.note_metadata(self.catalog.header_bytes as u64);
        self.owned_mem = Some(owned);
        info!(
            "HeteroScratch {:?} {} — {} KiB × {} host | {} dGPU mirror(s) | budget ~{} | hayai_owned ~{}",
            xfer,
            if resident { "resident" } else { "ping-pong" },
            layer_bytes / 1024,
            if resident { self.config.num_layers } else { 2 },
            scratch.dgpu_mirrors.len(),
            crate::metrics::format_bytes(budget.total_budget_bytes),
            crate::metrics::format_bytes(self.owned_mem.as_ref().map(|m| m.total()).unwrap_or(0))
        );

        // Resident mode: preload every layer pack once (disk → device memory) and
        // cache the final projection + embedding matrices for zero-I/O decode.
        if resident {
            info!(
                "Resident mode: pre-loading all {} layers into device memory",
                self.config.num_layers
            );
            let mut cat = self.catalog.fork_reader()?;
            for i in 0..self.config.num_layers {
                let t0 = Instant::now();
                let layout_total = {
                    let dst = scratch.host_slot_mut(i);
                    let (_, layout) = cat.load_layer_pack_into(i, dst)?;
                    layout.total
                };
                scratch.dma_layer_to_mirrors(&orch.pool, i, layout_total)?;
                self.io_secs += t0.elapsed().as_secs_f64();
            }
            info!("Resident preload complete: {} layers in device memory", self.config.num_layers);
            if self.has_output_weight {
                self.resident_output = Some(self.catalog.load_quant_matrix("output.weight")?);
            }
            self.resident_embed = Some(self.catalog.load_quant_matrix("token_embd.weight")?);
            self.io_bytes += self
                .resident_output
                .as_ref()
                .map(|m| m.nbytes() as u64)
                .unwrap_or(0)
                + self
                    .resident_embed
                    .as_ref()
                    .map(|m| m.nbytes() as u64)
                    .unwrap_or(0);
        }

        let wall0 = Instant::now();
        let prompt_len = prompt_ids.len();

        let mut last_logits;
        let mut all_new = Vec::new();

        if self.config.hrm.is_some() {
            // HRM: each token runs H×(L+1) stack passes (ExecPlan Recurrence).
            last_logits = Vec::new();
            for &tok in &prompt_ids {
                last_logits = self.forward_hrm_token(orch, &mut scratch, tok)?;
            }
            for _ in 0..max_new_tokens {
                let next = sample(&last_logits, self.sampler, &mut self.rng);
                all_new.push(next);
                if next == self.tokenizer.eos_id {
                    break;
                }
                last_logits = self.forward_hrm_token(orch, &mut scratch, next)?;
            }
        } else if self
            .exec_plan
            .as_ref()
            .map(|p| {
                p.known_ops
                    .iter()
                    .any(|o| matches!(o, crate::exec_plan::LayerOpKind::DeltaNet))
            })
            .unwrap_or(false)
        {
            // Qwen3.5 hybrid: route via plan-driven units (full-attn + DeltaNet/SSM).
            last_logits =
                crate::hybrid_infer::prefill_hybrid(self, orch, &prompt_ids, &mut scratch)?;
            for _ in 0..max_new_tokens {
                let next = sample(&last_logits, self.sampler, &mut self.rng);
                all_new.push(next);
                if next == self.tokenizer.eos_id {
                    break;
                }
                last_logits = crate::hybrid_infer::forward_hybrid(self, orch, next, &mut scratch)?;
            }
        } else if crate::gemma_infer::is_gemma4(self) {
            last_logits =
                crate::gemma_infer::prefill_gemma(self, orch, &prompt_ids, &mut scratch)?;
            for _ in 0..max_new_tokens {
                let next = sample(&last_logits, self.sampler, &mut self.rng);
                all_new.push(next);
                if next == self.tokenizer.eos_id {
                    break;
                }
                last_logits = crate::gemma_infer::forward_gemma(self, orch, next, &mut scratch)?;
            }
        } else {
            // Prefill with Attn∥FFN wavefront (PRD §3.3); decode one token at a time.
            last_logits = self.prefill_wavefront(orch, &prompt_ids, &mut scratch)?;
            for _ in 0..max_new_tokens {
                let next = sample(&last_logits, self.sampler, &mut self.rng);
                all_new.push(next);
                if next == self.tokenizer.eos_id {
                    break;
                }
                last_logits = self.forward_staged(orch, next, &mut scratch)?;
            }
        }
        self.wall_compute_secs = wall0.elapsed().as_secs_f64();

        if let (Some(mem), Some(budget)) = (self.owned_mem, self.memory_budget) {
            mem.check_budget(&budget)
                .map_err(StreamInferError::Msg)?;
        }

        Ok(GenerateStats {
            text: self.tokenizer.decode(&all_new),
            prompt_tokens: prompt_len,
            new_tokens: all_new.len(),
            total_positions: self.position,
        })
    }

    /// Forward with layer packs staged into StreamingScratch (DMA/SVM) before FFN.
    fn forward_staged(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
        scratch: &mut hayai_opencl::StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        self.forward_inner(orch, token, Some(scratch))
    }

    pub fn forward(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
    ) -> Result<Vec<f32>, StreamInferError> {
        self.forward_inner(orch, token, None)
    }
}

fn begin_gemv(
    eng: &hayai_opencl::OpenClEngine,
    m: &QuantMatrix,
    xn: &[f32],
) -> Result<PendingGemv, StreamInferError> {
    Ok(crate::orchestrator::begin_gemv_engine(eng, m, xn)?)
}

fn begin_gemv_from_scratch(
    eng: &hayai_opencl::OpenClEngine,
    m: &QuantMatrix,
    xn: &[f32],
    scratch: &hayai_opencl::StreamingScratch,
    layer: usize,
    tensor_off: usize,
) -> Result<PendingGemv, StreamInferError> {
    let (kernel, label) = match m.ggml_type {
        GgmlType::Q4_0 => (&eng.gemv_q4_0, "q4_0"),
        GgmlType::Q4_1 => (&eng.gemv_q4_1, "q4_1"),
        GgmlType::Q8_0 => (&eng.gemv_q8_0, "q8_0"),
        GgmlType::Q5_0 => (&eng.gemv_q5_0, "q5_0"),
        GgmlType::Q5_1 => (&eng.gemv_q5_1, "q5_1"),
        GgmlType::Q2_K => (&eng.gemv_q2_k, "q2_k"),
        GgmlType::Q3_K => (&eng.gemv_q3_k, "q3_k"),
        GgmlType::Q4_K => (&eng.gemv_q4_k, "q4_k"),
        GgmlType::Q5_K => (&eng.gemv_q5_k, "q5_k"),
        GgmlType::Q6_K => (&eng.gemv_q6_k, "q6_k"),
        GgmlType::IQ4_NL => (&eng.gemv_iq4_nl, "iq4_nl"),
        GgmlType::IQ4_XS => (&eng.gemv_iq4_xs, "iq4_xs"),
        GgmlType::IQ3_XXS => (&eng.gemv_iq3_xxs, "iq3_xxs"),
        GgmlType::IQ3_S => (&eng.gemv_iq3_s, "iq3_s"),
        GgmlType::IQ2_XXS => (&eng.gemv_iq2_xxs, "iq2_xxs"),
        GgmlType::IQ2_XS => (&eng.gemv_iq2_xs, "iq2_xs"),
        GgmlType::IQ2_S => (&eng.gemv_iq2_s, "iq2_s"),
        GgmlType::F32 => (&eng.gemv_f32, "f32"),
        GgmlType::F16 => (&eng.gemv_f16, "f16"),
        other => {
            return Err(StreamInferError::Msg(format!(
                "OpenCL GEMV missing for {other:?} (no CPU FFN fallback with GPU pool)"
            )));
        }
    };
    use hayai_opencl::WeightBind;
    let woff = scratch.weight_offset(layer, tensor_off);
    match scratch.weight_bind(eng, layer)? {
        WeightBind::Svm { ptr } => Ok(eng.ggml_gemv_async_from_svm(
            kernel,
            label,
            m.nrows,
            m.ncols,
            ptr,
            woff,
            xn,
        )?),
        WeightBind::Device { buf } => Ok(eng.ggml_gemv_async_from_device(
            kernel,
            label,
            m.nrows,
            m.ncols,
            buf,
            woff,
            xn,
        )?),
    }
}

pub(crate) enum GateUpInflight {
    /// Both devices async — wait both after overlap work.
    Dual {
        gate: PendingGemv,
        up: PendingGemv,
    },
    /// Both already computed on CPU (empty OpenCL pool).
    Done,
}

/// FFN gate∥up across the OpenCL device pool. CPU only if pool is empty.
pub(crate) fn ffn_begin_gate_up_scratch(
    orch: &mut EngineOrchestrator,
    gate: &QuantMatrix,
    up: &QuantMatrix,
    xn: &[f32],
    _gate_out: &mut [f32],
    _up_out: &mut [f32],
    used_dgpu: &mut bool,
    used_apu: &mut bool,
    scratch: Option<&hayai_opencl::StreamingScratch>,
    layer: usize,
    layout: Option<&hayai_model::LayerPackLayout>,
) -> Result<GateUpInflight, StreamInferError> {
    if orch.pool.is_empty() {
        orch.execute_quant_gemv(gate, xn, _gate_out)?;
        orch.execute_quant_gemv(up, xn, _up_out)?;
        return Ok(GateUpInflight::Done);
    }

    let gate_eng = orch.pool.for_role(0);
    let up_eng = orch.pool.for_role(1);
    *used_dgpu = true;
    if orch.pool.len() >= 2 {
        *used_apu = true;
    }

    let g = if let (Some(sc), Some(lay)) = (scratch, layout) {
        begin_gemv_from_scratch(gate_eng, gate, xn, sc, layer, lay.gate_off)?
    } else {
        begin_gemv(gate_eng, gate, xn)?
    };
    let u = if let (Some(sc), Some(lay)) = (scratch, layout) {
        begin_gemv_from_scratch(up_eng, up, xn, sc, layer, lay.up_off)?
    } else {
        begin_gemv(up_eng, up, xn)?
    };
    Ok(GateUpInflight::Dual { gate: g, up: u })
}

/// Finish FFN using Base+Offset scratch for `down` when available.
pub(crate) fn ffn_finish_scratch(
    orch: &mut EngineOrchestrator,
    inflight: GateUpInflight,
    down: &QuantMatrix,
    gate_out: &mut [f32],
    up_out: &mut [f32],
    down_out: &mut [f32],
    used_dgpu: &mut bool,
    scratch: Option<&hayai_opencl::StreamingScratch>,
    layer: usize,
    layout: Option<&hayai_model::LayerPackLayout>,
) -> Result<(), StreamInferError> {
    match inflight {
        GateUpInflight::Dual { gate, up } => {
            let gate_v = gate.wait()?;
            let up_v = up.wait()?;
            gate_out.copy_from_slice(&gate_v);
            up_out.copy_from_slice(&up_v);
        }
        GateUpInflight::Done => {}
    }

    // Default: SiLU(gate) * up (LLaMA / Qwen). Gemma4 overrides via `ffn_finish_gelu`.
    for i in 0..gate_out.len() {
        let g = gate_out[i];
        gate_out[i] = (g / (1.0 + (-g).exp())) * up_out[i];
    }

    if !orch.pool.is_empty() {
        let eng = orch.pool.for_role(2);
        *used_dgpu = true;
        let p = if let (Some(sc), Some(lay)) = (scratch, layout) {
            begin_gemv_from_scratch(eng, down, gate_out, sc, layer, lay.down_off)?
        } else {
            begin_gemv(eng, down, gate_out)?
        };
        let v = p.wait()?;
        down_out.copy_from_slice(&v);
        return Ok(());
    }
    orch.execute_quant_gemv(down, gate_out, down_out)?;
    Ok(())
}

/// Gemma4 FFN: GELU(gate) * up (not SiLU).
pub(crate) fn ffn_finish_gelu_scratch(
    orch: &mut EngineOrchestrator,
    inflight: GateUpInflight,
    down: &QuantMatrix,
    gate_out: &mut [f32],
    up_out: &mut [f32],
    down_out: &mut [f32],
    used_dgpu: &mut bool,
    scratch: Option<&hayai_opencl::StreamingScratch>,
    layer: usize,
    layout: Option<&hayai_model::LayerPackLayout>,
) -> Result<(), StreamInferError> {
    match inflight {
        GateUpInflight::Dual { gate, up } => {
            let gate_v = gate.wait()?;
            let up_v = up.wait()?;
            gate_out.copy_from_slice(&gate_v);
            up_out.copy_from_slice(&up_v);
        }
        GateUpInflight::Done => {}
    }

    // tanh approximation of GELU (same family as ggml_gelu_quick).
    for i in 0..gate_out.len() {
        let x = gate_out[i];
        let x3 = x * x * x;
        let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x3);
        gate_out[i] = 0.5 * x * (1.0 + inner.tanh()) * up_out[i];
    }

    if !orch.pool.is_empty() {
        let eng = orch.pool.for_role(2);
        *used_dgpu = true;
        let p = if let (Some(sc), Some(lay)) = (scratch, layout) {
            begin_gemv_from_scratch(eng, down, gate_out, sc, layer, lay.down_off)?
        } else {
            begin_gemv(eng, down, gate_out)?
        };
        let v = p.wait()?;
        down_out.copy_from_slice(&v);
        return Ok(());
    }
    orch.execute_quant_gemv(down, gate_out, down_out)?;
    Ok(())
}

fn build_attn_config(
    cat: &GgufCatalog,
    config: &ModelConfig,
) -> Result<AttentionConfig, StreamInferError> {
    // Prefer explicit head dim from metadata / first full-attn q tensor (Qwen3.5 ≠ hidden/heads).
    let head_dim_meta = cat
        .meta_u32(&format!("{}.attention.key_length", config.architecture))
        .or_else(|| cat.meta_u32("llama.attention.key_length"))
        .unwrap_or(0) as usize;
    let q_nrows = (0..config.num_layers).find_map(|i| {
        cat.tensor(&format!("blk.{i}.attn_q.weight"))
            .ok()
            .map(|t| t.nrows())
    });
    let head_dim = if head_dim_meta > 0 {
        head_dim_meta
    } else if let Some(qn) = q_nrows {
        (qn / config.num_attention_heads).max(1)
    } else {
        config.hidden_size / config.num_attention_heads
    };
    let mut cfg = AttentionConfig::from_model(
        head_dim * config.num_attention_heads,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.rope_theta,
    );
    cfg.head_dim = head_dim;
    // Partial RoPE (Qwen3.5: rope.dimension_count=64 with head_dim=256).
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

fn load_config(cat: &GgufCatalog) -> Result<ModelConfig, GgufError> {
    let arch = cat.meta_str("general.architecture").unwrap_or("llama");
    let prefix = if arch == "llama" || arch == "qwen2" || arch.contains("smollm") {
        "llama"
    } else {
        arch
    };
    let hidden = cat
        .meta_u32(&format!("{arch}.embedding_length"))
        .or_else(|| cat.meta_u32(&format!("{prefix}.embedding_length")))
        .or_else(|| cat.meta_u32("llama.embedding_length"))
        .or_else(|| cat.meta_u32("qwen2.embedding_length"))
        .or_else(|| cat.tensor("token_embd.weight").ok().map(|t| t.ncols() as u32))
        .ok_or_else(|| GgufError::MissingKey("embedding_length".into()))? as usize;
    let layers_meta = cat
        .meta_u32(&format!("{arch}.block_count"))
        .or_else(|| cat.meta_u32(&format!("{prefix}.block_count")))
        .or_else(|| cat.meta_u32("llama.block_count"))
        .or_else(|| cat.meta_u32("qwen2.block_count"))
        .ok_or_else(|| GgufError::MissingKey("block_count".into()))? as usize;
    // Physical `blk.N` count (HRM metadata may encode H×L×cycles ≫ resident blocks).
    let layers_physical = cat
        .tensors
        .iter()
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut layers_n = if layers_physical > 0 && layers_physical < layers_meta {
        tracing::info!(
            "block_count meta={layers_meta} physical_blk={layers_physical} — using physical"
        );
        layers_physical
    } else {
        layers_meta
    };
    // Qwen3.5 MTP/NextN is an extra trailing block — exclude from main trunk.
    if layers_n > 0
        && cat
            .tensor(&format!("blk.{}.nextn.eh_proj.weight", layers_n - 1))
            .is_ok()
    {
        tracing::info!("excluding NextN/MTP blk.{} from main decode stack", layers_n - 1);
        layers_n -= 1;
    }
    let intermediate = cat
        .meta_u32(&format!("{arch}.feed_forward_length"))
        .or_else(|| cat.meta_u32(&format!("{prefix}.feed_forward_length")))
        .or_else(|| cat.meta_u32("llama.feed_forward_length"))
        .or_else(|| cat.meta_u32("qwen2.feed_forward_length"))
        .unwrap_or((hidden * 8 / 3) as u32) as usize;
    let n_heads = cat
        .meta_u32(&format!("{arch}.attention.head_count"))
        .or_else(|| cat.meta_u32(&format!("{prefix}.attention.head_count")))
        .or_else(|| cat.meta_u32("llama.attention.head_count"))
        .or_else(|| cat.meta_u32("qwen2.attention.head_count"))
        .unwrap_or(8) as usize;
    let n_kv = cat
        .meta_u32(&format!("{arch}.attention.head_count_kv"))
        .or_else(|| cat.meta_u32(&format!("{prefix}.attention.head_count_kv")))
        .or_else(|| cat.meta_u32("llama.attention.head_count_kv"))
        .or_else(|| cat.meta_u32("qwen2.attention.head_count_kv"))
        .unwrap_or(n_heads as u32) as usize;
    let hrm = if arch == "hrm_text" || arch.contains("hrm") {
        let layers_per_stack = cat
            .meta_u32(&format!("{prefix}.layers_per_stack"))
            .unwrap_or((layers_n / 2).max(1) as u32) as usize;
        let h_cycles = cat.meta_u32(&format!("{prefix}.h_cycles")).unwrap_or(2) as usize;
        let l_cycles = cat.meta_u32(&format!("{prefix}.l_cycles")).unwrap_or(3) as usize;
        let embedding_scale = cat
            .meta_f32(&format!("{prefix}.embedding_scale"))
            .unwrap_or(1.0);
        Some(hayai_model::HrmConfig {
            h_cycles,
            l_cycles,
            layers_per_stack,
            embedding_scale,
        })
    } else {
        None
    };
    let tok = cat.tensor("token_embd.weight")?;
    Ok(ModelConfig {
        name: cat
            .meta_str("general.name")
            .unwrap_or("gguf-model")
            .to_string(),
        num_layers: layers_n,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv,
        architecture: arch.to_string(),
        vocab_size: tok.nrows(),
        max_position_embeddings: cat
            .meta_u32(&format!("{prefix}.context_length"))
            .unwrap_or(2048) as usize,
        rope_theta: cat
            .meta_f32(&format!("{prefix}.rope.freq_base"))
            .unwrap_or(10000.0),
        rms_norm_eps: cat
            .meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5),
        hrm,
    })
}
