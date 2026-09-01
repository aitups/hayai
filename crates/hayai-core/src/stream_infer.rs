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
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;
use tracing::{debug, info};

use crate::adaptive_window::{compute_window_plan, MemoryStrategy, WindowPlan};
use crate::infer::{spmm_adj, FfnOverride, GenerateStats, SparseAdj};
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
    /// Gemma4 extra per-block norms/scales — preloaded once at session open so the
    /// decode hot path never issues per-token norm disk reads.
    pub(crate) attn_q_norm: Option<Vec<f32>>,
    pub(crate) attn_k_norm: Option<Vec<f32>>,
    pub(crate) post_attn_norm: Option<Vec<f32>>,
    pub(crate) post_ffw_norm: Option<Vec<f32>>,
    pub(crate) layer_output_scale: Option<f32>,
}

/// Layer-role based model family, derived from **ops/tensors** — never from
/// `general.architecture`. This is what routes each model to its forward path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// Llama-shaped dense blocks (staged ping-pong path).
    Dense,
    /// Per-head q/k norms + GELU FFN + optional per-layer embeddings (Gemma4).
    Gemma,
    /// DeltaNet / SSM mixed blocks (Qwen3.5 hybrid).
    Hybrid,
    /// Router + per-expert FFN units (MoE).
    MoE,
}

impl ModelKind {
    /// Detect from catalog tensor presence (runs before the ExecPlan exists).
    pub fn from_catalog(cat: &GgufCatalog) -> Self {
        let has = |suffix: &str| cat.tensor(&format!("blk.0.{suffix}")).is_ok();
        if has("ssm_out.weight") || has("ssm_a") {
            ModelKind::Hybrid
        } else if has("ffn_gate_inp.weight")
            || has("ffn_exp.0.ffn_gate.weight")
            || has("ffn_shexp.ffn_gate.weight")
        {
            ModelKind::MoE
        } else if has("post_ffw_norm.weight")
            || has("layer_output_scale.weight")
            || has("attn_q_norm.weight")
            || has("attn_k_norm.weight")
        {
            ModelKind::Gemma
        } else {
            ModelKind::Dense
        }
    }
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
    /// Gemma4 proportional RoPE factors (`rope_freqs.weight`), cached once.
    pub(crate) gemma_rope_freqs: Option<Vec<f32>>,
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
    /// Dedicated background I/O worker (Phase H1) for block prefetches.
    pub(crate) io_worker: hayai_io::IoWorker,
    /// Qwen3.5 DeltaNet weights (None entries for full-attn layers).
    pub(crate) deltanet_weights: Option<Vec<Option<crate::deltanet::DeltaNetLayerWeights>>>,
    pub(crate) deltanet_states: Option<Vec<Option<crate::deltanet::DeltaNetState>>>,
    /// MoE expert LRU cache (colibri-style) — host RAM, sized at session start.
    pub(crate) moe_cache: crate::moe_infer::ExpertCache,
}

impl StreamingGenerator {
    /// FFN disperso embebido (D16): ejecuta gate/up/down vía CSR (CPU) cuando el
    /// pack trae bloques sustituidos; los bloques densos usan el orchestrator.
    /// `ov` (override de evolución, Vía B) tiene prioridad sobre el CSR embebido
    /// — agnóstico a arquitectura: lo usan el path Dense y los bloques híbridos.
    pub(crate) fn run_ffn_block(
        &mut self,
        orch: &mut EngineOrchestrator,
        pack: &hayai_model::LayerWeightPack,
        xn: &[f32],
        ov: Option<&FfnOverride>,
    ) -> Result<(), StreamInferError> {
        let gate_csr = ov.and_then(|o| o.gate.as_ref()).or(pack.gate_csr.as_ref());
        if let Some(c) = gate_csr {
            let out = self.spmm_csr(orch, xn, c)?;
            self.ws_gate.copy_from_slice(&out);
        } else {
            orch.execute_quant_gemv(&pack.gate, xn, &mut self.ws_gate)?;
        }
        let up_csr = ov.and_then(|o| o.up.as_ref()).or(pack.up_csr.as_ref());
        if let Some(c) = up_csr {
            let out = self.spmm_csr(orch, xn, c)?;
            self.ws_up.copy_from_slice(&out);
        } else {
            orch.execute_quant_gemv(&pack.up, xn, &mut self.ws_up)?;
        }
        for i in 0..self.ws_gate.len() {
            let g = self.ws_gate[i];
            self.ws_gate[i] = (g / (1.0 + (-g).exp())) * self.ws_up[i];
        }
        let down_csr = ov.and_then(|o| o.down.as_ref()).or(pack.down_csr.as_ref());
        if let Some(c) = down_csr {
            let out = self.spmm_csr(orch, &self.ws_gate, c)?;
            self.ws_down.copy_from_slice(&out);
        } else {
            orch.execute_quant_gemv(&pack.down, &self.ws_gate, &mut self.ws_down)?;
        }
        Ok(())
    }

    /// Aplica los overrides `SparseAdj` (bit-tensor + pesos compartidos por capa)
    /// de un bloque FFN (gate/up/down) de forma **batcheada en GPU** — Fase 2, C1/C4:
    /// `sel` candidatos en un único dispatch sobre `[N×n_pos, d_in]`. Prioriza el
    /// kernel Q4 (dequant en GPU, `w_q4`); cae al F32 (`w`) o CPU `spmm_adj`.
    /// Evita el build/gather del CSR por (candidato, capa, token).
    pub(crate) fn apply_sparse_adj_block(
        &self,
        orch: &EngineOrchestrator,
        ov: &[FfnOverride],
        pick: impl Fn(&FfnOverride) -> Option<&SparseAdj>,
        x_flat: &[f32],
        n: usize,
        n_pos: usize,
        d_in: usize,
        d_out: usize,
        w: Option<&Arc<Vec<f32>>>,
        w_q4: Option<&[u8]>,
        out_flat: &mut [f32],
    ) -> Result<(), StreamInferError> {
        let mut sel: Vec<usize> = Vec::new();
        let mut x_sub: Vec<f32> = Vec::new();
        let mut adjs: Vec<u8> = Vec::new();
        for c in 0..n {
            if let Some(sa) = pick(&ov[c]) {
                sel.push(c);
                x_sub.extend_from_slice(&x_flat[c * n_pos * d_in..(c + 1) * n_pos * d_in]);
                adjs.extend_from_slice(&sa.adjacency);
            }
        }
        if sel.is_empty() {
            return Ok(());
        }
        let mut out_sub = vec![0.0f32; sel.len() * n_pos * d_out];
        if let Some(eng) = orch.opencl_engine() {
            if let Some(q4) = w_q4 {
                eng.spmm_adj_batched_q4gpu(&x_sub, &adjs, q4, sel.len(), n_pos, d_in, d_out, &mut out_sub)?;
            } else if let Some(w0) = w {
                eng.spmm_adj_batched(&x_sub, &adjs, w0, sel.len(), n_pos, d_in, d_out, &mut out_sub)?;
            } else {
                return Ok(());
            }
        } else {
            let Some(_) = w else { return Ok(()) };
            let mut off = 0usize;
            for (i, &c) in sel.iter().enumerate() {
                let sa = pick(&ov[c]).expect("sel consistency");
                let out = spmm_adj(&x_sub[i * n_pos * d_in..(i + 1) * n_pos * d_in], sa);
                out_sub[off..off + n_pos * d_out].copy_from_slice(&out);
                off += n_pos * d_out;
            }
        }
        for (i, &c) in sel.iter().enumerate() {
            out_flat[c * n_pos * d_out..(c + 1) * n_pos * d_out]
                .copy_from_slice(&out_sub[i * n_pos * d_out..(i + 1) * n_pos * d_out]);
        }
        Ok(())
    }

    /// SpMM CSR del FFN disperso: OpenCL si hay pool, si no CPU.
    pub(crate) fn spmm_csr(
        &self,
        orch: &EngineOrchestrator,
        x: &[f32],
        c: &hayai_model::CsrSparse,
    ) -> Result<Vec<f32>, StreamInferError> {
        if let Some(eng) = orch.opencl_engine() {
            let mut out = vec![0.0f32; (x.len() / c.d_in.max(1)) * c.d_out];
            eng.spmm_csr(x, &c.row_ptr, &c.col_idx, &c.vals, c.d_in, c.d_out, &mut out)
                .map_err(|e| StreamInferError::Msg(format!("spmm_csr OpenCL: {e}")))?;
            Ok(out)
        } else {
            Ok(hayai_model::spmm_csr_cpu(
                x,
                &c.row_ptr,
                &c.col_idx,
                &c.vals,
                c.d_in,
                c.d_out,
            ))
        }
    }

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
            // Gemma4 extra per-block norms/scales: read once here so the decode
            // hot path never issues per-token norm disk reads.
            let attn_q_norm = catalog
                .dequant_f32(&format!("blk.{i}.attn_q_norm.weight"))
                .ok();
            let attn_k_norm = catalog
                .dequant_f32(&format!("blk.{i}.attn_k_norm.weight"))
                .ok();
            let post_attn_norm = catalog
                .dequant_f32(&format!("blk.{i}.post_attention_norm.weight"))
                .ok();
            let post_ffw_norm = catalog
                .dequant_f32(&format!("blk.{i}.post_ffw_norm.weight"))
                .ok();
            let layer_output_scale = catalog
                .dequant_f32(&format!("blk.{i}.layer_output_scale.weight"))
                .ok()
                .map(|v| v.first().copied().unwrap_or(1.0));
            layer_norms.push(LayerNorms {
                attn_norm,
                ffn_norm,
                attn_q_norm,
                attn_k_norm,
                post_attn_norm,
                post_ffw_norm,
                layer_output_scale,
            });
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
        let kv = match ModelKind::from_catalog(&catalog) {
            ModelKind::Gemma => {
                crate::gemma_infer::build_layer_kv_caches(&catalog, &config, sink, window)?
            }
            ModelKind::Hybrid => {
                // Per full-attn layer dims from tensors (not a single 4B-shaped AttentionConfig).
                crate::layer_cfg::build_hybrid_kv_caches(&catalog, &config, sink, window)?
            }
            _ => (0..kv_slots)
                .map(|_| LayerKvCache::new(attn_cfg.num_kv_heads, attn_cfg.head_dim, sink, window))
                .collect(),
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
        // Gemma4 proportional RoPE factors — loaded once, not per token.
        let gemma_rope_freqs = catalog.dequant_f32("rope_freqs.weight").ok();

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
            gemma_rope_freqs,
            exec_plan: None,
            io_worker: hayai_io::IoWorker::new("hayai-io-worker"),
            deltanet_weights: None,
            deltanet_states: None,
            moe_cache: crate::moe_infer::ExpertCache::new(0),
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

    /// MoE expert LRU cache hit/miss counters (host RAM, colibri-style).
    pub fn moe_cache_stats(&self) -> (usize, usize) {
        (self.moe_cache.hits, self.moe_cache.misses)
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
    /// Macro-chunk: loads the whole `block_k` block once per slot, then returns
    /// views over this layer's slice for every layer of the block.
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
        if scratch.block_k > 1 {
            let k = scratch.block_k;
            let block = layer / k;
            let bslot = scratch.slot_for(layer);
            let bs = block * k;
            let be = (bs + k).min(self.config.num_layers);
            // Un bloque que contenga capas DeltaNet/SSM (híbrido Qwen3.5) no se
            // puede cargar como packs llama (no tienen attn_q/k/v separados): cae
            // al ping-pong por capa (las capas deltanet viven en su propio cache).
            let has_hybrid = (bs..be)
                .any(|l| crate::deltanet::is_deltanet_layer(&self.catalog, l));
            if !has_hybrid && scratch.block_staged[bslot] != block + 1 {
                // Load the whole block into this slot once.
                let t_map = Instant::now();
                scratch.prepare_host_write(&orch.pool, layer)?;
                self.map_secs += t_map.elapsed().as_secs_f64();
                let mut cat = self.catalog.fork_reader()?;
                for l in bs..be {
                    let t0 = Instant::now();
                    let dst = scratch.host_slot_mut(l);
                    let (_, layout) = cat.load_layer_pack_into(l, dst)?;
                    self.io_secs += t0.elapsed().as_secs_f64();
                    self.io_bytes += layout.total as u64;
                }
                scratch.mark_block_staged(bslot, block);
            }
            if !has_hybrid {
                let t0 = Instant::now();
                let base = scratch.host_slot(layer);
                let (pack, layout) = self.catalog.layer_pack_views_from_base(layer, base)?;
                self.io_secs += t0.elapsed().as_secs_f64();
                self.io_bytes += layout.total as u64;
                return Ok((pack, layout));
            }
            // has_hybrid: continúa al ping-pong por capa de abajo.
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
        override_ffn: Option<&[FfnOverride]>,
    ) -> Result<Vec<f32>, StreamInferError> {
        let h = self.config.hidden_size;
        self.act_sel = 0;
        self.act_pp[0].fill(0.0);
        let mut embed_buf = vec![0.0f32; h];
        self.embed_row("token_embd.weight", token, h, &mut embed_buf)?;
        self.act_pp[0].copy_from_slice(&embed_buf);

        // Macro-chunk decode: blocks of `block_k` layers per I/O batch.
        if let Some(sc) = scratch.as_mut() {
            if sc.block_k > 1 && !sc.resident && override_ffn.is_none() {
                return self.forward_macro_chunk(orch, sc);
            }
        }

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
            let sparse_ffn = current.gate_csr.is_some()
                || current.up_csr.is_some()
                || current.down_csr.is_some();
            let ov = override_ffn.and_then(|o| o.get(layer_idx));
            let has_ov = match ov {
                Some(o) => o.gate.is_some() || o.up.is_some() || o.down.is_some(),
                None => false,
            };
            let inflight = if has_ov || sparse_ffn {
                None
            } else {
                Some(ffn_begin_gate_up_scratch(
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
                )?)
            };
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
            if let Some(inflight) = inflight {
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
            } else {
                // FFN disperso embebido (D16) u override de evolución: CSR en GPU/CPU.
                self.run_ffn_block(orch, &current, &xn, ov)?;
            }
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

    /// Macro-chunk decode: process blocks of `block_k` layers, loading each block
    /// once per token (one prefetch thread per block instead of per layer).
    fn forward_macro_chunk(
        &mut self,
        orch: &mut EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        let h = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let n_layers = self.config.num_layers;
        let k = scratch.block_k.max(1);
        let n_blocks = n_layers.div_ceil(k);
        let pos = self.position;

        for block in 0..n_blocks {
            let bs = block * k;
            let be = (bs + k).min(n_layers);

            // Ensure this block is staged in its ping-pong slot.
            self.stage_block(orch, scratch, bs, be)?;

            // Prefetch the next block on the dedicated I/O worker (overlaps compute).
            let mut prefetch: Option<std::sync::mpsc::Receiver<Result<(), GgufError>>> = None;
            if block + 1 < n_blocks {
                let next_bs = (block + 1) * k;
                let next_be = (next_bs + k).min(n_layers);
                let next_slot = (block + 1) % 2;
                let t_map = Instant::now();
                scratch.prepare_host_write(&orch.pool, next_bs)?;
                self.map_secs += t_map.elapsed().as_secs_f64();
                let (ptr, _len) = scratch.host_slot_block_ptr_mut(next_slot);
                let slot_ptr = PrefetchSlotPtr::new(ptr, scratch.slot_capacity());
                let stride = scratch.resident_stride;
                let mut cat = self.catalog.fork_reader()?;
                prefetch = self.io_worker.run(move || -> Result<(), GgufError> {
                    let dst = unsafe { slot_ptr.as_mut_slice() };
                    for l in next_bs..next_be {
                        let off = (l - next_bs) * stride;
                        let end = (off + stride).min(dst.len());
                        cat.load_layer_pack_into(l, &mut dst[off..end])?;
                    }
                    Ok(())
                });
            }

            // Process every layer of the block.
            for layer in bs..be {
                self.act_swap_prepare();

                let t_map = Instant::now();
                scratch.ensure_host_readable(&orch.pool, layer)?;
                self.map_secs += t_map.elapsed().as_secs_f64();
                let base = scratch.host_slot(layer);
                let (current, layout) =
                    self.catalog.layer_pack_views_from_base(layer, base)?;

                // ── CPU Attention ────────────────────────────────────────────────
                let t_attn = Instant::now();
                let mut xn = self.act().to_vec();
                rms_norm(&mut xn, &self.layer_norms[layer].attn_norm, eps);

                let q_dim = self.attn_cfg.hidden_size();
                let kv_dim = self.attn_cfg.kv_dim();
                let mut q = vec![0.0f32; q_dim];
                let mut kk = vec![0.0f32; kv_dim];
                let mut v = vec![0.0f32; kv_dim];
                current.wq.gemv(&xn, &mut q)?;
                current.wk.gemv(&xn, &mut kk)?;
                current.wv.gemv(&xn, &mut v)?;

                let mut attn_out = vec![0.0f32; q_dim];
                attention_decode_step(
                    &self.attn_cfg,
                    &mut self.kv[layer],
                    &mut q,
                    &mut kk,
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


                // ── FFN ──────────────────────────────────────────────────────────
                let mut xn = self.act().to_vec();
                rms_norm(&mut xn, &self.layer_norms[layer].ffn_norm, eps);

                self.ws_gate.fill(0.0);
                self.ws_up.fill(0.0);
                self.ws_down.fill(0.0);
                self.finish_ffn_unmap(orch, scratch, layer)?;

                let t_ffn = Instant::now();
                let sparse_ffn = current.gate_csr.is_some()
                    || current.up_csr.is_some()
                    || current.down_csr.is_some();
                if sparse_ffn {
                    // FFN disperso embebido (D16): gate/up/down vía CSR en CPU.
                    self.run_ffn_block(orch, &current, &xn, None)?;
                } else {
                    let inflight = ffn_begin_gate_up_scratch(
                        orch,
                        &current.gate,
                        &current.up,
                        &xn,
                        &mut self.ws_gate,
                        &mut self.ws_up,
                        &mut self.used_dgpu,
                        &mut self.used_apu,
                        Some(scratch),
                        layer,
                        Some(&layout),
                    )?;
                    ffn_finish_scratch(
                        orch,
                        inflight,
                        &current.down,
                        &mut self.ws_gate,
                        &mut self.ws_up,
                        &mut self.ws_down,
                        &mut self.used_dgpu,
                        Some(scratch),
                        layer,
                        Some(&layout),
                    )?;
                }
                self.ffn_secs += t_ffn.elapsed().as_secs_f64();

                {
                    let sel = self.act_sel;
                    for i in 0..h {
                        self.act_pp[sel][i] += self.ws_down[i];
                    }
                }
            }

            if let Some(rx) = prefetch {
                match rx.recv() {
                    Ok(Ok(())) => {
                        scratch.mark_block_staged((block + 1) % 2, block + 1);
                        self.prefetch_hits += 1;
                    }
                    Ok(Err(e)) => return Err(e.into()),
                    Err(_) => {
                        return Err(StreamInferError::Msg("block prefetch worker dropped".into()))
                    }
                }
            }
        }


        // Final projection (resident cache or on-demand).
        let mut xn = self.act().to_vec();
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

        self.position += 1;
        Ok(logits)
    }

    /// Load block `[bs, be)` into its ping-pong slot if not already staged.
    fn stage_block(
        &mut self,
        orch: &EngineOrchestrator,
        scratch: &mut hayai_opencl::StreamingScratch,
        bs: usize,
        be: usize,
    ) -> Result<(), StreamInferError> {
        let block = bs / scratch.block_k.max(1);
        let bslot = scratch.slot_for(bs);
        if scratch.block_staged[bslot] == block + 1 {
            return Ok(());
        }
        let t_map = Instant::now();
        scratch.prepare_host_write(&orch.pool, bs)?;
        self.map_secs += t_map.elapsed().as_secs_f64();
        let mut cat = self.catalog.fork_reader()?;
        for l in bs..be {
            let t0 = Instant::now();
            let dst = scratch.host_slot_mut(l);
            let (_, layout) = cat.load_layer_pack_into(l, dst)?;
            self.io_secs += t0.elapsed().as_secs_f64();
            self.io_bytes += layout.total as u64;
        }
        scratch.mark_block_staged(bslot, block);
        Ok(())
    }

    /// Prepare the generation session: build the ExecPlan, allocate the streaming
    /// or resident scratch, and (in resident mode) preload every layer pack into
    /// accelerator memory. Returns the session scratch.
    ///
    /// Call once per request before [`Self::prefill`].
    pub fn prepare_session(
        &mut self,
        orch: &mut EngineOrchestrator,
    ) -> Result<hayai_opencl::StreamingScratch, StreamInferError> {
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

        // MoE expert LRU cache (colibri): byte budget `HAYAI_MOE_CACHE_MB` (default
        // 512 MiB, 0 disables) → slots = min(n_layers×top_k, budget / max_expert_pack).
        // Hot experts stay resident → fewer disk reads/token on edge devices.
        let cache_mb: usize = std::env::var("HAYAI_MOE_CACHE_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let (cache_slots, cache_bytes) = self
            .exec_plan
            .as_ref()
            .and_then(|p| p.moe.map(|m| (p, m)))
            .map(|(p, m)| {
                let max_expert = p
                    .units
                    .iter()
                    .map(|u| u.max_expert_bytes)
                    .max()
                    .unwrap_or(0);
                let desired = self.config.num_layers.saturating_mul(m.top_k).max(1);
                if cache_mb == 0 || max_expert == 0 {
                    (0, 0)
                } else {
                    let slots = (cache_mb * 1024 * 1024 / max_expert).min(desired).max(1);
                    (slots, slots * max_expert)
                }
            })
            .unwrap_or((0, 0));
        self.moe_cache = crate::moe_infer::ExpertCache::new(cache_slots);
        if cache_slots > 0 {
            tracing::info!(
                "MoE expert LRU cache: {cache_slots} packs (~{} MiB host RAM)",
                cache_bytes / (1024 * 1024)
            );
        }

        // ── Adaptive Memory Window ──────────────────────────────────────────────────
        // Plan-driven sizing: streaming uses the sparse unit window (MoE: attn/router
        // + top_k × expert), while resident preload needs the full block (all experts).
        let window_bytes = self
            .exec_plan
            .as_ref()
            .map(|p| p.max_unit_bytes)
            .unwrap_or(self.catalog.max_layer_pack_nbytes().unwrap_or(0))
            .max(1);
        let full_bytes = self
            .exec_plan
            .as_ref()
            .map(|p| p.max_full_block_bytes)
            .unwrap_or(window_bytes)
            .max(1);
        self.layer_scratch_cap = window_bytes;
        let win = compute_window_plan(
            &orch.pool,
            full_bytes,
            self.config.num_layers,
            self.memory_strategy,
        );
        self.window_plan = Some(win);
        let resident = win.resident && win.k_chunk >= self.config.num_layers;
        let layer_bytes = if resident { full_bytes } else { window_bytes };
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
            hayai_opencl::StreamingScratch::allocate_for_pool(&orch.pool, layer_bytes, win.k_chunk)?
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
            let plan = self.exec_plan.clone();
            for i in 0..self.config.num_layers {
                // Hybrid (Qwen3.5 DeltaNet) layers keep their weights in the
                // DeltaNetLayerWeights cache (ensure_deltanet_cache) and their
                // scratch slot is not used by the generic llama pack.
                if crate::deltanet::is_deltanet_layer(&self.catalog, i) {
                    continue;
                }
                let t0 = Instant::now();
                let layout_total = {
                    let dst = scratch.host_slot_mut(i);
                    if let Some(p) = &plan {
                        if let Some(unit) = p.units.iter().find(|u| u.block_id == Some(i)) {
                            if !unit.experts.is_empty() {
                                // MoE: load the full block into the resident slot at the
                                // PLAN's compact layout. Non-expert tensors (attn/router/
                                // shared expert) go at COMPACT offsets 0..non_expert_bytes
                                // (matching the reader's `view_of`), then each expert at
                                // its fixed slot `non_expert_bytes + eid*max_expert_bytes`.
                                // Writing the fused 3D tensors at catalog order would
                                // MISALIGN any non-expert tensor that follows them in the
                                // catalog and corrupt the expert slots.
                                let mut specs: Vec<(&str, usize, usize, usize)> = Vec::new();
                                let mut off = 0usize;
                                for t in unit
                                    .tensors
                                    .iter()
                                    .filter(|t| !crate::exec_plan::is_expert_op(t.op))
                                {
                                    specs.push((t.name.as_str(), 0, off, t.nbytes));
                                    off += t.nbytes;
                                }
                                for e in &unit.experts {
                                    let base_off = unit.non_expert_bytes
                                        + e.expert_id * unit.max_expert_bytes;
                                    for t in &e.tensors {
                                        specs.push((
                                            t.name.as_str(),
                                            t.src_off,
                                            base_off + t.offset,
                                            t.nbytes,
                                        ));
                                    }
                                }
                                cat.load_tensors_into(&specs, dst)?;
                                unit.non_expert_bytes
                                    + unit.experts.len() * unit.max_expert_bytes
                            } else {
                                let (_, layout) = cat.load_layer_pack_into(i, dst)?;
                                layout.total
                            }
                        } else {
                            let (_, layout) = cat.load_layer_pack_into(i, dst)?;
                            layout.total
                        }
                    } else {
                        let (_, layout) = cat.load_layer_pack_into(i, dst)?;
                        layout.total
                    }
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

        Ok(scratch)
    }


    /// Layer-role derived model kind (op-based, never `general.architecture`).
    pub fn model_kind(&self) -> ModelKind {
        // Catálogo primero: los tensores `ssm_*` pueden no aparecer en los units
        // del plan (el clasificador `AttnQkv` gana al de `DeltaNet` en el orden de
        // fases), pero definen un híbrido Qwen3.5 y deben enrutar a `forward_hybrid`.
        for i in 0..self.config.num_layers.min(16) {
            if crate::deltanet::is_deltanet_layer(&self.catalog, i) {
                return ModelKind::Hybrid;
            }
        }
        if let Some(p) = &self.exec_plan {
            let has = |op: crate::LayerOpKind| p.known_ops.contains(&op);
            if has(crate::LayerOpKind::DeltaNet) {
                ModelKind::Hybrid
            } else if has(crate::LayerOpKind::Router)
                || has(crate::LayerOpKind::ExpertGate)
                || has(crate::LayerOpKind::SharedExpert)
            {
                ModelKind::MoE
            } else if has(crate::LayerOpKind::AttnQNorm)
                || has(crate::LayerOpKind::AttnKNorm)
                || has(crate::LayerOpKind::PostFfnNorm)
                || has(crate::LayerOpKind::LayerOutputScale)
            {
                ModelKind::Gemma
            } else {
                ModelKind::Dense
            }
        } else {
            ModelKind::Dense
        }
    }

    /// Prefill `prompt_ids` and return logits ready for sampling (family dispatch:
    /// HRM / Qwen3.5 hybrid / Gemma4 / llama). Advances the KV cache over the prompt.
    pub fn prefill(
        &mut self,
        orch: &mut EngineOrchestrator,
        prompt_ids: &[u32],
        scratch: &mut hayai_opencl::StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        if prompt_ids.is_empty() {
            return Err(StreamInferError::Msg("empty prompt tokenization".into()));
        }
        if self.config.hrm.is_some() {
            // HRM: each prompt token runs H×(L+1) stack passes (ExecPlan Recurrence).
            let mut last = Vec::new();
            for &tok in prompt_ids {
                last = self.forward_hrm_token(orch, scratch, tok)?;
            }
            Ok(last)
        } else {
            match self.model_kind() {
                ModelKind::Hybrid => {
                    crate::hybrid_infer::prefill_hybrid(self, orch, prompt_ids, scratch)
                }
                ModelKind::Gemma => {
                    crate::gemma_infer::prefill_gemma(self, orch, prompt_ids, scratch)
                }
                ModelKind::MoE => {
                    crate::moe_infer::prefill_moe(self, orch, prompt_ids, scratch)
                }
                ModelKind::Dense => {
                    // Prefill with Attn∥FFN wavefront (PRD §3.3).
                    self.prefill_wavefront(orch, prompt_ids, scratch)
                }
            }
        }
    }

    /// One decode step for `token`: run the model forward (family dispatch) and
    /// return the next logits. Advances `self.position` by one.
    pub fn decode_step(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
        scratch: &mut hayai_opencl::StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        if self.config.hrm.is_some() {
            self.forward_hrm_token(orch, scratch, token)
        } else {
            match self.model_kind() {
                ModelKind::Hybrid => crate::hybrid_infer::forward_hybrid(self, orch, token, scratch, None),
                ModelKind::Gemma => crate::gemma_infer::forward_gemma(self, orch, token, scratch),
                ModelKind::MoE => crate::moe_infer::forward_moe(self, orch, token, scratch),
                ModelKind::Dense => self.forward_staged(orch, token, scratch),
            }
        }
    }


    pub fn decode_step_with_override(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
        scratch: &mut hayai_opencl::StreamingScratch,
        override_ffn: &[FfnOverride],
    ) -> Result<Vec<f32>, StreamInferError> {
        if self.config.hrm.is_some() {
            return Err(StreamInferError::Msg(
                "decode_step_with_override: HRM sin soporte de override todavía".into(),
            ));
        }
        match self.model_kind() {
            ModelKind::Hybrid => {
                crate::hybrid_infer::forward_hybrid(self, orch, token, scratch, Some(override_ffn))
            }
            ModelKind::Dense => self.forward_inner(orch, token, Some(scratch), Some(override_ffn)),
            other => Err(StreamInferError::Msg(format!(
                "decode_step_with_override: ModelKind {} sin override (Dense/Hybrid soportados)",
                format!("{other:?}")
            ))),
        }
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
        let prompt_len = prompt_ids.len();

        let mut scratch = self.prepare_session(orch)?;
        let wall0 = Instant::now();
        let mut last_logits = self.prefill(orch, &prompt_ids, &mut scratch)?;
        let mut all_new = Vec::new();
        for _ in 0..max_new_tokens {
            let next = sample(&last_logits, self.sampler, &mut self.rng);
            all_new.push(next);
            if next == self.tokenizer.eos_id {
                break;
            }
            last_logits = self.decode_step(orch, next, &mut scratch)?;
        }
        self.wall_compute_secs = wall0.elapsed().as_secs_f64();

        if let (Some(mem), Some(budget)) = (self.owned_mem, self.memory_budget) {
            // El híbrido Qwen3.5 mantiene el cache DeltaNet/SSM (~1 GB) fuera del
            // estimado de capa: la verificación de presupuesto es orientativa.
            if self.model_kind() != ModelKind::Hybrid {
                mem.check_budget(&budget)
                    .map_err(StreamInferError::Msg)?;
            }
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
        self.forward_inner(orch, token, Some(scratch), None)
    }

    pub fn forward(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
    ) -> Result<Vec<f32>, StreamInferError> {
        self.forward_inner(orch, token, None, None)
    }

    /// Forward Dense con override de FFN por capa (Vía B) — sin re-embeder.
    /// Requiere `MemoryStrategy::Minimal` (block_k = 1).
    pub fn forward_with_override(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
        override_ffn: &[FfnOverride],
    ) -> Result<Vec<f32>, StreamInferError> {
        self.forward_inner(orch, token, None, Some(override_ffn))
    }

    /// Forward batcheado **agnóstico a arquitectura**: despacha Dense → [`Self::forward_batched`]
    /// y Hybrid → `hybrid_infer::forward_batched_hybrid` (estado KV + DeltaNet por
    /// candidato). `scratch` se requiere solo para el path Hybrid.
    pub fn forward_batched_any(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
        pos: usize,
        kv: &mut [Vec<LayerKvCache>],
        n_candidates: usize,
        mut get_override: impl FnMut(usize, usize) -> FfnOverride,
        scratch: &mut hayai_opencl::StreamingScratch,
    ) -> Result<Vec<Vec<f32>>, StreamInferError> {
        match self.model_kind() {
            ModelKind::Hybrid => {
                // Crea un estado DeltaNet fresco por candidato (solo capas deltanet).
                crate::hybrid_infer::ensure_deltanet_cache(self)?;
                let mut dn: Vec<Vec<Option<crate::deltanet::DeltaNetState>>> = Vec::with_capacity(n_candidates);
                for _ in 0..n_candidates {
                    let mut per_layer: Vec<Option<crate::deltanet::DeltaNetState>> = Vec::with_capacity(self.config.num_layers);
                    for layer in 0..self.config.num_layers {
                        if crate::deltanet::is_deltanet_layer(&self.catalog, layer) {
                            let w = self
                                .deltanet_weights
                                .as_ref()
                                .and_then(|v| v[layer].as_ref())
                                .ok_or_else(|| StreamInferError::Msg("deltanet weights ausentes".into()))?;
                            per_layer.push(Some(crate::deltanet::DeltaNetState::new(
                                w.conv_k, w.conv_dim, w.n_v_heads, w.head_k, w.head_v,
                            )));
                        } else {
                            per_layer.push(None);
                        }
                    }
                    dn.push(per_layer);
                }
                crate::hybrid_infer::forward_batched_hybrid_gemm(
                    self, orch, token, pos, scratch, kv, &mut dn, n_candidates, get_override,
                )
            }
            ModelKind::Dense => self.forward_batched(orch, token, pos, kv, n_candidates, get_override),
            other => Err(StreamInferError::Msg(format!(
                "forward_batched_any: ModelKind {} sin batch (Dense/Hybrid soportados)",
                format!("{other:?}")
            ))),
        }
    }

    /// Forward **por capas** agnóstico a arquitectura (layer-major): procesa TODA la
    /// secuencia `tokens` en una sola llamada. El override se construye UNA vez por
    /// (candidato, capa) y las proyecciones + FFN se batchean sobre `[N×n_pos]`.
    pub fn forward_batched_any_seq(
        &mut self,
        orch: &mut EngineOrchestrator,
        tokens: &[u32],
        kv: &mut [Vec<LayerKvCache>],
        n_candidates: usize,
        mut get_override: impl FnMut(
            usize,
            usize,
            &Option<Arc<Vec<f32>>>,
            &Option<Arc<Vec<f32>>>,
            &Option<Arc<Vec<f32>>>,
        ) -> FfnOverride,
    ) -> Result<Vec<Vec<f32>>, StreamInferError> {
        match self.model_kind() {
            ModelKind::Hybrid => {
                crate::hybrid_infer::ensure_deltanet_cache(self)?;
                let mut dn: Vec<Vec<Option<crate::deltanet::DeltaNetState>>> = Vec::with_capacity(n_candidates);
                for _ in 0..n_candidates {
                    let mut per_layer: Vec<Option<crate::deltanet::DeltaNetState>> = Vec::with_capacity(self.config.num_layers);
                    for layer in 0..self.config.num_layers {
                        if crate::deltanet::is_deltanet_layer(&self.catalog, layer) {
                            let w = self
                                .deltanet_weights
                                .as_ref()
                                .and_then(|v| v[layer].as_ref())
                                .ok_or_else(|| StreamInferError::Msg("deltanet weights ausentes".into()))?;
                            per_layer.push(Some(crate::deltanet::DeltaNetState::new(
                                w.conv_k, w.conv_dim, w.n_v_heads, w.head_k, w.head_v,
                            )));
                        } else {
                            per_layer.push(None);
                        }
                    }
                    dn.push(per_layer);
                }
                crate::hybrid_infer::forward_batched_hybrid_seq(
                    self, orch, tokens, kv, &mut dn, n_candidates, get_override,
                )
            }
            ModelKind::Dense => self.forward_batched_seq(orch, tokens, kv, n_candidates, get_override),
            other => Err(StreamInferError::Msg(format!(
                "forward_batched_any_seq: ModelKind {} sin seq (Dense/Hybrid soportados)",
                format!("{other:?}")
            ))),
        }
    }

    /// Forward **batcheado** de N candidatos (Fase 2, path Dense — llama/ALIA):
    /// un único paso por capa con override de FFN por (candidato, capa). Las
    /// proyecciones de atención y el FFN denso usan **GEMM batcheado** (un
    /// dispatch `[N×M]`, pesos leídos una vez — criterios C1/C4). `get_override`
    /// construye el CSR al vuelo y se libera al terminar la capa (RAM acotada).
    ///
    /// `pos` y `kv` (por candidato, por capa) son **persistentes**: en
    /// teacher-forcing el token `t` atiende a su historia (KV acumulada).
    pub fn forward_batched(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
        pos: usize,
        kv: &mut [Vec<LayerKvCache>],
        n_candidates: usize,
        mut get_override: impl FnMut(usize, usize) -> FfnOverride,
    ) -> Result<Vec<Vec<f32>>, StreamInferError> {
        let h = self.config.hidden_size;
        let n_layers = self.config.num_layers;
        let eps = self.config.rms_norm_eps;
        let q_dim = self.attn_cfg.hidden_size();
        let kv_dim = self.attn_cfg.kv_dim();
        let ff = self.config.intermediate_size;
        if kv.len() < n_candidates {
            return Err(StreamInferError::Msg(format!(
                "forward_batched: kv tiene {} filas, se necesitan {n_candidates}",
                kv.len()
            )));
        }
        let n = n_candidates;

        // Activaciones por candidato (mismo token).
        let mut x: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut emb = vec![0.0f32; h];
            self.embed_row("token_embd.weight", token, h, &mut emb)?;
            x.push(emb);
        }
        let mut x_flat = vec![0.0f32; n * h];
        let mut q_flat = vec![0.0f32; n * q_dim];
        let mut k_flat = vec![0.0f32; n * kv_dim];
        let mut v_flat = vec![0.0f32; n * kv_dim];
        let mut attn_out_flat = vec![0.0f32; n * q_dim];
        let mut attn_proj_flat = vec![0.0f32; n * h];
        let mut gate_flat = vec![0.0f32; n * ff];
        let mut up_flat = vec![0.0f32; n * ff];
        let mut down_flat = vec![0.0f32; n * h];

        for layer_idx in 0..n_layers {
            let pack = self.load_pack(layer_idx)?;
            let ov: Vec<FfnOverride> = (0..n).map(|c| get_override(c, layer_idx)).collect();

            // ── Atención: norm + proyecciones batcheadas + núcleo por candidato.
            for c in 0..n {
                x_flat[c * h..(c + 1) * h].copy_from_slice(&x[c]);
                rms_norm(
                    &mut x_flat[c * h..(c + 1) * h],
                    &self.layer_norms[layer_idx].attn_norm,
                    eps,
                );
            }
            orch.execute_quant_gemv_batched(&pack.wq, &x_flat, &mut q_flat, n)?;
            orch.execute_quant_gemv_batched(&pack.wk, &x_flat, &mut k_flat, n)?;
            orch.execute_quant_gemv_batched(&pack.wv, &x_flat, &mut v_flat, n)?;
            for c in 0..n {
                attention_decode_step(
                    &self.attn_cfg,
                    &mut kv[c][layer_idx],
                    &mut q_flat[c * q_dim..(c + 1) * q_dim],
                    &mut k_flat[c * kv_dim..(c + 1) * kv_dim],
                    &v_flat[c * kv_dim..(c + 1) * kv_dim],
                    pos,
                    &mut attn_out_flat[c * q_dim..(c + 1) * q_dim],
                );
            }
            if let Some(ref gate_w) = pack.attn_gate {
                let mut ag = vec![0.0f32; n * q_dim];
                orch.execute_quant_gemv_batched(gate_w, &x_flat, &mut ag, n)?;
                for i in 0..n * q_dim {
                    attn_out_flat[i] *= 1.0 / (1.0 + (-ag[i]).exp());
                }
            }
            orch.execute_quant_gemv_batched(&pack.wo, &attn_out_flat, &mut attn_proj_flat, n)?;
            for c in 0..n {
                for i in 0..h {
                    x[c][i] += attn_proj_flat[c * h + i];
                }
            }

            // ── FFN: norm + gate/up denso batcheado + override CSR + down.
            for c in 0..n {
                x_flat[c * h..(c + 1) * h].copy_from_slice(&x[c]);
                rms_norm(
                    &mut x_flat[c * h..(c + 1) * h],
                    &self.layer_norms[layer_idx].ffn_norm,
                    eps,
                );
            }
            let has_any_ov = ov
                .iter()
                .any(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some());
            orch.execute_quant_gemv_batched(&pack.gate, &x_flat, &mut gate_flat, n)?;
            orch.execute_quant_gemv_batched(&pack.up, &x_flat, &mut up_flat, n)?;
            if has_any_ov {
                for c in 0..n {
                    if let Some(cs) = ov[c].gate.as_ref() {
                        let out = self.spmm_csr(orch, &x_flat[c * h..(c + 1) * h], cs)?;
                        gate_flat[c * ff..(c + 1) * ff].copy_from_slice(&out);
                    }
                    if let Some(cs) = ov[c].up.as_ref() {
                        let out = self.spmm_csr(orch, &x_flat[c * h..(c + 1) * h], cs)?;
                        up_flat[c * ff..(c + 1) * ff].copy_from_slice(&out);
                    }
                }
            }
            for i in 0..n * ff {
                let g = gate_flat[i];
                gate_flat[i] = (g / (1.0 + (-g).exp())) * up_flat[i];
            }
            orch.execute_quant_gemv_batched(&pack.down, &gate_flat, &mut down_flat, n)?;
            if has_any_ov {
                for c in 0..n {
                    if let Some(cs) = ov[c].down.as_ref() {
                        let out = self.spmm_csr(orch, &gate_flat[c * ff..(c + 1) * ff], cs)?;
                        down_flat[c * h..(c + 1) * h].copy_from_slice(&out);
                    }
                }
            }
            for c in 0..n {
                for i in 0..h {
                    x[c][i] += down_flat[c * h + i];
                }
            }
            // `ov` (y sus CSR) se libera aquí.
        }

        // ── lm_head batcheado.
        let vocab = self.config.vocab_size;
        let ow = self
            .catalog
            .load_quant_matrix("output.weight")
            .or_else(|_| self.catalog.load_quant_matrix("token_embd.weight"))?;
        for c in 0..n {
            x_flat[c * h..(c + 1) * h].copy_from_slice(&x[c]);
            rms_norm(&mut x_flat[c * h..(c + 1) * h], &self.output_norm, eps);
        }
        let mut logits_flat = vec![0.0f32; n * vocab];
        orch.execute_quant_gemv_batched(&ow, &x_flat, &mut logits_flat, n)?;
        let mut out = Vec::with_capacity(n);
        for c in 0..n {
            out.push(logits_flat[c * vocab..(c + 1) * vocab].to_vec());
        }
        Ok(out)
    }
/// Forward **por capas** Dense (Fase 2, layer-major): procesa TODA la secuencia de
/// tokens por capa en una sola llamada. Los pesos de cada capa se cargan UNA vez por
/// generación, el override se construye UNA vez por (candidato, capa) y el FFN + las
/// proyecciones de atención se batchean sobre `[N×n_pos]` (criterios C1/C4).
pub fn forward_batched_seq(
    &mut self,
    orch: &mut EngineOrchestrator,
    tokens: &[u32],
    kv: &mut [Vec<LayerKvCache>],
    n_candidates: usize,
    mut get_override: impl FnMut(
        usize,
        usize,
        &Option<Arc<Vec<f32>>>,
        &Option<Arc<Vec<f32>>>,
        &Option<Arc<Vec<f32>>>,
    ) -> FfnOverride,
) -> Result<Vec<Vec<f32>>, StreamInferError> {
    let h = self.config.hidden_size;
    let n_layers = self.config.num_layers;
    let n_pos = tokens.len();
    let n = n_candidates;
    let eps = self.config.rms_norm_eps;
    let q_dim = self.attn_cfg.hidden_size();
    let kv_dim = self.attn_cfg.kv_dim();
    let ff = self.config.intermediate_size;
    if kv.len() < n || n_pos == 0 {
        return Err(StreamInferError::Msg(format!(
            "forward_batched_seq: kv tiene {} filas, se necesitan {n}",
            kv.len()
        )));
    }
    let mut x: Vec<Vec<f32>> = (0..n).map(|_| vec![0.0f32; n_pos * h]).collect();
    for c in 0..n {
        for t in 0..n_pos {
            let mut emb = vec![0.0f32; h];
            self.embed_row("token_embd.weight", tokens[t], h, &mut emb)?;
            x[c][t * h..(t + 1) * h].copy_from_slice(&emb);
        }
    }
    let batch = n * n_pos;
    let mut x_flat = vec![0.0f32; batch * h];
    let mut q_flat = vec![0.0f32; batch * q_dim];
    let mut k_flat = vec![0.0f32; batch * kv_dim];
    let mut v_flat = vec![0.0f32; batch * kv_dim];
    let mut attn_out_flat = vec![0.0f32; batch * q_dim];
    let mut attn_proj_flat = vec![0.0f32; batch * h];
    let mut gate_flat = vec![0.0f32; batch * ff];
    let mut up_flat = vec![0.0f32; batch * ff];
    let mut down_flat = vec![0.0f32; batch * h];
    for layer_idx in 0..n_layers {
        let pack = self.load_pack(layer_idx)?;
        // Pesos compartidos por capa (Fase 2, C1/C4): si hay GPU y el bloque es
        // Q4_K, el kernel `spmm_adj_batched_q4` dequantiza en la GPU (sin
        // materializar 23 GB F32/gen); si no, se dequantiza el F32 una vez por capa.
        let has_gpu = orch.opencl_engine().is_some();
        // Dequant Q4_K en GPU es opt-in (HAYAI_SPMM_Q4=1): bit-exacto y VRAM en
        // rango C2 (2.0 GB) pero ~2x mas lento en la RTX 4050 (686s vs 360s del
        // F32 compartido, buffer F32 de 356 MB por capa). F32 es el default (C4).
        let can_q4 = std::env::var("HAYAI_SPMM_Q4").ok().as_deref() == Some("1")
            && pack.gate.ggml_type == GgmlType::Q4_K;
        let gate_w = if has_gpu && can_q4 {
            None
        } else {
            self.catalog
                .dequant_f32(&format!("blk.{layer_idx}.ffn_gate.weight"))
                .ok()
                .map(Arc::new)
        };
        let up_w = if has_gpu && can_q4 {
            None
        } else {
            self.catalog
                .dequant_f32(&format!("blk.{layer_idx}.ffn_up.weight"))
                .ok()
                .map(Arc::new)
        };
        let down_w = if has_gpu && can_q4 {
            None
        } else {
            self.catalog
                .dequant_f32(&format!("blk.{layer_idx}.ffn_down.weight"))
                .ok()
                .map(Arc::new)
        };
        let gate_q4 = if has_gpu && can_q4 {
            Some(pack.gate.raw_bytes())
        } else {
            None
        };
        let up_q4 = if has_gpu && can_q4 {
            Some(pack.up.raw_bytes())
        } else {
            None
        };
        let down_q4 = if has_gpu && can_q4 {
            Some(pack.down.raw_bytes())
        } else {
            None
        };
        let ov: Vec<FfnOverride> = (0..n)
            .map(|c| get_override(c, layer_idx, &gate_w, &up_w, &down_w))
            .collect();
        // Atención: norm + proyecciones batcheadas sobre N×n_pos.
        for c in 0..n {
            for t in 0..n_pos {
                x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h]
                    .copy_from_slice(&x[c][t * h..(t + 1) * h]);
                rms_norm(
                    &mut x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h],
                    &self.layer_norms[layer_idx].attn_norm,
                    eps,
                );
            }
        }
        orch.execute_quant_gemv_batched(&pack.wq, &x_flat, &mut q_flat, batch)?;
        orch.execute_quant_gemv_batched(&pack.wk, &x_flat, &mut k_flat, batch)?;
        orch.execute_quant_gemv_batched(&pack.wv, &x_flat, &mut v_flat, batch)?;
        for c in 0..n {
            for t in 0..n_pos {
                let idx = c * n_pos + t;
                attention_decode_step(
                    &self.attn_cfg,
                    &mut kv[c][layer_idx],
                    &mut q_flat[idx * q_dim..(idx + 1) * q_dim],
                    &mut k_flat[idx * kv_dim..(idx + 1) * kv_dim],
                    &v_flat[idx * kv_dim..(idx + 1) * kv_dim],
                    t,
                    &mut attn_out_flat[idx * q_dim..(idx + 1) * q_dim],
                );
            }
        }
        if let Some(ref gate_w) = pack.attn_gate {
            let mut ag = vec![0.0f32; batch * q_dim];
            orch.execute_quant_gemv_batched(gate_w, &x_flat, &mut ag, batch)?;
            for i in 0..batch * q_dim {
                attn_out_flat[i] *= 1.0 / (1.0 + (-ag[i]).exp());
            }
        }
        orch.execute_quant_gemv_batched(&pack.wo, &attn_out_flat, &mut attn_proj_flat, batch)?;
        for c in 0..n {
            for t in 0..n_pos {
                for i in 0..h {
                    x[c][t * h + i] += attn_proj_flat[(c * n_pos + t) * h + i];
                }
            }
        }
        // FFN: norm + gate/up denso batcheado + override CSR + down.
        for c in 0..n {
            for t in 0..n_pos {
                x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h]
                    .copy_from_slice(&x[c][t * h..(t + 1) * h]);
                rms_norm(
                    &mut x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h],
                    &self.layer_norms[layer_idx].ffn_norm,
                    eps,
                );
            }
        }
        let has_any_csr = ov
            .iter()
            .any(|o| o.gate.is_some() || o.up.is_some() || o.down.is_some());
        let has_any_adj = ov
            .iter()
            .any(|o| o.gate_adj.is_some() || o.up_adj.is_some() || o.down_adj.is_some());
        orch.execute_quant_gemv_batched(&pack.gate, &x_flat, &mut gate_flat, batch)?;
        orch.execute_quant_gemv_batched(&pack.up, &x_flat, &mut up_flat, batch)?;
        if has_any_adj {
            self.apply_sparse_adj_block(
                orch, &ov, |o| o.gate_adj.as_ref(), &x_flat, n, n_pos, h, ff,
                gate_w.as_ref(), gate_q4.as_deref(), &mut gate_flat,
            )?;
            self.apply_sparse_adj_block(
                orch, &ov, |o| o.up_adj.as_ref(), &x_flat, n, n_pos, h, ff,
                up_w.as_ref(), up_q4.as_deref(), &mut up_flat,
            )?;
        }
        if has_any_csr {
            for c in 0..n {
                if let Some(cs) = ov[c].gate.as_ref() {
                    let out = self.spmm_csr(orch, &x_flat[c * n_pos * h..(c + 1) * n_pos * h], cs)?;
                    gate_flat[c * n_pos * ff..(c + 1) * n_pos * ff].copy_from_slice(&out);
                }
                if let Some(cs) = ov[c].up.as_ref() {
                    let out = self.spmm_csr(orch, &x_flat[c * n_pos * h..(c + 1) * n_pos * h], cs)?;
                    up_flat[c * n_pos * ff..(c + 1) * n_pos * ff].copy_from_slice(&out);
                }
            }
        }
        for i in 0..batch * ff {
            let g = gate_flat[i];
            gate_flat[i] = (g / (1.0 + (-g).exp())) * up_flat[i];
        }
        orch.execute_quant_gemv_batched(&pack.down, &gate_flat, &mut down_flat, batch)?;
        if has_any_adj {
            self.apply_sparse_adj_block(
                orch, &ov, |o| o.down_adj.as_ref(), &gate_flat, n, n_pos, ff, h,
                down_w.as_ref(), down_q4.as_deref(), &mut down_flat,
            )?;
        }
        if has_any_csr {
            for c in 0..n {
                if let Some(cs) = ov[c].down.as_ref() {
                    let out = self.spmm_csr(orch, &gate_flat[c * n_pos * ff..(c + 1) * n_pos * ff], cs)?;
                    down_flat[c * n_pos * h..(c + 1) * n_pos * h].copy_from_slice(&out);
                }
            }
        }
        for c in 0..n {
            for t in 0..n_pos {
                for i in 0..h {
                    x[c][t * h + i] += down_flat[(c * n_pos + t) * h + i];
                }
            }
        }
        // `ov` (y sus CSR) se libera aquí.
    }
    // lm_head batcheado sobre N×n_pos.
    let vocab = self.config.vocab_size;
    let ow = self
        .catalog
        .load_quant_matrix("output.weight")
        .or_else(|_| self.catalog.load_quant_matrix("token_embd.weight"))?;
    for c in 0..n {
        for t in 0..n_pos {
            x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h]
                .copy_from_slice(&x[c][t * h..(t + 1) * h]);
            rms_norm(
                &mut x_flat[(c * n_pos + t) * h..(c * n_pos + t + 1) * h],
                &self.output_norm,
                eps,
            );
        }
    }
    let mut logits_flat = vec![0.0f32; batch * vocab];
    orch.execute_quant_gemv_batched(&ow, &x_flat, &mut logits_flat, batch)?;
    let mut out = Vec::with_capacity(n);
    for c in 0..n {
        out.push(logits_flat[c * n_pos * vocab..(c + 1) * n_pos * vocab].to_vec());
    }
    Ok(out)
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
    if std::env::var("HAYAI_DUMP_GATE").ok().as_deref() == Some("1") && layer == 0 {
        eprintln!("PROD_GATE_PRE L0: {:?}", &gate_out[0..8]);
        eprintln!("PROD_UP_PRE L0: {:?}", &up_out[0..8]);
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

pub fn load_config(cat: &GgufCatalog) -> Result<ModelConfig, GgufError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use hayai_model::gguf::write_minimal_gguf;
    use hayai_model::MetadataValue;
    use std::env::temp_dir;

    /// ModelKind must be derived from tensors/ops, never `general.architecture`.
    fn kind_for(name: &str, tensors: &[(&str, Vec<u64>)]) -> ModelKind {
        let path = temp_dir().join(format!("hayai_modelkind_{name}.gguf"));
        let values: Vec<f32> = tensors
            .iter()
            .flat_map(|(_, dims)| vec![1.0f32; dims.iter().product::<u64>() as usize])
            .collect();
        // write_minimal_gguf needs (name, dims, values) triples; build them inline.
        let mut t: Vec<(&str, Vec<u64>, Vec<f32>)> = Vec::new();
        let mut cursor = 0usize;
        for (n, d) in tensors {
            let len = d.iter().product::<u64>() as usize;
            t.push((n, d.clone(), values[cursor..cursor + len].to_vec()));
            cursor += len;
        }
        write_minimal_gguf(
            &path,
            &[(
                "general.architecture",
                MetadataValue::String("test".into()),
            )],
            &t,
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        let kind = ModelKind::from_catalog(&cat);
        let _ = std::fs::remove_file(&path);
        kind
    }

    #[test]
    fn model_kind_detects_dense() {
        assert_eq!(
            kind_for(
                "dense",
                &[("token_embd.weight", vec![16, 8])]
            ),
            ModelKind::Dense
        );
    }

    #[test]
    fn model_kind_detects_hybrid_from_ssm() {
        assert_eq!(
            kind_for(
                "hybrid",
                &[
                    ("token_embd.weight", vec![16, 8]),
                    ("blk.0.ssm_out.weight", vec![8, 8]),
                    ("blk.0.attn_qkv.weight", vec![24, 8]),
                ]
            ),
            ModelKind::Hybrid
        );
    }

    #[test]
    fn model_kind_detects_moe_from_router() {
        assert_eq!(
            kind_for(
                "moe",
                &[
                    ("token_embd.weight", vec![16, 8]),
                    ("blk.0.ffn_gate_inp.weight", vec![4, 8]),
                    ("blk.0.ffn_exp.0.ffn_gate.weight", vec![8, 8]),
                ]
            ),
            ModelKind::MoE
        );
    }

    #[test]
    fn model_kind_detects_gemma_from_per_head_norms() {
        assert_eq!(
            kind_for(
                "gemma",
                &[
                    ("token_embd.weight", vec![16, 8]),
                    ("blk.0.attn_q_norm.weight", vec![8]),
                    ("blk.0.attn_k_norm.weight", vec![8]),
                ]
            ),
            ModelKind::Gemma
        );
    }
}

