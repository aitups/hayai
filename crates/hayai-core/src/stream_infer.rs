//! PRD generate: deterministic layer streaming + Attn∥FFN + dGPU∥APU.
//!
//! Pipeline per layer (decode):
//! 1. Prefetch layer N+1 into ping-pong (second `fork_reader` handle).
//! 2. CPU Attention on pack N.
//! 3. Enqueue FFN gate/up async (`cl_event`); while in flight, join prefetch (I/O∥FFN).
//! 4. SiLU + down; residual. Next iteration's Attn starts as soon as FFN finishes
//!    (macro-pipeline: GPU FFN(N) overlaps CPU/I/O prep for N+1; Attn(N+1) follows
//!    immediately — residual deps prevent Attn(N+1) before FFN(N) completes).

use hayai_cpu::{
    attention_decode_step_ex, rms_norm, AttentionConfig, LayerKvCache,
};
use hayai_model::{
    sample_with, GgmlType, GgufCatalog, GgufError, LayerPackLayout, LayerWeightPack, ModelConfig,
    Penalties, QuantMatrix, SamplerConfig, Tokenizer,
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
    pub(crate) unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.addr as *mut u8, self.len) }
    }
}

/// Activation applied by the generic causal depthwise conv.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ConvActivation {
    Silu,
    Gelu,
    None,
}

impl ConvActivation {
    fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "none" | "identity" | "linear" => Self::None,
            "gelu" => Self::Gelu,
            _ => Self::Silu,
        }
    }

    fn apply(self, a: f32) -> f32 {
        match self {
            Self::None => a,
            Self::Gelu => {
                // tanh approximation (ggml `GELU`).
                0.5 * a * (1.0 + (0.797_884_6 * (a + 0.044_715 * a * a * a)).tanh())
            }
            Self::Silu => a / (1.0 + (-a).exp()),
        }
    }
}

/// Depthwise causal conv1d + activation, added as a residual to `x`:
/// `x[c] += act(sum_t w[c*k + t] * (history ++ x)[t])`, with `t=0` the oldest
/// tap (PyTorch / HF / ggml convention). `state` holds the last `k-1` inputs,
/// oldest→newest, in `[(k-1) * channels]`. `channels` may differ from `hidden`
/// (a conv over a projected subspace); the caller passes the matching buffer.
fn apply_depthwise_conv(
    x: &mut [f32],
    state: &mut [f32],
    k: usize,
    w: &[f32],
    act: ConvActivation,
) {
    let ch = x.len();
    let hist = k.saturating_sub(1);
    debug_assert!(state.len() >= hist * ch);
    let mut acc = vec![0.0f32; ch];
    for c in 0..ch {
        let row = c * k;
        let mut a = 0.0f32;
        for t in 0..hist {
            a += w[row + t] * state[t * ch + c];
        }
        a += w[row + hist] * x[c];
        acc[c] = a;
    }
    if hist > 0 {
        if hist > 1 {
            state.copy_within(ch..hist * ch, 0);
        }
        state[(hist - 1) * ch..hist * ch].copy_from_slice(x);
    }
    for c in 0..ch {
        x[c] += act.apply(acc[c]);
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
    /// LayerNorm biases (GPT-2/BLOOM/OPT): present only for LayerNorm models.
    pub(crate) attn_norm_bias: Option<Vec<f32>>,
    pub(crate) ffn_norm_bias: Option<Vec<f32>>,
    /// Gemma4 extra per-block norms/scales — preloaded once at session open so the
    /// decode hot path never issues per-token norm disk reads.
    pub(crate) attn_q_norm: Option<Vec<f32>>,
    pub(crate) attn_k_norm: Option<Vec<f32>>,
    pub(crate) post_attn_norm: Option<Vec<f32>>,
    pub(crate) post_ffw_norm: Option<Vec<f32>>,
    pub(crate) layer_output_scale: Option<f32>,
}

/// Dense FFN biases (`ffn_gate/up/down.bias`), preloaded once per model.
#[derive(Clone, Default)]
pub(crate) struct LayerFfnBias {
    pub(crate) gate: Option<Vec<f32>>,
    pub(crate) up: Option<Vec<f32>>,
    pub(crate) down: Option<Vec<f32>>,
}

/// MLA (DeepSeek-V2/V3, Kimi) attention config resolved from GGUF metadata.
#[derive(Clone, Copy)]
pub(crate) struct MlaMeta {
    pub(crate) n_heads: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) qk_nope: usize,
    pub(crate) qk_rope: usize,
    pub(crate) v_head_dim: usize,
    /// `true` for the absorbed path (`attn_k_b`/`attn_v_b`, 1 KV head MQA).
    pub(crate) absorbed: bool,
}

impl MlaMeta {
    pub(crate) fn qk_head(&self) -> usize {
        self.qk_nope + self.qk_rope
    }
}

/// MLA compressed cache: K is `[n_heads × (qk_nope + qk_rope)]`, V is
/// `[n_heads × v_head_dim]` (the fused `attn_kv_b` decompression is non-absorbed).
pub(crate) struct MlaKvCache {
    pub(crate) k: LayerKvCache,
    pub(crate) v: LayerKvCache,
}

/// LayerNorm *in place* (mean/variance + weight + optional bias). GPT-2/BLOOM/OPT.
pub(crate) fn layernorm_inplace(
    x: &mut [f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        let w = weight.get(i).copied().unwrap_or(1.0);
        let b = bias.and_then(|b| b.get(i)).copied().unwrap_or(0.0);
        x[i] = (x[i] - mean) * inv * w + b;
    }
}

/// Apply a norm with the model's norm kind (RMSNorm or LayerNorm).
pub(crate) fn apply_norm(
    x: &mut [f32],
    weight: &[f32],
    bias: &Option<Vec<f32>>,
    eps: f32,
    layernorm: bool,
) {
    if layernorm {
        layernorm_inplace(x, weight, bias.as_deref(), eps);
    } else {
        rms_norm(x, weight, eps);
    }
}

/// GELU (exact, `erf`-based) — matches llama.cpp `LLM_FFN_GELU` (`ggml_gelu`) for
/// the classic ungated FFNs (GPT-2 / BLOOM / OPT).
#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * 0.707_106_77))
}

/// `erf` via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t) + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

/// ALiBi slopes (HF `get_alibi_slopes`): `2^(-8*(h+1)/n)` for power-of-two head
/// counts, interleaved otherwise (BLOOM/Falcon/MPT).
pub(crate) fn alibi_slopes(n_head: usize) -> Vec<f32> {
    fn pow2(n: usize) -> Vec<f32> {
        let start = 2f64.powf(-(2f64.powf(-((n as f64).log2() - 3.0))));
        (0..n)
            .map(|i| (start * start.powi(i as i32)) as f32)
            .collect()
    }
    if n_head == 0 {
        return Vec::new();
    }
    let log2 = (n_head as f64).log2();
    if log2.fract() == 0.0 {
        pow2(n_head)
    } else {
        let closest = 2f64.powf(log2.floor()) as usize;
        let mut slopes = pow2(closest);
        let extra = alibi_slopes(2 * closest);
        for &s in extra.iter().step_by(2).take(n_head - closest) {
            slopes.push(s);
        }
        slopes
    }
}

/// Attention projection biases (`attn_q/k/v/o.bias`), preloaded once per model.
/// Qwen2/Qwen2.5 (`attention_bias=true`) need these or the attention is wrong.
#[derive(Clone, Default)]
pub(crate) struct LayerAttnBias {
    pub(crate) q: Option<Vec<f32>>,
    pub(crate) k: Option<Vec<f32>>,
    pub(crate) v: Option<Vec<f32>>,
    pub(crate) o: Option<Vec<f32>>,
}

impl LayerAttnBias {
    fn is_empty(&self) -> bool {
        self.q.is_none() && self.k.is_none() && self.v.is_none() && self.o.is_none()
    }
}

/// Add an optional bias vector element-wise (no-op when absent).
pub(crate) fn add_bias(buf: &mut [f32], bias: &Option<Vec<f32>>) {
    if let Some(b) = bias {
        for (x, y) in buf.iter_mut().zip(b.iter()) {
            *x += y;
        }
    }
}

/// Add q/k/v biases from a layer's bias record.
pub(crate) fn add_qkv_bias(b: &LayerAttnBias, q: &mut [f32], k: &mut [f32], v: &mut [f32]) {
    add_bias(q, &b.q);
    add_bias(k, &b.k);
    add_bias(v, &b.v);
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
    /// Mamba-1 selective-scan SSM blocks (no attention / no KV cache).
    Mamba,
    /// MLA (DeepSeek-V2/V3, Kimi): latent attention + dense-lead / MoE FFN.
    Mla,
}

/// One item of a multimodal prompt: a text token, or a media embedding row
/// (image/audio) injected in place of a placeholder token.
#[derive(Clone)]
pub enum MediaInput {
    Token(u32),
    Emb(Vec<f32>),
}

impl ModelKind {
    /// Detect from catalog tensor presence (runs before the ExecPlan exists).
    ///
    /// Scans **every** block (not just `blk.0`): a hybrid Qwen3.5 whose first
    /// layer is full-attention must still be detected as `Hybrid`, otherwise
    /// `open()` would build dense KV caches while `decode_step` dispatches to the
    /// hybrid path.
    pub fn from_catalog(cat: &GgufCatalog) -> Self {
        let arch = cat.meta_str("general.architecture").unwrap_or("");
        let block_count = cat
            .meta_u32(&format!("{arch}.block_count"))
            .or_else(|| cat.meta_u32("llama.block_count"))
            .unwrap_or(1)
            .max(1) as usize;
        let has_any = |suffix: &str| {
            (0..block_count).any(|i| cat.tensor(&format!("blk.{i}.{suffix}")).is_ok())
        };
        if has_any("ssm_x.weight") || has_any("ssm_d") {
            // Mamba-1 selective scan: x_proj (`ssm_x`) + skip (`ssm_d`) are unique.
            ModelKind::Mamba
        } else if has_any("attn_kv_a_mqa.weight") {
            // MLA latent attention (DeepSeek-V2/V3, Kimi).
            ModelKind::Mla
        } else if has_any("ssm_out.weight") || has_any("ssm_a") {
            ModelKind::Hybrid
        } else if has_any("ffn_gate_inp.weight")
            || has_any("ffn_exp.0.ffn_gate.weight")
            || has_any("ffn_shexp.ffn_gate.weight")
        {
            ModelKind::MoE
        } else if has_any("post_ffw_norm.weight")
            || has_any("layer_output_scale.weight")
            || has_any("attn_q_norm.weight")
            || has_any("attn_k_norm.weight")
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
    /// Per-layer attention biases (empty vectors when absent — the common case).
    pub(crate) attn_bias: Vec<LayerAttnBias>,
    /// Index of the trailing NextN/MTP draft block (one past the main trunk), if any.
    pub(crate) mtp_slot: Option<usize>,
    /// Post-final-norm hidden of the last processed token (`t_h_nextn`). Saved by
    /// every forward path so the MTP draft head can consume it.
    pub(crate) last_hidden_nextn: Vec<f32>,
    /// Cached MTP/NextN block weights (loaded on first draft).
    pub(crate) mtp_cache: Option<crate::mtp::MtpCache>,
    /// Cached Mamba selective-scan weights + recurrent state (op-driven SSM).
    pub(crate) mamba_cache: Option<crate::mamba_infer::MambaCache>,
    pub(crate) output_norm: Vec<f32>,
    /// LayerNorm bias for the final norm (GPT-2/BLOOM/OPT), if any.
    pub(crate) output_norm_bias: Option<Vec<f32>>,
    /// Input-embedding LayerNorm (BLOOM `token_embd_norm`): `(weight, bias)`.
    pub(crate) token_embed_norm: Option<(Vec<f32>, Vec<f32>)>,
    /// Learned absolute position embeddings (`position_embd.weight`), flattened
    /// `[max_pos × hidden]` (GPT-2). `None` for RoPE models.
    pub(crate) learned_pos: Option<Vec<f32>>,
    /// Per-layer dense FFN biases (`ffn_gate/up/down.bias`).
    pub(crate) ffn_bias: Vec<LayerFfnBias>,
    /// True when the model uses LayerNorm (mean+variance) instead of RMSNorm.
    pub(crate) use_layernorm: bool,
    /// Classic-transformer Dense features (LayerNorm / learned positions / input
    /// LayerNorm): prefill falls back to the per-token Dense path.
    pub(crate) simple_dense: bool,
    /// Parallel residual + single shared norm (Phi-2/GPT-J/PaLM): `x = x + attn(ln(x))
    /// + ffn(ln(x))` — the block has `attn_norm` but no `ffn_norm`.
    pub(crate) parallel_residual: bool,
    /// Per-layer RoPE enable (Cohere2: SWA layers use RoPE, global layers NoPE).
    /// Empty ⇒ RoPE for every layer.
    pub(crate) layer_apply_rope: Vec<bool>,
    /// Architecture scalars (Granite `embedding_multiplier` / `residual_multiplier`
    /// / `logits_scaling`); 1.0 for every other family.
    pub(crate) embedding_scale: f32,
    pub(crate) residual_scale: f32,
    pub(crate) logit_scale: f32,
    /// ALiBi slopes (one per query head) for BLOOM/Falcon/MPT; `None` otherwise.
    pub(crate) alibi_slopes: Option<Vec<f32>>,
    pub(crate) has_output_weight: bool,
    pub sampler: SamplerConfig,
    /// Repetition/presence/frequency/logit-bias adjustments applied at sampling time.
    pub penalties: Penalties,
    /// Optional GBNF grammar for constrained decoding.
    pub grammar: Option<hayai_model::Grammar>,
    /// Live grammar state (reset at the start of each [`Self::generate`]).
    pub grammar_state: Option<hayai_model::GrammarState>,
    /// Cached per-token text for the grammar mask (built by [`Self::set_grammar`]).
    pub grammar_token_texts: Vec<Option<String>>,
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
    /// Reusable per-token scratch buffers (avoid per-layer heap allocs in decode).
    pub(crate) scratch_xn: Vec<f32>,
    pub(crate) scratch_q: Vec<f32>,
    pub(crate) scratch_k: Vec<f32>,
    pub(crate) scratch_v: Vec<f32>,
    pub(crate) scratch_attn: Vec<f32>,
    pub(crate) scratch_gate: Vec<f32>,
    pub(crate) scratch_proj: Vec<f32>,
    /// HRM frozen low-cycle init state (`hrm.z_l_init`), length = hidden.
    pub(crate) z_l_init: Option<Vec<f32>>,
    /// Gemma4 proportional RoPE factors (`rope_freqs.weight`), cached once.
    pub(crate) gemma_rope_freqs: Option<Vec<f32>>,
    /// LongRoPE (Phi-3-128k): `(short_factors, long_factors, original_ctx_len)`.
    pub(crate) longrope: Option<(Vec<f32>, Vec<f32>, usize)>,
    /// Per-dim RoPE `freq_factors` for the Dense path: `rope_freqs.weight` (Gemma4
    /// proportional / Llama-3 NTK-by-parts baked by the converter) or LongRoPE
    /// `rope_factors_{short,long}` selected by sequence length.
    pub(crate) rope_freq_factors: Option<Vec<f32>>,
    /// Gemma4 per-layer-embedding projection (`per_layer_model_proj.weight`): a
    /// single global tensor, loaded once on first use instead of once per token.
    pub(crate) ple_model_proj: Option<QuantMatrix>,
    /// Gemma4 PLE projection norm (`per_layer_proj_norm.weight`), loaded once.
    pub(crate) ple_proj_norm: Option<Vec<f32>>,
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
    /// MoE resident non-expert (attn + router + shared expert) bytes per layer,
    /// cached after the first read so subsequent tokens skip the disk. Bounded by
    /// `moe_non_expert_cap`. Experts are streamed sparsely (never cached here).
    pub(crate) moe_non_expert: Option<Vec<Option<Box<[u8]>>>>,
    pub(crate) moe_non_expert_bytes: usize,
    pub(crate) moe_non_expert_cap: usize,
    /// Per-layer depthwise causal short-conv weights (`Conv` op):
    /// `(kernel, channels, w[c*kernel + t])`; `None` for layers without a conv.
    pub(crate) conv_weights: Option<Vec<Option<(usize, usize, Vec<f32>)>>>,
    /// Per-layer conv history `[(kernel-1) * channels]`, oldest→newest.
    pub(crate) conv_states: Option<Vec<Option<Vec<f32>>>>,
    /// Per-layer `Conv` tensor name, resolved once from the ExecPlan.
    pub(crate) conv_names: Option<Vec<Option<String>>>,
    /// Activation for the generic causal conv (`hayai.conv_activation`, default SiLU).
    pub(crate) conv_activation: ConvActivation,
    /// Stored ExecPlan used to drive the forward graph (Ola 2). Shared via `Arc`
    /// so the per-token forward paths can borrow it while mutating the generator.
    pub exec_plan: Option<std::sync::Arc<crate::exec_plan::ExecPlan>>,
    /// Dedicated background I/O worker (Phase H1) for block prefetches.
    pub(crate) io_worker: hayai_io::IoWorker,
    /// Qwen3.5 DeltaNet weights (None entries for full-attn layers).
    pub(crate) deltanet_weights: Option<Vec<Option<crate::deltanet::DeltaNetLayerWeights>>>,
    pub(crate) deltanet_states: Option<Vec<Option<crate::deltanet::DeltaNetState>>>,
    /// MoE expert LRU cache (colibri-style) — host RAM, sized at session start.
    pub(crate) moe_cache: crate::moe_infer::ExpertCache,
    /// MoE router selection bias (`blk.N.ffn_gate_inp.bias`, DeepSeek V3), per layer.
    pub(crate) moe_router_bias: Option<Vec<Option<Vec<f32>>>>,
    /// MLA attention config (DeepSeek-V2/V3, Kimi); `None` otherwise.
    pub(crate) mla: Option<MlaMeta>,
    /// MLA compressed KV caches (one per layer).
    pub(crate) mla_kv: Option<Vec<MlaKvCache>>,
    /// MLA `attn_kv_a_norm` (dim `kv_lora_rank`) per layer.
    pub(crate) mla_kv_a_norm: Option<Vec<Option<Vec<f32>>>>,
    /// MLA `attn_q_a_norm` (dim `q_lora_rank`) per layer (full MLA only).
    pub(crate) mla_q_a_norm: Option<Vec<Option<Vec<f32>>>>,
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
        // Encoder-decoder (T5/BART) is a distinct architecture class handled by
        // `encoder_decoder_infer`; the decoder-only paths would misread `enc.blk`/`dec.blk`.
        if crate::encoder_decoder_infer::is_encoder_decoder(&catalog) {
            return Err(StreamInferError::Msg(
                "encoder-decoder model (T5/BART): use the encoder-decoder path \
                 (`hayai_core::encoder_decoder_infer`), not the decoder-only streaming generator"
                    .into(),
            ));
        }
        let config = load_config(&catalog)?;
        let mut attn_cfg = build_attn_config(&catalog, &config)?;

        // HRM / some hybrids use parameterless RMSNorm (no weight tensors).
        let ones = |n: usize| vec![1.0f32; n];
        let h = config.hidden_size;
        // One extra slot for the trailing NextN/MTP draft block (index `num_layers`),
        // which the main trunk skips but the MTP head executes.
        let mtp_slot = if crate::deltanet::is_nextn_layer(&catalog, config.num_layers) {
            Some(config.num_layers)
        } else {
            None
        };
        let n_slots = config.num_layers + usize::from(mtp_slot.is_some());
        let mut layer_norms = Vec::with_capacity(n_slots);
        for i in 0..n_slots {
            let attn_norm = catalog
                .dequant_f32(&format!("blk.{i}.attn_norm.weight"))
                .or_else(|_| catalog.dequant_f32(&format!("blk.{i}.attention_norm.weight")))
                .unwrap_or_else(|_| ones(h));
            let ffn_norm = catalog
                .dequant_f32(&format!("blk.{i}.ffn_norm.weight"))
                .or_else(|_| catalog.dequant_f32(&format!("blk.{i}.post_attention_norm.weight")))
                .unwrap_or_else(|_| ones(h));
            // LayerNorm biases (GPT-2/BLOOM): present only for LayerNorm models.
            let attn_norm_bias = catalog
                .dequant_f32(&format!("blk.{i}.attn_norm.bias"))
                .or_else(|_| catalog.dequant_f32(&format!("blk.{i}.attention_norm.bias")))
                .ok();
            let ffn_norm_bias = catalog
                .dequant_f32(&format!("blk.{i}.ffn_norm.bias"))
                .ok();
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
                attn_norm_bias,
                ffn_norm_bias,
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
        let output_norm_bias = catalog.dequant_f32("output_norm.bias").ok();
        // LayerNorm (`attention.layer_norm_epsilon`) vs RMSNorm (`..._rms_epsilon`).
        let arch = config.architecture.clone();
        let use_layernorm = catalog
            .meta_f32(&format!("{arch}.attention.layer_norm_epsilon"))
            .is_some()
            && catalog
                .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
                .is_none();
        // BLOOM input-embedding LayerNorm (`token_embd_norm` / `word_embeddings_layernorm`).
        let token_embed_norm = match (
            catalog
                .dequant_f32("token_embd_norm.weight")
                .or_else(|_| catalog.dequant_f32("word_embeddings_layernorm.weight")),
            catalog
                .dequant_f32("token_embd_norm.bias")
                .or_else(|_| catalog.dequant_f32("word_embeddings_layernorm.bias")),
        ) {
            (Ok(w), Ok(b)) => Some((w, b)),
            _ => None,
        };
        // Learned absolute position embeddings (GPT-2 `position_embd.weight`).
        let learned_pos = catalog.dequant_f32("position_embd.weight").ok();
        if std::env::var("HAYAI_DUMP_POS").ok().as_deref() == Some("1") {
            if let Ok(t) = catalog.tensor("position_embd.weight") {
                eprintln!("POS_EMBD dims={:?} type={:?} len={}", t.dims, t.ggml_type, learned_pos.as_ref().map(|v| v.len()).unwrap_or(0));
            }
            eprintln!(
                "DENSE_FLAGS use_layernorm={} learned_pos={} token_embed_norm={}",
                use_layernorm,
                learned_pos.is_some(),
                token_embed_norm.is_some(),
            );
        }
        let ffn_bias: Vec<LayerFfnBias> = (0..n_slots)
            .map(|l| LayerFfnBias {
                gate: catalog.dequant_f32(&format!("blk.{l}.ffn_gate.bias")).ok(),
                up: catalog.dequant_f32(&format!("blk.{l}.ffn_up.bias")).ok(),
                down: catalog.dequant_f32(&format!("blk.{l}.ffn_down.bias")).ok(),
            })
            .collect();
        let has_output_weight = catalog.tensor("output.weight").is_ok();
        // Parallel residual: two sublayers read the residual *before* either is added.
        // Detected from the tensor layout (one shared norm, no `ffn_norm`), not from
        // `{arch}.use_parallel_residual` — that metadata is unreliable (e.g. the
        // StableLM-2 GGUF sets it true while the HF config is sequential).
        let parallel_residual = detect_parallel_residual(&catalog);
        // Architecture scalars (Granite): embedding / residual / logit multipliers.
        let meta_scale = |keys: &[&str]| -> f32 {
            for k in keys {
                if let Some(v) = catalog.meta_f32(&format!("{arch}.{k}")) {
                    return v;
                }
            }
            1.0
        };
        let embedding_scale = meta_scale(&["embedding_scale", "embedding_multiplier"]);
        let residual_scale = meta_scale(&["residual_scale", "residual_multiplier"]);
        // `logit_scale` is a **multiplier** on the final logits: Cohere2 stores it
        // directly (`ggml_scale(logits, logit_scale)`); Granite stores the divisor
        // (`logits / logits_scaling`).
        let raw_logit_scale = meta_scale(&["logit_scale", "logits_scaling"]);
        let logit_scale = if raw_logit_scale == 1.0 {
            1.0
        } else if arch == "cohere2" {
            raw_logit_scale
        } else {
            1.0 / raw_logit_scale
        };
        // `AttentionConfig` already supports a scale override (Gemma4 uses 1.0).
        if residual_scale != 1.0 || embedding_scale != 1.0 || logit_scale != 1.0 {
            info!(
                "arch={arch}: embedding_scale={embedding_scale} residual_scale={residual_scale} logit_scale={logit_scale}"
            );
        }
        let simple_dense = use_layernorm
            || learned_pos.is_some()
            || token_embed_norm.is_some()
            || parallel_residual;
        // ALiBi (BLOOM/Falcon/MPT): score bias from absolute q/k positions. Falcon is
        // special: the "new" multiquery architecture (`tensor_data_layout = "jploski"`,
        // falcon-7B/40B) uses **RoPE**, not ALiBi; only the old arch (falcon-rw-1b) does.
        let alibi_slopes = if matches!(
            arch.as_str(),
            "bloom" | "falcon" | "mpt" | "starcoder" | "refact" | "jais"
        ) {
            let falcon_rope = arch == "falcon"
                && catalog.meta_str("falcon.tensor_data_layout") == Some("jploski");
            if falcon_rope {
                None
            } else {
                Some(alibi_slopes(config.num_attention_heads))
            }
        } else {
            None
        };
        // Models with learned positions (GPT-2) or ALiBi (BLOOM/Falcon/MPT) do not
        // rotate Q/K — disable RoPE in the shared attention path.
        if learned_pos.is_some() || alibi_slopes.is_some() {
            attn_cfg.use_rope = false;
        }

        let kv_slots = config
            .hrm
            .as_ref()
            .map(|h| h.kv_slots())
            .unwrap_or(config.num_layers);
        // Cohere2 (Command-R7B): `is_swa(i) = i % 4 != 3`. SWA layers use RoPE + a
        // `sliding_window` cache; global layers use **NoPE** (no RoPE) + full attention.
        let cohere2: Option<(usize, Vec<bool>)> = if arch == "cohere2" {
            let n_swa = catalog
                .meta_u32("cohere2.attention.sliding_window")
                .unwrap_or(4096) as usize;
            Some((n_swa, (0..kv_slots).map(|i| i % 4 != 3).collect()))
        } else {
            None
        };
        let layer_apply_rope: Vec<bool> = cohere2
            .as_ref()
            .map(|(_, s)| s.clone())
            .unwrap_or_default();
        let kv = match ModelKind::from_catalog(&catalog) {
            ModelKind::Gemma => {
                crate::gemma_infer::build_layer_kv_caches(&catalog, &config, sink, window)?
            }
            ModelKind::Hybrid => {
                // Per full-attn layer dims from tensors (not a single 4B-shaped AttentionConfig).
                crate::layer_cfg::build_hybrid_kv_caches(&catalog, &config, sink, window)?
            }
            ModelKind::Mamba => {
                // Selective-scan SSM: recurrent state only, no KV cache.
                Vec::new()
            }
            ModelKind::Mla => {
                // MLA uses its own compressed latent cache (`gen.mla_kv`).
                Vec::new()
            }
            _ => (0..kv_slots)
                .map(|i| {
                    let w = match &cohere2 {
                        Some((n_swa, swa)) if swa[i] => (*n_swa).max(1),
                        Some(_) => config.max_position_embeddings.max(1),
                        None => window,
                    };
                    LayerKvCache::new(attn_cfg.num_kv_heads, attn_cfg.head_dim, sink, w)
                })
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
        // LongRoPE (Phi-3-128k): `short`/`long` per-dim factors + original ctx length.
        let longrope = match (
            catalog.dequant_f32("rope_factors_short.weight").ok(),
            catalog.dequant_f32("rope_factors_long.weight").ok(),
        ) {
            (Some(s), Some(l)) => {
                let orig = catalog
                    .meta_u32(&format!("{arch}.rope.scaling.original_context_length"))
                    .unwrap_or(4096) as usize;
                Some((s, l, orig))
            }
            _ => None,
        };
        // Attention biases (Qwen2/2.5 `attention_bias=true`, GPT-2/BLOOM fused QKV):
        // small F32 vectors, loaded once for all layers so the hot path only adds them.
        let q_dim_b = attn_cfg.hidden_size();
        let kv_dim_b = attn_cfg.kv_dim();
        let attn_bias: Vec<LayerAttnBias> = (0..n_slots)
            .map(|l| {
                let o = catalog.dequant_f32(&format!("blk.{l}.attn_output.bias")).ok();
                let q = catalog.dequant_f32(&format!("blk.{l}.attn_q.bias")).ok();
                let k = catalog.dequant_f32(&format!("blk.{l}.attn_k.bias")).ok();
                let v = catalog.dequant_f32(&format!("blk.{l}.attn_v.bias")).ok();
                // Fused `attn_qkv.bias` (concat [q|k|v]) → split per projection.
                if q.is_none() && k.is_none() && v.is_none() {
                    if let Ok(b) = catalog.dequant_f32(&format!("blk.{l}.attn_qkv.bias")) {
                        if b.len() == q_dim_b + 2 * kv_dim_b {
                            return LayerAttnBias {
                                q: Some(b[..q_dim_b].to_vec()),
                                k: Some(b[q_dim_b..q_dim_b + kv_dim_b].to_vec()),
                                v: Some(b[q_dim_b + kv_dim_b..].to_vec()),
                                o,
                            };
                        }
                    }
                }
                LayerAttnBias { q, k, v, o }
            })
            .collect();
        let n_attn_bias = attn_bias.iter().filter(|b| !b.is_empty()).count();
        if n_attn_bias > 0 {
            info!("Attention biases: {n_attn_bias} layer(s) with q/k/v/o bias");
        }
        // MoE router `e_score_correction_bias` (`ffn_gate_inp.bias`), preloaded once.
        let model_kind = ModelKind::from_catalog(&catalog);
        let moe_like = matches!(model_kind, ModelKind::MoE | ModelKind::Mla);
        let moe_router_bias = if moe_like {
            let v: Vec<Option<Vec<f32>>> = (0..n_slots)
                .map(|l| catalog.dequant_f32(&format!("blk.{l}.ffn_gate_inp.bias")).ok())
                .collect();
            if v.iter().any(|b| b.is_some()) {
                info!("MoE: router correction bias loaded for {} layer(s)", v.iter().filter(|b| b.is_some()).count());
            }
            Some(v)
        } else {
            None
        };
        // MLA attention config + compressed KV caches (DeepSeek-V2/V3, Kimi).
        let (mla, mla_kv, mla_kv_a_norm, mla_q_a_norm) = if model_kind == ModelKind::Mla {
            let arch = &config.architecture;
            let cat_u32 = |key: &str| {
                catalog
                    .meta_u32(&format!("{arch}.{key}"))
                    .or_else(|| catalog.meta_u32(&format!("llama.{key}")))
            };
            let kv_lora_rank = cat_u32("attention.kv_lora_rank").unwrap_or(512) as usize;
            let key_length = cat_u32("attention.key_length")
                .or_else(|| cat_u32("attention.key_length_mla"))
                .unwrap_or(128 + 64) as usize;
            let qk_rope = cat_u32("rope.dimension_count").unwrap_or(64) as usize;
            let v_head_dim = cat_u32("attention.value_length")
                .or_else(|| cat_u32("attention.value_length_mla"))
                .unwrap_or(128) as usize;
            let n_heads = config.num_attention_heads;
            // Absorbed MLA (`attn_k_b`/`attn_v_b`, full DeepSeek-V2/V3 + Kimi) uses a
            // single MQA KV head over the latent; the Lite path decompresses per head.
            let absorbed = (0..config.num_layers)
                .any(|i| catalog.tensor(&format!("blk.{i}.attn_k_b.weight")).is_ok());
            let meta = MlaMeta {
                n_heads,
                kv_lora_rank,
                qk_nope: key_length.saturating_sub(qk_rope),
                qk_rope,
                v_head_dim,
                absorbed,
            };
            let caches: Vec<MlaKvCache> = (0..n_slots)
                .map(|_| {
                    if absorbed {
                        MlaKvCache {
                            k: LayerKvCache::new(1, kv_lora_rank + qk_rope, sink, window),
                            v: LayerKvCache::new(1, kv_lora_rank, sink, window),
                        }
                    } else {
                        MlaKvCache {
                            k: LayerKvCache::new(n_heads, meta.qk_head(), sink, window),
                            v: LayerKvCache::new(n_heads, v_head_dim, sink, window),
                        }
                    }
                })
                .collect();
            let kv_norms: Vec<Option<Vec<f32>>> = (0..n_slots)
                .map(|l| catalog.dequant_f32(&format!("blk.{l}.attn_kv_a_norm.weight")).ok())
                .collect();
            let q_norms: Vec<Option<Vec<f32>>> = (0..n_slots)
                .map(|l| catalog.dequant_f32(&format!("blk.{l}.attn_q_a_norm.weight")).ok())
                .collect();
            info!(
                "MLA: n_heads={} kv_lora_rank={} qk_nope={} qk_rope={} v_head_dim={}",
                n_heads, kv_lora_rank, meta.qk_nope, qk_rope, v_head_dim
            );
            (Some(meta), Some(caches), Some(kv_norms), Some(q_norms))
        } else {
            (None, None, None, None)
        };
        let conv_activation = catalog
            .meta_str("hayai.conv_activation")
            .map(ConvActivation::parse)
            .unwrap_or(ConvActivation::Silu);

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
            attn_bias,
            mtp_slot,
            last_hidden_nextn: Vec::new(),
            mtp_cache: None,
            mamba_cache: None,
            output_norm,
            output_norm_bias,
            token_embed_norm,
            learned_pos,
            ffn_bias,
            use_layernorm,
            simple_dense,
            parallel_residual,
            layer_apply_rope,
            embedding_scale,
            residual_scale,
            logit_scale,
            alibi_slopes,
            has_output_weight,
            sampler,
            penalties: Penalties::default(),
            grammar: None,
            grammar_state: None,
            grammar_token_texts: Vec::new(),
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
            scratch_xn: vec![0.0; hidden],
            scratch_q: vec![0.0; hidden],
            scratch_k: vec![0.0; hidden],
            scratch_v: vec![0.0; hidden],
            scratch_attn: vec![0.0; hidden],
            scratch_gate: vec![0.0; hidden],
            scratch_proj: vec![0.0; hidden],
            z_l_init,
            rope_freq_factors: gemma_rope_freqs.clone(),
            gemma_rope_freqs,
            longrope,
            ple_model_proj: None,
            ple_proj_norm: None,
            moe_non_expert: None,
            moe_non_expert_bytes: 0,
            moe_non_expert_cap: std::env::var("HAYAI_MOE_NONEXPERT_MB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1024)
                .saturating_mul(1024 * 1024),
            conv_weights: None,
            conv_states: None,
            conv_names: None,
            conv_activation,
            exec_plan: None,
            io_worker: hayai_io::IoWorker::new("hayai-io-worker"),
            deltanet_weights: None,
            deltanet_states: None,
            moe_cache: crate::moe_infer::ExpertCache::new(0),
            moe_router_bias,
            mla,
            mla_kv,
            mla_kv_a_norm,
            mla_q_a_norm,
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

    pub(crate) fn act(&self) -> &[f32] {
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
        let pack = match self.fused_qkv_dims(layer) {
            Some((q, kv)) => self.catalog.load_layer_pack_fused(layer, q, kv)?.0,
            None => self.catalog.load_layer_pack(layer)?,
        };
        self.io_secs += t0.elapsed().as_secs_f64();
        self.io_bytes += pack.nbytes() as u64;
        Ok(pack)
    }

    /// Fused `attn_qkv` split dims `(q_dim, kv_dim)` for this layer, or `None` when
    /// the model uses separate `attn_q/k/v` tensors. Fused QKV is split by output
    /// rows assuming the concat `[q | k | v]` layout.
    pub(crate) fn fused_qkv_dims(&self, layer: usize) -> Option<(usize, usize)> {
        if self
            .catalog
            .tensor(&format!("blk.{layer}.attn_q.weight"))
            .is_ok()
        {
            return None;
        }
        if self
            .catalog
            .tensor(&format!("blk.{layer}.attn_qkv.weight"))
            .is_err()
        {
            return None;
        }
        let q = self.attn_cfg.num_heads * self.attn_cfg.head_dim;
        let kv = self.attn_cfg.num_kv_heads * self.attn_cfg.head_dim;
        Some((q, kv))
    }

    /// Load one layer pack into `dst`, using the fused-QKV split when needed.
    pub(crate) fn load_pack_into(
        &self,
        cat: &mut hayai_model::GgufCatalog,
        layer: usize,
        dst: &mut [u8],
    ) -> Result<(LayerWeightPack, LayerPackLayout), GgufError> {
        match self.fused_qkv_dims(layer) {
            Some((q, kv)) => cat.load_layer_pack_into_fused(layer, dst, q, kv),
            None => cat.load_layer_pack_into(layer, dst),
        }
    }

    /// Re-derive pack views over an already-resident/staged base, splitting fused
    /// QKV when needed.
    pub(crate) fn pack_views_from_base(
        &mut self,
        layer: usize,
        base: &[u8],
    ) -> Result<(LayerWeightPack, LayerPackLayout), StreamInferError> {
        match self.fused_qkv_dims(layer) {
            Some((q, kv)) => {
                Ok(self
                    .catalog
                    .layer_pack_views_from_base_fused(layer, base, q, kv)?)
            }
            None => Ok(self.catalog.layer_pack_views_from_base(layer, base)?),
        }
    }

    /// Load (once) the depthwise causal short-conv weights of `layer`, if present.
    fn ensure_conv_layer(&mut self, layer: usize) -> Result<(), StreamInferError> {
        let n = self.config.num_layers + usize::from(self.mtp_slot.is_some());
        if self.conv_names.is_none() {
            // Resolve every `Conv` tensor name once (no plan scan per token).
            let mut names: Vec<Option<String>> = vec![None; n];
            if let Some(plan) = self.exec_plan.as_ref() {
                for u in &plan.units {
                    let Some(b) = u.block_id else { continue };
                    if b >= n {
                        continue;
                    }
                    for t in &u.tensors {
                        if t.op == crate::LayerOpKind::Conv && names[b].is_none() {
                            names[b] = Some(t.name.clone());
                        }
                    }
                }
            }
            self.conv_names = Some(names);
            self.conv_weights = Some((0..n).map(|_| None).collect());
            self.conv_states = Some((0..n).map(|_| None).collect());
        }
        if self.conv_names.as_ref().unwrap()[layer].is_none() {
            return Ok(());
        }
        if self.conv_weights.as_ref().unwrap()[layer].is_some() {
            return Ok(());
        }
        let name = self.conv_names.as_ref().unwrap()[layer]
            .clone()
            .expect("checked above");
        let info = self.catalog.tensor(&name)?.clone();
        let data = self.catalog.dequant_f32(&name)?;
        // GGUF layout `[ne0=kernel, ne1=channels]`: linear index `c*kernel + t`.
        // `channels` may differ from `hidden` (a conv over a projected subspace);
        // the caller must pass the matching buffer to `apply_conv`.
        let k = info.ncols().max(1);
        let ch = info.nrows().max(1);
        if data.len() < k * ch {
            return Err(StreamInferError::Msg(format!(
                "conv tensor {name}: {} values < kernel({k})*channels({ch})",
                data.len()
            )));
        }
        self.io_bytes += (data.len() * 4) as u64;
        self.conv_weights.as_mut().unwrap()[layer] = Some((k, ch, data[..k * ch].to_vec()));
        self.conv_states.as_mut().unwrap()[layer] = Some(vec![0.0f32; (k - 1) * ch]);
        Ok(())
    }

    /// Apply the optional short-conv residual to an explicit activation buffer
    /// (`x = x + silu(depthwise_causal_conv(x))`).
    pub(crate) fn apply_conv(
        &mut self,
        layer: usize,
        x: &mut [f32],
    ) -> Result<(), StreamInferError> {
        self.ensure_conv_layer(layer)?;
        let Some(Some((k, ch, w))) = self.conv_weights.as_ref().and_then(|v| v.get(layer)) else {
            return Ok(());
        };
        let (k, ch) = (*k, *ch);
        if x.len() != ch {
            return Err(StreamInferError::Msg(format!(
                "conv layer {layer}: activation len {} != channels {ch}",
                x.len()
            )));
        }
        let state = self.conv_states.as_mut().unwrap()[layer]
            .as_mut()
            .unwrap();
        apply_depthwise_conv(x, state, k, w.as_slice(), self.conv_activation);
        Ok(())
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
                self.scale_embed(dst);
                // No disk I/O: resident embedding. Do not inflate `io_bytes`.
                return Ok(());
            }
        }
        let t0 = Instant::now();
        self.catalog.read_embed_row(embd_name, token, h, dst)?;
        self.scale_embed(dst);
        self.io_secs += t0.elapsed().as_secs_f64();
        self.io_bytes += (h * 2) as u64;
        Ok(())
    }

    /// Granite `embedding_multiplier` (`embedding_scale`); no-op for other families.
    #[inline]
    fn scale_embed(&self, dst: &mut [f32]) {
        if self.embedding_scale != 1.0 {
            for v in dst.iter_mut() {
                *v *= self.embedding_scale;
            }
        }
    }

    /// Final-logit multiplier (Granite `logits_scaling` ÷, Cohere2 `logit_scale` ×).
    #[inline]
    pub(crate) fn apply_logit_scale(&self, logits: &mut [f32]) {
        if self.logit_scale != 1.0 {
            for v in logits.iter_mut() {
                *v *= self.logit_scale;
            }
        }
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
            let (pack, layout) = self.pack_views_from_base(layer, base)?;
            self.io_secs += t0.elapsed().as_secs_f64();
            // Views over resident memory: no disk read, so no `io_bytes`.
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
                    let (_, layout) = self.load_pack_into(&mut cat, l, dst)?;
                    self.io_secs += t0.elapsed().as_secs_f64();
                    self.io_bytes += layout.total as u64;
                }
                scratch.mark_block_staged(bslot, block);
            }
            if !has_hybrid {
                let t0 = Instant::now();
                let base = scratch.host_slot(layer);
                let (pack, layout) = self.pack_views_from_base(layer, base)?;
                self.io_secs += t0.elapsed().as_secs_f64();
                // Views over the already-staged block: no disk read.
                return Ok((pack, layout));
            }
            // has_hybrid: continúa al ping-pong por capa de abajo.
        }
        let t_map = Instant::now();
        scratch.prepare_host_write(&orch.pool, slot)?;
        self.map_secs += t_map.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let fused = self.fused_qkv_dims(layer);
        let dst = scratch.host_slot_mut(slot);
        let (pack, layout) = match fused {
            Some((q, kv)) => self.catalog.load_layer_pack_into_fused(layer, dst, q, kv)?,
            None => self.catalog.load_layer_pack_into(layer, dst)?,
        };
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
        // GPT-2: add the learned absolute position embedding for this position.
        if let Some(pos_emb) = &self.learned_pos {
            let base = self.position * h;
            if base + h <= pos_emb.len() {
                for i in 0..h {
                    embed_buf[i] += pos_emb[base + i];
                }
            }
        }
        // BLOOM: LayerNorm the input embeddings before the first block.
        if let Some((w, b)) = &self.token_embed_norm {
            layernorm_inplace(&mut embed_buf, w, Some(b.as_slice()), self.config.rms_norm_eps);
        }
        self.act_pp[0].copy_from_slice(&embed_buf);

        // Macro-chunk decode: blocks of `block_k` layers per I/O batch.
        if let Some(sc) = scratch.as_mut() {
            if sc.block_k > 1 && !sc.resident && override_ffn.is_none() && !self.simple_dense {
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
                    (current, layout) = self.pack_views_from_base(layer_idx, base)?;
                    // Resident: views over preloaded memory, no disk read.
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
                let next_fused = self.fused_qkv_dims(next);
                if let Some(sc) = scratch.as_mut() {
                    let next_slot = (layer_idx + 1) % 2;
                    let mut slot_ptr = self.prepare_prefetch_slot(orch, sc, next_slot)?;
                    if let Some(m) = self.owned_mem.as_mut() {
                        m.note_prefetch_staging(0);
                    }
                    prefetch = Some(thread::spawn(move || {
                        let dst = unsafe { slot_ptr.as_mut_slice() };
                        match next_fused {
                            Some((q, kv)) => cat.load_layer_pack_into_fused(next, dst, q, kv),
                            None => cat.load_layer_pack_into(next, dst),
                        }
                    }));
                } else {
                    // No scratch: the prefetch must return **owned** matrices.
                    // Returning views into a `blob` local to this closure would
                    // dangle the moment the closure returns (the pack is used by
                    // the main thread afterwards). The owned loaders copy every
                    // tensor into its own allocation, so the pack owns its bytes.
                    prefetch = Some(thread::spawn(move || match next_fused {
                        Some((q, kv)) => {
                            let pack = cat.load_layer_pack_fused(next, q, kv)?.0;
                            let lay = pack.layout();
                            Ok((pack, lay))
                        }
                        None => {
                            let pack = cat.load_layer_pack(next)?;
                            let lay = pack.layout();
                            Ok((pack, lay))
                        }
                    }));
                }
            }

            // --- CPU Attention (∥ DMA already enqueued) ---
            let t_attn = Instant::now();
            let sel = self.act_sel;
            self.scratch_xn.resize(h, 0.0);
            self.scratch_xn.copy_from_slice(&self.act_pp[sel]);
            apply_norm(
                &mut self.scratch_xn,
                &self.layer_norms[layer_idx].attn_norm,
                &self.layer_norms[layer_idx].attn_norm_bias,
                eps,
                self.use_layernorm,
            );

            let q_dim = self.attn_cfg.hidden_size();
            let kv_dim = self.attn_cfg.kv_dim();
            self.scratch_q.resize(q_dim, 0.0);
            self.scratch_k.resize(kv_dim, 0.0);
            self.scratch_v.resize(kv_dim, 0.0);
            current.wq.gemv(&self.scratch_xn, &mut self.scratch_q)?;
            current.wk.gemv(&self.scratch_xn, &mut self.scratch_k)?;
            current.wv.gemv(&self.scratch_xn, &mut self.scratch_v)?;
            if let Some(b) = self.attn_bias.get(layer_idx) {
                add_qkv_bias(b, &mut self.scratch_q, &mut self.scratch_k, &mut self.scratch_v);
            }

            self.scratch_attn.resize(q_dim, 0.0);
            attention_decode_step_ex(
                &self.attn_cfg,
                &mut self.kv[layer_idx],
                &mut self.scratch_q,
                &mut self.scratch_k,
                &self.scratch_v,
                pos,
                &mut self.scratch_attn,
                true,
                self.rope_freq_factors.as_deref(),
                self.alibi_slopes.as_deref(),
                self.layer_apply_rope.get(layer_idx).copied().unwrap_or(true),
            );
            if let Some(ref gate_w) = current.attn_gate {
                self.scratch_gate.resize(q_dim, 0.0);
                gate_w.gemv(&self.scratch_xn, &mut self.scratch_gate)?;
                for i in 0..q_dim {
                    self.scratch_attn[i] *= 1.0 / (1.0 + (-self.scratch_gate[i]).exp());
                }
            }
            self.scratch_proj.resize(h, 0.0);
            current.wo.gemv(&self.scratch_attn, &mut self.scratch_proj)?;
            if let Some(b) = self.attn_bias.get(layer_idx) {
                add_bias(&mut self.scratch_proj, &b.o);
            }
            if !self.parallel_residual {
                let x = &mut self.act_pp[sel];
                for i in 0..h {
                    x[i] += self.residual_scale * self.scratch_proj[i];
                }
            }
            self.attn_secs += t_attn.elapsed().as_secs_f64();
            // Optional depthwise causal short-conv residual (generic `Conv` op),
            // applied after attention and before the FFN norm.
            self.ensure_conv_layer(layer_idx)?;
            if let Some(Some((k, ch, w))) =
                self.conv_weights.as_ref().and_then(|v| v.get(layer_idx))
            {
                let (k, ch) = (*k, *ch);
                let act = self.conv_activation;
                let state = self.conv_states.as_mut().unwrap()[layer_idx]
                    .as_mut()
                    .unwrap();
                let x = &mut self.act_pp[sel];
                if x.len() == ch {
                    apply_depthwise_conv(x, state, k, w.as_slice(), act);
                } else {
                    return Err(StreamInferError::Msg(format!(
                        "conv layer {layer_idx}: tensor channels {ch} != residual len {}; a \
                         non-residual conv needs a family-specific path (no silent skip)",
                        x.len()
                    )));
                }
            }

            // --- FFN: unmap after Attn, then enqueue async ---
            // Parallel residual (Phi-2/GPT-J): the FFN reuses the attention-normed input.
            if !self.parallel_residual {
                self.scratch_xn.copy_from_slice(&self.act_pp[sel]);
                apply_norm(
                    &mut self.scratch_xn,
                    &self.layer_norms[layer_idx].ffn_norm,
                    &self.layer_norms[layer_idx].ffn_norm_bias,
                    eps,
                    self.use_layernorm,
                );
            }

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
            // Ungated FFN (GPT-2/BLOOM/Falcon: `up → act → down`, no gate).
            let ungated = current.gate.nrows == 0 && current.up.nrows > 0;
            let inflight = if has_ov || sparse_ffn || ungated {
                None
            } else {
                Some(ffn_begin_gate_up_scratch(
                    orch,
                    &current.gate,
                    &current.up,
                    &self.scratch_xn,
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
            } else if ungated {
                // CPU ungated FFN: `down(act(up(x) + b_up)) + b_down`.
                let mut up = vec![0.0f32; current.up.nrows];
                current.up.gemv(&self.scratch_xn, &mut up)?;
                if let Some(lb) = self.ffn_bias.get(layer_idx) {
                    add_bias(&mut up, &lb.up);
                }
                for v in up.iter_mut() {
                    *v = gelu(*v);
                }
                self.ws_down.resize(current.down.nrows, 0.0);
                current.down.gemv(&up, &mut self.ws_down)?;
                if let Some(lb) = self.ffn_bias.get(layer_idx) {
                    add_bias(&mut self.ws_down, &lb.down);
                }
            } else {
                // FFN disperso embebido (D16) u override de evolución: CSR en GPU/CPU.
                let xn = self.scratch_xn.clone();
                self.run_ffn_block(orch, &current, &xn, ov)?;
            }
            self.ffn_secs += t_ffn_fin.elapsed().as_secs_f64();

            {
                let sel = self.act_sel;
                if self.parallel_residual {
                    // `x = x + attn(ln(x)) + ffn(ln(x))` (both from the shared norm).
                    for i in 0..h {
                        self.act_pp[sel][i] +=
                            self.residual_scale * (self.scratch_proj[i] + self.ws_down[i]);
                    }
                } else {
                    for i in 0..h {
                        self.act_pp[sel][i] += self.residual_scale * self.ws_down[i];
                    }
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
        apply_norm(&mut xn, &self.output_norm, &self.output_norm_bias, eps, self.use_layernorm);
        self.last_hidden_nextn.clear();
        self.last_hidden_nextn.extend_from_slice(&xn);
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
        if std::env::var("HAYAI_DUMP_TOP").ok().as_deref() == Some("1") {
            let at = std::env::var("HAYAI_DUMP_AT_POS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
            if self.position == at {
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
                let top: Vec<String> = idx[..8].iter().map(|&i| format!("{}:{:.3}", i, logits[i])).collect();
                eprintln!("DENSE_DUMP_TOP pos={}: {}", self.position, top.join(" "));
            }
        }
        self.apply_logit_scale(&mut logits);
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
                let mut slot_ptr = PrefetchSlotPtr::new(ptr, scratch.slot_capacity());
                let stride = scratch.resident_stride;
                let mut cat = self.catalog.fork_reader()?;
                let fused: Vec<Option<(usize, usize)>> = (next_bs..next_be)
                    .map(|l| self.fused_qkv_dims(l))
                    .collect();
                prefetch = self.io_worker.run(move || -> Result<(), GgufError> {
                    let dst = unsafe { slot_ptr.as_mut_slice() };
                    for l in next_bs..next_be {
                        let off = (l - next_bs) * stride;
                        let end = (off + stride).min(dst.len());
                        match fused[l - next_bs] {
                            Some((q, kv)) => {
                                cat.load_layer_pack_into_fused(l, &mut dst[off..end], q, kv)?;
                            }
                            None => {
                                cat.load_layer_pack_into(l, &mut dst[off..end])?;
                            }
                        }
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
                let (current, layout) = self.pack_views_from_base(layer, base)?;
                // DMA this layer's FFN slices into every dGPU mirror (resolved at the
                // layer's in-block offset) so the GPU FFN reads fresh weights instead
                // of a stale mirror. Enqueued before Attn to overlap the copy.
                self.begin_ffn_dma(orch, scratch, layer, &layout)?;

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
                if let Some(b) = self.attn_bias.get(layer) {
                    add_qkv_bias(b, &mut q, &mut kk, &mut v);
                }

                let mut attn_out = vec![0.0f32; q_dim];
                attention_decode_step_ex(
                    &self.attn_cfg,
                    &mut self.kv[layer],
                    &mut q,
                    &mut kk,
                    &v,
                    pos,
                    &mut attn_out,
                    true,
                    self.rope_freq_factors.as_deref(),
                    self.alibi_slopes.as_deref(),
                    self.layer_apply_rope.get(layer).copied().unwrap_or(true),
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
                if let Some(b) = self.attn_bias.get(layer) {
                    add_bias(&mut attn_proj, &b.o);
                }
                {
                    let rs = self.residual_scale;
                    let x = self.act_mut();
                    for i in 0..h {
                        x[i] += rs * attn_proj[i];
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
                        self.act_pp[sel][i] += self.residual_scale * self.ws_down[i];
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
        apply_norm(&mut xn, &self.output_norm, &self.output_norm_bias, eps, self.use_layernorm);
        self.last_hidden_nextn.clear();
        self.last_hidden_nextn.extend_from_slice(&xn);
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
        self.apply_logit_scale(&mut logits);
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
            let (_, layout) = self.load_pack_into(&mut cat, l, dst)?;
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
                    self.exec_plan = Some(std::sync::Arc::new(plan));
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
        // Reserve KV cache + activations + metadata before sizing the weight
        // window so the 75–80 % target covers the whole Hayai-owned footprint.
        let reserve_bytes = crate::metrics::StreamingMemoryBudget::kv_activation_bytes(
            &self.config,
            self.kv_sinks,
            self.kv_window,
        ) + 64 * 1024 * 1024;
        let win = compute_window_plan(
            &orch.pool,
            full_bytes,
            self.config.num_layers,
            self.memory_strategy,
            reserve_bytes,
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
        // Two ping-pong slots of `k_chunk` layers each (resident → k_chunk = n_layers).
        let slots = if resident {
            self.config.num_layers
        } else {
            win.k_chunk
        };
        let mut scratch_bytes = 2u64
            .saturating_mul(slots as u64)
            .saturating_mul(layer_bytes as u64);
        // Resident mode also keeps the final projection and embedding resident.
        if resident {
            if let Some(ow) = &self.resident_output {
                scratch_bytes = scratch_bytes.saturating_add(ow.nbytes() as u64);
            }
            if let Some(emb) = &self.resident_embed {
                scratch_bytes = scratch_bytes.saturating_add(emb.nbytes() as u64);
            }
        }
        owned.note_scratch(scratch_bytes);
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
        // Mamba has no llama-shaped layer pack (its weights live in `mamba_cache`).
        if resident && self.model_kind() != ModelKind::Mamba {
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
                                let (_, layout) = self.load_pack_into(&mut cat, i, dst)?;
                                layout.total
                            }
                        } else {
                            let (_, layout) = self.load_pack_into(&mut cat, i, dst)?;
                            layout.total
                        }
                    } else {
                        let (_, layout) = self.load_pack_into(&mut cat, i, dst)?;
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
    ///
    /// Uses the **same** detection as [`ModelKind::from_catalog`] (which `open()`
    /// uses to size the KV caches), then falls back to the stored `ExecPlan` for
    /// ops whose tensor names the probe does not recognize.
    pub fn model_kind(&self) -> ModelKind {
        let catalog_kind = ModelKind::from_catalog(&self.catalog);
        if catalog_kind != ModelKind::Dense {
            return catalog_kind;
        }
        if let Some(p) = &self.exec_plan {
            let has = |op: crate::LayerOpKind| p.known_ops.contains(&op);
            if has(crate::LayerOpKind::Mamba) {
                ModelKind::Mamba
            } else if has(crate::LayerOpKind::MlaKvA) || has(crate::LayerOpKind::MlaQa) {
                ModelKind::Mla
            } else if has(crate::LayerOpKind::DeltaNet) {
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
    /// Prefill a multimodal input sequence (text tokens + media embedding rows).
    /// Currently supported on Gemma-family models (vision/audio injection).
    pub fn prefill_media(
        &mut self,
        orch: &mut EngineOrchestrator,
        items: &[MediaInput],
        scratch: &mut hayai_opencl::StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        if items.is_empty() {
            return Err(StreamInferError::Msg("empty multimodal prompt".into()));
        }
        if self.model_kind() != ModelKind::Gemma {
            return Err(StreamInferError::Msg(
                "multimodal input is currently supported on Gemma models only".into(),
            ));
        }
        let mut last = Vec::new();
        for it in items {
            last = match it {
                MediaInput::Token(t) => {
                    crate::gemma_infer::forward_gemma(self, orch, *t, scratch)?
                }
                MediaInput::Emb(e) => {
                    crate::gemma_infer::forward_gemma_embd(self, orch, e, scratch)?
                }
            };
        }
        Ok(last)
    }

    /// Generate from a multimodal input sequence (see [`Self::prefill_media`]).
    pub fn generate_media(
        &mut self,
        orch: &mut EngineOrchestrator,
        items: &[MediaInput],
        max_new_tokens: usize,
    ) -> Result<GenerateStats, StreamInferError> {
        let prompt_tokens = items.len();
        let mut scratch = self.prepare_session(orch)?;
        let wall0 = Instant::now();
        let mut last_logits = self.prefill_media(orch, items, &mut scratch)?;
        if let Some(g) = &self.grammar {
            self.grammar_state = Some(g.initial_state());
        }
        let mut all_new = Vec::new();
        for _ in 0..max_new_tokens {
            if self.grammar.is_some() {
                self.mask_logits_with_grammar(&mut last_logits);
            }
            let next = sample_with(
                &last_logits,
                self.sampler,
                &mut self.rng,
                &self.penalties,
                &all_new,
            );
            self.advance_grammar(next);
            if self.tokenizer.is_stop(next) {
                break;
            }
            all_new.push(next);
            last_logits = self.decode_step(orch, next, &mut scratch)?;
        }
        self.wall_compute_secs = wall0.elapsed().as_secs_f64();
        Ok(GenerateStats {
            text: self.tokenizer.decode(&all_new),
            prompt_tokens,
            new_tokens: all_new.len(),
            total_positions: self.position,
        })
    }

    pub fn prefill(
        &mut self,
        orch: &mut EngineOrchestrator,
        prompt_ids: &[u32],
        scratch: &mut hayai_opencl::StreamingScratch,
    ) -> Result<Vec<f32>, StreamInferError> {
        if prompt_ids.is_empty() {
            return Err(StreamInferError::Msg("empty prompt tokenization".into()));
        }
        // LongRoPE: pick short/long per-dim factors by sequence length (Phi-3-128k).
        if let Some((short, long, orig)) = &self.longrope {
            self.rope_freq_factors = Some(if prompt_ids.len() > *orig {
                long.clone()
            } else {
                short.clone()
            });
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
                ModelKind::Mamba => {
                    crate::mamba_infer::prefill_mamba(self, orch, prompt_ids, scratch)
                }
                ModelKind::Mla => {
                    crate::moe_infer::prefill_moe(self, orch, prompt_ids, scratch)
                }
                ModelKind::Dense => {
                    // Classic-transformer Dense features (LayerNorm/positions/ungated
                    // FFN) run through the per-token Dense path.
                    if self.simple_dense {
                        let mut last = Vec::new();
                        for &tok in prompt_ids {
                            last = self.forward_inner(orch, tok, Some(scratch), None)?;
                        }
                        Ok(last)
                    } else if std::env::var("HAYAI_PREFILL_BATCH").ok().as_deref() == Some("1") {
                        self.prefill_batched(orch, prompt_ids, scratch)
                    } else {
                        self.prefill_wavefront(orch, prompt_ids, scratch)
                    }
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
                ModelKind::Mamba => self.forward_mamba(orch, token),
                ModelKind::Mla => crate::moe_infer::forward_moe(self, orch, token, scratch),
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


    /// Enable grammar-constrained decoding. Builds the per-token text cache and
    /// resets the live grammar state to the grammar's initial state.
    pub fn set_grammar(&mut self, grammar: hayai_model::Grammar) {
        self.grammar_token_texts = self.tokenizer.token_texts();
        self.grammar_state = Some(grammar.initial_state());
        self.grammar = Some(grammar);
    }

    pub fn clear_grammar(&mut self) {
        self.grammar = None;
        self.grammar_state = None;
        self.grammar_token_texts.clear();
    }

    /// Set the logits of every token that would leave the grammar to `-inf`.
    /// EOS is allowed only when the grammar may stop here.
    fn mask_logits_with_grammar(&self, logits: &mut [f32]) {
        let (Some(grammar), Some(state)) = (&self.grammar, &self.grammar_state) else {
            return;
        };
        for (id, text) in self.grammar_token_texts.iter().enumerate() {
            if id >= logits.len() {
                break;
            }
            if self.tokenizer.is_stop(id as u32) {
                if !grammar.is_accepting(state) {
                    logits[id] = f32::NEG_INFINITY;
                }
                continue;
            }
            match text {
                Some(t) if !t.is_empty() => {
                    if grammar.advance_str(state, t).is_none() {
                        logits[id] = f32::NEG_INFINITY;
                    }
                }
                // Special / partial-UTF-8 tokens emit no clean text: disallow.
                _ => logits[id] = f32::NEG_INFINITY,
            }
        }
    }

    /// Advance the live grammar state after `token` was sampled.
    fn advance_grammar(&mut self, token: u32) {
        let (Some(grammar), Some(state)) = (&self.grammar, self.grammar_state.take()) else {
            return;
        };
        if self.tokenizer.is_stop(token) {
            self.grammar_state = Some(state);
            return;
        }
        let next = self
            .grammar_token_texts
            .get(token as usize)
            .and_then(|t| t.as_deref())
            .and_then(|t| grammar.advance_str(&state, t));
        // `None` only if the mask was bypassed; keep the previous state.
        self.grammar_state = Some(next.unwrap_or(state));
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
        if let Some(g) = &self.grammar {
            self.grammar_state = Some(g.initial_state());
        }
        let mut all_new = Vec::new();
        for _ in 0..max_new_tokens {
            if self.grammar.is_some() {
                self.mask_logits_with_grammar(&mut last_logits);
            }
            let next = sample_with(
                &last_logits,
                self.sampler,
                &mut self.rng,
                &self.penalties,
                &all_new,
            );
            self.advance_grammar(next);
            if self.tokenizer.is_stop(next) {
                break;
            }
            all_new.push(next);
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

    /// Build per-layer CSR FFN overrides from the global sparse genome
    /// (`saor.sparse` + `saor.genome` + `saor.tau`). Returns `None` for dense
    /// GGUFs. The dense FFN weights are read once, pruned by the decoded CPPN
    /// topology and kept as CSR (research "Vía B" runtime path).
    pub fn build_global_sparse_overrides(
        &mut self,
    ) -> Result<Option<Vec<FfnOverride>>, StreamInferError> {
        use hayai_model::sparse_dag::{META_GENOME, META_SPARSE, META_TAU};
        let is_sparse = matches!(
            self.catalog.metadata.get(META_SPARSE),
            Some(hayai_model::MetadataValue::Bool(true))
        );
        if !is_sparse {
            return Ok(None);
        }
        let genome = self
            .catalog
            .metadata
            .get(META_GENOME)
            .and_then(|v| v.as_f32_array())
            .ok_or_else(|| StreamInferError::Msg("sparse model is missing saor.genome".into()))?;
        let tau = self
            .catalog
            .metadata
            .get(META_TAU)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| StreamInferError::Msg("sparse model is missing saor.tau".into()))?;
        let n_layers = self.config.num_layers;
        let mut out = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let y = hayai_model::layer_coord(layer, n_layers);
            let mut ov = FfnOverride::default();
            for (block, slot) in [("ffn_gate", 0u8), ("ffn_up", 1), ("ffn_down", 2)] {
                let name = format!("blk.{layer}.{block}.weight");
                if self.catalog.tensor(&name).is_err() {
                    continue;
                }
                let info = self.catalog.tensor(&name)?.clone();
                let nbytes = hayai_model::tensor_nbytes(&info)?;
                let mut buf = vec![0u8; nbytes];
                self.catalog.read_tensor_into(&name, &mut buf)?;
                let dense = hayai_model::gguf::dequantize(&info, &buf)?;
                let csr = hayai_model::sparse_layer_csr(
                    &genome,
                    tau,
                    y,
                    &dense,
                    info.ncols(),
                    info.nrows(),
                );
                match slot {
                    0 => ov.gate = Some(csr),
                    1 => ov.up = Some(csr),
                    _ => ov.down = Some(csr),
                }
            }
            out.push(ov);
        }
        Ok(Some(out))
    }

    /// Prefill with per-layer FFN overrides (Dense only, used by the global
    /// sparse genome runtime). Token-by-token; the wavefront path has no override.
    pub fn prefill_with_override(
        &mut self,
        orch: &mut EngineOrchestrator,
        prompt_ids: &[u32],
        scratch: &mut hayai_opencl::StreamingScratch,
        override_ffn: &[FfnOverride],
    ) -> Result<Vec<f32>, StreamInferError> {
        if prompt_ids.is_empty() {
            return Err(StreamInferError::Msg("empty prompt tokenization".into()));
        }
        if self.model_kind() != ModelKind::Dense {
            return Err(StreamInferError::Msg(
                "sparse-global override is only supported for Dense models".into(),
            ));
        }
        let mut last = Vec::new();
        for &tok in prompt_ids {
            last = self.forward_inner(orch, tok, Some(scratch), Some(override_ffn))?;
        }
        Ok(last)
    }

    /// Like [`Self::generate`] but with per-layer FFN overrides (Dense only).
    pub fn generate_with_override(
        &mut self,
        orch: &mut EngineOrchestrator,
        prompt: &str,
        max_new_tokens: usize,
        override_ffn: &[FfnOverride],
    ) -> Result<GenerateStats, StreamInferError> {
        let prompt_ids = self.tokenizer.encode(prompt, self.tokenizer.add_bos);
        if prompt_ids.is_empty() {
            return Err(StreamInferError::Msg("empty prompt tokenization".into()));
        }
        let prompt_len = prompt_ids.len();
        let mut scratch = self.prepare_session(orch)?;
        let wall0 = Instant::now();
        let mut last_logits =
            self.prefill_with_override(orch, &prompt_ids, &mut scratch, override_ffn)?;
        if let Some(g) = &self.grammar {
            self.grammar_state = Some(g.initial_state());
        }
        let mut all_new = Vec::new();
        for _ in 0..max_new_tokens {
            if self.grammar.is_some() {
                self.mask_logits_with_grammar(&mut last_logits);
            }
            let next = sample_with(
                &last_logits,
                self.sampler,
                &mut self.rng,
                &self.penalties,
                &all_new,
            );
            self.advance_grammar(next);
            if self.tokenizer.is_stop(next) {
                break;
            }
            all_new.push(next);
            last_logits = self.decode_step_with_override(orch, next, &mut scratch, override_ffn)?;
        }
        self.wall_compute_secs = wall0.elapsed().as_secs_f64();
        Ok(GenerateStats {
            text: self.tokenizer.decode(&all_new),
            prompt_tokens: prompt_len,
            new_tokens: all_new.len(),
            total_positions: self.position,
        })
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
        get_override: impl FnMut(usize, usize) -> FfnOverride,
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
        get_override: impl FnMut(
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
                attention_decode_step_ex(
                    &self.attn_cfg,
                    &mut kv[c][layer_idx],
                    &mut q_flat[c * q_dim..(c + 1) * q_dim],
                    &mut k_flat[c * kv_dim..(c + 1) * kv_dim],
                    &v_flat[c * kv_dim..(c + 1) * kv_dim],
                    pos,
                    &mut attn_out_flat[c * q_dim..(c + 1) * q_dim],
                    true,
                    self.rope_freq_factors.as_deref(),
                    self.alibi_slopes.as_deref(),
                    self.layer_apply_rope.get(layer_idx).copied().unwrap_or(true),
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
                attention_decode_step_ex(
                    &self.attn_cfg,
                    &mut kv[c][layer_idx],
                    &mut q_flat[idx * q_dim..(idx + 1) * q_dim],
                    &mut k_flat[idx * kv_dim..(idx + 1) * kv_dim],
                    &v_flat[idx * kv_dim..(idx + 1) * kv_dim],
                    t,
                    &mut attn_out_flat[idx * q_dim..(idx + 1) * q_dim],
                    true,
                    self.rope_freq_factors.as_deref(),
                    self.alibi_slopes.as_deref(),
                    self.layer_apply_rope.get(layer_idx).copied().unwrap_or(true),
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
    eng.ffn_calls
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    match scratch.weight_bind(eng, layer) {
        Ok(WeightBind::Svm { ptr }) => Ok(eng.ggml_gemv_async_from_svm(
            kernel,
            label,
            m.nrows,
            m.ncols,
            ptr,
            woff,
            xn,
        )?),
        Ok(WeightBind::Device { buf }) => Ok(eng.ggml_gemv_async_from_device(
            kernel,
            label,
            m.nrows,
            m.ncols,
            buf,
            woff,
            xn,
        )?),
        // No SVM ownership and no VRAM mirror (e.g. the mirror did not fit in
        // `CL_DEVICE_MAX_MEM_ALLOC_SIZE`, or the pool had no mirror): upload the
        // host bytes for this GEMV instead of failing. Correct, just slower.
        Err(_) => begin_gemv(eng, m, xn),
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

    let gate_eng = orch
        .pool
        .for_role(0)
        .ok_or_else(|| StreamInferError::Msg("empty GPU pool".into()))?;
    let up_eng = orch
        .pool
        .for_role(1)
        .ok_or_else(|| StreamInferError::Msg("empty GPU pool".into()))?;
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
        let eng = orch
            .pool
            .for_role(2)
            .ok_or_else(|| StreamInferError::Msg("empty GPU pool".into()))?;
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
        let eng = orch
            .pool
            .for_role(2)
            .ok_or_else(|| StreamInferError::Msg("empty GPU pool".into()))?;
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
    // Attention-less families (Mamba SSM) report head_count=0; clamp so the config
    // is harmless (their forward never uses it) and never divides by zero.
    let n_heads = config.num_attention_heads.max(1);
    let n_kv = config.num_key_value_heads.max(1);
    let q_nrows = (0..config.num_layers).find_map(|i| {
        cat.tensor(&format!("blk.{i}.attn_q.weight"))
            .ok()
            .map(|t| t.nrows())
    });
    let head_dim = if head_dim_meta > 0 {
        head_dim_meta
    } else if let Some(qn) = q_nrows {
        (qn / n_heads).max(1)
    } else {
        (config.hidden_size / n_heads).max(1)
    };
    let mut cfg = AttentionConfig::from_model(
        head_dim * n_heads,
        n_heads,
        n_kv,
        config.rope_theta,
    );
    cfg.head_dim = head_dim;
    // ALiBi bias clamp (`mpt.attention.max_alibi_bias`); 0 disables (BLOOM/Falcon).
    cfg.alibi_max_bias = cat
        .meta_f32(&format!("{}.attention.max_alibi_bias", config.architecture))
        .or_else(|| cat.meta_f32("mpt.attention.max_alibi_bias"))
        .unwrap_or(0.0);
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
    // Architecture attention scale (`attention_multiplier`, Granite): replaces the
    // default `1/sqrt(head_dim)`.
    if let Some(s) = cat
        .meta_f32(&format!("{}.attention.scale", config.architecture))
        .or_else(|| cat.meta_f32(&format!("{}.attention_multiplier", config.architecture)))
    {
        cfg.scale_override = Some(s);
    }
    // RoPE scaling: linear (`freq_scale = 1/factor`) and YaRN (`ext_factor`/mscale).
    let arch = &config.architecture;
    let rs_type = cat
        .meta_str(&format!("{arch}.rope.scaling.type"))
        .or_else(|| cat.meta_str("llama.rope.scaling.type"))
        .unwrap_or("none");
    let rs_factor = cat
        .meta_f32(&format!("{arch}.rope.scaling.factor"))
        .or_else(|| cat.meta_f32("llama.rope.scaling.factor"))
        .unwrap_or(1.0);
    match rs_type {
        "linear" => {
            cfg.rope = hayai_cpu::RopeScaling {
                freq_scale: 1.0 / rs_factor,
                ..hayai_cpu::RopeScaling::NONE
            };
        }
        "yarn" => {
            let n_ctx_orig = cat
                .meta_u32(&format!("{arch}.rope.scaling.original_context_length"))
                .or_else(|| cat.meta_u32(&format!("{arch}.context_length")))
                .unwrap_or(0) as usize;
            cfg.rope = hayai_cpu::RopeScaling {
                freq_scale: 1.0 / rs_factor,
                ext_factor: 1.0,
                attn_factor: cat
                    .meta_f32(&format!("{arch}.rope.scaling.attn_factor"))
                    .unwrap_or(1.0),
                beta_fast: cat
                    .meta_f32(&format!("{arch}.rope.scaling.beta_fast"))
                    .unwrap_or(32.0),
                beta_slow: cat
                    .meta_f32(&format!("{arch}.rope.scaling.beta_slow"))
                    .unwrap_or(1.0),
                n_ctx_orig,
                // Convert-script stores `yarn_log_multiplier * 0.1`; llama.cpp cancels it.
                yarn_log_mul: cat
                    .meta_f32(&format!("{arch}.rope.scaling.yarn_log_multiplier"))
                    .map(|v| v / 0.1)
                    .unwrap_or(0.0),
            };
        }
        _ => {}
    }
    // LongRoPE (Phi-3-128k): per-dim `rope_factors_{short,long}` tensors + `attn_factor`.
    // The per-dim factors are passed to the attention call; here we set the cos/sin
    // `attn_factor` (mscale) and the original context length.
    if cat.tensor("rope_factors_short.weight").is_ok() {
        let attn_factor = cat
            .meta_f32(&format!("{arch}.rope.scaling.attn_factor"))
            .unwrap_or(1.0);
        let n_ctx_orig = cat
            .meta_u32(&format!("{arch}.rope.scaling.original_context_length"))
            .unwrap_or(4096) as usize;
        cfg.rope = hayai_cpu::RopeScaling {
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig,
            yarn_log_mul: 0.0,
        };
    }
    Ok(cfg)
}

/// Detect the parallel-residual + single shared norm layout (Phi-2/GPT-J/PaLM):
/// block 0 has `attn_norm` and `ffn_up` but no `ffn_norm`, so both sublayers read
/// the same normed activation and add to the residual together.
pub(crate) fn detect_parallel_residual(cat: &GgufCatalog) -> bool {
    cat.tensor("blk.0.ffn_norm.weight").is_err()
        && cat.tensor("blk.0.attn_norm.weight").is_ok()
        && cat.tensor("blk.0.ffn_up.weight").is_ok()
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
        .unwrap_or(0) as usize;
    // Physical `blk.N` count (HRM metadata may encode H×L×cycles ≫ resident blocks;
    // also lets unknown architectures load without a `block_count` key).
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
    if layers_meta == 0 && layers_physical == 0 {
        return Err(GgufError::MissingKey("block_count".into()));
    }
    let mut layers_n = if layers_physical > 0 && (layers_meta == 0 || layers_physical < layers_meta) {
        if layers_meta != 0 && layers_physical < layers_meta {
            tracing::info!(
                "block_count meta={layers_meta} physical_blk={layers_physical} — using physical"
            );
        }
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
    // `head_count_kv` may be a scalar OR a per-layer array (Granite: `[4,4,...]`);
    // take the first element for the generic Dense path (per-layer families resolve
    // it themselves, e.g. `GemmaMeta`).
    let n_kv = cat
        .metadata
        .get(&format!("{arch}.attention.head_count_kv"))
        .or_else(|| cat.metadata.get(&format!("{prefix}.attention.head_count_kv")))
        .or_else(|| cat.metadata.get("llama.attention.head_count_kv"))
        .or_else(|| cat.metadata.get("qwen2.attention.head_count_kv"))
        .and_then(|v| {
            v.as_u32()
                .or_else(|| v.as_u32_array().and_then(|a| a.first().copied()))
        })
        .unwrap_or(n_heads as u32) as usize;
    // HRM detection is op/tensor-driven: the recurrence marker tensor is authoritative,
    // the architecture name is only a fallback (another name with the same ops works).
    let hrm = if arch == "hrm_text" || arch.contains("hrm") || cat.tensor("hrm.z_l_init").is_ok() {
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
            prefix_lm: cat
                .metadata
                .get(&format!("{prefix}.prefix_lm"))
                .and_then(|v| match v {
                    hayai_model::MetadataValue::Bool(b) => Some(*b),
                    _ => None,
                })
                .unwrap_or(false),
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

    /// Parallel-residual detection: block 0 with `attn_norm` + `ffn_up` but no
    /// `ffn_norm` is the Phi-2/GPT-J single-norm layout; a separate `ffn_norm` is
    /// the sequential layout.
    #[test]
    fn detect_parallel_residual_layout() {
        let phi = temp_dir().join("hayai_parallel_residual.gguf");
        let t: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("blk.0.attn_norm.weight", vec![8], vec![1.0; 8]),
            ("blk.0.ffn_up.weight", vec![8, 16], vec![1.0; 128]),
            ("blk.0.ffn_down.weight", vec![16, 8], vec![1.0; 128]),
        ];
        write_minimal_gguf(
            &phi,
            &[("general.architecture", MetadataValue::String("phi2".into()))],
            &t,
        )
        .unwrap();
        let cat = GgufCatalog::open(&phi).unwrap();
        assert!(detect_parallel_residual(&cat));
        let _ = std::fs::remove_file(&phi);

        let seq = temp_dir().join("hayai_sequential_residual.gguf");
        let t2: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("blk.0.attn_norm.weight", vec![8], vec![1.0; 8]),
            ("blk.0.ffn_norm.weight", vec![8], vec![1.0; 8]),
            ("blk.0.ffn_up.weight", vec![8, 16], vec![1.0; 128]),
        ];
        write_minimal_gguf(
            &seq,
            &[("general.architecture", MetadataValue::String("llama".into()))],
            &t2,
        )
        .unwrap();
        let cat2 = GgufCatalog::open(&seq).unwrap();
        assert!(!detect_parallel_residual(&cat2));
        let _ = std::fs::remove_file(&seq);
    }

    /// `head_count_kv` may be a per-layer array (Granite): `load_config` takes the
    /// first element so the generic Dense path gets the right `n_kv`.
    #[test]
    fn load_config_head_count_kv_array() {
        let path = temp_dir().join("hayai_kv_array.gguf");
        write_minimal_gguf(
            &path,
            &[
                ("general.architecture", MetadataValue::String("granite".into())),
                ("granite.embedding_length", MetadataValue::U32(8)),
                ("granite.block_count", MetadataValue::U32(1)),
                ("granite.attention.head_count", MetadataValue::U32(4)),
                (
                    "granite.attention.head_count_kv",
                    MetadataValue::Array(vec![MetadataValue::U32(1), MetadataValue::U32(1)]),
                ),
            ],
            &[("token_embd.weight", vec![8, 8], vec![1.0f32; 64])],
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        let cfg = load_config(&cat).unwrap();
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.num_attention_heads, 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn depthwise_causal_conv_silu_residual() {
        // k=3, channels=2, weights w[c*k + t] (t=0 oldest).
        let w = [1.0f32, 2.0, 3.0, 0.5, 1.0, 1.5];
        let mut state = vec![0.0f32; 2 * 2]; // (k-1)*channels
        // Token 0: x = [1, 1]; taps 0,1 read zero history.
        let mut x = [1.0f32, 1.0];
        super::apply_depthwise_conv(&mut x, &mut state, 3, &w, ConvActivation::Silu);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        assert!((x[0] - (1.0 + silu(3.0))).abs() < 1e-5, "x0={}", x[0]);
        assert!((x[1] - (1.0 + silu(1.5))).abs() < 1e-5, "x1={}", x[1]);
        // History now holds the first input as the newest sample.
        assert_eq!(&state[2..4], &[1.0, 1.0]);
        // Token 1: x = [1, 1]; c0 = 1*0 + 2*1 + 3*1 = 5; c1 = 0.5*0 + 1*1 + 1.5*1 = 2.5.
        let mut x = [1.0f32, 1.0];
        super::apply_depthwise_conv(&mut x, &mut state, 3, &w, ConvActivation::Silu);
        assert!((x[0] - (1.0 + silu(5.0))).abs() < 1e-5, "x0={}", x[0]);
        assert!((x[1] - (1.0 + silu(2.5))).abs() < 1e-5, "x1={}", x[1]);
    }

    #[test]
    fn depthwise_conv_kernel_one_is_pointwise() {
        let w = [0.5f32, -1.0];
        let mut state: Vec<f32> = Vec::new();
        let mut x = [2.0f32, 4.0];
        super::apply_depthwise_conv(&mut x, &mut state, 1, &w, ConvActivation::Silu);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        assert!((x[0] - (2.0 + silu(1.0))).abs() < 1e-5);
        assert!((x[1] - (4.0 + silu(-4.0))).abs() < 1e-5);
        assert!(state.is_empty());
    }

    #[test]
    fn fused_qkv_detected_on_open_and_separate_not() {
        // llama-shaped 1-block model with a fused attn_qkv (concat [q|k|v]).
        let hidden = 8u64;
        let n_heads = 2u64;
        let n_kv = 1u64;
        let q_dim = (n_heads * 4) as usize;
        let kv_dim = (n_kv * 4) as usize;
        let inter = 16u64;
        let vocab = 16u64;
        let z = |n: u64| vec![0.0f32; n as usize];

        let path = temp_dir().join("hayai_fused_open.gguf");
        let tokens: Vec<MetadataValue> = (0..vocab)
            .map(|i| MetadataValue::String(format!("t{i}")))
            .collect();
        write_minimal_gguf(
            &path,
            &[
                ("general.architecture", MetadataValue::String("llama".into())),
                ("llama.block_count", MetadataValue::U32(1)),
                ("llama.embedding_length", MetadataValue::U32(hidden as u32)),
                ("llama.attention.head_count", MetadataValue::U32(n_heads as u32)),
                ("llama.attention.head_count_kv", MetadataValue::U32(n_kv as u32)),
                ("llama.attention.key_length", MetadataValue::U32(4)),
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
            ],
            &[
                ("token_embd.weight", vec![hidden, vocab], z(hidden * vocab)),
                ("output_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.attn_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.ffn_norm.weight", vec![hidden], z(hidden)),
                (
                    "blk.0.attn_qkv.weight",
                    vec![hidden, (q_dim + 2 * kv_dim) as u64],
                    z(hidden * (q_dim + 2 * kv_dim) as u64),
                ),
                (
                    "blk.0.attn_output.weight",
                    vec![q_dim as u64, hidden],
                    z(q_dim as u64 * hidden),
                ),
                ("blk.0.ffn_gate.weight", vec![hidden, inter], z(hidden * inter)),
                ("blk.0.ffn_up.weight", vec![hidden, inter], z(hidden * inter)),
                ("blk.0.ffn_down.weight", vec![inter, hidden], z(inter * hidden)),
            ],
        )
        .unwrap();
        let meta = {
            let cat = GgufCatalog::open(&path).unwrap();
            cat.metadata.clone()
        };
        let tok = hayai_model::Tokenizer::from_metadata(&meta).unwrap();
        let gen = StreamingGenerator::open(
            &path,
            tok,
            4,
            128,
            hayai_model::SamplerConfig::Greedy,
            42,
        )
        .unwrap();
        assert_eq!(gen.fused_qkv_dims(0), Some((q_dim, kv_dim)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn global_sparse_overrides_build_from_genome() {
        let hidden = 8u64;
        let n_heads = 2u64;
        let n_kv = 1u64;
        let inter = 16u64;
        let vocab = 16u64;
        let z = |n: u64| vec![0.0f32; n as usize];
        let genome: Vec<MetadataValue> =
            (0..hayai_model::cppn::genome_len()).map(|_| MetadataValue::F32(0.0)).collect();
        let tokens: Vec<MetadataValue> = (0..vocab)
            .map(|i| MetadataValue::String(format!("t{i}")))
            .collect();
        let path = temp_dir().join("hayai_sparse_global.gguf");
        let q_dim = n_heads * 4;
        let kv_dim = n_kv * 4;
        write_minimal_gguf(
            &path,
            &[
                ("general.architecture", MetadataValue::String("llama".into())),
                ("llama.block_count", MetadataValue::U32(1)),
                ("llama.embedding_length", MetadataValue::U32(hidden as u32)),
                ("llama.attention.head_count", MetadataValue::U32(n_heads as u32)),
                ("llama.attention.head_count_kv", MetadataValue::U32(n_kv as u32)),
                ("llama.attention.key_length", MetadataValue::U32(4)),
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
                ("saor.sparse", MetadataValue::Bool(true)),
                ("saor.tau", MetadataValue::F32(-1.0)),
                ("saor.genome", MetadataValue::Array(genome)),
            ],
            &[
                ("token_embd.weight", vec![hidden, vocab], z(hidden * vocab)),
                ("output_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.attn_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.ffn_norm.weight", vec![hidden], z(hidden)),
                (
                    "blk.0.attn_qkv.weight",
                    vec![hidden, q_dim + 2 * kv_dim],
                    z(hidden * (q_dim + 2 * kv_dim)),
                ),
                ("blk.0.attn_output.weight", vec![q_dim, hidden], z(q_dim * hidden)),
                ("blk.0.ffn_gate.weight", vec![hidden, inter], z(hidden * inter)),
                ("blk.0.ffn_up.weight", vec![hidden, inter], z(hidden * inter)),
                ("blk.0.ffn_down.weight", vec![inter, hidden], z(inter * hidden)),
            ],
        )
        .unwrap();
        let meta = {
            let cat = GgufCatalog::open(&path).unwrap();
            cat.metadata.clone()
        };
        let tok = hayai_model::Tokenizer::from_metadata(&meta).unwrap();
        let mut gen = StreamingGenerator::open(
            &path,
            tok,
            4,
            128,
            hayai_model::SamplerConfig::Greedy,
            42,
        )
        .unwrap();
        assert_eq!(gen.model_kind(), ModelKind::Dense);
        let ov = gen.build_global_sparse_overrides().unwrap().expect("sparse");
        assert_eq!(ov.len(), 1);
        let gate = ov[0].gate.as_ref().expect("gate csr");
        assert_eq!((gate.d_in, gate.d_out), (hidden as usize, inter as usize));
        assert_eq!(gate.vals.len(), (hidden * inter) as usize); // tau=-1 → all active
        let down = ov[0].down.as_ref().expect("down csr");
        assert_eq!((down.d_in, down.d_out), (inter as usize, hidden as usize));
        assert_eq!(down.vals.len(), (inter * hidden) as usize);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn depthwise_conv_is_causal_and_channel_generic() {
        // k=2, ch=3 (deliberately != hidden); per-channel taps [t_old, t_cur].
        let w = [0.5f32, 1.0, -1.0, 2.0, 0.25, 0.0];
        let mut state = vec![0.0f32; 3]; // (k-1) * ch
        let mut x = vec![1.0f32, 1.0, 1.0];
        apply_depthwise_conv(&mut x, &mut state, 2, &w, ConvActivation::Silu);
        // First step: only the current tap contributes.
        let silu = |a: f32| a / (1.0 + (-a).exp());
        assert!((x[0] - (1.0 + silu(1.0))).abs() < 1e-4);
        assert!((x[1] - (1.0 + silu(2.0))).abs() < 1e-4);
        assert!((x[2] - (1.0 + silu(0.0))).abs() < 1e-4);
        // History now holds the pre-activation inputs.
        assert_eq!(state, vec![1.0, 1.0, 1.0]);
        let mut x2 = vec![0.0f32, 0.0, 0.0];
        apply_depthwise_conv(&mut x2, &mut state, 2, &w, ConvActivation::Silu);
        // Second step: the current tap is 0, so only history survives (causal, t=0 oldest).
        assert!((x2[0] - silu(0.5)).abs() < 1e-4);
        assert!((x2[1] - silu(-1.0)).abs() < 1e-4);
        assert!((x2[2] - silu(0.25)).abs() < 1e-4);
    }

    #[test]
    fn conv_activation_none_is_identity() {
        let w = [0.0f32, 2.0]; // k=2, ch=1
        let mut state = vec![0.0f32; 1];
        let mut x = vec![3.0f32];
        apply_depthwise_conv(&mut x, &mut state, 2, &w, ConvActivation::None);
        assert!((x[0] - 9.0).abs() < 1e-6);
        assert!(ConvActivation::parse("gelu") == ConvActivation::Gelu);
        assert!(ConvActivation::parse("nonsense") == ConvActivation::Silu);
    }

    /// Framework invariant: execution is driven by **ops/tensors**, never by the
    /// `general.architecture` string. An unknown family name with llama-shaped
    /// tensors must load and run as `Dense`.
    #[test]
    fn unknown_architecture_loads_from_ops_not_name() {
        let path = temp_dir().join("hayai_unknown_arch.gguf");
        let hidden = 16u64;
        let vocab = 16u64;
        let inter = 32u64;
        let q_dim = 16u64; // 2 heads * head_dim 8
        let kv_dim = 16u64;
        let z = |n: u64| vec![0.0f32; n as usize];
        let tokens: Vec<MetadataValue> = (0..vocab)
            .map(|i| MetadataValue::String(format!("t{i}")))
            .collect();
        write_minimal_gguf(
            &path,
            &[
                (
                    "general.architecture",
                    MetadataValue::String("future_family".into()),
                ),
                ("future_family.block_count", MetadataValue::U32(1)),
                (
                    "future_family.embedding_length",
                    MetadataValue::U32(hidden as u32),
                ),
                ("future_family.attention.head_count", MetadataValue::U32(2)),
                (
                    "future_family.attention.head_count_kv",
                    MetadataValue::U32(2),
                ),
                ("future_family.attention.key_length", MetadataValue::U32(8)),
                (
                    "future_family.feed_forward_length",
                    MetadataValue::U32(inter as u32),
                ),
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
            ],
            &[
                ("token_embd.weight", vec![hidden, vocab], z(hidden * vocab)),
                ("output_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.attn_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.ffn_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.attn_q.weight", vec![hidden, q_dim], z(hidden * q_dim)),
                ("blk.0.attn_k.weight", vec![hidden, kv_dim], z(hidden * kv_dim)),
                ("blk.0.attn_v.weight", vec![hidden, kv_dim], z(hidden * kv_dim)),
                ("blk.0.attn_output.weight", vec![q_dim, hidden], z(q_dim * hidden)),
                ("blk.0.ffn_gate.weight", vec![hidden, inter], z(hidden * inter)),
                ("blk.0.ffn_up.weight", vec![hidden, inter], z(hidden * inter)),
                ("blk.0.ffn_down.weight", vec![inter, hidden], z(inter * hidden)),
            ],
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        assert_eq!(ModelKind::from_catalog(&cat), ModelKind::Dense);
        let cfg = load_config(&cat).unwrap();
        assert_eq!(cfg.hidden_size, hidden as usize);
        assert_eq!(cfg.num_layers, 1);
        assert_eq!(cfg.num_attention_heads, 2);
        let meta = cat.metadata.clone();
        let tok = hayai_model::Tokenizer::from_metadata(&meta).unwrap();
        let gen =
            StreamingGenerator::open(&path, tok, 4, 128, hayai_model::SamplerConfig::Greedy, 42)
                .unwrap();
        assert_eq!(gen.model_kind(), ModelKind::Dense);
        let _ = std::fs::remove_file(&path);
    }

    /// Mamba ops under an unknown family name must still route to the Mamba path.
    #[test]
    fn mamba_detected_from_ops_not_name() {
        let path = temp_dir().join("hayai_unknown_mamba.gguf");
        let hidden = 8u64;
        let vocab = 8u64;
        let z = |n: u64| vec![0.0f32; n as usize];
        let tokens: Vec<MetadataValue> = (0..vocab)
            .map(|i| MetadataValue::String(format!("t{i}")))
            .collect();
        write_minimal_gguf(
            &path,
            &[
                (
                    "general.architecture",
                    MetadataValue::String("future_ssm".into()),
                ),
                ("future_ssm.block_count", MetadataValue::U32(1)),
                ("future_ssm.embedding_length", MetadataValue::U32(hidden as u32)),
                ("tokenizer.ggml.tokens", MetadataValue::Array(tokens)),
                ("tokenizer.ggml.merges", MetadataValue::Array(vec![])),
                ("tokenizer.ggml.bos_token_id", MetadataValue::U32(1)),
                ("tokenizer.ggml.eos_token_id", MetadataValue::U32(2)),
            ],
            &[
                ("token_embd.weight", vec![hidden, vocab], z(hidden * vocab)),
                ("blk.0.attn_norm.weight", vec![hidden], z(hidden)),
                ("blk.0.ssm_in.weight", vec![hidden, 2 * hidden], z(hidden * 2 * hidden)),
                ("blk.0.ssm_x.weight", vec![hidden, 4], z(hidden * 4)),
                ("blk.0.ssm_d", vec![hidden], z(hidden)),
                ("blk.0.ssm_out.weight", vec![hidden, hidden], z(hidden * hidden)),
            ],
        )
        .unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        assert_eq!(ModelKind::from_catalog(&cat), ModelKind::Mamba);
        let _ = std::fs::remove_file(&path);
    }
}

