//! Encoder-decoder (T5 / BART-style) inference with deterministic weight streaming.
//!
//! The decoder-only [`crate::stream_infer::StreamingGenerator`] assumes a single
//! causal stack. T5 has two stacks: a bidirectional encoder and a causal decoder
//! with **cross-attention** over the encoder output, plus a learned **relative
//! position bias** (no RoPE / no absolute positions, and no `1/√d` score scaling).
//! This module implements that architecture class directly on top of the shared
//! [`crate::exec_plan::ExecPlan`] + GGUF slice loader, so weights still stream from
//! disk layer by layer (never `mmap`).
//!
//! Validated against HuggingFace `google/flan-t5-small` (top-k logits at a fixed
//! position). T5 FFN is the gated variant with the `gelu_new` (tanh) activation;
//! the GGUF does not record `feed_forward_proj`, so that is the assumed default.

use std::fmt;
use std::path::Path;

use hayai_cpu::{rms_norm, softmax};
use hayai_model::{GgufCatalog, GgufError, QuantMatrix, Tokenizer};

use crate::exec_plan::{build_exec_plan, LayerOpKind, StreamingUnit, TensorRef};

/// Disjoint block-id base used by [`crate::exec_plan`] for the decoder stack.
pub(crate) const DEC_BLOCK_OFFSET: usize = 100_000;

#[derive(Debug)]
pub enum EdeError {
    Gguf(GgufError),
    Msg(String),
}

impl fmt::Display for EdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdeError::Gguf(e) => write!(f, "gguf: {e}"),
            EdeError::Msg(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for EdeError {}

impl From<GgufError> for EdeError {
    fn from(e: GgufError) -> Self {
        EdeError::Gguf(e)
    }
}

pub type Result<T> = std::result::Result<T, EdeError>;

/// T5 model geometry resolved from GGUF metadata / tensor shapes.
#[derive(Debug, Clone)]
pub struct T5Config {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub vocab: usize,
    pub eps: f32,
    pub n_buckets: usize,
    pub max_distance: usize,
    pub decoder_start: u32,
    pub eos_id: u32,
    pub pad_id: u32,
}

/// One streamable unit plus the views needed to execute it.
struct Unit {
    tensors: Vec<TensorRef>,
    total_bytes: usize,
}

impl Unit {
    fn from_streaming(u: &StreamingUnit) -> Self {
        Self {
            tensors: u.tensors.clone(),
            total_bytes: u.total_bytes,
        }
    }

    fn view(&self, base: &[u8], op: LayerOpKind) -> Result<QuantMatrix> {
        let t = self
            .tensors
            .iter()
            .find(|t| t.op == op)
            .ok_or_else(|| EdeError::Msg(format!("T5 unit missing {op:?}")))?;
        Ok(QuantMatrix::view(
            t.name.clone(),
            t.ncols,
            t.nrows,
            t.ggml_type,
            &base[t.offset..t.offset + t.nbytes],
        ))
    }

    fn specs(&self) -> Vec<(&str, usize, usize, usize)> {
        self.tensors
            .iter()
            .map(|t| (t.name.as_str(), t.src_off, t.offset, t.nbytes))
            .collect()
    }
}

/// A per-layer encoder unit's seven projections.
struct EncViews {
    q: QuantMatrix,
    k: QuantMatrix,
    v: QuantMatrix,
    o: QuantMatrix,
    gate: QuantMatrix,
    up: QuantMatrix,
    down: QuantMatrix,
}

/// A per-layer decoder unit's projections (self-attn + cross-attn + FFN).
struct DecViews {
    q: QuantMatrix,
    k: QuantMatrix,
    v: QuantMatrix,
    o: QuantMatrix,
    cq: QuantMatrix,
    co: QuantMatrix,
    gate: QuantMatrix,
    up: QuantMatrix,
    down: QuantMatrix,
}

const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// `gelu_new`: the tanh approximation used by T5's `gated-gelu` FFN.
#[inline]
fn gelu_new(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x)).tanh())
}

/// HF `T5Attention._relative_position_bucket` for one `memory - query` distance.
fn rel_bucket(mut rel: i64, bidirectional: bool, num_buckets: usize, max_distance: usize) -> usize {
    let mut nb = num_buckets;
    let mut buckets: i64 = 0;
    if bidirectional {
        nb /= 2;
        if rel > 0 {
            buckets += nb as i64;
        }
        rel = rel.abs();
    } else {
        rel = -rel.min(0);
    }
    let max_exact = nb / 2;
    let is_small = (rel as usize) < max_exact;
    if is_small {
        return (buckets + rel) as usize;
    }
    let rp = rel as f64;
    let large = max_exact as f64
        + (rp / max_exact as f64).ln() / (max_distance as f64 / max_exact as f64).ln()
            * ((nb - max_exact) as f64);
    let large = (large as i64).min((nb - 1) as i64);
    (buckets + large) as usize
}

/// Loaded T5 model. Weights stream from disk per layer; only the small norm /
/// relative-bias tables and the embedding/output matrices are resident.
pub struct T5Model {
    pub cat: GgufCatalog,
    pub cfg: T5Config,
    enc: Vec<Unit>,
    dec: Vec<Unit>,
    unit_buf: Vec<u8>,
    /// Per encoder layer `(attn_norm, ffn_norm)`.
    enc_norms: Vec<(Vec<f32>, Vec<f32>)>,
    /// Per decoder layer `(attn_norm, cross_attn_norm, ffn_norm)`.
    dec_norms: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    enc_out_norm: Vec<f32>,
    dec_out_norm: Vec<f32>,
    /// Relative position bias tables, `[bucket * n_heads + head]`.
    enc_rel_bias: Vec<f32>,
    dec_rel_bias: Vec<f32>,
    output: QuantMatrix,
}

impl T5Model {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut cat = GgufCatalog::open(path)?;
        let arch = cat.meta_str("general.architecture").unwrap_or("t5").to_string();
        let md = |c: &GgufCatalog, k: &str| {
            c.meta_u32(&format!("{arch}.{k}"))
                .or_else(|| c.meta_u32(&format!("t5.{k}")))
                .or_else(|| c.meta_u32(&format!("bart.{k}")))
        };
        let d_model = md(&cat, "embedding_length").unwrap_or(512) as usize;
        let d_ff = md(&cat, "feed_forward_length").unwrap_or(2048) as usize;
        let n_heads = md(&cat, "attention.head_count").unwrap_or(8) as usize;
        let head_dim = md(&cat, "attention.key_length").unwrap_or(64) as usize;
        let eps = cat
            .meta_f32(&format!("{arch}.attention.layer_norm_epsilon"))
            .or_else(|| cat.meta_f32("t5.attention.layer_norm_epsilon"))
            .unwrap_or(1e-6);
        let n_buckets = md(&cat, "attention.relative_buckets_count").unwrap_or(32) as usize;
        // Not recorded in the GGUF; HF's default.
        let max_distance = 128usize;
        let decoder_start = md(&cat, "decoder_start_token_id").unwrap_or(0);
        let eos_id = cat.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(1);
        let pad_id = cat.meta_u32("tokenizer.ggml.padding_token_id").unwrap_or(0);

        // Distinct encoder / decoder block ids straight from the tensor names.
        let mk = |prefixes: &[&str]| -> Vec<usize> {
            let mut ids: Vec<usize> = Vec::new();
            for t in &cat.tensors {
                for p in prefixes {
                    if let Some(rest) = t.name.strip_prefix(p) {
                        let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                        if let Ok(v) = num.parse::<usize>() {
                            if !ids.contains(&v) {
                                ids.push(v);
                            }
                        }
                        break;
                    }
                }
            }
            ids.sort_unstable();
            ids
        };
        let enc_ids = mk(&["enc.blk.", "encoder.blk."]);
        let dec_ids = mk(&["dec.blk.", "decoder.blk."]);
        let vocab = cat.tensor("token_embd.weight").map(|t| t.nrows()).unwrap_or(32128);
        let cfg = T5Config {
            d_model,
            d_ff,
            n_heads,
            head_dim,
            enc_layers: enc_ids.len(),
            dec_layers: dec_ids.len(),
            vocab,
            eps,
            n_buckets,
            max_distance,
            decoder_start,
            eos_id,
            pad_id,
        };

        let plan = build_exec_plan(&cat, 0, false)
            .map_err(|u| EdeError::Msg(format!("T5 exec plan failed on {}: {}", u.tensor_name, u.hint)))?;

        // Encoder / decoder streaming units, indexed by original block id.
        let enc_units: Vec<Unit> = (0..cfg.enc_layers)
            .map(|want| {
                plan.units
                    .iter()
                    .find(|u| u.block_id == Some(want))
                    .map(Unit::from_streaming)
                    .ok_or_else(|| EdeError::Msg(format!("T5 missing encoder block {want}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let dec_units: Vec<Unit> = (0..cfg.dec_layers)
            .map(|want| {
                plan.units
                    .iter()
                    .find(|u| u.block_id == Some(DEC_BLOCK_OFFSET + want))
                    .map(Unit::from_streaming)
                    .ok_or_else(|| EdeError::Msg(format!("T5 missing decoder block {want}")))
            })
            .collect::<Result<Vec<_>>>()?;

        let enc_norms = (0..cfg.enc_layers)
            .map(|l| {
                Ok((
                    cat.dequant_f32(&format!("enc.blk.{l}.attn_norm.weight"))?,
                    cat.dequant_f32(&format!("enc.blk.{l}.ffn_norm.weight"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let dec_norms = (0..cfg.dec_layers)
            .map(|l| {
                Ok((
                    cat.dequant_f32(&format!("dec.blk.{l}.attn_norm.weight"))?,
                    cat.dequant_f32(&format!("dec.blk.{l}.cross_attn_norm.weight"))?,
                    cat.dequant_f32(&format!("dec.blk.{l}.ffn_norm.weight"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let enc_out_norm = cat.dequant_f32("enc.output_norm.weight")?;
        let dec_out_norm = cat.dequant_f32("dec.output_norm.weight")?;
        let enc_rel_bias = cat.dequant_f32("enc.blk.0.attn_rel_b.weight")?;
        let dec_rel_bias = cat.dequant_f32("dec.blk.0.attn_rel_b.weight")?;
        let output = cat.load_quant_matrix("output.weight")?;

        let unit_buf = vec![0u8; plan.max_unit_bytes.max(1)];

        Ok(Self {
            cat,
            cfg,
            enc: enc_units,
            dec: dec_units,
            unit_buf,
            enc_norms,
            dec_norms,
            enc_out_norm,
            dec_out_norm,
            enc_rel_bias,
            dec_rel_bias,
            output,
        })
    }

    fn stream_enc(&mut self, idx: usize) -> Result<()> {
        let total = self.enc[idx].total_bytes;
        let specs = self.enc[idx].specs();
        self.cat.load_tensors_into(&specs, &mut self.unit_buf[..total])?;
        Ok(())
    }

    fn stream_dec(&mut self, idx: usize) -> Result<()> {
        let total = self.dec[idx].total_bytes;
        let specs = self.dec[idx].specs();
        self.cat.load_tensors_into(&specs, &mut self.unit_buf[..total])?;
        Ok(())
    }

    fn embed_row(&mut self, token: u32, dst: &mut [f32]) -> Result<()> {
        let d = self.cfg.d_model;
        self.cat.read_embed_row("token_embd.weight", token, d, dst)?;
        Ok(())
    }

    /// Encode a token sequence with the bidirectional encoder stack. Returns the
    /// final-layer, final-norm hidden states `[seq * d_model]`.
    pub fn encode(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let (seq, d, nh, hd) = (ids.len(), cfg.d_model, cfg.n_heads, cfg.head_dim);
        let inner = nh * hd;
        let mut h = vec![0.0f32; seq * d];
        for (t, &id) in ids.iter().enumerate() {
            self.embed_row(id, &mut h[t * d..(t + 1) * d])?;
        }

        let bias = self.enc_rel_bias.clone();
        for l in 0..cfg.enc_layers {
            let (an, fnorm) = self.enc_norms[l].clone();
            self.stream_enc(l)?;
            let u = &self.enc[l];
            let buf: &[u8] = &self.unit_buf[..u.total_bytes];
            let v = EncViews {
                q: u.view(buf, LayerOpKind::AttnQ)?,
                k: u.view(buf, LayerOpKind::AttnK)?,
                v: u.view(buf, LayerOpKind::AttnV)?,
                o: u.view(buf, LayerOpKind::AttnO)?,
                gate: u.view(buf, LayerOpKind::FfnGate)?,
                up: u.view(buf, LayerOpKind::FfnUp)?,
                down: u.view(buf, LayerOpKind::FfnDown)?,
            };

            let mut q = vec![0.0f32; seq * inner];
            let mut k = vec![0.0f32; seq * inner];
            let mut vv = vec![0.0f32; seq * inner];
            let mut xn = vec![0.0f32; d];
            for t in 0..seq {
                xn.copy_from_slice(&h[t * d..(t + 1) * d]);
                rms_norm(&mut xn, &an, cfg.eps);
                v.q.gemv(&xn, &mut q[t * inner..(t + 1) * inner])?;
                v.k.gemv(&xn, &mut k[t * inner..(t + 1) * inner])?;
                v.v.gemv(&xn, &mut vv[t * inner..(t + 1) * inner])?;
            }
            let mut attn = vec![0.0f32; seq * inner];
            let mut scores = vec![0.0f32; seq];
            for t in 0..seq {
                for hdi in 0..nh {
                    for s in 0..seq {
                        let qt = &q[t * inner + hdi * hd..t * inner + (hdi + 1) * hd];
                        let ks = &k[s * inner + hdi * hd..s * inner + (hdi + 1) * hd];
                        let dot: f32 = qt.iter().zip(ks).map(|(a, b)| a * b).sum();
                        let b = rel_bucket(s as i64 - t as i64, true, cfg.n_buckets, cfg.max_distance);
                        scores[s] = dot + bias[b * nh + hdi];
                    }
                    softmax(&mut scores);
                    let acc = &mut attn[t * inner + hdi * hd..t * inner + (hdi + 1) * hd];
                    for s in 0..seq {
                        let vs = &vv[s * inner + hdi * hd..s * inner + (hdi + 1) * hd];
                        let w = scores[s];
                        for i in 0..hd {
                            acc[i] += w * vs[i];
                        }
                    }
                }
            }
            let mut proj = vec![0.0f32; d];
            for t in 0..seq {
                v.o.gemv(&attn[t * inner..(t + 1) * inner], &mut proj)?;
                for i in 0..d {
                    h[t * d + i] += proj[i];
                }
            }
            let mut gate = vec![0.0f32; cfg.d_ff];
            let mut up = vec![0.0f32; cfg.d_ff];
            let mut down = vec![0.0f32; d];
            for t in 0..seq {
                xn.copy_from_slice(&h[t * d..(t + 1) * d]);
                rms_norm(&mut xn, &fnorm, cfg.eps);
                v.gate.gemv(&xn, &mut gate)?;
                v.up.gemv(&xn, &mut up)?;
                for i in 0..cfg.d_ff {
                    gate[i] = gelu_new(gate[i]) * up[i];
                }
                v.down.gemv(&gate, &mut down)?;
                for i in 0..d {
                    h[t * d + i] += down[i];
                }
            }
        }
        for t in 0..seq {
            rms_norm(&mut h[t * d..(t + 1) * d], &self.enc_out_norm, cfg.eps);
        }
        Ok(h)
    }

    /// Precompute cross-attention K/V for every decoder layer from the encoder
    /// output. Returns `(ck, cv)` per layer, each `[enc_len * n_heads * head_dim]`.
    fn prepare_cross(
        &mut self,
        enc: &[f32],
        enc_len: usize,
    ) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        let cfg = self.cfg.clone();
        let (d, nh, hd) = (cfg.d_model, cfg.n_heads, cfg.head_dim);
        let inner = nh * hd;
        let mut out = Vec::with_capacity(cfg.dec_layers);
        for l in 0..cfg.dec_layers {
            self.stream_dec(l)?;
            let u = &self.dec[l];
            let buf: &[u8] = &self.unit_buf[..u.total_bytes];
            let wck = u.view(buf, LayerOpKind::CrossAttnK)?;
            let wcv = u.view(buf, LayerOpKind::CrossAttnV)?;
            let mut ck = vec![0.0f32; enc_len * inner];
            let mut cv = vec![0.0f32; enc_len * inner];
            for s in 0..enc_len {
                wck.gemv(&enc[s * d..(s + 1) * d], &mut ck[s * inner..(s + 1) * inner])?;
                wcv.gemv(&enc[s * d..(s + 1) * d], &mut cv[s * inner..(s + 1) * inner])?;
            }
            out.push((ck, cv));
        }
        Ok(out)
    }

    /// One decoder step: `token` at position `pos`, with the self-attention cache
    /// `self_kv` (per layer `(k,v)` grown in place) and precomputed `cross`. Returns
    /// the logits over the vocabulary.
    fn decode_step(
        &mut self,
        token: u32,
        pos: usize,
        cross: &[(Vec<f32>, Vec<f32>)],
        self_kv: &mut [(Vec<f32>, Vec<f32>)],
        enc_len: usize,
    ) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let (d, nh, hd) = (cfg.d_model, cfg.n_heads, cfg.head_dim);
        let inner = nh * hd;
        let mut x = vec![0.0f32; d];
        self.embed_row(token, &mut x)?;

        for l in 0..cfg.dec_layers {
            let (an, cn, fnorm) = self.dec_norms[l].clone();
            self.stream_dec(l)?;
            let u = &self.dec[l];
            let buf: &[u8] = &self.unit_buf[..u.total_bytes];
            let v = DecViews {
                q: u.view(buf, LayerOpKind::AttnQ)?,
                k: u.view(buf, LayerOpKind::AttnK)?,
                v: u.view(buf, LayerOpKind::AttnV)?,
                o: u.view(buf, LayerOpKind::AttnO)?,
                cq: u.view(buf, LayerOpKind::CrossAttnQ)?,
                co: u.view(buf, LayerOpKind::CrossAttnO)?,
                gate: u.view(buf, LayerOpKind::FfnGate)?,
                up: u.view(buf, LayerOpKind::FfnUp)?,
                down: u.view(buf, LayerOpKind::FfnDown)?,
            };

            let mut xn = x.clone();
            rms_norm(&mut xn, &an, cfg.eps);
            let mut q = vec![0.0f32; inner];
            let mut k = vec![0.0f32; inner];
            let mut vv = vec![0.0f32; inner];
            v.q.gemv(&xn, &mut q)?;
            v.k.gemv(&xn, &mut k)?;
            v.v.gemv(&xn, &mut vv)?;
            self_kv[l].0.extend_from_slice(&k);
            self_kv[l].1.extend_from_slice(&vv);
            let n_keys = pos + 1;
            let mut attn = vec![0.0f32; inner];
            let mut scores = vec![0.0f32; n_keys];
            let (kk, vcache) = &self_kv[l];
            for hdi in 0..nh {
                for s in 0..n_keys {
                    let qt = &q[hdi * hd..(hdi + 1) * hd];
                    let ks = &kk[s * inner + hdi * hd..s * inner + (hdi + 1) * hd];
                    let dot: f32 = qt.iter().zip(ks).map(|(a, b)| a * b).sum();
                    let b = rel_bucket(
                        pos as i64 - s as i64,
                        false,
                        cfg.n_buckets,
                        cfg.max_distance,
                    );
                    scores[s] = dot + self.dec_rel_bias[b * nh + hdi];
                }
                softmax(&mut scores);
                let acc = &mut attn[hdi * hd..(hdi + 1) * hd];
                for s in 0..n_keys {
                    let vs = &vcache[s * inner + hdi * hd..s * inner + (hdi + 1) * hd];
                    let w = scores[s];
                    for i in 0..hd {
                        acc[i] += w * vs[i];
                    }
                }
            }
            let mut proj = vec![0.0f32; d];
            v.o.gemv(&attn, &mut proj)?;
            for i in 0..d {
                x[i] += proj[i];
            }

            // Cross-attention over the encoder output (no position bias).
            let mut xn = x.clone();
            rms_norm(&mut xn, &cn, cfg.eps);
            let mut cq = vec![0.0f32; inner];
            v.cq.gemv(&xn, &mut cq)?;
            let (ck, cv) = &cross[l];
            let mut cattn = vec![0.0f32; inner];
            let mut cscores = vec![0.0f32; enc_len];
            for hdi in 0..nh {
                for s in 0..enc_len {
                    let qt = &cq[hdi * hd..(hdi + 1) * hd];
                    let ks = &ck[s * inner + hdi * hd..s * inner + (hdi + 1) * hd];
                    cscores[s] = qt.iter().zip(ks).map(|(a, b)| a * b).sum();
                }
                softmax(&mut cscores);
                let acc = &mut cattn[hdi * hd..(hdi + 1) * hd];
                for s in 0..enc_len {
                    let vs = &cv[s * inner + hdi * hd..s * inner + (hdi + 1) * hd];
                    let w = cscores[s];
                    for i in 0..hd {
                        acc[i] += w * vs[i];
                    }
                }
            }
            v.co.gemv(&cattn, &mut proj)?;
            for i in 0..d {
                x[i] += proj[i];
            }

            // Gated gelu_new FFN.
            let mut xn = x.clone();
            rms_norm(&mut xn, &fnorm, cfg.eps);
            let mut gate = vec![0.0f32; cfg.d_ff];
            let mut up = vec![0.0f32; cfg.d_ff];
            let mut down = vec![0.0f32; d];
            v.gate.gemv(&xn, &mut gate)?;
            v.up.gemv(&xn, &mut up)?;
            for i in 0..cfg.d_ff {
                gate[i] = gelu_new(gate[i]) * up[i];
            }
            v.down.gemv(&gate, &mut down)?;
            for i in 0..d {
                x[i] += down[i];
            }
        }
        rms_norm(&mut x, &self.dec_out_norm, cfg.eps);
        let mut logits = vec![0.0f32; cfg.vocab];
        self.output.gemv(&x, &mut logits)?;
        Ok(logits)
    }

    /// Logits at the first decoder position for a fixed encoder sequence. Used by
    /// validation tests against HuggingFace.
    pub fn first_logits(&mut self, enc_ids: &[u32], dec_start: u32) -> Result<Vec<f32>> {
        let enc = self.encode(enc_ids)?;
        let enc_len = enc_ids.len();
        let cross = self.prepare_cross(&enc, enc_len)?;
        let mut self_kv: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); self.cfg.dec_layers];
        self.decode_step(dec_start, 0, &cross, &mut self_kv, enc_len)
    }

    /// Greedy generation from tokenized input `ids` (encoder input). Returns the
    /// generated ids (excluding the decoder start token, including the stop token).
    pub fn generate_ids(&mut self, enc_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let enc = self.encode(enc_ids)?;
        let enc_len = enc_ids.len();
        let cross = self.prepare_cross(&enc, enc_len)?;
        let mut self_kv: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); self.cfg.dec_layers];
        let mut token = self.cfg.decoder_start;
        let mut out = Vec::new();
        for pos in 0..max_new_tokens {
            let logits = self.decode_step(token, pos, &cross, &mut self_kv, enc_len)?;
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            token = best as u32;
            out.push(token);
            if token == self.cfg.eos_id {
                break;
            }
        }
        Ok(out)
    }
}

/// Greedy T5 text-to-text generation.
pub fn generate(
    model: &mut T5Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<Vec<u32>> {
    // T5's SentencePiece applies `add_dummy_prefix`; emulate it with a leading space
    // and append EOS (the GGUF requests `add_eos_token`). No BOS for T5.
    let text = format!(" {prompt}");
    let mut ids = tokenizer.encode(&text, false);
    ids.push(model.cfg.eos_id);
    model.generate_ids(&ids, max_new_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Validation vs HuggingFace `google/flan-t5-small` (F32):
    /// prompt `translate English to German: The house is wonderful.`,
    /// encoder ids `[13959,1566,12,2968,10,37,629,19,1627,5,1]`, decoder start 0,
    /// first-step logits top-5 `[644,316,37,660,1122]`.
    #[test]
    fn t5_first_logits_match_hf() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("models/flan-t5-small.F16.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let mut m = T5Model::open(&path).unwrap();
        let enc_ids: [u32; 11] = [13959, 1566, 12, 2968, 10, 37, 629, 19, 1627, 5, 1];
        let logits = m.first_logits(&enc_ids, 0).unwrap();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let top5: Vec<usize> = order[..5].to_vec();
        eprintln!(
            "T5 top5 = {:?} logits = {:?}",
            top5,
            order[..5].iter().map(|&i| logits[i]).collect::<Vec<_>>()
        );
        assert_eq!(top5, vec![644, 316, 37, 660, 1122]);
    }

    /// End-to-end greedy: tokenizer + encoder + decoder. HF reference:
    /// `[644, 4598, 229, 9685, 5, 1]` (`Das Haus ist schön.`).
    #[test]
    fn t5_greedy_matches_hf() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("models/flan-t5-small.F16.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let mut m = T5Model::open(&path).unwrap();
        let cat = GgufCatalog::open(&path).unwrap();
        let tok = hayai_model::Tokenizer::from_catalog(&cat).unwrap();
        let ids = generate(
            &mut m,
            &tok,
            "translate English to German: The house is wonderful.",
            8,
        )
        .unwrap();
        eprintln!("T5 greedy ids = {ids:?}");
        assert_eq!(ids, vec![644, 4598, 229, 9685, 5, 1]);
    }
}
