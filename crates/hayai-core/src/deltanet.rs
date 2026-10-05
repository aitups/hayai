//! CPU decode-step for Qwen3.5 gated DeltaNet / SSM layers (GGUF `ssm_*` + `attn_qkv`).
//!
//! Matches HuggingFace `torch_recurrent_gated_delta_rule` + llama.cpp `qwen35`:
//! - state per V-head: `[head_k, head_v]` (square when head_k == head_v, as in llama GDN)
//! - query scaled by `1/sqrt(head_k)` before readout
//! - decay = `exp(ssm_a * softplus(alpha + dt_bias))` where `ssm_a` is already `-exp(A_log)`
//! - causal depthwise conv: `weight[0]` multiplies the oldest sample (PyTorch Conv1d / HF)
//! - K↔V head map: `kh = vh % n_k` (llama `ggml_repeat` tile), not HF `repeat_interleave`

use hayai_model::{GgufCatalog, GgufError, QuantMatrix};

#[derive(Clone)]
pub struct DeltaNetLayerWeights {
    pub qkv: QuantMatrix,
    pub gate: QuantMatrix,
    pub conv1d: Vec<f32>,
    pub conv_k: usize,
    pub conv_dim: usize,
    pub a: Vec<f32>,
    pub alpha: QuantMatrix,
    pub beta: QuantMatrix,
    pub dt_bias: Vec<f32>,
    pub norm: Vec<f32>,
    pub out: QuantMatrix,
    pub n_v_heads: usize,
    pub head_v: usize,
    pub n_k_heads: usize,
    pub head_k: usize,
    pub eps: f32,
}

/// Per-layer recurrent state for decode.
#[derive(Clone)]
pub struct DeltaNetState {
    /// Last `(conv_k - 1) * conv_dim` mixed channels, oldest→newest.
    pub conv: Vec<f32>,
    /// `n_v_heads * head_k * head_v` — row-major `[k, v]` per V-head.
    pub ssm: Vec<f32>,
}

impl DeltaNetState {
    pub fn new(conv_k: usize, conv_dim: usize, n_v_heads: usize, head_k: usize, head_v: usize) -> Self {
        Self {
            conv: vec![0.0; (conv_k.saturating_sub(1)) * conv_dim],
            ssm: vec![0.0; n_v_heads * head_k * head_v],
        }
    }
}

impl DeltaNetLayerWeights {
    /// Load one DeltaNet block. Dimensions come from **this layer's tensors**;
    /// arch `ssm.*` meta is only a consistency check when present (no 4B fallbacks).
    pub fn load(cat: &mut GgufCatalog, layer: usize) -> Result<Self, GgufError> {
        let arch = cat
            .meta_str("general.architecture")
            .unwrap_or("qwen35")
            .to_string();
        let eps = cat
            .meta_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);

        let qkv = cat.load_quant_matrix(&format!("blk.{layer}.attn_qkv.weight"))?;
        let gate = cat.load_quant_matrix(&format!("blk.{layer}.attn_gate.weight"))?;
        let out = cat.load_quant_matrix(&format!("blk.{layer}.ssm_out.weight"))?;
        let alpha = cat.load_quant_matrix(&format!("blk.{layer}.ssm_alpha.weight"))?;
        let beta = cat.load_quant_matrix(&format!("blk.{layer}.ssm_beta.weight"))?;
        let a = cat.dequant_f32(&format!("blk.{layer}.ssm_a"))?;
        let norm = cat.dequant_f32(&format!("blk.{layer}.ssm_norm.weight"))?;
        let dt_bias = cat
            .dequant_f32(&format!("blk.{layer}.ssm_dt.bias"))
            .or_else(|_| cat.dequant_f32(&format!("blk.{layer}.ssm_dt")))
            .unwrap_or_else(|_| vec![0.0; a.len()]);
        let conv_info = cat.tensor(&format!("blk.{layer}.ssm_conv1d.weight"))?.clone();
        let conv1d = cat.dequant_f32(&format!("blk.{layer}.ssm_conv1d.weight"))?;
        // GGUF layout: [ne0=kernel, ne1=channels] → ncols=kernel, nrows=channels
        let conv_k = conv_info.ncols().max(1);
        let conv_dim = conv_info.nrows().max(1);
        if conv1d.len() < conv_k * conv_dim {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: ssm_conv1d size {} < {conv_k}*{conv_dim}",
                conv1d.len()
            )));
        }

        // Tensor-first: n_v from ssm_a / alpha.nrows / beta.nrows
        let n_v_heads = a.len().max(1);
        if alpha.nrows != n_v_heads || beta.nrows != n_v_heads {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: ssm_a len={} vs alpha.nrows={} beta.nrows={}",
                a.len(),
                alpha.nrows,
                beta.nrows
            )));
        }
        if dt_bias.len() != n_v_heads {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: ssm_dt len={} != n_v_heads={n_v_heads}",
                dt_bias.len()
            )));
        }

        let head_v = if !norm.is_empty() && gate.nrows % norm.len() == 0 {
            norm.len()
        } else if gate.nrows % n_v_heads == 0 {
            gate.nrows / n_v_heads
        } else {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: cannot derive head_v from gate.nrows={} n_v={n_v_heads} norm={}",
                gate.nrows,
                norm.len()
            )));
        };
        if gate.nrows != n_v_heads * head_v {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: gate.nrows={} != n_v*{head_v}",
                gate.nrows
            )));
        }
        let value_dim = n_v_heads * head_v;

        // qkv = [key | key | value]; remaining after value_dim is 2*key_dim
        if qkv.nrows <= value_dim || (qkv.nrows - value_dim) % 2 != 0 {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: attn_qkv.nrows={} incompatible with value_dim={value_dim}",
                qkv.nrows
            )));
        }
        let key_dim = (qkv.nrows - value_dim) / 2;
        if conv_dim != key_dim * 2 + value_dim && conv_dim != qkv.nrows {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: conv_dim={conv_dim} != 2*{key_dim}+{value_dim} (qkv={})",
                qkv.nrows
            )));
        }

        // Prefer meta state_size / group_count when they divide key_dim cleanly.
        let head_k_meta = cat
            .meta_u32(&format!("{arch}.ssm.state_size"))
            .map(|v| v as usize)
            .filter(|&s| s > 0 && key_dim % s == 0);
        let n_k_meta = cat
            .meta_u32(&format!("{arch}.ssm.group_count"))
            .map(|v| v as usize)
            .filter(|&g| g > 0 && key_dim % g == 0);

        let (n_k_heads, head_k) = if let (Some(hk), Some(nk)) = (head_k_meta, n_k_meta) {
            if nk * hk != key_dim {
                return Err(GgufError::Msg(format!(
                    "blk.{layer}: meta n_k*head_k={nk}*{hk} != key_dim={key_dim}"
                )));
            }
            (nk, hk)
        } else if let Some(hk) = head_k_meta {
            (key_dim / hk, hk)
        } else if let Some(nk) = n_k_meta {
            (nk, key_dim / nk)
        } else if head_v > 0 && key_dim % head_v == 0 {
            // llama GDN asserts head_k == head_v; common when meta missing.
            (key_dim / head_v, head_v)
        } else {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: cannot split key_dim={key_dim} (no ssm.state_size/group_count)"
            )));
        };

        if n_v_heads % n_k_heads != 0 {
            return Err(GgufError::Msg(format!(
                "blk.{layer}: n_v={n_v_heads} not divisible by n_k={n_k_heads}"
            )));
        }

        // Optional meta cross-check (warn via error only on hard conflict with tensors).
        if let Some(inner) = cat.meta_u32(&format!("{arch}.ssm.inner_size")) {
            let inner = inner as usize;
            if inner != value_dim && inner != key_dim * 2 + value_dim {
                // inner_size in Qwen GGUF is value_dim (= head_v * n_v); tolerate key-inclusive aliases.
                if inner != n_v_heads * head_v {
                    return Err(GgufError::Msg(format!(
                        "blk.{layer}: ssm.inner_size={inner} != value_dim={value_dim}"
                    )));
                }
            }
        }

        Ok(Self {
            qkv,
            gate,
            conv1d,
            conv_k,
            conv_dim,
            a,
            alpha,
            beta,
            dt_bias,
            norm,
            out,
            n_v_heads,
            head_v,
            n_k_heads,
            head_k,
            eps,
        })
    }

    /// Decode step: `y += DeltaNet(x)` (caller applies residual).
    ///
    /// `gpu` (opt-in, `HAYAI_DN_GPU=1`) runs the recurrent state update on a device via
    /// the `hayai_deltanet_step` kernel; `None` keeps the host implementation. The
    /// device state is the source of truth on the GPU path (the host `ssm` is untouched).
    pub fn decode_step(
        &self,
        x: &[f32],
        state: &mut DeltaNetState,
        y: &mut [f32],
        gpu: Option<(&hayai_opencl::OpenClEngine, *const hayai_opencl::DeviceDeltanetState)>,
    ) -> Result<(), GgufError> {
        let n_embd = x.len();
        let mix_dim = self.qkv.nrows;
        let mut mixed = vec![0.0f32; mix_dim];
        self.qkv.gemv(x, &mut mixed)?;

        let k = self.conv_k;
        let cd = self.conv_dim.min(mix_dim);
        let hist = k.saturating_sub(1);
        let mut conv_out = vec![0.0f32; cd];
        // PyTorch/HF/ggml causal Conv1d: weight[0]·oldest … weight[k-1]·newest.
        // HAYAI_CONV_FLIP=1 reverses taps (diagnose GGUF tap order).
        let flip = std::env::var("HAYAI_CONV_FLIP").ok().as_deref() == Some("1");
        for c in 0..cd {
            let row = c * k;
            let mut acc = 0.0f32;
            for t in 0..hist {
                let st = &state.conv[t * cd..t * cd + cd];
                let w_idx = if flip { hist - t } else { t };
                acc += self.conv1d[row + w_idx] * st[c];
            }
            let w_new = if flip { 0 } else { hist };
            acc += self.conv1d[row + w_new] * mixed[c];
            conv_out[c] = silu(acc);
        }
        if hist > 0 {
            if hist > 1 {
                state.conv.copy_within(cd..hist * cd, 0);
            }
            state.conv[(hist - 1) * cd..hist * cd].copy_from_slice(&mixed[..cd]);
        }

        let key_dim = self.n_k_heads * self.head_k;
        let value_dim = self.n_v_heads * self.head_v;
        let mut q = conv_out[..key_dim].to_vec();
        let mut kk = conv_out[key_dim..key_dim * 2].to_vec();
        let v = conv_out[key_dim * 2..key_dim * 2 + value_dim].to_vec();

        l2_norm_heads(&mut q, self.n_k_heads, self.head_k, self.eps);
        l2_norm_heads(&mut kk, self.n_k_heads, self.head_k, self.eps);

        let q_scale = 1.0 / (self.head_k as f32).sqrt();
        for e in q.iter_mut() {
            *e *= q_scale;
        }

        let mut alpha_h = vec![0.0f32; self.n_v_heads];
        let mut beta_h = vec![0.0f32; self.n_v_heads];
        self.alpha.gemv(x, &mut alpha_h)?;
        self.beta.gemv(x, &mut beta_h)?;

        // Per-head decay / beta (host, tiny) — shared by both paths.
        let mut decay = vec![0.0f32; self.n_v_heads];
        let mut beta = vec![0.0f32; self.n_v_heads];
        for vh in 0..self.n_v_heads {
            let a = self.a[vh.min(self.a.len() - 1)];
            let soft = softplus(self.dt_bias.get(vh).copied().unwrap_or(0.0) + alpha_h[vh]);
            decay[vh] = (a * soft).exp();
            beta[vh] = sigmoid(beta_h[vh]);
        }

        let hv = self.head_v;
        let hk = self.head_k;
        let mut o = vec![0.0f32; value_dim];
        if let Some((eng, dev)) = gpu {
            // Recurrent state decay/read/delta-write/output on the device. `dev` is a
            // raw pointer to the generator's per-layer state (the caller holds it out of
            // the aliased `state` borrow, same trick as the weight/state pointers).
            let dev = unsafe { &*dev };
            eng.deltanet_step(&q, &kk, &v, &decay, &beta, dev, &mut o)
                .map_err(|e| GgufError::Msg(format!("deltanet_step: {e}")))?;
        } else {
            // GQA over V-heads: llama.cpp expands K-heads with `ggml_repeat` (tile),
            // so V-head `vh` maps to K-head `vh % n_k` — NOT HF `repeat_interleave`
            // (`vh / ratio`). Wrong mapping yields coherent-looking garbage until fixed.
            for vh in 0..self.n_v_heads {
                let kh = vh % self.n_k_heads;
                let decay = decay[vh];
                let b = beta[vh];

                let s = &mut state.ssm[vh * hk * hv..(vh + 1) * hk * hv];
                let qh = &q[kh * hk..kh * hk + hk];
                let khv = &kk[kh * hk..kh * hk + hk];
                let vv = &v[vh * hv..(vh + 1) * hv];

                for e in s.iter_mut() {
                    *e *= decay;
                }
                let mut kv_mem = vec![0.0f32; hv];
                for ki in 0..hk {
                    let ks = khv[ki];
                    let row = ki * hv;
                    for vi in 0..hv {
                        kv_mem[vi] += s[row + vi] * ks;
                    }
                }
                for ki in 0..hk {
                    let ks = khv[ki];
                    let row = ki * hv;
                    for vi in 0..hv {
                        let delta = (vv[vi] - kv_mem[vi]) * b;
                        s[row + vi] += ks * delta;
                    }
                }
                let oh = &mut o[vh * hv..(vh + 1) * hv];
                for vi in 0..hv {
                    let mut acc = 0.0;
                    for ki in 0..hk {
                        acc += s[ki * hv + vi] * qh[ki];
                    }
                    oh[vi] = acc;
                }
            }
        }

        let mut z = vec![0.0f32; value_dim];
        self.gate.gemv(x, &mut z)?;
        for vh in 0..self.n_v_heads {
            let base = vh * hv;
            let oh = &mut o[base..base + hv];
            let mut ms = 0.0f32;
            for &v in oh.iter() {
                ms += v * v;
            }
            let inv = 1.0 / (ms / hv as f32 + self.eps).sqrt();
            for i in 0..hv {
                let n = self.norm.get(i).copied().unwrap_or(1.0);
                oh[i] = oh[i] * inv * n * silu(z[base + i]);
            }
        }

        let mut proj = vec![0.0f32; n_embd];
        self.out.gemv(&o, &mut proj)?;
        for i in 0..n_embd {
            y[i] += proj[i];
        }
        Ok(())
    }
}

fn l2_norm_heads(x: &mut [f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let base = h * head_dim;
        let mut ms = 0.0f32;
        for i in 0..head_dim {
            ms += x[base + i] * x[base + i];
        }
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..head_dim {
            x[base + i] *= inv;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// True if this block is a DeltaNet/SSM hybrid layer (not full MHA).
pub fn is_deltanet_layer(cat: &GgufCatalog, layer: usize) -> bool {
    cat.tensor(&format!("blk.{layer}.ssm_a")).is_ok()
        && cat.tensor(&format!("blk.{layer}.attn_qkv.weight")).is_ok()
}

/// True if this block is a NextN/MTP draft head (skip in main decode).
pub fn is_nextn_layer(cat: &GgufCatalog, layer: usize) -> bool {
    cat.tensor(&format!("blk.{layer}.nextn.eh_proj.weight")).is_ok()
}
