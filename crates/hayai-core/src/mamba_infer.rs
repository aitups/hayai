//! Mamba-1 selective-scan forward (op-driven SSM, no attention / no KV cache).
//!
//! Tensor roles (driven by presence, never the family name): `ssm_in` (x∥z),
//! `ssm_conv1d` (+bias), `ssm_x` (dt∥B∥C), `ssm_dt` (+bias), `ssm_a`, `ssm_d`,
//! `ssm_out`, `attn_norm`. Matches llama.cpp `build_mamba_layer` (`ggml_ssm_scan`):
//! `dt = softplus(dt)`, `h = exp(dt·A)·h + dt·B·x`, `y = C·h + D·x`, then `y *= silu(z)`.

use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{StreamInferError, StreamingGenerator};
use hayai_cpu::rms_norm;
use hayai_model::{GgufCatalog, QuantMatrix};

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `out[o] = sum_c w[o*ncols + c] * x[c]` for an F32 matrix stored row-major.
fn gemv_f32(w: &[f32], ncols: usize, nrows: usize, x: &[f32], out: &mut [f32]) {
    for o in 0..nrows {
        let row = &w[o * ncols..(o + 1) * ncols];
        let mut s = 0.0f32;
        for c in 0..ncols {
            s += row[c] * x[c];
        }
        out[o] = s;
    }
}

pub(crate) struct MambaLayerWeights {
    pub in_proj: QuantMatrix,
    pub out_proj: QuantMatrix,
    pub conv_w: Vec<f32>,
    pub conv_b: Vec<f32>,
    pub x_proj: Vec<f32>,
    pub dt_w: Vec<f32>,
    pub dt_b: Vec<f32>,
    pub a: Vec<f32>,
    pub d: Vec<f32>,
    pub norm: Vec<f32>,
    pub d_inner: usize,
    pub d_state: usize,
    pub dt_rank: usize,
    pub d_conv: usize,
}

struct MambaState {
    conv: Vec<f32>,
    ssm: Vec<f32>,
}

pub(crate) struct MambaCache {
    layers: Vec<MambaLayerWeights>,
    states: Vec<MambaState>,
    /// Tied LM head (`token_embd`) when the model has no `output.weight`.
    lm_head: Option<QuantMatrix>,
}

impl MambaLayerWeights {
    fn load(cat: &mut GgufCatalog, layer: usize) -> Result<Self, StreamInferError> {
        let t = |s: &str| format!("blk.{layer}.{s}");
        Ok(Self {
            in_proj: cat.load_quant_matrix(&t("ssm_in.weight"))?,
            out_proj: cat.load_quant_matrix(&t("ssm_out.weight"))?,
            conv_w: cat.dequant_f32(&t("ssm_conv1d.weight"))?,
            conv_b: cat.dequant_f32(&t("ssm_conv1d.bias"))?,
            x_proj: cat.dequant_f32(&t("ssm_x.weight"))?,
            dt_w: cat.dequant_f32(&t("ssm_dt.weight"))?,
            dt_b: cat.dequant_f32(&t("ssm_dt.bias"))?,
            a: cat.dequant_f32(&t("ssm_a"))?,
            d: cat.dequant_f32(&t("ssm_d"))?,
            norm: cat
                .dequant_f32(&t("attn_norm.weight"))
                .or_else(|_| cat.dequant_f32(&t("attention_norm.weight")))?,
            d_inner: cat.tensor(&t("ssm_x.weight"))?.ncols(),
            d_state: cat.tensor(&t("ssm_a"))?.ncols(),
            dt_rank: cat.tensor(&t("ssm_dt.weight"))?.ncols(),
            d_conv: cat.tensor(&t("ssm_conv1d.weight"))?.ncols().max(1),
        })
    }
}

impl StreamingGenerator {
    fn ensure_mamba_cache(&mut self) -> Result<(), StreamInferError> {
        if self.mamba_cache.is_some() {
            return Ok(());
        }
        let n = self.config.num_layers;
        let mut layers = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        for l in 0..n {
            let w = MambaLayerWeights::load(&mut self.catalog, l)?;
            states.push(MambaState {
                conv: vec![0.0; (w.d_conv.saturating_sub(1)) * w.d_inner],
                ssm: vec![0.0; w.d_inner * w.d_state],
            });
            layers.push(w);
        }
        let lm_head = if self.has_output_weight {
            None
        } else {
            Some(self.catalog.load_quant_matrix("token_embd.weight")?)
        };
        self.mamba_cache = Some(MambaCache {
            layers,
            states,
            lm_head,
        });
        Ok(())
    }

    /// One Mamba token: mixer recurrence over all layers, then the LM head.
    pub(crate) fn forward_mamba(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
    ) -> Result<Vec<f32>, StreamInferError> {
        self.ensure_mamba_cache()?;
        let eps = self.config.rms_norm_eps;
        let d_model = self.config.hidden_size;
        let mut x = vec![0.0f32; d_model];
        self.embed_row("token_embd.weight", token, d_model, &mut x)?;

        let mut cache = self.mamba_cache.take().expect("mamba cache");
        self.mamba_layers(orch, &mut cache, eps, &mut x)?;

        let mut xn = x;
        rms_norm(&mut xn, &self.output_norm, eps);
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        let head = if let Some(ow) = &self.resident_output {
            Some(ow)
        } else if self.has_output_weight {
            None
        } else {
            cache.lm_head.as_ref()
        };
        match head {
            Some(ow) => orch.execute_quant_gemv(ow, &xn, &mut logits)?,
            None => {
                let ow = self.catalog.load_quant_matrix("output.weight")?;
                orch.execute_quant_gemv(&ow, &xn, &mut logits)?;
            }
        }
        self.mamba_cache = Some(cache);
        self.position += 1;
        Ok(logits)
    }

    fn mamba_layers(
        &mut self,
        orch: &mut EngineOrchestrator,
        cache: &mut MambaCache,
        eps: f32,
        x: &mut [f32],
    ) -> Result<(), StreamInferError> {
        let d_model = self.config.hidden_size;
        for l in 0..cache.layers.len() {
            let lw = &cache.layers[l];
            let (d_inner, d_state, d_conv) = (lw.d_inner, lw.d_state, lw.d_conv);

            let mut xn = x.to_vec();
            rms_norm(&mut xn, &lw.norm, eps);
            let mut proj = vec![0.0f32; 2 * d_inner];
            orch.execute_quant_gemv(&lw.in_proj, &xn, &mut proj)?;
            let (xpart, zpart) = proj.split_at(d_inner);

            let st = &mut cache.states[l];
            // Causal depthwise conv over x + bias, then SiLU.
            let mut xc = vec![0.0f32; d_inner];
            for c in 0..d_inner {
                let row = c * d_conv;
                let mut acc = lw.conv_b[c];
                for tt in 0..d_conv - 1 {
                    acc += lw.conv_w[row + tt] * st.conv[tt * d_inner + c];
                }
                acc += lw.conv_w[row + d_conv - 1] * xpart[c];
                xc[c] = silu(acc);
            }
            if d_conv > 1 {
                let hist = d_conv - 1;
                if hist > 1 {
                    st.conv.copy_within(d_inner..hist * d_inner, 0);
                }
                st.conv[(hist - 1) * d_inner..hist * d_inner].copy_from_slice(xpart);
            }

            // x_proj -> dt(rank) ∥ B(state) ∥ C(state)
            let mut xd = vec![0.0f32; lw.dt_rank + 2 * d_state];
            gemv_f32(&lw.x_proj, d_inner, lw.dt_rank + 2 * d_state, &xc, &mut xd);
            let dt_raw = &xd[..lw.dt_rank];
            let bvec = &xd[lw.dt_rank..lw.dt_rank + d_state];
            let cvec = &xd[lw.dt_rank + d_state..];

            // dt = dt_proj(dt_raw) + dt_bias, then softplus.
            let mut dt = vec![0.0f32; d_inner];
            gemv_f32(&lw.dt_w, lw.dt_rank, d_inner, dt_raw, &mut dt);
            for i in 0..d_inner {
                dt[i] = (1.0 + (dt[i] + lw.dt_b[i]).exp()).ln();
            }

            // Selective scan (y uses the updated state).
            let mut y = vec![0.0f32; d_inner];
            for i in 0..d_inner {
                let dt_i = dt[i];
                let x_i = xc[i];
                let mut acc = lw.d[i] * x_i;
                let base = i * d_state;
                for n in 0..d_state {
                    let h =
                        (dt_i * lw.a[base + n]).exp() * st.ssm[base + n] + dt_i * bvec[n] * x_i;
                    st.ssm[base + n] = h;
                    acc += h * cvec[n];
                }
                y[i] = acc * silu(zpart[i]);
            }

            let mut out = vec![0.0f32; d_model];
            orch.execute_quant_gemv(&lw.out_proj, &y, &mut out)?;
            for i in 0..d_model {
                x[i] += out[i];
            }
        }
        Ok(())
    }
}

/// Prompt prefill = token-by-token (SSM recurrence is inherently sequential).
pub(crate) fn prefill_mamba(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    tokens: &[u32],
    _scratch: &mut hayai_opencl::StreamingScratch,
) -> Result<Vec<f32>, StreamInferError> {
    let mut last = Vec::new();
    for &t in tokens {
        last = gen.forward_mamba(orch, t)?;
    }
    Ok(last)
}
