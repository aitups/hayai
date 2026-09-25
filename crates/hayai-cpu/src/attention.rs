use crate::kv_cache::LayerKvCache;

/// Rotary Position Embedding (RoPE) on a single head vector of `head_dim`.
///
/// When `rope_dim < head_dim` (Qwen3.5: rope_dim=64, head_dim=256), only the
/// first `rope_dim` elements are rotated; the remainder is left unchanged.
pub fn apply_rope(vec: &mut [f32], position: usize, head_dim: usize, base_freq: f32) {
    apply_rope_partial(vec, position, head_dim, head_dim, base_freq);
}

/// Partial RoPE: rotate only the leading `rope_dim` dims of a `head_dim` head.
pub fn apply_rope_partial(
    vec: &mut [f32],
    position: usize,
    head_dim: usize,
    rope_dim: usize,
    base_freq: f32,
) {
    apply_rope_partial_factors(vec, position, head_dim, rope_dim, base_freq, None);
}

/// NeoX-style RoPE with optional `freq_factors` (llama `ggml_rope_ext` / Gemma4
/// `rope_freqs.weight`): angle = `pos * inv_freq / factor[i]`.
pub fn apply_rope_partial_factors(
    vec: &mut [f32],
    position: usize,
    head_dim: usize,
    rope_dim: usize,
    base_freq: f32,
    freq_factors: Option<&[f32]>,
) {
    assert_eq!(vec.len(), head_dim);
    let rd = rope_dim.min(head_dim);
    if rd < 2 {
        return;
    }
    let half_dim = rd / 2;
    for i in 0..half_dim {
        let mut freq = 1.0 / base_freq.powf((2 * i) as f32 / rd as f32);
        if let Some(ff) = freq_factors.and_then(|f| f.get(i)).copied() {
            if ff != 0.0 {
                freq /= ff;
            }
        }
        let val = position as f32 * freq;
        let cos_val = val.cos();
        let sin_val = val.sin();

        let v0 = vec[i];
        let v1 = vec[i + half_dim];

        vec[i] = v0 * cos_val - v1 * sin_val;
        vec[i + half_dim] = v0 * sin_val + v1 * cos_val;
    }
}

/// RoPE with linear / YaRN scaling (llama.cpp `ggml_rope_ext`). Falls back to
/// [`apply_rope_partial_factors`] when `rope.is_none()`.
pub fn apply_rope_partial_factors_scaled(
    vec: &mut [f32],
    position: usize,
    head_dim: usize,
    rope_dim: usize,
    base_freq: f32,
    freq_factors: Option<&[f32]>,
    rope: RopeScaling,
) {
    if rope.is_none() {
        return apply_rope_partial_factors(
            vec,
            position,
            head_dim,
            rope_dim,
            base_freq,
            freq_factors,
        );
    }
    assert_eq!(vec.len(), head_dim);
    let rd = rope_dim.min(head_dim);
    if rd < 2 {
        return;
    }
    let half_dim = rd / 2;
    // ggml `rope_yarn_corr_dims`: corrected dims for the YaRN ramp.
    let n_ctx_orig = rope.n_ctx_orig.max(1) as f32;
    let corr = |beta: f32| -> f32 {
        rd as f32 * (n_ctx_orig / (beta * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * base_freq.ln())
    };
    let (corr_low, corr_high) = if rope.ext_factor != 0.0 {
        (corr(rope.beta_fast).floor(), corr(rope.beta_slow).ceil())
    } else {
        (0.0, 0.0)
    };
    for i in 0..half_dim {
        let mut freq = 1.0 / base_freq.powf((2 * i) as f32 / rd as f32);
        if let Some(ff) = freq_factors.and_then(|f| f.get(i)).copied() {
            if ff != 0.0 {
                freq /= ff;
            }
        }
        let theta_extrap = position as f32 * freq;
        let theta_interp = rope.freq_scale * theta_extrap;
        let mut theta = theta_interp;
        let mut mscale = rope.attn_factor;
        if rope.ext_factor != 0.0 {
            let i0 = (2 * i) as f32;
            let y = (i0 / 2.0 - corr_low) / (corr_high - corr_low).max(0.001);
            let ramp = 1.0 - y.clamp(0.0, 1.0);
            let ramp_mix = ramp * rope.ext_factor;
            theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0 + 0.1 * (1.0 / rope.freq_scale).ln();
        }
        let (sin_val, cos_val) = (theta.sin() * mscale, theta.cos() * mscale);
        let v0 = vec[i];
        let v1 = vec[i + half_dim];
        vec[i] = v0 * cos_val - v1 * sin_val;
        vec[i + half_dim] = v0 * sin_val + v1 * cos_val;
    }
}

/// Numerically stable Softmax in-place.
pub fn softmax(slice: &mut [f32]) {
    if slice.is_empty() {
        return;
    }
    let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in slice.iter_mut() {
        *x = (*x - max_val).exp();
        sum += *x;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in slice.iter_mut() {
            *x *= inv;
        }
    }
}

/// RMSNorm in-place (Llama-style). Uses `std::simd` (PRD §2 / §3.3).
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len());
    let ms = crate::simd_ops::simd_mean_square(x);
    let inv = 1.0 / (ms + eps).sqrt();
    crate::simd_ops::simd_scale_mul(x, weight, inv);
}

/// Dot product (PRD §2 CPU SIMD via `std::simd`).
#[inline]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    crate::simd_ops::simd_dot(a, b)
}

/// RoPE scaling (llama.cpp `ggml_rope_ext`): linear (`freq_scale`) and YaRN
/// (`ext_factor`/`attn_factor`/`beta_*`/`n_ctx_orig`).
#[derive(Debug, Clone, Copy)]
pub struct RopeScaling {
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub n_ctx_orig: usize,
    /// DeepSeek-V2 `rope.scaling.yarn_log_multiplier` (pre-scales `kq_scale`).
    pub yarn_log_mul: f32,
}

impl RopeScaling {
    pub const NONE: Self = Self {
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        n_ctx_orig: 0,
        yarn_log_mul: 0.0,
    };

    pub fn is_none(&self) -> bool {
        self.ext_factor == 0.0
            && (self.freq_scale - 1.0).abs() < 1e-9
            && self.attn_factor == 1.0
    }
}

/// GQA / MHA configuration.
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    /// Elements of each head that receive RoPE (`<= head_dim`). Qwen3.5 uses 64 of 256.
    pub rope_dim: usize,
    /// If `Some`, use this attention scale instead of `1/sqrt(head_dim)`. Gemma4 uses `1.0`.
    pub scale_override: Option<f32>,
    /// ALiBi bias clamp (`{arch}.attention.max_alibi_bias`, MPT): `bias = max(bias, -max)`.
    /// `0.0` disables clamping (BLOOM/Falcon).
    pub alibi_max_bias: f32,
    /// Linear / YaRN RoPE scaling.
    pub rope: RopeScaling,
    /// `false` for models with learned positions (GPT-2) or ALiBi (BLOOM/Falcon/MPT),
    /// which do not rotate Q/K.
    pub use_rope: bool,
}

impl AttentionConfig {
    pub fn from_model(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f32,
    ) -> Self {
        assert_eq!(hidden_size % num_heads, 0);
        assert_eq!(num_heads % num_kv_heads, 0);
        let head_dim = hidden_size / num_heads;
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
            rope_dim: head_dim,
            scale_override: None,
            alibi_max_bias: 0.0,
            rope: RopeScaling::NONE,
            use_rope: true,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn groups_per_kv(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

/// One decode-step of causal GQA attention against an INT8 KV cache.
///
/// `q` / `k` / `v` are already projected: shapes
/// `[num_heads * head_dim]`, `[num_kv_heads * head_dim]`, `[num_kv_heads * head_dim]`.
/// Writes attention output into `out` (`[num_heads * head_dim]`).
pub fn attention_decode_step(
    cfg: &AttentionConfig,
    cache: &mut LayerKvCache,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    position: usize,
    out: &mut [f32],
) {
    attention_decode_step_ex(cfg, cache, q, k, v, position, out, true, None, None);
}

/// Like [`attention_decode_step`] but with ALiBi slopes (one per query head).
pub fn attention_decode_step_alibi(
    cfg: &AttentionConfig,
    cache: &mut LayerKvCache,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    position: usize,
    out: &mut [f32],
    alibi_slopes: &[f32],
) {
    attention_decode_step_ex(
        cfg,
        cache,
        q,
        k,
        v,
        position,
        out,
        true,
        None,
        Some(alibi_slopes),
    );
}

/// Like [`attention_decode_step`], but when `write_kv` is false only Q is RoPE'd and
/// attention reads the existing cache (Gemma4 shared-KV layers).
///
/// `freq_factors`: optional per-pair divisors (length `rope_dim/2`), used by Gemma4
/// global layers via `rope_freqs.weight`.
pub fn attention_decode_step_ex(
    cfg: &AttentionConfig,
    cache: &mut LayerKvCache,
    q: &mut [f32],
    k: &mut [f32],
    v: &[f32],
    position: usize,
    out: &mut [f32],
    write_kv: bool,
    freq_factors: Option<&[f32]>,
    alibi_slopes: Option<&[f32]>,
) {
    assert_eq!(q.len(), cfg.hidden_size());
    assert_eq!(out.len(), cfg.hidden_size());
    assert_eq!(cache.heads.len(), cfg.num_kv_heads);

    // RoPE on all Q heads and KV heads (partial when rope_dim < head_dim).
    // Skipped for learned-position (GPT-2) / ALiBi (BLOOM/Falcon/MPT) models.
    let rd = cfg.rope_dim.max(1).min(cfg.head_dim);
    if cfg.use_rope {
        for h in 0..cfg.num_heads {
            let s = h * cfg.head_dim;
            apply_rope_partial_factors_scaled(
                &mut q[s..s + cfg.head_dim],
                position,
                cfg.head_dim,
                rd,
                cfg.rope_theta,
                freq_factors,
                cfg.rope,
            );
        }
    }
    if write_kv {
        assert_eq!(k.len(), cfg.kv_dim());
        assert_eq!(v.len(), cfg.kv_dim());
        if cfg.use_rope {
            for h in 0..cfg.num_kv_heads {
                let s = h * cfg.head_dim;
                apply_rope_partial_factors_scaled(
                    &mut k[s..s + cfg.head_dim],
                    position,
                    cfg.head_dim,
                    rd,
                    cfg.rope_theta,
                    freq_factors,
                    cfg.rope,
                );
            }
        }
        // Append K/V into INT8 cache (post-RoPE keys).
        for h in 0..cfg.num_kv_heads {
            let s = h * cfg.head_dim;
            cache.heads[h].append(&k[s..s + cfg.head_dim], &v[s..s + cfg.head_dim]);
        }
    }

    let scale = cfg
        .scale_override
        .unwrap_or(1.0 / (cfg.head_dim as f32).sqrt());
    let groups = cfg.groups_per_kv();
    let mut scores = Vec::new();
    let mut acc = vec![0.0f32; cfg.head_dim];

    for q_head in 0..cfg.num_heads {
        let kv_head = q_head / groups;
        let q_slice = &q[q_head * cfg.head_dim..(q_head + 1) * cfg.head_dim];
        let kv = &cache.heads[kv_head];
        let alibi = alibi_slopes.map(|s| (s[q_head], position, cfg.alibi_max_bias));
        kv.attend(q_slice, scale, &mut scores, &mut acc, alibi);
        out[q_head * cfg.head_dim..(q_head + 1) * cfg.head_dim].copy_from_slice(&acc);
    }
}

/// Dense FP32 reference attention (no KV quant) for a single head over a full K/V history.
/// Used only in tests. `keys`/`values` are `[seq * head_dim]` row-major.
pub fn attention_fp32_reference(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    head_dim: usize,
    out: &mut [f32],
) {
    let seq = keys.len() / head_dim;
    assert_eq!(values.len(), keys.len());
    assert_eq!(query.len(), head_dim);
    assert_eq!(out.len(), head_dim);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq];
    for t in 0..seq {
        let mut dot = 0.0f32;
        let k = &keys[t * head_dim..(t + 1) * head_dim];
        for i in 0..head_dim {
            dot += query[i] * k[i];
        }
        scores[t] = dot * scale;
    }
    softmax(&mut scores);
    out.fill(0.0);
    for t in 0..seq {
        let v = &values[t * head_dim..(t + 1) * head_dim];
        for i in 0..head_dim {
            out[i] += scores[t] * v[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::LayerKvCache;
    use crate::matmul::max_abs_diff;

    #[test]
    fn rope_is_norm_preserving() {
        let mut v = [1.0f32, 0.0, 0.0, 1.0];
        let before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope(&mut v, 7, 4, 10000.0);
        let after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((before - after).abs() < 1e-5);
    }

    #[test]
    fn rope_scaling_none_matches_base_and_linear_matches_position_scale() {
        // NONE must be bit-identical to the unscaled RoPE.
        let mut a = [0.5f32, -0.3, 0.8, 0.1];
        let mut b = a;
        apply_rope_partial_factors(&mut a, 7, 4, 4, 10000.0, None);
        apply_rope_partial_factors_scaled(&mut b, 7, 4, 4, 10000.0, None, RopeScaling::NONE);
        assert_eq!(a, b);

        // Linear scaling by `factor` == RoPE at position/factor.
        let rope = RopeScaling {
            freq_scale: 0.5,
            ..RopeScaling::NONE
        };
        let mut scaled = [1.0f32, 0.0, 1.0, 0.0];
        let mut manual = [1.0f32, 0.0, 1.0, 0.0];
        apply_rope_partial_factors_scaled(&mut scaled, 8, 4, 4, 10000.0, None, rope);
        apply_rope_partial_factors(&mut manual, 4, 4, 4, 10000.0, None);
        assert_eq!(scaled, manual);
    }


    #[test]
    fn gqa_decode_matches_fp32_short_context() {
        let cfg = AttentionConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            rope_theta: 10000.0,
            rope_dim: 8,
            scale_override: None,
            alibi_max_bias: 0.0,
            rope: RopeScaling::NONE,
            use_rope: true,
        };
        let mut cache = LayerKvCache::new(cfg.num_kv_heads, cfg.head_dim, 2, 8);

        // Warm cache with a few tokens (no window eviction yet).
        for pos in 0..3 {
            let mut q = vec![0.1f32; cfg.hidden_size()];
            let mut k = vec![0.2f32; cfg.kv_dim()];
            let v = vec![0.3f32; cfg.kv_dim()];
            for i in 0..q.len() {
                q[i] = ((i + pos) as f32) * 0.01;
            }
            for i in 0..k.len() {
                k[i] = ((i + pos * 3) as f32) * 0.02 - 0.1;
            }
            let mut out = vec![0.0f32; cfg.hidden_size()];
            attention_decode_step(&cfg, &mut cache, &mut q, &mut k, &v, pos, &mut out);
            assert!(out.iter().all(|x| x.is_finite()));
        }
        assert_eq!(cache.resident_len(), 3);
    }

    #[test]
    fn single_head_int8_close_to_fp32() {
        let head_dim = 16;
        let mut cache = LayerKvCache::new(1, head_dim, 0, 16);

        let mut keys_fp = Vec::new();
        let mut vals_fp = Vec::new();
        for t in 0..5 {
            let mut k: Vec<f32> = (0..head_dim).map(|i| ((i + t) as f32) * 0.05 - 0.2).collect();
            let v: Vec<f32> = (0..head_dim).map(|i| ((i * 2 + t) as f32) * 0.03).collect();
            apply_rope(&mut k, t, head_dim, 10000.0);
            keys_fp.extend_from_slice(&k);
            vals_fp.extend_from_slice(&v);
            cache.heads[0].append(&k, &v);
        }

        let mut q: Vec<f32> = (0..head_dim).map(|i| i as f32 * 0.04).collect();
        apply_rope(&mut q, 5, head_dim, 10000.0);

        // INT8 path via cache scores
        let slots = cache.heads[0].attention_slots();
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; slots.len()];
        for (i, &slot) in slots.iter().enumerate() {
            scores[i] = cache.heads[0].score_key(slot, &q) * scale;
        }
        softmax(&mut scores);
        let mut out_q = vec![0.0f32; head_dim];
        for (i, &slot) in slots.iter().enumerate() {
            cache.heads[0].accumulate_value(slot, scores[i], &mut out_q);
        }

        let mut out_fp = vec![0.0f32; head_dim];
        attention_fp32_reference(&q, &keys_fp, &vals_fp, head_dim, &mut out_fp);

        let err = max_abs_diff(&out_q, &out_fp);
        assert!(err < 0.15, "INT8 attn too far from FP32: {err}");
    }
}

