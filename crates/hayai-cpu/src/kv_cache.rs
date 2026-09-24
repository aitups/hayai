//! Bounded KV cache: Attention Sinks + Sliding Window + channel-wise INT8 + recent FP.
//!
//! PRD §3.4 / design (KIVI-style):
//! - History compressed with **per-channel** INT8 scales (outlier-safe per dim).
//! - Newest `recent_fp_tokens` also kept in FP32 for attention (small full-precision window).
//! - Resident token count never exceeds `sinks + window`.

use crate::simd_ops::{simd_dot, simd_dot_i8_channel};

/// Default recent full-precision tokens.
pub const DEFAULT_RECENT_FP_TOKENS: usize = 32;

/// Bounded KV with sinks + sliding window.
#[derive(Clone)]
pub struct BoundedKvCache {
    pub max_seq_len: usize,
    pub num_sink_tokens: usize,
    pub window_size: usize,
    pub recent_fp_tokens: usize,
    pub key_cache: Vec<i8>,
    pub value_cache: Vec<i8>,
    /// Per-channel scales: `[max_seq_len * dim]`
    pub k_scales: Vec<f32>,
    pub v_scales: Vec<f32>,
    /// FP overlay for newest tokens: `[recent_fp_tokens * dim]`
    recent_k: Vec<f32>,
    recent_v: Vec<f32>,
    /// Physical slot id for each recent ring entry (`usize::MAX` if empty).
    recent_slot: Vec<usize>,
    recent_len: usize,
    recent_write: usize,
    pub current_len: usize,
    pub dim: usize,
}

impl BoundedKvCache {
    pub fn new(dim: usize, num_sink_tokens: usize, window_size: usize) -> Self {
        Self::with_recent_fp(dim, num_sink_tokens, window_size, DEFAULT_RECENT_FP_TOKENS)
    }

    pub fn with_recent_fp(
        dim: usize,
        num_sink_tokens: usize,
        window_size: usize,
        recent_fp_tokens: usize,
    ) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        let recent_fp_tokens = recent_fp_tokens.max(1).min(window_size);
        let max_seq_len = num_sink_tokens + window_size;
        Self {
            max_seq_len,
            num_sink_tokens,
            window_size,
            recent_fp_tokens,
            key_cache: vec![0; max_seq_len * dim],
            value_cache: vec![0; max_seq_len * dim],
            k_scales: vec![1.0; max_seq_len * dim],
            v_scales: vec![1.0; max_seq_len * dim],
            recent_k: vec![0.0; recent_fp_tokens * dim],
            recent_v: vec![0.0; recent_fp_tokens * dim],
            recent_slot: vec![usize::MAX; recent_fp_tokens],
            recent_len: 0,
            recent_write: 0,
            current_len: 0,
            dim,
        }
    }

    pub fn resident_len(&self) -> usize {
        self.current_len.min(self.max_seq_len)
    }

    fn write_slot(current_len: usize, num_sink: usize, window: usize, max_seq: usize) -> usize {
        if current_len < max_seq {
            current_len
        } else {
            num_sink + ((current_len - num_sink) % window)
        }
    }

    fn quantize_channel_wise(src: &[f32], dst: &mut [i8], scales: &mut [f32]) {
        for i in 0..src.len() {
            let a = src[i].abs().max(1e-8);
            let scale = a / 127.0;
            scales[i] = scale;
            dst[i] = (src[i] / scale).clamp(-127.0, 127.0) as i8;
        }
    }

    /// Append K/V: always compress into sinks/window (channel-wise INT8) and
    /// mirror into the recent FP ring for high-fidelity attention on newest tokens.
    pub fn append(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.dim);
        assert_eq!(value.len(), self.dim);

        let target_idx = Self::write_slot(
            self.current_len,
            self.num_sink_tokens,
            self.window_size,
            self.max_seq_len,
        );
        let start = target_idx * self.dim;
        Self::quantize_channel_wise(
            key,
            &mut self.key_cache[start..start + self.dim],
            &mut self.k_scales[start..start + self.dim],
        );
        Self::quantize_channel_wise(
            value,
            &mut self.value_cache[start..start + self.dim],
            &mut self.v_scales[start..start + self.dim],
        );

        // FP overlay (ring).
        if self.recent_len < self.recent_fp_tokens {
            self.recent_len += 1;
        }
        let w = self.recent_write;
        self.recent_k[w * self.dim..(w + 1) * self.dim].copy_from_slice(key);
        self.recent_v[w * self.dim..(w + 1) * self.dim].copy_from_slice(value);
        self.recent_slot[w] = target_idx;
        self.recent_write = (w + 1) % self.recent_fp_tokens;

        self.current_len += 1;
    }

    /// Physical slot indices in causal attention order.
    pub fn attention_slots(&self) -> Vec<usize> {
        let resident = self.resident_len();
        if resident == 0 {
            return Vec::new();
        }
        if self.current_len <= self.max_seq_len {
            return (0..resident).collect();
        }
        let mut slots = Vec::with_capacity(self.max_seq_len);
        slots.extend(0..self.num_sink_tokens);
        let oldest = (self.current_len - self.num_sink_tokens) % self.window_size;
        for i in 0..self.window_size {
            slots.push(self.num_sink_tokens + ((oldest + i) % self.window_size));
        }
        slots
    }

    /// Absolute token position of each entry of [`Self::attention_slots`] (same order).
    /// Used by ALiBi (BLOOM/Falcon/MPT), whose score bias depends on `q_pos - k_pos`.
    pub fn attention_slot_positions(&self) -> Vec<usize> {
        let resident = self.resident_len();
        if resident == 0 {
            return Vec::new();
        }
        if self.current_len <= self.max_seq_len {
            return (0..resident).collect();
        }
        let mut pos = Vec::with_capacity(self.max_seq_len);
        pos.extend(0..self.num_sink_tokens);
        let win_start = self.current_len - self.window_size;
        for i in 0..self.window_size {
            pos.push(win_start + i);
        }
        pos
    }

    fn recent_fp_for_slot(&self, slot: usize) -> Option<usize> {
        for i in 0..self.recent_len {
            let idx = if self.recent_len < self.recent_fp_tokens {
                i
            } else {
                (self.recent_write + i) % self.recent_fp_tokens
            };
            if self.recent_slot[idx] == slot {
                return Some(idx);
            }
        }
        None
    }

    pub fn dequantize_key(&self, slot: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        if let Some(ri) = self.recent_fp_for_slot(slot) {
            out.copy_from_slice(&self.recent_k[ri * self.dim..(ri + 1) * self.dim]);
            return;
        }
        let start = slot * self.dim;
        for i in 0..self.dim {
            out[i] = self.key_cache[start + i] as f32 * self.k_scales[start + i];
        }
    }

    pub fn dequantize_value(&self, slot: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        if let Some(ri) = self.recent_fp_for_slot(slot) {
            out.copy_from_slice(&self.recent_v[ri * self.dim..(ri + 1) * self.dim]);
            return;
        }
        let start = slot * self.dim;
        for i in 0..self.dim {
            out[i] = self.value_cache[start + i] as f32 * self.v_scales[start + i];
        }
    }

    pub fn score_key(&self, slot: usize, query: &[f32]) -> f32 {
        if let Some(ri) = self.recent_fp_for_slot(slot) {
            return simd_dot(
                query,
                &self.recent_k[ri * self.dim..(ri + 1) * self.dim],
            );
        }
        let start = slot * self.dim;
        simd_dot_i8_channel(
            query,
            &self.key_cache[start..start + self.dim],
            &self.k_scales[start..start + self.dim],
        )
    }

    pub fn accumulate_value(&self, slot: usize, weight: f32, acc: &mut [f32]) {
        if let Some(ri) = self.recent_fp_for_slot(slot) {
            let v = &self.recent_v[ri * self.dim..(ri + 1) * self.dim];
            for i in 0..self.dim {
                acc[i] += v[i] * weight;
            }
            return;
        }
        let start = slot * self.dim;
        for i in 0..self.dim {
            acc[i] += self.value_cache[start + i] as f32 * self.v_scales[start + i] * weight;
        }
    }

    /// Softmax attention over resident slots (FP for recent, INT8 channel-wise otherwise).
    ///
    /// `alibi`: optional `(slope, query_position)` — adds `-slope * (q_pos - k_pos)`
    /// to every score before softmax (BLOOM/Falcon/MPT).
    pub fn attend(
        &self,
        query: &[f32],
        scale: f32,
        scores_out: &mut Vec<f32>,
        acc: &mut [f32],
        alibi: Option<(f32, usize)>,
    ) {
        assert_eq!(query.len(), self.dim);
        assert_eq!(acc.len(), self.dim);
        acc.fill(0.0);
        let slots = self.attention_slots();
        let positions = if alibi.is_some() {
            Some(self.attention_slot_positions())
        } else {
            None
        };
        scores_out.clear();
        scores_out.reserve(slots.len());
        for (idx, &slot) in slots.iter().enumerate() {
            let mut s = self.score_key(slot, query) * scale;
            if let (Some((slope, q_pos)), Some(pos)) = (alibi, positions.as_ref()) {
                s -= slope * (q_pos as f32 - pos[idx] as f32);
            }
            scores_out.push(s);
        }
        if scores_out.is_empty() {
            return;
        }
        let max_val = scores_out
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores_out.iter_mut() {
            *s = (*s - max_val).exp();
            sum += *s;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for s in scores_out.iter_mut() {
            *s *= inv;
        }
        for (i, &slot) in slots.iter().enumerate() {
            self.accumulate_value(slot, scores_out[i], acc);
        }
    }
}

/// Per-layer KV state: one bounded cache per KV head (GQA-friendly).
#[derive(Clone)]
pub struct LayerKvCache {
    pub heads: Vec<BoundedKvCache>,
}

impl LayerKvCache {
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        num_sink_tokens: usize,
        window_size: usize,
    ) -> Self {
        Self {
            heads: (0..num_kv_heads)
                .map(|_| BoundedKvCache::new(head_dim, num_sink_tokens, window_size))
                .collect(),
        }
    }

    pub fn resident_len(&self) -> usize {
        self.heads.first().map(|h| h.resident_len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinks_preserved_under_eviction() {
        let mut cache = BoundedKvCache::with_recent_fp(4, 2, 3, 2);
        for t in 0..10 {
            let k = [t as f32, 0.0, 0.0, 0.0];
            let v = [t as f32 * 10.0, 0.0, 0.0, 0.0];
            cache.append(&k, &v);
        }
        assert_eq!(cache.resident_len(), 5);
        assert_eq!(cache.current_len, 10);

        let mut out = [0.0f32; 4];
        cache.dequantize_key(0, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-3, "sink0 overwritten: {:?}", out);
        cache.dequantize_key(1, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-3, "sink1 overwritten: {:?}", out);

        let slots = cache.attention_slots();
        assert_eq!(slots.len(), 5);
        assert_eq!(&slots[..2], &[0, 1]);
    }

    #[test]
    fn channel_wise_preserves_outlier_channel() {
        let mut cache = BoundedKvCache::with_recent_fp(4, 0, 4, 1);
        cache.append(&[1.0, 100.0, 0.0, 0.0], &[1.0, 1.0, 1.0, 1.0]);
        // Force slot 0 out of recent FP by filling recent with another token on same window.
        cache.append(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]);
        // Slot 0 may still be recent if recent_fp=1 holds token1. Append more.
        cache.append(&[0.5, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]);
        let mut out = [0.0f32; 4];
        // Read INT8 path: slot 0 should be token0 if not overwritten; window size 4 so still there.
        // recent holds latest slot only — slot 0 uses INT8 channel-wise.
        cache.dequantize_key(0, &mut out);
        assert!((out[0] - 1.0).abs() < 0.05, "{out:?}");
        assert!(
            (out[1] - 100.0).abs() < 0.5,
            "outlier channel destroyed: {out:?}"
        );
    }

    #[test]
    fn score_and_accumulate_roundtrip() {
        let mut cache = BoundedKvCache::with_recent_fp(4, 0, 4, 4);
        let key = [1.0, -1.0, 0.5, 0.0];
        let value = [2.0, 3.0, 4.0, 5.0];
        cache.append(&key, &value);
        let q = [1.0, 0.0, 0.0, 0.0];
        let score = cache.score_key(0, &q);
        assert!((score - 1.0).abs() < 0.05);

        let mut acc = [0.0f32; 4];
        cache.accumulate_value(0, 1.0, &mut acc);
        for i in 0..4 {
            assert!((acc[i] - value[i]).abs() < 0.05);
        }
    }
}
