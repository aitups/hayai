//! Bounded KV cache: Attention Sinks + Sliding Window + channel-wise INT8 + recent FP.
//!
//! PRD §3.4 / design (KIVI-style):
//! - History compressed with **per-channel** INT8 scales (outlier-safe per dim).
//! - Newest `recent_fp_tokens` also kept in FP32 for attention (small full-precision window).
//! - Resident token count never exceeds `sinks + window`.

use crate::simd_ops::{simd_dot, simd_dot_i8_block};

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
    /// Per-block scales: `[max_seq_len * n_blocks]` (audit C.3 — ~1.06 B/elem vs 5).
    pub k_scales: Vec<f32>,
    pub v_scales: Vec<f32>,
    /// Elements per INT8 scale block (32, or `dim` if smaller).
    pub k_block: usize,
    /// Number of scale blocks per vector (`ceil(dim / k_block)`).
    pub n_blocks: usize,
    /// FP overlay for newest tokens: `[recent_fp_tokens * dim]`
    recent_k: Vec<f32>,
    recent_v: Vec<f32>,
    /// Physical slot id for each recent ring entry (`usize::MAX` if empty).
    recent_slot: Vec<usize>,
    /// Reverse map: physical slot → recent ring index (`None` when not in the ring).
    /// O(1) lookup replacing the linear scan in `recent_fp_for_slot` (audit C.3.2).
    slot_recent: Vec<Option<usize>>,
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
        let k_block = 32usize.min(dim).max(1);
        let n_blocks = (dim + k_block - 1) / k_block;
        Self {
            max_seq_len,
            num_sink_tokens,
            window_size,
            recent_fp_tokens,
            key_cache: vec![0; max_seq_len * dim],
            value_cache: vec![0; max_seq_len * dim],
            k_scales: vec![1.0; max_seq_len * n_blocks],
            v_scales: vec![1.0; max_seq_len * n_blocks],
            k_block,
            n_blocks,
            recent_k: vec![0.0; recent_fp_tokens * dim],
            recent_v: vec![0.0; recent_fp_tokens * dim],
            recent_slot: vec![usize::MAX; recent_fp_tokens],
            slot_recent: vec![None; max_seq_len],
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

    fn quantize_block_wise(src: &[f32], dst: &mut [i8], scales: &mut [f32], block: usize) {
        let mut b = 0;
        while b < src.len() {
            let end = (b + block).min(src.len());
            let max_abs = src[b..end]
                .iter()
                .fold(0.0f32, |m, &v| m.max(v.abs()))
                .max(1e-8);
            let scale = max_abs / 127.0;
            scales[b / block] = scale;
            for i in b..end {
                dst[i] = (src[i] / scale).clamp(-127.0, 127.0) as i8;
            }
            b = end;
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
        let sb = target_idx * self.n_blocks;
        Self::quantize_block_wise(
            key,
            &mut self.key_cache[start..start + self.dim],
            &mut self.k_scales[sb..sb + self.n_blocks],
            self.k_block,
        );
        Self::quantize_block_wise(
            value,
            &mut self.value_cache[start..start + self.dim],
            &mut self.v_scales[sb..sb + self.n_blocks],
            self.k_block,
        );

        // FP overlay (ring).
        if self.recent_len < self.recent_fp_tokens {
            self.recent_len += 1;
        }
        let w = self.recent_write;
        // Clear the reverse mapping of the slot this ring position is about to evict.
        let old = self.recent_slot[w];
        if old != usize::MAX && old < self.slot_recent.len() {
            self.slot_recent[old] = None;
        }
        self.recent_k[w * self.dim..(w + 1) * self.dim].copy_from_slice(key);
        self.recent_v[w * self.dim..(w + 1) * self.dim].copy_from_slice(value);
        self.recent_slot[w] = target_idx;
        self.slot_recent[target_idx] = Some(w);
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
        // O(1) via the reverse map (audit C.3.2 — was a linear scan per head per layer).
        self.slot_recent.get(slot).copied().flatten()
    }

    pub fn dequantize_key(&self, slot: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        if let Some(ri) = self.recent_fp_for_slot(slot) {
            out.copy_from_slice(&self.recent_k[ri * self.dim..(ri + 1) * self.dim]);
            return;
        }
        let start = slot * self.dim;
        let sb = slot * self.n_blocks;
        for i in 0..self.dim {
            out[i] = self.key_cache[start + i] as f32 * self.k_scales[sb + i / self.k_block];
        }
    }

    pub fn dequantize_value(&self, slot: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        if let Some(ri) = self.recent_fp_for_slot(slot) {
            out.copy_from_slice(&self.recent_v[ri * self.dim..(ri + 1) * self.dim]);
            return;
        }
        let start = slot * self.dim;
        let sb = slot * self.n_blocks;
        for i in 0..self.dim {
            out[i] = self.value_cache[start + i] as f32 * self.v_scales[sb + i / self.k_block];
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
        let sb = slot * self.n_blocks;
        simd_dot_i8_block(
            query,
            &self.key_cache[start..start + self.dim],
            &self.k_scales[sb..sb + self.n_blocks],
            self.k_block,
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
        let sb = slot * self.n_blocks;
        for i in 0..self.dim {
            acc[i] += self.value_cache[start + i] as f32 * self.v_scales[sb + i / self.k_block] * weight;
        }
    }

    /// Softmax attention over resident slots (FP for recent, INT8 channel-wise otherwise).
    ///
    /// `alibi`: optional `(slope, query_position, max_bias)` — adds
    /// `max(-slope * (q_pos - k_pos), -max_bias)` to every score before softmax
    /// (BLOOM/Falcon/MPT; `max_bias <= 0` disables the clamp).
    pub fn attend(
        &self,
        query: &[f32],
        scale: f32,
        scores_out: &mut Vec<f32>,
        acc: &mut [f32],
        alibi: Option<(f32, usize, f32)>,
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
            if let (Some((slope, q_pos, max_bias)), Some(pos)) = (alibi, positions.as_ref()) {
                let mut b = -slope * (q_pos as f32 - pos[idx] as f32);
                if max_bias > 0.0 {
                    b = b.max(-max_bias);
                }
                s += b;
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
    fn alibi_bias_is_clamped_to_max() {
        // 3 keys with zero scores; slope 1, q_pos 2 → biases -2,-1,0; clamp at -1 → -1,-1,0.
        let mut cache = BoundedKvCache::with_recent_fp(1, 0, 8, 8);
        for _ in 0..3 {
            cache.append(&[0.0], &[0.0]);
        }
        let mut scores = Vec::new();
        let mut acc = [0.0f32; 1];
        cache.attend(&[0.0], 1.0, &mut scores, &mut acc, Some((1.0, 2, 1.0)));
        assert!(
            (scores[0] - scores[1]).abs() < 1e-6,
            "bias not clamped: {scores:?}"
        );
        assert!(scores[2] > scores[1], "recent key should dominate: {scores:?}");
    }

    #[test]
    fn block_wise_scales_roundtrip() {
        // Per-block INT8 (audit C.3): dim 64 → 2 blocks of 32, one scale each.
        let mut cache = BoundedKvCache::with_recent_fp(64, 0, 8, 1);
        assert_eq!(cache.n_blocks, 2);
        let mut key = vec![0.0f32; 64];
        key[0] = 3.0;
        key[40] = -2.0;
        let mut val = vec![0.0f32; 64];
        for (i, v) in val.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 3.0;
        }
        cache.append(&key, &val);
        // Push slot 0 out of the recent FP ring (recent_fp=1 holds the latest token).
        cache.append(&vec![0.0f32; 64], &vec![0.0f32; 64]);
        let mut out = vec![0.0f32; 64];
        cache.dequantize_key(0, &mut out);
        assert!((out[0] - 3.0).abs() < 0.1, "block0 {out:?}");
        assert!((out[40] + 2.0).abs() < 0.1, "block1 {out:?}");
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


