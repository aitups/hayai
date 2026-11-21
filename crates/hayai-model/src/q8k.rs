//! Q8 input quantisation for integer-dot GEMV (audit C.2).
//!
//! llama.cpp quantises the activation vector once per GEMV — `Q8_K` (256-element blocks)
//! for the K-quants, `Q8_0` (32-element blocks) for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 — and then
//! computes the weight×activation dot in f32/integer units. Doing the same here both
//! matches llama.cpp's numerics and lets the inner loop avoid the full-precision input.

/// Elements per Q8_K block (matches GGML `QK_K`).
pub const QK_K: usize = 256;

/// A quantised activation vector: one `f32` scale per `block` elements + int8 quants.
#[derive(Clone)]
pub struct Q8K {
    /// Block size (`256` for the K-quants, `32` for the Q8_0 family).
    pub block: usize,
    /// Per-block scale `max(|x|)/127`.
    pub d: Vec<f32>,
    /// `n_blocks * block` int8 quants (input length must be a multiple of `block`).
    pub qs: Vec<i8>,
    /// The dequantised activations `d * qs` as f32, ready for the f32 dot product.
    pub scaled: Vec<f32>,
    pub n_blocks: usize,
}

impl Q8K {
    /// Quantise with 256-element blocks (Q8_K, for Q4_K/Q5_K/Q6_K/Q8_K).
    pub fn quantize(input: &[f32]) -> Self {
        Self::quantize_blocked(input, QK_K)
    }

    /// Quantise with an explicit block size (`32` for Q8_0-style inputs).
    pub fn quantize_blocked(input: &[f32], block: usize) -> Self {
        assert!(block > 0);
        assert_eq!(input.len() % block, 0, "input length must be a multiple of block");
        let n_blocks = input.len() / block;
        let mut d = Vec::with_capacity(n_blocks);
        let mut qs = Vec::with_capacity(input.len());
        let mut scaled = Vec::with_capacity(input.len());
        for blk in input.chunks_exact(block) {
            let amax = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let scale = amax / 127.0;
            d.push(scale);
            let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
            for &x in blk {
                let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
                qs.push(q);
                scaled.push(scale * q as f32);
            }
        }
        Self {
            block,
            d,
            qs,
            scaled,
            n_blocks,
        }
    }

    #[inline]
    pub fn qs_block(&self, block: usize) -> &[i8] {
        &self.qs[block * self.block..(block + 1) * self.block]
    }
}
