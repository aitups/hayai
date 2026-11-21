//! Q8_K input quantisation for integer-dot GEMV (audit C.2).
//!
//! llama.cpp quantises the activation vector to `Q8_K` (256-element blocks, one `f32`
//! scale per block) once per GEMV and then computes the dot product with integer
//! arithmetic (Q4_K/Q5_K/Q6_K/Q8_K weights × Q8_K activations). Doing the same here both
//! matches llama.cpp's numerics and lets the inner loop use integer SIMD.

/// Elements per Q8_K block (matches GGML `QK_K`).
pub const QK_K: usize = 256;

/// A quantised activation vector: one `f32` scale per 256-element block + int8 quants.
#[derive(Clone)]
pub struct Q8K {
    /// Per-block scale `max(|x|)/127`.
    pub d: Vec<f32>,
    /// `n_blocks * QK_K` int8 quants (the last block is not padded — Q4_K/Q6_K `ncols`
    /// is always a multiple of `QK_K`).
    pub qs: Vec<i8>,
    /// The dequantised activations `d * qs` as f32, ready for the f32 dot product.
    pub scaled: Vec<f32>,
    pub n_blocks: usize,
}

impl Q8K {
    /// Quantise `input` (length must be a multiple of [`QK_K`]).
    pub fn quantize(input: &[f32]) -> Self {
        debug_assert_eq!(input.len() % QK_K, 0);
        let n_blocks = input.len() / QK_K;
        let mut d = Vec::with_capacity(n_blocks);
        let mut qs = Vec::with_capacity(n_blocks * QK_K);
        let mut scaled = Vec::with_capacity(n_blocks * QK_K);
        for block in input.chunks_exact(QK_K) {
            let amax = block.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let scale = amax / 127.0;
            d.push(scale);
            let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
            for &x in block {
                let q = (x * inv).round().clamp(-127.0, 127.0) as i8;
                qs.push(q);
                scaled.push(scale * q as f32);
            }
        }
        Self {
            d,
            qs,
            scaled,
            n_blocks,
        }
    }

    #[inline]
    pub fn qs_block(&self, block: usize) -> &[i8] {
        &self.qs[block * QK_K..(block + 1) * QK_K]
    }
}
