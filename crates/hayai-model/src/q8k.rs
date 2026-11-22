//! Q8 input quantisation for integer-dot GEMV (audit C.2 / I1).
//!
//! The activation vector is quantised once per GEMV (`Q8_K` 256-element blocks for the
//! K-quants, `Q8_0` 32-element blocks for Q4_0/Q8_0) and the weight·activation dot is
//! done in **integer** accumulators (`i32`), then scaled once per block. No dequantised
//! f32 copy of the activations is produced.

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
        assert_eq!(
            input.len() % block,
            0,
            "input length must be a multiple of block"
        );
        let n_blocks = input.len() / block;
        let mut d = Vec::with_capacity(n_blocks);
        let mut qs = Vec::with_capacity(input.len());
        for blk in input.chunks_exact(block) {
            let amax = blk.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let scale = amax / 127.0;
            d.push(scale);
            let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
            for &x in blk {
                qs.push((x * inv).round().clamp(-127.0, 127.0) as i8);
            }
        }
        Self {
            block,
            d,
            qs,
            n_blocks,
        }
    }

    #[inline]
    pub fn qs_block(&self, block: usize) -> &[i8] {
        &self.qs[block * self.block..(block + 1) * self.block]
    }
}

/// Integer dot product of two `i8` slices with an `i32` accumulator.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 availability was just checked.
            return unsafe { dot_i8_avx2(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = 0i32;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += (*x as i32) * (*y as i32);
    }
    acc
}

/// Real integer SIMD: widen to `i16` and use `_mm256_madd_epi16` (pairwise i16 products
/// summed into i32 lanes), eight lanes per instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let alo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(av));
        let blo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bv));
        let ahi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
        let bhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 1));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(alo, blo));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ahi, bhi));
        i += 32;
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut total: i32 = lanes.iter().sum();
    while i < n {
        total += (a[i] as i32) * (b[i] as i32);
        i += 1;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_i8_matches_scalar() {
        let a: Vec<i8> = (0..257).map(|i| ((i * 7) % 251) as i8).collect();
        let b: Vec<i8> = (0..257).map(|i| ((i * 13) % 251) as i8).collect();
        assert_eq!(dot_i8(&a, &b), dot_i8_scalar(&a, &b));
    }

    #[test]
    fn quantize_roundtrips_within_scale() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.3).collect();
        let q = Q8K::quantize(&x);
        assert_eq!(q.qs.len(), 256);
        for (i, &xi) in x.iter().enumerate() {
            let deq = q.d[0] * q.qs[i] as f32;
            assert!((deq - xi).abs() <= q.d[0] * 0.5 + 1e-5);
        }
    }
}
