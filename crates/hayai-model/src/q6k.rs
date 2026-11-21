//! GGML Q6_K (used heavily in Q4_K_M / Q5_K_M mixes for output / some FFN tensors).

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use crate::q4k::QK_K;
use rayon::prelude::*;

/// ql(128) + qh(64) + scales(16) + d(2) = 210
pub const Q6_K_BLOCK_BYTES: usize = 210;

pub fn dequant_q6_k(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK_K != 0 {
        return Err(GgufError::Truncated("q6_k n"));
    }
    let blocks = n / QK_K;
    if bytes.len() < blocks * Q6_K_BLOCK_BYTES {
        return Err(GgufError::Truncated("q6_k tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_q6_k_block(&bytes[i * Q6_K_BLOCK_BYTES..], &mut out[i * QK_K..]);
    }
    Ok(out)
}

fn dequant_q6_k_block(block: &[u8], y: &mut [f32]) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut yo = 0usize;
    let mut qlo = 0usize;
    let mut qho = 0usize;
    let mut sco = 0usize;
    for _ in 0..(QK_K / 128) {
        for l in 0..32 {
            let is = l / 16;
            let sc = scales;
            let q1 = ((ql[qlo + l] & 0x0F) | (((qh[qho + l] >> 0) & 3) << 4)) as i8 as i32 - 32;
            let q2 = ((ql[qlo + l + 32] & 0x0F) | (((qh[qho + l] >> 2) & 3) << 4)) as i8 as i32 - 32;
            let q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) as i8 as i32 - 32;
            let q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) as i8 as i32 - 32;
            let s0 = sc[sco + is] as i8 as f32;
            let s2 = sc[sco + is + 2] as i8 as f32;
            let s4 = sc[sco + is + 4] as i8 as f32;
            let s6 = sc[sco + is + 6] as i8 as f32;
            y[yo + l] = d * s0 * q1 as f32;
            y[yo + l + 32] = d * s2 * q2 as f32;
            y[yo + l + 64] = d * s4 * q3 as f32;
            y[yo + l + 96] = d * s6 * q4 as f32;
        }
        yo += 128;
        qlo += 64;
        qho += 32;
        sco += 8;
    }
}

pub fn gemv_q6_k(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!("Q6_K ncols={ncols}")));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q6_k gemv"));
    }
    // Audit C.2.1: dequantise one 256-element block into a stack buffer and dot it,
    // instead of `vec![0.0f32; ncols]` per row (the lm_head fired ~152k malloc+free per
    // token and thrashed the global allocator across the Rayon threads).
    let use_avx2 = crate::simd_dispatch::x86_avx2_fma();
    output
        .par_iter_mut()
        .with_min_len(64)
        .enumerate()
        .for_each(|(row, out)| {
            let row_base = row * row_bytes;
            *out = gemv_q6k_dispatch(use_avx2, data, input, row_base, blocks_per_row);
        });
    Ok(())
}

#[inline(always)]
fn gemv_q6k_row(data: &[u8], input: &[f32], row_base: usize, blocks_per_row: usize) -> f32 {
    use std::simd::f32x8;
    use std::simd::num::SimdFloat;
    let mut sum = 0.0f32;
    let mut block_vals = [0.0f32; QK_K];
    for b in 0..blocks_per_row {
        dequant_q6_k_block(&data[row_base + b * Q6_K_BLOCK_BYTES..], &mut block_vals);
        let xs = &input[b * QK_K..(b + 1) * QK_K];
        let mut acc = [f32x8::splat(0.0); 4];
        let mut c = 0;
        let mut k = 0;
        while c + 8 <= QK_K {
            acc[k] += f32x8::from_slice(&block_vals[c..c + 8]) * f32x8::from_slice(&xs[c..c + 8]);
            k = (k + 1) & 3;
            c += 8;
        }
        sum += (acc[0] + acc[1] + acc[2] + acc[3]).reduce_sum();
        while c < QK_K {
            sum += block_vals[c] * xs[c];
            c += 1;
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_q6k_row_avx2(
    data: &[u8],
    input: &[f32],
    row_base: usize,
    blocks_per_row: usize,
) -> f32 {
    gemv_q6k_row(data, input, row_base, blocks_per_row)
}

#[inline]
fn gemv_q6k_dispatch(
    use_avx2: bool,
    data: &[u8],
    input: &[f32],
    row_base: usize,
    blocks_per_row: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2 {
            return unsafe { gemv_q6k_row_avx2(data, input, row_base, blocks_per_row) };
        }
    }
    let _ = use_avx2;
    gemv_q6k_row(data, input, row_base, blocks_per_row)
}

/// Q6_K × Q8_K GEMV (audit C.2): quantise the activation once to Q8_K, then the existing
/// f32 dot on the pre-scaled int8 activations (exact; 4 accumulators for ILP).
pub fn gemv_q6_k_q8k(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!("Q6_K ncols={ncols}")));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q6_K_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q6_k gemv"));
    }
    let q8 = crate::q8k::Q8K::quantize(input);
    let use_avx2 = crate::simd_dispatch::x86_avx2_fma();
    output
        .par_iter_mut()
        .with_min_len(64)
        .enumerate()
        .for_each(|(row, out)| {
            let row_base = row * row_bytes;
            *out = gemv_q6k_dispatch(use_avx2, data, &q8.scaled, row_base, blocks_per_row);
        });
    Ok(())
}

pub fn extract_q6_k_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    let blocks = ncols / QK_K;
    let row_bytes = blocks * Q6_K_BLOCK_BYTES;
    let start = row * row_bytes;
    for b in 0..blocks {
        dequant_q6_k_block(
            &data[start + b * Q6_K_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}
