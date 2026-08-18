//! GGML Q4_K (used by Q4_K_S / Q4_K_M GGUF files) — block size QK_K=256, 144 bytes/block.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK_K: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144; // d(2)+dmin(2)+scales(12)+qs(128)

#[inline]
fn gemv_q4k_sub32(d: f32, minv: f32, q: &[u8], x: &[f32], high_nibble: bool) -> f32 {
    use std::simd::f32x8;
    use std::simd::num::SimdFloat;
    let mut acc = f32x8::splat(0.0);
    let d_v = f32x8::splat(d);
    let m_v = f32x8::splat(minv);
    for chunk in 0..4 {
        let o = chunk * 8;
        let mut w = [0.0f32; 8];
        for i in 0..8 {
            let nibble = if high_nibble {
                q[o + i] >> 4
            } else {
                q[o + i] & 0x0F
            };
            w[i] = nibble as f32;
        }
        let wv = f32x8::from_array(w) * d_v - m_v;
        let xv = f32x8::from_slice(&x[o..o + 8]);
        acc += wv * xv;
    }
    acc.reduce_sum()
}

#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize a contiguous Q4_K blob of `n` elements (n % 256 == 0) into FP32.
pub fn dequant_q4_k(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK_K != 0 {
        return Err(GgufError::Truncated("q4_k n not multiple of 256"));
    }
    let blocks = n / QK_K;
    if bytes.len() < blocks * Q4_K_BLOCK_BYTES {
        return Err(GgufError::Truncated("q4_k tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_q4_k_block(&bytes[i * Q4_K_BLOCK_BYTES..], &mut out[i * QK_K..]);
    }
    Ok(out)
}

fn dequant_q4_k_block(block: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let mut q = &block[16..144];
    let mut is = 0usize;
    let mut yo = 0usize;
    for _ in 0..(QK_K / 64) {
        let (sc0, m0) = get_scale_min_k4(is, scales);
        let (sc1, m1) = get_scale_min_k4(is + 1, scales);
        let d1 = d * sc0 as f32;
        let m1v = min * m0 as f32;
        let d2 = d * sc1 as f32;
        let m2v = min * m1 as f32;
        for l in 0..32 {
            y[yo + l] = d1 * (q[l] & 0x0F) as f32 - m1v;
        }
        yo += 32;
        for l in 0..32 {
            y[yo + l] = d2 * (q[l] >> 4) as f32 - m2v;
        }
        yo += 32;
        q = &q[32..];
        is += 2;
    }
}

/// GEMV: `output[nrows] = W @ input[ncols]` with W stored as row-major Q4_K (ne0=ncols).
pub fn gemv_q4_k(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!(
            "Q4_K ncols={ncols} not multiple of {QK_K}"
        )));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q4_k gemv"));
    }
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * Q4_K_BLOCK_BYTES..];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let mut q = &block[16..144];
            let mut is = 0usize;
            let x_base = b * QK_K;
            let mut xo = 0usize;
            for _ in 0..(QK_K / 64) {
                let (sc0, m0) = get_scale_min_k4(is, scales);
                let (sc1, m1) = get_scale_min_k4(is + 1, scales);
                let d1 = d * sc0 as f32;
                let m1v = min * m0 as f32;
                let d2 = d * sc1 as f32;
                let m2v = min * m1 as f32;
                sum += gemv_q4k_sub32(d1, m1v, &q[..32], &input[x_base + xo..], false);
                xo += 32;
                sum += gemv_q4k_sub32(d2, m2v, &q[..32], &input[x_base + xo..], true);
                xo += 32;
                q = &q[32..];
                is += 2;
            }
        }
        *out = sum;
    });
    Ok(())
}

/// Extract one embedding row from a Q4_K matrix (nrows = vocab, ncols = hidden).
pub fn extract_q4_k_row(data: &[u8], ncols: usize, row: usize, out: &mut [f32]) -> Result<(), GgufError> {
    if ncols % QK_K != 0 || out.len() != ncols {
        return Err(GgufError::Truncated("q4_k embed row"));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q4_K_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {
        return Err(GgufError::Truncated("q4_k embed OOB"));
    }
    for b in 0..blocks_per_row {
        dequant_q4_k_block(
            &data[start + b * Q4_K_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ok() {
        let out = dequant_q4_k(&[], 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn block_roundtrip_finite() {
        // Synthetic block: d=1.0 f16, dmin=0, scales small, qs mid
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        // f16 1.0 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        // dmin = 0
        for i in 4..16 {
            block[i] = 1; // tiny scales
        }
        for i in 16..144 {
            block[i] = 0x11;
        }
        let mut y = vec![0.0f32; QK_K];
        dequant_q4_k_block(&block, &mut y);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
