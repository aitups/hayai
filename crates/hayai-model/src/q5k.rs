//! GGML Q5_K — QK_K=256, 176 bytes/block.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK_K: usize = 256;
pub const Q5_K_BLOCK_BYTES: usize = 176; // d(2)+dmin(2)+scales(12)+qh(32)+qs(128)

fn scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

pub fn dequant_q5_k(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK_K != 0 {
        return Err(GgufError::Truncated("q5_k n not multiple of 256"));
    }
    let blocks = n / QK_K;
    if bytes.len() < blocks * Q5_K_BLOCK_BYTES {
        return Err(GgufError::Truncated("q5_k tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_q5_k_block(&bytes[i * Q5_K_BLOCK_BYTES..], &mut out[i * QK_K..]);
    }
    Ok(out)
}

fn dequant_q5_k_block(block: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let mut ql = &block[48..176];
    let mut yo = 0usize;
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for _ in 0..(QK_K / 64) {
        let (sc0, m0) = scale_min(is, scales);
        let (sc1, m1) = scale_min(is + 1, scales);
        let d1 = d * sc0 as f32;
        let m1v = min * m0 as f32;
        let d2 = d * sc1 as f32;
        let m2v = min * m1 as f32;
        for l in 0..32 {
            let v = (ql[l] & 0x0F) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
            y[yo + l] = d1 * v as f32 - m1v;
        }
        yo += 32;
        for l in 0..32 {
            let v = (ql[l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
            y[yo + l] = d2 * v as f32 - m2v;
        }
        yo += 32;
        ql = &ql[32..];
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

pub fn gemv_q5_k(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!(
            "Q5_K ncols={ncols} not multiple of {QK_K}"
        )));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q5_k gemv"));
    }
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().with_min_len(64).enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * Q5_K_BLOCK_BYTES..];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let scales = &block[4..16];
            let qh = &block[16..48];
            let mut ql = &block[48..176];
            let x_base = b * QK_K;
            let mut xo = 0usize;
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for _ in 0..(QK_K / 64) {
                let (sc0, m0) = scale_min(is, scales);
                let (sc1, m1) = scale_min(is + 1, scales);
                let d1 = d * sc0 as f32;
                let m1v = min * m0 as f32;
                let d2 = d * sc1 as f32;
                let m2v = min * m1 as f32;
                for l in 0..32 {
                    let v = (ql[l] & 0x0F) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
                    sum += (d1 * v as f32 - m1v) * input[x_base + xo + l];
                }
                xo += 32;
                for l in 0..32 {
                    let v = (ql[l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                    sum += (d2 * v as f32 - m2v) * input[x_base + xo + l];
                }
                xo += 32;
                ql = &ql[32..];
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        *out = sum;
    });
    Ok(())
}

/// Q5_K × Q8_K GEMV (audit C.2): quantise the activation to Q8_K (256-element blocks),
/// then the f32 dot on the pre-scaled int8 activations (exact for these small ints).
pub fn gemv_q5_k_q8k(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!("Q5_K ncols={ncols}")));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q5_k gemv"));
    }
    // Real integer dot (audit C.2/I1): 5-bit q (nibble + high bit) in i32, scale/min once.
    let q8 = crate::q8k::Q8K::quantize(input);
    for (row, out) in output.iter_mut().enumerate() {
        *out = gemv_q5k_row_q8k(data, &q8, row, blocks_per_row);
    }
    Ok(())
}

#[inline]
fn gemv_q5k_row_q8k(data: &[u8], q8: &crate::q8k::Q8K, row: usize, blocks: usize) -> f32 {
    let row_base = row * blocks * Q5_K_BLOCK_BYTES;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let block = &data[row_base + b * Q5_K_BLOCK_BYTES..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let ql = &block[48..176];
        let dq = q8.d[b];
        let xq = q8.qs_block(b);
        let mut is = 0usize;
        let mut xo = 0usize;
        let mut qo = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..4 {
            let (sc0, m0) = scale_min(is, scales);
            let (sc1, m1) = scale_min(is + 1, scales);
            let (mut sqx0, mut sx0) = (0i32, 0i32);
            for l in 0..32 {
                let v = (ql[qo + l] & 0x0F) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 };
                let x = xq[xo + l] as i32;
                sqx0 += v * x;
                sx0 += x;
            }
            let (mut sqx1, mut sx1) = (0i32, 0i32);
            for l in 0..32 {
                let v = (ql[qo + l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 };
                let x = xq[xo + 32 + l] as i32;
                sqx1 += v * x;
                sx1 += x;
            }
            sum += dq
                * (d * sc0 as f32 * sqx0 as f32 - min * m0 as f32 * sx0 as f32
                    + d * sc1 as f32 * sqx1 as f32
                    - min * m1 as f32 * sx1 as f32);
            xo += 64;
            qo += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    sum
}

pub fn extract_q5_k_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 || out.len() != ncols {
        return Err(GgufError::Truncated("q5_k embed row"));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {
        return Err(GgufError::Truncated("q5_k embed OOB"));
    }
    for b in 0..blocks_per_row {
        dequant_q5_k_block(
            &data[start + b * Q5_K_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}

