//! GGML Q2_K — QK_K=256, 84 bytes/block.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK_K: usize = 256;
pub const Q2_K_BLOCK_BYTES: usize = 84; // scales(16)+qs(64)+d(2)+dmin(2)

pub fn dequant_q2_k(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK_K != 0 {
        return Err(GgufError::Truncated("q2_k n not multiple of 256"));
    }
    let blocks = n / QK_K;
    if bytes.len() < blocks * Q2_K_BLOCK_BYTES {
        return Err(GgufError::Truncated("q2_k tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_q2_k_block(&bytes[i * Q2_K_BLOCK_BYTES..], &mut out[i * QK_K..]);
    }
    Ok(out)
}

fn dequant_q2_k_block(block: &[u8], y: &mut [f32]) {
    let scales = &block[0..16];
    let mut q = &block[16..80];
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let min = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    let mut yo = 0usize;
    let mut is = 0usize;
    for _ in 0..(QK_K / 128) {
        let mut shift = 0u32;
        for _ in 0..4 {
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                y[yo + l] = dl * ((q[l] >> shift) & 3) as f32 - ml;
            }
            yo += 16;
            let sc = scales[is];
            is += 1;
            let dl = d * (sc & 0xF) as f32;
            let ml = min * (sc >> 4) as f32;
            for l in 0..16 {
                y[yo + l] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
            }
            yo += 16;
            shift += 2;
        }
        q = &q[32..];
    }
}

pub fn gemv_q2_k(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!(
            "Q2_K ncols={ncols} not multiple of {QK_K}"
        )));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q2_K_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q2_k gemv"));
    }
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * Q2_K_BLOCK_BYTES..];
            let scales = &block[0..16];
            let mut q = &block[16..80];
            let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
            let min = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
            let x_base = b * QK_K;
            let mut xo = 0usize;
            let mut is = 0usize;
            for _ in 0..(QK_K / 128) {
                let mut shift = 0u32;
                for _ in 0..4 {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for l in 0..16 {
                        let w = dl * ((q[l] >> shift) & 3) as f32 - ml;
                        sum += w * input[x_base + xo + l];
                    }
                    xo += 16;
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for l in 0..16 {
                        let w = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
                        sum += w * input[x_base + xo + l];
                    }
                    xo += 16;
                    shift += 2;
                }
                q = &q[32..];
            }
        }
        *out = sum;
    });
    Ok(())
}

pub fn extract_q2_k_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 || out.len() != ncols {
        return Err(GgufError::Truncated("q2_k embed row"));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q2_K_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {
        return Err(GgufError::Truncated("q2_k embed OOB"));
    }
    for b in 0..blocks_per_row {
        dequant_q2_k_block(
            &data[start + b * Q2_K_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}
