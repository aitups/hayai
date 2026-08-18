//! GGML Q3_K — QK_K=256, 110 bytes/block.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK_K: usize = 256;
pub const Q3_K_BLOCK_BYTES: usize = 110; // hmask(32)+qs(64)+scales(12)+d(2)

fn unpack_q3_k_scales(scales12: &[u8]) -> [i8; 16] {
    let kmask1 = 0x0303_0303u32;
    let kmask2 = 0x0f0f_0f0fu32;
    let mut aux = [0u32; 4];
    aux[0] = u32::from_le_bytes(scales12[0..4].try_into().unwrap());
    aux[1] = u32::from_le_bytes(scales12[4..8].try_into().unwrap());
    aux[2] = u32::from_le_bytes(scales12[8..12].try_into().unwrap());
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    let mut out = [0i8; 16];
    for (i, a) in aux.iter().enumerate() {
        let b = a.to_le_bytes();
        for j in 0..4 {
            out[i * 4 + j] = b[j] as i8;
        }
    }
    out
}

pub fn dequant_q3_k(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK_K != 0 {
        return Err(GgufError::Truncated("q3_k n not multiple of 256"));
    }
    let blocks = n / QK_K;
    if bytes.len() < blocks * Q3_K_BLOCK_BYTES {
        return Err(GgufError::Truncated("q3_k tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_q3_k_block(&bytes[i * Q3_K_BLOCK_BYTES..], &mut out[i * QK_K..]);
    }
    Ok(out)
}

fn dequant_q3_k_block(block: &[u8], y: &mut [f32]) {
    let hmask = &block[0..32];
    let mut q = &block[32..96];
    let scales = unpack_q3_k_scales(&block[96..108]);
    let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let mut yo = 0usize;
    let mut is = 0usize;
    let mut m: u8 = 1;
    for _ in 0..(QK_K / 128) {
        let mut shift = 0u32;
        for _ in 0..4 {
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let qv = ((q[l] >> shift) & 3) as i8;
                let hm = if hmask[l] & m != 0 { 0 } else { 4 };
                y[yo + l] = dl * (qv - hm) as f32;
            }
            yo += 16;
            let dl = d_all * (scales[is] as i32 - 32) as f32;
            is += 1;
            for l in 0..16 {
                let qv = ((q[l + 16] >> shift) & 3) as i8;
                let hm = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                y[yo + l] = dl * (qv - hm) as f32;
            }
            yo += 16;
            shift += 2;
            m <<= 1;
        }
        q = &q[32..];
    }
}

pub fn gemv_q3_k(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!(
            "Q3_K ncols={ncols} not multiple of {QK_K}"
        )));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q3_K_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q3_k gemv"));
    }
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * Q3_K_BLOCK_BYTES..];
            let hmask = &block[0..32];
            let mut q = &block[32..96];
            let scales = unpack_q3_k_scales(&block[96..108]);
            let d_all = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
            let x_base = b * QK_K;
            let mut xo = 0usize;
            let mut is = 0usize;
            let mut m: u8 = 1;
            for _ in 0..(QK_K / 128) {
                let mut shift = 0u32;
                for _ in 0..4 {
                    let dl = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let qv = ((q[l] >> shift) & 3) as i8;
                        let hm = if hmask[l] & m != 0 { 0 } else { 4 };
                        sum += dl * (qv - hm) as f32 * input[x_base + xo + l];
                    }
                    xo += 16;
                    let dl = d_all * (scales[is] as i32 - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let qv = ((q[l + 16] >> shift) & 3) as i8;
                        let hm = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                        sum += dl * (qv - hm) as f32 * input[x_base + xo + l];
                    }
                    xo += 16;
                    shift += 2;
                    m <<= 1;
                }
                q = &q[32..];
            }
        }
        *out = sum;
    });
    Ok(())
}

pub fn extract_q3_k_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 || out.len() != ncols {
        return Err(GgufError::Truncated("q3_k embed row"));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q3_K_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {
        return Err(GgufError::Truncated("q3_k embed OOB"));
    }
    for b in 0..blocks_per_row {
        dequant_q3_k_block(
            &data[start + b * Q3_K_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}
