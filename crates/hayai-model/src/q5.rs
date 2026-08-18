//! GGML Q5_0 / Q5_1 blocks (common inside “Q4_K_M” GGUF mixes).

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK5: usize = 32;
pub const Q5_0_BLOCK_BYTES: usize = 22; // d(2) + qh(4) + qs(16)
pub const Q5_1_BLOCK_BYTES: usize = 24; // d(2)+m(2)+qh(4)+qs(16)

pub fn dequant_q5_0(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK5 != 0 {
        return Err(GgufError::Truncated("q5_0 n"));
    }
    let blocks = n / QK5;
    if bytes.len() < blocks * Q5_0_BLOCK_BYTES {
        return Err(GgufError::Truncated("q5_0 tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_q5_0_block(&bytes[i * Q5_0_BLOCK_BYTES..], &mut out[i * QK5..]);
    }
    Ok(out)
}

fn dequant_q5_0_block(block: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    for j in 0..16 {
        let xh_0 = ((qh >> j) << 4) & 0x10;
        let xh_1 = (qh >> (j + 12)) & 0x10;
        let x0 = ((qs[j] as u32 & 0x0F) | xh_0) as i32 - 16;
        let x1 = ((qs[j] as u32 >> 4) | xh_1) as i32 - 16;
        y[j] = x0 as f32 * d;
        y[j + 16] = x1 as f32 * d;
    }
}

pub fn gemv_q5_0(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK5 != 0 {
        return Err(GgufError::Msg(format!("Q5_0 ncols={ncols}")));
    }
    let blocks_per_row = ncols / QK5;
    let row_bytes = blocks_per_row * Q5_0_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("q5_0 gemv"));
    }
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * Q5_0_BLOCK_BYTES..];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let qs = &block[6..22];
            let x_base = b * QK5;
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((qs[j] as u32 & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((qs[j] as u32 >> 4) | xh_1) as i32 - 16;
                sum += x0 as f32 * d * input[x_base + j];
                sum += x1 as f32 * d * input[x_base + j + 16];
            }
        }
        *out = sum;
    });
    Ok(())
}

pub fn extract_q5_0_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    let blocks = ncols / QK5;
    let row_bytes = blocks * Q5_0_BLOCK_BYTES;
    let start = row * row_bytes;
    for b in 0..blocks {
        dequant_q5_0_block(
            &data[start + b * Q5_0_BLOCK_BYTES..],
            &mut out[b * QK5..(b + 1) * QK5],
        );
    }
    Ok(())
}

pub fn dequant_q5_1(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK5 != 0 {
        return Err(GgufError::Truncated("q5_1 n"));
    }
    let blocks = n / QK5;
    if bytes.len() < blocks * Q5_1_BLOCK_BYTES {
        return Err(GgufError::Truncated("q5_1 tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        let block = &bytes[i * Q5_1_BLOCK_BYTES..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        let y = &mut out[i * QK5..];
        for j in 0..16 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = (qs[j] as u32 & 0x0F) | xh_0;
            let x1 = (qs[j] as u32 >> 4) | xh_1;
            y[j] = x0 as f32 * d + m;
            y[j + 16] = x1 as f32 * d + m;
        }
    }
    Ok(out)
}

pub fn gemv_q5_1(
    _nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK5 != 0 {
        return Err(GgufError::Msg(format!("Q5_1 ncols={ncols}")));
    }
    let blocks_per_row = ncols / QK5;
    let row_bytes = blocks_per_row * Q5_1_BLOCK_BYTES;
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * Q5_1_BLOCK_BYTES..];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
            let qs = &block[8..24];
            let x_base = b * QK5;
            for j in 0..16 {
                let xh_0 = ((qh >> j) << 4) & 0x10;
                let xh_1 = (qh >> (j + 12)) & 0x10;
                let x0 = (qs[j] as u32 & 0x0F) | xh_0;
                let x1 = (qs[j] as u32 >> 4) | xh_1;
                sum += (x0 as f32 * d + m) * input[x_base + j];
                sum += (x1 as f32 * d + m) * input[x_base + j + 16];
            }
        }
        *out = sum;
    });
    Ok(())
}
