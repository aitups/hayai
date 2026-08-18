//! GGML IQ4_NL (block 32) and IQ4_XS (block 256) — non-linear 4-bit quants.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK4_NL: usize = 32;
pub const IQ4_NL_BLOCK_BYTES: usize = 18; // d(2)+qs(16)

pub const QK_K: usize = 256;
pub const IQ4_XS_BLOCK_BYTES: usize = 136; // d(2)+scales_h(2)+scales_l(4)+qs(128)

/// Nonlinear lookup shared by IQ4_NL / IQ4_XS (llama.cpp `kvalues_iq4nl`).
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

pub fn dequant_iq4_nl(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK4_NL != 0 {
        return Err(GgufError::Truncated("iq4_nl n not multiple of 32"));
    }
    let blocks = n / QK4_NL;
    if bytes.len() < blocks * IQ4_NL_BLOCK_BYTES {
        return Err(GgufError::Truncated("iq4_nl tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_iq4_nl_block(
            &bytes[i * IQ4_NL_BLOCK_BYTES..],
            &mut out[i * QK4_NL..(i + 1) * QK4_NL],
        );
    }
    Ok(out)
}

fn dequant_iq4_nl_block(block: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..18];
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
    }
}

pub fn gemv_iq4_nl(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK4_NL != 0 {
        return Err(GgufError::Msg(format!(
            "IQ4_NL ncols={ncols} not multiple of {QK4_NL}"
        )));
    }
    let blocks_per_row = ncols / QK4_NL;
    let row_bytes = blocks_per_row * IQ4_NL_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("iq4_nl gemv"));
    }
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * IQ4_NL_BLOCK_BYTES..];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let qs = &block[2..18];
            let x_base = b * QK4_NL;
            for j in 0..16 {
                sum += d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32 * input[x_base + j];
                sum += d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32 * input[x_base + 16 + j];
            }
        }
        *out = sum;
    });
    Ok(())
}

pub fn extract_iq4_nl_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK4_NL != 0 || out.len() != ncols {
        return Err(GgufError::Truncated("iq4_nl embed row"));
    }
    let blocks_per_row = ncols / QK4_NL;
    let row_bytes = blocks_per_row * IQ4_NL_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {
        return Err(GgufError::Truncated("iq4_nl embed OOB"));
    }
    for b in 0..blocks_per_row {
        dequant_iq4_nl_block(
            &data[start + b * IQ4_NL_BLOCK_BYTES..],
            &mut out[b * QK4_NL..(b + 1) * QK4_NL],
        );
    }
    Ok(())
}

pub fn dequant_iq4_xs(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {
    if n % QK_K != 0 {
        return Err(GgufError::Truncated("iq4_xs n not multiple of 256"));
    }
    let blocks = n / QK_K;
    if bytes.len() < blocks * IQ4_XS_BLOCK_BYTES {
        return Err(GgufError::Truncated("iq4_xs tensor"));
    }
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {
        dequant_iq4_xs_block(
            &bytes[i * IQ4_XS_BLOCK_BYTES..],
            &mut out[i * QK_K..(i + 1) * QK_K],
        );
    }
    Ok(out)
}

fn dequant_iq4_xs_block(block: &[u8], y: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let mut qs = &block[8..136];
    let mut yo = 0usize;
    for ib in 0..8 {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32
            | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            y[yo + j] = dl * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
            y[yo + 16 + j] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
        yo += 32;
        qs = &qs[16..];
    }
}

pub fn gemv_iq4_xs(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 {
        return Err(GgufError::Msg(format!(
            "IQ4_XS ncols={ncols} not multiple of {QK_K}"
        )));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * IQ4_XS_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {
        return Err(GgufError::Truncated("iq4_xs gemv"));
    }
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let block = &data[row_base + b * IQ4_XS_BLOCK_BYTES..];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let mut qs = &block[8..136];
            let x_base = b * QK_K;
            let mut xo = 0usize;
            for ib in 0..8 {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) as i32
                    | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
                let dl = d * (ls - 32) as f32;
                for j in 0..16 {
                    sum += dl
                        * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32
                        * input[x_base + xo + j];
                    sum += dl
                        * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32
                        * input[x_base + xo + 16 + j];
                }
                xo += 32;
                qs = &qs[16..];
            }
        }
        *out = sum;
    });
    Ok(())
}

pub fn extract_iq4_xs_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {
    if ncols % QK_K != 0 || out.len() != ncols {
        return Err(GgufError::Truncated("iq4_xs embed row"));
    }
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * IQ4_XS_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {
        return Err(GgufError::Truncated("iq4_xs embed OOB"));
    }
    for b in 0..blocks_per_row {
        dequant_iq4_xs_block(
            &data[start + b * IQ4_XS_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq4_nl_empty() {
        assert!(dequant_iq4_nl(&[], 0).unwrap().is_empty());
    }

    #[test]
    fn iq4_nl_block_finite() {
        let mut block = vec![0u8; IQ4_NL_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        for j in 0..16 {
            block[2 + j] = 0x88; // mid codes
        }
        let y = dequant_iq4_nl(&block, 32).unwrap();
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
