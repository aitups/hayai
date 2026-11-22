//! GGML Q4_K (used by Q4_K_S / Q4_K_M GGUF files) — block size QK_K=256, 144 bytes/block.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK_K: usize = 256;
pub const Q4_K_BLOCK_BYTES: usize = 144; // d(2)+dmin(2)+scales(12)+qs(128)

#[inline]
fn gemv_q4k_sub32(d: f32, minv: f32, q: &[u8], x: &[f32], high_nibble: bool) -> f32 {
    use std::simd::num::{SimdFloat, SimdUint};
    use std::simd::{f32x8, u32x8, u8x8};
    let d_v = f32x8::splat(d);
    let m_v = f32x8::splat(minv);
    let shift = u8x8::splat(4);
    let mask = u8x8::splat(0x0F);
    let magic = u32x8::splat(0x4B00_0000);
    let magic_f = f32x8::splat(8_388_608.0);
    // Four independent per-chunk products (no serial accumulator chain) to hide the f32
    // multiply/add latency (audit C.2).
    let mut acc = [f32x8::splat(0.0); 4];
    for (chunk, a) in acc.iter_mut().enumerate() {
        let o = chunk * 8;
        let bytes = u8x8::from_slice(&q[o..o + 8]);
        let nib = if high_nibble { bytes >> shift } else { bytes & mask };
        // u8 nibble -> f32 via the "magic number" int-to-float bit trick, all SIMD.
        let f = f32x8::from_bits(nib.cast::<u32>() | magic) - magic_f;
        *a = (f * d_v - m_v) * f32x8::from_slice(&x[o..o + 8]);
    }
    (acc[0] + acc[1] + acc[2] + acc[3]).reduce_sum()
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

/// Convierte `f32` a bits `f16` (IEEE 754 half).
pub fn f32_to_f16(value: f32) -> u16 {
    let f = value.to_bits();
    let sign = (f >> 16) & 0x8000;
    let mut exponent = ((f >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = f & 0x7fffff;
    if exponent <= 0 {
        if exponent < -10 {
            return sign as u16;
        }
        // subnormal f16
        let m = mantissa | 0x800000;
        let shift = (14 - exponent) as u32;
        let mut m = m >> shift;
        if mantissa & (1 << (shift.saturating_sub(1))) != 0 {
            m += 1;
        }
        return (sign as u16) | m as u16;
    }
    if exponent >= 31 {
        return (sign as u16) | 0x7c00; // inf/overflow
    }
    let mut m = mantissa >> 13;
    if mantissa & 0x1000 != 0 {
        m += 1;
    }
    if m & 0x400 == 0x400 {
        m = 0;
        exponent += 1;
    }
    if exponent >= 31 {
        return (sign as u16) | 0x7c00;
    }
    (sign as u16) | ((exponent as u16) << 10) | m as u16
}

/// Cuantiza un bloque contiguo de `n` f32 a Q4_K (n % 256 == 0). Espejo de la
/// referencia `quantize_row_q4_K_reference` de ggml: por sub-bloque de 32 se
/// deriva `scale=(max-min)/15` y `min`; el bloque empaqueta `d=max_scale/63` y
/// `dmin=max_min/63` (f16) más escalas de 6 bits y nibbles de 4 bits.
pub fn quantize_q4_k(x: &[f32], n: usize) -> Result<Vec<u8>, GgufError> {
    if n % QK_K != 0 {
        return Err(GgufError::Truncated("q4_k n not multiple of 256"));
    }
    let blocks = n / QK_K;
    let mut out = vec![0u8; blocks * Q4_K_BLOCK_BYTES];
    for b in 0..blocks {
        quantize_q4_k_block(&x[b * QK_K..b * QK_K + QK_K], &mut out[b * Q4_K_BLOCK_BYTES..]);
    }
    Ok(out)
}

fn quantize_q4_k_block(x: &[f32], out: &mut [u8]) {
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    let mut l = [0u8; QK_K];
    for j in 0..8 {
        let sub = &x[j * 32..j * 32 + 32];
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in sub {
            mn = mn.min(v);
            mx = mx.max(v);
        }
        scales[j] = (mx - mn) / 15.0;
        mins[j] = mn;
    }
    let max_scale = scales.iter().cloned().fold(0.0f32, f32::max).max(1e-30);
    let max_min = mins.iter().map(|m| m.abs()).fold(0.0f32, f32::max).max(1e-30);
    let d = max_scale / 63.0;
    let dmin = max_min / 63.0;
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..8 {
        ls[j] = (scales[j] * 63.0 / max_scale).round().clamp(0.0, 63.0) as u8;
        // El dequant resta `dmin*m`; para que `-dmin*m ≈ mins[j]` se usa el
        // offset positivo: `m = -mins[j] / dmin`.
        lm[j] = (-mins[j] * 63.0 / max_min).round().clamp(0.0, 63.0) as u8;
    }
    for j in 0..8 {
        let d1 = d * ls[j] as f32;
        if d1 == 0.0 {
            for ii in 0..32 {
                l[j * 32 + ii] = 0;
            }
            continue;
        }
        let dm = dmin * lm[j] as f32;
        for ii in 0..32 {
            let q = ((x[j * 32 + ii] + dm) / d1).round().clamp(0.0, 15.0) as u8;
            l[j * 32 + ii] = q;
        }
    }
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
    let mut scales_pack = [0u8; 12];
    for j in 0..8 {
        let sc = ls[j];
        let m = lm[j];
        if j < 4 {
            scales_pack[j] = sc;
            scales_pack[j + 4] = m;
        } else {
            scales_pack[j + 4] = (sc & 0xF) | ((m & 0xF) << 4);
            scales_pack[j - 4] |= (sc >> 4) << 6;
            scales_pack[j] |= (m >> 4) << 6;
        }
    }
    out[4..16].copy_from_slice(&scales_pack);
    for j in (0..QK_K).step_by(64) {
        let base = j / 64 * 32;
        for ii in 0..32 {
            out[16 + base + ii] = l[j + ii] | (l[j + ii + 32] << 4);
        }
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

    let use_avx2 = crate::simd_dispatch::x86_avx2_fma();
    output.par_iter_mut().with_min_len(64).enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        *out = gemv_q4k_dispatch(use_avx2, data, input, row_base, blocks_per_row);
    });
    Ok(())
}

#[inline(always)]
fn gemv_q4k_row(data: &[u8], input: &[f32], row_base: usize, blocks_per_row: usize) -> f32 {
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
    sum
}

/// AVX2/FMA/F16C clone of [`gemv_q4k_row`]. The `#[inline(always)]` body is inlined
/// here so its `f32x8` ops use 256-bit AVX registers (2× the SSE2 baseline) instead of
/// being compiled at the base target. Bit-exact (same per-lane f32 ops; Rust does not
/// contract to FMA by default).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemv_q4k_row_avx2(
    data: &[u8],
    input: &[f32],
    row_base: usize,
    blocks_per_row: usize,
) -> f32 {
    gemv_q4k_row(data, input, row_base, blocks_per_row)
}

#[inline]
fn gemv_q4k_dispatch(
    use_avx2: bool,
    data: &[u8],
    input: &[f32],
    row_base: usize,
    blocks_per_row: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if use_avx2 {
            return unsafe { gemv_q4k_row_avx2(data, input, row_base, blocks_per_row) };
        }
    }
    let _ = use_avx2;
    gemv_q4k_row(data, input, row_base, blocks_per_row)
}

/// Q4_K × Q8_K GEMV (audit C.2): quantise the activation to Q8_K once (256-element blocks,
/// one f32 scale each, matching llama.cpp), then run the existing f32 dot on the
/// pre-scaled int8 activations. The values are small exact ints, so the f32 dot is
/// bit-identical to an integer dot; the dot keeps four independent accumulators for ILP
/// (the audit allows `i32x8 / f32x8` accs).
pub fn gemv_q4_k_q8k(
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
    // Real integer dot (audit C.2/I1): quantise the activation once and accumulate
    // `sum(q_i * xq_i)` / `sum(xq_i)` per 32-sub-block in i32, applying the block
    // scale/min once. Consumes `Q8K.qs`; no dequantised f32 copy.
    let q8 = crate::q8k::Q8K::quantize(input);
    for (row, out) in output.iter_mut().enumerate() {
        *out = gemv_q4k_row_q8k(data, &q8, row, blocks_per_row);
    }
    Ok(())
}

/// Integer Q4_K row dot against a Q8_K activation for one row.
#[inline]
fn gemv_q4k_row_q8k(data: &[u8], q8: &crate::q8k::Q8K, row: usize, blocks: usize) -> f32 {
    let row_base = row * blocks * Q4_K_BLOCK_BYTES;
    let mut sum = 0.0f32;
    for b in 0..blocks {
        let block = &data[row_base + b * Q4_K_BLOCK_BYTES..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let dq = q8.d[b];
        let xq = q8.qs_block(b);
        let mut is = 0usize;
        let mut xo = 0usize;
        let mut qo = 0usize;
        for _ in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);
            let (sqx0, sx0) = int_dot_nibbles_low(&qs[qo..qo + 32], &xq[xo..xo + 32]);
            sum += dq * (d * sc0 as f32 * sqx0 as f32 - min * m0 as f32 * sx0 as f32);
            let (sqx1, sx1) = int_dot_nibbles_high(&qs[qo..qo + 32], &xq[xo + 32..xo + 64]);
            sum += dq * (d * sc1 as f32 * sqx1 as f32 - min * m1 as f32 * sx1 as f32);
            xo += 64;
            qo += 32;
            is += 2;
        }
    }
    sum
}

#[inline]
fn int_dot_nibbles_low(bytes: &[u8], xq: &[i8]) -> (i32, i32) {
    let mut sqx = 0i32;
    let mut sx = 0i32;
    for j in 0..32 {
        let q = (bytes[j] & 0x0F) as i32;
        let x = xq[j] as i32;
        sqx += q * x;
        sx += x;
    }
    (sqx, sx)
}

#[inline]
fn int_dot_nibbles_high(bytes: &[u8], xq: &[i8]) -> (i32, i32) {
    let mut sqx = 0i32;
    let mut sx = 0i32;
    for j in 0..32 {
        let q = (bytes[j] >> 4) as i32;
        let x = xq[j] as i32;
        sqx += q * x;
        sx += x;
    }
    (sqx, sx)
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

    #[test]
    fn f16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 123.45, 0.0001, -9876.5, 1.2345] {
            let h = f32_to_f16(v);
            let back = crate::gguf::f16_to_f32(h);
            assert!(
                (v - back).abs() < 0.01 * v.abs().max(1e-3),
                "{v} -> {back}"
            );
        }
    }

    #[test]
    fn q4k_quantize_roundtrip_error_small() {
        let mut x = vec![0.0f32; 1024];
        for (i, v) in x.iter_mut().enumerate() {
            let b = i / 256;
            let j = i % 256;
            *v = match b {
                0 => (j as f32 - 128.0) * 0.01,
                1 => ((j % 17) as f32) * 0.5 - 4.0,
                2 => (j as f32).sin() * 3.0,
                _ => {
                    if j % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
            };
        }
        let packed = quantize_q4_k(&x, x.len()).unwrap();
        let back = dequant_q4_k(&packed, x.len()).unwrap();
        // Error Q4_K es absoluto (~media unidad de paso); se mide relativo al
        // rango del bloque, no relativo al valor (que explota cerca de 0).
        let mut max_abs = 0.0f32;
        let xmax = x.iter().cloned().fold(0.0f32, f32::max);
        let xmin = x.iter().cloned().fold(0.0f32, f32::min);
        for (a, b) in x.iter().zip(back.iter()) {
            max_abs = max_abs.max((a - b).abs());
        }
        let range_rel = max_abs / (xmax - xmin).max(1e-6);
        assert!(range_rel < 0.2, "range_rel_err={range_rel} max_abs={max_abs}");
    }

    #[test]
    fn q8k_gemv_matches_f32_within_quant_error() {
        // The Q8_K path quantises the activation (int8) — compare it to the exact f32 dot
        // on one Q4_K row (4 blocks) with a non-trivial activation pattern.
        let ncols = 1024usize;
        let mut w = vec![0.0f32; ncols];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (((i * 37) % 211) as f32 - 105.0) * 0.05;
        }
        let packed = quantize_q4_k(&w, w.len()).unwrap();
        let mut input = vec![0.0f32; ncols];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (((i * 53) % 97) as f32 - 48.0) * 0.1;
        }
        let mut out_f32 = vec![0.0f32; 1];
        let mut out_q8 = vec![0.0f32; 1];
        gemv_q4_k(1, ncols, &packed, &input, &mut out_f32).unwrap();
        gemv_q4_k_q8k(1, ncols, &packed, &input, &mut out_q8).unwrap();
        let err = (out_f32[0] - out_q8[0]).abs() / out_f32[0].abs().max(1e-3);
        assert!(
            err < 0.05,
            "f32={} q8={} rel_err={err}",
            out_f32[0],
            out_q8[0]
        );
    }
}


