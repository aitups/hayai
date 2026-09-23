//! Packed GGUF tensors + GGML-compatible GEMV.
//!
//! Production streaming uses `QuantStorage::External` views over scratch slots;
//! `QuantStorage::Mapped` is the **dev/test** full-file mmap path only.

use crate::gguf::f16_to_f32;
use crate::gguf::GgufFile;
use crate::gguf_types::{GgmlType, GgufError, TensorInfo};
use crate::iq2::{
    extract_iq2_s_row, extract_iq2_xs_row, extract_iq2_xxs_row, gemv_iq2_s, gemv_iq2_xs,
    gemv_iq2_xxs,
};
use crate::iq3::{
    extract_iq3_s_row, extract_iq3_xxs_row, gemv_iq3_s, gemv_iq3_xxs,
};
use crate::iq4::{
    extract_iq4_nl_row, extract_iq4_xs_row, gemv_iq4_nl, gemv_iq4_xs,
};
use crate::q2k::{extract_q2_k_row, gemv_q2_k};
use crate::q3k::{extract_q3_k_row, gemv_q3_k};
use crate::q4k::{extract_q4_k_row, gemv_q4_k};
use crate::q5::{extract_q5_0_row, gemv_q5_0, gemv_q5_1};
use crate::q5k::{extract_q5_k_row, gemv_q5_k};
use crate::q6k::{extract_q6_k_row, gemv_q6_k};
use rayon::prelude::*;
use std::sync::Arc;

/// Matrix in GGUF/GGML layout: `ne0` contiguous, shape `[ncols=ne0, nrows=ne1]`.
/// GEMV: `y[nrows] = W @ x[ncols]`.
#[derive(Clone)]
pub struct QuantMatrix {
    pub name: String,
    pub ncols: usize,
    pub nrows: usize,
    pub ggml_type: GgmlType,
    storage: QuantStorage,
}

#[derive(Clone)]
enum QuantStorage {
    /// Owned blob (tests / small tensors / CPU-only fallback).
    Owned(Vec<u8>),
    /// Zero-copy slice into a shared GGUF mmap.
    Mapped {
        file: Arc<GgufFile>,
        start: usize,
        len: usize,
    },
    /// Borrowed bytes (StreamingScratch / prefetch blob). Caller keeps buffer alive.
    External {
        ptr: *const u8,
        len: usize,
    },
}

// External pointers are only cloned as shallow views into caller-owned scratch.
unsafe impl Send for QuantStorage {}
unsafe impl Sync for QuantStorage {}

impl QuantMatrix {
    pub fn from_gguf(gguf: Arc<GgufFile>, name: &str) -> Result<Self, GgufError> {
        let info = gguf.tensor(name)?.clone();
        Self::mapped(gguf, &info)
    }

    pub fn mapped(gguf: Arc<GgufFile>, info: &TensorInfo) -> Result<Self, GgufError> {
        let (start, len) = gguf.tensor_range(info)?;
        Ok(Self {
            name: info.name.clone(),
            ncols: info.ncols(),
            nrows: info.nrows(),
            ggml_type: info.ggml_type,
            storage: QuantStorage::Mapped {
                file: gguf,
                start,
                len,
            },
        })
    }

    pub fn owned(
        name: impl Into<String>,
        ncols: usize,
        nrows: usize,
        ggml_type: GgmlType,
        data: Vec<u8>,
    ) -> Self {
        Self {
            name: name.into(),
            ncols,
            nrows,
            ggml_type,
            storage: QuantStorage::Owned(data),
        }
    }

    /// View into `data` without copying (Attn/FFN over StreamingScratch).
    ///
    /// The returned matrix must not outlive `data`.
    pub fn view(
        name: impl Into<String>,
        ncols: usize,
        nrows: usize,
        ggml_type: GgmlType,
        data: &[u8],
    ) -> Self {
        Self {
            name: name.into(),
            ncols,
            nrows,
            ggml_type,
            storage: QuantStorage::External {
                ptr: data.as_ptr(),
                len: data.len(),
            },
        }
    }

    /// Bytes Q4 crudos del tensor (copia) — SpMM esparso en GPU con dequant
    /// en-kernel (Fase 2, C1): evita materializar el F32 (23 GB/gen en 27B).
    pub fn raw_bytes(&self) -> Vec<u8> {
        match &self.storage {
            QuantStorage::Owned(d) => d.clone(),
            QuantStorage::Mapped { file, start, len } => {
                let m = file.mmap_bytes();
                m[*start..*start + *len].to_vec()
            }
            QuantStorage::External { ptr, len } => {
                let bytes = unsafe { std::slice::from_raw_parts(*ptr as *const u8, *len) };
                bytes.to_vec()
            }
        }
    }

    /// Metadata-only shell (nbytes known, no payload) — FFN GPU path uses scratch offsets.
    pub fn meta_only(
        name: impl Into<String>,
        ncols: usize,
        nrows: usize,
        ggml_type: GgmlType,
        nbytes: usize,
    ) -> Self {
        Self {
            name: name.into(),
            ncols,
            nrows,
            ggml_type,
            storage: QuantStorage::External {
                ptr: std::ptr::null(),
                len: nbytes,
            },
        }
    }

    pub fn data(&self) -> &[u8] {
        match &self.storage {
            QuantStorage::Owned(v) => v,
            QuantStorage::Mapped { file, start, len } => &file.mmap_bytes()[*start..*start + *len],
            QuantStorage::External { ptr, len } => {
                if ptr.is_null() || *len == 0 {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts(*ptr, *len) }
                }
            }
        }
    }

    pub fn nbytes(&self) -> usize {
        match &self.storage {
            QuantStorage::Owned(v) => v.len(),
            QuantStorage::Mapped { len, .. } => *len,
            QuantStorage::External { len, .. } => *len,
        }
    }

    pub fn is_mapped(&self) -> bool {
        matches!(self.storage, QuantStorage::Mapped { .. })
    }

    pub fn is_view(&self) -> bool {
        matches!(self.storage, QuantStorage::External { .. })
    }

    /// Copy one embedding row (`token` in `0..nrows`) into `out[ncols]`.
    pub fn embed_row(&self, token: u32, out: &mut [f32]) -> Result<(), GgufError> {
        let row = token as usize;
        if row >= self.nrows || out.len() != self.ncols {
            return Err(GgufError::Truncated("embed row OOB"));
        }
        let data = self.data();
        match self.ggml_type {
            GgmlType::F32 => {
                let start = row * self.ncols * 4;
                for i in 0..self.ncols {
                    let o = start + i * 4;
                    out[i] = f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
                }
            }
            GgmlType::F16 => {
                let start = row * self.ncols * 2;
                for i in 0..self.ncols {
                    let o = start + i * 2;
                    let h = u16::from_le_bytes(data[o..o + 2].try_into().unwrap());
                    out[i] = f16_to_f32(h);
                }
            }
            GgmlType::BF16 => {
                let start = row * self.ncols * 2;
                for i in 0..self.ncols {
                    let o = start + i * 2;
                    let bits = (u16::from_le_bytes(data[o..o + 2].try_into().unwrap()) as u32) << 16;
                    out[i] = f32::from_bits(bits);
                }
            }
            GgmlType::F64 => {
                let start = row * self.ncols * 8;
                for i in 0..self.ncols {
                    let o = start + i * 8;
                    out[i] = f64::from_le_bytes(data[o..o + 8].try_into().unwrap()) as f32;
                }
            }
            GgmlType::I8 => {
                let start = row * self.ncols;
                for i in 0..self.ncols {
                    out[i] = data[start + i] as i8 as f32;
                }
            }
            GgmlType::I16 => {
                let start = row * self.ncols * 2;
                for i in 0..self.ncols {
                    let o = start + i * 2;
                    out[i] = i16::from_le_bytes(data[o..o + 2].try_into().unwrap()) as f32;
                }
            }
            GgmlType::I32 => {
                let start = row * self.ncols * 4;
                for i in 0..self.ncols {
                    let o = start + i * 4;
                    out[i] = i32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as f32;
                }
            }
            GgmlType::I64 => {
                let start = row * self.ncols * 8;
                for i in 0..self.ncols {
                    let o = start + i * 8;
                    out[i] = i64::from_le_bytes(data[o..o + 8].try_into().unwrap()) as f32;
                }
            }
            GgmlType::Q8_1 => extract_q8_1_row(data, self.ncols, row, out)?,
            GgmlType::Q8_K => extract_q8_k_row(data, self.ncols, row, out)?,
            GgmlType::Q4_0 => extract_q4_0_row(data, self.ncols, row, out)?,
            GgmlType::Q4_1 => extract_q4_1_row(data, self.ncols, row, out)?,
            GgmlType::Q8_0 => extract_q8_0_row(data, self.ncols, row, out)?,
            GgmlType::Q2_K => extract_q2_k_row(data, self.ncols, row, out)?,
            GgmlType::Q3_K => extract_q3_k_row(data, self.ncols, row, out)?,
            GgmlType::Q4_K => extract_q4_k_row(data, self.ncols, row, out)?,
            GgmlType::Q5_0 => extract_q5_0_row(data, self.ncols, row, out)?,
            GgmlType::Q5_K => extract_q5_k_row(data, self.ncols, row, out)?,
            GgmlType::Q6_K => extract_q6_k_row(data, self.ncols, row, out)?,
            GgmlType::IQ4_NL => extract_iq4_nl_row(data, self.ncols, row, out)?,
            GgmlType::IQ4_XS => extract_iq4_xs_row(data, self.ncols, row, out)?,
            GgmlType::IQ3_XXS => extract_iq3_xxs_row(data, self.ncols, row, out)?,
            GgmlType::IQ3_S => extract_iq3_s_row(data, self.ncols, row, out)?,
            GgmlType::IQ2_XXS => extract_iq2_xxs_row(data, self.ncols, row, out)?,
            GgmlType::IQ2_XS => extract_iq2_xs_row(data, self.ncols, row, out)?,
            GgmlType::IQ2_S => extract_iq2_s_row(data, self.ncols, row, out)?,
            other => return Err(GgufError::UnsupportedType(other, self.name.clone())),
        }
        Ok(())
    }

    /// Extract a single row by index into `out[ncols]`.
    pub fn extract_row(&self, row: usize, out: &mut [f32]) -> Result<(), GgufError> {
        self.embed_row(row as u32, out)
    }

    pub fn gemv(&self, input: &[f32], output: &mut [f32]) -> Result<(), GgufError> {
        assert_eq!(input.len(), self.ncols);
        assert_eq!(output.len(), self.nrows);
        let data = self.data();
        match self.ggml_type {
            GgmlType::F32 => {
                gemv_f32(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::F16 => {
                gemv_f16(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::BF16 => {
                gemv_bf16(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::F64 => {
                gemv_f64(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::I8 => {
                gemv_i8(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::I16 => {
                gemv_i16(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::I32 => {
                gemv_i32(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::I64 => {
                gemv_i64(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::Q8_1 => {
                gemv_q8_1(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::Q8_K => gemv_q8_k(self.nrows, self.ncols, data, input, output),
            GgmlType::Q4_0 => {
                gemv_q4_0(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::Q4_1 => {
                gemv_q4_1(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::Q8_0 => {
                gemv_q8_0(self.ncols, data, input, output);
                Ok(())
            }
            GgmlType::Q2_K => gemv_q2_k(self.nrows, self.ncols, data, input, output),
            GgmlType::Q3_K => gemv_q3_k(self.nrows, self.ncols, data, input, output),
            GgmlType::Q4_K => gemv_q4_k(self.nrows, self.ncols, data, input, output),
            GgmlType::Q5_0 => gemv_q5_0(self.nrows, self.ncols, data, input, output),
            GgmlType::Q5_1 => gemv_q5_1(self.nrows, self.ncols, data, input, output),
            GgmlType::Q5_K => gemv_q5_k(self.nrows, self.ncols, data, input, output),
            GgmlType::Q6_K => gemv_q6_k(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ4_NL => gemv_iq4_nl(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ4_XS => gemv_iq4_xs(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ3_XXS => gemv_iq3_xxs(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ3_S => gemv_iq3_s(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ2_XXS => gemv_iq2_xxs(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ2_XS => gemv_iq2_xs(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ2_S => gemv_iq2_s(self.nrows, self.ncols, data, input, output),
            GgmlType::IQ1_S | GgmlType::IQ1_M | GgmlType::TQ1_0 | GgmlType::TQ2_0 => {
                Err(GgufError::Msg(format!(
                    "unsupported quant {:?} on '{}' — IQ1/TQ GEMV not implemented (hard fail, no silent fallback)",
                    self.ggml_type, self.name
                )))
            }
            other => Err(GgufError::UnsupportedType(other, self.name.clone())),
        }
    }

}

fn gemv_f32(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols * 4;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            let o = base + col * 4;
            let w = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
            sum += w * input[col];
        }
        *out = sum;
    });
}

fn gemv_f16(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols * 2;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            let o = base + col * 2;
            let w = f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]]));
            sum += w * input[col];
        }
        *out = sum;
    });
}

fn gemv_bf16(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols * 2;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            let o = base + col * 2;
            let bits = (u16::from_le_bytes([data[o], data[o + 1]]) as u32) << 16;
            let w = f32::from_bits(bits);
            sum += w * input[col];
        }
        *out = sum;
    });
}

fn gemv_f64(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols * 8;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            let o = base + col * 8;
            let w = f64::from_le_bytes(data[o..o + 8].try_into().unwrap()) as f32;
            sum += w * input[col];
        }
        *out = sum;
    });
}

fn gemv_i8(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            sum += (data[base + col] as i8) as f32 * input[col];
        }
        *out = sum;
    });
}

fn gemv_i16(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols * 2;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            let o = base + col * 2;
            sum += i16::from_le_bytes(data[o..o + 2].try_into().unwrap()) as f32 * input[col];
        }
        *out = sum;
    });
}

fn gemv_i32(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols * 4;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            let o = base + col * 4;
            sum += i32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as f32 * input[col];
        }
        *out = sum;
    });
}

fn gemv_i64(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let base = row * ncols * 8;
        let mut sum = 0.0f32;
        for col in 0..ncols {
            let o = base + col * 8;
            sum += i64::from_le_bytes(data[o..o + 8].try_into().unwrap()) as f32 * input[col];
        }
        *out = sum;
    });
}

fn gemv_q8_1(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    assert_eq!(ncols % 32, 0);
    let blocks_per_row = ncols / 32;
    let row_bytes = blocks_per_row * 36;
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let base = row_base + b * 36;
            let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let x_base = b * 32;
            for j in 0..32 {
                sum += (data[base + 4 + j] as i8) as f32 * d * input[x_base + j];
            }
        }
        *out = sum;
    });
}

fn gemv_q8_k(nrows: usize, ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) -> Result<(), GgufError> {
    if ncols % 256 != 0 {
        return Err(GgufError::Msg(format!("q8_k GEMV requires ncols%256==0, got {ncols}")));
    }
    let blocks_per_row = ncols / 256;
    let row_bytes = blocks_per_row * 292;
    let need = nrows * row_bytes;
    if data.len() < need {
        return Err(GgufError::Truncated("q8_k weights"));
    }
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let base = row_base + b * 292;
            let d = f32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]]);
            let x_base = b * 256;
            for j in 0..256 {
                sum += (data[base + 4 + j] as i8) as f32 * d * input[x_base + j];
            }
        }
        *out = sum;
    });
    Ok(())
}

fn extract_q8_1_row(data: &[u8], ncols: usize, row: usize, out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = ncols / 32;
    let row_bytes_n = blocks * 36;
    let base = row * row_bytes_n;
    if data.len() < base + row_bytes_n {
        return Err(GgufError::Truncated("q8_1 row"));
    }
    for b in 0..blocks {
        let bb = base + b * 36;
        let d = f16_to_f32(u16::from_le_bytes([data[bb], data[bb + 1]]));
        for j in 0..32 {
            out[b * 32 + j] = (data[bb + 4 + j] as i8) as f32 * d;
        }
    }
    Ok(())
}

fn extract_q8_k_row(data: &[u8], ncols: usize, row: usize, out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = ncols / 256;
    let row_bytes_n = blocks * 292;
    let base = row * row_bytes_n;
    if data.len() < base + row_bytes_n {
        return Err(GgufError::Truncated("q8_k row"));
    }
    for b in 0..blocks {
        let bb = base + b * 292;
        let d = f32::from_le_bytes([data[bb], data[bb + 1], data[bb + 2], data[bb + 3]]);
        for j in 0..256 {
            out[b * 256 + j] = (data[bb + 4 + j] as i8) as f32 * d;
        }
    }
    Ok(())
}

/// Q4_0 GEMV with `f32x8` MAC on each half-block (Attn hot path, PRD §2 SIMD).
pub fn gemv_q4_0(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    use std::simd::f32x8;
    use std::simd::num::SimdFloat;

    assert_eq!(ncols % 32, 0);
    let blocks_per_row = ncols / 32;
    let row_bytes = blocks_per_row * 18;
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let base = row_base + b * 18;
            let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let qs = &data[base + 2..base + 18];
            let x_base = b * 32;
            let d_v = f32x8::splat(d);
            // Low nibbles × input[0..16]
            let mut q_lo = [0.0f32; 16];
            let mut q_hi = [0.0f32; 16];
            for j in 0..16 {
                q_lo[j] = ((qs[j] & 0x0F) as i8 - 8) as f32;
                q_hi[j] = ((qs[j] >> 4) as i8 - 8) as f32;
            }
            let mut acc = f32x8::splat(0.0);
            for chunk in 0..2 {
                let o = chunk * 8;
                let ql = f32x8::from_slice(&q_lo[o..o + 8]);
                let qh = f32x8::from_slice(&q_hi[o..o + 8]);
                let xl = f32x8::from_slice(&input[x_base + o..x_base + o + 8]);
                let xh = f32x8::from_slice(&input[x_base + 16 + o..x_base + 24 + o]);
                acc += ql * d_v * xl + qh * d_v * xh;
            }
            sum += acc.reduce_sum();
        }
        *out = sum;
    });
}

pub fn gemv_q4_1(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    assert_eq!(ncols % 32, 0);
    let blocks_per_row = ncols / 32;
    let row_bytes = blocks_per_row * 20;
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let base = row_base + b * 20;
            let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let m = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
            let qs = &data[base + 4..base + 20];
            let x_base = b * 32;
            for j in 0..16 {
                sum += ((qs[j] & 0x0F) as f32 * d + m) * input[x_base + j];
                sum += ((qs[j] >> 4) as f32 * d + m) * input[x_base + j + 16];
            }
        }
        *out = sum;
    });
}

pub fn gemv_q8_0(ncols: usize, data: &[u8], input: &[f32], output: &mut [f32]) {
    assert_eq!(ncols % 32, 0);
    let blocks_per_row = ncols / 32;
    let row_bytes = blocks_per_row * 34;
    output.par_iter_mut().enumerate().for_each(|(row, out)| {
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        for b in 0..blocks_per_row {
            let base = row_base + b * 34;
            let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
            let x_base = b * 32;
            for j in 0..32 {
                sum += (data[base + 2 + j] as i8) as f32 * d * input[x_base + j];
            }
        }
        *out = sum;
    });
}

fn extract_q4_0_row(data: &[u8], ncols: usize, row: usize, out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = ncols / 32;
    let row_bytes_n = blocks * 18;
    let base = row * row_bytes_n;
    let data = &data[base..base + row_bytes_n];
    for b in 0..blocks {
        let bb = b * 18;
        let d = f16_to_f32(u16::from_le_bytes([data[bb], data[bb + 1]]));
        let qs = &data[bb + 2..bb + 18];
        for j in 0..16 {
            out[b * 32 + j] = ((qs[j] & 0x0F) as i8 - 8) as f32 * d;
            out[b * 32 + j + 16] = ((qs[j] >> 4) as i8 - 8) as f32 * d;
        }
    }
    Ok(())
}

fn extract_q4_1_row(data: &[u8], ncols: usize, row: usize, out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = ncols / 32;
    let row_bytes_n = blocks * 20;
    let base = row * row_bytes_n;
    let data = &data[base..base + row_bytes_n];
    for b in 0..blocks {
        let bb = b * 20;
        let d = f16_to_f32(u16::from_le_bytes([data[bb], data[bb + 1]]));
        let minv = f16_to_f32(u16::from_le_bytes([data[bb + 2], data[bb + 3]]));
        let qs = &data[bb + 4..bb + 20];
        for j in 0..16 {
            out[b * 32 + j] = (qs[j] & 0x0F) as f32 * d + minv;
            out[b * 32 + j + 16] = (qs[j] >> 4) as f32 * d + minv;
        }
    }
    Ok(())
}

fn extract_q8_0_row(data: &[u8], ncols: usize, row: usize, out: &mut [f32]) -> Result<(), GgufError> {
    let blocks = ncols / 32;
    let row_bytes_n = blocks * 34;
    let base = row * row_bytes_n;
    let data = &data[base..base + row_bytes_n];
    for b in 0..blocks {
        let bb = b * 34;
        let d = f16_to_f32(u16::from_le_bytes([data[bb], data[bb + 1]]));
        for j in 0..32 {
            out[b * 32 + j] = (data[bb + 2 + j] as i8) as f32 * d;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{dequantize, write_minimal_gguf, GgufFile};
    use crate::gguf_types::MetadataValue;
    use std::sync::Arc;

    #[test]
    fn mmap_view_gemv_matches_dense() {
        let dir = std::env::temp_dir().join("hayai_quant_mmap");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("t.gguf");
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        write_minimal_gguf(
            &path,
            &[("general.architecture", MetadataValue::String("llama".into()))],
            &[("w.weight", vec![4, 2], values)],
        )
        .unwrap();
        let gguf = Arc::new(GgufFile::open(&path).unwrap());
        let qm = QuantMatrix::from_gguf(gguf.clone(), "w.weight").unwrap();
        assert!(qm.is_mapped());
        let info = gguf.tensor("w.weight").unwrap();
        let dense = dequantize(info, gguf.tensor_bytes(info).unwrap()).unwrap();
        let x = vec![1.0f32, 0.0, 0.0, 0.0];
        let mut y_q = vec![0.0; 2];
        qm.gemv(&x, &mut y_q).unwrap();
        for row in 0..2 {
            let mut s = 0.0;
            for col in 0..4 {
                s += dense[row * 4 + col] * x[col];
            }
            assert!((y_q[row] - s).abs() < 1e-5);
        }
    }

    #[test]
    fn bf16_f64_i16_gemv_match_dense() {
        // BF16 2×4.
        let w = [1.0f32, -2.0, 0.5, 3.0, 0.0, 1.0, -1.0, 2.0];
        let mut bf = Vec::new();
        for v in w {
            bf.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
        }
        let qm = QuantMatrix::owned("bf", 4, 2, GgmlType::BF16, bf);
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![0.0; 2];
        qm.gemv(&x, &mut y).unwrap();
        assert!((y[0] - 10.5).abs() < 1e-3);
        assert!((y[1] - 7.0).abs() < 1e-3);

        // F64 dequant + GEMV.
        let vals = [1.0f64, 2.0, -3.0, 4.0];
        let mut b = Vec::new();
        for v in vals {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let qm = QuantMatrix::owned("f", 4, 1, GgmlType::F64, b);
        let ones = [1.0f32, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 1];
        qm.gemv(&ones, &mut y).unwrap();
        assert!((y[0] - 4.0).abs() < 1e-6);

        // I16 dequant + GEMV.
        let iv: [i16; 4] = [1, -2, 3, -4];
        let mut b = Vec::new();
        for v in iv {
            b.extend_from_slice(&v.to_le_bytes());
        }
        let qm = QuantMatrix::owned("i", 4, 1, GgmlType::I16, b);
        let mut y = vec![0.0; 1];
        qm.gemv(&ones, &mut y).unwrap();
        assert_eq!(y[0], -2.0);
    }

    #[test]
    fn q8_1_and_q8_k_gemv() {
        // Q8_1: 1×32, d = 1.0, all qs = 1.
        let mut b = vec![0x00u8, 0x3Cu8]; // f16 1.0 = 0x3C00
        b.extend_from_slice(&[0u8, 0u8]); // s
        b.extend_from_slice(&[1u8; 32]);
        let qm = QuantMatrix::owned("q8_1", 32, 1, GgmlType::Q8_1, b);
        let ones = vec![1.0f32; 32];
        let mut y = vec![0.0; 1];
        qm.gemv(&ones, &mut y).unwrap();
        assert!((y[0] - 32.0).abs() < 1e-3);

        // Q8_K: 1×256, d = 1.0, all qs = 1.
        let mut b = Vec::new();
        b.extend_from_slice(&1.0f32.to_le_bytes());
        b.extend_from_slice(&[1u8; 256]);
        b.extend_from_slice(&[0u8; 32]);
        let qm = QuantMatrix::owned("q8_k", 256, 1, GgmlType::Q8_K, b.clone());
        let ones = vec![1.0f32; 256];
        let mut y = vec![0.0; 1];
        qm.gemv(&ones, &mut y).unwrap();
        assert!((y[0] - 256.0).abs() < 1e-3);
        // dequantize agrees.
        let info = crate::gguf_types::TensorInfo {
            name: "q8_k".into(),
            dims: vec![256, 1],
            ggml_type: GgmlType::Q8_K,
            type_id: 15,
            offset: 0,
        };
        let d = dequantize(&info, &b).unwrap();
        assert_eq!(d.len(), 256);
        assert!((d[0] - 1.0).abs() < 1e-6);
    }
}
