#!/usr/bin/env python3
"""Generate crates/hayai-model/src/iq3.rs from a local ggml-common.h dump."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = Path(
    r"C:\Users\epoke\.cursor\projects\d-Documents-pySrc-hayai\agent-tools"
    r"\f4f7a0e1-46f8-48cd-b55e-42b5d5f05832.txt"
)
OUT = ROOT / "crates" / "hayai-model" / "src" / "iq3.rs"


def extract(src: str, name: str, ty: str, n: int) -> str:
    m = re.search(
        rf"GGML_TABLE_BEGIN\({ty}, {name}, {n}\)\n(.*?)GGML_TABLE_END\(\)",
        src,
        re.S,
    )
    if not m:
        raise SystemExit(f"missing table {name}")
    return m.group(1)


def fmt_vals(vals: list[str], per: int) -> str:
    lines = []
    for i in range(0, len(vals), per):
        lines.append("    " + ", ".join(vals[i : i + per]) + ",")
    return "\n".join(lines)


def main() -> None:
    src = COMMON.read_text(encoding="utf-8", errors="ignore")
    xxs = re.findall(r"0x[0-9a-fA-F]+", extract(src, "iq3xxs_grid", "uint32_t", 256))
    sgrid = re.findall(r"0x[0-9a-fA-F]+", extract(src, "iq3s_grid", "uint32_t", 512))
    ksigns = re.findall(r"\d+", extract(src, "ksigns_iq2xs", "uint8_t", 128))
    assert len(xxs) == 256 and len(sgrid) == 512 and len(ksigns) == 128

    content = f'''//! GGML IQ3_XXS / IQ3_S (block 256) — 3-bit IQ quants with lookup grids.
//! Layout and dequant match llama.cpp `ggml-quants.c` / `ggml-common.h`.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK_K: usize = 256;
/// `sizeof(block_iq3_xxs)` = 2 + 96
pub const IQ3_XXS_BLOCK_BYTES: usize = 98;
/// `sizeof(block_iq3_s)` = 2 + 64 + 8 + 32 + 4
pub const IQ3_S_BLOCK_BYTES: usize = 110;

const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// llama.cpp `ksigns_iq2xs` — sign bitmasks for 8 values.
pub static KSIGNS_IQ2XS: [u8; 128] = [
{fmt_vals(ksigns, 16)}
];

/// llama.cpp `iq3xxs_grid` — 256 entries, each packed as 4×u8 grid points.
pub static IQ3XXS_GRID: [u32; 256] = [
{fmt_vals(xxs, 8)}
];

/// llama.cpp `iq3s_grid` — 512 entries.
pub static IQ3S_GRID: [u32; 512] = [
{fmt_vals(sgrid, 8)}
];

#[inline]
fn grid4(grid: u32) -> [u8; 4] {{
    grid.to_le_bytes()
}}

#[inline]
fn signed_mul(db: f32, g: u8, signs: u8, j: usize) -> f32 {{
    let s = if signs & KMASK_IQ2XS[j] != 0 {{
        -1.0f32
    }} else {{
        1.0f32
    }};
    db * g as f32 * s
}}

fn dequant_iq3_xxs_block(block: &[u8], y: &mut [f32]) {{
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let scales_and_signs = &block[66..98];
    let mut yo = 0usize;
    let mut qoff = 0usize;
    for ib32 in 0..8 {{
        let aux32 = u32::from_le_bytes([
            scales_and_signs[4 * ib32],
            scales_and_signs[4 * ib32 + 1],
            scales_and_signs[4 * ib32 + 2],
            scales_and_signs[4 * ib32 + 3],
        ]);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {{
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
            let g1 = grid4(IQ3XXS_GRID[qs[qoff + 2 * l] as usize]);
            let g2 = grid4(IQ3XXS_GRID[qs[qoff + 2 * l + 1] as usize]);
            for j in 0..4 {{
                y[yo + j] = signed_mul(db, g1[j], signs, j);
                y[yo + 4 + j] = signed_mul(db, g2[j], signs, j + 4);
            }}
            yo += 8;
        }}
        qoff += 8;
    }}
}}

fn dequant_iq3_s_block(block: &[u8], y: &mut [f32]) {{
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs_all = &block[2..66];
    let qh_all = &block[66..74];
    let signs_all = &block[74..106];
    let scales = &block[106..110];
    let mut yo = 0usize;
    let mut qoff = 0usize;
    let mut soff = 0usize;
    let mut qh_off = 0usize;
    for ib32 in (0..8).step_by(2) {{
        let db1 = d * (1.0 + 2.0 * (scales[ib32 / 2] & 0x0f) as f32);
        let db2 = d * (1.0 + 2.0 * (scales[ib32 / 2] >> 4) as f32);
        let qh0 = qh_all[qh_off];
        for l in 0..4 {{
            let idx1 = qs_all[qoff + 2 * l] as usize | (((qh0 as usize) << (8 - 2 * l)) & 256);
            let idx2 = qs_all[qoff + 2 * l + 1] as usize | (((qh0 as usize) << (7 - 2 * l)) & 256);
            let g1 = grid4(IQ3S_GRID[idx1]);
            let g2 = grid4(IQ3S_GRID[idx2]);
            let signs = signs_all[soff + l];
            for j in 0..4 {{
                y[yo + j] = signed_mul(db1, g1[j], signs, j);
                y[yo + 4 + j] = signed_mul(db1, g2[j], signs, j + 4);
            }}
            yo += 8;
        }}
        qoff += 8;
        soff += 4;
        let qh1 = qh_all[qh_off + 1];
        for l in 0..4 {{
            let idx1 = qs_all[qoff + 2 * l] as usize | (((qh1 as usize) << (8 - 2 * l)) & 256);
            let idx2 = qs_all[qoff + 2 * l + 1] as usize | (((qh1 as usize) << (7 - 2 * l)) & 256);
            let g1 = grid4(IQ3S_GRID[idx1]);
            let g2 = grid4(IQ3S_GRID[idx2]);
            let signs = signs_all[soff + l];
            for j in 0..4 {{
                y[yo + j] = signed_mul(db2, g1[j], signs, j);
                y[yo + 4 + j] = signed_mul(db2, g2[j], signs, j + 4);
            }}
            yo += 8;
        }}
        qh_off += 2;
        qoff += 8;
        soff += 4;
    }}
}}

pub fn dequant_iq3_xxs(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {{
    if n % QK_K != 0 {{
        return Err(GgufError::Truncated("iq3_xxs n not multiple of 256"));
    }}
    let blocks = n / QK_K;
    if bytes.len() < blocks * IQ3_XXS_BLOCK_BYTES {{
        return Err(GgufError::Truncated("iq3_xxs tensor"));
    }}
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {{
        dequant_iq3_xxs_block(
            &bytes[i * IQ3_XXS_BLOCK_BYTES..],
            &mut out[i * QK_K..(i + 1) * QK_K],
        );
    }}
    Ok(out)
}}

pub fn dequant_iq3_s(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {{
    if n % QK_K != 0 {{
        return Err(GgufError::Truncated("iq3_s n not multiple of 256"));
    }}
    let blocks = n / QK_K;
    if bytes.len() < blocks * IQ3_S_BLOCK_BYTES {{
        return Err(GgufError::Truncated("iq3_s tensor"));
    }}
    let mut out = vec![0.0f32; n];
    for i in 0..blocks {{
        dequant_iq3_s_block(
            &bytes[i * IQ3_S_BLOCK_BYTES..],
            &mut out[i * QK_K..(i + 1) * QK_K],
        );
    }}
    Ok(out)
}}

pub fn gemv_iq3_xxs(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {{
    if ncols % QK_K != 0 {{
        return Err(GgufError::Msg(format!(
            "IQ3_XXS ncols={{ncols}} not multiple of {{QK_K}}"
        )));
    }}
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * IQ3_XXS_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {{
        return Err(GgufError::Truncated("iq3_xxs gemv"));
    }}
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {{
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        let mut tmp = [0.0f32; QK_K];
        for b in 0..blocks_per_row {{
            dequant_iq3_xxs_block(
                &data[row_base + b * IQ3_XXS_BLOCK_BYTES..],
                &mut tmp,
            );
            let x_base = b * QK_K;
            for j in 0..QK_K {{
                sum += tmp[j] * input[x_base + j];
            }}
        }}
        *out = sum;
    }});
    Ok(())
}}

pub fn gemv_iq3_s(
    nrows: usize,
    ncols: usize,
    data: &[u8],
    input: &[f32],
    output: &mut [f32],
) -> Result<(), GgufError> {{
    if ncols % QK_K != 0 {{
        return Err(GgufError::Msg(format!(
            "IQ3_S ncols={{ncols}} not multiple of {{QK_K}}"
        )));
    }}
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * IQ3_S_BLOCK_BYTES;
    if data.len() < nrows * row_bytes {{
        return Err(GgufError::Truncated("iq3_s gemv"));
    }}
    assert_eq!(input.len(), ncols);
    assert_eq!(output.len(), nrows);

    output.par_iter_mut().enumerate().for_each(|(row, out)| {{
        let row_base = row * row_bytes;
        let mut sum = 0.0f32;
        let mut tmp = [0.0f32; QK_K];
        for b in 0..blocks_per_row {{
            dequant_iq3_s_block(&data[row_base + b * IQ3_S_BLOCK_BYTES..], &mut tmp);
            let x_base = b * QK_K;
            for j in 0..QK_K {{
                sum += tmp[j] * input[x_base + j];
            }}
        }}
        *out = sum;
    }});
    Ok(())
}}

pub fn extract_iq3_xxs_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {{
    if ncols % QK_K != 0 || out.len() != ncols {{
        return Err(GgufError::Truncated("iq3_xxs embed row"));
    }}
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * IQ3_XXS_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {{
        return Err(GgufError::Truncated("iq3_xxs embed OOB"));
    }}
    for b in 0..blocks_per_row {{
        dequant_iq3_xxs_block(
            &data[start + b * IQ3_XXS_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }}
    Ok(())
}}

pub fn extract_iq3_s_row(
    data: &[u8],
    ncols: usize,
    row: usize,
    out: &mut [f32],
) -> Result<(), GgufError> {{
    if ncols % QK_K != 0 || out.len() != ncols {{
        return Err(GgufError::Truncated("iq3_s embed row"));
    }}
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * IQ3_S_BLOCK_BYTES;
    let start = row * row_bytes;
    if data.len() < start + row_bytes {{
        return Err(GgufError::Truncated("iq3_s embed OOB"));
    }}
    for b in 0..blocks_per_row {{
        dequant_iq3_s_block(
            &data[start + b * IQ3_S_BLOCK_BYTES..],
            &mut out[b * QK_K..(b + 1) * QK_K],
        );
    }}
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn iq3_xxs_block_roundtrip_len() {{
        let block = vec![0u8; IQ3_XXS_BLOCK_BYTES];
        let mut y = [0.0f32; QK_K];
        dequant_iq3_xxs_block(&block, &mut y);
        assert_eq!(y.len(), 256);
    }}

    #[test]
    fn iq3_s_block_roundtrip_len() {{
        let block = vec![0u8; IQ3_S_BLOCK_BYTES];
        let mut y = [0.0f32; QK_K];
        dequant_iq3_s_block(&block, &mut y);
        assert_eq!(y.len(), 256);
    }}
}}
'''
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
