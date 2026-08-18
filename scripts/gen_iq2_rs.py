#!/usr/bin/env python3
"""Generate crates/hayai-model/src/iq2.rs from a local ggml-common.h dump."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = Path(
    r"C:\Users\epoke\.cursor\projects\d-Documents-pySrc-hayai\agent-tools"
    r"\f4f7a0e1-46f8-48cd-b55e-42b5d5f05832.txt"
)
OUT = ROOT / "crates" / "hayai-model" / "src" / "iq2.rs"


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
    xxs = re.findall(r"0x[0-9a-fA-F]+", extract(src, "iq2xxs_grid", "uint64_t", 256))
    xs = re.findall(r"0x[0-9a-fA-F]+", extract(src, "iq2xs_grid", "uint64_t", 512))
    sgrid = re.findall(r"0x[0-9a-fA-F]+", extract(src, "iq2s_grid", "uint64_t", 1024))
    ksigns = re.findall(r"\d+", extract(src, "ksigns_iq2xs", "uint8_t", 128))
    assert len(xxs) == 256 and len(xs) == 512 and len(sgrid) == 1024 and len(ksigns) == 128

    content = f'''//! GGML IQ2_XXS / IQ2_XS / IQ2_S (block 256) — 2-bit IQ quants with lookup grids.
//! Layout and dequant match llama.cpp `ggml-quants.c` / `ggml-common.h`.

use crate::gguf::f16_to_f32;
use crate::gguf_types::GgufError;
use rayon::prelude::*;

pub const QK_K: usize = 256;
/// `sizeof(block_iq2_xxs)` = 2 + 64
pub const IQ2_XXS_BLOCK_BYTES: usize = 66;
/// `sizeof(block_iq2_xs)` = 2 + 64 + 8
pub const IQ2_XS_BLOCK_BYTES: usize = 74;
/// `sizeof(block_iq2_s)` = 2 + 64 + 8 + 8
pub const IQ2_S_BLOCK_BYTES: usize = 82;

const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

pub static KSIGNS_IQ2XS: [u8; 128] = [
{fmt_vals(ksigns, 16)}
];

/// llama.cpp `iq2xxs_grid` — 256 × packed 8×u8.
pub static IQ2XXS_GRID: [u64; 256] = [
{fmt_vals(xxs, 4)}
];

/// llama.cpp `iq2xs_grid` — 512 × packed 8×u8.
pub static IQ2XS_GRID: [u64; 512] = [
{fmt_vals(xs, 4)}
];

/// llama.cpp `iq2s_grid` — 1024 × packed 8×u8.
pub static IQ2S_GRID: [u64; 1024] = [
{fmt_vals(sgrid, 4)}
];

#[inline]
fn grid8(grid: u64) -> [u8; 8] {{
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

fn dequant_iq2_xxs_block(block: &[u8], y: &mut [f32]) {{
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let mut yo = 0usize;
    for ib32 in 0..8 {{
        let base = 8 * ib32;
        let aux0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
        let aux1 = u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);
        let aux8 = aux0.to_le_bytes();
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {{
            let g = grid8(IQ2XXS_GRID[aux8[l] as usize]);
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {{
                y[yo + j] = signed_mul(db, g[j], signs, j);
            }}
            yo += 8;
        }}
    }}
}}

fn dequant_iq2_xs_block(block: &[u8], y: &mut [f32]) {{
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66]; // 32 × u16
    let scales = &block[66..74];
    let mut yo = 0usize;
    for ib32 in 0..8 {{
        let db0 = d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25;
        let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
        for l in 0..4 {{
            let q = u16::from_le_bytes([qs[8 * ib32 + 2 * l], qs[8 * ib32 + 2 * l + 1]]);
            let g = grid8(IQ2XS_GRID[(q & 511) as usize]);
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            let db = if l < 2 {{ db0 }} else {{ db1 }};
            for j in 0..8 {{
                y[yo + j] = signed_mul(db, g[j], signs, j);
            }}
            yo += 8;
        }}
    }}
}}

fn dequant_iq2_s_block(block: &[u8], y: &mut [f32]) {{
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..34];
    let signs_all = &block[34..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    let mut yo = 0usize;
    let mut qoff = 0usize;
    let mut soff = 0usize;
    for ib32 in 0..8 {{
        let db0 = d * (0.5 + (scales[ib32] & 0x0f) as f32) * 0.25;
        let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
        for l in 0..4 {{
            let db = if l < 2 {{ db0 }} else {{ db1 }};
            let idx = qs[qoff + l] as usize
                | ((((qh[ib32] as usize) << (8 - 2 * l)) & 0x300) as usize);
            let g = grid8(IQ2S_GRID[idx]);
            let signs = signs_all[soff + l];
            for j in 0..8 {{
                y[yo + j] = signed_mul(db, g[j], signs, j);
            }}
            yo += 8;
        }}
        qoff += 4;
        soff += 4;
    }}
}}

macro_rules! iq2_api {{
    ($dequant:ident, $gemv:ident, $extract:ident, $block_fn:ident, $blk:expr, $name:literal) => {{
        pub fn $dequant(bytes: &[u8], n: usize) -> Result<Vec<f32>, GgufError> {{
            if n % QK_K != 0 {{
                return Err(GgufError::Truncated(concat!($name, " n not multiple of 256")));
            }}
            let blocks = n / QK_K;
            if bytes.len() < blocks * $blk {{
                return Err(GgufError::Truncated(concat!($name, " tensor")));
            }}
            let mut out = vec![0.0f32; n];
            for i in 0..blocks {{
                $block_fn(&bytes[i * $blk..], &mut out[i * QK_K..(i + 1) * QK_K]);
            }}
            Ok(out)
        }}

        pub fn $gemv(
            nrows: usize,
            ncols: usize,
            data: &[u8],
            input: &[f32],
            output: &mut [f32],
        ) -> Result<(), GgufError> {{
            if ncols % QK_K != 0 {{
                return Err(GgufError::Msg(format!(
                    concat!($name, " ncols={{}} not multiple of {{}}"),
                    ncols,
                    QK_K
                )));
            }}
            let blocks_per_row = ncols / QK_K;
            let row_bytes = blocks_per_row * $blk;
            if data.len() < nrows * row_bytes {{
                return Err(GgufError::Truncated(concat!($name, " gemv")));
            }}
            assert_eq!(input.len(), ncols);
            assert_eq!(output.len(), nrows);
            output.par_iter_mut().enumerate().for_each(|(row, out)| {{
                let row_base = row * row_bytes;
                let mut sum = 0.0f32;
                let mut tmp = [0.0f32; QK_K];
                for b in 0..blocks_per_row {{
                    $block_fn(&data[row_base + b * $blk..], &mut tmp);
                    let x_base = b * QK_K;
                    for j in 0..QK_K {{
                        sum += tmp[j] * input[x_base + j];
                    }}
                }}
                *out = sum;
            }});
            Ok(())
        }}

        pub fn $extract(
            data: &[u8],
            ncols: usize,
            row: usize,
            out: &mut [f32],
        ) -> Result<(), GgufError> {{
            if ncols % QK_K != 0 || out.len() != ncols {{
                return Err(GgufError::Truncated(concat!($name, " embed row")));
            }}
            let blocks_per_row = ncols / QK_K;
            let row_bytes = blocks_per_row * $blk;
            let start = row * row_bytes;
            if data.len() < start + row_bytes {{
                return Err(GgufError::Truncated(concat!($name, " embed OOB")));
            }}
            for b in 0..blocks_per_row {{
                $block_fn(
                    &data[start + b * $blk..],
                    &mut out[b * QK_K..(b + 1) * QK_K],
                );
            }}
            Ok(())
        }}
    }};
}}

iq2_api!(
    dequant_iq2_xxs,
    gemv_iq2_xxs,
    extract_iq2_xxs_row,
    dequant_iq2_xxs_block,
    IQ2_XXS_BLOCK_BYTES,
    "iq2_xxs"
);
iq2_api!(
    dequant_iq2_xs,
    gemv_iq2_xs,
    extract_iq2_xs_row,
    dequant_iq2_xs_block,
    IQ2_XS_BLOCK_BYTES,
    "iq2_xs"
);
iq2_api!(
    dequant_iq2_s,
    gemv_iq2_s,
    extract_iq2_s_row,
    dequant_iq2_s_block,
    IQ2_S_BLOCK_BYTES,
    "iq2_s"
);

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn iq2_block_lens() {{
        let mut y = [0.0f32; QK_K];
        dequant_iq2_xxs_block(&vec![0u8; IQ2_XXS_BLOCK_BYTES], &mut y);
        dequant_iq2_xs_block(&vec![0u8; IQ2_XS_BLOCK_BYTES], &mut y);
        dequant_iq2_s_block(&vec![0u8; IQ2_S_BLOCK_BYTES], &mut y);
    }}
}}
'''
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
