#!/usr/bin/env python3
"""Append IQ2 OpenCL kernels into ggml_gemv_q4.cl."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CL = ROOT / "crates" / "hayai-kernels" / "kernels" / "ggml_gemv_q4.cl"
COMMON = Path(
    r"C:\Users\epoke\.cursor\projects\d-Documents-pySrc-hayai\agent-tools"
    r"\f4f7a0e1-46f8-48cd-b55e-42b5d5f05832.txt"
)


def extract(src: str, name: str, ty: str, n: int) -> list[str]:
    m = re.search(
        rf"GGML_TABLE_BEGIN\({ty}, {name}, {n}\)\n(.*?)GGML_TABLE_END\(\)",
        src,
        re.S,
    )
    if not m:
        raise SystemExit(f"missing {name}")
    return re.findall(r"0x[0-9a-fA-F]+", m.group(1))


def main() -> None:
    text = CL.read_text(encoding="utf-8")
    if "ggml_gemv_iq2_xxs" in text:
        print("IQ2 kernels already present")
        return

    src = COMMON.read_text(encoding="utf-8", errors="ignore")
    xxs = extract(src, "iq2xxs_grid", "uint64_t", 256)
    xs = extract(src, "iq2xs_grid", "uint64_t", 512)
    sgrid = extract(src, "iq2s_grid", "uint64_t", 1024)
    assert len(xxs) == 256 and len(xs) == 512 and len(sgrid) == 1024

    parts: list[str] = ["", "/* ---- IQ2 grids / kernels ---- */", ""]
    parts.append("__constant ulong hayai_iq2xxs_grid[256] = {")
    for i in range(0, 256, 4):
        parts.append("    " + ", ".join(f"{v}UL" for v in xxs[i : i + 4]) + ",")
    parts.append("};")
    parts.append("")
    parts.append("__constant ulong hayai_iq2xs_grid[512] = {")
    for i in range(0, 512, 4):
        parts.append("    " + ", ".join(f"{v}UL" for v in xs[i : i + 4]) + ",")
    parts.append("};")
    parts.append("")
    parts.append("__constant ulong hayai_iq2s_grid[1024] = {")
    for i in range(0, 1024, 4):
        parts.append("    " + ", ".join(f"{v}UL" for v in sgrid[i : i + 4]) + ",")
    parts.append("};")
    parts.append(
        r"""
inline uchar hayai_grid8_byte(ulong g, int j) {
    return (uchar)((g >> (8*j)) & 0xff);
}

inline float hayai_iq2_signed(float db, uchar g, uchar signs, int j) {
    float s = (signs & hayai_kmask_iq2xs[j]) ? -1.0f : 1.0f;
    return db * (float)g * s;
}

__kernel void ggml_gemv_iq2_xxs(
    const int M, const int N, const int weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output
) {
    int row = get_global_id(0);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 66;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row * row_bytes + bi * 66;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        __global const uchar* qs = block + 2;
        int x_base = bi * 256;
        int xo = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            int base = 8 * ib32;
            uint aux0 = (uint)qs[base] | ((uint)qs[base+1] << 8) | ((uint)qs[base+2] << 16) | ((uint)qs[base+3] << 24);
            uint aux1 = (uint)qs[base+4] | ((uint)qs[base+5] << 8) | ((uint)qs[base+6] << 16) | ((uint)qs[base+7] << 24);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                uchar idx = (uchar)((aux0 >> (8*l)) & 0xff);
                ulong gu = hayai_iq2xxs_grid[idx];
                uchar signs = hayai_ksigns_iq2xs[(aux1 >> (7*l)) & 127];
                for (int j = 0; j < 8; j++) {
                    sum += hayai_iq2_signed(db, hayai_grid8_byte(gu, j), signs, j) * input[x_base + xo + j];
                }
                xo += 8;
            }
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_iq2_xs(
    const int M, const int N, const int weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output
) {
    int row = get_global_id(0);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 74;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row * row_bytes + bi * 74;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        __global const uchar* qs = block + 2;
        __global const uchar* scales = block + 66;
        int x_base = bi * 256;
        int xo = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db0 = d * (0.5f + (float)(scales[ib32] & 0x0f)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                ushort q = (ushort)qs[8*ib32 + 2*l] | ((ushort)qs[8*ib32 + 2*l + 1] << 8);
                ulong gu = hayai_iq2xs_grid[q & 511];
                uchar signs = hayai_ksigns_iq2xs[q >> 9];
                float db = (l < 2) ? db0 : db1;
                for (int j = 0; j < 8; j++) {
                    sum += hayai_iq2_signed(db, hayai_grid8_byte(gu, j), signs, j) * input[x_base + xo + j];
                }
                xo += 8;
            }
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_iq2_s(
    const int M, const int N, const int weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output
) {
    int row = get_global_id(0);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 82;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row * row_bytes + bi * 82;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        __global const uchar* qs = block + 2;
        __global const uchar* signs_all = block + 34;
        __global const uchar* qh = block + 66;
        __global const uchar* scales = block + 74;
        int x_base = bi * 256;
        int xo = 0;
        int qoff = 0;
        int soff = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            float db0 = d * (0.5f + (float)(scales[ib32] & 0x0f)) * 0.25f;
            float db1 = d * (0.5f + (float)(scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                float db = (l < 2) ? db0 : db1;
                int idx = (int)qs[qoff + l] | ((((int)qh[ib32]) << (8 - 2*l)) & 0x300);
                ulong gu = hayai_iq2s_grid[idx];
                uchar signs = signs_all[soff + l];
                for (int j = 0; j < 8; j++) {
                    sum += hayai_iq2_signed(db, hayai_grid8_byte(gu, j), signs, j) * input[x_base + xo + j];
                }
                xo += 8;
            }
            qoff += 4;
            soff += 4;
        }
    }
    output[row] = sum;
}
"""
    )
    CL.write_text(text.rstrip() + "\n" + "\n".join(parts), encoding="utf-8", newline="\n")
    print(f"appended IQ2 kernels -> {CL} ({CL.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
