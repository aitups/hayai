#!/usr/bin/env python3
"""Rewrite IQ3 OpenCL kernels into ggml_gemv_q4.cl with real newlines."""

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
    if "uint32" in ty:
        return re.findall(r"0x[0-9a-fA-F]+", m.group(1))
    return re.findall(r"\d+", m.group(1))


def main() -> None:
    text = CL.read_text(encoding="utf-8")
    cut = text.find("__constant uchar hayai_kmask_iq2xs")
    if cut < 0:
        # also try broken literal version start after iq4_xs
        cut = text.find("__kernel void ggml_gemv_iq3_xxs")
        if cut < 0:
            head = text.rstrip() + "\n"
        else:
            head = text[:cut].rstrip() + "\n"
            # if kmask missing, cut may be mid-broken section; prefer end of iq4_xs
    else:
        head = text[:cut].rstrip() + "\n"

    # Prefer cutting after iq4_xs kernel cleanly
    m = re.search(
        r"(__kernel void ggml_gemv_iq4_xs\b.*?^\})",
        text,
        re.M | re.S,
    )
    if m:
        head = text[: m.end()].rstrip() + "\n"

    src = COMMON.read_text(encoding="utf-8", errors="ignore")
    xxs = extract(src, "iq3xxs_grid", "uint32_t", 256)
    sgrid = extract(src, "iq3s_grid", "uint32_t", 512)
    ks = extract(src, "ksigns_iq2xs", "uint8_t", 128)
    assert len(xxs) == 256 and len(sgrid) == 512 and len(ks) == 128

    parts: list[str] = ["", "__constant uchar hayai_kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };", ""]
    parts.append("__constant uint hayai_iq3xxs_grid[256] = {")
    for i in range(0, 256, 8):
        parts.append("    " + ", ".join(xxs[i : i + 8]) + ",")
    parts.append("};")
    parts.append("")
    parts.append("__constant uint hayai_iq3s_grid[512] = {")
    for i in range(0, 512, 8):
        parts.append("    " + ", ".join(sgrid[i : i + 8]) + ",")
    parts.append("};")
    parts.append("")
    parts.append("__constant uchar hayai_ksigns_iq2xs[128] = {")
    for i in range(0, 128, 16):
        parts.append("    " + ", ".join(ks[i : i + 16]) + ",")
    parts.append("};")
    parts.append(
        """
inline float hayai_iq3_signed(float db, uchar g, uchar signs, int j) {
    float s = (signs & hayai_kmask_iq2xs[j]) ? -1.0f : 1.0f;
    return db * (float)g * s;
}

inline uchar hayai_grid_byte(uint g, int j) {
    return (uchar)((g >> (8*j)) & 0xff);
}

__kernel void ggml_gemv_iq3_xxs(
    const int M,
    const int N,
    const int weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output
) {
    int row = get_global_id(0);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 98;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row_base + bi * 98;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        __global const uchar* qs = block + 2;
        __global const uchar* scales_and_signs = block + 66;
        int x_base = bi * 256;
        int xo = 0;
        int qoff = 0;
        for (int ib32 = 0; ib32 < 8; ib32++) {
            uint aux32 = (uint)scales_and_signs[4*ib32]
                | ((uint)scales_and_signs[4*ib32+1] << 8)
                | ((uint)scales_and_signs[4*ib32+2] << 16)
                | ((uint)scales_and_signs[4*ib32+3] << 24);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                uchar signs = hayai_ksigns_iq2xs[(aux32 >> (7*l)) & 127];
                uint g1u = hayai_iq3xxs_grid[qs[qoff + 2*l]];
                uint g2u = hayai_iq3xxs_grid[qs[qoff + 2*l + 1]];
                for (int j = 0; j < 4; j++) {
                    sum += hayai_iq3_signed(db, hayai_grid_byte(g1u, j), signs, j) * input[x_base + xo + j];
                    sum += hayai_iq3_signed(db, hayai_grid_byte(g2u, j), signs, j + 4) * input[x_base + xo + 4 + j];
                }
                xo += 8;
            }
            qoff += 8;
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_iq3_s(
    const int M,
    const int N,
    const int weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output
) {
    int row = get_global_id(0);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 110;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row_base + bi * 110;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        __global const uchar* qs = block + 2;
        __global const uchar* qh = block + 66;
        __global const uchar* signs_all = block + 74;
        __global const uchar* scales = block + 106;
        int x_base = bi * 256;
        int xo = 0;
        int qoff = 0;
        int soff = 0;
        int qh_off = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            float db1 = d * (1.0f + 2.0f * (float)(scales[ib32/2] & 0x0f));
            float db2 = d * (1.0f + 2.0f * (float)(scales[ib32/2] >> 4));
            uchar qh0 = qh[qh_off];
            for (int l = 0; l < 4; l++) {
                int idx1 = (int)qs[qoff + 2*l] | ((((int)qh0) << (8 - 2*l)) & 256);
                int idx2 = (int)qs[qoff + 2*l + 1] | ((((int)qh0) << (7 - 2*l)) & 256);
                uint g1u = hayai_iq3s_grid[idx1];
                uint g2u = hayai_iq3s_grid[idx2];
                uchar signs = signs_all[soff + l];
                for (int j = 0; j < 4; j++) {
                    sum += hayai_iq3_signed(db1, hayai_grid_byte(g1u, j), signs, j) * input[x_base + xo + j];
                    sum += hayai_iq3_signed(db1, hayai_grid_byte(g2u, j), signs, j + 4) * input[x_base + xo + 4 + j];
                }
                xo += 8;
            }
            qoff += 8;
            soff += 4;
            uchar qh1 = qh[qh_off + 1];
            for (int l = 0; l < 4; l++) {
                int idx1 = (int)qs[qoff + 2*l] | ((((int)qh1) << (8 - 2*l)) & 256);
                int idx2 = (int)qs[qoff + 2*l + 1] | ((((int)qh1) << (7 - 2*l)) & 256);
                uint g1u = hayai_iq3s_grid[idx1];
                uint g2u = hayai_iq3s_grid[idx2];
                uchar signs = signs_all[soff + l];
                for (int j = 0; j < 4; j++) {
                    sum += hayai_iq3_signed(db2, hayai_grid_byte(g1u, j), signs, j) * input[x_base + xo + j];
                    sum += hayai_iq3_signed(db2, hayai_grid_byte(g2u, j), signs, j + 4) * input[x_base + xo + 4 + j];
                }
                xo += 8;
            }
            qh_off += 2;
            qoff += 8;
            soff += 4;
        }
    }
    output[row] = sum;
}
"""
    )

    CL.write_text(head + "\n".join(parts), encoding="utf-8", newline="\n")
    print(f"rewrote {CL} ({CL.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
