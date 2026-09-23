// GGML GEMV kernels for GGUF packed weights.
// One work-item per output row. `weight_off` = byte offset into `weights` (scratch pack).
//
// OpenCL 3.0 is a HARD requirement. Kernels are written in the mandatory OpenCL C
// subset (C 1.2) — the ONLY language every OpenCL 3.0 device must support; OpenCL C
// 2.0/3.0 language is optional per-device and NVIDIA's OpenCL 3.0 compiles C 1.2 only.
// Vector quant bytes are indexed through a union of vector + byte array: `.s<N>` needs
// a constant index and dynamic `v[i]` needs C 2.0+ — the union works on every vendor.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

inline float hayai_half_bits_to_float(ushort h) {
    uint sign = ((uint)(h >> 15) & 1u) << 31;
    uint exp = (h >> 10) & 0x1Fu;
    uint mant = h & 0x3FFu;
    uint bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return as_float(bits);
}


// Host always passes __local float[HAYAI_X_TILE]. Hot kernels tile over N.
#ifndef HAYAI_X_TILE
#define HAYAI_X_TILE 2048
#endif

inline void hayai_load_input_tile(int base, int ntile, __global const float* g_in, __local float* lx) {
    int lid = get_local_id(0);
    int ls = get_local_size(0);
    for (int i = lid; i < ntile; i += ls) {
        lx[i] = g_in[base + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void hayai_wg_barrier(__local float* lx) {
    if (get_local_id(0) == 0) lx[0] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void ggml_gemv_q4_0(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 32;
    int row_bytes = blocks * 18;
    float sum = 0.0f;
    // Tile input into __local (all WIs must hit barriers).
    for (int t0 = 0; t0 < N; t0 += HAYAI_X_TILE) {
        int ntile = min(HAYAI_X_TILE, N - t0);
        hayai_load_input_tile(t0, ntile, input, local_input);
        if (row < M) {
            int b0 = t0 / 32;
            int b1 = (t0 + ntile) / 32;
            int row_base = row * row_bytes;
            for (int b = b0; b < b1; b++) {
                int base = row_base + b * 18;
                ushort hd = (ushort)wbase[base] | ((ushort)wbase[base + 1] << 8);
                float d = hayai_half_bits_to_float(hd);
                int x_base = b * 32 - t0;
                // Vectorized loads (vload8 tolerates the 18-byte unaligned GGML block).
                // `.s<N>` needs a constant index and dynamic `v[i]` needs C 2.0+ (NVIDIA's
                // OpenCL 3.0 compiles C 1.2 only) — index via a union of vector + array.
                union { uchar8 v; uchar a[8]; } q0u, q1u;
                q0u.v = vload8(0, &wbase[base + 2]);
                q1u.v = vload8(0, &wbase[base + 10]);
                for (int j = 0; j < 8; j++) {
                    float x0 = (float)((int)(q0u.a[j] & 0x0F) - 8);
                    float x1 = (float)((int)(q0u.a[j] >> 4) - 8);
                    float x2 = (float)((int)(q1u.a[j] & 0x0F) - 8);
                    float x3 = (float)((int)(q1u.a[j] >> 4) - 8);
                    sum += x0 * d * local_input[x_base + j];
                    sum += x1 * d * local_input[x_base + j + 16];
                    sum += x2 * d * local_input[x_base + j + 8];
                    sum += x3 * d * local_input[x_base + j + 24];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M) output[row] = sum;
}

__kernel void ggml_gemv_q4_1(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 32;
    int row_bytes = blocks * 20;
    float sum = 0.0f;
    for (int t0 = 0; t0 < N; t0 += HAYAI_X_TILE) {
        int ntile = min(HAYAI_X_TILE, N - t0);
        hayai_load_input_tile(t0, ntile, input, local_input);
        if (row < M) {
            int b0 = t0 / 32;
            int b1 = (t0 + ntile) / 32;
            int row_base = row * row_bytes;
            for (int b = b0; b < b1; b++) {
                int base = row_base + b * 20;
                ushort hd = (ushort)wbase[base] | ((ushort)wbase[base + 1] << 8);
                ushort hm = (ushort)wbase[base + 2] | ((ushort)wbase[base + 3] << 8);
                float d = hayai_half_bits_to_float(hd);
                float m = hayai_half_bits_to_float(hm);
                int x_base = b * 32 - t0;
                // Vectorized loads (vload8 tolerates the 20-byte unaligned GGML block).
                union { uchar8 v; uchar a[8]; } q0u, q1u;
                q0u.v = vload8(0, &wbase[base + 4]);
                q1u.v = vload8(0, &wbase[base + 12]);
                for (int j = 0; j < 8; j++) {
                    float a0 = (float)(q0u.a[j] & 0x0F) * d + m;
                    float a1 = (float)(q0u.a[j] >> 4) * d + m;
                    float a2 = (float)(q1u.a[j] & 0x0F) * d + m;
                    float a3 = (float)(q1u.a[j] >> 4) * d + m;
                    sum += a0 * local_input[x_base + j];
                    sum += a1 * local_input[x_base + j + 16];
                    sum += a2 * local_input[x_base + j + 8];
                    sum += a3 * local_input[x_base + j + 24];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M) output[row] = sum;
}

__kernel void ggml_gemv_q8_0(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 32;
    int row_bytes = blocks * 34;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        int base = row_base + b * 34;
        ushort hd = (ushort)wbase[base] | ((ushort)wbase[base + 1] << 8);
        float d = hayai_half_bits_to_float(hd);
        int x_base = b * 32;
        // Vectorized loads (vload8 tolerates the 34-byte unaligned GGML block).
        union { uchar8 v; uchar a[8]; } q0u, q1u, q2u, q3u;
        q0u.v = vload8(0, &wbase[base + 2]);
        q1u.v = vload8(0, &wbase[base + 10]);
        q2u.v = vload8(0, &wbase[base + 18]);
        q3u.v = vload8(0, &wbase[base + 26]);
        for (int j = 0; j < 8; j++) {
            sum += (float)((char)q0u.a[j]) * d * input[x_base + j];
            sum += (float)((char)q1u.a[j]) * d * input[x_base + j + 8];
            sum += (float)((char)q2u.a[j]) * d * input[x_base + j + 16];
            sum += (float)((char)q3u.a[j]) * d * input[x_base + j + 24];
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_q5_0(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 32;
    int row_bytes = blocks * 22;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        int base = row_base + b * 22;
        ushort hd = (ushort)wbase[base] | ((ushort)wbase[base + 1] << 8);
        float d = hayai_half_bits_to_float(hd);
        uint qh = (uint)wbase[base + 2] | ((uint)wbase[base + 3] << 8)
                | ((uint)wbase[base + 4] << 16) | ((uint)wbase[base + 5] << 24);
        int x_base = b * 32;
        for (int j = 0; j < 16; j++) {
            uint xh_0 = ((qh >> j) << 4) & 0x10u;
            uint xh_1 = (qh >> (j + 12)) & 0x10u;
            uchar qs = wbase[base + 6 + j];
            int x0 = (int)((qs & 0x0Fu) | xh_0) - 16;
            int x1 = (int)((qs >> 4) | xh_1) - 16;
            sum += (float)x0 * d * input[x_base + j];
            sum += (float)x1 * d * input[x_base + j + 16];
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_q5_1(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 32;
    int row_bytes = blocks * 24;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int b = 0; b < blocks; b++) {
        int base = row_base + b * 24;
        ushort hd = (ushort)wbase[base] | ((ushort)wbase[base + 1] << 8);
        ushort hm = (ushort)wbase[base + 2] | ((ushort)wbase[base + 3] << 8);
        float d = hayai_half_bits_to_float(hd);
        float m = hayai_half_bits_to_float(hm);
        uint qh = (uint)wbase[base + 4] | ((uint)wbase[base + 5] << 8)
                | ((uint)wbase[base + 6] << 16) | ((uint)wbase[base + 7] << 24);
        int x_base = b * 32;
        for (int j = 0; j < 16; j++) {
            uint xh_0 = ((qh >> j) << 4) & 0x10u;
            uint xh_1 = (qh >> (j + 12)) & 0x10u;
            uchar qs = wbase[base + 8 + j];
            int x0 = (int)((qs & 0x0Fu) | xh_0);
            int x1 = (int)((qs >> 4) | xh_1);
            sum += ((float)x0 * d + m) * input[x_base + j];
            sum += ((float)x1 * d + m) * input[x_base + j + 16];
        }
    }
    output[row] = sum;
}

inline void hayai_get_scale_min_k4(int j, __global const uchar* scales, uchar* sc, uchar* m) {
    if (j < 4) {
        *sc = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
}

__kernel void ggml_gemv_q4_k(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 144;
    float sum = 0.0f;
    for (int t0 = 0; t0 < N; t0 += HAYAI_X_TILE) {
        int ntile = min(HAYAI_X_TILE, N - t0);
        hayai_load_input_tile(t0, ntile, input, local_input);
        if (row < M) {
            int bi0 = t0 / 256;
            int bi1 = (t0 + ntile) / 256;
            int row_base = row * row_bytes;
            for (int bi = bi0; bi < bi1; bi++) {
                __global const uchar* block = wbase + row_base + bi * 144;
                float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
                float minv = hayai_half_bits_to_float((ushort)block[2] | ((ushort)block[3] << 8));
                __global const uchar* scales = block + 4;
                __global const uchar* q = block + 16;
                int x_base = bi * 256 - t0;
                int is = 0;
                int qo = 0;
                for (int sub = 0; sub < 4; sub++) {
                    uchar sc0, m0, sc1, m1;
                    hayai_get_scale_min_k4(is, scales, &sc0, &m0);
                    hayai_get_scale_min_k4(is + 1, scales, &sc1, &m1);
                    float d1 = d * (float)sc0;
                    float m1v = minv * (float)m0;
                    float d2 = d * (float)sc1;
                    float m2v = minv * (float)m1;
                    // Vectorized loads: 4 × vload8 cover the 32 qs bytes of the sub-block.
                    union { uchar8 v; uchar a[8]; } qv0u, qv1u, qv2u, qv3u;
                    qv0u.v = vload8(0, &q[qo]);
                    qv1u.v = vload8(0, &q[qo + 8]);
                    qv2u.v = vload8(0, &q[qo + 16]);
                    qv3u.v = vload8(0, &q[qo + 24]);
                    for (int l = 0; l < 8; l++) {
                        int c0 = qv0u.a[l] & 0x0F;
                        int c1 = qv0u.a[l] >> 4;
                        int c2 = qv1u.a[l] & 0x0F;
                        int c3 = qv1u.a[l] >> 4;
                        int c4 = qv2u.a[l] & 0x0F;
                        int c5 = qv2u.a[l] >> 4;
                        int c6 = qv3u.a[l] & 0x0F;
                        int c7 = qv3u.a[l] >> 4;
                        int xb = x_base + sub * 64;
                        sum += (d1 * (float)c0 - m1v) * local_input[xb + l];
                        sum += (d2 * (float)c1 - m2v) * local_input[xb + 32 + l];
                        sum += (d1 * (float)c2 - m1v) * local_input[xb + 8 + l];
                        sum += (d2 * (float)c3 - m2v) * local_input[xb + 40 + l];
                        sum += (d1 * (float)c4 - m1v) * local_input[xb + 16 + l];
                        sum += (d2 * (float)c5 - m2v) * local_input[xb + 48 + l];
                        sum += (d1 * (float)c6 - m1v) * local_input[xb + 24 + l];
                        sum += (d2 * (float)c7 - m2v) * local_input[xb + 56 + l];
                    }
                    qo += 32;
                    is += 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M) output[row] = sum;
}


// Variante BATCHEADA: un dispatch para B candidatos. Rejilla [B*M]; cada work-item
// (b, row) lee la fila `row` de los pesos UNA vez y su input[b]. El tile __local
// es uniforme por workgroup (M múltiplo del tamaño de grupo en el host).
// Fase 2 (criterios C1/C4): satura la GPU en proyecciones de atención + FFN.
__kernel void ggml_gemv_batched_q4_k(
    const int M,
    const int N,
    const long weight_off,
    const int batch,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output
) {
    int gid = get_global_id(0);
    int b = gid / M;
    int row = gid - b * M;
    if (b >= batch || row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    __global const float* xrow = input + b * N;
    int blocks = N / 256;
    int row_bytes = blocks * 144;
    float sum = 0.0f;
    for (int t0 = 0; t0 < N; t0 += HAYAI_X_TILE) {
        int ntile = min(HAYAI_X_TILE, N - t0);
        int bi0 = t0 / 256;
        int bi1 = (t0 + ntile) / 256;
        int row_base = row * row_bytes;
        for (int bi = bi0; bi < bi1; bi++) {
            __global const uchar* block = wbase + row_base + bi * 144;
            float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
            float minv = hayai_half_bits_to_float((ushort)block[2] | ((ushort)block[3] << 8));
            __global const uchar* scales = block + 4;
            __global const uchar* q = block + 16;
            int x_base = bi * 256 - t0;
            int is = 0;
            int qo = 0;
            for (int sub = 0; sub < 4; sub++) {
                uchar sc0, m0, sc1, m1;
                hayai_get_scale_min_k4(is, scales, &sc0, &m0);
                hayai_get_scale_min_k4(is + 1, scales, &sc1, &m1);
                float d1 = d * (float)sc0;
                float m1v = minv * (float)m0;
                float d2 = d * (float)sc1;
                float m2v = minv * (float)m1;
                union { uchar8 v; uchar a[8]; } qv0u, qv1u, qv2u, qv3u;
                qv0u.v = vload8(0, &q[qo]);
                qv1u.v = vload8(0, &q[qo + 8]);
                qv2u.v = vload8(0, &q[qo + 16]);
                qv3u.v = vload8(0, &q[qo + 24]);
                for (int l = 0; l < 8; l++) {
                    int c0 = qv0u.a[l] & 0x0F;
                    int c1 = qv0u.a[l] >> 4;
                    int c2 = qv1u.a[l] & 0x0F;
                    int c3 = qv1u.a[l] >> 4;
                    int c4 = qv2u.a[l] & 0x0F;
                    int c5 = qv2u.a[l] >> 4;
                    int c6 = qv3u.a[l] & 0x0F;
                    int c7 = qv3u.a[l] >> 4;
                    int xb = x_base + sub * 64;
                    sum += (d1 * (float)c0 - m1v) * xrow[t0 + xb + l];
                    sum += (d2 * (float)c1 - m2v) * xrow[t0 + xb + 32 + l];
                    sum += (d1 * (float)c2 - m1v) * xrow[t0 + xb + 8 + l];
                    sum += (d2 * (float)c3 - m2v) * xrow[t0 + xb + 40 + l];
                    sum += (d1 * (float)c4 - m1v) * xrow[t0 + xb + 16 + l];
                    sum += (d2 * (float)c5 - m2v) * xrow[t0 + xb + 48 + l];
                    sum += (d1 * (float)c6 - m1v) * xrow[t0 + xb + 24 + l];
                    sum += (d2 * (float)c7 - m2v) * xrow[t0 + xb + 56 + l];
                }
                qo += 32;
                is += 2;
            }
        }
    }
    output[b * M + row] = sum;
}


__kernel void ggml_gemv_q6_k(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 210;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row_base + bi * 210;
        __global const uchar* ql = block;
        __global const uchar* qh = block + 128;
        __global const char* scales = (__global const char*)(block + 192);
        float d = hayai_half_bits_to_float((ushort)block[208] | ((ushort)block[209] << 8));
        int x_base = bi * 256;
        int qlo = 0;
        int qho = 0;
        int sco = 0;
        for (int part = 0; part < 2; part++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = ((ql[qlo + l] & 0x0F) | (((qh[qho + l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[qlo + l + 32] & 0x0F) | (((qh[qho + l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32;
                float s0 = (float)scales[sco + is];
                float s2 = (float)scales[sco + is + 2];
                float s4 = (float)scales[sco + is + 4];
                float s6 = (float)scales[sco + is + 6];
                int xb = x_base + part * 128 + l;
                sum += d * s0 * (float)q1 * input[xb];
                sum += d * s2 * (float)q2 * input[xb + 32];
                sum += d * s4 * (float)q3 * input[xb + 64];
                sum += d * s6 * (float)q4 * input[xb + 96];
            }
            qlo += 64;
            qho += 32;
            sco += 8;
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_f32(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const float* w = (__global const float*)(weights + weight_off + row * N * 4);
    float sum = 0.0f;
    for (int c = 0; c < N; c++) sum += w[c] * input[c];
    output[row] = sum;
}

__kernel void ggml_gemv_f16(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off + row * N * 2;
    float sum = 0.0f;
    for (int c = 0; c < N; c++) {
        ushort hd = (ushort)wbase[c * 2] | ((ushort)wbase[c * 2 + 1] << 8);
        sum += hayai_half_bits_to_float(hd) * input[c];
    }
    output[row] = sum;
}

__kernel void ggml_gemv_q2_k(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 84;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row_base + bi * 84;
        __global const uchar* scales = block;
        __global const uchar* q = block + 16;
        float d = hayai_half_bits_to_float((ushort)block[80] | ((ushort)block[81] << 8));
        float minv = hayai_half_bits_to_float((ushort)block[82] | ((ushort)block[83] << 8));
        int x_base = bi * 256;
        int xo = 0;
        int is = 0;
        int qo = 0;
        for (int part = 0; part < 2; part++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uchar sc = scales[is++];
                float dl = d * (float)(sc & 0xF);
                float ml = minv * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) {
                    sum += (dl * (float)((q[qo + l] >> shift) & 3) - ml) * input[x_base + xo + l];
                }
                xo += 16;
                sc = scales[is++];
                dl = d * (float)(sc & 0xF);
                ml = minv * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) {
                    sum += (dl * (float)((q[qo + 16 + l] >> shift) & 3) - ml) * input[x_base + xo + l];
                }
                xo += 16;
                shift += 2;
            }
            qo += 32;
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_q3_k(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 110;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row_base + bi * 110;
        __global const uchar* hmask = block;
        __global const uchar* q = block + 32;
        uint aux0 = (uint)block[96] | ((uint)block[97] << 8) | ((uint)block[98] << 16) | ((uint)block[99] << 24);
        uint aux1 = (uint)block[100] | ((uint)block[101] << 8) | ((uint)block[102] << 16) | ((uint)block[103] << 24);
        uint tmp = (uint)block[104] | ((uint)block[105] << 8) | ((uint)block[106] << 16) | ((uint)block[107] << 24);
        uint aux2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        uint aux3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
        char scales[16];
        scales[0] = (char)(aux0 & 0xFF); scales[1] = (char)((aux0 >> 8) & 0xFF);
        scales[2] = (char)((aux0 >> 16) & 0xFF); scales[3] = (char)((aux0 >> 24) & 0xFF);
        scales[4] = (char)(aux1 & 0xFF); scales[5] = (char)((aux1 >> 8) & 0xFF);
        scales[6] = (char)((aux1 >> 16) & 0xFF); scales[7] = (char)((aux1 >> 24) & 0xFF);
        scales[8] = (char)(aux2 & 0xFF); scales[9] = (char)((aux2 >> 8) & 0xFF);
        scales[10] = (char)((aux2 >> 16) & 0xFF); scales[11] = (char)((aux2 >> 24) & 0xFF);
        scales[12] = (char)(aux3 & 0xFF); scales[13] = (char)((aux3 >> 8) & 0xFF);
        scales[14] = (char)((aux3 >> 16) & 0xFF); scales[15] = (char)((aux3 >> 24) & 0xFF);
        float d_all = hayai_half_bits_to_float((ushort)block[108] | ((ushort)block[109] << 8));
        int x_base = bi * 256;
        int xo = 0;
        int is = 0;
        int qo = 0;
        uchar m = 1;
        for (int part = 0; part < 2; part++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (float)((int)scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int qv = (int)((q[qo + l] >> shift) & 3) - ((hmask[l] & m) ? 0 : 4);
                    sum += dl * (float)qv * input[x_base + xo + l];
                }
                xo += 16;
                dl = d_all * (float)((int)scales[is++] - 32);
                for (int l = 0; l < 16; l++) {
                    int qv = (int)((q[qo + 16 + l] >> shift) & 3) - ((hmask[l + 16] & m) ? 0 : 4);
                    sum += dl * (float)qv * input[x_base + xo + l];
                }
                xo += 16;
                shift += 2;
                m <<= 1;
            }
            qo += 32;
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_q5_k(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 176;
    float sum = 0.0f;
    for (int t0 = 0; t0 < N; t0 += HAYAI_X_TILE) {
        int ntile = min(HAYAI_X_TILE, N - t0);
        hayai_load_input_tile(t0, ntile, input, local_input);
        if (row < M) {
            int bi0 = t0 / 256;
            int bi1 = (t0 + ntile) / 256;
            int row_base = row * row_bytes;
            for (int bi = bi0; bi < bi1; bi++) {
                __global const uchar* block = wbase + row_base + bi * 176;
                float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
                float minv = hayai_half_bits_to_float((ushort)block[2] | ((ushort)block[3] << 8));
                __global const uchar* scales = block + 4;
                __global const uchar* qh = block + 16;
                __global const uchar* ql = block + 48;
                int x_base = bi * 256 - t0;
                int xo = 0;
                int is = 0;
                int qlo = 0;
                uchar u1 = 1;
                uchar u2 = 2;
                for (int sub = 0; sub < 4; sub++) {
                    uchar sc0, m0, sc1, m1;
                    hayai_get_scale_min_k4(is, scales, &sc0, &m0);
                    hayai_get_scale_min_k4(is + 1, scales, &sc1, &m1);
                    float d1 = d * (float)sc0;
                    float m1v = minv * (float)m0;
                    float d2 = d * (float)sc1;
                    float m2v = minv * (float)m1;
                    for (int l = 0; l < 32; l++) {
                        int v = (ql[qlo + l] & 0x0F) + ((qh[l] & u1) ? 16 : 0);
                        sum += (d1 * (float)v - m1v) * local_input[x_base + xo + l];
                    }
                    xo += 32;
                    for (int l = 0; l < 32; l++) {
                        int v = (ql[qlo + l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                        sum += (d2 * (float)v - m2v) * local_input[x_base + xo + l];
                    }
                    xo += 32;
                    qlo += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M) output[row] = sum;
}

__constant char hayai_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

__kernel void ggml_gemv_iq4_nl(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 32;
    int row_bytes = blocks * 18;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row_base + bi * 18;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        __global const uchar* qs = block + 2;
        int x_base = bi * 32;
        for (int j = 0; j < 16; j++) {
            sum += d * (float)hayai_kvalues_iq4nl[qs[j] & 0x0F] * input[x_base + j];
            sum += d * (float)hayai_kvalues_iq4nl[qs[j] >> 4] * input[x_base + 16 + j];
        }
    }
    output[row] = sum;
}

__kernel void ggml_gemv_iq4_xs(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off;
    int blocks = N / 256;
    int row_bytes = blocks * 136;
    int row_base = row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + row_base + bi * 136;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        ushort scales_h = (ushort)block[2] | ((ushort)block[3] << 8);
        __global const uchar* scales_l = block + 4;
        __global const uchar* qs = block + 8;
        int x_base = bi * 256;
        int xo = 0;
        int qso = 0;
        for (int ib = 0; ib < 8; ib++) {
            int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((int)((scales_h >> (2 * ib)) & 3)) << 4);
            float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; j++) {
                sum += dl * (float)hayai_kvalues_iq4nl[qs[qso + j] & 0x0F] * input[x_base + xo + j];
                sum += dl * (float)hayai_kvalues_iq4nl[qs[qso + j] >> 4] * input[x_base + xo + 16 + j];
            }
            xo += 32;
            qso += 16;
        }
    }
    output[row] = sum;
}

__constant uchar hayai_kmask_iq2xs[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };

__constant uint hayai_iq3xxs_grid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

__constant uint hayai_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

__constant uchar hayai_ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

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
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
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
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
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

/* ---- IQ2 grids / kernels ---- */

__constant ulong hayai_iq2xxs_grid[256] = {
    0x0808080808080808UL, 0x080808080808082bUL, 0x0808080808081919UL, 0x0808080808082b08UL,
    0x0808080808082b2bUL, 0x0808080808190819UL, 0x0808080808191908UL, 0x08080808082b0808UL,
    0x08080808082b082bUL, 0x08080808082b2b08UL, 0x08080808082b2b2bUL, 0x0808080819080819UL,
    0x0808080819081908UL, 0x0808080819190808UL, 0x0808080819192b08UL, 0x08080808192b0819UL,
    0x08080808192b1908UL, 0x080808082b080808UL, 0x080808082b08082bUL, 0x080808082b082b2bUL,
    0x080808082b2b082bUL, 0x0808081908080819UL, 0x0808081908081908UL, 0x0808081908190808UL,
    0x0808081908191919UL, 0x0808081919080808UL, 0x080808192b081908UL, 0x080808192b192b08UL,
    0x0808082b08080808UL, 0x0808082b0808082bUL, 0x0808082b082b082bUL, 0x0808082b2b08082bUL,
    0x0808190808080819UL, 0x0808190808081908UL, 0x0808190808190808UL, 0x08081908082b0819UL,
    0x08081908082b1908UL, 0x0808190819080808UL, 0x080819081908082bUL, 0x0808190819082b08UL,
    0x08081908192b0808UL, 0x080819082b080819UL, 0x080819082b081908UL, 0x080819082b190808UL,
    0x080819082b2b1908UL, 0x0808191908080808UL, 0x080819190808082bUL, 0x0808191908082b08UL,
    0x08081919082b0808UL, 0x080819191908192bUL, 0x08081919192b2b19UL, 0x080819192b080808UL,
    0x080819192b190819UL, 0x0808192b08082b19UL, 0x0808192b08190808UL, 0x0808192b19080808UL,
    0x0808192b2b081908UL, 0x0808192b2b2b1908UL, 0x08082b0808080808UL, 0x08082b0808081919UL,
    0x08082b0808082b08UL, 0x08082b0808191908UL, 0x08082b08082b2b08UL, 0x08082b0819080819UL,
    0x08082b0819081908UL, 0x08082b0819190808UL, 0x08082b081919082bUL, 0x08082b082b082b08UL,
    0x08082b1908081908UL, 0x08082b1919080808UL, 0x08082b2b0808082bUL, 0x08082b2b08191908UL,
    0x0819080808080819UL, 0x0819080808081908UL, 0x0819080808190808UL, 0x08190808082b0819UL,
    0x0819080819080808UL, 0x08190808192b0808UL, 0x081908082b081908UL, 0x081908082b190808UL,
    0x081908082b191919UL, 0x0819081908080808UL, 0x0819081908082b08UL, 0x08190819082b0808UL,
    0x0819081919190808UL, 0x0819081919192b2bUL, 0x081908192b080808UL, 0x0819082b082b1908UL,
    0x0819082b19081919UL, 0x0819190808080808UL, 0x0819190808082b08UL, 0x08191908082b0808UL,
    0x08191908082b1919UL, 0x0819190819082b19UL, 0x081919082b080808UL, 0x0819191908192b08UL,
    0x08191919192b082bUL, 0x0819192b08080808UL, 0x0819192b0819192bUL, 0x08192b0808080819UL,
    0x08192b0808081908UL, 0x08192b0808190808UL, 0x08192b0819080808UL, 0x08192b082b080819UL,
    0x08192b1908080808UL, 0x08192b1908081919UL, 0x08192b192b2b0808UL, 0x08192b2b19190819UL,
    0x082b080808080808UL, 0x082b08080808082bUL, 0x082b080808082b2bUL, 0x082b080819081908UL,
    0x082b0808192b0819UL, 0x082b08082b080808UL, 0x082b08082b08082bUL, 0x082b0819082b2b19UL,
    0x082b081919082b08UL, 0x082b082b08080808UL, 0x082b082b0808082bUL, 0x082b190808080819UL,
    0x082b190808081908UL, 0x082b190808190808UL, 0x082b190819080808UL, 0x082b19081919192bUL,
    0x082b191908080808UL, 0x082b191919080819UL, 0x082b1919192b1908UL, 0x082b192b2b190808UL,
    0x082b2b0808082b08UL, 0x082b2b08082b0808UL, 0x082b2b082b191908UL, 0x082b2b2b19081908UL,
    0x1908080808080819UL, 0x1908080808081908UL, 0x1908080808190808UL, 0x1908080808192b08UL,
    0x19080808082b0819UL, 0x19080808082b1908UL, 0x1908080819080808UL, 0x1908080819082b08UL,
    0x190808081919192bUL, 0x19080808192b0808UL, 0x190808082b080819UL, 0x190808082b081908UL,
    0x190808082b190808UL, 0x1908081908080808UL, 0x19080819082b0808UL, 0x19080819192b0819UL,
    0x190808192b080808UL, 0x190808192b081919UL, 0x1908082b08080819UL, 0x1908082b08190808UL,
    0x1908082b19082b08UL, 0x1908082b1919192bUL, 0x1908082b192b2b08UL, 0x1908190808080808UL,
    0x1908190808082b08UL, 0x19081908082b0808UL, 0x190819082b080808UL, 0x190819082b192b19UL,
    0x190819190819082bUL, 0x19081919082b1908UL, 0x1908192b08080808UL, 0x19082b0808080819UL,
    0x19082b0808081908UL, 0x19082b0808190808UL, 0x19082b0819080808UL, 0x19082b0819081919UL,
    0x19082b1908080808UL, 0x19082b1919192b08UL, 0x19082b19192b0819UL, 0x19082b192b08082bUL,
    0x19082b2b19081919UL, 0x19082b2b2b190808UL, 0x1919080808080808UL, 0x1919080808082b08UL,
    0x1919080808190819UL, 0x1919080808192b19UL, 0x19190808082b0808UL, 0x191908082b080808UL,
    0x191908082b082b08UL, 0x1919081908081908UL, 0x191908191908082bUL, 0x191908192b2b1908UL,
    0x1919082b2b190819UL, 0x191919082b190808UL, 0x191919082b19082bUL, 0x1919191908082b2bUL,
    0x1919192b08080819UL, 0x1919192b19191908UL, 0x19192b0808080808UL, 0x19192b0808190819UL,
    0x19192b0808192b19UL, 0x19192b08192b1908UL, 0x19192b1919080808UL, 0x19192b2b08082b08UL,
    0x192b080808081908UL, 0x192b080808190808UL, 0x192b080819080808UL, 0x192b0808192b2b08UL,
    0x192b081908080808UL, 0x192b081919191919UL, 0x192b082b08192b08UL, 0x192b082b192b0808UL,
    0x192b190808080808UL, 0x192b190808081919UL, 0x192b191908190808UL, 0x192b19190819082bUL,
    0x192b19192b081908UL, 0x192b2b081908082bUL, 0x2b08080808080808UL, 0x2b0808080808082bUL,
    0x2b08080808082b2bUL, 0x2b08080819080819UL, 0x2b0808082b08082bUL, 0x2b08081908081908UL,
    0x2b08081908192b08UL, 0x2b08081919080808UL, 0x2b08082b08190819UL, 0x2b08190808080819UL,
    0x2b08190808081908UL, 0x2b08190808190808UL, 0x2b08190808191919UL, 0x2b08190819080808UL,
    0x2b081908192b0808UL, 0x2b08191908080808UL, 0x2b0819191908192bUL, 0x2b0819192b191908UL,
    0x2b08192b08082b19UL, 0x2b08192b19080808UL, 0x2b08192b192b0808UL, 0x2b082b080808082bUL,
    0x2b082b1908081908UL, 0x2b082b2b08190819UL, 0x2b19080808081908UL, 0x2b19080808190808UL,
    0x2b190808082b1908UL, 0x2b19080819080808UL, 0x2b1908082b2b0819UL, 0x2b1908190819192bUL,
    0x2b1908192b080808UL, 0x2b19082b19081919UL, 0x2b19190808080808UL, 0x2b191908082b082bUL,
    0x2b19190819081908UL, 0x2b19191919190819UL, 0x2b192b082b080819UL, 0x2b192b19082b0808UL,
    0x2b2b08080808082bUL, 0x2b2b080819190808UL, 0x2b2b08082b081919UL, 0x2b2b081908082b19UL,
    0x2b2b082b08080808UL, 0x2b2b190808192b08UL, 0x2b2b2b0819190808UL, 0x2b2b2b1908081908UL,
};

__constant ulong hayai_iq2xs_grid[512] = {
    0x0808080808080808UL, 0x080808080808082bUL, 0x0808080808081919UL, 0x0808080808082b08UL,
    0x0808080808082b2bUL, 0x0808080808190819UL, 0x0808080808191908UL, 0x080808080819192bUL,
    0x0808080808192b19UL, 0x08080808082b0808UL, 0x08080808082b082bUL, 0x08080808082b1919UL,
    0x08080808082b2b08UL, 0x0808080819080819UL, 0x0808080819081908UL, 0x080808081908192bUL,
    0x0808080819082b19UL, 0x0808080819190808UL, 0x080808081919082bUL, 0x0808080819191919UL,
    0x0808080819192b08UL, 0x08080808192b0819UL, 0x08080808192b1908UL, 0x080808082b080808UL,
    0x080808082b08082bUL, 0x080808082b081919UL, 0x080808082b082b08UL, 0x080808082b190819UL,
    0x080808082b191908UL, 0x080808082b192b19UL, 0x080808082b2b0808UL, 0x0808081908080819UL,
    0x0808081908081908UL, 0x080808190808192bUL, 0x0808081908082b19UL, 0x0808081908190808UL,
    0x080808190819082bUL, 0x0808081908191919UL, 0x0808081908192b08UL, 0x0808081908192b2bUL,
    0x08080819082b0819UL, 0x08080819082b1908UL, 0x0808081919080808UL, 0x080808191908082bUL,
    0x0808081919081919UL, 0x0808081919082b08UL, 0x0808081919190819UL, 0x0808081919191908UL,
    0x08080819192b0808UL, 0x08080819192b2b08UL, 0x080808192b080819UL, 0x080808192b081908UL,
    0x080808192b190808UL, 0x0808082b08080808UL, 0x0808082b0808082bUL, 0x0808082b08081919UL,
    0x0808082b08082b08UL, 0x0808082b08190819UL, 0x0808082b08191908UL, 0x0808082b082b0808UL,
    0x0808082b19080819UL, 0x0808082b19081908UL, 0x0808082b19190808UL, 0x0808082b19191919UL,
    0x0808082b2b080808UL, 0x0808082b2b082b2bUL, 0x0808190808080819UL, 0x0808190808081908UL,
    0x080819080808192bUL, 0x0808190808082b19UL, 0x0808190808190808UL, 0x080819080819082bUL,
    0x0808190808191919UL, 0x0808190808192b08UL, 0x08081908082b0819UL, 0x08081908082b1908UL,
    0x0808190819080808UL, 0x080819081908082bUL, 0x0808190819081919UL, 0x0808190819082b08UL,
    0x0808190819190819UL, 0x0808190819191908UL, 0x080819081919192bUL, 0x08081908192b0808UL,
    0x080819082b080819UL, 0x080819082b081908UL, 0x080819082b190808UL, 0x0808191908080808UL,
    0x080819190808082bUL, 0x0808191908081919UL, 0x0808191908082b08UL, 0x0808191908190819UL,
    0x0808191908191908UL, 0x08081919082b0808UL, 0x0808191919080819UL, 0x0808191919081908UL,
    0x0808191919190808UL, 0x08081919192b0819UL, 0x080819192b080808UL, 0x0808192b08080819UL,
    0x0808192b08081908UL, 0x0808192b08190808UL, 0x0808192b082b192bUL, 0x0808192b19080808UL,
    0x0808192b1908082bUL, 0x0808192b2b081908UL, 0x08082b0808080808UL, 0x08082b080808082bUL,
    0x08082b0808081919UL, 0x08082b0808082b08UL, 0x08082b0808082b2bUL, 0x08082b0808190819UL,
    0x08082b0808191908UL, 0x08082b08082b0808UL, 0x08082b08082b1919UL, 0x08082b0819080819UL,
    0x08082b0819081908UL, 0x08082b0819190808UL, 0x08082b0819192b08UL, 0x08082b082b080808UL,
    0x08082b082b2b0808UL, 0x08082b082b2b2b2bUL, 0x08082b1908080819UL, 0x08082b1908081908UL,
    0x08082b1908190808UL, 0x08082b1919080808UL, 0x08082b192b080819UL, 0x08082b192b082b19UL,
    0x08082b2b08080808UL, 0x08082b2b082b0808UL, 0x08082b2b082b2b08UL, 0x08082b2b2b19192bUL,
    0x08082b2b2b2b0808UL, 0x0819080808080819UL, 0x0819080808081908UL, 0x081908080808192bUL,
    0x0819080808082b19UL, 0x0819080808190808UL, 0x081908080819082bUL, 0x0819080808191919UL,
    0x0819080808192b08UL, 0x08190808082b0819UL, 0x08190808082b1908UL, 0x0819080819080808UL,
    0x081908081908082bUL, 0x0819080819081919UL, 0x0819080819082b08UL, 0x0819080819190819UL,
    0x0819080819191908UL, 0x08190808192b0808UL, 0x08190808192b2b2bUL, 0x081908082b080819UL,
    0x081908082b081908UL, 0x081908082b190808UL, 0x0819081908080808UL, 0x081908190808082bUL,
    0x0819081908081919UL, 0x0819081908082b08UL, 0x0819081908190819UL, 0x0819081908191908UL,
    0x08190819082b0808UL, 0x0819081919080819UL, 0x0819081919081908UL, 0x0819081919190808UL,
    0x081908192b080808UL, 0x081908192b191908UL, 0x081908192b19192bUL, 0x0819082b08080819UL,
    0x0819082b08081908UL, 0x0819082b0808192bUL, 0x0819082b08190808UL, 0x0819082b19080808UL,
    0x0819082b192b0808UL, 0x0819190808080808UL, 0x081919080808082bUL, 0x0819190808081919UL,
    0x0819190808082b08UL, 0x0819190808190819UL, 0x0819190808191908UL, 0x08191908082b0808UL,
    0x0819190819080819UL, 0x0819190819081908UL, 0x0819190819082b19UL, 0x0819190819190808UL,
    0x08191908192b1908UL, 0x081919082b080808UL, 0x0819191908080819UL, 0x0819191908081908UL,
    0x0819191908190808UL, 0x0819191919080808UL, 0x0819192b08080808UL, 0x0819192b08191908UL,
    0x0819192b19082b19UL, 0x08192b0808080819UL, 0x08192b0808081908UL, 0x08192b0808190808UL,
    0x08192b080819082bUL, 0x08192b0819080808UL, 0x08192b0819191908UL, 0x08192b082b08192bUL,
    0x08192b1908080808UL, 0x08192b1908081919UL, 0x08192b19192b192bUL, 0x08192b2b19190819UL,
    0x08192b2b2b2b2b19UL, 0x082b080808080808UL, 0x082b08080808082bUL, 0x082b080808081919UL,
    0x082b080808082b08UL, 0x082b080808082b2bUL, 0x082b080808190819UL, 0x082b080808191908UL,
    0x082b0808082b0808UL, 0x082b080819080819UL, 0x082b080819081908UL, 0x082b080819190808UL,
    0x082b08082b080808UL, 0x082b08082b2b0808UL, 0x082b081908080819UL, 0x082b081908081908UL,
    0x082b081908190808UL, 0x082b081919080808UL, 0x082b081919082b08UL, 0x082b0819192b1919UL,
    0x082b082b08080808UL, 0x082b082b082b082bUL, 0x082b082b2b080808UL, 0x082b082b2b2b2b08UL,
    0x082b190808080819UL, 0x082b190808081908UL, 0x082b190808190808UL, 0x082b1908082b2b19UL,
    0x082b190819080808UL, 0x082b191908080808UL, 0x082b191919080819UL, 0x082b19191919082bUL,
    0x082b19192b192b19UL, 0x082b192b08080819UL, 0x082b192b08192b2bUL, 0x082b192b2b2b192bUL,
    0x082b2b0808080808UL, 0x082b2b0808082b08UL, 0x082b2b0808082b2bUL, 0x082b2b08082b0808UL,
    0x082b2b0819191919UL, 0x082b2b082b082b08UL, 0x082b2b082b2b082bUL, 0x082b2b19192b2b08UL,
    0x082b2b192b190808UL, 0x082b2b2b08082b08UL, 0x082b2b2b082b0808UL, 0x082b2b2b2b08082bUL,
    0x082b2b2b2b082b08UL, 0x082b2b2b2b082b2bUL, 0x1908080808080819UL, 0x1908080808081908UL,
    0x190808080808192bUL, 0x1908080808082b19UL, 0x1908080808190808UL, 0x190808080819082bUL,
    0x1908080808191919UL, 0x1908080808192b08UL, 0x19080808082b0819UL, 0x19080808082b1908UL,
    0x1908080819080808UL, 0x190808081908082bUL, 0x1908080819081919UL, 0x1908080819082b08UL,
    0x1908080819082b2bUL, 0x1908080819190819UL, 0x1908080819191908UL, 0x19080808192b0808UL,
    0x19080808192b1919UL, 0x190808082b080819UL, 0x190808082b081908UL, 0x190808082b190808UL,
    0x1908081908080808UL, 0x190808190808082bUL, 0x1908081908081919UL, 0x1908081908082b08UL,
    0x1908081908190819UL, 0x1908081908191908UL, 0x19080819082b0808UL, 0x1908081919080819UL,
    0x1908081919081908UL, 0x1908081919190808UL, 0x190808192b080808UL, 0x190808192b081919UL,
    0x190808192b2b082bUL, 0x1908082b08080819UL, 0x1908082b08081908UL, 0x1908082b08190808UL,
    0x1908082b0819082bUL, 0x1908082b082b2b19UL, 0x1908082b19080808UL, 0x1908190808080808UL,
    0x190819080808082bUL, 0x1908190808081919UL, 0x1908190808082b08UL, 0x1908190808190819UL,
    0x1908190808191908UL, 0x1908190808192b19UL, 0x19081908082b0808UL, 0x1908190819080819UL,
    0x1908190819081908UL, 0x1908190819190808UL, 0x190819082b080808UL, 0x190819082b191908UL,
    0x1908191908080819UL, 0x1908191908081908UL, 0x1908191908190808UL, 0x19081919082b1908UL,
    0x1908191919080808UL, 0x190819192b192b2bUL, 0x1908192b08080808UL, 0x1908192b08082b2bUL,
    0x1908192b19081908UL, 0x1908192b19190808UL, 0x19082b0808080819UL, 0x19082b0808081908UL,
    0x19082b0808190808UL, 0x19082b0819080808UL, 0x19082b0819081919UL, 0x19082b0819191908UL,
    0x19082b08192b082bUL, 0x19082b1908080808UL, 0x19082b1908190819UL, 0x19082b1919081908UL,
    0x19082b1919190808UL, 0x19082b19192b2b19UL, 0x19082b2b08081908UL, 0x1919080808080808UL,
    0x191908080808082bUL, 0x1919080808081919UL, 0x1919080808082b08UL, 0x1919080808190819UL,
    0x1919080808191908UL, 0x19190808082b0808UL, 0x19190808082b2b08UL, 0x1919080819080819UL,
    0x1919080819081908UL, 0x1919080819190808UL, 0x191908082b080808UL, 0x1919081908080819UL,
    0x1919081908081908UL, 0x1919081908190808UL, 0x1919081908191919UL, 0x1919081919080808UL,
    0x191908191908082bUL, 0x1919082b08080808UL, 0x1919082b19081908UL, 0x1919082b2b2b2b2bUL,
    0x1919190808080819UL, 0x1919190808081908UL, 0x1919190808190808UL, 0x19191908082b0819UL,
    0x1919190819080808UL, 0x19191908192b0808UL, 0x191919082b080819UL, 0x191919082b2b0819UL,
    0x1919191908080808UL, 0x1919191908082b08UL, 0x191919192b080808UL, 0x191919192b082b08UL,
    0x1919192b082b0819UL, 0x1919192b192b2b08UL, 0x1919192b2b2b0819UL, 0x19192b0808080808UL,
    0x19192b0808191908UL, 0x19192b0819080819UL, 0x19192b0819190808UL, 0x19192b082b192b19UL,
    0x19192b1908192b2bUL, 0x19192b1919080808UL, 0x19192b191908082bUL, 0x19192b2b2b081919UL,
    0x192b080808080819UL, 0x192b080808081908UL, 0x192b080808190808UL, 0x192b080819080808UL,
    0x192b080819191908UL, 0x192b0808192b082bUL, 0x192b08082b08192bUL, 0x192b08082b2b2b19UL,
    0x192b081908080808UL, 0x192b082b082b1908UL, 0x192b082b19082b2bUL, 0x192b082b2b19082bUL,
    0x192b190808080808UL, 0x192b19080819192bUL, 0x192b191908190808UL, 0x192b191919080808UL,
    0x192b191919081919UL, 0x192b19192b2b1908UL, 0x192b2b0808080819UL, 0x192b2b08192b2b2bUL,
    0x192b2b19082b1919UL, 0x192b2b2b0808192bUL, 0x192b2b2b19191908UL, 0x192b2b2b192b082bUL,
    0x2b08080808080808UL, 0x2b0808080808082bUL, 0x2b08080808081919UL, 0x2b08080808082b08UL,
    0x2b08080808190819UL, 0x2b08080808191908UL, 0x2b080808082b0808UL, 0x2b080808082b2b2bUL,
    0x2b08080819080819UL, 0x2b08080819081908UL, 0x2b08080819190808UL, 0x2b0808082b080808UL,
    0x2b0808082b08082bUL, 0x2b0808082b2b2b08UL, 0x2b0808082b2b2b2bUL, 0x2b08081908080819UL,
    0x2b08081908081908UL, 0x2b0808190808192bUL, 0x2b08081908190808UL, 0x2b08081919080808UL,
    0x2b08081919190819UL, 0x2b08081919192b19UL, 0x2b08082b08080808UL, 0x2b08082b082b0808UL,
    0x2b08082b2b080808UL, 0x2b08082b2b08082bUL, 0x2b08082b2b2b0808UL, 0x2b08082b2b2b2b08UL,
    0x2b08190808080819UL, 0x2b08190808081908UL, 0x2b08190808190808UL, 0x2b0819080819082bUL,
    0x2b08190808191919UL, 0x2b08190819080808UL, 0x2b081908192b0808UL, 0x2b0819082b082b19UL,
    0x2b08191908080808UL, 0x2b08191919081908UL, 0x2b0819192b2b1919UL, 0x2b08192b08192b08UL,
    0x2b08192b192b2b2bUL, 0x2b082b0808080808UL, 0x2b082b0808082b08UL, 0x2b082b08082b1919UL,
    0x2b082b0819192b2bUL, 0x2b082b082b080808UL, 0x2b082b082b08082bUL, 0x2b082b082b2b2b08UL,
    0x2b082b190808192bUL, 0x2b082b2b082b082bUL, 0x2b082b2b2b080808UL, 0x2b082b2b2b082b08UL,
    0x2b082b2b2b19192bUL, 0x2b082b2b2b2b2b08UL, 0x2b19080808080819UL, 0x2b19080808081908UL,
    0x2b19080808190808UL, 0x2b19080819080808UL, 0x2b1908081919192bUL, 0x2b1908082b081908UL,
    0x2b19081908080808UL, 0x2b190819082b082bUL, 0x2b190819192b1908UL, 0x2b19082b1919192bUL,
    0x2b19082b2b082b19UL, 0x2b19190808080808UL, 0x2b19190808081919UL, 0x2b19190819081908UL,
    0x2b19190819190808UL, 0x2b19190819192b08UL, 0x2b191919082b2b19UL, 0x2b1919192b190808UL,
    0x2b1919192b19082bUL, 0x2b19192b19080819UL, 0x2b192b0819190819UL, 0x2b192b082b2b192bUL,
    0x2b192b1919082b19UL, 0x2b192b2b08191919UL, 0x2b192b2b192b0808UL, 0x2b2b080808080808UL,
    0x2b2b08080808082bUL, 0x2b2b080808082b08UL, 0x2b2b080808082b2bUL, 0x2b2b0808082b0808UL,
    0x2b2b0808082b2b2bUL, 0x2b2b08082b2b0808UL, 0x2b2b081919190819UL, 0x2b2b081919192b19UL,
    0x2b2b08192b2b192bUL, 0x2b2b082b08080808UL, 0x2b2b082b0808082bUL, 0x2b2b082b08082b08UL,
    0x2b2b082b082b2b2bUL, 0x2b2b082b2b080808UL, 0x2b2b082b2b2b0808UL, 0x2b2b190819080808UL,
    0x2b2b19082b191919UL, 0x2b2b192b192b1919UL, 0x2b2b192b2b192b08UL, 0x2b2b2b0808082b2bUL,
    0x2b2b2b08082b0808UL, 0x2b2b2b08082b082bUL, 0x2b2b2b08082b2b08UL, 0x2b2b2b082b2b0808UL,
    0x2b2b2b082b2b2b08UL, 0x2b2b2b1908081908UL, 0x2b2b2b192b081908UL, 0x2b2b2b192b08192bUL,
    0x2b2b2b2b082b2b08UL, 0x2b2b2b2b082b2b2bUL, 0x2b2b2b2b2b190819UL, 0x2b2b2b2b2b2b2b2bUL,
};

__constant ulong hayai_iq2s_grid[1024] = {
    0x0808080808080808UL, 0x080808080808082bUL, 0x0808080808081919UL, 0x0808080808082b08UL,
    0x0808080808082b2bUL, 0x0808080808190819UL, 0x0808080808191908UL, 0x080808080819192bUL,
    0x0808080808192b19UL, 0x08080808082b0808UL, 0x08080808082b082bUL, 0x08080808082b1919UL,
    0x08080808082b2b08UL, 0x0808080819080819UL, 0x0808080819081908UL, 0x080808081908192bUL,
    0x0808080819082b19UL, 0x0808080819190808UL, 0x080808081919082bUL, 0x0808080819191919UL,
    0x0808080819192b08UL, 0x08080808192b0819UL, 0x08080808192b1908UL, 0x08080808192b192bUL,
    0x08080808192b2b19UL, 0x080808082b080808UL, 0x080808082b08082bUL, 0x080808082b081919UL,
    0x080808082b082b08UL, 0x080808082b190819UL, 0x080808082b191908UL, 0x080808082b2b0808UL,
    0x080808082b2b1919UL, 0x080808082b2b2b2bUL, 0x0808081908080819UL, 0x0808081908081908UL,
    0x080808190808192bUL, 0x0808081908082b19UL, 0x0808081908190808UL, 0x080808190819082bUL,
    0x0808081908191919UL, 0x0808081908192b08UL, 0x08080819082b0819UL, 0x08080819082b1908UL,
    0x0808081919080808UL, 0x080808191908082bUL, 0x0808081919081919UL, 0x0808081919082b08UL,
    0x0808081919190819UL, 0x0808081919191908UL, 0x080808191919192bUL, 0x0808081919192b19UL,
    0x08080819192b0808UL, 0x08080819192b1919UL, 0x08080819192b2b08UL, 0x080808192b080819UL,
    0x080808192b081908UL, 0x080808192b190808UL, 0x080808192b19082bUL, 0x080808192b191919UL,
    0x080808192b2b0819UL, 0x080808192b2b1908UL, 0x0808082b08080808UL, 0x0808082b0808082bUL,
    0x0808082b08081919UL, 0x0808082b08082b08UL, 0x0808082b08190819UL, 0x0808082b08191908UL,
    0x0808082b082b0808UL, 0x0808082b082b2b2bUL, 0x0808082b19080819UL, 0x0808082b19081908UL,
    0x0808082b1908192bUL, 0x0808082b19082b19UL, 0x0808082b19190808UL, 0x0808082b19191919UL,
    0x0808082b2b080808UL, 0x0808082b2b081919UL, 0x0808082b2b082b2bUL, 0x0808082b2b191908UL,
    0x0808082b2b2b082bUL, 0x0808190808080819UL, 0x0808190808081908UL, 0x080819080808192bUL,
    0x0808190808082b19UL, 0x0808190808190808UL, 0x080819080819082bUL, 0x0808190808191919UL,
    0x0808190808192b08UL, 0x08081908082b0819UL, 0x08081908082b1908UL, 0x08081908082b192bUL,
    0x08081908082b2b19UL, 0x0808190819080808UL, 0x080819081908082bUL, 0x0808190819081919UL,
    0x0808190819082b08UL, 0x0808190819082b2bUL, 0x0808190819190819UL, 0x0808190819191908UL,
    0x080819081919192bUL, 0x0808190819192b19UL, 0x08081908192b0808UL, 0x08081908192b082bUL,
    0x08081908192b1919UL, 0x080819082b080819UL, 0x080819082b081908UL, 0x080819082b08192bUL,
    0x080819082b082b19UL, 0x080819082b190808UL, 0x080819082b191919UL, 0x080819082b192b08UL,
    0x080819082b2b0819UL, 0x080819082b2b1908UL, 0x0808191908080808UL, 0x080819190808082bUL,
    0x0808191908081919UL, 0x0808191908082b08UL, 0x0808191908082b2bUL, 0x0808191908190819UL,
    0x0808191908191908UL, 0x080819190819192bUL, 0x0808191908192b19UL, 0x08081919082b0808UL,
    0x08081919082b1919UL, 0x08081919082b2b08UL, 0x0808191919080819UL, 0x0808191919081908UL,
    0x080819191908192bUL, 0x0808191919082b19UL, 0x0808191919190808UL, 0x080819191919082bUL,
    0x0808191919191919UL, 0x0808191919192b08UL, 0x08081919192b0819UL, 0x08081919192b1908UL,
    0x080819192b080808UL, 0x080819192b08082bUL, 0x080819192b081919UL, 0x080819192b082b08UL,
    0x080819192b190819UL, 0x080819192b191908UL, 0x080819192b2b0808UL, 0x0808192b08080819UL,
    0x0808192b08081908UL, 0x0808192b0808192bUL, 0x0808192b08082b19UL, 0x0808192b08190808UL,
    0x0808192b08191919UL, 0x0808192b19080808UL, 0x0808192b19081919UL, 0x0808192b19082b08UL,
    0x0808192b19190819UL, 0x0808192b19191908UL, 0x0808192b192b0808UL, 0x0808192b2b080819UL,
    0x0808192b2b081908UL, 0x0808192b2b190808UL, 0x08082b0808080808UL, 0x08082b080808082bUL,
    0x08082b0808081919UL, 0x08082b0808082b08UL, 0x08082b0808190819UL, 0x08082b0808191908UL,
    0x08082b080819192bUL, 0x08082b0808192b19UL, 0x08082b08082b0808UL, 0x08082b08082b1919UL,
    0x08082b08082b2b2bUL, 0x08082b0819080819UL, 0x08082b0819081908UL, 0x08082b081908192bUL,
    0x08082b0819082b19UL, 0x08082b0819190808UL, 0x08082b081919082bUL, 0x08082b0819191919UL,
    0x08082b0819192b08UL, 0x08082b08192b0819UL, 0x08082b08192b1908UL, 0x08082b082b080808UL,
    0x08082b082b081919UL, 0x08082b082b191908UL, 0x08082b082b2b2b2bUL, 0x08082b1908080819UL,
    0x08082b1908081908UL, 0x08082b1908190808UL, 0x08082b190819082bUL, 0x08082b1908191919UL,
    0x08082b1908192b08UL, 0x08082b19082b0819UL, 0x08082b1919080808UL, 0x08082b1919081919UL,
    0x08082b1919082b08UL, 0x08082b1919190819UL, 0x08082b1919191908UL, 0x08082b19192b0808UL,
    0x08082b192b080819UL, 0x08082b192b190808UL, 0x08082b2b08080808UL, 0x08082b2b08190819UL,
    0x08082b2b08191908UL, 0x08082b2b082b082bUL, 0x08082b2b082b2b08UL, 0x08082b2b082b2b2bUL,
    0x08082b2b19190808UL, 0x08082b2b2b192b19UL, 0x0819080808080819UL, 0x0819080808081908UL,
    0x081908080808192bUL, 0x0819080808082b19UL, 0x0819080808190808UL, 0x081908080819082bUL,
    0x0819080808191919UL, 0x0819080808192b08UL, 0x08190808082b0819UL, 0x08190808082b1908UL,
    0x08190808082b192bUL, 0x0819080819080808UL, 0x081908081908082bUL, 0x0819080819081919UL,
    0x0819080819082b08UL, 0x0819080819190819UL, 0x0819080819191908UL, 0x081908081919192bUL,
    0x0819080819192b19UL, 0x08190808192b0808UL, 0x08190808192b082bUL, 0x08190808192b1919UL,
    0x08190808192b2b08UL, 0x081908082b080819UL, 0x081908082b081908UL, 0x081908082b08192bUL,
    0x081908082b190808UL, 0x081908082b191919UL, 0x081908082b192b08UL, 0x081908082b2b0819UL,
    0x081908082b2b1908UL, 0x0819081908080808UL, 0x081908190808082bUL, 0x0819081908081919UL,
    0x0819081908082b08UL, 0x0819081908082b2bUL, 0x0819081908190819UL, 0x0819081908191908UL,
    0x081908190819192bUL, 0x0819081908192b19UL, 0x08190819082b0808UL, 0x08190819082b082bUL,
    0x08190819082b1919UL, 0x08190819082b2b08UL, 0x0819081919080819UL, 0x0819081919081908UL,
    0x081908191908192bUL, 0x0819081919082b19UL, 0x0819081919190808UL, 0x081908191919082bUL,
    0x0819081919191919UL, 0x0819081919192b08UL, 0x08190819192b0819UL, 0x08190819192b1908UL,
    0x081908192b080808UL, 0x081908192b08082bUL, 0x081908192b081919UL, 0x081908192b082b08UL,
    0x081908192b190819UL, 0x081908192b191908UL, 0x0819082b08080819UL, 0x0819082b08081908UL,
    0x0819082b08082b19UL, 0x0819082b08190808UL, 0x0819082b08191919UL, 0x0819082b082b0819UL,
    0x0819082b082b1908UL, 0x0819082b19080808UL, 0x0819082b19081919UL, 0x0819082b19190819UL,
    0x0819082b19191908UL, 0x0819082b2b080819UL, 0x0819082b2b081908UL, 0x0819082b2b190808UL,
    0x0819190808080808UL, 0x081919080808082bUL, 0x0819190808081919UL, 0x0819190808082b08UL,
    0x0819190808190819UL, 0x0819190808191908UL, 0x081919080819192bUL, 0x0819190808192b19UL,
    0x08191908082b0808UL, 0x08191908082b1919UL, 0x08191908082b2b08UL, 0x0819190819080819UL,
    0x0819190819081908UL, 0x081919081908192bUL, 0x0819190819082b19UL, 0x0819190819190808UL,
    0x081919081919082bUL, 0x0819190819191919UL, 0x0819190819192b08UL, 0x08191908192b0819UL,
    0x08191908192b1908UL, 0x081919082b080808UL, 0x081919082b08082bUL, 0x081919082b081919UL,
    0x081919082b082b08UL, 0x081919082b190819UL, 0x081919082b191908UL, 0x081919082b2b0808UL,
    0x0819191908080819UL, 0x0819191908081908UL, 0x081919190808192bUL, 0x0819191908082b19UL,
    0x0819191908190808UL, 0x081919190819082bUL, 0x0819191908191919UL, 0x0819191908192b08UL,
    0x08191919082b0819UL, 0x08191919082b1908UL, 0x0819191919080808UL, 0x081919191908082bUL,
    0x0819191919081919UL, 0x0819191919082b08UL, 0x0819191919190819UL, 0x0819191919191908UL,
    0x08191919192b0808UL, 0x081919192b080819UL, 0x081919192b081908UL, 0x081919192b190808UL,
    0x0819192b08080808UL, 0x0819192b08081919UL, 0x0819192b08082b08UL, 0x0819192b08190819UL,
    0x0819192b08191908UL, 0x0819192b082b0808UL, 0x0819192b19080819UL, 0x0819192b19081908UL,
    0x0819192b19190808UL, 0x0819192b2b080808UL, 0x0819192b2b2b2b2bUL, 0x08192b0808080819UL,
    0x08192b0808081908UL, 0x08192b080808192bUL, 0x08192b0808082b19UL, 0x08192b0808190808UL,
    0x08192b0808191919UL, 0x08192b0808192b08UL, 0x08192b08082b0819UL, 0x08192b0819080808UL,
    0x08192b081908082bUL, 0x08192b0819081919UL, 0x08192b0819082b08UL, 0x08192b0819190819UL,
    0x08192b0819191908UL, 0x08192b08192b0808UL, 0x08192b082b080819UL, 0x08192b082b081908UL,
    0x08192b1908080808UL, 0x08192b190808082bUL, 0x08192b1908081919UL, 0x08192b1908082b08UL,
    0x08192b1908190819UL, 0x08192b1908191908UL, 0x08192b19082b0808UL, 0x08192b1919080819UL,
    0x08192b1919081908UL, 0x08192b1919190808UL, 0x08192b19192b2b19UL, 0x08192b192b2b082bUL,
    0x08192b2b08081908UL, 0x08192b2b08190808UL, 0x08192b2b19080808UL, 0x08192b2b1919192bUL,
    0x082b080808080808UL, 0x082b08080808082bUL, 0x082b080808081919UL, 0x082b080808082b08UL,
    0x082b080808190819UL, 0x082b080808191908UL, 0x082b08080819192bUL, 0x082b080808192b19UL,
    0x082b0808082b0808UL, 0x082b0808082b1919UL, 0x082b0808082b2b2bUL, 0x082b080819080819UL,
    0x082b080819081908UL, 0x082b080819190808UL, 0x082b08081919082bUL, 0x082b080819191919UL,
    0x082b0808192b1908UL, 0x082b08082b080808UL, 0x082b08082b082b2bUL, 0x082b08082b191908UL,
    0x082b08082b2b2b2bUL, 0x082b081908080819UL, 0x082b081908081908UL, 0x082b081908190808UL,
    0x082b08190819082bUL, 0x082b081908191919UL, 0x082b0819082b0819UL, 0x082b081919080808UL,
    0x082b08191908082bUL, 0x082b081919081919UL, 0x082b081919190819UL, 0x082b081919191908UL,
    0x082b0819192b0808UL, 0x082b08192b080819UL, 0x082b08192b081908UL, 0x082b08192b190808UL,
    0x082b082b08080808UL, 0x082b082b08082b2bUL, 0x082b082b082b082bUL, 0x082b082b082b2b08UL,
    0x082b082b082b2b2bUL, 0x082b082b19081908UL, 0x082b082b19190808UL, 0x082b082b2b082b08UL,
    0x082b082b2b082b2bUL, 0x082b082b2b2b2b08UL, 0x082b190808080819UL, 0x082b190808081908UL,
    0x082b19080808192bUL, 0x082b190808082b19UL, 0x082b190808190808UL, 0x082b190808191919UL,
    0x082b190808192b08UL, 0x082b1908082b0819UL, 0x082b1908082b1908UL, 0x082b190819080808UL,
    0x082b19081908082bUL, 0x082b190819081919UL, 0x082b190819082b08UL, 0x082b190819190819UL,
    0x082b190819191908UL, 0x082b1908192b0808UL, 0x082b19082b080819UL, 0x082b19082b081908UL,
    0x082b19082b190808UL, 0x082b191908080808UL, 0x082b191908081919UL, 0x082b191908082b08UL,
    0x082b191908190819UL, 0x082b191908191908UL, 0x082b1919082b0808UL, 0x082b191919080819UL,
    0x082b191919081908UL, 0x082b191919190808UL, 0x082b1919192b192bUL, 0x082b19192b080808UL,
    0x082b192b08080819UL, 0x082b192b08081908UL, 0x082b192b08190808UL, 0x082b192b19080808UL,
    0x082b192b19192b19UL, 0x082b2b0808080808UL, 0x082b2b0808081919UL, 0x082b2b0808190819UL,
    0x082b2b0808191908UL, 0x082b2b0819080819UL, 0x082b2b0819081908UL, 0x082b2b0819190808UL,
    0x082b2b082b082b2bUL, 0x082b2b082b2b2b2bUL, 0x082b2b1908080819UL, 0x082b2b1908081908UL,
    0x082b2b1908190808UL, 0x082b2b192b191919UL, 0x082b2b2b08082b2bUL, 0x082b2b2b082b082bUL,
    0x082b2b2b192b1908UL, 0x082b2b2b2b082b08UL, 0x082b2b2b2b082b2bUL, 0x1908080808080819UL,
    0x1908080808081908UL, 0x190808080808192bUL, 0x1908080808082b19UL, 0x1908080808190808UL,
    0x190808080819082bUL, 0x1908080808191919UL, 0x1908080808192b08UL, 0x1908080808192b2bUL,
    0x19080808082b0819UL, 0x19080808082b1908UL, 0x19080808082b192bUL, 0x1908080819080808UL,
    0x190808081908082bUL, 0x1908080819081919UL, 0x1908080819082b08UL, 0x1908080819082b2bUL,
    0x1908080819190819UL, 0x1908080819191908UL, 0x190808081919192bUL, 0x1908080819192b19UL,
    0x19080808192b0808UL, 0x19080808192b082bUL, 0x19080808192b1919UL, 0x190808082b080819UL,
    0x190808082b081908UL, 0x190808082b190808UL, 0x190808082b191919UL, 0x190808082b192b08UL,
    0x190808082b2b0819UL, 0x190808082b2b1908UL, 0x1908081908080808UL, 0x190808190808082bUL,
    0x1908081908081919UL, 0x1908081908082b08UL, 0x1908081908190819UL, 0x1908081908191908UL,
    0x190808190819192bUL, 0x1908081908192b19UL, 0x19080819082b0808UL, 0x19080819082b082bUL,
    0x19080819082b1919UL, 0x1908081919080819UL, 0x1908081919081908UL, 0x190808191908192bUL,
    0x1908081919082b19UL, 0x1908081919190808UL, 0x190808191919082bUL, 0x1908081919191919UL,
    0x1908081919192b08UL, 0x19080819192b0819UL, 0x19080819192b1908UL, 0x190808192b080808UL,
    0x190808192b08082bUL, 0x190808192b081919UL, 0x190808192b082b08UL, 0x190808192b190819UL,
    0x190808192b191908UL, 0x190808192b2b0808UL, 0x1908082b08080819UL, 0x1908082b08081908UL,
    0x1908082b08190808UL, 0x1908082b0819082bUL, 0x1908082b08191919UL, 0x1908082b08192b08UL,
    0x1908082b082b1908UL, 0x1908082b19080808UL, 0x1908082b19081919UL, 0x1908082b19082b08UL,
    0x1908082b19190819UL, 0x1908082b19191908UL, 0x1908082b192b0808UL, 0x1908082b2b080819UL,
    0x1908082b2b081908UL, 0x1908190808080808UL, 0x190819080808082bUL, 0x1908190808081919UL,
    0x1908190808082b08UL, 0x1908190808082b2bUL, 0x1908190808190819UL, 0x1908190808191908UL,
    0x190819080819192bUL, 0x1908190808192b19UL, 0x19081908082b0808UL, 0x19081908082b082bUL,
    0x19081908082b1919UL, 0x19081908082b2b08UL, 0x1908190819080819UL, 0x1908190819081908UL,
    0x190819081908192bUL, 0x1908190819082b19UL, 0x1908190819190808UL, 0x190819081919082bUL,
    0x1908190819191919UL, 0x1908190819192b08UL, 0x19081908192b0819UL, 0x19081908192b1908UL,
    0x190819082b080808UL, 0x190819082b08082bUL, 0x190819082b081919UL, 0x190819082b082b08UL,
    0x190819082b190819UL, 0x190819082b191908UL, 0x190819082b2b0808UL, 0x1908191908080819UL,
    0x1908191908081908UL, 0x190819190808192bUL, 0x1908191908082b19UL, 0x1908191908190808UL,
    0x190819190819082bUL, 0x1908191908191919UL, 0x1908191908192b08UL, 0x19081919082b0819UL,
    0x19081919082b1908UL, 0x1908191919080808UL, 0x190819191908082bUL, 0x1908191919081919UL,
    0x1908191919082b08UL, 0x1908191919190819UL, 0x1908191919191908UL, 0x19081919192b0808UL,
    0x19081919192b2b2bUL, 0x190819192b080819UL, 0x190819192b081908UL, 0x190819192b190808UL,
    0x1908192b08080808UL, 0x1908192b0808082bUL, 0x1908192b08081919UL, 0x1908192b08082b08UL,
    0x1908192b08190819UL, 0x1908192b08191908UL, 0x1908192b082b0808UL, 0x1908192b19080819UL,
    0x1908192b19081908UL, 0x1908192b19190808UL, 0x1908192b2b080808UL, 0x1908192b2b2b1919UL,
    0x19082b0808080819UL, 0x19082b0808081908UL, 0x19082b0808082b19UL, 0x19082b0808190808UL,
    0x19082b080819082bUL, 0x19082b0808191919UL, 0x19082b0808192b08UL, 0x19082b08082b0819UL,
    0x19082b08082b1908UL, 0x19082b0819080808UL, 0x19082b081908082bUL, 0x19082b0819081919UL,
    0x19082b0819082b08UL, 0x19082b0819190819UL, 0x19082b0819191908UL, 0x19082b08192b0808UL,
    0x19082b082b081908UL, 0x19082b082b190808UL, 0x19082b1908080808UL, 0x19082b190808082bUL,
    0x19082b1908081919UL, 0x19082b1908082b08UL, 0x19082b1908190819UL, 0x19082b1908191908UL,
    0x19082b19082b0808UL, 0x19082b1919080819UL, 0x19082b1919081908UL, 0x19082b1919190808UL,
    0x19082b192b080808UL, 0x19082b192b19192bUL, 0x19082b2b08080819UL, 0x19082b2b08081908UL,
    0x19082b2b08190808UL, 0x19082b2b19080808UL, 0x1919080808080808UL, 0x191908080808082bUL,
    0x1919080808081919UL, 0x1919080808082b08UL, 0x1919080808190819UL, 0x1919080808191908UL,
    0x191908080819192bUL, 0x1919080808192b19UL, 0x19190808082b0808UL, 0x19190808082b082bUL,
    0x19190808082b1919UL, 0x19190808082b2b08UL, 0x1919080819080819UL, 0x1919080819081908UL,
    0x191908081908192bUL, 0x1919080819082b19UL, 0x1919080819190808UL, 0x191908081919082bUL,
    0x1919080819191919UL, 0x1919080819192b08UL, 0x19190808192b0819UL, 0x19190808192b1908UL,
    0x191908082b080808UL, 0x191908082b08082bUL, 0x191908082b081919UL, 0x191908082b082b08UL,
    0x191908082b190819UL, 0x191908082b191908UL, 0x1919081908080819UL, 0x1919081908081908UL,
    0x191908190808192bUL, 0x1919081908082b19UL, 0x1919081908190808UL, 0x191908190819082bUL,
    0x1919081908191919UL, 0x1919081908192b08UL, 0x19190819082b0819UL, 0x19190819082b1908UL,
    0x1919081919080808UL, 0x191908191908082bUL, 0x1919081919081919UL, 0x1919081919082b08UL,
    0x1919081919190819UL, 0x1919081919191908UL, 0x19190819192b0808UL, 0x191908192b080819UL,
    0x191908192b081908UL, 0x191908192b190808UL, 0x1919082b08080808UL, 0x1919082b08081919UL,
    0x1919082b08082b08UL, 0x1919082b08190819UL, 0x1919082b08191908UL, 0x1919082b082b0808UL,
    0x1919082b19080819UL, 0x1919082b19081908UL, 0x1919082b19190808UL, 0x1919082b192b2b19UL,
    0x1919082b2b080808UL, 0x1919190808080819UL, 0x1919190808081908UL, 0x191919080808192bUL,
    0x1919190808082b19UL, 0x1919190808190808UL, 0x191919080819082bUL, 0x1919190808191919UL,
    0x1919190808192b08UL, 0x19191908082b0819UL, 0x19191908082b1908UL, 0x1919190819080808UL,
    0x191919081908082bUL, 0x1919190819081919UL, 0x1919190819082b08UL, 0x1919190819190819UL,
    0x1919190819191908UL, 0x19191908192b0808UL, 0x191919082b080819UL, 0x191919082b081908UL,
    0x191919082b190808UL, 0x1919191908080808UL, 0x191919190808082bUL, 0x1919191908081919UL,
    0x1919191908082b08UL, 0x1919191908190819UL, 0x1919191908191908UL, 0x19191919082b0808UL,
    0x1919191919080819UL, 0x1919191919081908UL, 0x1919191919190808UL, 0x191919192b080808UL,
    0x1919192b08080819UL, 0x1919192b08081908UL, 0x1919192b08190808UL, 0x1919192b082b192bUL,
    0x1919192b19080808UL, 0x19192b0808080808UL, 0x19192b080808082bUL, 0x19192b0808081919UL,
    0x19192b0808082b08UL, 0x19192b0808190819UL, 0x19192b0808191908UL, 0x19192b08082b0808UL,
    0x19192b0819080819UL, 0x19192b0819081908UL, 0x19192b0819190808UL, 0x19192b0819192b2bUL,
    0x19192b082b080808UL, 0x19192b1908080819UL, 0x19192b1908081908UL, 0x19192b1908190808UL,
    0x19192b1919080808UL, 0x19192b2b08080808UL, 0x19192b2b08192b19UL, 0x19192b2b2b081919UL,
    0x19192b2b2b2b2b08UL, 0x192b080808080819UL, 0x192b080808081908UL, 0x192b08080808192bUL,
    0x192b080808190808UL, 0x192b08080819082bUL, 0x192b080808191919UL, 0x192b080808192b08UL,
    0x192b0808082b0819UL, 0x192b0808082b1908UL, 0x192b080819080808UL, 0x192b080819081919UL,
    0x192b080819082b08UL, 0x192b080819190819UL, 0x192b080819191908UL, 0x192b0808192b0808UL,
    0x192b08082b081908UL, 0x192b08082b190808UL, 0x192b081908080808UL, 0x192b08190808082bUL,
    0x192b081908081919UL, 0x192b081908082b08UL, 0x192b081908190819UL, 0x192b081908191908UL,
    0x192b0819082b0808UL, 0x192b081919080819UL, 0x192b081919081908UL, 0x192b081919190808UL,
    0x192b08192b080808UL, 0x192b08192b192b19UL, 0x192b082b08081908UL, 0x192b082b08190808UL,
    0x192b082b19080808UL, 0x192b082b1919192bUL, 0x192b082b2b2b0819UL, 0x192b190808080808UL,
    0x192b190808081919UL, 0x192b190808082b08UL, 0x192b190808190819UL, 0x192b190808191908UL,
    0x192b1908082b0808UL, 0x192b190819080819UL, 0x192b190819081908UL, 0x192b190819190808UL,
    0x192b19082b080808UL, 0x192b191908080819UL, 0x192b191908081908UL, 0x192b191908190808UL,
    0x192b191919080808UL, 0x192b191919082b2bUL, 0x192b1919192b2b08UL, 0x192b19192b19082bUL,
    0x192b192b08080808UL, 0x192b192b2b191908UL, 0x192b2b0808080819UL, 0x192b2b0808081908UL,
    0x192b2b0808190808UL, 0x192b2b08192b1919UL, 0x192b2b082b192b08UL, 0x192b2b1908080808UL,
    0x192b2b19082b2b2bUL, 0x192b2b2b1908082bUL, 0x192b2b2b2b2b0819UL, 0x2b08080808080808UL,
    0x2b0808080808082bUL, 0x2b08080808081919UL, 0x2b08080808082b08UL, 0x2b08080808190819UL,
    0x2b08080808191908UL, 0x2b08080808192b19UL, 0x2b080808082b0808UL, 0x2b080808082b1919UL,
    0x2b08080819080819UL, 0x2b08080819081908UL, 0x2b08080819190808UL, 0x2b0808081919082bUL,
    0x2b08080819191919UL, 0x2b08080819192b08UL, 0x2b080808192b0819UL, 0x2b0808082b080808UL,
    0x2b0808082b081919UL, 0x2b0808082b190819UL, 0x2b0808082b191908UL, 0x2b08081908080819UL,
    0x2b08081908081908UL, 0x2b08081908082b19UL, 0x2b08081908190808UL, 0x2b0808190819082bUL,
    0x2b08081908191919UL, 0x2b08081908192b08UL, 0x2b080819082b0819UL, 0x2b080819082b1908UL,
    0x2b08081919080808UL, 0x2b0808191908082bUL, 0x2b08081919081919UL, 0x2b08081919082b08UL,
    0x2b08081919190819UL, 0x2b08081919191908UL, 0x2b0808192b080819UL, 0x2b0808192b081908UL,
    0x2b0808192b190808UL, 0x2b0808192b2b2b19UL, 0x2b08082b08080808UL, 0x2b08082b08081919UL,
    0x2b08082b08082b2bUL, 0x2b08082b08190819UL, 0x2b08082b08191908UL, 0x2b08082b19080819UL,
    0x2b08082b19081908UL, 0x2b08082b19190808UL, 0x2b08190808080819UL, 0x2b08190808081908UL,
    0x2b0819080808192bUL, 0x2b08190808082b19UL, 0x2b08190808190808UL, 0x2b0819080819082bUL,
    0x2b08190808191919UL, 0x2b08190808192b08UL, 0x2b081908082b0819UL, 0x2b08190819080808UL,
    0x2b0819081908082bUL, 0x2b08190819081919UL, 0x2b08190819082b08UL, 0x2b08190819190819UL,
    0x2b08190819191908UL, 0x2b081908192b0808UL, 0x2b0819082b080819UL, 0x2b0819082b081908UL,
    0x2b0819082b190808UL, 0x2b08191908080808UL, 0x2b0819190808082bUL, 0x2b08191908081919UL,
    0x2b08191908082b08UL, 0x2b08191908190819UL, 0x2b08191908191908UL, 0x2b081919082b0808UL,
    0x2b08191919080819UL, 0x2b08191919081908UL, 0x2b08191919190808UL, 0x2b0819192b080808UL,
    0x2b0819192b082b2bUL, 0x2b08192b08080819UL, 0x2b08192b08081908UL, 0x2b08192b08190808UL,
    0x2b08192b082b2b19UL, 0x2b08192b19080808UL, 0x2b082b0808080808UL, 0x2b082b0808081919UL,
    0x2b082b0808190819UL, 0x2b082b0808191908UL, 0x2b082b0819080819UL, 0x2b082b0819081908UL,
    0x2b082b0819190808UL, 0x2b082b082b2b082bUL, 0x2b082b1908080819UL, 0x2b082b1908081908UL,
    0x2b082b1919080808UL, 0x2b082b19192b1919UL, 0x2b082b2b082b082bUL, 0x2b082b2b19192b08UL,
    0x2b082b2b19192b2bUL, 0x2b082b2b2b08082bUL, 0x2b082b2b2b2b082bUL, 0x2b19080808080819UL,
    0x2b19080808081908UL, 0x2b19080808082b19UL, 0x2b19080808190808UL, 0x2b1908080819082bUL,
    0x2b19080808191919UL, 0x2b19080808192b08UL, 0x2b190808082b1908UL, 0x2b19080819080808UL,
    0x2b1908081908082bUL, 0x2b19080819081919UL, 0x2b19080819082b08UL, 0x2b19080819190819UL,
    0x2b19080819191908UL, 0x2b190808192b0808UL, 0x2b1908082b080819UL, 0x2b1908082b081908UL,
    0x2b1908082b190808UL, 0x2b19081908080808UL, 0x2b19081908081919UL, 0x2b19081908190819UL,
    0x2b19081908191908UL, 0x2b19081919080819UL, 0x2b19081919081908UL, 0x2b19081919190808UL,
    0x2b19081919192b2bUL, 0x2b19082b08080819UL, 0x2b19082b08081908UL, 0x2b19082b08190808UL,
    0x2b19082b19080808UL, 0x2b19082b2b2b192bUL, 0x2b19190808080808UL, 0x2b1919080808082bUL,
    0x2b19190808081919UL, 0x2b19190808082b08UL, 0x2b19190808190819UL, 0x2b19190808191908UL,
    0x2b191908082b0808UL, 0x2b19190819080819UL, 0x2b19190819081908UL, 0x2b19190819190808UL,
    0x2b1919082b080808UL, 0x2b1919082b19192bUL, 0x2b19191908080819UL, 0x2b19191908081908UL,
    0x2b19191908190808UL, 0x2b19191919080808UL, 0x2b1919192b192b08UL, 0x2b1919192b2b0819UL,
    0x2b19192b08080808UL, 0x2b19192b1908192bUL, 0x2b19192b192b1908UL, 0x2b192b0808080819UL,
    0x2b192b0808081908UL, 0x2b192b0808190808UL, 0x2b192b08082b192bUL, 0x2b192b0819080808UL,
    0x2b192b082b2b2b19UL, 0x2b192b1908080808UL, 0x2b192b1919082b19UL, 0x2b192b191919082bUL,
    0x2b192b2b2b190808UL, 0x2b2b080808080808UL, 0x2b2b080808081919UL, 0x2b2b080808082b2bUL,
    0x2b2b080808191908UL, 0x2b2b0808082b082bUL, 0x2b2b0808082b2b2bUL, 0x2b2b080819080819UL,
    0x2b2b080819081908UL, 0x2b2b080819190808UL, 0x2b2b08082b2b082bUL, 0x2b2b08082b2b2b2bUL,
    0x2b2b081919080808UL, 0x2b2b0819192b1919UL, 0x2b2b082b0808082bUL, 0x2b2b082b08082b2bUL,
    0x2b2b082b082b082bUL, 0x2b2b082b082b2b08UL, 0x2b2b082b082b2b2bUL, 0x2b2b082b2b08082bUL,
    0x2b2b082b2b082b08UL, 0x2b2b082b2b082b2bUL, 0x2b2b082b2b2b2b08UL, 0x2b2b190808080819UL,
    0x2b2b190808081908UL, 0x2b2b190808190808UL, 0x2b2b190819080808UL, 0x2b2b19082b082b19UL,
    0x2b2b19082b2b1908UL, 0x2b2b191908080808UL, 0x2b2b191908192b19UL, 0x2b2b192b19190819UL,
    0x2b2b2b0808082b2bUL, 0x2b2b2b08082b2b08UL, 0x2b2b2b082b2b082bUL, 0x2b2b2b1919191908UL,
    0x2b2b2b192b08192bUL, 0x2b2b2b2b08082b08UL, 0x2b2b2b2b08082b2bUL, 0x2b2b2b2b082b0808UL,
    0x2b2b2b2b082b082bUL, 0x2b2b2b2b082b2b08UL, 0x2b2b2b2b2b082b08UL, 0x2b2b2b2b2b2b2b2bUL,
};

inline uchar hayai_grid8_byte(ulong g, int j) {
    return (uchar)((g >> (8*j)) & 0xff);
}

inline float hayai_iq2_signed(float db, uchar g, uchar signs, int j) {
    float s = (signs & hayai_kmask_iq2xs[j]) ? -1.0f : 1.0f;
    return db * (float)g * s;
}

__kernel void ggml_gemv_iq2_xxs(
    const int M, const int N, const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
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
    const int M, const int N, const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
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
    const int M, const int N, const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
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

// BF16: high 16 bits of an IEEE-754 float32.
__kernel void ggml_gemv_bf16(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    __global const uchar* wbase = weights + weight_off + row * N * 2;
    float sum = 0.0f;
    for (int c = 0; c < N; c++) {
        uint bits = (uint)wbase[c * 2] | ((uint)wbase[c * 2 + 1] << 8);
        sum += as_float(bits << 16) * input[c];
    }
    output[row] = sum;
}

// Q8_1: block = { half d; half s; int8 qs[32] } = 36 bytes / 32 elems.
__kernel void ggml_gemv_q8_1(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    int blocks = N / 32;
    int row_bytes = blocks * 36;
    __global const uchar* wbase = weights + weight_off + row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + bi * 36;
        float d = hayai_half_bits_to_float((ushort)block[0] | ((ushort)block[1] << 8));
        __global const uchar* qs = block + 4;
        int x_base = bi * 32;
        for (int j = 0; j < 32; j++) {
            sum += (float)((char)qs[j]) * d * input[x_base + j];
        }
    }
    output[row] = sum;
}

// Q8_K: block = { float d; int8 qs[256]; int16 bsums[16] } = 292 bytes / 256 elems.
__kernel void ggml_gemv_q8_k(
    const int M,
    const int N,
    const long weight_off,
    __global const uchar* restrict weights,
    __global const float* restrict input,
    __global float* restrict output,
    __local float* restrict local_input
) {
    int row = get_global_id(0);
    hayai_wg_barrier(local_input);
    if (row >= M) return;
    int blocks = N / 256;
    int row_bytes = blocks * 292;
    __global const uchar* wbase = weights + weight_off + row * row_bytes;
    float sum = 0.0f;
    for (int bi = 0; bi < blocks; bi++) {
        __global const uchar* block = wbase + bi * 292;
        uint db = (uint)block[0] | ((uint)block[1] << 8) | ((uint)block[2] << 16) | ((uint)block[3] << 24);
        float d = as_float(db);
        __global const uchar* qs = block + 4;
        int x_base = bi * 256;
        for (int j = 0; j < 256; j++) {
            sum += (float)((char)qs[j]) * d * input[x_base + j];
        }
    }
    output[row] = sum;
}
