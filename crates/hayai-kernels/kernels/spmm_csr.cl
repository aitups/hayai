// SpMM CSR para el FFN disperso (DAG irregular, GGUF de `saor`).
// OpenCL C 1.2: un work-item por par (b, j); el bucle recorre solo los no-cero.
// Y[b][j] = sum_k X[b][col_idx[k]] * vals[k], para k in [row_ptr[j], row_ptr[j+1]).

__kernel void spmm_csr(
    __global const float* x,     // B * d_in
    __global const int* row_ptr, // d_out + 1
    __global const int* col_idx, // nnz
    __global const float* vals,  // nnz
    const int d_in,
    const int d_out,
    __global float* y)           // B * d_out
{
    const int gid = get_global_id(0);
    const int b = gid / d_out;
    const int j = gid % d_out;
    float acc = 0.0f;
    for (int k = row_ptr[j]; k < row_ptr[j + 1]; k++) {
        acc += x[b * d_in + col_idx[k]] * vals[k];
    }
    y[b * d_out + j] = acc;
}

// SpMM esparso batcheado desde bit-tensor (SparseAdj) + pesos F32 compartidos por
// capa (Fase 2, criterio C4): N candidatos x n_pos tokens en un unico dispatch.
// Un work-item por (b, j); el bucle recorre TODOS los bits de la fila j del
// candidato (conn = i*d_out+j, LSB-first) acumulando solo los activos. Los pesos
// se comparten (dequant UNA vez por capa) y se evita el gather del CSR por
// (candidato, capa, token) que dominaba el tiempo en 27B/40B.
__kernel void spmm_adj_batched(
    __global const float* x,       // N * n_pos * d_in
    __global const uchar* adjs,    // N * (d_in*d_out/8)
    __global const float* w,       // d_out * d_in (F32 compartido por capa)
    const int d_in,
    const int d_out,
    const int n_pos,               // tokens por candidato
    __global float* y)             // N * n_pos * d_out
{
    const int gid = get_global_id(0);
    const int b = gid / d_out;
    const int j = gid % d_out;
    const int c = b / n_pos;
    const __global uchar* adj = adjs + (size_t)c * (((size_t)d_in * (size_t)d_out) >> 3);
    float acc = 0.0f;
    for (int i = 0; i < d_in; i++) {
        const int conn = i * d_out + j;
        if (adj[conn >> 3] & (1 << (conn & 7))) {
            acc += x[b * d_in + i] * w[j * d_in + i];
        }
    }
    y[b * d_out + j] = acc;
}

// IEEE 754 half -> float (OpenCL C 1.2 no garantiza fp16).
inline float f16_to_f32(unsigned short h) {
    const unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    const unsigned int exp = (h >> 10) & 0x1Fu;
    const unsigned int mant = h & 0x3FFu;
    union { unsigned int u; float f; } c;
    if (exp == 0u) {
        if (mant == 0u) { c.u = sign; return c.f; }
        // subnormal: mismo algoritmo que hayai_model::gguf::f16_to_f32
        unsigned int m = mant;
        int e = 113; // 127 - 15 + 1
        while (!(m & 0x400u)) { m <<= 1; e--; }
        m &= 0x3FFu;
        c.u = sign | (((unsigned int)e) << 23) | (m << 13);
        return c.f;
    }
    if (exp == 31u) {
        return mant == 0u ? (sign ? -INFINITY : INFINITY) : (sign ? -NAN : NAN);
    }
    c.u = sign | (((exp + 127u - 15u)) << 23) | (mant << 13);
    return c.f;
}

// Dequant de UN valor Q4_K en fila-mayor [d_out, d_in]: bloque de 256 valores
// (144 B: d f16, min f16, 12 B de escalas, 128 B de nibbles).
inline float q4k_val(const __global uchar* w4, long pos, int d_in) {
    const long block = pos / 256;
    const int off = (int)(pos % 256);
    const int sub = off >> 5;          // 0..7 sub-bloques de 32
    const int l = off & 31;
    const __global uchar* b = w4 + block * 144;
    const float d = f16_to_f32((unsigned short)(b[0] | (b[1] << 8)));
    const float minv = f16_to_f32((unsigned short)(b[2] | (b[3] << 8)));
    int sc, m;
    if (sub < 4) {
        sc = b[4 + sub] & 63;
        m = b[8 + sub] & 63;
    } else {
        // Rust get_scale_min_k4(j=sub): d=(scales[sub+4]&0x0F)|((scales[sub-4]>>6)<<4);
        //                              m=(scales[sub+4]>>4)|((scales[sub]>>6)<<4)
        // scales[k] = b[4+k].
        sc = (b[8 + sub] & 0x0Fu) | ((b[sub] >> 6) << 4);
        m = (b[8 + sub] >> 4) | ((b[4 + sub] >> 6) << 4);
    }
    const int qb = (sub >> 1) * 32 + l;
    const int nib = (sub & 1) ? (b[16 + qb] >> 4) : (b[16 + qb] & 0x0Fu);
    return d * (float)sc * (float)nib - minv * (float)m;
}

// SpMM esparso batcheado con dequant Q4_K en el kernel (Fase 2, criterio C1/C4):
// elimina el dequant F32 + upload de ~23 GB/gen del path SparseAdj-F32; el Q4 ya
// está en el streaming y el trabajo de dequant pasa a la GPU.
__kernel void spmm_adj_batched_q4(
    __global const float* x,       // N*n_pos*d_in
    __global const uchar* adjs,    // N*(d_in*d_out/8)
    __global const uchar* w4,      // Q4_K [d_out*d_in] (144 B/256 val)
    const int d_in,
    const int d_out,
    const int n_pos,
    __global float* y)             // N*n_pos*d_out
{
    const int gid = get_global_id(0);
    const int b = gid / d_out;
    const int j = gid % d_out;
    const int c = b / n_pos;
    const __global uchar* adj = adjs + (size_t)c * (((size_t)d_in * (size_t)d_out) >> 3);
    const long row_base = (long)j * d_in;
    float acc = 0.0f;
    for (int i = 0; i < d_in; i++) {
        const int conn = i * d_out + j;
        if (adj[conn >> 3] & (1 << (conn & 7))) {
            acc += x[b * d_in + i] * q4k_val(w4, row_base + i, d_in);
        }
    }
    y[b * d_out + j] = acc;
}

// Dequant Q4_K -> F32 batcheado (un work-item por bloque de 256): rellena el
// buffer F32 que consume `spmm_adj_batched` — se encadena en la MISMA cola tras el
// upload del Q4 (8× menos PCIe que el F32) y antes del SpMM (Fase 2, C1/C4).
__kernel void dequant_q4_k_to_f32(
    __global const uchar* w4,      // Q4_K [n] packed (144 B/256 val)
    const int n,                   // total de elementos (múltiplo de 256)
    __global float* out)           // n floats
{
    const int gid = get_global_id(0);
    const long block = (long)gid * 144;
    const __global uchar* b = w4 + block;
    const float d = f16_to_f32((unsigned short)(b[0] | (b[1] << 8)));
    const float minv = f16_to_f32((unsigned short)(b[2] | (b[3] << 8)));
    const __global uchar* scales = b + 4;
    const __global uchar* q = b + 16;
    long yo = (long)gid * 256;
    for (int g = 0; g < 4; g++) {
        const int sub0 = 2 * g;
        const int sub1 = sub0 + 1;
        int sc0, m0, sc1, m1;
        if (sub0 < 4) { sc0 = scales[sub0] & 63; m0 = scales[sub0 + 4] & 63; }
        else { sc0 = (scales[sub0 + 4] & 0x0Fu) | ((scales[sub0 - 4] >> 6) << 4); m0 = (scales[sub0 + 4] >> 4) | ((scales[sub0] >> 6) << 4); }
        if (sub1 < 4) { sc1 = scales[sub1] & 63; m1 = scales[sub1 + 4] & 63; }
        else { sc1 = (scales[sub1 + 4] & 0x0Fu) | ((scales[sub1 - 4] >> 6) << 4); m1 = (scales[sub1 + 4] >> 4) | ((scales[sub1] >> 6) << 4); }
        const float d1 = d * (float)sc0;
        const float m1v = minv * (float)m0;
        const float d2 = d * (float)sc1;
        const float m2v = minv * (float)m1;
        for (int l = 0; l < 32; l++) {
            out[yo + l] = d1 * (float)(q[l] & 0x0Fu) - m1v;
            out[yo + 32 + l] = d2 * (float)(q[l] >> 4) - m2v;
        }
        q += 32;
        yo += 64;
    }
}
