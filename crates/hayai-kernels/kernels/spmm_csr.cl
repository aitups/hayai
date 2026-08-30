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
