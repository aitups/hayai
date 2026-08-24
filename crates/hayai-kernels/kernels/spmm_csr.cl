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
