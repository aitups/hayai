// OpenCL 3.0 Custom Kernel for LUT (Look-Up Table) Quantized Matrix Multiplication
// Implements Shift & Add / Dictionary lookup in __local memory for maximum bandwidth efficiency.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// 4-Bit LUT MatMul Kernel:
// M: Rows of weight matrix (e.g. output dim)
// N: Columns of weight matrix (e.g. input dim)
// K: Batch/Vector size (typically 1 for token generation)
// weights_q4: Packed 4-bit indices (2 indices per byte)
// lut_table: Dictionary of 16 float values per block/row
// input_vec: FP32 activation vector (length N)
// output_vec: FP32 output vector (length M)
__kernel void lut_matmul_q4_v1(
    const int M,
    const int N,
    __global const uchar* restrict weights_q4,
    __global const float* restrict lut_table,
    __global const float* restrict input_vec,
    __global float* restrict output_vec,
    __local float* local_lut
) {
    int row = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);

    // Load LUT (16 entries) into local work-group memory
    if (local_id < 16) {
        local_lut[local_id] = lut_table[local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (row >= M) return;

    float sum = 0.0f;
    int packed_cols = N / 2;
    int row_offset = row * packed_cols;

    for (int col = 0; col < packed_cols; col++) {
        uchar packed_val = weights_q4[row_offset + col];
        uchar idx0 = packed_val & 0x0F;
        uchar idx1 = (packed_val >> 4) & 0x0F;

        float w0 = local_lut[idx0];
        float w1 = local_lut[idx1];

        float x0 = input_vec[col * 2];
        float x1 = input_vec[col * 2 + 1];

        sum += x0 * w0 + x1 * w1;
    }

    output_vec[row] = sum;
}
