// Activation / norm kernels so the FFN can stay on-device (audit B.2 #3): the gate
// and up GEMVs write device buffers, `hayai_silu_mul` combines them in place, and the
// down GEMV reads the result without a host round-trip. `hayai_rms_norm` is the
// matching normalization pass. Written in the OpenCL C 1.2 subset.

__kernel void hayai_silu_mul(
    __global const float* restrict g,   // gate GEMV output
    __global const float* restrict u,   // up GEMV output
    __global float* restrict out,       // may alias `g`
    const int n)
{
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    float v = g[i];
    out[i] = (v / (1.0f + exp(-v))) * u[i];
}

// Row-wise RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w. One work-group per call;
// the row is strided across lanes and reduced through `red` (size = local size).
__kernel void hayai_rms_norm(
    __global const float* restrict x,
    __global const float* restrict w,
    __global float* restrict y,
    const int n,
    const float eps,
    __local float* restrict red)
{
    int lid = (int)get_local_id(0);
    int ls = (int)get_local_size(0);
    float s = 0.0f;
    for (int i = lid; i < n; i += ls) {
        float v = x[i];
        s += v * v;
    }
    red[lid] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = ls >> 1; off > 0; off >>= 1) {
        if (lid < off) {
            red[lid] += red[lid + off];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float inv = rsqrt(red[0] / (float)n + eps);
    for (int i = lid; i < n; i += ls) {
        y[i] = x[i] * inv * w[i];
    }
}
// Residual add: x += y (in place on x). Keeps the residual stream on-device.
__kernel void hayai_residual_add(
    __global float* restrict x,
    __global const float* restrict y,
    const int n)
{
    int i = (int)get_global_id(0);
    if (i >= n) {
        return;
    }
    x[i] += y[i];
}

// Rotary position embedding (GPT-NeoX / llama pair layout): rotate (x[2i], x[2i+1])
// with the per-position cos/sin table. `n_rows` rows of `head_dim`; only the first
// `rotary_dim` columns are rotated (partial RoPE).
__kernel void hayai_rope_apply(
    __global float* restrict x,
    const int n_rows,
    const int head_dim,
    const int rotary_dim,
    const int pos,
    __global const float* restrict cosv,
    __global const float* restrict sinv)
{
    int idx = (int)get_global_id(0);          // one work-item per (row, pair)
    int hd2 = rotary_dim >> 1;
    int total = n_rows * hd2;
    if (idx >= total) {
        return;
    }
    int row = idx / hd2;
    int p = idx - row * hd2;
    int base = row * head_dim;
    int ci = pos * hd2 + p;                    // precomputed table for every position
    float a = x[base + p];
    float b = x[base + hd2 + p];
    float c = cosv[ci];
    float s = sinv[ci];
    x[base + p] = a * c - b * s;
    x[base + hd2 + p] = a * s + b * c;
}

