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