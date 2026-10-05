// Flash-style single-query attention decode with an online (running) softmax.
//
// One work-item per query head. K/V come from a device FP32 cache laid out
// `[seq * n_kv * head_dim]`. GQA: query head `h` reads KV head `h % n_kv`.
// The kernel never materialises the `seq` score vector, so it works at any
// context length with `head_dim` floats of private state per head.
//
// Targets the mandatory OpenCL C 1.2 subset of OpenCL 3.0 (NVIDIA reports 3.0
// but only compiles C 1.2): no dynamic vector-component selectors, no C 2.0
// features. `head_dim` must be <= HAYAI_MAX_HEAD_DIM.

#ifndef HAYAI_MAX_HEAD_DIM
#define HAYAI_MAX_HEAD_DIM 256
#endif

__kernel void hayai_attn_decode(
    __global const float* restrict q,   // [n_heads * head_dim]
    __global const float* restrict k,   // [max_seq * n_kv * head_dim]
    __global const float* restrict v,   // [max_seq * n_kv * head_dim]
    __global float* restrict out,       // [n_heads * head_dim]
    const int n_heads,
    const int n_kv,
    const int groups,                   // n_heads / n_kv
    const int head_dim,
    const int seq,                      // number of valid positions (0 => empty)
    const float scale)
{
    int h = (int)get_global_id(0);
    if (h >= n_heads || head_dim > HAYAI_MAX_HEAD_DIM) {
        return;
    }
    // GQA mapping used by the host engine: query head h reads KV head h / groups
    // (HF `repeat_interleave`), NOT h % n_kv.
    int kh = (groups > 0) ? (h / groups) : 0;

    __private float qv[HAYAI_MAX_HEAD_DIM];
    __private float acc[HAYAI_MAX_HEAD_DIM];
    for (int d = 0; d < head_dim; ++d) {
        qv[d] = q[h * head_dim + d];
        acc[d] = 0.0f;
    }
    if (seq <= 0) {
        for (int d = 0; d < head_dim; ++d) {
            out[h * head_dim + d] = 0.0f;
        }
        return;
    }

    float m;
    float l;
    // Seed the running softmax with position 0 (avoids a huge -inf sentinel, which
    // trips denormal/inf handling on some drivers).
    {
        __global const float* k0 = k + kh * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qv[d] * k0[d];
        }
        m = dot * scale;
        l = 1.0f;
        __global const float* v0 = v + kh * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = v0[d];
        }
    }
    for (int j = 1; j < seq; ++j) {
        __global const float* kj = k + (j * n_kv + kh) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += qv[d] * kj[d];
        }
        dot *= scale;
        float m_new = fmax(m, dot);
        float p = exp(dot - m_new);
        float corr = exp(m - m_new);
        l = l * corr + p;
        __global const float* vj = v + (j * n_kv + kh) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * corr + p * vj[d];
        }
        m = m_new;
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        out[h * head_dim + d] = acc[d] * inv;
    }
}




