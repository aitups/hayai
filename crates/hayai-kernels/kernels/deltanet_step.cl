// DeltaNet / gated linear attention: recurrent state update for one decode step.
//
// One work-item per (value head, value dim) — i.e. per column `vd` of that head's
// `h_k × h_v` state. Columns are independent, so no cross-work-item synchronisation.
// Per step (llama.cpp `build_delta_net` order):
//   s[:, vd] *= decay
//   kv_mem    = sum_ki s[ki, vd] * k[ki]
//   delta     = (v[vd] - kv_mem) * beta
//   s[ki, vd] = s[ki, vd] + k[ki] * delta
//   out[vd]   = sum_ki s[ki, vd] * q[ki]
// `decay`/`beta` are the per-head scalars the host already derived (beta sigmoid'd,
// decay = exp(A_log_softplus(...))). `q` is post L2-norm and q-scale.
//
// OpenCL C 1.2 subset; `state` is `[n_v_heads * h_k * h_v]`, q/kk `[n_k_heads * h_k]`.

__kernel void hayai_deltanet_step(
    __global const float* restrict q,     // [n_k_heads * h_k]
    __global const float* restrict kk,    // [n_k_heads * h_k]
    __global const float* restrict v,     // [n_v_heads * h_v]
    __global const float* restrict decay, // [n_v_heads]
    __global const float* restrict beta,  // [n_v_heads]
    __global float* restrict state,       // [n_v_heads * h_k * h_v]
    __global float* restrict out,         // [n_v_heads * h_v]
    const int n_k_heads,
    const int n_v_heads,
    const int h_k,
    const int h_v)
{
    int vi = (int)get_global_id(0);
    if (vi >= n_v_heads * h_v) {
        return;
    }
    int vh = vi / h_v;
    int vd = vi - vh * h_v;
    // llama.cpp `ggml_repeat` tile: V-head vh maps to K-head vh % n_k_heads.
    int kh = vh % n_k_heads;
    __global const float* qh = q + kh * h_k;
    __global const float* khv = kk + kh * h_k;
    __global float* s = state + (size_t)vh * h_k * h_v;
    float d = decay[vh];
    float b = beta[vh];

    float kv_mem = 0.0f;
    for (int ki = 0; ki < h_k; ++ki) {
        int idx = ki * h_v + vd;
        float sv = s[idx] * d;
        s[idx] = sv;
        kv_mem += sv * khv[ki];
    }
    float delta = (v[vh * h_v + vd] - kv_mem) * b;
    float o = 0.0f;
    for (int ki = 0; ki < h_k; ++ki) {
        int idx = ki * h_v + vd;
        float sv = s[idx] + khv[ki] * delta;
        s[idx] = sv;
        o += sv * qh[ki];
    }
    out[vh * h_v + vd] = o;
}
