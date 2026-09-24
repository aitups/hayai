//! CPU path: Attention / RoPE / KV + SIMD (`std::simd`, PRD §2 / §3.3).

#![feature(portable_simd)]

pub mod attention;
pub mod kv_cache;
pub mod matmul;
pub mod simd_ops;

pub use attention::{
    apply_rope, apply_rope_partial, apply_rope_partial_factors,
    apply_rope_partial_factors_scaled, attention_decode_step,
    attention_decode_step_alibi, attention_decode_step_ex, attention_fp32_reference, rms_norm,
    simd_dot, softmax,
    AttentionConfig, RopeScaling,
};
pub use kv_cache::{BoundedKvCache, LayerKvCache, DEFAULT_RECENT_FP_TOKENS};
pub use matmul::{cpu_lut_matmul_q4, fp32_matmul, max_abs_diff, unpack_q4_to_fp32};
