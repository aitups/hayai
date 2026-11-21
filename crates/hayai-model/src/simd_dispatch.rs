//! Runtime SIMD feature detection for multiversioned CPU GEMV.
//!
//! The CPU GEMV kernels use `std::simd` (`f32x8`), whose codegen depends on the compile
//! target's enabled features. A baseline build (x86-64-v1, SSE2) emits two 128-bit ops
//! per `f32x8`; the `#[target_feature(enable = "avx2,fma,f16c")]` clones use one 256-bit
//! op instead (audit C.1). This is bit-exact — the per-lane f32 operations are unchanged
//! and Rust does not contract `a * b + c` into FMA by default — so validating the base
//! path validates the AVX2 path.

/// True when the running x86-64 CPU supports AVX2 + FMA + F16C. Always false on other
/// architectures (ARM/NEON etc. keep the portable `std::simd` path).
#[inline]
pub fn x86_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}
