//! Portable SIMD helpers via `std::simd` (PRD §2 / §3.3).

#![allow(unstable_name_collisions)]

use std::simd::f32x8;
use std::simd::num::SimdFloat;

/// Mean of squares (for RMSNorm).
#[inline]
pub fn simd_mean_square(x: &[f32]) -> f32 {
    let mut sum = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= x.len() {
        let v = f32x8::from_slice(&x[i..i + 8]);
        sum += v * v;
        i += 8;
    }
    let mut ms = sum.reduce_sum();
    while i < x.len() {
        ms += x[i] * x[i];
        i += 1;
    }
    ms / x.len() as f32
}

/// `x[i] *= inv * weight[i]`
#[inline]
pub fn simd_scale_mul(x: &mut [f32], weight: &[f32], inv: f32) {
    let inv_v = f32x8::splat(inv);
    let mut i = 0;
    while i + 8 <= x.len() {
        let xv = f32x8::from_slice(&x[i..i + 8]);
        let wv = f32x8::from_slice(&weight[i..i + 8]);
        (xv * inv_v * wv).copy_to_slice(&mut x[i..i + 8]);
        i += 8;
    }
    while i < x.len() {
        x[i] *= inv * weight[i];
        i += 1;
    }
}

/// Dot product.
#[inline]
pub fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= a.len() {
        let av = f32x8::from_slice(&a[i..i + 8]);
        let bv = f32x8::from_slice(&b[i..i + 8]);
        sum += av * bv;
        i += 8;
    }
    let mut acc = sum.reduce_sum();
    while i < a.len() {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// Dot `query` with INT8 vector dequantized by per-channel scales.
#[inline]
pub fn simd_dot_i8_channel(query: &[f32], keys: &[i8], scales: &[f32]) -> f32 {
    assert_eq!(query.len(), keys.len());
    assert_eq!(query.len(), scales.len());
    let mut sum = f32x8::splat(0.0);
    let mut i = 0;
    while i + 8 <= query.len() {
        let q = f32x8::from_slice(&query[i..i + 8]);
        let s = f32x8::from_slice(&scales[i..i + 8]);
        let k = f32x8::from_array([
            keys[i] as f32,
            keys[i + 1] as f32,
            keys[i + 2] as f32,
            keys[i + 3] as f32,
            keys[i + 4] as f32,
            keys[i + 5] as f32,
            keys[i + 6] as f32,
            keys[i + 7] as f32,
        ]);
        sum += q * (k * s);
        i += 8;
    }
    let mut acc = sum.reduce_sum();
    while i < query.len() {
        acc += query[i] * (keys[i] as f32 * scales[i]);
        i += 1;
    }
    acc
}
