use rayon::prelude::*;

/// Dense FP32 matrix-vector multiply: `output[m] = sum_n weights[m,n] * input[n]`.
/// `weights` is row-major `[M * N]`.
pub fn fp32_matmul(m: usize, n: usize, weights: &[f32], input: &[f32], output: &mut [f32]) {
    assert_eq!(weights.len(), m * n);
    assert_eq!(input.len(), n);
    assert_eq!(output.len(), m);

    output.par_iter_mut().enumerate().for_each(|(row, out_val)| {
        let row_offset = row * n;
        let mut sum = 0.0f32;
        for col in 0..n {
            sum += weights[row_offset + col] * input[col];
        }
        *out_val = sum;
    });
}

/// Expand packed Q4 indices + LUT into dense FP32 weights (row-major).
pub fn unpack_q4_to_fp32(m: usize, n: usize, weights_q4: &[u8], lut: &[f32; 16]) -> Vec<f32> {
    assert_eq!(n % 2, 0);
    let packed_cols = n / 2;
    assert_eq!(weights_q4.len(), m * packed_cols);

    let mut weights = vec![0.0f32; m * n];
    for row in 0..m {
        let row_offset = row * packed_cols;
        for col in 0..packed_cols {
            let packed = weights_q4[row_offset + col];
            let idx0 = (packed & 0x0F) as usize;
            let idx1 = ((packed >> 4) & 0x0F) as usize;
            let out_base = row * n + col * 2;
            weights[out_base] = lut[idx0];
            weights[out_base + 1] = lut[idx1];
        }
    }
    weights
}

/// CPU 4-Bit Quantized Matrix-Vector Multiplication (LUT unpack + MAC).
/// M: Output Dimension (rows)
/// N: Input Dimension (columns)
/// weights_q4: M * (N / 2) bytes
/// lut: 16 float dictionary values
/// input: N floats
/// output: M floats
pub fn cpu_lut_matmul_q4(
    m: usize,
    n: usize,
    weights_q4: &[u8],
    lut: &[f32; 16],
    input: &[f32],
    output: &mut [f32],
) {
    assert_eq!(n % 2, 0);
    let packed_cols = n / 2;
    assert_eq!(weights_q4.len(), m * packed_cols);
    assert_eq!(input.len(), n);
    assert_eq!(output.len(), m);

    output.par_iter_mut().enumerate().for_each(|(row, out_val)| {
        let row_offset = row * packed_cols;
        let mut sum = 0.0f32;

        for col in 0..packed_cols {
            let packed = weights_q4[row_offset + col];
            let idx0 = (packed & 0x0F) as usize;
            let idx1 = ((packed >> 4) & 0x0F) as usize;

            let w0 = lut[idx0];
            let w1 = lut[idx1];

            let x0 = input[col * 2];
            let x1 = input[col * 2 + 1];

            sum += x0 * w0 + x1 * w1;
        }

        *out_val = sum;
    });
}

/// Max absolute difference between two vectors.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lut() -> [f32; 16] {
        [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ]
    }

    #[test]
    fn q4_matches_fp32_reference() {
        let m = 32;
        let n = 64;
        let lut = sample_lut();
        let weights_q4: Vec<u8> = (0..(m * n / 2)).map(|i| (i % 256) as u8).collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();

        let dense = unpack_q4_to_fp32(m, n, &weights_q4, &lut);
        let mut expected = vec![0.0f32; m];
        let mut got = vec![0.0f32; m];
        fp32_matmul(m, n, &dense, &input, &mut expected);
        cpu_lut_matmul_q4(m, n, &weights_q4, &lut, &input, &mut got);

        let err = max_abs_diff(&expected, &got);
        assert!(
            err < 1e-4,
            "Q4 LUT MatMul diverges from FP32 reference: max_abs_diff={err}"
        );
    }

    #[test]
    fn q4_smollm_ffn_shape_smoke() {
        // SmolLM-135M gate/up shape: intermediate × hidden
        let m = 96;
        let n = 64;
        let lut = sample_lut();
        let weights_q4 = vec![0xABu8; m * n / 2];
        let input = vec![1.0f32; n];
        let mut output = vec![0.0f32; m];
        cpu_lut_matmul_q4(m, n, &weights_q4, &lut, &input, &mut output);
        assert!(output.iter().all(|v| v.is_finite()));
    }
}
