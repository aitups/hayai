//! OpenCL Kernel Source Registry for Hayai Engine.

/// OpenCL C source for 4-bit LUT MatMul kernel (Hayai packed format).
pub const LUT_MATMUL_CL: &str = include_str!("../kernels/lut_matmul.cl");

/// OpenCL C source for GGML Q4_0 / Q4_1 / Q8_0 GEMV (GGUF weights).
pub const GGML_GEMV_Q4_CL: &str = include_str!("../kernels/ggml_gemv_q4.cl");

/// OpenCL C source for sparse FFN DAG SpMM (CSR, GGUF disperso de `saor`).
pub const SPMM_CSR_CL: &str = include_str!("../kernels/spmm_csr.cl");

/// OpenCL C source for single-query attention decode (online softmax, GQA).
pub const ATTN_DECODE_CL: &str = include_str!("../kernels/attn_decode.cl");

/// OpenCL C source for the DeltaNet recurrent state update (KV-heads, decode step).
pub const DELTANET_STEP_CL: &str = include_str!("../kernels/deltanet_step.cl");

/// Combined program source built by the OpenCL engine.
pub fn opencl_program_source() -> String {
    let mut src = String::with_capacity(
        LUT_MATMUL_CL.len()
            + GGML_GEMV_Q4_CL.len()
            + SPMM_CSR_CL.len()
            + ATTN_DECODE_CL.len()
            + DELTANET_STEP_CL.len()
            + 16,
    );
    src.push_str(LUT_MATMUL_CL);
    src.push('\n');
    src.push_str(GGML_GEMV_Q4_CL);
    src.push('\n');
    src.push_str(SPMM_CSR_CL);
    src.push('\n');
    src.push_str(ATTN_DECODE_CL);
    src.push('\n');
    src.push_str(DELTANET_STEP_CL);
    src
}

#[cfg(test)]
mod tests {
    use super::*;

    // `.s[`/`.x[`/`.lo[` … selectors with a runtime index are invalid in EVERY OpenCL C
    // version — dynamic vector indexing is the bare subscript `v[i]` (OpenCL C 2.0/3.0).
    const BAD_DYNAMIC_SELECTORS: [&str; 8] = [
        ".s[", ".v[", ".x[", ".y[", ".z[", ".w[", ".lo[", ".hi[",
    ];

    #[test]
    fn kernel_sources_never_dynamically_index_vector_components() {
        // Guards the regression that broke kernel compilation on NVIDIA/Intel in v0.2.1
        // (`q0.s[j]` → `illegal vector component name 's'`). Kernels target the mandatory
        // OpenCL C 1.2 subset of OpenCL 3.0 (NVIDIA's OpenCL 3.0 compiles C 1.2 only), so
        // dynamic component selectors `.s[i]` must never reappear — index via union arrays.
        for (name, src) in [
            ("lut_matmul.cl", LUT_MATMUL_CL),
            ("ggml_gemv_q4.cl", GGML_GEMV_Q4_CL),
            ("spmm_csr.cl", SPMM_CSR_CL),
        ] {
            for (i, line) in src.lines().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue; // documentation may mention `.s<N>` literally
                }
                for sel in BAD_DYNAMIC_SELECTORS {
                    assert!(
                        !line.contains(sel),
                        "{name}:{} has invalid dynamic vector component selector '{sel}': {line}",
                        i + 1
                    );
                }
            }
        }
    }
}