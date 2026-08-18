//! OpenCL Kernel Source Registry for Hayai Engine.

/// OpenCL C source for 4-bit LUT MatMul kernel (Hayai packed format).
pub const LUT_MATMUL_CL: &str = include_str!("../kernels/lut_matmul.cl");

/// OpenCL C source for GGML Q4_0 / Q4_1 / Q8_0 GEMV (GGUF weights).
pub const GGML_GEMV_Q4_CL: &str = include_str!("../kernels/ggml_gemv_q4.cl");

/// Combined program source built by the OpenCL engine.
pub fn opencl_program_source() -> String {
    let mut src = String::with_capacity(LUT_MATMUL_CL.len() + GGML_GEMV_Q4_CL.len() + 8);
    src.push_str(LUT_MATMUL_CL);
    src.push('\n');
    src.push_str(GGML_GEMV_Q4_CL);
    src
}