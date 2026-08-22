use crate::context::{OpenClEngine, OpenClError};
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::types::{cl_float, cl_int, cl_uchar, CL_BLOCKING, CL_NON_BLOCKING};
use std::ptr;
use tracing::debug;

impl OpenClEngine {
    /// Run Q4 LUT MatMul on the active OpenCL device and write results into `output`.
    ///
    /// Allocates transient device buffers, uploads inputs, enqueues `lut_matmul_q4_v1`,
    /// and reads back the output vector. Suitable for correctness validation and
    /// micro-benchmarks; the streaming path reuses persistent ping-pong buffers.
    pub fn lut_matmul_q4(
        &self,
        m: usize,
        n: usize,
        weights_q4: &[u8],
        lut: &[f32; 16],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        assert_eq!(n % 2, 0, "N must be even for packed Q4 weights");
        let packed_cols = n / 2;
        assert_eq!(weights_q4.len(), m * packed_cols);
        assert_eq!(input.len(), n);
        assert_eq!(output.len(), m);

        let m_i = m as cl_int;
        let n_i = n as cl_int;

        let mut weights_buf = unsafe {
            Buffer::<cl_uchar>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                weights_q4.len(),
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("weights buffer: {e}")))?
        };
        let mut lut_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, 16, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("lut buffer: {e}")))?
        };
        let mut input_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("input buffer: {e}")))?
        };
        let output_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_WRITE_ONLY, m, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("output buffer: {e}")))?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut weights_buf, CL_BLOCKING, 0, weights_q4, &[])
                .map_err(|e| OpenClError::ClError(format!("write weights: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, lut, &[])
                .map_err(|e| OpenClError::ClError(format!("write lut: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|e| OpenClError::ClError(format!("write input: {e}")))?;
        }

        // Prefer a local size that covers the 16-entry LUT load and divides M when possible.
        let local = preferred_local_size(m);
        let global = ((m + local - 1) / local) * local;

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.lut_kernel)
                .set_arg(&m_i)
                .set_arg(&n_i)
                .set_arg(&weights_buf)
                .set_arg(&lut_buf)
                .set_arg(&input_buf)
                .set_arg(&output_buf)
                .set_arg_local_buffer(16 * std::mem::size_of::<cl_float>())
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue lut_matmul: {e}")))?
        };

        let wait = [kernel_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, output, &wait)
                .map_err(|e| OpenClError::ClError(format!("read output: {e}")))?;
        }

        debug!(
            "OpenCL lut_matmul_q4 [{m}×{n}] done on {}",
            self.device_info.device_name
        );
        Ok(())
    }

    /// GGML Q4_0 GEMV on device (GGUF layout).
    pub fn ggml_gemv_q4_0(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        self.ggml_gemv_dispatch(&self.gemv_q4_0, "q4_0", m, n, weights, input, output)
    }

    pub fn ggml_gemv_q4_1(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        self.ggml_gemv_dispatch(&self.gemv_q4_1, "q4_1", m, n, weights, input, output)
    }

    pub fn ggml_gemv_q8_0(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        self.ggml_gemv_dispatch(&self.gemv_q8_0, "q8_0", m, n, weights, input, output)
    }

    pub fn ggml_gemv_q4_k(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        self.ggml_gemv_dispatch(&self.gemv_q4_k, "q4_k", m, n, weights, input, output)
    }

    pub fn ggml_gemv_q6_k(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        self.ggml_gemv_dispatch(&self.gemv_q6_k, "q6_k", m, n, weights, input, output)
    }

    pub fn ggml_gemv_q5_k(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        self.ggml_gemv_dispatch(&self.gemv_q5_k, "q5_k", m, n, weights, input, output)
    }

    fn ggml_gemv_dispatch(
        &self,
        kernel: &opencl3::kernel::Kernel,
        label: &str,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        assert_eq!(input.len(), n);
        assert_eq!(output.len(), m);
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let off_i = 0i64; // cl_long (kernel arg)

        let mut weights_buf = unsafe {
            Buffer::<cl_uchar>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                weights.len(),
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("weights buffer: {e}")))?
        };
        let mut input_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("input buffer: {e}")))?
        };
        let output_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_WRITE_ONLY, m, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("output buffer: {e}")))?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut weights_buf, CL_BLOCKING, 0, weights, &[])
                .map_err(|e| OpenClError::ClError(format!("write weights: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|e| OpenClError::ClError(format!("write input: {e}")))?;
        }

        let local = preferred_local_size(m);
        let global = ((m + local - 1) / local) * local;

        // Must match HAYAI_X_TILE in ggml_gemv_q4.cl (tiled __local input).
        let local_bytes = 2048 * std::mem::size_of::<cl_float>();
        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&m_i)
                .set_arg(&n_i)
                .set_arg(&off_i)
                .set_arg(&weights_buf)
                .set_arg(&input_buf)
                .set_arg(&output_buf)
                .set_arg_local_buffer(local_bytes)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue ggml_gemv_{label}: {e}")))?
        };

        let wait = [kernel_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, output, &wait)
                .map_err(|e| OpenClError::ClError(format!("read output: {e}")))?;
        }
        debug!(
            "OpenCL ggml_gemv_{label} [{m}×{n}] on {}",
            self.device_info.device_name
        );
        Ok(())
    }

    /// Non-blocking write of host bytes into a device weight buffer (streaming upload).
    pub fn enqueue_write_layer(
        &self,
        device_buf: &mut Buffer<cl_uchar>,
        host_bytes: &[u8],
    ) -> Result<opencl3::event::Event, OpenClError> {
        unsafe {
            self.queue
                .enqueue_write_buffer(device_buf, CL_NON_BLOCKING, 0, host_bytes, &[])
                .map_err(|e| OpenClError::ClError(format!("enqueue_write_layer: {e}")))
        }
    }
}

fn preferred_local_size(m: usize) -> usize {
    const CANDIDATES: [usize; 4] = [256, 128, 64, 32];
    for &ls in &CANDIDATES {
        if m % ls == 0 {
            return ls;
        }
    }
    64.min(m.max(16))
}

#[cfg(test)]
mod tests {
    use crate::device::{discover_opencl_devices, DeviceKind};
    use crate::OpenClEngine;
    use hayai_cpu::{cpu_lut_matmul_q4, max_abs_diff};
    use hayai_model::quant::gemv_q4_0;

    /// Init the engine, skipping ONLY when no GPU OpenCL device exists. If a GPU is
    /// present but engine init (kernel compilation) fails, the test FAILS — a compile
    /// regression must never hide behind a silent skip (that is exactly how the
    /// v0.2.1 `v.s[j]` kernel breakage slipped through the suite).
    fn init_engine_or_skip(label: &str) -> Option<OpenClEngine> {
        let has_gpu = discover_opencl_devices()
            .iter()
            .any(|d| d.device_kind != DeviceKind::CpuOpenCl);
        match OpenClEngine::try_init_any() {
            Ok(eng) => Some(eng),
            Err(e) if !has_gpu => {
                eprintln!("skipping {label}: no OpenCL device");
                None
            }
            Err(e) => panic!("OpenCL device(s) present but engine init failed: {e}"),
        }
    }

    #[test]
    fn opencl_lut_matches_cpu_when_available() {
        let Some(engine) = init_engine_or_skip("OpenCL LUT test") else {
            return;
        };

        let m = 64;
        let n = 128;
        let lut: [f32; 16] = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ];
        let weights_q4: Vec<u8> = (0..(m * n / 2)).map(|i| ((i * 17) % 256) as u8).collect();
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.5).collect();

        let mut cpu_out = vec![0.0f32; m];
        let mut gpu_out = vec![0.0f32; m];
        cpu_lut_matmul_q4(m, n, &weights_q4, &lut, &input, &mut cpu_out);
        engine
            .lut_matmul_q4(m, n, &weights_q4, &lut, &input, &mut gpu_out)
            .expect("OpenCL lut_matmul_q4 failed");

        let err = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            err < 1e-4,
            "OpenCL vs CPU LUT mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    #[test]
    fn opencl_ggml_q4_0_matches_cpu_when_available() {
        let Some(engine) = init_engine_or_skip("OpenCL q4_0 gemv test") else {
            return;
        };

        let m = 64usize;
        let n = 128usize;
        let blocks = n / 32;
        let row_bytes = blocks * 18;
        let mut weights = vec![0u8; m * row_bytes];
        for row in 0..m {
            for b in 0..blocks {
                let base = row * row_bytes + b * 18;
                weights[base] = 0x66;
                weights[base + 1] = 0x2E;
                for j in 0..16 {
                    weights[base + 2 + j] = ((row + b + j) % 256) as u8;
                }
            }
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let mut cpu_out = vec![0.0f32; m];
        let mut gpu_out = vec![0.0f32; m];
        gemv_q4_0(n, &weights, &input, &mut cpu_out);
        engine
            .ggml_gemv_q4_0(m, n, &weights, &input, &mut gpu_out)
            .expect("OpenCL ggml_gemv_q4_0 failed");

        let err = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            err < 1e-3,
            "OpenCL ggml Q4_0 vs CPU mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    #[test]
    fn opencl_ggml_q4_k_matches_cpu_when_available() {
        use hayai_model::q4k::gemv_q4_k;

        let Some(engine) = init_engine_or_skip("OpenCL q4_k gemv test") else {
            return;
        };

        let m = 32usize;
        let n = 256usize; // 1 block
        let row_bytes = 144;
        let mut weights = vec![0u8; m * row_bytes];
        for row in 0..m {
            let base = row * row_bytes;
            // d=1.0 (fp16 0x3C00), min=0.0
            weights[base] = 0x00;
            weights[base + 1] = 0x3C;
            weights[base + 2] = 0x00;
            weights[base + 3] = 0x00;
            // scales: 1
            for s in 4..16 {
                weights[base + s] = 1;
            }
            // q
            for q in 16..144 {
                weights[base + q] = ((row + q) % 256) as u8;
            }
        }

        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 - 0.2).collect();
        let mut cpu_out = vec![0.0f32; m];
        let mut gpu_out = vec![0.0f32; m];
        gemv_q4_k(m, n, &weights, &input, &mut cpu_out).unwrap();
        engine
            .ggml_gemv_q4_k(m, n, &weights, &input, &mut gpu_out)
            .expect("OpenCL ggml_gemv_q4_k failed");

        let err = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            err < 1e-3,
            "OpenCL ggml Q4_K vs CPU mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    #[test]
    fn opencl_ggml_q6_k_matches_cpu_when_available() {
        use hayai_model::q6k::gemv_q6_k;

        let Some(engine) = init_engine_or_skip("OpenCL q6_k gemv test") else {
            return;
        };

        let m = 32usize;
        let n = 256usize; // 1 block
        let row_bytes = 210;
        let mut weights = vec![0u8; m * row_bytes];
        for row in 0..m {
            let base = row * row_bytes;
            // ql (128)
            for q in 0..128 {
                weights[base + q] = ((row + q) % 256) as u8;
            }
            // qh (64)
            for q in 128..192 {
                weights[base + q] = ((row * 3 + q) % 256) as u8;
            }
            // scales (16)
            for s in 192..208 {
                weights[base + s] = 1;
            }
            // d=1.0 (fp16 0x3C00)
            weights[base + 208] = 0x00;
            weights[base + 209] = 0x3C;
        }

        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.005 - 0.2).collect();
        let mut cpu_out = vec![0.0f32; m];
        let mut gpu_out = vec![0.0f32; m];
        gemv_q6_k(m, n, &weights, &input, &mut cpu_out).unwrap();
        engine
            .ggml_gemv_q6_k(m, n, &weights, &input, &mut gpu_out)
            .expect("OpenCL ggml_gemv_q6_k failed");

        let err = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            err < 1e-3,
            "OpenCL ggml Q6_K vs CPU mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    #[test]
    fn opencl_ggml_q4_1_matches_cpu_when_available() {
        use hayai_model::quant::gemv_q4_1;

        let Some(engine) = init_engine_or_skip("OpenCL q4_1 gemv test") else {
            return;
        };

        let m = 32usize;
        let n = 256usize; // 8 blocks
        let blocks = n / 32;
        let row_bytes = blocks * 20;
        let mut weights = vec![0u8; m * row_bytes];
        for row in 0..m {
            for b in 0..blocks {
                let base = row * row_bytes + b * 20;
                weights[base] = 0x66; // d fp16
                weights[base + 1] = 0x2E;
                weights[base + 2] = 0x00; // m fp16 = 0
                weights[base + 3] = 0x00;
                for j in 0..16 {
                    weights[base + 4 + j] = ((row + b + j) % 256) as u8;
                }
            }
        }

        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let mut cpu_out = vec![0.0f32; m];
        let mut gpu_out = vec![0.0f32; m];
        gemv_q4_1(n, &weights, &input, &mut cpu_out);
        engine
            .ggml_gemv_q4_1(m, n, &weights, &input, &mut gpu_out)
            .expect("OpenCL ggml_gemv_q4_1 failed");

        let err = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            err < 1e-3,
            "OpenCL ggml Q4_1 vs CPU mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    #[test]
    fn opencl_ggml_q8_0_matches_cpu_when_available() {
        use hayai_model::quant::gemv_q8_0;

        let Some(engine) = init_engine_or_skip("OpenCL q8_0 gemv test") else {
            return;
        };

        let m = 32usize;
        let n = 256usize; // 8 blocks
        let blocks = n / 32;
        let row_bytes = blocks * 34;
        let mut weights = vec![0u8; m * row_bytes];
        for row in 0..m {
            for b in 0..blocks {
                let base = row * row_bytes + b * 34;
                weights[base] = 0x66; // d fp16
                weights[base + 1] = 0x2E;
                for j in 0..32 {
                    weights[base + 2 + j] = ((row + b + j) % 256) as u8;
                }
            }
        }

        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.4).collect();
        let mut cpu_out = vec![0.0f32; m];
        let mut gpu_out = vec![0.0f32; m];
        gemv_q8_0(n, &weights, &input, &mut cpu_out);
        engine
            .ggml_gemv_q8_0(m, n, &weights, &input, &mut gpu_out)
            .expect("OpenCL ggml_gemv_q8_0 failed");

        let err = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            err < 1e-3,
            "OpenCL ggml Q8_0 vs CPU mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    #[test]
    fn opencl_all_gpus_compile_and_match_cpu() {
        // Validates EVERY OpenCL 3.0 GPU in the pool (dGPU + iGPU): kernels must compile
        // and produce CPU-identical results on each device, not just the primary.
        use crate::pool::OpenClDevicePool;

        let has_gpu = discover_opencl_devices()
            .iter()
            .any(|d| d.device_kind != DeviceKind::CpuOpenCl);
        let pool = match OpenClDevicePool::try_init_all_gpus() {
            Ok(p) => p,
            Err(_) if !has_gpu => {
                eprintln!("skipping pool test: no OpenCL device");
                return;
            }
            Err(e) => panic!("OpenCL device(s) present but pool init failed: {e}"),
        };

        let m = 64usize;
        let n = 128usize;
        let blocks = n / 32;
        let row_bytes = blocks * 18;
        let mut weights = vec![0u8; m * row_bytes];
        for row in 0..m {
            for b in 0..blocks {
                let base = row * row_bytes + b * 18;
                weights[base] = 0x66;
                weights[base + 1] = 0x2E;
                for j in 0..16 {
                    weights[base + 2 + j] = ((row + b + j) % 256) as u8;
                }
            }
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.4).collect();

        for eng in pool.engines.iter() {
            let mut cpu_out = vec![0.0f32; m];
            let mut gpu_out = vec![0.0f32; m];
            gemv_q4_0(n, &weights, &input, &mut cpu_out);
            eng.ggml_gemv_q4_0(m, n, &weights, &input, &mut gpu_out)
                .expect("OpenCL ggml_gemv_q4_0 failed");
            let err = max_abs_diff(&cpu_out, &gpu_out);
            assert!(
                err < 1e-3,
                "OpenCL ggml Q4_0 vs CPU mismatch: {err} on {}",
                eng.device_info.device_name
            );
        }
    }
}

