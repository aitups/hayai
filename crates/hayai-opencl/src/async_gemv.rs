//! Non-blocking OpenCL GEMV: enqueue now, wait later (`cl_event`).
//! Supports host upload **or** weights already resident in a device/scratch buffer + byte offset.

use crate::context::{OpenClEngine, OpenClError};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::types::{cl_float, cl_int, cl_long, cl_uchar, CL_BLOCKING, CL_NON_BLOCKING};
use std::ptr;

/// In-flight GGML GEMV. Device buffers + host output stay alive until the
/// completion event is waited on. If the value is dropped without calling
/// [`PendingGemv::wait`], the event is still waited on (see [`Drop`]) so the
/// driver never writes into an already-freed host buffer.
pub struct PendingGemv {
    _weights_owned: Option<Buffer<cl_uchar>>,
    _input: Buffer<cl_float>,
    _output_dev: Buffer<cl_float>,
    host_out: Vec<f32>,
    complete: Option<Event>,
    device_name: String,
    label: &'static str,
}

impl PendingGemv {
    /// Block until the device GEMV completes and take the host output vector.
    pub fn wait(mut self) -> Result<Vec<f32>, OpenClError> {
        self.finish()?;
        tracing::debug!(
            "OpenCL async ggml_gemv_{} done on {}",
            self.label,
            self.device_name
        );
        Ok(std::mem::take(&mut self.host_out))
    }

    /// Wait on the completion event at most once. Idempotent.
    fn finish(&mut self) -> Result<(), OpenClError> {
        if let Some(event) = self.complete.take() {
            event
                .wait()
                .map_err(|e| OpenClError::ClError(format!("wait {}: {e}", self.label)))?;
        }
        Ok(())
    }
}

impl Drop for PendingGemv {
    fn drop(&mut self) {
        // Never free `host_out` while a non-blocking read may still target it.
        // A driver error here cannot be propagated from `Drop`, but waiting is
        // the soundness requirement; report it for debugging.
        if let Err(e) = self.finish() {
            tracing::warn!("PendingGemv dropped before completion: {e}");
        }
    }
}

impl OpenClEngine {
    pub fn ggml_gemv_q4_0_async(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        self.ggml_gemv_begin(&self.gemv_q4_0, "q4_0", m, n, 0, weights, input)
    }

    pub fn ggml_gemv_q4_1_async(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        self.ggml_gemv_begin(&self.gemv_q4_1, "q4_1", m, n, 0, weights, input)
    }

    pub fn ggml_gemv_q8_0_async(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        self.ggml_gemv_begin(&self.gemv_q8_0, "q8_0", m, n, 0, weights, input)
    }

    pub fn ggml_gemv_async(
        &self,
        kernel: &Kernel,
        label: &'static str,
        m: usize,
        n: usize,
        weights: &[u8],
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        self.ggml_gemv_begin(kernel, label, m, n, 0, weights, input)
    }

    /// GEMV reading weights from an existing device buffer at `weight_off` (dGPU mirror).
    pub fn ggml_gemv_async_from_device(
        &self,
        kernel: &Kernel,
        label: &'static str,
        m: usize,
        n: usize,
        weights_dev: &Buffer<cl_uchar>,
        weight_off: usize,
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        self.ggml_gemv_begin_dev(kernel, label, m, n, weights_dev, weight_off, input)
    }

    /// GEMV with SVM base pointer + byte offset (APU zero-copy; expert Base+Offset).
    ///
    /// The raw `svm_base` is handed to `clSetKernelArg` (SVM pointer argument); it
    /// is not dereferenced by Rust, so this is not an `unsafe fn` — the caller is
    /// responsible for the pointer being a valid SVM allocation.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn ggml_gemv_async_from_svm(
        &self,
        kernel: &Kernel,
        label: &'static str,
        m: usize,
        n: usize,
        svm_base: *const u8,
        weight_off: usize,
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        assert_eq!(input.len(), n);
        crate::compute::validate_gemv_shape(label, m, n)?;
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let off_i = weight_off as cl_long;

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
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|e| OpenClError::ClError(format!("write input: {e}")))?;
        }

        let local = preferred_local_async(m, self.device_info.max_work_group_size);
        let split = crate::compute::gemv_split(label);
        let local = if split > 1 { (local / split * split).max(split) } else { local };
        let global = ((m.saturating_mul(split) + local - 1) / local) * local;

        let local_bytes = 2048 * std::mem::size_of::<cl_float>();
        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&m_i)
                .set_arg(&n_i)
                .set_arg(&off_i)
                .set_arg_svm(svm_base)
                .set_arg(&input_buf)
                .set_arg(&output_buf)
                .set_arg_local_buffer(local_bytes)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue async SVM {label}: {e}")))?
        };

        let mut host_out = vec![0.0f32; m];
        let complete = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &output_buf,
                    CL_NON_BLOCKING,
                    0,
                    &mut host_out,
                    &[kernel_event.get()],
                )
                .map_err(|e| OpenClError::ClError(format!("read async SVM {label}: {e}")))?
        };

        Ok(PendingGemv {
            _weights_owned: None,
            _input: input_buf,
            _output_dev: output_buf,
            host_out,
            complete: Some(complete),
            device_name: self.device_info.device_name.clone(),
            label,
        })
    }

    fn ggml_gemv_begin(
        &self,
        kernel: &Kernel,
        label: &'static str,
        m: usize,
        n: usize,
        weight_off: usize,
        weights: &[u8],
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        assert_eq!(input.len(), n);
        crate::compute::validate_gemv_weights(label, m, n, weights.len())?;
        let mut weights_buf = unsafe {
            Buffer::<cl_uchar>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                weights.len(),
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("weights buffer: {e}")))?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut weights_buf, CL_BLOCKING, 0, weights, &[])
                .map_err(|e| OpenClError::ClError(format!("write weights: {e}")))?;
        }
        let mut pending =
            self.ggml_gemv_begin_dev(kernel, label, m, n, &weights_buf, weight_off, input)?;
        // Keep the weights buffer alive for the lifetime of the in-flight GEMV.
        pending._weights_owned = Some(weights_buf);
        Ok(pending)
    }

    fn ggml_gemv_begin_dev(
        &self,
        kernel: &Kernel,
        label: &'static str,
        m: usize,
        n: usize,
        weights_dev: &Buffer<cl_uchar>,
        weight_off: usize,
        input: &[f32],
    ) -> Result<PendingGemv, OpenClError> {
        assert_eq!(input.len(), n);
        crate::compute::validate_gemv_shape(label, m, n)?;
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let off_i = weight_off as cl_long;

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
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|e| OpenClError::ClError(format!("write input: {e}")))?;
        }

        let local = preferred_local_async(m, self.device_info.max_work_group_size);
        let split = crate::compute::gemv_split(label);
        let local = if split > 1 { (local / split * split).max(split) } else { local };
        let global = ((m.saturating_mul(split) + local - 1) / local) * local;

        let local_bytes = 2048 * std::mem::size_of::<cl_float>();
        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&m_i)
                .set_arg(&n_i)
                .set_arg(&off_i)
                .set_arg(weights_dev)
                .set_arg(&input_buf)
                .set_arg(&output_buf)
                .set_arg_local_buffer(local_bytes)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue async {label}: {e}")))?
        };

        let mut host_out = vec![0.0f32; m];
        let complete = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &output_buf,
                    CL_NON_BLOCKING,
                    0,
                    &mut host_out,
                    &[kernel_event.get()],
                )
                .map_err(|e| OpenClError::ClError(format!("read async {label}: {e}")))?
        };

        Ok(PendingGemv {
            _weights_owned: None,
            _input: input_buf,
            _output_dev: output_buf,
            host_out,
            complete: Some(complete),
            device_name: self.device_info.device_name.clone(),
            label,
        })
    }

    /// Fixed per-op launch overhead (seconds), measured with a tiny resident GEMV.
    /// This is what dominates small-op devices and what the planner must price.
    pub fn bench_launch_seconds(&self) -> Result<f64, OpenClError> {
        let (m, n) = (64usize, 64usize);
        let weights = vec![0u8; m * n * 4];
        let mut wbuf = unsafe {
            Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, weights.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("bench weights: {e}")))?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut wbuf, CL_BLOCKING, 0, &weights, &[])
        }
        .map_err(|e| OpenClError::ClError(format!("bench write: {e}")))?;
        let input = vec![0.01f32; n];
        let _ = self
            .ggml_gemv_begin_dev(&self.gemv_f32, "f32", m, n, &wbuf, 0, &input)?
            .wait()?;
        let iters = 50usize;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self
                .ggml_gemv_begin_dev(&self.gemv_f32, "f32", m, n, &wbuf, 0, &input)?
                .wait()?;
        }
        Ok(t0.elapsed().as_secs_f64() / iters as f64)
    }

    /// Effective GEMV bandwidth (bytes/s) with the weights **already resident** on the
    /// device (no per-call upload) — the mirror/SVM-owned path the planner prices.
    pub fn bench_resident_gemv_gbytes_s(&self, m: usize, n: usize) -> Result<f64, OpenClError> {
        self.bench_resident_gemv_label_gbytes_s(&self.gemv_f32, "f32", m, n)
    }

    /// Resident GEMV bandwidth for the 4-bit K-quant that dominates real FFN weights.
    pub fn bench_resident_gemv_q4k_gbytes_s(&self, m: usize, n: usize) -> Result<f64, OpenClError> {
        if n % 256 != 0 {
            return Ok(0.0);
        }
        self.bench_resident_gemv_label_gbytes_s(&self.gemv_q4_k, "q4_k", m, n)
    }

    pub fn bench_resident_gemv_label_gbytes_s(
        &self,
        kernel: &Kernel,
        label: &str,
        m: usize,
        n: usize,
    ) -> Result<f64, OpenClError> {
        let row_bytes = match label {
            "q4_k" => (n / 256) * 144,
            _ => n * 4,
        };
        let nbytes = m * row_bytes;
        let weights = vec![0u8; nbytes];
        let mut wbuf = unsafe {
            Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, nbytes, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("bench weights: {e}")))?
        };
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut wbuf, CL_BLOCKING, 0, &weights, &[])
        }
        .map_err(|e| OpenClError::ClError(format!("bench write: {e}")))?;
        let input = vec![0.01f32; n];
        let mut out = vec![0.0f32; m];
        let _ = self.ggml_gemv_dispatch_bound(
            kernel,
            label,
            m,
            n,
            crate::compute::GgmlWeightBind::Device(&wbuf),
            0,
            &input,
            &mut out,
        )?;
        let iters = 10usize;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self.ggml_gemv_dispatch_bound(
                kernel,
                label,
                m,
                n,
                crate::compute::GgmlWeightBind::Device(&wbuf),
                0,
                &input,
                &mut out,
            )?;
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt <= 0.0 {
            return Ok(0.0);
        }
        Ok((nbytes * iters) as f64 / dt / 1e9)
    }

    /// Host→device transfer bandwidth (bytes/s) for a non-blocking write of `bytes`.
    pub fn bench_dma_gbytes_s(&self, bytes: usize) -> Result<f64, OpenClError> {
        let data = vec![0u8; bytes];
        let iters = 5usize;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let mut buf = unsafe {
                Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, bytes, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("bench dma buf: {e}")))?
            };
            let ev = unsafe {
                self.queue
                    .enqueue_write_buffer(&mut buf, CL_NON_BLOCKING, 0, &data, &[])
            }
            .map_err(|e| OpenClError::ClError(format!("bench dma write: {e}")))?;
            ev.wait()
                .map_err(|e| OpenClError::ClError(format!("bench dma wait: {e}")))?;
        }
        let dt = t0.elapsed().as_secs_f64();
        if dt <= 0.0 {
            return Ok(0.0);
        }
        Ok((bytes * iters) as f64 / dt / 1e9)
    }
}

fn preferred_local_async(m: usize, max_work_group: usize) -> usize {
    let cap = max_work_group.max(1);
    const CANDIDATES: [usize; 4] = [256, 128, 64, 32];
    for &ls in &CANDIDATES {
        if ls <= cap && m % ls == 0 {
            return ls;
        }
    }
    64.min(m.max(16)).min(cap)
}


