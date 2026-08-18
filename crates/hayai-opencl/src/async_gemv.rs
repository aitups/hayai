//! Non-blocking OpenCL GEMV: enqueue now, wait later (`cl_event`).
//! Supports host upload **or** weights already resident in a device/scratch buffer + byte offset.

use crate::context::{OpenClEngine, OpenClError};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::types::{cl_float, cl_int, cl_uchar, CL_BLOCKING, CL_NON_BLOCKING};
use std::ptr;

/// In-flight GGML GEMV. Device buffers + host output stay alive until [`PendingGemv::wait`].
pub struct PendingGemv {
    _weights_owned: Option<Buffer<cl_uchar>>,
    _input: Buffer<cl_float>,
    _output_dev: Buffer<cl_float>,
    host_out: Vec<f32>,
    complete: Event,
    device_name: String,
    label: &'static str,
}

impl PendingGemv {
    pub fn wait(self) -> Result<Vec<f32>, OpenClError> {
        self.complete
            .wait()
            .map_err(|e| OpenClError::ClError(format!("wait {}: {e}", self.label)))?;
        tracing::debug!(
            "OpenCL async ggml_gemv_{} done on {}",
            self.label,
            self.device_name
        );
        Ok(self.host_out)
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
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let off_i = weight_off as cl_int;

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

        let local = preferred_local_async(m);
        let global = ((m + local - 1) / local) * local;

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
            complete,
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
        let pending =
            self.ggml_gemv_begin_dev(kernel, label, m, n, &weights_buf, weight_off, input)?;
        Ok(PendingGemv {
            _weights_owned: Some(weights_buf),
            _input: pending._input,
            _output_dev: pending._output_dev,
            host_out: pending.host_out,
            complete: pending.complete,
            device_name: pending.device_name,
            label: pending.label,
        })
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
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let off_i = weight_off as cl_int;

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

        let local = preferred_local_async(m);
        let global = ((m + local - 1) / local) * local;

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
            complete,
            device_name: self.device_info.device_name.clone(),
            label,
        })
    }
}

fn preferred_local_async(m: usize) -> usize {
    const CANDIDATES: [usize; 4] = [256, 128, 64, 32];
    for &ls in &CANDIDATES {
        if m % ls == 0 {
            return ls;
        }
    }
    64.min(m.max(16))
}
