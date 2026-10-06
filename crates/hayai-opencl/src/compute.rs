use crate::context::{OpenClEngine, OpenClError};
use opencl3::kernel::ExecuteKernel;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::types::{cl_float, cl_int, cl_uchar, CL_BLOCKING, CL_NON_BLOCKING};
use std::ptr;
use tracing::debug;

/// Persistent device buffers for [`OpenClEngine::ggml_gemv_dispatch`].
#[derive(Default)]
pub struct GemvSyncWorkspace {
    pub(crate) weights: Option<Buffer<cl_uchar>>,
    pub(crate) input: Option<Buffer<cl_float>>,
    pub(crate) output: Option<Buffer<cl_float>>,
    pub(crate) cap_w: usize,
    pub(crate) cap_n: usize,
    pub(crate) cap_m: usize,
}

/// Weight source for [`OpenClEngine::ggml_gemv_dispatch_bound`]: a device mirror buffer
/// at a byte offset, or an owned SVM base pointer at a byte offset.
pub enum GgmlWeightBind<'a> {
    Device(&'a Buffer<cl_uchar>),
    /// # Safety
    /// The caller guarantees `*const u8` is a valid owned SVM allocation readable by
    /// this engine's device (the same contract as `ggml_gemv_async_from_svm`).
    Svm(*const u8),
}

/// Work-items cooperating per output row for the row-split GEMV kernels. The host must
/// launch `global = M * split` for the matching kernel (see `ggml_gemv_q4_k`).
pub(crate) fn gemv_split(label: &str) -> usize {
    match label {
        "q4_k" => 8,
        _ => 1,
    }
}

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
        let local = preferred_local_size(m, self.device_info.max_work_group_size);
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

    /// GEMV Q4_K **batcheado** (Fase 2, criterios C1/C4): `batch` candidatos en un
    /// único dispatch `[batch*m]`; los pesos se leen una vez por work-item y el
    /// L2/L1 los reutiliza entre candidatos. `inputs` = `batch*n`, `outputs` = `batch*m`.
    #[allow(clippy::too_many_arguments)]
    pub fn ggml_gemv_batched_q4_k(
        &self,
        m: usize,
        n: usize,
        weights: &[u8],
        inputs: &[f32],
        outputs: &mut [f32],
        batch: usize,
    ) -> Result<(), OpenClError> {
        assert_eq!(inputs.len(), batch * n);
        assert_eq!(outputs.len(), batch * m);
        validate_gemv_weights("q4_k", m, n, weights.len())?;
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let batch_i = batch as cl_int;
        let off_i = 0i64;

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
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, batch * n, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("input buffer: {e}")))?
        };
        let output_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                batch * m,
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("output buffer: {e}")))?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut weights_buf, CL_BLOCKING, 0, weights, &[])
                .map_err(|e| OpenClError::ClError(format!("write weights: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, inputs, &[])
                .map_err(|e| OpenClError::ClError(format!("write input: {e}")))?;
        }

        let local = preferred_local_size(m, self.device_info.max_work_group_size);
        let global = (((batch * m) + local - 1) / local) * local;

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.gemv_batched_q4_k)
                .set_arg(&m_i)
                .set_arg(&n_i)
                .set_arg(&off_i)
                .set_arg(&batch_i)
                .set_arg(&weights_buf)
                .set_arg(&input_buf)
                .set_arg(&output_buf)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue batched q4_k: {e}")))?
        };

        let wait = [kernel_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, outputs, &wait)
                .map_err(|e| OpenClError::ClError(format!("read output: {e}")))?;
        }
        debug!(
            "OpenCL ggml_gemv_batched_q4_k [{batch}×{m}×{n}] on {}",
            self.device_info.device_name
        );
        Ok(())
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

    pub fn ggml_gemv_dispatch(
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
        validate_gemv_weights(label, m, n, weights.len())?;
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let off_i = 0i64; // cl_long (kernel arg)

        let mut ws = self
            .sync_ws
            .lock()
            .map_err(|_| OpenClError::ClError("gemv workspace poisoned".into()))?;
        // Grow the persistent buffers only when a larger op arrives; otherwise reuse.
        if ws.cap_w < weights.len().max(1) {
            let cap = weights.len().max(1);
            ws.weights = Some(unsafe {
                Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, cap, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("weights buffer: {e}")))?
            });
            ws.cap_w = cap;
        }
        if ws.cap_n < n.max(1) {
            let cap = n.max(1);
            ws.input = Some(unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, cap, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("input buffer: {e}")))?
            });
            ws.cap_n = cap;
        }
        if ws.cap_m < m.max(1) {
            let cap = m.max(1);
            ws.output = Some(unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_WRITE_ONLY, cap, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("output buffer: {e}")))?
            });
            ws.cap_m = cap;
        }
        {
            let wbuf = ws.weights.as_mut().unwrap();
            unsafe {
                self.queue
                    .enqueue_write_buffer(wbuf, CL_BLOCKING, 0, weights, &[])
                    .map_err(|e| OpenClError::ClError(format!("write weights: {e}")))?;
            }
        }
        {
            let ibuf = ws.input.as_mut().unwrap();
            unsafe {
                self.queue
                    .enqueue_write_buffer(ibuf, CL_BLOCKING, 0, input, &[])
                    .map_err(|e| OpenClError::ClError(format!("write input: {e}")))?;
            }
        }

        let local = preferred_local_size(m, self.device_info.max_work_group_size);
        let global = ((m.saturating_mul(gemv_split(label)) + local - 1) / local) * local;

        // Must match HAYAI_X_TILE in ggml_gemv_q4.cl (tiled __local input).
        let local_bytes = 2048 * std::mem::size_of::<cl_float>();
        let kernel_event = unsafe {
            ExecuteKernel::new(kernel)
                .set_arg(&m_i)
                .set_arg(&n_i)
                .set_arg(&off_i)
                .set_arg(ws.weights.as_ref().unwrap())
                .set_arg(ws.input.as_ref().unwrap())
                .set_arg(ws.output.as_ref().unwrap())
                .set_arg_local_buffer(local_bytes)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue ggml_gemv_{label}: {e}")))?
        };

        let wait = [kernel_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(
                    ws.output.as_ref().unwrap(),
                    CL_BLOCKING,
                    0,
                    output,
                    &wait,
                )
                .map_err(|e| OpenClError::ClError(format!("read output: {e}")))?;
        }
        debug!(
            "OpenCL ggml_gemv_{label} [{m}×{n}] on {}",
            self.device_info.device_name
        );
        Ok(())
    }

    /// Synchronous GEMV reading the weights from a device mirror / owned SVM, reusing
    /// the persistent input/output workspace. No per-op `clCreateBuffer` and no weight
    /// upload — this is the resident path the planner prices.
    pub fn ggml_gemv_dispatch_bound(
        &self,
        kernel: &opencl3::kernel::Kernel,
        label: &str,
        m: usize,
        n: usize,
        weights: GgmlWeightBind<'_>,
        weight_off: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        assert_eq!(input.len(), n);
        assert_eq!(output.len(), m);
        validate_gemv_shape(label, m, n)?;
        let m_i = m as cl_int;
        let n_i = n as cl_int;
        let off_i = weight_off as i64;
        let mut ws = self
            .sync_ws
            .lock()
            .map_err(|_| OpenClError::ClError("gemv workspace poisoned".into()))?;
        if ws.cap_n < n.max(1) {
            let cap = n.max(1);
            ws.input = Some(unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, cap, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("input buffer: {e}")))?
            });
            ws.cap_n = cap;
        }
        if ws.cap_m < m.max(1) {
            let cap = m.max(1);
            ws.output = Some(unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_WRITE_ONLY, cap, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("output buffer: {e}")))?
            });
            ws.cap_m = cap;
        }
        {
            let ib = ws.input.as_mut().unwrap();
            unsafe {
                self.queue
                    .enqueue_write_buffer(ib, CL_BLOCKING, 0, input, &[])
                    .map_err(|e| OpenClError::ClError(format!("write input: {e}")))?;
            }
        }
        let local = preferred_local_size(m, self.device_info.max_work_group_size);
        let global = ((m.saturating_mul(gemv_split(label)) + local - 1) / local) * local;
        let local_bytes = 2048 * std::mem::size_of::<cl_float>();
        let ev = unsafe {
            let mut exec = ExecuteKernel::new(kernel);
            exec.set_arg(&m_i).set_arg(&n_i).set_arg(&off_i);
            let exec_ref = match weights {
                GgmlWeightBind::Device(b) => exec.set_arg(b),
                GgmlWeightBind::Svm(p) => exec.set_arg_svm(p),
            };
            exec_ref
                .set_arg(ws.input.as_ref().unwrap())
                .set_arg(ws.output.as_ref().unwrap())
                .set_arg_local_buffer(local_bytes)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue bound {label}: {e}")))?
        };
        let wait = [ev.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(ws.output.as_ref().unwrap(), CL_BLOCKING, 0, output, &wait)
                .map_err(|e| OpenClError::ClError(format!("read output: {e}")))?;
        }
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

    /// SpMM CSR del FFN disperso (DAG irregular, GGUF de `saor`):
    /// `Y[b][j] = sum_k X[b][col_idx[k]] * vals[k]`, un work-item por `(b, j)`.
    ///
    /// Topología vacía (τ alto): `row_ptr`/`col_idx`/`vals` vacíos devuelven
    /// ceros sin tocar OpenCL (no admite buffers de tamaño 0).
    #[allow(clippy::too_many_arguments)]
    pub fn spmm_csr(
        &self,
        x: &[f32],
        row_ptr: &[i32],
        col_idx: &[i32],
        vals: &[f32],
        d_in: usize,
        d_out: usize,
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        let batch = if d_in > 0 { x.len() / d_in } else { 0 };
        assert_eq!(output.len(), batch * d_out);
        if col_idx.is_empty() || vals.is_empty() {
            output.fill(0.0);
            return Ok(());
        }

        let mut x_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, x.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("x buffer: {e}")))?
        };
        let mut rp_buf = unsafe {
            Buffer::<cl_int>::create(&self.context, CL_MEM_READ_ONLY, row_ptr.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("row_ptr buffer: {e}")))?
        };
        let mut ci_buf = unsafe {
            Buffer::<cl_int>::create(&self.context, CL_MEM_READ_ONLY, col_idx.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("col_idx buffer: {e}")))?
        };
        let mut v_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, vals.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("vals buffer: {e}")))?
        };
        let output_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                output.len(),
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("output buffer: {e}")))?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut x_buf, CL_BLOCKING, 0, x, &[])
                .map_err(|e| OpenClError::ClError(format!("write x: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut rp_buf, CL_BLOCKING, 0, row_ptr, &[])
                .map_err(|e| OpenClError::ClError(format!("write row_ptr: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut ci_buf, CL_BLOCKING, 0, col_idx, &[])
                .map_err(|e| OpenClError::ClError(format!("write col_idx: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut v_buf, CL_BLOCKING, 0, vals, &[])
                .map_err(|e| OpenClError::ClError(format!("write vals: {e}")))?;
        }

        let global = batch * d_out;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.spmm_csr)
                .set_arg(&x_buf)
                .set_arg(&rp_buf)
                .set_arg(&ci_buf)
                .set_arg(&v_buf)
                .set_arg(&(d_in as cl_int))
                .set_arg(&(d_out as cl_int))
                .set_arg(&output_buf)
                .set_global_work_size(global)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue spmm_csr: {e}")))?
        };

        let wait = [kernel_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, output, &wait)
                .map_err(|e| OpenClError::ClError(format!("read spmm_csr: {e}")))?;
        }
        debug!(
            "OpenCL spmm_csr [{batch}×{d_out}×{d_in}] on {}",
            self.device_info.device_name
        );
        Ok(())
    }

    /// SpMM esparso batcheado desde bit-tensor + pesos F32 compartidos por capa
    /// (SparseAdj — Fase 2, criterio C4): `N` candidatos × `n_pos` tokens en un
    /// único dispatch. `adjs` es la concatenación de los bit-tensores
    /// (candidato-major, `conn = i*d_out+j`, LSB-first), `w` el F32 compartido
    /// `[d_out, d_in]`. Evita el gather del CSR por (candidato, capa, token).
    #[allow(clippy::too_many_arguments)]
    pub fn spmm_adj_batched(
        &self,
        x: &[f32],
        adjs: &[u8],
        w: &[f32],
        n_cands: usize,
        n_pos: usize,
        d_in: usize,
        d_out: usize,
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        let batch = n_cands * n_pos;
        assert_eq!(output.len(), batch * d_out);
        if adjs.is_empty() || w.is_empty() {
            output.fill(0.0);
            return Ok(());
        }
        validate_spmm_inputs(x.len(), adjs.len(), n_cands, n_pos, d_in, d_out)?;
        let w_need = d_in
            .checked_mul(d_out)
            .ok_or_else(|| OpenClError::ClError("spmm_adj: d_in*d_out overflow".into()))?;
        if w.len() < w_need {
            return Err(OpenClError::ClError(format!(
                "spmm_adj: weight buffer {} < {w_need} required",
                w.len()
            )));
        }

        let mut x_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, x.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_adj x buffer: {e}")))? 
        };
        let mut a_buf = unsafe {
            Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, adjs.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_adj adj buffer: {e}")))?
        };
        let mut w_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, w.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_adj w buffer: {e}")))?
        };
        let output_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                output.len(),
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("spmm_adj output buffer: {e}")))?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut x_buf, CL_BLOCKING, 0, x, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_adj x: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut a_buf, CL_BLOCKING, 0, adjs, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_adj adjs: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut w_buf, CL_BLOCKING, 0, w, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_adj w: {e}")))?;
        }

        let global = batch * d_out;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.spmm_adj_batched)
                .set_arg(&x_buf)
                .set_arg(&a_buf)
                .set_arg(&w_buf)
                .set_arg(&(d_in as cl_int))
                .set_arg(&(d_out as cl_int))
                .set_arg(&(n_pos as cl_int))
                .set_arg(&output_buf)
                .set_global_work_size(global)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue spmm_adj_batched: {e}")))?
        };

        let wait = [kernel_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, output, &wait)
                .map_err(|e| OpenClError::ClError(format!("read spmm_adj_batched: {e}")))?;
        }
        debug!(
            "OpenCL spmm_adj_batched [{batch}×{d_out}×{d_in}] on {}",
            self.device_info.device_name
        );
        Ok(())
    }

    /// SpMM esparso batcheado con dequant Q4_K en el kernel (Fase 2, criterio
    /// C1/C4): mismo contrato que [`spmm_adj_batched`] pero `w4` son los bytes Q4_K
    /// crudos del tensor (fila-mayor `[d_out, d_in]`, 144 B/bloque de 256) — se
    /// elimina el dequant F32 + upload de ~23 GB/gen en 27B/40B y el trabajo de
    /// dequant pasa a la GPU (C1).
    #[allow(clippy::too_many_arguments)]
    pub fn spmm_adj_batched_q4(
        &self,
        x: &[f32],
        adjs: &[u8],
        w4: &[u8],
        n_cands: usize,
        n_pos: usize,
        d_in: usize,
        d_out: usize,
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        let batch = n_cands * n_pos;
        assert_eq!(output.len(), batch * d_out);
        if adjs.is_empty() || w4.is_empty() {
            output.fill(0.0);
            return Ok(());
        }
        validate_spmm_inputs(x.len(), adjs.len(), n_cands, n_pos, d_in, d_out)?;
        validate_q4_k_weights(d_in, d_out, w4.len())?;

        let mut x_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, x.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_adj_q4 x buffer: {e}")))?
        };
        let mut a_buf = unsafe {
            Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, adjs.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_adj_q4 adj buffer: {e}")))?
        };
        let mut w4_buf = unsafe {
            Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, w4.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_adj_q4 w4 buffer: {e}")))?
        };
        let output_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                output.len(),
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("spmm_adj_q4 output buffer: {e}")))?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut x_buf, CL_BLOCKING, 0, x, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_adj_q4 x: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut a_buf, CL_BLOCKING, 0, adjs, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_adj_q4 adjs: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut w4_buf, CL_BLOCKING, 0, w4, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_adj_q4 w4: {e}")))?;
        }

        let global = batch * d_out;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.spmm_adj_batched_q4)
                .set_arg(&x_buf)
                .set_arg(&a_buf)
                .set_arg(&w4_buf)
                .set_arg(&(d_in as cl_int))
                .set_arg(&(d_out as cl_int))
                .set_arg(&(n_pos as cl_int))
                .set_arg(&output_buf)
                .set_global_work_size(global)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue spmm_adj_batched_q4: {e}")))?
        };

        let wait = [kernel_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, output, &wait)
                .map_err(|e| OpenClError::ClError(format!("read spmm_adj_batched_q4: {e}")))?;
        }
        debug!(
            "OpenCL spmm_adj_batched_q4 [{batch}×{d_out}×{d_in}] on {}",
            self.device_info.device_name
        );
        Ok(())
    }

    /// SpMM esparso batcheado con **dequant Q4_K en la GPU** (Fase 2, criterio
    /// C1/C4): sube el Q4 (8× menos PCIe que el F32) y encadena `dequant_q4_k_to_f32`
    /// → `spmm_adj_batched` en la MISMA cola (un único wait). Elimina el dequant CPU
    /// de 23 GB/gen y el upload F32; el trabajo de dequant pasa a la GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn spmm_adj_batched_q4gpu(
        &self,
        x: &[f32],
        adjs: &[u8],
        w4: &[u8],
        n_cands: usize,
        n_pos: usize,
        d_in: usize,
        d_out: usize,
        output: &mut [f32],
    ) -> Result<(), OpenClError> {
        let batch = n_cands * n_pos;
        let n = d_out * d_in;
        assert_eq!(output.len(), batch * d_out);
        if adjs.is_empty() || w4.is_empty() {
            output.fill(0.0);
            return Ok(());
        }
        // Same validation as `spmm_adj_batched_q4`: never launch a kernel whose
        // indexing would run past the adjacency/weight buffers.
        validate_spmm_inputs(x.len(), adjs.len(), n_cands, n_pos, d_in, d_out)?;
        validate_q4_k_weights(d_in, d_out, w4.len())?;

        let mut x_buf = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, x.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_q4gpu x buffer: {e}")))?
        };
        let mut a_buf = unsafe {
            Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, adjs.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_q4gpu adj buffer: {e}")))?
        };
        let mut w4_buf = unsafe {
            Buffer::<cl_uchar>::create(&self.context, CL_MEM_READ_ONLY, w4.len(), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("spmm_q4gpu w4 buffer: {e}")))?
        };
        let f32_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                n,
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("spmm_q4gpu f32 buffer: {e}")))?
        };
        let output_buf = unsafe {
            Buffer::<cl_float>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                output.len(),
                ptr::null_mut(),
            )
            .map_err(|e| OpenClError::ClError(format!("spmm_q4gpu output buffer: {e}")))?
        };

        unsafe {
            self.queue
                .enqueue_write_buffer(&mut x_buf, CL_NON_BLOCKING, 0, x, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_q4gpu x: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut a_buf, CL_NON_BLOCKING, 0, adjs, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_q4gpu adjs: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut w4_buf, CL_NON_BLOCKING, 0, w4, &[])
                .map_err(|e| OpenClError::ClError(format!("write spmm_q4gpu w4: {e}")))?;
        }

        let n_blocks = n / 256;
        let deq_event = unsafe {
            ExecuteKernel::new(&self.dequant_q4_k_to_f32)
                .set_arg(&w4_buf)
                .set_arg(&(n as cl_int))
                .set_arg(&f32_buf)
                .set_global_work_size(n_blocks)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue dequant_q4_k_to_f32: {e}")))?
        };

        let global = batch * d_out;
        let spmm_event = unsafe {
            ExecuteKernel::new(&self.spmm_adj_batched)
                .set_arg(&x_buf)
                .set_arg(&a_buf)
                .set_arg(&f32_buf)
                .set_arg(&(d_in as cl_int))
                .set_arg(&(d_out as cl_int))
                .set_arg(&(n_pos as cl_int))
                .set_arg(&output_buf)
                .set_global_work_size(global)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue spmm_adj_batched: {e}")))?
        };

        let wait = [deq_event.get(), spmm_event.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, output, &wait)
                .map_err(|e| OpenClError::ClError(format!("read spmm_adj_batched_q4gpu: {e}")))?;
        }
        debug!(
            "OpenCL spmm_adj_batched_q4gpu [{batch}×{d_out}×{d_in}] on {}",
            self.device_info.device_name
        );
        Ok(())
    }
}

fn preferred_local_size(m: usize, max_work_group: usize) -> usize {
    let cap = max_work_group.max(1);
    const CANDIDATES: [usize; 4] = [256, 128, 64, 32];
    for &ls in &CANDIDATES {
        if ls <= cap && m % ls == 0 {
            return ls;
        }
    }
    64.min(m.max(16)).min(cap)
}

/// Packed bytes per GEMV output row for `n` columns, per kernel layout. Returns
/// `None` for an unknown label or an `n` that is not a multiple of the block
/// size (the kernels would then read past the end of the weight buffer).
pub(crate) fn gemv_row_bytes(label: &str, n: usize) -> Option<usize> {
    let (block, bytes) = match label {
        "q4_0" => (32, 18),
        "q4_1" => (32, 20),
        "q5_0" => (32, 22),
        "q5_1" => (32, 24),
        "q8_0" => (32, 34),
        "q8_1" => (32, 36),
        "q8_k" => (256, 292),
        "iq4_nl" => (32, 18),
        "q2_k" => (256, 84),
        "q3_k" => (256, 110),
        "q4_k" => (256, 144),
        "q5_k" => (256, 176),
        "q6_k" => (256, 210),
        "iq4_xs" => (256, 136),
        "iq3_xxs" => (256, 98),
        "iq3_s" => (256, 110),
        "iq2_xxs" => (256, 66),
        "iq2_xs" => (256, 74),
        "iq2_s" => (256, 82),
        "f32" => return n.checked_mul(4),
        "f16" => return n.checked_mul(2),
        "bf16" => return n.checked_mul(2),
        _ => return None,
    };
    if block == 0 || n % block != 0 {
        return None;
    }
    (n / block).checked_mul(bytes)
}

/// Reject a GEMV whose weight buffer is too small for the `m × n` the kernel is
/// launched with. Without this an inconsistent GGUF can make the device read
/// past the end of the `cl_mem` (driver fault / garbage results).
pub(crate) fn validate_gemv_weights(
    label: &str,
    m: usize,
    n: usize,
    weights_len: usize,
) -> Result<(), OpenClError> {
    let row = gemv_row_bytes(label, n).ok_or_else(|| {
        OpenClError::ClError(format!(
            "ggml_gemv_{label}: n={n} is not a valid column count for this layout"
        ))
    })?;
    let expected = m.checked_mul(row).ok_or_else(|| {
        OpenClError::ClError(format!("ggml_gemv_{label}: weight size overflow for {m}x{n}"))
    })?;
    if weights_len < expected {
        return Err(OpenClError::ClError(format!(
            "ggml_gemv_{label}: weight buffer {weights_len} B < {expected} B required for {m}x{n}"
        )));
    }
    Ok(())
}

/// Validate only the `m`/`n` shape (no weight buffer): every block kernel derives
/// `blocks = n / block` and silently drops a shorter tail, so a non-multiple `n`
/// must be rejected before launch.
pub(crate) fn validate_gemv_shape(label: &str, m: usize, n: usize) -> Result<(), OpenClError> {
    if m == 0 || n == 0 {
        return Err(OpenClError::ClError(format!(
            "ggml_gemv_{label}: empty shape m={m} n={n}"
        )));
    }
    if gemv_row_bytes(label, n).is_none() {
        return Err(OpenClError::ClError(format!(
            "ggml_gemv_{label}: n={n} is not a valid column count (multiple of the block size)"
        )));
    }
    Ok(())
}

/// Validate the inputs shared by every batched sparse SpMM entry point.
fn validate_spmm_inputs(
    x_len: usize,
    adjs_len: usize,
    n_cands: usize,
    n_pos: usize,
    d_in: usize,
    d_out: usize,
) -> Result<usize, OpenClError> {
    let batch = n_cands
        .checked_mul(n_pos)
        .ok_or_else(|| OpenClError::ClError("spmm: n_cands*n_pos overflow".into()))?;
    let conns = d_in
        .checked_mul(d_out)
        .ok_or_else(|| OpenClError::ClError("spmm: d_in*d_out overflow".into()))?;
    if conns % 8 != 0 {
        return Err(OpenClError::ClError(format!(
            "spmm: d_in*d_out={conns} is not a multiple of 8"
        )));
    }
    let x_need = batch
        .checked_mul(d_in)
        .ok_or_else(|| OpenClError::ClError("spmm: x size overflow".into()))?;
    if x_len < x_need {
        return Err(OpenClError::ClError(format!(
            "spmm: x buffer {x_len} < {x_need} required"
        )));
    }
    let adj_need = n_cands
        .checked_mul(conns / 8)
        .ok_or_else(|| OpenClError::ClError("spmm: adj size overflow".into()))?;
    if adjs_len < adj_need {
        return Err(OpenClError::ClError(format!(
            "spmm: adjacency buffer {adjs_len} < {adj_need} required"
        )));
    }
    Ok(batch)
}

/// Validate the Q4_K weight payload for `d_out × d_in` (144 B per 256 values).
fn validate_q4_k_weights(d_in: usize, d_out: usize, w4_len: usize) -> Result<(), OpenClError> {
    if d_in % 256 != 0 {
        return Err(OpenClError::ClError(format!(
            "spmm q4: d_in={d_in} is not a multiple of 256"
        )));
    }
    let need = d_out
        .checked_mul(d_in / 256)
        .and_then(|b| b.checked_mul(144))
        .ok_or_else(|| OpenClError::ClError("spmm q4: weight size overflow".into()))?;
    if w4_len < need {
        return Err(OpenClError::ClError(format!(
            "spmm q4: weight buffer {w4_len} < {need} required for {d_out}x{d_in}"
        )));
    }
    Ok(())
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
            Err(_e) if !has_gpu => {
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

    /// Equivalencia del SpMM CSR del FFN disperso frente a la referencia densa
    /// enmascarada (CPU), tolerancia FP32 1e-5.
    #[test]
    fn opencl_spmm_csr_matches_cpu_when_available() {
        use hayai_model::sparse_dag::{sparse_dag_to_csr, spmm_dense_masked};

        let Some(engine) = init_engine_or_skip("OpenCL spmm_csr test") else {
            return;
        };

        // d_in=64, d_out=32, ~40% de conexiones activas, batch=3.
        let d_in = 64usize;
        let d_out = 32usize;
        let total = d_in * d_out;
        let mut adjacency = vec![0u8; total.div_ceil(8)];
        let mut weights = Vec::new();
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        for conn in 0..total {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let keep = (rng >> 33) % 100 < 40;
            if keep {
                adjacency[conn / 8] |= 1 << (conn % 8);
                weights.push(((conn % 13) as f32 - 6.0) * 0.25);
            }
        }
        assert!(!weights.is_empty());

        let x: Vec<f32> = (0..d_in * 3).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let expected = spmm_dense_masked(&x, &adjacency, &weights, d_in, d_out);
        let (row_ptr, col_idx, vals) = sparse_dag_to_csr(&adjacency, &weights, d_in, d_out);
        let mut got = vec![0.0f32; expected.len()];
        engine
            .spmm_csr(&x, &row_ptr, &col_idx, &vals, d_in, d_out, &mut got)
            .expect("OpenCL spmm_csr failed");

        let err = max_abs_diff(&expected, &got);
        assert!(
            err < 1e-5,
            "OpenCL spmm_csr vs CPU mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    /// Run one GEMV kernel against the CPU reference on deterministic bytes. Bytes
    /// are masked to `0x1F` so every embedded f16 scale stays finite (no NaN/Inf),
    /// which lets arbitrary bit patterns exercise the whole layout safely.
    fn gemv_parity<F>(
        engine: &OpenClEngine,
        label: &'static str,
        kernel: &opencl3::kernel::Kernel,
        m: usize,
        n: usize,
        row_bytes: usize,
        cpu: F,
    ) where
        F: Fn(&[u8], &[f32], &mut [f32]),
    {
        let mut weights = vec![0u8; m * row_bytes];
        for (i, b) in weights.iter_mut().enumerate() {
            *b = ((i * 131 + 17) & 0x1F) as u8;
        }
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.003 - 0.2).collect();
        let mut cpu_out = vec![0.0f32; m];
        let mut gpu_out = vec![0.0f32; m];
        cpu(&weights, &input, &mut cpu_out);
        let pending = engine
            .ggml_gemv_async(kernel, label, m, n, &weights, &input)
            .unwrap_or_else(|e| panic!("enqueue {label}: {e}"));
        gpu_out.copy_from_slice(&pending.wait().expect("wait"));
        let err = max_abs_diff(&cpu_out, &gpu_out);
        assert!(
            err < 1e-3,
            "OpenCL {label} vs CPU mismatch: {err} on {}",
            engine.device_info.device_name
        );
    }

    #[test]
    fn opencl_gemv_remaining_quants_match_cpu_when_available() {
        use hayai_model::iq2::{gemv_iq2_s, gemv_iq2_xs, gemv_iq2_xxs};
        use hayai_model::iq3::{gemv_iq3_s, gemv_iq3_xxs};
        use hayai_model::iq4::{gemv_iq4_nl, gemv_iq4_xs};
        use hayai_model::q2k::gemv_q2_k;
        use hayai_model::q3k::gemv_q3_k;
        use hayai_model::q5::{gemv_q5_0, gemv_q5_1};
        use hayai_model::q5k::gemv_q5_k;

        let Some(engine) = init_engine_or_skip("OpenCL all-quants gemv test") else {
            return;
        };
        let m = 32usize;
        let n = 256usize;
        let (b32, b256) = (n / 32, n / 256);

        gemv_parity(&engine, "q5_0", &engine.gemv_q5_0, m, n, b32 * 22, |w, i, o| {
            gemv_q5_0(m, n, w, i, o).unwrap()
        });
        gemv_parity(&engine, "q5_1", &engine.gemv_q5_1, m, n, b32 * 24, |w, i, o| {
            gemv_q5_1(m, n, w, i, o).unwrap()
        });
        gemv_parity(&engine, "q2_k", &engine.gemv_q2_k, m, n, b256 * 84, |w, i, o| {
            gemv_q2_k(m, n, w, i, o).unwrap()
        });
        gemv_parity(&engine, "q3_k", &engine.gemv_q3_k, m, n, b256 * 110, |w, i, o| {
            gemv_q3_k(m, n, w, i, o).unwrap()
        });
        gemv_parity(&engine, "q5_k", &engine.gemv_q5_k, m, n, b256 * 176, |w, i, o| {
            gemv_q5_k(m, n, w, i, o).unwrap()
        });
        gemv_parity(
            &engine,
            "iq4_nl",
            &engine.gemv_iq4_nl,
            m,
            n,
            b32 * 18,
            |w, i, o| gemv_iq4_nl(m, n, w, i, o).unwrap(),
        );
        gemv_parity(
            &engine,
            "iq4_xs",
            &engine.gemv_iq4_xs,
            m,
            n,
            b256 * 136,
            |w, i, o| gemv_iq4_xs(m, n, w, i, o).unwrap(),
        );
        gemv_parity(
            &engine,
            "iq3_xxs",
            &engine.gemv_iq3_xxs,
            m,
            n,
            b256 * 98,
            |w, i, o| gemv_iq3_xxs(m, n, w, i, o).unwrap(),
        );
        gemv_parity(
            &engine,
            "iq3_s",
            &engine.gemv_iq3_s,
            m,
            n,
            b256 * 110,
            |w, i, o| gemv_iq3_s(m, n, w, i, o).unwrap(),
        );
        gemv_parity(
            &engine,
            "iq2_xxs",
            &engine.gemv_iq2_xxs,
            m,
            n,
            b256 * 66,
            |w, i, o| gemv_iq2_xxs(m, n, w, i, o).unwrap(),
        );
        gemv_parity(
            &engine,
            "iq2_xs",
            &engine.gemv_iq2_xs,
            m,
            n,
            b256 * 74,
            |w, i, o| gemv_iq2_xs(m, n, w, i, o).unwrap(),
        );
        gemv_parity(
            &engine,
            "iq2_s",
            &engine.gemv_iq2_s,
            m,
            n,
            b256 * 82,
            |w, i, o| gemv_iq2_s(m, n, w, i, o).unwrap(),
        );
    }

    #[test]
    fn opencl_gemv_f32_f16_match_cpu_when_available() {
        use hayai_model::gguf::f16_to_f32;

        let Some(engine) = init_engine_or_skip("OpenCL f32/f16 gemv test") else {
            return;
        };
        let m = 16usize;
        let n = 64usize;
        let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.3).collect();

        // F32
        let mut w32 = vec![0u8; m * n * 4];
        for i in 0..(m * n) {
            let v = ((i % 17) as f32 - 8.0) * 0.03;
            w32[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut cpu = vec![0.0f32; m];
        for r in 0..m {
            let mut s = 0.0f32;
            for c in 0..n {
                let o = (r * n + c) * 4;
                s += f32::from_le_bytes(w32[o..o + 4].try_into().unwrap()) * input[c];
            }
            cpu[r] = s;
        }
        let p = engine
            .ggml_gemv_async(&engine.gemv_f32, "f32", m, n, &w32, &input)
            .expect("enqueue f32");
        let gpu = p.wait().expect("wait f32");
        assert!(max_abs_diff(&cpu, &gpu) < 1e-3, "f32 mismatch");

        // F16 (raw masked bytes; CPU uses the same f16 conversion)
        let mut w16 = vec![0u8; m * n * 2];
        for (i, b) in w16.iter_mut().enumerate() {
            *b = ((i * 131 + 17) & 0x1F) as u8;
        }
        let mut cpu = vec![0.0f32; m];
        for r in 0..m {
            let mut s = 0.0f32;
            for c in 0..n {
                let o = (r * n + c) * 2;
                s += f16_to_f32(u16::from_le_bytes(w16[o..o + 2].try_into().unwrap())) * input[c];
            }
            cpu[r] = s;
        }
        let p = engine
            .ggml_gemv_async(&engine.gemv_f16, "f16", m, n, &w16, &input)
            .expect("enqueue f16");
        let gpu = p.wait().expect("wait f16");
        assert!(max_abs_diff(&cpu, &gpu) < 1e-3, "f16 mismatch");
    }

    #[test]
    fn opencl_bf16_q8_1_q8_k_match_cpu_when_available() {
        use hayai_model::gguf::f16_to_f32;
        let Some(engine) = init_engine_or_skip("OpenCL bf16/q8_1/q8_k gemv test") else {
            return;
        };
        let m = 16usize;
        let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 0.7).collect();

        // BF16: bits<<16.
        let n = 64usize;
        let inp = &input[..n];
        let mut w = vec![0u8; m * n * 2];
        for i in 0..(m * n) {
            let v = ((i % 23) as f32 - 11.0) * 0.02;
            let bits = (v.to_bits() >> 16) as u16;
            w[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        let mut cpu = vec![0.0f32; m];
        for r in 0..m {
            let mut s = 0.0f32;
            for c in 0..n {
                let o = (r * n + c) * 2;
                let bits = (u16::from_le_bytes(w[o..o + 2].try_into().unwrap()) as u32) << 16;
                s += f32::from_bits(bits) * inp[c];
            }
            cpu[r] = s;
        }
        let gpu = engine
            .ggml_gemv_async(&engine.gemv_bf16, "bf16", m, n, &w, inp)
            .expect("enqueue bf16")
            .wait()
            .expect("wait bf16");
        assert!(max_abs_diff(&cpu, &gpu) < 1e-2, "bf16 mismatch");

        // Q8_1: { half d; half s; int8 qs[32] } = 36 B / 32 elems.
        let n = 64usize;
        let blocks = n / 32;
        let row_bytes = blocks * 36;
        let mut w = vec![0u8; m * row_bytes];
        for r in 0..m {
            for bi in 0..blocks {
                let base = r * row_bytes + bi * 36;
                w[base..base + 2].copy_from_slice(&0x3800u16.to_le_bytes()); // f16 0.5
                for j in 0..32 {
                    w[base + 4 + j] = (((r + bi + j) % 11) as i8 - 5) as u8;
                }
            }
        }
        let mut cpu = vec![0.0f32; m];
        for r in 0..m {
            let mut s = 0.0f32;
            for bi in 0..blocks {
                let base = r * row_bytes + bi * 36;
                let d = f16_to_f32(u16::from_le_bytes(w[base..base + 2].try_into().unwrap()));
                for j in 0..32 {
                    s += (w[base + 4 + j] as i8) as f32 * d * input[bi * 32 + j];
                }
            }
            cpu[r] = s;
        }
        let gpu = engine
            .ggml_gemv_async(&engine.gemv_q8_1, "q8_1", m, n, &w, &input[..n])
            .expect("enqueue q8_1")
            .wait()
            .expect("wait q8_1");
        assert!(max_abs_diff(&cpu, &gpu) < 1e-2, "q8_1 mismatch");

        // Q8_K: { float d; int8 qs[256]; int16 bsums[16] } = 292 B / 256 elems.
        let n = 256usize;
        let row_bytes = 292;
        let mut w = vec![0u8; m * row_bytes];
        for r in 0..m {
            let base = r * row_bytes;
            w[base..base + 4].copy_from_slice(&0.013f32.to_le_bytes());
            for j in 0..256 {
                w[base + 4 + j] = (((r * 7 + j) % 13) as i8 - 6) as u8;
            }
        }
        let mut cpu = vec![0.0f32; m];
        for r in 0..m {
            let base = r * row_bytes;
            let d = f32::from_le_bytes(w[base..base + 4].try_into().unwrap());
            let mut s = 0.0f32;
            for j in 0..256 {
                s += (w[base + 4 + j] as i8) as f32 * d * input[j];
            }
            cpu[r] = s;
        }
        let gpu = engine
            .ggml_gemv_async(&engine.gemv_q8_k, "q8_k", m, n, &w, &input)
            .expect("enqueue q8_k")
            .wait()
            .expect("wait q8_k");
        assert!(max_abs_diff(&cpu, &gpu) < 1e-2, "q8_k mismatch");
    }

    #[test]
    fn opencl_batched_q4_k_matches_cpu_when_available() {        use hayai_model::q4k::gemv_q4_k;

        let Some(engine) = init_engine_or_skip("OpenCL batched q4_k test") else {
            return;
        };
        let (m, n, batch) = (32usize, 256usize, 3usize);
        let mut weights = vec![0u8; m * 144];
        for (i, b) in weights.iter_mut().enumerate() {
            *b = ((i * 131 + 17) & 0x1F) as u8;
        }
        let inputs: Vec<f32> = (0..batch * n).map(|i| (i as f32) * 0.003 - 0.2).collect();
        let mut outputs = vec![0.0f32; batch * m];
        engine
            .ggml_gemv_batched_q4_k(m, n, &weights, &inputs, &mut outputs, batch)
            .expect("batched q4_k");
        let mut expected = vec![0.0f32; batch * m];
        for b in 0..batch {
            gemv_q4_k(m, n, &weights, &inputs[b * n..(b + 1) * n], &mut expected[b * m..(b + 1) * m])
                .unwrap();
        }
        assert!(max_abs_diff(&expected, &outputs) < 1e-3, "batched q4_k mismatch");
    }

    #[test]
    fn opencl_spmm_adj_batched_matches_cpu_when_available() {
        let Some(engine) = init_engine_or_skip("OpenCL spmm_adj test") else {
            return;
        };
        let (d_in, d_out, n_cands, n_pos) = (64usize, 32usize, 2usize, 2usize);
        let conns = d_in * d_out;
        let mut adjs = vec![0u8; n_cands * (conns / 8)];
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        for b in adjs.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((rng >> 33) as u8) & 0b0101_0101;
        }
        let w: Vec<f32> = (0..d_out * d_in).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let x: Vec<f32> = (0..n_cands * n_pos * d_in)
            .map(|i| (i as f32) * 0.002 - 0.1)
            .collect();
        let mut out = vec![0.0f32; n_cands * n_pos * d_out];
        engine
            .spmm_adj_batched(&x, &adjs, &w, n_cands, n_pos, d_in, d_out, &mut out)
            .expect("spmm_adj_batched");
        let mut exp = vec![0.0f32; out.len()];
        for c in 0..n_cands {
            for p in 0..n_pos {
                let b = c * n_pos + p;
                for j in 0..d_out {
                    let mut acc = 0.0f32;
                    for i in 0..d_in {
                        let conn = i * d_out + j;
                        if adjs[c * (conns / 8) + conn / 8] & (1 << (conn % 8)) != 0 {
                            acc += x[b * d_in + i] * w[j * d_in + i];
                        }
                    }
                    exp[b * d_out + j] = acc;
                }
            }
        }
        assert!(max_abs_diff(&exp, &out) < 1e-3, "spmm_adj_batched mismatch");
    }

    #[test]
    fn opencl_spmm_adj_batched_q4_matches_cpu_when_available() {
        use hayai_model::q4k::dequant_q4_k;

        let Some(engine) = init_engine_or_skip("OpenCL spmm_adj_q4 test") else {
            return;
        };
        let (d_in, d_out, n_cands, n_pos) = (256usize, 32usize, 1usize, 1usize);
        let conns = d_in * d_out;
        let mut adjs = vec![0u8; n_cands * (conns / 8)];
        let mut rng: u64 = 0xdead_beef_1234_5678;
        for b in adjs.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = ((rng >> 33) as u8) & 0b0101_0101;
        }
        let row_bytes = (d_in / 256) * 144;
        let mut w4 = vec![0u8; d_out * row_bytes];
        for (i, b) in w4.iter_mut().enumerate() {
            *b = ((i * 131 + 17) & 0x1F) as u8;
        }
        let wf = dequant_q4_k(&w4, d_out * d_in).unwrap();
        let x: Vec<f32> = (0..d_in).map(|i| (i as f32) * 0.002 - 0.1).collect();
        let mut out = vec![0.0f32; d_out];
        engine
            .spmm_adj_batched_q4(&x, &adjs, &w4, n_cands, n_pos, d_in, d_out, &mut out)
            .expect("spmm_adj_batched_q4");
        let mut out_gpu = vec![0.0f32; d_out];
        engine
            .spmm_adj_batched_q4gpu(&x, &adjs, &w4, n_cands, n_pos, d_in, d_out, &mut out_gpu)
            .expect("spmm_adj_batched_q4gpu");
        let mut exp = vec![0.0f32; d_out];
        for j in 0..d_out {
            let mut acc = 0.0f32;
            for i in 0..d_in {
                let conn = i * d_out + j;
                if adjs[conn / 8] & (1 << (conn % 8)) != 0 {
                    acc += x[i] * wf[j * d_in + i];
                }
            }
            exp[j] = acc;
        }
        assert!(max_abs_diff(&exp, &out) < 1e-3, "spmm_adj_batched_q4 mismatch");
        assert!(
            max_abs_diff(&exp, &out_gpu) < 1e-3,
            "spmm_adj_batched_q4gpu mismatch"
        );
    }
}


