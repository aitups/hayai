//! Device-resident FP32 KV cache + single-query attention decode (OpenCL).
//!
//! Kernel `hayai_attn_decode` (`kernels/attn_decode.cl`) runs an online-softmax
//! attention over a device KV cache, so the attention node executes on the accelerator
//! instead of the host. The cache is FP32 and per layer; it is grown by appending one
//! `n_kv * head_dim` row per token via [`OpenClEngine::kv_append`].

use crate::context::{OpenClEngine, OpenClError};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::types::{cl_float, cl_int, CL_BLOCKING};
use std::ptr;

/// A per-layer FP32 KV cache resident on the device: `[max_seq * n_kv * head_dim]`.
pub struct DeviceKvCache {
    k: Buffer<cl_float>,
    v: Buffer<cl_float>,
    pub n_kv: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    /// Number of positions written so far.
    pub len: usize,
}

impl DeviceKvCache {
    pub fn new(
        eng: &OpenClEngine,
        n_kv: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Result<Self, OpenClError> {
        let n = (max_seq * n_kv * head_dim).max(1);
        let k = unsafe {
            Buffer::<cl_float>::create(&eng.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("kv k buffer: {e}")))?
        };
        let v = unsafe {
            Buffer::<cl_float>::create(&eng.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("kv v buffer: {e}")))?
        };
        Ok(Self {
            k,
            v,
            n_kv,
            head_dim,
            max_seq,
            len: 0,
        })
    }

    pub fn resident_len(&self) -> usize {
        self.len.min(self.max_seq)
    }
}

impl OpenClEngine {
    /// Append one K/V row (`n_kv * head_dim`) at `pos` in the device cache.
    pub fn kv_append(
        &self,
        cache: &mut DeviceKvCache,
        pos: usize,
        k: &[f32],
        v: &[f32],
    ) -> Result<(), OpenClError> {
        let row = cache.n_kv * cache.head_dim;
        if pos >= cache.max_seq || k.len() != row || v.len() != row {
            return Err(OpenClError::ClError(format!(
                "kv_append shape: pos {pos} max {} k {} v {} row {row}",
                cache.max_seq,
                k.len(),
                v.len()
            )));
        }
        unsafe {
            self.queue
                .enqueue_write_buffer(&mut cache.k, CL_BLOCKING, pos * row * std::mem::size_of::<cl_float>(), k, &[])
                .map_err(|e| OpenClError::ClError(format!("kv write k: {e}")))?;
            self.queue
                .enqueue_write_buffer(&mut cache.v, CL_BLOCKING, pos * row * std::mem::size_of::<cl_float>(), v, &[])
                .map_err(|e| OpenClError::ClError(format!("kv write v: {e}")))?;
        }
        cache.len = cache.len.max(pos + 1);
        Ok(())
    }

    /// Append one K/V row from **device buffers** (device→device copy; no host round-trip).
    pub fn kv_append_dev(
        &self,
        cache: &mut DeviceKvCache,
        pos: usize,
        k_dev: &Buffer<cl_float>,
        v_dev: &Buffer<cl_float>,
    ) -> Result<(), OpenClError> {
        let row = cache.n_kv * cache.head_dim;
        if pos >= cache.max_seq {
            return Err(OpenClError::ClError(format!(
                "kv_append_dev pos {pos} >= max {}",
                cache.max_seq
            )));
        }
        let off = pos * row * std::mem::size_of::<cl_float>();
        let bytes = row * std::mem::size_of::<cl_float>();
        unsafe {
            self.queue
                .enqueue_copy_buffer(k_dev, &mut cache.k, 0, off, bytes, &[])
                .map_err(|e| OpenClError::ClError(format!("kv copy k: {e}")))?;
            self.queue
                .enqueue_copy_buffer(v_dev, &mut cache.v, 0, off, bytes, &[])
                .map_err(|e| OpenClError::ClError(format!("kv copy v: {e}")))?;
        }
        cache.len = cache.len.max(pos + 1);
        Ok(())
    }

    /// Attention decode with **device-resident** q and out (no host round-trip).
    pub fn attn_decode_dev(
        &self,
        q_dev: &Buffer<cl_float>,
        cache: &DeviceKvCache,
        out_dev: &Buffer<cl_float>,
        n_heads: usize,
        seq: usize,
        scale: f32,
    ) -> Result<(), OpenClError> {
        use opencl3::kernel::ExecuteKernel;
        let hd = cache.head_dim;
        let n_heads_i = n_heads as cl_int;
        let n_kv_i = cache.n_kv as cl_int;
        let groups_i = (n_heads / cache.n_kv.max(1)).max(1) as cl_int;
        let hd_i = hd as cl_int;
        let seq_i = seq as cl_int;
        let scale_f = scale;
        let local = 64usize.min(self.device_info.max_work_group_size.max(1));
        let global = ((n_heads + local - 1) / local) * local;
        unsafe {
            let mut exec = ExecuteKernel::new(&self.attn_decode);
            exec.set_arg(q_dev)
                .set_arg(&cache.k)
                .set_arg(&cache.v)
                .set_arg(out_dev)
                .set_arg(&n_heads_i)
                .set_arg(&n_kv_i)
                .set_arg(&groups_i)
                .set_arg(&hd_i)
                .set_arg(&seq_i)
                .set_arg(&scale_f)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue attn_decode_dev: {e}")))?;
        }
        Ok(())
    }
    /// `n_heads * head_dim`. Reuses the persistent input/output workspace.
    pub fn attn_decode(
        &self,
        q: &[f32],
        cache: &DeviceKvCache,
        out: &mut [f32],
        n_heads: usize,
        seq: usize,
        scale: f32,
    ) -> Result<(), OpenClError> {
        let hd = cache.head_dim;
        if q.len() != n_heads * hd || out.len() != n_heads * hd {
            return Err(OpenClError::ClError(format!(
                "attn_decode shape: q {} out {} expect {}",
                q.len(),
                out.len(),
                n_heads * hd
            )));
        }
        let mut ws = self
            .sync_ws
            .lock()
            .map_err(|_| OpenClError::ClError("gemv workspace poisoned".into()))?;
        let need = n_heads * hd;
        if ws.cap_n < need.max(1) {
            let cap = need.max(1);
            ws.input = Some(unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, cap, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("attn q buffer: {e}")))?
            });
            ws.cap_n = cap;
        }
        if ws.cap_m < need.max(1) {
            let cap = need.max(1);
            ws.output = Some(unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_WRITE_ONLY, cap, ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("attn out buffer: {e}")))?
            });
            ws.cap_m = cap;
        }
        {
            let qb = ws.input.as_mut().unwrap();
            unsafe {
                self.queue
                    .enqueue_write_buffer(qb, CL_BLOCKING, 0, q, &[])
                    .map_err(|e| OpenClError::ClError(format!("attn write q: {e}")))?;
            }
        }
        let n_heads_i = n_heads as cl_int;
        let n_kv_i = cache.n_kv as cl_int;
        let groups_i = (n_heads / cache.n_kv.max(1)).max(1) as cl_int;
        let hd_i = hd as cl_int;
        let seq_i = seq as cl_int;
        let scale_f = scale;
        // Long contexts: split the sequence across (head, chunk) work-items so the GPU
        // gets `n_heads * n_splits` threads instead of `n_heads` (audit B.2 #4). Short
        // contexts keep the single-work-item kernel (no combine launch).
        let target_chunk = 128usize;
        let max_splits = 16usize;
        let n_splits = if seq > target_chunk {
            ((seq + target_chunk - 1) / target_chunk)
                .min(max_splits)
                .max(1)
        } else {
            1
        };
        if n_splits > 1 {
            let chunk = (seq + n_splits - 1) / n_splits;
            let stride = hd + 2;
            let need_p = n_heads * n_splits * stride;
            let mut pws = self
                .attn_split_ws
                .lock()
                .map_err(|_| OpenClError::ClError("attn split workspace poisoned".into()))?;
            if pws.cap < need_p.max(1) {
                let cap = need_p.max(1);
                pws.partial = Some(unsafe {
                    Buffer::<cl_float>::create(&self.context, CL_MEM_READ_WRITE, cap, ptr::null_mut())
                        .map_err(|e| OpenClError::ClError(format!("attn partial buffer: {e}")))?
                });
                pws.cap = cap;
            }
            let local = 64usize.min(self.device_info.max_work_group_size.max(1));
            let n_splits_i = n_splits as cl_int;
            let chunk_i = chunk as cl_int;
            let total = n_heads * n_splits;
            let gsplit = ((total + local - 1) / local) * local;
            let ev_split = unsafe {
                let mut exec = ExecuteKernel::new(&self.attn_decode_split);
                exec.set_arg(ws.input.as_ref().unwrap())
                    .set_arg(&cache.k)
                    .set_arg(&cache.v)
                    .set_arg(pws.partial.as_ref().unwrap())
                    .set_arg(&n_heads_i)
                    .set_arg(&n_kv_i)
                    .set_arg(&groups_i)
                    .set_arg(&hd_i)
                    .set_arg(&seq_i)
                    .set_arg(&scale_f)
                    .set_arg(&n_splits_i)
                    .set_arg(&chunk_i)
                    .set_global_work_size(gsplit)
                    .set_local_work_size(local)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|e| OpenClError::ClError(format!("enqueue attn_split: {e}")))?
            };
            let gcomb = ((n_heads + local - 1) / local) * local;
            let ev_comb = unsafe {
                let mut exec = ExecuteKernel::new(&self.attn_decode_combine);
                exec.set_arg(pws.partial.as_ref().unwrap())
                    .set_arg(ws.output.as_ref().unwrap())
                    .set_arg(&n_heads_i)
                    .set_arg(&n_splits_i)
                    .set_arg(&hd_i)
                    .set_global_work_size(gcomb)
                    .set_local_work_size(local)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|e| OpenClError::ClError(format!("enqueue attn_combine: {e}")))?
            };
            let _ = ev_split.get();
            let wait = [ev_comb.get()];
            unsafe {
                self.queue
                    .enqueue_read_buffer(ws.output.as_ref().unwrap(), CL_BLOCKING, 0, out, &wait)
                    .map_err(|e| OpenClError::ClError(format!("attn read out: {e}")))?;
            }
            return Ok(());
        }
        // Global size = n_heads work-items, rounded to the device work-group size.
        let local = 64usize.min(self.device_info.max_work_group_size.max(1));
        let global = ((n_heads + local - 1) / local) * local;
        use opencl3::kernel::ExecuteKernel;
        let ev = unsafe {
            let mut exec = ExecuteKernel::new(&self.attn_decode);
            exec.set_arg(ws.input.as_ref().unwrap())
                .set_arg(&cache.k)
                .set_arg(&cache.v)
                .set_arg(ws.output.as_ref().unwrap())
                .set_arg(&n_heads_i)
                .set_arg(&n_kv_i)
                .set_arg(&groups_i)
                .set_arg(&hd_i)
                .set_arg(&seq_i)
                .set_arg(&scale_f)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue attn_decode: {e}")))?
        };
        let wait = [ev.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(ws.output.as_ref().unwrap(), CL_BLOCKING, 0, out, &wait)
                .map_err(|e| OpenClError::ClError(format!("attn read out: {e}")))?;
        }
        Ok(())
    }
}

/// Recurrent DeltaNet state on the device: `[n_v_heads * h_k * h_v]` f32, per layer.
pub struct DeviceDeltanetState {
    s: Buffer<cl_float>,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub h_k: usize,
    pub h_v: usize,
}

impl DeviceDeltanetState {
    pub fn new(
        eng: &OpenClEngine,
        n_k_heads: usize,
        n_v_heads: usize,
        h_k: usize,
        h_v: usize,
    ) -> Result<Self, OpenClError> {
        let n = (n_v_heads * h_k * h_v).max(1);
        let s = unsafe {
            Buffer::<cl_float>::create(&eng.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("deltanet state buffer: {e}")))?
        };
        Ok(Self {
            s,
            n_k_heads,
            n_v_heads,
            h_k,
            h_v,
        })
    }

    /// Reset the recurrent state to zero.
    pub fn zero(&mut self, eng: &OpenClEngine) -> Result<(), OpenClError> {
        let z = vec![0.0f32; (self.n_v_heads * self.h_k * self.h_v).max(1)];
        unsafe {
            eng.queue
                .enqueue_write_buffer(&mut self.s, CL_BLOCKING, 0, &z, &[])
        }
        .map_err(|e| OpenClError::ClError(format!("deltanet zero: {e}")))?;
        Ok(())
    }
}

/// Reusable transient buffers for [`OpenClEngine::deltanet_step`] so the ~6 per-layer
/// inputs/outputs are not `clCreateBuffer`'d and destroyed on every DeltaNet layer
/// (audit B.2 #4). Buffers grow on demand and are reused across calls.
#[derive(Default)]
pub struct DeltanetWorkspace {
    q: Option<Buffer<cl_float>>,
    kk: Option<Buffer<cl_float>>,
    v: Option<Buffer<cl_float>>,
    decay: Option<Buffer<cl_float>>,
    beta: Option<Buffer<cl_float>>,
    out: Option<Buffer<cl_float>>,
    cap_q: usize,
    cap_kk: usize,
    cap_v: usize,
    cap_decay: usize,
    cap_beta: usize,
    cap_out: usize,
}

fn ensure_buf(
    slot: &mut Option<Buffer<cl_float>>,
    cap: &mut usize,
    len: usize,
    flags: opencl3::types::cl_mem_flags,
    ctx: &opencl3::context::Context,
) -> Result<(), OpenClError> {
    let need = len.max(1);
    if *cap < need {
        *slot = Some(unsafe {
            Buffer::<cl_float>::create(ctx, flags, need, ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("deltanet buf: {e}")))?
        });
        *cap = need;
    }
    Ok(())
}

/// Reusable partial-state buffer for the FlashDecoding split-K attention path
/// (`hayai_attn_decode_split` → `hayai_attn_decode_combine`).
#[derive(Default)]
pub struct AttnSplitWorkspace {
    partial: Option<Buffer<cl_float>>,
    cap: usize,
}

impl OpenClEngine {
    /// One DeltaNet decode step: `decay`/`beta` are the per-value-head scalars the host
    /// derived; the device `state` is updated in place and `out` (`n_v_heads * h_v`) is
    /// returned. `q` is post L2-norm/q-scale, `kk`/`v` are the post-conv key/value.
    pub fn deltanet_step(
        &self,
        q: &[f32],
        kk: &[f32],
        v: &[f32],
        decay: &[f32],
        beta: &[f32],
        state: &DeviceDeltanetState,
        out: &mut [f32],
    ) -> Result<(), OpenClError> {
        let (n_kh, n_vh, h_k, h_v) = (
            state.n_k_heads,
            state.n_v_heads,
            state.h_k,
            state.h_v,
        );
        if q.len() != n_kh * h_k
            || kk.len() != n_kh * h_k
            || v.len() != n_vh * h_v
            || decay.len() != n_vh
            || beta.len() != n_vh
            || out.len() != n_vh * h_v
        {
            return Err(OpenClError::ClError("deltanet_step shape mismatch".into()));
        }
        let mut guard = self
            .deltanet_ws
            .lock()
            .map_err(|_| OpenClError::ClError("deltanet workspace poisoned".into()))?;
        let ws = &mut *guard;
        ensure_buf(&mut ws.q, &mut ws.cap_q, q.len(), CL_MEM_READ_ONLY, &self.context)?;
        ensure_buf(&mut ws.kk, &mut ws.cap_kk, kk.len(), CL_MEM_READ_ONLY, &self.context)?;
        ensure_buf(&mut ws.v, &mut ws.cap_v, v.len(), CL_MEM_READ_ONLY, &self.context)?;
        ensure_buf(&mut ws.decay, &mut ws.cap_decay, decay.len(), CL_MEM_READ_ONLY, &self.context)?;
        ensure_buf(&mut ws.beta, &mut ws.cap_beta, beta.len(), CL_MEM_READ_ONLY, &self.context)?;
        ensure_buf(&mut ws.out, &mut ws.cap_out, out.len(), CL_MEM_WRITE_ONLY, &self.context)?;
        {
            let wr = |b: &mut Buffer<cl_float>, data: &[f32]| -> Result<(), OpenClError> {
                unsafe {
                    self.queue
                        .enqueue_write_buffer(b, CL_BLOCKING, 0, data, &[])
                        .map_err(|e| OpenClError::ClError(format!("deltanet write: {e}")))?;
                }
                Ok(())
            };
            wr(ws.q.as_mut().unwrap(), q)?;
            wr(ws.kk.as_mut().unwrap(), kk)?;
            wr(ws.v.as_mut().unwrap(), v)?;
            wr(ws.decay.as_mut().unwrap(), decay)?;
            wr(ws.beta.as_mut().unwrap(), beta)?;
        }
        let (n_kh_i, n_vh_i, h_k_i, h_v_i) =
            (n_kh as cl_int, n_vh as cl_int, h_k as cl_int, h_v as cl_int);
        use opencl3::kernel::ExecuteKernel;
        let ev = unsafe {
            let mut exec = ExecuteKernel::new(&self.deltanet_step);
            exec.set_arg(ws.q.as_ref().unwrap())
                .set_arg(ws.kk.as_ref().unwrap())
                .set_arg(ws.v.as_ref().unwrap())
                .set_arg(ws.decay.as_ref().unwrap())
                .set_arg(ws.beta.as_ref().unwrap())
                .set_arg(&state.s)
                .set_arg(ws.out.as_ref().unwrap())
                .set_arg(&n_kh_i)
                .set_arg(&n_vh_i)
                .set_arg(&h_k_i)
                .set_arg(&h_v_i)
                .set_global_work_size(n_vh * h_v)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue deltanet_step: {e}")))?
        };
        let wait = [ev.get()];
        unsafe {
            self.queue
                .enqueue_read_buffer(ws.out.as_ref().unwrap(), CL_BLOCKING, 0, out, &wait)
                .map_err(|e| OpenClError::ClError(format!("deltanet read out: {e}")))?;
        }
        Ok(())
    }

    /// `out = silu(g) * u` on device buffers (`out` may alias `g`). Lets the FFN keep
    /// its gate/up merge on the device instead of downloading both to the host.
    pub fn silu_mul_dev(
        &self,
        g: &Buffer<cl_float>,
        u: &Buffer<cl_float>,
        out: &Buffer<cl_float>,
        n: usize,
    ) -> Result<(), OpenClError> {
        if n == 0 {
            return Ok(());
        }
        use opencl3::kernel::ExecuteKernel;
        let n_i = n as cl_int;
        let local = 256usize.min(self.device_info.max_work_group_size.max(1));
        let global = ((n + local - 1) / local) * local;
        unsafe {
            let mut exec = ExecuteKernel::new(&self.silu_mul);
            exec.set_arg(g)
                .set_arg(u)
                .set_arg(out)
                .set_arg(&n_i)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue silu_mul: {e}")))?;
        }
        Ok(())
    }

    /// Row-wise RMSNorm `y = x * rsqrt(mean(x^2) + eps) * w` on device buffers.
    pub fn rms_norm_dev(
        &self,
        x: &Buffer<cl_float>,
        w: &Buffer<cl_float>,
        y: &Buffer<cl_float>,
        n: usize,
        eps: f32,
    ) -> Result<(), OpenClError> {
        if n == 0 {
            return Ok(());
        }
        use opencl3::kernel::ExecuteKernel;
        let n_i = n as cl_int;
        let eps_f = eps;
        let max_ls = 256usize.min(self.device_info.max_work_group_size.max(1));
        let mut ls = 1usize;
        while ls * 2 <= max_ls {
            ls *= 2;
        }
        let local_bytes = ls * std::mem::size_of::<cl_float>();
        unsafe {
            let mut exec = ExecuteKernel::new(&self.rms_norm);
            exec.set_arg(x)
                .set_arg(w)
                .set_arg(y)
                .set_arg(&n_i)
                .set_arg(&eps_f)
                .set_arg_local_buffer(local_bytes)
                .set_global_work_size(ls)
                .set_local_work_size(ls)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue rms_norm: {e}")))?;
        }
        Ok(())
    }

    /// `x += y` on device buffers (residual stream stays on-device).
    pub fn residual_add_dev(
        &self,
        x: &Buffer<cl_float>,
        y: &Buffer<cl_float>,
        n: usize,
    ) -> Result<(), OpenClError> {
        if n == 0 {
            return Ok(());
        }
        use opencl3::kernel::ExecuteKernel;
        let n_i = n as cl_int;
        let local = 256usize.min(self.device_info.max_work_group_size.max(1));
        let global = ((n + local - 1) / local) * local;
        unsafe {
            let mut exec = ExecuteKernel::new(&self.residual_add);
            exec.set_arg(x)
                .set_arg(y)
                .set_arg(&n_i)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue residual_add: {e}")))?;
        }
        Ok(())
    }

    /// In-place rotary embedding of `x` (`n_rows × head_dim`, GPT-NeoX pair layout)
    /// using a per-position `(cos, sin)` table of `rotary_dim/2` entries.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_apply_dev(
        &self,
        x: &Buffer<cl_float>,
        n_rows: usize,
        head_dim: usize,
        rotary_dim: usize,
        cos: &Buffer<cl_float>,
        sin: &Buffer<cl_float>,
    ) -> Result<(), OpenClError> {
        let half = rotary_dim / 2;
        let total = n_rows * half;
        if total == 0 {
            return Ok(());
        }
        use opencl3::kernel::ExecuteKernel;
        let (nr_i, hd_i, rd_i) = (n_rows as cl_int, head_dim as cl_int, rotary_dim as cl_int);
        let local = 64usize.min(self.device_info.max_work_group_size.max(1));
        let global = ((total + local - 1) / local) * local;
        unsafe {
            let mut exec = ExecuteKernel::new(&self.rope_apply);
            exec.set_arg(x)
                .set_arg(&nr_i)
                .set_arg(&hd_i)
                .set_arg(&rd_i)
                .set_arg(cos)
                .set_arg(sin)
                .set_global_work_size(global)
                .set_local_work_size(local)
                .enqueue_nd_range(&self.queue)
                .map_err(|e| OpenClError::ClError(format!("enqueue rope_apply: {e}")))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attn_decode_matches_reference() {
        // Auto-skip when no OpenCL device is present (CI).
        let Ok(eng) = OpenClEngine::try_init_any() else {
            eprintln!("attn_decode parity: SKIP (no OpenCL device)");
            return;
        };
        let (n_heads, n_kv, hd, seq) = (4usize, 2usize, 8usize, 5usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let q: Vec<f32> = (0..n_heads * hd).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let k: Vec<f32> = (0..seq * n_kv * hd).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
        let v: Vec<f32> = (0..seq * n_kv * hd).map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2).collect();

        // Reference (host).
        let groups = n_heads / n_kv;
        let mut ref_out = vec![0.0f32; n_heads * hd];
        for h in 0..n_heads {
            let kh = h / groups;
            let mut scores = vec![0.0f32; seq];
            let mut m = f32::NEG_INFINITY;
            for j in 0..seq {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[h * hd + d] * k[(j * n_kv + kh) * hd + d];
                }
                scores[j] = dot * scale;
                m = m.max(scores[j]);
            }
            let mut l = 0.0f32;
            for j in 0..seq {
                scores[j] = (scores[j] - m).exp();
                l += scores[j];
            }
            for d in 0..hd {
                let mut acc = 0.0f32;
                for j in 0..seq {
                    acc += scores[j] * v[(j * n_kv + kh) * hd + d];
                }
                ref_out[h * hd + d] = acc / l;
            }
        }

        let mut cache = DeviceKvCache::new(&eng, n_kv, hd, seq).unwrap();
        for j in 0..seq {
            let krow = &k[j * n_kv * hd..(j + 1) * n_kv * hd];
            let vrow = &v[j * n_kv * hd..(j + 1) * n_kv * hd];
            eng.kv_append(&mut cache, j, krow, vrow).unwrap();
        }
        let mut out = vec![0.0f32; n_heads * hd];
        eng.attn_decode(&q, &cache, &mut out, n_heads, seq, scale)
            .unwrap();
        for (a, b) in out.iter().zip(ref_out.iter()) {
            assert!((a - b).abs() < 1e-4, "attn mismatch: {a} vs {b}");
        }

        // Long context exercises the FlashDecoding split-K + combine path.
        let (nh2, nkv2, hd2, seq2) = (8usize, 2usize, 16usize, 300usize);
        let scale2 = 1.0 / (hd2 as f32).sqrt();
        let q2: Vec<f32> = (0..nh2 * hd2).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let k2: Vec<f32> = (0..seq2 * nkv2 * hd2)
            .map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let v2: Vec<f32> = (0..seq2 * nkv2 * hd2)
            .map(|i| ((i * 3 % 7) as f32 - 3.0) * 0.2)
            .collect();
        let g2 = nh2 / nkv2;
        let mut ref2 = vec![0.0f32; nh2 * hd2];
        for h in 0..nh2 {
            let kh = h / g2;
            let mut scores = vec![0.0f32; seq2];
            let mut m = f32::NEG_INFINITY;
            for j in 0..seq2 {
                let mut dot = 0.0f32;
                for d in 0..hd2 {
                    dot += q2[h * hd2 + d] * k2[(j * nkv2 + kh) * hd2 + d];
                }
                scores[j] = dot * scale2;
                m = m.max(scores[j]);
            }
            let mut l = 0.0f32;
            for j in 0..seq2 {
                scores[j] = (scores[j] - m).exp();
                l += scores[j];
            }
            for d in 0..hd2 {
                let mut acc = 0.0f32;
                for j in 0..seq2 {
                    acc += scores[j] * v2[(j * nkv2 + kh) * hd2 + d];
                }
                ref2[h * hd2 + d] = acc / l;
            }
        }
        let mut cache2 = DeviceKvCache::new(&eng, nkv2, hd2, seq2).unwrap();
        for j in 0..seq2 {
            let krow = &k2[j * nkv2 * hd2..(j + 1) * nkv2 * hd2];
            let vrow = &v2[j * nkv2 * hd2..(j + 1) * nkv2 * hd2];
            eng.kv_append(&mut cache2, j, krow, vrow).unwrap();
        }
        let mut out2 = vec![0.0f32; nh2 * hd2];
        eng.attn_decode(&q2, &cache2, &mut out2, nh2, seq2, scale2)
            .unwrap();
        for (a, b) in out2.iter().zip(ref2.iter()) {
            assert!((a - b).abs() < 1e-4, "attn split mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn deltanet_step_matches_reference() {
        let Ok(eng) = OpenClEngine::try_init_any() else {
            eprintln!("deltanet parity: SKIP (no OpenCL device)");
            return;
        };
        let (n_k, n_v, h_k, h_v) = (2usize, 4usize, 4usize, 3usize);
        let q: Vec<f32> = (0..n_k * h_k).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let kk: Vec<f32> = (0..n_k * h_k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let v: Vec<f32> = (0..n_v * h_v).map(|i| ((i % 6) as f32 - 3.0) * 0.2).collect();
        let decay: Vec<f32> = vec![0.9, 0.8, 0.7, 0.6];
        let beta: Vec<f32> = vec![0.5, 0.4, 0.3, 0.2];

        // Reference: two steps from a zero state.
        let mut ref_state = vec![0.0f32; n_v * h_k * h_v];
        let step = |state: &mut [f32], out: &mut [f32]| {
            for vh in 0..n_v {
                let kh = vh % n_k;
                let s = &mut state[vh * h_k * h_v..(vh + 1) * h_k * h_v];
                for vd in 0..h_v {
                    for ki in 0..h_k {
                        s[ki * h_v + vd] *= decay[vh];
                    }
                    let mut kv_mem = 0.0f32;
                    for ki in 0..h_k {
                        kv_mem += s[ki * h_v + vd] * kk[kh * h_k + ki];
                    }
                    let delta = (v[vh * h_v + vd] - kv_mem) * beta[vh];
                    let mut o = 0.0f32;
                    for ki in 0..h_k {
                        s[ki * h_v + vd] += kk[kh * h_k + ki] * delta;
                        o += s[ki * h_v + vd] * q[kh * h_k + ki];
                    }
                    out[vh * h_v + vd] = o;
                }
            }
        };
        let mut ref_out = vec![0.0f32; n_v * h_v];
        step(&mut ref_state, &mut ref_out); // step 1
        step(&mut ref_state, &mut ref_out); // step 2 (state carries)

        let mut dev = DeviceDeltanetState::new(&eng, n_k, n_v, h_k, h_v).unwrap();
        dev.zero(&eng).unwrap();
        let mut out = vec![0.0f32; n_v * h_v];
        eng.deltanet_step(&q, &kk, &v, &decay, &beta, &dev, &mut out)
            .unwrap();
        eng.deltanet_step(&q, &kk, &v, &decay, &beta, &dev, &mut out)
            .unwrap();
        for (a, b) in out.iter().zip(ref_out.iter()) {
            assert!((a - b).abs() < 1e-4, "deltanet mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn nn_ops_match_reference() {
        use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
        use opencl3::types::CL_BLOCKING;
        use std::ptr;

        let Ok(eng) = OpenClEngine::try_init_any() else {
            eprintln!("nn_ops parity: SKIP (no OpenCL device)");
            return;
        };
        let n = 1000usize;
        let upload = |data: &[f32], flags| -> Buffer<cl_float> {
            let mut b = unsafe {
                Buffer::<cl_float>::create(&eng.context, flags, n, ptr::null_mut()).unwrap()
            };
            unsafe {
                eng.queue
                    .enqueue_write_buffer(&mut b, CL_BLOCKING, 0, data, &[])
                    .unwrap();
            }
            b
        };
        let g: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let u: Vec<f32> = (0..n).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();
        let bg = upload(&g, CL_MEM_READ_ONLY);
        let bu = upload(&u, CL_MEM_READ_ONLY);
        let bo = unsafe {
            Buffer::<cl_float>::create(&eng.context, CL_MEM_WRITE_ONLY, n, ptr::null_mut()).unwrap()
        };
        eng.silu_mul_dev(&bg, &bu, &bo, n).unwrap();
        let mut got = vec![0.0f32; n];
        unsafe {
            eng.queue
                .enqueue_read_buffer(&bo, CL_BLOCKING, 0, &mut got, &[])
                .unwrap();
        }
        for i in 0..n {
            let want = (g[i] / (1.0 + (-g[i]).exp())) * u[i];
            assert!((got[i] - want).abs() < 1e-4, "silu_mul {i}: {} vs {}", got[i], want);
        }

        let x: Vec<f32> = (0..n).map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.2).collect();
        let w: Vec<f32> = (0..n).map(|i| 1.0 + (i % 3) as f32 * 0.1).collect();
        let eps = 1e-5f32;
        let bx = upload(&x, CL_MEM_READ_ONLY);
        let bw = upload(&w, CL_MEM_READ_ONLY);
        let by = unsafe {
            Buffer::<cl_float>::create(&eng.context, CL_MEM_WRITE_ONLY, n, ptr::null_mut()).unwrap()
        };
        eng.rms_norm_dev(&bx, &bw, &by, n, eps).unwrap();
        let mut goty = vec![0.0f32; n];
        unsafe {
            eng.queue
                .enqueue_read_buffer(&by, CL_BLOCKING, 0, &mut goty, &[])
                .unwrap();
        }
        let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (mean + eps).sqrt();
        for i in 0..n {
            let want = x[i] * inv * w[i];
            assert!((goty[i] - want).abs() < 1e-3, "rms_norm {i}: {} vs {}", goty[i], want);
        }
    }

    #[test]
    fn residual_add_and_rope_match_reference() {
        use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
        use opencl3::types::{cl_float, CL_BLOCKING};
        use std::ptr;

        let Ok(eng) = OpenClEngine::try_init_any() else {
            eprintln!("dev ops parity: SKIP (no OpenCL device)");
            return;
        };
        let mk = |data: &[f32], flags| {
            let n = data.len();
            let mut b = unsafe {
                Buffer::<cl_float>::create(&eng.context, flags, n.max(1), ptr::null_mut()).unwrap()
            };
            unsafe {
                eng.queue
                    .enqueue_write_buffer(&mut b, CL_BLOCKING, 0, data, &[])
                    .unwrap();
            }
            b
        };
        // residual_add
        let n = 64usize;
        let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| 0.5 * i as f32).collect();
        let bx = mk(&x, CL_MEM_READ_WRITE);
        let by = mk(&y, CL_MEM_READ_ONLY);
        eng.residual_add_dev(&bx, &by, n).unwrap();
        let mut got = vec![0.0f32; n];
        unsafe {
            eng.queue
                .enqueue_read_buffer(&bx, CL_BLOCKING, 0, &mut got, &[])
                .unwrap();
        }
        for i in 0..n {
            assert!((got[i] - (x[i] + y[i])).abs() < 1e-6);
        }

        // rope_apply (rotate_half pairs i, i+half within head_dim)
        let (rows, hd, rd) = (2usize, 8usize, 8usize);
        let half = rd / 2;
        let cos: Vec<f32> = (0..half).map(|i| (0.3 * i as f32).cos()).collect();
        let sin: Vec<f32> = (0..half).map(|i| (0.3 * i as f32).sin()).collect();
        let xr: Vec<f32> = (0..rows * hd).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let bxr = mk(&xr, CL_MEM_READ_WRITE);
        let bc = mk(&cos, CL_MEM_READ_ONLY);
        let bs = mk(&sin, CL_MEM_READ_ONLY);
        eng.rope_apply_dev(&bxr, rows, hd, rd, &bc, &bs).unwrap();
        let mut gr = vec![0.0f32; rows * hd];
        unsafe {
            eng.queue
                .enqueue_read_buffer(&bxr, CL_BLOCKING, 0, &mut gr, &[])
                .unwrap();
        }
        for r in 0..rows {
            for p in 0..half {
                let a = xr[r * hd + p];
                let b = xr[r * hd + half + p];
                let want0 = a * cos[p] - b * sin[p];
                let want1 = a * sin[p] + b * cos[p];
                assert!((gr[r * hd + p] - want0).abs() < 1e-5, "rope 0");
                assert!((gr[r * hd + half + p] - want1).abs() < 1e-5, "rope 1");
            }
        }
    }
}
