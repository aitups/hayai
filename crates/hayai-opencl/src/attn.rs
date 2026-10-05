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

    /// Single-query attention over the device cache: `q` and `out` are
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
        let hd_i = hd as cl_int;
        let seq_i = seq as cl_int;
        let scale_f = scale;
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
        let mut ref_out = vec![0.0f32; n_heads * hd];
        for h in 0..n_heads {
            let kh = h % n_kv;
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
    }
}
