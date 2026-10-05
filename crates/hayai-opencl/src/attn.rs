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
        let groups_i = (n_heads / cache.n_kv.max(1)).max(1) as cl_int;
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
        let mk = |len: usize, data: &[f32]| -> Result<Buffer<cl_float>, OpenClError> {
            let mut b = unsafe {
                Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, len.max(1), ptr::null_mut())
                    .map_err(|e| OpenClError::ClError(format!("deltanet buf: {e}")))?
            };
            unsafe {
                self.queue
                    .enqueue_write_buffer(&mut b, CL_BLOCKING, 0, data, &[])
                    .map_err(|e| OpenClError::ClError(format!("deltanet write: {e}")))?;
            }
            Ok(b)
        };
        let bq = mk(q.len(), q)?;
        let bkk = mk(kk.len(), kk)?;
        let bv = mk(v.len(), v)?;
        let bdecay = mk(decay.len(), decay)?;
        let bbeta = mk(beta.len(), beta)?;
        let mut bout = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_WRITE_ONLY, out.len().max(1), ptr::null_mut())
                .map_err(|e| OpenClError::ClError(format!("deltanet out: {e}")))?
        };
        let (n_kh_i, n_vh_i, h_k_i, h_v_i) =
            (n_kh as cl_int, n_vh as cl_int, h_k as cl_int, h_v as cl_int);
        use opencl3::kernel::ExecuteKernel;
        let ev = unsafe {
            let mut exec = ExecuteKernel::new(&self.deltanet_step);
            exec.set_arg(&bq)
                .set_arg(&bkk)
                .set_arg(&bv)
                .set_arg(&bdecay)
                .set_arg(&bbeta)
                .set_arg(&state.s)
                .set_arg(&bout)
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
                .enqueue_read_buffer(&bout, CL_BLOCKING, 0, out, &wait)
                .map_err(|e| OpenClError::ClError(format!("deltanet read out: {e}")))?;
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
}
