use hayai_cpu::cpu_lut_matmul_q4;
use hayai_model::{GgmlType, ModelConfig, QuantMatrix};
use hayai_opencl::{
    discover_opencl_devices, OpenClDevicePool, OpenClEngine, OpenClError, PendingGemv,
};
use thiserror::Error;
use tracing::{info, warn};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionMode {
    Auto,
    OpenClDevice(String),
    CpuOnly,
}

impl ExecutionMode {
    /// Parse a `--device` style string: `"auto"`, `"cpu"`, or an OpenCL device
    /// name substring.
    pub fn parse(device: &str) -> Self {
        match device {
            "cpu" => ExecutionMode::CpuOnly,
            "auto" => ExecutionMode::Auto,
            other => ExecutionMode::OpenClDevice(other.to_string()),
        }
    }
}

#[derive(Error, Debug)]
pub enum OrchestratorError {
    #[error("OpenCL error: {0}")]
    OpenCl(#[from] OpenClError),
    #[error("{0}")]
    Msg(String),
}

pub struct EngineOrchestrator {
    pub mode: ExecutionMode,
    /// All OpenCL GPUs in the pool (empty ⇒ CPU-only).
    pub pool: OpenClDevicePool,
    pub model_config: ModelConfig,
}

// SAFETY: OpenCL handles are raw pointers; the orchestrator is always accessed
// behind a `Mutex` (API server serializes requests per model), so the underlying
// OpenCL objects are never used concurrently. `Sync` comes from `Arc<Mutex<_>>`.
unsafe impl Send for EngineOrchestrator {}

impl EngineOrchestrator {
    pub fn new(requested_mode: ExecutionMode, model_config: ModelConfig) -> Self {
        info!(
            "Initializing Engine Orchestrator with mode: {:?}",
            requested_mode
        );

        let (final_mode, pool) = match requested_mode {
            ExecutionMode::CpuOnly => {
                info!("CPU-Only mode explicitly requested. OpenCL disabled.");
                (
                    ExecutionMode::CpuOnly,
                    OpenClDevicePool::empty(),
                )
            }
            ExecutionMode::OpenClDevice(ref name) => {
                let devices = discover_opencl_devices();
                if let Some(dev_info) = devices.iter().find(|d| d.device_name.contains(name)) {
                    match OpenClEngine::init_device(dev_info) {
                        Ok(eng) => (
                            ExecutionMode::OpenClDevice(dev_info.device_name.clone()),
                            OpenClDevicePool::single(eng),
                        ),
                        Err(e) => {
                            warn!(
                                "Failed to init OpenCL device '{}': {}. Falling back to CPU-Only.",
                                name, e
                            );
                            (
                                ExecutionMode::CpuOnly,
                                OpenClDevicePool::empty(),
                            )
                        }
                    }
                } else {
                    warn!(
                        "Specified device '{}' not found. Falling back to CPU-Only.",
                        name
                    );
                    (
                        ExecutionMode::CpuOnly,
                        OpenClDevicePool::empty(),
                    )
                }
            }
            ExecutionMode::Auto => {
                // Small models lose on a discrete GPU: the per-layer FFN DMA (coarse
                // SVM transfers) costs more than the CPU GEMV it saves. Only offload
                // once the FFN is big enough (`hidden*ff`), overridable via
                // `HAYAI_FFN_MIN_GPU_PARAMS`.
                let ffn = model_config
                    .hidden_size
                    .saturating_mul(model_config.intermediate_size);
                let min_gpu = std::env::var("HAYAI_FFN_MIN_GPU_PARAMS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(6_000_000usize);
                if ffn < min_gpu {
                    info!(
                        "Auto: FFN hidden*ff={ffn} < {min_gpu} params → CPU (GPU DMA dominates for small models)"
                    );
                    (ExecutionMode::CpuOnly, OpenClDevicePool::empty())
                } else {
                    match OpenClDevicePool::try_init_all_gpus() {
                        Ok(pool) => {
                            info!(
                                "OpenCL pool ({}): {}",
                                pool.len(),
                                pool.names().join(" | ")
                            );
                            (ExecutionMode::Auto, pool)
                        }
                        Err(e) => {
                            info!(
                                "No OpenCL GPU available ({}). Falling back to CPU-Only mode.",
                                e
                            );
                            (ExecutionMode::CpuOnly, OpenClDevicePool::empty())
                        }
                    }
                }
            }
        };

        Self {
            mode: final_mode,
            pool,
            model_config,
        }
    }

    /// Primary FFN/scratch device (first in pool).
    pub fn opencl_engine(&self) -> Option<&OpenClEngine> {
        self.pool.engines.first()
    }

    /// Second device when present (legacy APU aux name).
    pub fn apu_engine(&self) -> Option<&OpenClEngine> {
        self.pool.engines.get(1)
    }

    pub fn execute_lut_matmul(
        &mut self,
        m: usize,
        n: usize,
        weights_q4: &[u8],
        lut: &[f32; 16],
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OrchestratorError> {
        if let Some(cl_engine) = self.opencl_engine() {
            match cl_engine.lut_matmul_q4(m, n, weights_q4, lut, input, output) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    warn!("OpenCL lut_matmul failed ({e}); falling back to CPU for this call");
                }
            }
        }

        cpu_lut_matmul_q4(m, n, weights_q4, lut, input, output);
        Ok(())
    }

    /// FFN / weight GEMV. With a GPU pool: OpenCL only (no quant-type CPU fallback).
    /// CPU path only when the pool is empty.
    pub fn execute_quant_gemv(
        &mut self,
        matrix: &QuantMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OrchestratorError> {
        if let Some(cl) = self.opencl_engine() {
            return self.gpu_gemv(cl, matrix, input, output);
        }
        matrix
            .gemv(input, output)
            .map_err(|e| OrchestratorError::Msg(e.to_string()))
    }

    /// GEMV **batcheado** de `batch` candidatos (Fase 2, criterios C1/C4): un único
    /// dispatch `[batch×M]` para Q4_K (los pesos se leen una vez); para otros
    /// tipos o sin GPU, N gemvs secuenciales. `inputs` = `batch*ncols`,
    /// `outputs` = `batch*nrows`.
    pub fn execute_quant_gemv_batched(
        &mut self,
        matrix: &QuantMatrix,
        inputs: &[f32],
        outputs: &mut [f32],
        batch: usize,
    ) -> Result<(), OrchestratorError> {
        let m = matrix.nrows;
        let n = matrix.ncols;
        if let Some(cl) = self.opencl_engine() {
            if matrix.ggml_type == GgmlType::Q4_K {
                if std::env::var("HAYAI_DEBUG_KERNEL").ok().as_deref() == Some("1") {
                    eprintln!("[gemv_batched] kernel Q4_K {m}×{n} batch={batch}");
                }
                return cl
                    .ggml_gemv_batched_q4_k(m, n, matrix.data(), inputs, outputs, batch)
                    .map_err(OrchestratorError::OpenCl);
            }
            for b in 0..batch {
                self.execute_quant_gemv(matrix, &inputs[b * n..(b + 1) * n], &mut outputs[b * m..(b + 1) * m])?;
            }
            return Ok(());
        }
        for b in 0..batch {
            matrix
                .gemv(&inputs[b * n..(b + 1) * n], &mut outputs[b * m..(b + 1) * m])
                .map_err(|e| OrchestratorError::Msg(e.to_string()))?;
        }
        Ok(())
    }

    /// Execute a GEMV honoring the **plan's per-op binding** (Track L2): ops bound
    /// to `Cpu`/`HostRowRead` (attention, router, norms) stay on CPU even when a
    /// GPU pool exists; `GpuAsync` ops (FFN/experts/output) use the pool. Passive
    /// `Discard` ops are no-ops. This is what the executor consumes instead of
    /// hardcoded call sites.
    pub fn execute_op(
        &mut self,
        _op: crate::LayerOpKind,
        binding: crate::exec_plan::OpBinding,
        matrix: &QuantMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OrchestratorError> {
        match binding.device {
            crate::exec_plan::OpDevice::GpuAsync => self.execute_quant_gemv(matrix, input, output),
            crate::exec_plan::OpDevice::Cpu | crate::exec_plan::OpDevice::HostRowRead => {
                matrix
                    .gemv(input, output)
                    .map_err(|e| OrchestratorError::Msg(e.to_string()))
            }
            crate::exec_plan::OpDevice::Discard => Ok(()),
        }
    }

    fn gpu_gemv(
        &self,
        cl: &OpenClEngine,
        matrix: &QuantMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OrchestratorError> {
        cl.ffn_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let m = matrix.nrows;
        let n = matrix.ncols;
        let w = matrix.data();
        let result = match matrix.ggml_type {
            GgmlType::Q4_0 => cl.ggml_gemv_q4_0(m, n, w, input, output),
            GgmlType::Q4_1 => cl.ggml_gemv_q4_1(m, n, w, input, output),
            GgmlType::Q8_0 => cl.ggml_gemv_q8_0(m, n, w, input, output),
            GgmlType::Q5_0 => cl.ggml_gemv_async(
                &cl.gemv_q5_0,
                "q5_0",
                m,
                n,
                w,
                input,
            )
            .and_then(|p| {
                let v = p.wait()?;
                output.copy_from_slice(&v);
                Ok(())
            }),
            GgmlType::Q5_1 => cl
                .ggml_gemv_async(&cl.gemv_q5_1, "q5_1", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::Q2_K => cl
                .ggml_gemv_async(&cl.gemv_q2_k, "q2_k", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::Q3_K => cl
                .ggml_gemv_async(&cl.gemv_q3_k, "q3_k", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::Q4_K => cl
                .ggml_gemv_async(&cl.gemv_q4_k, "q4_k", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::Q5_K => cl
                .ggml_gemv_async(&cl.gemv_q5_k, "q5_k", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::Q6_K => cl
                .ggml_gemv_async(&cl.gemv_q6_k, "q6_k", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::IQ4_NL => cl
                .ggml_gemv_async(&cl.gemv_iq4_nl, "iq4_nl", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::IQ4_XS => cl
                .ggml_gemv_async(&cl.gemv_iq4_xs, "iq4_xs", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::IQ3_XXS => cl
                .ggml_gemv_async(&cl.gemv_iq3_xxs, "iq3_xxs", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::IQ3_S => cl
                .ggml_gemv_async(&cl.gemv_iq3_s, "iq3_s", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::IQ2_XXS => cl
                .ggml_gemv_async(&cl.gemv_iq2_xxs, "iq2_xxs", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::IQ2_XS => cl
                .ggml_gemv_async(&cl.gemv_iq2_xs, "iq2_xs", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::IQ2_S => cl
                .ggml_gemv_async(&cl.gemv_iq2_s, "iq2_s", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::F32 => cl
                .ggml_gemv_async(&cl.gemv_f32, "f32", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::F16 => cl
                .ggml_gemv_async(&cl.gemv_f16, "f16", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            // BF16 / Q8_1 / Q8_K now have OpenCL kernels (async).
            GgmlType::BF16 => cl
                .ggml_gemv_async(&cl.gemv_bf16, "bf16", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::Q8_1 => cl
                .ggml_gemv_async(&cl.gemv_q8_1, "q8_1", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            GgmlType::Q8_K => cl
                .ggml_gemv_async(&cl.gemv_q8_k, "q8_k", m, n, w, input)
                .and_then(|p| {
                    let v = p.wait()?;
                    output.copy_from_slice(&v);
                    Ok(())
                }),
            other => {
                return Err(OrchestratorError::Msg(format!(
                    "OpenCL GEMV missing for {other:?} — GPU present, CPU fallback disabled (PRD). \
                     Need OpenCL kernel for this quant."
                )));
            }
        };
        result.map_err(OrchestratorError::from)
    }

    /// Begin async GEMV on a specific pool device (host weight bytes).
    pub fn begin_gemv_on(
        &self,
        role: usize,
        matrix: &QuantMatrix,
        input: &[f32],
    ) -> Result<PendingGemv, OrchestratorError> {
        if self.pool.is_empty() {
            return Err(OrchestratorError::Msg(
                "begin_gemv_on called with empty GPU pool".into(),
            ));
        }
        let eng = self.pool.for_role(role).ok_or_else(|| {
            OrchestratorError::Msg("begin_gemv_on called with empty GPU pool".into())
        })?;
        begin_gemv_engine(eng, matrix, input)
    }

    pub fn using_opencl(&self) -> bool {
        !self.pool.is_empty()
    }

    pub fn ffn_device_name(&self) -> &str {
        self.opencl_engine()
            .map(|e| e.device_info.device_name.as_str())
            .unwrap_or("CPU")
    }

    /// Report the FFN role→device assignment and the per-device GEMV counts
    /// (load-distribution / utilization proxy).
    pub fn format_compute_plan(&self) -> String {
        let mut s = String::from("Compute plan (FFN roles):\n");
        for (role, name) in ["gate", "up", "down"].iter().enumerate() {
            let dev = self
                .pool
                .for_role(role)
                .map(|e| e.device_info.device_name.as_str())
                .unwrap_or("CPU");
            s.push_str(&format!("  {name:<5} -> {dev}\n"));
        }
        for e in &self.pool.engines {
            s.push_str(&format!(
                "  {}: {} FFN GEMVs\n",
                e.device_info.device_name,
                e.ffn_calls.load(std::sync::atomic::Ordering::Relaxed)
            ));
        }
        s
    }

    pub fn hetero_devices_active(&self) -> bool {
        self.pool.len() >= 2
    }
}

pub fn begin_gemv_engine(
    eng: &OpenClEngine,
    m: &QuantMatrix,
    xn: &[f32],
) -> Result<PendingGemv, OrchestratorError> {
    let kernel_label = match m.ggml_type {
        GgmlType::Q4_0 => (&eng.gemv_q4_0, "q4_0"),
        GgmlType::Q4_1 => (&eng.gemv_q4_1, "q4_1"),
        GgmlType::Q8_0 => (&eng.gemv_q8_0, "q8_0"),
        GgmlType::Q5_0 => (&eng.gemv_q5_0, "q5_0"),
        GgmlType::Q5_1 => (&eng.gemv_q5_1, "q5_1"),
        GgmlType::Q2_K => (&eng.gemv_q2_k, "q2_k"),
        GgmlType::Q3_K => (&eng.gemv_q3_k, "q3_k"),
        GgmlType::Q4_K => (&eng.gemv_q4_k, "q4_k"),
        GgmlType::Q5_K => (&eng.gemv_q5_k, "q5_k"),
        GgmlType::Q6_K => (&eng.gemv_q6_k, "q6_k"),
        GgmlType::IQ4_NL => (&eng.gemv_iq4_nl, "iq4_nl"),
        GgmlType::IQ4_XS => (&eng.gemv_iq4_xs, "iq4_xs"),
        GgmlType::IQ3_XXS => (&eng.gemv_iq3_xxs, "iq3_xxs"),
        GgmlType::IQ3_S => (&eng.gemv_iq3_s, "iq3_s"),
        GgmlType::IQ2_XXS => (&eng.gemv_iq2_xxs, "iq2_xxs"),
        GgmlType::IQ2_XS => (&eng.gemv_iq2_xs, "iq2_xs"),
        GgmlType::IQ2_S => (&eng.gemv_iq2_s, "iq2_s"),
        GgmlType::F32 => (&eng.gemv_f32, "f32"),
        GgmlType::F16 => (&eng.gemv_f16, "f16"),
        GgmlType::BF16 => (&eng.gemv_bf16, "bf16"),
        GgmlType::Q8_1 => (&eng.gemv_q8_1, "q8_1"),
        GgmlType::Q8_K => (&eng.gemv_q8_k, "q8_k"),
        other => {
            return Err(OrchestratorError::Msg(format!(
                "OpenCL GEMV missing for {other:?} (GPU pool non-empty; no CPU FFN fallback)"
            )));
        }
    };
    eng.ffn_calls
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    eng.ggml_gemv_async(kernel_label.0, kernel_label.1, m.nrows, m.ncols, m.data(), xn)
        .map_err(OrchestratorError::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hayai_cpu::{fp32_matmul, max_abs_diff, unpack_q4_to_fp32};

    #[test]
    fn orchestrator_cpu_path_matches_fp32() {
        let config = ModelConfig::smollm_135m();
        let mut orch = EngineOrchestrator::new(ExecutionMode::CpuOnly, config);
        let m = 48;
        let n = 64;
        let lut = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ];
        let weights_q4: Vec<u8> = (0..(m * n / 2)).map(|i| (i % 256) as u8).collect();
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.02).collect();
        let mut out = vec![0.0f32; m];
        orch.execute_lut_matmul(m, n, &weights_q4, &lut, &input, &mut out)
            .unwrap();

        let dense = unpack_q4_to_fp32(m, n, &weights_q4, &lut);
        let mut expected = vec![0.0f32; m];
        fp32_matmul(m, n, &dense, &input, &mut expected);
        assert!(max_abs_diff(&out, &expected) < 1e-4);
    }

    /// Track L2: `execute_op` honors the plan's per-op binding — CPU-bound ops
    /// (router/attn) stay on CPU, GPU-bound FFN falls back to CPU with an empty
    /// pool, and passive ops are no-ops.
    #[test]
    fn execute_op_honors_binding() {
        use crate::LayerOpKind;
        use hayai_model::{gguf::write_minimal_gguf, GgufCatalog, MetadataValue};
        let path = std::env::temp_dir().join("hayai_exec_op_binding.gguf");
        write_minimal_gguf(
            &path,
            &[(
                "general.architecture",
                MetadataValue::String("llama".into()),
            )],
            &[(
                "w.weight",
                vec![4, 6],
                vec![
                    0.5, -1.0, 0.25, 2.0, 1.0, 0.0, -0.5, 1.5, 2.0, -2.0, 0.75, -1.25, 0.1, 0.2,
                    0.3, 0.4, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
                ],
            )],
        )
        .unwrap();
        let mut cat = GgufCatalog::open(&path).unwrap();
        let m = cat.load_quant_matrix("w.weight").unwrap();
        let _ = std::fs::remove_file(&path);

        let mut orch = EngineOrchestrator::new(ExecutionMode::CpuOnly, ModelConfig::smollm_135m());
        let x: Vec<f32> = (0..4).map(|i| i as f32 * 0.5 - 0.25).collect();

        // GpuAsync binding (FFN) with an empty pool → CPU fallback, matches CPU gemv.
        let mut out_gpu = vec![0.0f32; m.nrows];
        orch.execute_op(
            LayerOpKind::FfnGate,
            crate::exec_plan::op_binding(LayerOpKind::FfnGate),
            &m,
            &x,
            &mut out_gpu,
        )
        .unwrap();
        let mut out_cpu = vec![0.0f32; m.nrows];
        m.gemv(&x, &mut out_cpu).unwrap();
        assert!(max_abs_diff(&out_gpu, &out_cpu) < 1e-5);

        // CPU binding (router) is forced to CPU, same result.
        let mut out_router = vec![0.0f32; m.nrows];
        orch.execute_op(
            LayerOpKind::Router,
            crate::exec_plan::op_binding(LayerOpKind::Router),
            &m,
            &x,
            &mut out_router,
        )
        .unwrap();
        assert!(max_abs_diff(&out_router, &out_cpu) < 1e-5);

        // Discard binding is a no-op: output untouched.
        let mut out_discard = vec![9.9f32; m.nrows];
        orch.execute_op(
            LayerOpKind::LayerOutputScale,
            crate::exec_plan::op_binding(LayerOpKind::LayerOutputScale),
            &m,
            &x,
            &mut out_discard,
        )
        .unwrap();
        assert!(out_discard.iter().all(|&v| (v - 9.9).abs() < 1e-6));
    }
}
