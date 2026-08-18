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
                    OpenClDevicePool { engines: Vec::new() },
                )
            }
            ExecutionMode::OpenClDevice(ref name) => {
                let devices = discover_opencl_devices();
                if let Some(dev_info) = devices.iter().find(|d| d.device_name.contains(name)) {
                    match OpenClEngine::init_device(dev_info) {
                        Ok(eng) => (
                            ExecutionMode::OpenClDevice(dev_info.device_name.clone()),
                            OpenClDevicePool {
                                engines: vec![eng],
                            },
                        ),
                        Err(e) => {
                            warn!(
                                "Failed to init OpenCL device '{}': {}. Falling back to CPU-Only.",
                                name, e
                            );
                            (
                                ExecutionMode::CpuOnly,
                                OpenClDevicePool { engines: Vec::new() },
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
                        OpenClDevicePool { engines: Vec::new() },
                    )
                }
            }
            ExecutionMode::Auto => match OpenClDevicePool::try_init_all_gpus() {
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
                    (
                        ExecutionMode::CpuOnly,
                        OpenClDevicePool { engines: Vec::new() },
                    )
                }
            },
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

    fn gpu_gemv(
        &self,
        cl: &OpenClEngine,
        matrix: &QuantMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), OrchestratorError> {
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
            // BF16 only appears in small Gemma4 PLE proj — CPU path is fine.
            GgmlType::BF16 => {
                return matrix
                    .gemv(input, output)
                    .map_err(|e| OrchestratorError::Msg(e.to_string()));
            }
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
        let eng = self.pool.for_role(role);
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
        other => {
            return Err(OrchestratorError::Msg(format!(
                "OpenCL GEMV missing for {other:?} (GPU pool non-empty; no CPU FFN fallback)"
            )));
        }
    };
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
}
