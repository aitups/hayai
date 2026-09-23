use crate::device::{discover_opencl_devices, DeviceKind, OpenClDeviceInfo};
use crate::pool::OpenClDevicePool; // try_init_hetero
use hayai_kernels::opencl_program_source;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::Kernel;
use opencl3::program::Program;

use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum OpenClError {
    #[error("No suitable OpenCL device found on system")]
    NoDeviceFound,
    #[error("OpenCL runtime error: {0}")]
    ClError(String),
    #[error("OpenCL 3.0 required, but device reports {0}")]
    UnsupportedPlatformVersion(String),
}

pub struct OpenClEngine {
    pub device_info: OpenClDeviceInfo,
    pub context: Context,
    pub queue: CommandQueue,
    pub lut_kernel: Kernel,
    pub gemv_q4_0: Kernel,
    pub gemv_q4_1: Kernel,
    pub gemv_q8_0: Kernel,
    pub gemv_q5_0: Kernel,
    pub gemv_q5_1: Kernel,
    pub gemv_q2_k: Kernel,
    pub gemv_q3_k: Kernel,
    pub gemv_q4_k: Kernel,
    /// Batcheado Q4_K: N candidatos en un dispatch (Fase 2, criterios C1/C4).
    pub gemv_batched_q4_k: Kernel,
    pub gemv_q5_k: Kernel,
    pub gemv_q6_k: Kernel,
    pub gemv_iq4_nl: Kernel,
    pub gemv_iq4_xs: Kernel,
    pub gemv_iq3_xxs: Kernel,
    pub gemv_iq3_s: Kernel,
    pub gemv_iq2_xxs: Kernel,
    pub gemv_iq2_xs: Kernel,
    pub gemv_iq2_s: Kernel,
    pub gemv_f32: Kernel,
    pub gemv_f16: Kernel,
    pub gemv_bf16: Kernel,
    pub gemv_q8_1: Kernel,
    pub gemv_q8_k: Kernel,
    /// SpMM CSR del FFN disperso (DAG irregular, GGUF de `saor`).
    pub spmm_csr: Kernel,
    /// SpMM esparso batcheado desde bit-tensor + pesos F32 compartidos (Fase 2, C4).
    pub spmm_adj_batched: Kernel,
    /// SpMM esparso batcheado con dequant Q4_K en el kernel (Fase 2, C1).
    pub spmm_adj_batched_q4: Kernel,
    /// Dequant Q4_K -> F32 batcheado (encadenado con `spmm_adj_batched`).
    pub dequant_q4_k_to_f32: Kernel,
    /// Number of FFN GEMVs dispatched to this device (load-distribution metric).
    pub ffn_calls: std::sync::atomic::AtomicU64,
}

impl OpenClEngine {
    pub fn try_init_any() -> Result<Self, OpenClError> {
        let devices = discover_opencl_devices();
        let best = devices
            .iter()
            .find(|d| d.device_kind == DeviceKind::DiscreteGpu)
            .or_else(|| devices.iter().find(|d| d.device_kind == DeviceKind::Apu))
            .or_else(|| {
                devices
                    .iter()
                    .find(|d| d.device_kind == DeviceKind::IntegratedGpu)
            })
            .or_else(|| {
                devices
                    .iter()
                    .find(|d| d.device_kind == DeviceKind::Accelerator)
            })
            .or_else(|| {
                devices
                    .iter()
                    .find(|d| d.device_kind != DeviceKind::CpuOpenCl)
            })
            .ok_or(OpenClError::NoDeviceFound)?;
        Self::init_device(best)
    }

    /// Legacy hetero pair (primary + second). Prefer [`OpenClDevicePool::try_init_all_gpus`].
    pub fn try_init_hetero() -> Result<(Self, Option<Self>), OpenClError> {
        let pool = OpenClDevicePool::try_init_all_gpus()?;
        let mut engines = pool.engines;
        let primary = engines.remove(0);
        let secondary = engines.into_iter().next();
        Ok((primary, secondary))
    }

    pub fn init_device(info: &OpenClDeviceInfo) -> Result<Self, OpenClError> {
        let device = Device::new(info.device_id);
        let context =
            Context::from_device(&device).map_err(|e| OpenClError::ClError(e.to_string()))?;

        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
                .map_err(|e| OpenClError::ClError(e.to_string()))?;

        // Hard requirement: Hayai targets OpenCL 3.0 PLATFORMS. Kernels are written in
        // the mandatory OpenCL C subset (C 1.2) — the ONLY language every OpenCL 3.0
        // device must support; OpenCL C 2.0/3.0 language is optional per-device and
        // NVIDIA's OpenCL 3.0 compiles C 1.2 only. We therefore build with the default
        // `-cl-std` (C 1.2), which every OpenCL 3.0 platform is required to accept.
        let platform_version = info.opencl_version.clone();
        if !platform_version.contains("3.0") {
            return Err(OpenClError::UnsupportedPlatformVersion(platform_version));
        }

        let source = opencl_program_source();
        let program = Program::create_and_build_from_source(&context, &source, "")
            .map_err(|e| OpenClError::ClError(format!("Kernel compilation failed: {}", e)))?;

        let lut_kernel = Kernel::create(&program, "lut_matmul_q4_v1")
            .map_err(|e| OpenClError::ClError(format!("lut kernel: {}", e)))?;
        let gemv_q4_0 = Kernel::create(&program, "ggml_gemv_q4_0")
            .map_err(|e| OpenClError::ClError(format!("q4_0 kernel: {}", e)))?;
        let gemv_q4_1 = Kernel::create(&program, "ggml_gemv_q4_1")
            .map_err(|e| OpenClError::ClError(format!("q4_1 kernel: {}", e)))?;
        let gemv_q8_0 = Kernel::create(&program, "ggml_gemv_q8_0")
            .map_err(|e| OpenClError::ClError(format!("q8_0 kernel: {}", e)))?;
        let gemv_q5_0 = Kernel::create(&program, "ggml_gemv_q5_0")
            .map_err(|e| OpenClError::ClError(format!("q5_0 kernel: {}", e)))?;
        let gemv_q5_1 = Kernel::create(&program, "ggml_gemv_q5_1")
            .map_err(|e| OpenClError::ClError(format!("q5_1 kernel: {}", e)))?;
        let gemv_q2_k = Kernel::create(&program, "ggml_gemv_q2_k")
            .map_err(|e| OpenClError::ClError(format!("q2_k kernel: {}", e)))?;
        let gemv_q3_k = Kernel::create(&program, "ggml_gemv_q3_k")
            .map_err(|e| OpenClError::ClError(format!("q3_k kernel: {}", e)))?;
        let gemv_q4_k = Kernel::create(&program, "ggml_gemv_q4_k")
            .map_err(|e| OpenClError::ClError(format!("q4_k kernel: {}", e)))?;
        let gemv_batched_q4_k = Kernel::create(&program, "ggml_gemv_batched_q4_k")
            .map_err(|e| OpenClError::ClError(format!("batched q4_k kernel: {}", e)))?;
        let gemv_q5_k = Kernel::create(&program, "ggml_gemv_q5_k")
            .map_err(|e| OpenClError::ClError(format!("q5_k kernel: {}", e)))?;
        let gemv_q6_k = Kernel::create(&program, "ggml_gemv_q6_k")
            .map_err(|e| OpenClError::ClError(format!("q6_k kernel: {}", e)))?;
        let gemv_iq4_nl = Kernel::create(&program, "ggml_gemv_iq4_nl")
            .map_err(|e| OpenClError::ClError(format!("iq4_nl kernel: {}", e)))?;
        let gemv_iq4_xs = Kernel::create(&program, "ggml_gemv_iq4_xs")
            .map_err(|e| OpenClError::ClError(format!("iq4_xs kernel: {}", e)))?;
        let gemv_iq3_xxs = Kernel::create(&program, "ggml_gemv_iq3_xxs")
            .map_err(|e| OpenClError::ClError(format!("iq3_xxs kernel: {}", e)))?;
        let gemv_iq3_s = Kernel::create(&program, "ggml_gemv_iq3_s")
            .map_err(|e| OpenClError::ClError(format!("iq3_s kernel: {}", e)))?;
        let gemv_iq2_xxs = Kernel::create(&program, "ggml_gemv_iq2_xxs")
            .map_err(|e| OpenClError::ClError(format!("iq2_xxs kernel: {}", e)))?;
        let gemv_iq2_xs = Kernel::create(&program, "ggml_gemv_iq2_xs")
            .map_err(|e| OpenClError::ClError(format!("iq2_xs kernel: {}", e)))?;
        let gemv_iq2_s = Kernel::create(&program, "ggml_gemv_iq2_s")
            .map_err(|e| OpenClError::ClError(format!("iq2_s kernel: {}", e)))?;
        let gemv_f32 = Kernel::create(&program, "ggml_gemv_f32")
            .map_err(|e| OpenClError::ClError(format!("f32 kernel: {}", e)))?;
        let gemv_f16 = Kernel::create(&program, "ggml_gemv_f16")
            .map_err(|e| OpenClError::ClError(format!("f16 kernel: {}", e)))?;
        let gemv_bf16 = Kernel::create(&program, "ggml_gemv_bf16")
            .map_err(|e| OpenClError::ClError(format!("bf16 kernel: {}", e)))?;
        let gemv_q8_1 = Kernel::create(&program, "ggml_gemv_q8_1")
            .map_err(|e| OpenClError::ClError(format!("q8_1 kernel: {}", e)))?;
        let gemv_q8_k = Kernel::create(&program, "ggml_gemv_q8_k")
            .map_err(|e| OpenClError::ClError(format!("q8_k kernel: {}", e)))?;
        let spmm_csr = Kernel::create(&program, "spmm_csr")
            .map_err(|e| OpenClError::ClError(format!("spmm_csr kernel: {}", e)))?;
        let spmm_adj_batched = Kernel::create(&program, "spmm_adj_batched")
            .map_err(|e| OpenClError::ClError(format!("spmm_adj_batched kernel: {}", e)))?;
        let spmm_adj_batched_q4 = Kernel::create(&program, "spmm_adj_batched_q4")
            .map_err(|e| OpenClError::ClError(format!("spmm_adj_batched_q4 kernel: {}", e)))?;
        let dequant_q4_k_to_f32 = Kernel::create(&program, "dequant_q4_k_to_f32")
            .map_err(|e| OpenClError::ClError(format!("dequant_q4_k_to_f32 kernel: {}", e)))?;

        info!(
            "Successfully initialized OpenCL Engine on device: {}",
            info.device_name
        );

        Ok(Self {
            device_info: info.clone(),
            context,
            queue,
            lut_kernel,
            gemv_q4_0,
            gemv_q4_1,
            gemv_q8_0,
            gemv_q5_0,
            gemv_q5_1,
            gemv_q2_k,
            gemv_q3_k,
            gemv_q4_k,
            gemv_batched_q4_k,
            gemv_q5_k,
            gemv_q6_k,
            gemv_iq4_nl,
            gemv_iq4_xs,
            gemv_iq3_xxs,
            gemv_iq3_s,
            gemv_iq2_xxs,
            gemv_iq2_xs,
            gemv_iq2_s,
            gemv_f32,
            gemv_f16,
            gemv_bf16,
            gemv_q8_1,
            gemv_q8_k,
            spmm_csr,
            spmm_adj_batched,
            spmm_adj_batched_q4,
            dequant_q4_k_to_f32,
            ffn_calls: std::sync::atomic::AtomicU64::new(0),
        })
    }

}
