//! OpenCL device pool: every GPU that exposes OpenCL joins the compute pool (PRD premise).

use crate::context::{OpenClEngine, OpenClError};
use crate::device::{discover_opencl_devices, DeviceKind, OpenClDeviceInfo};
use tracing::{info, warn};

/// All usable OpenCL accelerators (discrete + integrated + other GPUs).
/// CPU OpenCL devices are excluded from the FFN pool (host CPU already runs Attn/KV).
pub struct OpenClDevicePool {
    pub engines: Vec<OpenClEngine>,
}

impl OpenClDevicePool {
    /// Initialize **every** non-CPU OpenCL device. Product policy: if it speaks OpenCL
    /// as a GPU, it is in the pool — vendor/type agnostic (NVIDIA/Intel/AMD/Mali/…).
    pub fn try_init_all_gpus() -> Result<Self, OpenClError> {
        let devices = discover_opencl_devices();
        let gpu_infos: Vec<&OpenClDeviceInfo> = devices
            .iter()
            .filter(|d| d.device_kind != DeviceKind::CpuOpenCl)
            .collect();

        if gpu_infos.is_empty() {
            return Err(OpenClError::NoDeviceFound);
        }

        // Prefer discrete first for stable primary/scratch ownership, then integrated, then other.
        let mut ordered: Vec<&OpenClDeviceInfo> = Vec::new();
        for d in &gpu_infos {
            if d.device_kind == DeviceKind::DiscreteGpu {
                ordered.push(d);
            }
        }
        for d in &gpu_infos {
            if d.device_kind == DeviceKind::IntegratedGpu {
                ordered.push(d);
            }
        }
        for d in &gpu_infos {
            if !matches!(
                d.device_kind,
                DeviceKind::DiscreteGpu | DeviceKind::IntegratedGpu
            ) {
                ordered.push(d);
            }
        }

        let mut engines = Vec::new();
        for info in ordered {
            match OpenClEngine::init_device(info) {
                Ok(eng) => {
                    info!(
                        "Pool += {} ({:?}, SVM={})",
                        eng.device_info.device_name,
                        eng.device_info.device_kind,
                        eng.device_info.supports_svm
                    );
                    engines.push(eng);
                }
                Err(e) => {
                    warn!(
                        "Skipping OpenCL device '{}': {e}",
                        info.device_name
                    );
                }
            }
        }

        if engines.is_empty() {
            return Err(OpenClError::NoDeviceFound);
        }

        info!("OpenCL device pool size: {}", engines.len());
        Ok(Self { engines })
    }

    /// Create an empty pool (no OpenCL devices). Used in tests and CPU-only mode.
    pub fn empty() -> Self {
        Self { engines: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.engines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    pub fn primary(&self) -> &OpenClEngine {
        &self.engines[0]
    }

    pub fn primary_mut(&mut self) -> &mut OpenClEngine {
        &mut self.engines[0]
    }

    pub fn get(&self, idx: usize) -> &OpenClEngine {
        &self.engines[idx % self.engines.len()]
    }

    /// Round-robin pick for matrix `role` (0=gate, 1=up, 2=down, …).
    pub fn for_role(&self, role: usize) -> &OpenClEngine {
        self.get(role)
    }

    pub fn names(&self) -> Vec<String> {
        self.engines
            .iter()
            .map(|e| e.device_info.device_name.clone())
            .collect()
    }

    pub fn has_discrete_and_integrated(&self) -> bool {
        let has_d = self
            .engines
            .iter()
            .any(|e| e.device_info.device_kind == DeviceKind::DiscreteGpu);
        let has_i = self
            .engines
            .iter()
            .any(|e| e.device_info.device_kind == DeviceKind::IntegratedGpu);
        has_d && has_i
    }

    /// APU / integrated GPU with SVM — preferred host for `clSVMAlloc` (expert guidance).
    pub fn apu_svm(&self) -> Option<&OpenClEngine> {
        self.engines
            .iter()
            .find(|e| {
                e.device_info.device_kind == DeviceKind::IntegratedGpu
                    && e.device_info.supports_svm
            })
            .or_else(|| {
                self.engines
                    .iter()
                    .find(|e| e.device_info.supports_svm)
            })
    }
}
