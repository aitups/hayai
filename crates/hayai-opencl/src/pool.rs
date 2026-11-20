//! OpenCL device pool: every GPU that exposes OpenCL joins the compute pool (PRD premise).

use crate::context::{OpenClEngine, OpenClError};
use crate::device::{discover_opencl_devices, DeviceKind, OpenClDeviceInfo};
use tracing::{info, warn};

/// All usable OpenCL accelerators (discrete + integrated + other GPUs).
/// CPU OpenCL devices are excluded from the FFN pool (host CPU already runs Attn/KV).
pub struct OpenClDevicePool {
    pub engines: Vec<OpenClEngine>,
    /// FFN role (`0=gate, 1=up, 2=down`) → engine index, balanced by capability.
    role_map: Vec<usize>,
}

impl OpenClDevicePool {
    /// Initialize **every** OpenCL device, GPU first. Product policy: any device
    /// that speaks OpenCL 3.0 joins the pool — vendor/type agnostic (NVIDIA/Intel/
    /// AMD/ARM/…). CPU-OpenCL is included **last** (the host CPU already runs
    /// attention, but an OpenCL CPU device is still a usable FFN target).
    pub fn try_init_all_gpus() -> Result<Self, OpenClError> {
        let devices = discover_opencl_devices();
        if devices.is_empty() {
            return Err(OpenClError::NoDeviceFound);
        }
        // Discrete → APU → integrated → accelerator → other → CPU-OpenCL.
        let mut ordered: Vec<&OpenClDeviceInfo> = devices.iter().collect();
        ordered.sort_by_key(|d| d.device_kind.pool_rank());

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

        let role_map = plan_roles(&engines);
        info!("OpenCL device pool size: {}", engines.len());
        Ok(Self { engines, role_map })
    }

    /// Create an empty pool (no OpenCL devices). Used in tests and CPU-only mode.
    pub fn empty() -> Self {
        Self {
            engines: Vec::new(),
            role_map: Vec::new(),
        }
    }

    /// Pool with a single explicitly-selected engine (`--device <name>`).
    pub fn single(engine: OpenClEngine) -> Self {
        let role_map = plan_roles(std::slice::from_ref(&engine));
        Self {
            engines: vec![engine],
            role_map,
        }
    }

    pub fn len(&self) -> usize {
        self.engines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    /// Primary (first) engine, or `None` when the pool is empty (CPU-only).
    pub fn primary(&self) -> Option<&OpenClEngine> {
        self.engines.first()
    }

    pub fn primary_mut(&mut self) -> Option<&mut OpenClEngine> {
        self.engines.first_mut()
    }

    /// Round-robin engine for matrix `idx`, or `None` for an empty pool.
    pub fn get(&self, idx: usize) -> Option<&OpenClEngine> {
        if self.engines.is_empty() {
            None
        } else {
            Some(&self.engines[idx % self.engines.len()])
        }
    }

    /// Round-robin pick for matrix `role` (0=gate, 1=up, 2=down, …).
    pub fn for_role(&self, role: usize) -> Option<&OpenClEngine> {
        if self.engines.is_empty() {
            return None;
        }
        let idx = self
            .role_map
            .get(role)
            .copied()
            .unwrap_or(role % self.engines.len());
        self.engines.get(idx)
    }

    /// Device index assigned to each FFN role (empty = round-robin fallback).
    pub fn role_map(&self) -> &[usize] {
        &self.role_map
    }

    /// Override the role assignment (used by the hardware-aware planner).
    pub fn set_role_map(&mut self, map: Vec<usize>) {
        if !self.engines.is_empty() {
            self.role_map = map.into_iter().map(|i| i % self.engines.len()).collect();
        }
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
        let has_i = self.engines.iter().any(|e| {
            matches!(
                e.device_info.device_kind,
                DeviceKind::Apu | DeviceKind::IntegratedGpu
            )
        });
        has_d && has_i
    }

    /// APU / unified-memory device with SVM — preferred host for `clSVMAlloc`.
    pub fn apu_svm(&self) -> Option<&OpenClEngine> {
        self.engines
            .iter()
            .find(|e| e.device_info.device_kind == DeviceKind::Apu && e.device_info.supports_svm)
            .or_else(|| {
                self.engines.iter().find(|e| {
                    e.device_info.device_kind == DeviceKind::IntegratedGpu
                        && e.device_info.supports_svm
                })
            })
            .or_else(|| {
                self.engines
                    .iter()
                    .find(|e| e.device_info.supports_svm)
            })
    }
}

/// Assign the 3 FFN roles (gate/up/down) across the pool.
///
/// Without measured throughputs and with the gate/up → down dependency, the safe default
/// is to keep all three roles on the single fastest device *class* — never blind
/// round-robin `up` onto a slow iGPU (the old formula normalised by the same weight it
/// accumulated, always yielding `[0, 1, 0]`). The measured planner is the authority: it
/// calls [`OpenClDevicePool::set_role_map`] with the per-op placement, and the hot FFN
/// path reads `orchestrator.placement` directly.
pub fn plan_roles(engines: &[OpenClEngine]) -> Vec<usize> {
    let n = engines.len();
    if n == 0 {
        return Vec::new();
    }
    let best = (0..n)
        .min_by_key(|&i| (engines[i].device_info.device_kind.pool_rank(), i))
        .unwrap_or(0);
    vec![best; 3]
}
