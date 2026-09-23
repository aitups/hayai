use opencl3::device::{Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::platform::get_platforms;
use tracing::info;

/// OpenCL device-type bits not re-exported by `opencl3`.
const CL_DEVICE_TYPE_ACCELERATOR: u64 = 1 << 3;
const CL_DEVICE_TYPE_CUSTOM: u64 = 1 << 4;

/// SVM capability bits (OpenCL 2.0+).
pub const SVM_COARSE_GRAIN_BUFFER: u64 = 1 << 0;
pub const SVM_FINE_GRAIN_BUFFER: u64 = 1 << 1;
pub const SVM_FINE_GRAIN_SYSTEM: u64 = 1 << 2;

/// Hardware class of an OpenCL device. Universal: any OpenCL 3 platform maps to
/// one of these, regardless of vendor or architecture (x86-64 / ARM64).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceKind {
    /// Integrated GPU on **unified memory** (APU: AMD Strix Halo, Intel, GB10…).
    Apu,
    /// Integrated GPU without the unified-memory flag.
    IntegratedGpu,
    /// Dedicated GPU (NVIDIA RTX/GTX, AMD Radeon RX…).
    DiscreteGpu,
    /// OpenCL on the host CPU driver.
    CpuOpenCl,
    /// OpenCL accelerator / NPU / custom device.
    Accelerator,
    Other(String),
}

impl DeviceKind {
    /// True for any GPU-class device (APU, integrated or discrete).
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            DeviceKind::Apu | DeviceKind::IntegratedGpu | DeviceKind::DiscreteGpu
        )
    }

    /// Priority for pool ordering (lower = tried first for FFN roles).
    pub fn pool_rank(&self) -> u8 {
        match self {
            DeviceKind::DiscreteGpu => 0,
            DeviceKind::Apu => 1,
            DeviceKind::IntegratedGpu => 2,
            DeviceKind::Accelerator => 3,
            DeviceKind::Other(_) => 4,
            DeviceKind::CpuOpenCl => 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenClDeviceInfo {
    pub platform_name: String,
    pub device_name: String,
    pub vendor: String,
    pub device_kind: DeviceKind,
    pub device_id: opencl3::types::cl_device_id,
    pub supports_svm: bool,
    /// Raw SVM capability bits (coarse / fine buffer / fine system).
    pub svm_capability: u64,
    /// `CL_DEVICE_HOST_UNIFIED_MEMORY` (shared host/device DRAM).
    pub unified_memory: bool,
    pub opencl_version: String,
    pub opencl_c_version: String,
    pub max_compute_units: u32,
    pub global_mem_size: u64,
    pub max_alloc_size: u64,
    pub max_work_group_size: usize,
}

impl OpenClDeviceInfo {
    pub fn svm_coarse(&self) -> bool {
        self.svm_capability & SVM_COARSE_GRAIN_BUFFER != 0
    }
    pub fn svm_fine_buffer(&self) -> bool {
        self.svm_capability & SVM_FINE_GRAIN_BUFFER != 0
    }
    pub fn svm_fine_system(&self) -> bool {
        self.svm_capability & SVM_FINE_GRAIN_SYSTEM != 0
    }
}

/// Discovers all available OpenCL devices on the host system.
pub fn discover_opencl_devices() -> Vec<OpenClDeviceInfo> {
    let mut results = Vec::new();

    let platforms = match get_platforms() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("Failed to query OpenCL platforms: {}", e);
            return results;
        }
    };

    for platform in platforms {
        let platform_name = platform.name().unwrap_or_else(|_| "Unknown".into());
        let device_ids = match platform.get_devices(CL_DEVICE_TYPE_ALL) {
            Ok(devs) => devs,
            Err(_) => continue,
        };

        for dev_id in device_ids {
            let dev = Device::new(dev_id);
            let device_name = dev.name().unwrap_or_else(|_| "Unknown Device".into());
            let vendor = dev.vendor().unwrap_or_else(|_| "Unknown Vendor".into());
            let dev_type = dev.dev_type().unwrap_or(0) as u64;
            let max_compute_units = dev.max_compute_units().unwrap_or(1);
            let global_mem_size = dev.global_mem_size().unwrap_or(0);
            let max_alloc_size = dev.max_mem_alloc_size().unwrap_or(0);
            let max_work_group_size = dev.max_work_group_size().unwrap_or(0) as usize;

            // SVM capability bits (OpenCL 2.0+ / 3.0 SVM coarse/fine grain).
            let svm_capability = dev.svm_mem_capability() as u64;
            let supports_svm = svm_capability != 0;
            let unified_memory = dev.host_unified_memory().unwrap_or(false);

            // OpenCL platform version (Hayai hard-requires OpenCL 3.0 platforms).
            let opencl_version = dev.version().unwrap_or_else(|_| "OpenCL Unknown".into());

            // OpenCL C language version (diagnostic; OpenCL 3.0 only guarantees C 1.2,
            // e.g. NVIDIA's OpenCL 3.0 exposes "OpenCL C 1.2").
            let opencl_c_version = dev
                .opencl_c_version()
                .unwrap_or_else(|_| "OpenCL C Unknown".into());

            let gl_gpu = dev_type & CL_DEVICE_TYPE_GPU as u64 != 0;
            let device_kind = if gl_gpu {
                let name_lower = device_name.to_lowercase();
                if unified_memory {
                    // Unified host/device memory (APU / superchip) — first-class
                    // zero-copy target (SVM fine-system when available).
                    DeviceKind::Apu
                } else if name_lower.contains("intel")
                    || name_lower.contains("graphics")
                    || name_lower.contains("apu")
                    || (supports_svm && name_lower.contains("radeon"))
                {
                    DeviceKind::IntegratedGpu
                } else {
                    DeviceKind::DiscreteGpu
                }
            } else if dev_type & CL_DEVICE_TYPE_CPU as u64 != 0 {
                DeviceKind::CpuOpenCl
            } else if dev_type & (CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_CUSTOM) != 0 {
                DeviceKind::Accelerator
            } else {
                DeviceKind::Other(format!("{dev_type:x}"))
            };

            info!(
                "Discovered OpenCL Device: {} ({}) | Type: {:?} | SVM: {:#x} | unified: {} | OpenCL: {} | C: {} | VRAM/RAM: {} MB",
                device_name,
                platform_name,
                device_kind,
                svm_capability,
                unified_memory,
                opencl_version,
                opencl_c_version,
                global_mem_size / (1024 * 1024)
            );

            results.push(OpenClDeviceInfo {
                platform_name: platform_name.clone(),
                device_name,
                vendor,
                device_kind,
                device_id: dev_id,
                supports_svm,
                svm_capability,
                unified_memory,
                opencl_version,
                opencl_c_version,
                max_compute_units,
                global_mem_size,
                max_alloc_size,
                max_work_group_size,
            });
        }
    }

    results
}
