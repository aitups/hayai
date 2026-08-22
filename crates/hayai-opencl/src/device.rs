use opencl3::device::{Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::platform::get_platforms;
use tracing::info;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceKind {
    IntegratedGpu, // APU / Integrated graphics (Intel UHD, AMD Radeon 780M, etc.)
    DiscreteGpu,   // Dedicated GPU (NVIDIA RTX/GTX, AMD Radeon RX, etc.)
    CpuOpenCl,     // OpenCL on CPU driver
    Other(String),
}

#[derive(Debug, Clone)]
pub struct OpenClDeviceInfo {
    pub platform_name: String,
    pub device_name: String,
    pub vendor: String,
    pub device_kind: DeviceKind,
    pub device_id: opencl3::types::cl_device_id,
    pub supports_svm: bool,
    pub opencl_version: String,
    pub opencl_c_version: String,
    pub max_compute_units: u32,
    pub global_mem_size: u64,
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
            let dev_type = dev.dev_type().unwrap_or(0);
            let max_compute_units = dev.max_compute_units().unwrap_or(1);
            let global_mem_size = dev.global_mem_size().unwrap_or(0);

            // SVM Capability Check (OpenCL 2.0+ / 3.0 SVM coarse/fine grain)
            let supports_svm = dev.svm_mem_capability() != 0;

            // OpenCL platform version (Hayai hard-requires OpenCL 3.0 platforms).
            let opencl_version = dev.version().unwrap_or_else(|_| "OpenCL Unknown".into());

            // OpenCL C language version (diagnostic; OpenCL 3.0 only guarantees C 1.2,
            // e.g. NVIDIA's OpenCL 3.0 exposes "OpenCL C 1.2").
            let opencl_c_version = dev
                .opencl_c_version()
                .unwrap_or_else(|_| "OpenCL C Unknown".into());

            let device_kind = if dev_type & CL_DEVICE_TYPE_GPU != 0 {
                // Heuristic for Integrated vs Discrete GPU based on vendor/name and SVM
                let name_lower = device_name.to_lowercase();
                if name_lower.contains("intel") || name_lower.contains("graphics") || (supports_svm && name_lower.contains("amd radeon(tm)")) {
                    DeviceKind::IntegratedGpu
                } else {
                    DeviceKind::DiscreteGpu
                }
            } else if dev_type & CL_DEVICE_TYPE_CPU != 0 {
                DeviceKind::CpuOpenCl
            } else {
                DeviceKind::Other(format!("{:x}", dev_type))
            };

            info!(
                "Discovered OpenCL Device: {} ({}) | Type: {:?} | SVM: {} | OpenCL: {} | C: {} | VRAM/RAM: {} MB",
                device_name,
                platform_name,
                device_kind,
                supports_svm,
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
                opencl_version,
                opencl_c_version,
                max_compute_units,
                global_mem_size,
            });
        }
    }

    results
}
