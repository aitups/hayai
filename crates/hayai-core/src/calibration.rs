//! Hardware calibration probe.
//!
//! Measures the bandwidth ceilings the planner needs to target **75–80 %**
//! utilization of RAM and compute, agnostic of the concrete OpenCL device
//! (CPU/APU/NPU/GPU, x86-64 or ARM64):
//!
//! * host memory bandwidth (read+write),
//! * disk read bandwidth (through the same [`hayai_io`] path as inference),
//! * per-device *effective* GEMV bandwidth (upload + compute, i.e. the real
//!   streaming throughput the engine can achieve on that device).
//!
//! The derived [`HwProfile`] is the input to the memory/compute planner and the
//! origin of the numeric tok/s floor for a given `bytes_per_token`.

use hayai_opencl::{OpenClDevicePool, OpenClEngine};
use std::path::Path;
use std::time::Instant;

/// Bandwidth / memory profile of one accelerator.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub name: String,
    pub kind: String,
    /// Index into the OpenCL pool (the planner's `ComputeTarget::Device(i)`).
    pub device_index: usize,
    pub global_mem_bytes: u64,
    /// Effective streaming GEMV bandwidth (bytes/s) including host→device upload.
    pub effective_gemv_gbytes_s: f64,
    /// Effective GEMV bandwidth (bytes/s) with weights already resident on the device.
    pub resident_gemv_gbytes_s: f64,
    /// Resident GEMV bandwidth (bytes/s) for the Q4_K quant (the real FFN weights).
    pub resident_gemv_q4k_gbytes_s: f64,
    /// Fixed per-op launch overhead (µs).
    pub launch_us: f64,
    /// Host→device transfer bandwidth (bytes/s).
    pub dma_gbytes_s: f64,
}

/// Full hardware profile for one host.
#[derive(Debug, Clone)]
pub struct HwProfile {
    pub host_bw_gbytes_s: f64,
    /// Measured CPU Q4_K GEMV bandwidth (the real host compute rate).
    pub host_gemv_gbytes_s: f64,
    pub disk_bw_gbytes_s: f64,
    pub devices: Vec<DeviceProfile>,
}

impl HwProfile {
    /// Best accelerator streaming bandwidth in GB/s.
    pub fn best_accel_gbytes_s(&self) -> f64 {
        self.devices
            .iter()
            .map(|d| d.effective_gemv_gbytes_s)
            .fold(0.0, f64::max)
    }

    /// Effective stream-from-disk bandwidth: the pipeline is limited by the
    /// slower of disk and accelerator; with no disk measurement use the accel.
    pub fn effective_stream_gbytes_s(&self) -> f64 {
        let accel = self.best_accel_gbytes_s();
        if self.disk_bw_gbytes_s <= 0.0 {
            accel
        } else if accel <= 0.0 {
            self.disk_bw_gbytes_s
        } else {
            accel.min(self.disk_bw_gbytes_s)
        }
    }

    /// Bandwidth-bound tok/s for a model that streams `bytes_per_token` at the
    /// given utilization fraction (e.g. 0.75–0.80).
    pub fn target_tok_s(&self, bytes_per_token: f64, utilization: f64) -> f64 {
        if bytes_per_token <= 0.0 {
            return f64::INFINITY;
        }
        self.effective_stream_gbytes_s() * 1e9 * utilization / bytes_per_token
    }

    pub fn to_json(&self) -> String {
        let devices: Vec<String> = self
            .devices
            .iter()
            .map(|d| {
                format!(
                    "{{\"name\":\"{}\",\"kind\":\"{}\",\"global_mem_bytes\":{},\"effective_gemv_gbytes_s\":{:.4}}}",
                    d.name.replace('"', "'"),
                    d.kind.replace('"', "'"),
                    d.global_mem_bytes,
                    d.effective_gemv_gbytes_s
                )
            })
            .collect();
        format!(
            "{{\"host_bw_gbytes_s\":{:.4},\"disk_bw_gbytes_s\":{:.4},\"devices\":[{}]}}",
            self.host_bw_gbytes_s,
            self.disk_bw_gbytes_s,
            devices.join(",")
        )
    }

    /// Human-readable report with the 75 %/80 % targets for `bytes_per_token`.
    pub fn format_report(&self, bytes_per_token: Option<f64>) -> String {
        let mut s = String::new();
        s.push_str("HAYAI HARDWARE CALIBRATION\n");
        s.push_str("------------------------------------------------------------\n");
        s.push_str(&format!(
            "Host memory bandwidth : {:6.2} GB/s\n",
            self.host_bw_gbytes_s
        ));
        s.push_str(&format!(
            "Host Q4_K GEMV        : {:6.2} GB/s\n",
            self.host_gemv_gbytes_s
        ));
        s.push_str(&format!(
            "Disk read bandwidth   : {:6.2} GB/s\n",
            self.disk_bw_gbytes_s
        ));
        for d in &self.devices {
            s.push_str(&format!(
                "Device {:<24} GEMV stream {:>6.2} GB/s | resident {:>6.2} GB/s | q4_k {:>6.2} GB/s | launch {:>5.0} us | DMA {:>5.2} GB/s  ({} MiB, {})\n",
                d.name,
                d.effective_gemv_gbytes_s,
                d.resident_gemv_gbytes_s,
                d.resident_gemv_q4k_gbytes_s,
                d.launch_us,
                d.dma_gbytes_s,
                d.global_mem_bytes / (1024 * 1024),
                d.kind
            ));
        }
        s.push_str(&format!(
            "Effective stream BW   : {:6.2} GB/s\n",
            self.effective_stream_gbytes_s()
        ));
        if let Some(bpt) = bytes_per_token {
            s.push_str("------------------------------------------------------------\n");
            s.push_str(&format!("Model stream bytes/token: {:.1} MiB\n", bpt / 1_048_576.0));
            s.push_str(&format!(
                "Target @75%            : {:8.3} tok/s\n",
                self.target_tok_s(bpt, 0.75)
            ));
            s.push_str(&format!(
                "Target @80%            : {:8.3} tok/s\n",
                self.target_tok_s(bpt, 0.80)
            ));
        }
        s
    }
}

/// Host RAM bandwidth (read+write) in GB/s.
pub fn measure_host_bandwidth_gbytes_s() -> f64 {
    let len = 32 * 1024 * 1024usize;
    let src = vec![0xA5u8; len];
    let mut dst = vec![0u8; len];
    let iters = 4usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        dst.copy_from_slice(std::hint::black_box(&src));
    }
    // Keep the copy observable so the optimizer cannot elide the loop.
    std::hint::black_box(&dst);
    let dt = t0.elapsed().as_secs_f64();
    if dt <= 0.0 {
        return 0.0;
    }
    // read + write
    2.0 * (len * iters) as f64 / dt / 1e9
}

/// Host (CPU) Q4_K GEMV bandwidth in GB/s — the rate the real CPU compute path
/// achieves, not the RAM memcpy peak. The planner must price the CPU by this, or it
/// over-values the host and leaves a fast GPU idle.
pub fn measure_host_gemv_gbytes_s() -> f64 {
    const M: usize = 4096;
    const N: usize = 4096;
    let blocks_per_row = N / 256;
    let row_bytes = blocks_per_row * 144;
    let weights = vec![0x11u8; M * row_bytes];
    let input = vec![0.01f32; N];
    let mut out = vec![0.0f32; M];
    // Warm up (rayon thread pool + branch predictor).
    let _ = hayai_model::q4k::gemv_q4_k(M, N, &weights, &input, &mut out);
    let iters = 5usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        hayai_model::q4k::gemv_q4_k(M, N, &weights, &input, &mut out).ok();
        std::hint::black_box(&out);
    }
    let dt = t0.elapsed().as_secs_f64();
    if dt <= 0.0 {
        return 0.0;
    }
    (M * row_bytes * iters) as f64 / dt / 1e9
}

/// Sequential disk read bandwidth through the production [`hayai_io`] path.
pub fn measure_disk_bandwidth_gbytes_s(path: &Path) -> Result<f64, String> {
    let size = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if size == 0 {
        return Ok(0.0);
    }
    let mut io = hayai_io::open_weight_io(path).map_err(|e| e.to_string())?;
    let chunk = (4 * 1024 * 1024usize).min(size as usize).max(4096);
    let mut buf = vec![0u8; chunk];
    // Sample up to 16 chunks (64 MiB) to stay fast.
    let nread = (size / chunk as u64).clamp(1, 16);
    let t0 = Instant::now();
    let mut off = 0u64;
    for _ in 0..nread {
        let len = chunk.min((size - off) as usize);
        if len == 0 {
            break;
        }
        io.read_at(off, &mut buf[..len]).map_err(|e| e.to_string())?;
        off += len as u64;
    }
    let dt = t0.elapsed().as_secs_f64();
    if dt <= 0.0 || off == 0 {
        return Ok(0.0);
    }
    Ok(off as f64 / dt / 1e9)
}

/// Effective GEMV bandwidth (upload + compute) of one device, in GB/s.
///
/// Uses the F32 GEMV kernel on a 16 MiB matrix; this is the same streaming shape
/// the engine uses for FFN weights, so it is the bandwidth the planner can plan
/// against on any device kind.
pub fn measure_device_gemv_gbytes_s(engine: &OpenClEngine) -> Result<f64, String> {
    let (m, n) = (2048usize, 2048usize);
    let mut weights = vec![0u8; m * n * 4];
    for (i, b) in weights.iter_mut().enumerate() {
        if i % 4 == 0 {
            *b = (i & 0x3F) as u8;
        }
    }
    let input = vec![0.01f32; n];
    // Warm up (kernel + buffers).
    let _ = engine
        .ggml_gemv_async(&engine.gemv_f32, "f32", m, n, &weights, &input)
        .map_err(|e| e.to_string())?
        .wait()
        .map_err(|e| e.to_string())?;
    let iters = 5usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let out = engine
            .ggml_gemv_async(&engine.gemv_f32, "f32", m, n, &weights, &input)
            .map_err(|e| e.to_string())?
            .wait()
            .map_err(|e| e.to_string())?;
        std::hint::black_box(&out);
    }
    let dt = t0.elapsed().as_secs_f64();
    if dt <= 0.0 {
        return Ok(0.0);
    }
    Ok((m * n * 4 * iters) as f64 / dt / 1e9)
}

/// Run the full calibration over the OpenCL pool and (optionally) a model file.
pub fn calibrate(pool: &OpenClDevicePool, disk_path: Option<&Path>) -> HwProfile {
    let host_bw_gbytes_s = measure_host_bandwidth_gbytes_s();
    let host_gemv_gbytes_s = measure_host_gemv_gbytes_s();
    let disk_bw_gbytes_s = disk_path
        .and_then(|p| measure_disk_bandwidth_gbytes_s(p).ok())
        .unwrap_or(0.0);
    let mut devices = Vec::new();
    for (idx, e) in pool.engines.iter().enumerate() {
        let bw = measure_device_gemv_gbytes_s(e).unwrap_or(0.0);
        // Use a large GEMV so the fixed per-launch overhead (~100 us) is amortised;
        // a 2048x2048 problem is dominated by launch, not bandwidth.
        let resident_bw = e.bench_resident_gemv_gbytes_s(8192, 4096).unwrap_or(bw);
        let resident_q4k_bw = e
            .bench_resident_gemv_q4k_gbytes_s(8192, 4096)
            .unwrap_or(resident_bw);
        let launch_us = e
            .bench_launch_seconds()
            .map(|s| s * 1e6)
            .unwrap_or(0.0);
        let dma_bw = e
            .bench_dma_gbytes_s(16 * 1024 * 1024)
            .unwrap_or(0.0);
        devices.push(DeviceProfile {
            name: e.device_info.device_name.clone(),
            kind: format!("{:?}", e.device_info.device_kind),
            device_index: idx,
            global_mem_bytes: e.device_info.global_mem_size,
            effective_gemv_gbytes_s: bw,
            resident_gemv_gbytes_s: resident_bw,
            resident_gemv_q4k_gbytes_s: resident_q4k_bw,
            launch_us,
            dma_gbytes_s: dma_bw,
        });
    }
    HwProfile {
        host_bw_gbytes_s,
        host_gemv_gbytes_s,
        disk_bw_gbytes_s,
        devices,
    }
}
