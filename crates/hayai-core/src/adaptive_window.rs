//! Hardware-aware adaptive memory window.
//!
//! At inference startup (mode `auto`), Hayai queries available VRAM / RAM from the
//! already-enumerated OpenCL pool and picks the largest *layer-chunk* `k_chunk` that
//! fits within the available headroom — thereby trading disk-I/O trips for compute
//! throughput.
//!
//! Three operating modes
//! ─────────────────────
//! * **Resident**  (`k_chunk >= n_layers`): all layer packs fit in VRAM/SVM → I/O
//!   drops to zero after the first forward pass; decode speed reaches hardware maximum.
//! * **Macro-chunk** (`1 < k_chunk < n_layers`): one large sequential read per
//!   k-layer block instead of N individual micro-reads; saturates PCIe / SSD BW.
//! * **Minimal**   (`k_chunk == 1`): current 2-slot Ping-Pong (unchanged behaviour).

use hayai_opencl::OpenClDevicePool;

/// Target utilization of usable host/device memory (low end of the 75–80 % band).
pub const DEFAULT_UTILIZATION: f64 = 0.75;

/// User-visible / CLI strategy selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Detect free VRAM + system RAM and pick the largest safe k_chunk (default).
    AutoFit,
    /// Reserve at most `cap_bytes` of (V)RAM for the resident layer window.
    CapBytes(u64),
    /// Strict minimal mode: exactly 2 ping-pong slots regardless of VRAM.
    Minimal,
}

impl Default for MemoryStrategy {
    fn default() -> Self {
        Self::AutoFit
    }
}

impl MemoryStrategy {
    /// Parse from CLI string (`"auto"`, `"minimal"`, or `"4096"` / `"4096mb"`).
    pub fn parse(s: &str) -> Self {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "auto" | "autofit" | "" => Self::AutoFit,
            "minimal" | "min" | "strict" => Self::Minimal,
            _ => {
                // Accept numeric MiB or raw bytes: "2048", "2048mb", "2048mib"
                let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(n) = digits.parse::<u64>() {
                    // Treat values < 64 GB as MiB, larger as bytes
                    let bytes = if n < 64 * 1024 { n * 1024 * 1024 } else { n };
                    Self::CapBytes(bytes)
                } else {
                    Self::AutoFit
                }
            }
        }
    }
}

/// Result of the window computation passed into the generate loop.
#[derive(Debug, Clone, Copy)]
pub struct WindowPlan {
    /// Number of consecutive layers loaded in one I/O batch.
    pub k_chunk: usize,
    /// Whether all layers fit resident in accelerator memory (I/O-free decode).
    pub resident: bool,
    /// Total bytes allocated for the resident layer window.
    pub window_bytes: u64,
}

/// Compute the optimal `k_chunk` (layers per I/O batch) given:
/// - the per-layer pack size in bytes,
/// - the number of physical layers in the model,
/// - the OpenCL device pool (from which available VRAM is queried),
/// - the user memory strategy,
/// - `reserve_bytes`: KV cache + activations + metadata that must stay resident.
///
/// `AutoFit` targets [`DEFAULT_UTILIZATION`] (75 %) of the **usable** memory —
/// `min(host RAM, tightest device)`. Each dGPU mirror holds the whole base, so a
/// single mirror allocation is capped by that device's
/// `CL_DEVICE_MAX_MEM_ALLOC_SIZE`. A full-resident base uses one slot of
/// `n_layers`; otherwise ping-pong allocates two slots of `k_chunk` layers each.
pub fn compute_window_plan(
    pool: &OpenClDevicePool,
    layer_bytes: usize,
    n_layers: usize,
    strategy: MemoryStrategy,
    reserve_bytes: u64,
) -> WindowPlan {
    if layer_bytes == 0 || n_layers == 0 {
        return WindowPlan {
            k_chunk: 1,
            resident: false,
            window_bytes: 0,
        };
    }
    let layer_u64 = layer_bytes as u64;

    let available_bytes: u64 = match strategy {
        MemoryStrategy::Minimal => {
            return WindowPlan {
                k_chunk: 1,
                resident: false,
                window_bytes: layer_u64 * 2,
            };
        }
        MemoryStrategy::CapBytes(cap) => cap.saturating_sub(reserve_bytes),
        MemoryStrategy::AutoFit => {
            let host = system_available_ram_bytes()
                .map(|b| (b as f64 * DEFAULT_UTILIZATION) as u64);
            // The base lives in host RAM (SVM/pinned) and is mirrored to each
            // device that can hold it. Devices that cannot (base > their
            // max single-allocation) are skipped and use host-upload instead, so
            // the *best* device bounds the accelerator side (not the min).
            let dev = pool
                .engines
                .iter()
                .map(|e| {
                    let by_mem = (e.device_info.global_mem_size as f64 * DEFAULT_UTILIZATION) as u64;
                    let by_alloc = if e.device_info.max_alloc_size > 0 {
                        e.device_info.max_alloc_size
                    } else {
                        u64::MAX
                    };
                    by_mem.min(by_alloc)
                })
                .max();
            let total = match (host, dev) {
                (Some(h), Some(d)) => h.min(d),
                (Some(h), None) => h,
                (None, Some(d)) => d,
                (None, None) => 0,
            };
            total.saturating_sub(reserve_bytes)
        }
    };

    plan_from_available(layer_bytes, n_layers, strategy, available_bytes)
}

/// Map an available byte budget to a [`WindowPlan`] (pure; unit-testable).
pub fn plan_from_available(
    layer_bytes: usize,
    n_layers: usize,
    strategy: MemoryStrategy,
    available_bytes: u64,
) -> WindowPlan {
    if layer_bytes == 0 || n_layers == 0 {
        return WindowPlan {
            k_chunk: 1,
            resident: false,
            window_bytes: 0,
        };
    }
    let layer_u64 = layer_bytes as u64;
    if strategy == MemoryStrategy::Minimal {
        return WindowPlan {
            k_chunk: 1,
            resident: false,
            window_bytes: layer_u64 * 2,
        };
    }

    // Full resident base: one slot of `n_layers` layers (slot 1 is a dummy).
    let resident_needed = layer_u64.saturating_mul(n_layers as u64);
    if resident_needed > 0 && resident_needed <= available_bytes {
        return WindowPlan {
            k_chunk: n_layers,
            resident: true,
            window_bytes: resident_needed,
        };
    }

    // Ping-pong: two slots of `k_chunk` layers each, clamped to `< n_layers`.
    let max_k = n_layers.saturating_sub(1).max(1);
    let k_chunk = ((available_bytes / (2 * layer_u64)).max(1) as usize).min(max_k);
    WindowPlan {
        k_chunk,
        resident: false,
        window_bytes: 2 * layer_u64 * k_chunk as u64,
    }
}

/// Estimate available system RAM (best-effort, platform-specific).
fn system_available_ram_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        windows_available_ram()
    }
    #[cfg(target_os = "linux")]
    {
        linux_available_ram()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

#[cfg(windows)]
fn windows_available_ram() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }
    unsafe {
        let mut ms = MemoryStatusEx {
            dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };
        if GlobalMemoryStatusEx(&mut ms) != 0 {
            Some(ms.ull_avail_phys)
        } else {
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn linux_available_ram() -> Option<u64> {
    let data = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in data.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_strategy_gives_k1() {
        let pool = OpenClDevicePool::empty();
        let plan = compute_window_plan(&pool, 100_000, 32, MemoryStrategy::Minimal, 0);
        assert_eq!(plan.k_chunk, 1);
        assert!(!plan.resident);
    }

    #[test]
    fn cap_bytes_strategy() {
        let pool = OpenClDevicePool::empty();
        // 1 MB/layer, 10 MB cap → two slots → k_chunk = 10 / 2 = 5.
        let plan = compute_window_plan(&pool, 1_000_000, 32, MemoryStrategy::CapBytes(10_000_000), 0);
        assert_eq!(plan.k_chunk, 5);
        assert!(!plan.resident);
    }

    #[test]
    fn cap_bytes_reserves_kv_and_activations() {
        let pool = OpenClDevicePool::empty();
        // Half the cap reserved → k_chunk = (10 - 5) / 2 = 2 (integer div 5/2).
        let plan = compute_window_plan(&pool, 1_000_000, 32, MemoryStrategy::CapBytes(10_000_000), 5_000_000);
        assert_eq!(plan.k_chunk, 2);
    }

    #[test]
    fn plan_resident_when_base_fits() {
        // 32 layers × 1 MB = 32 MB base fits in 40 MB → resident, one slot.
        let p = plan_from_available(1_000_000, 32, MemoryStrategy::AutoFit, 40_000_000);
        assert!(p.resident);
        assert_eq!(p.k_chunk, 32);
        assert_eq!(p.window_bytes, 32_000_000);
    }

    #[test]
    fn plan_ping_pong_when_base_does_not_fit() {
        // 32 MB base > 10 MB → two slots of k_chunk = 10/2 = 5 layers.
        let p = plan_from_available(1_000_000, 32, MemoryStrategy::AutoFit, 10_000_000);
        assert!(!p.resident);
        assert_eq!(p.k_chunk, 5);
        assert_eq!(p.window_bytes, 10_000_000);
    }

    #[test]
    fn parse_strategy() {
        assert_eq!(MemoryStrategy::parse("auto"), MemoryStrategy::AutoFit);
        assert_eq!(MemoryStrategy::parse("minimal"), MemoryStrategy::Minimal);
        assert_eq!(MemoryStrategy::parse("2048"), MemoryStrategy::CapBytes(2048 * 1024 * 1024));
    }
}
