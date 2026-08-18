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
/// - the user memory strategy.
///
/// Safety margin: keep 20 % of each device's reported memory free for OS/driver/KV.
pub fn compute_window_plan(
    pool: &OpenClDevicePool,
    layer_bytes: usize,
    n_layers: usize,
    strategy: MemoryStrategy,
) -> WindowPlan {
    if layer_bytes == 0 || n_layers == 0 {
        return WindowPlan { k_chunk: 1, resident: false, window_bytes: 0 };
    }

    let available_bytes: u64 = match strategy {
        MemoryStrategy::Minimal => {
            // Exactly 2 slots (current behaviour).
            return WindowPlan {
                k_chunk: 1,
                resident: false,
                window_bytes: layer_bytes as u64 * 2,
            };
        }
        MemoryStrategy::CapBytes(cap) => cap,
        MemoryStrategy::AutoFit => {
            // Sum usable headroom across all accelerators.
            // For each device take 80 % of global_mem_size.
            let gpu_mem: u64 = pool
                .engines
                .iter()
                .map(|e| {
                    let m = e.device_info.global_mem_size;
                    // Apply 80 % safety margin.
                    m.saturating_mul(4) / 5
                })
                .max()
                .unwrap_or(0);

            // Also consider system RAM (conservative: half of physical RAM, capped at 8 GiB).
            // We use a simple heuristic: if no GPU, fall back to RAM estimate.
            let sys_ram_estimate: u64 = system_available_ram_bytes()
                .map(|b| b.saturating_mul(1) / 2)
                .unwrap_or(0)
                .min(8 * 1024 * 1024 * 1024);

            if pool.is_empty() {
                sys_ram_estimate
            } else {
                gpu_mem.max(sys_ram_estimate)
            }
        }
    };

    // How many layers fit in the available headroom (always at least 1, at most n_layers)?
    let layer_bytes_u64 = layer_bytes as u64;

    // Reserve 2× base slots regardless (ping-pong overhead), then add extra layers.
    let base = layer_bytes_u64.saturating_mul(2);
    let extra_bytes = available_bytes.saturating_sub(base);
    let extra_layers = (extra_bytes / layer_bytes_u64) as usize;
    let k_chunk = (1 + extra_layers).min(n_layers);

    let resident = k_chunk >= n_layers;
    let window_bytes = layer_bytes_u64.saturating_mul(k_chunk as u64 + 1); // +1 for ping-pong

    WindowPlan { k_chunk, resident, window_bytes }
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
        let plan = compute_window_plan(&pool, 100_000, 32, MemoryStrategy::Minimal);
        assert_eq!(plan.k_chunk, 1);
        assert!(!plan.resident);
    }

    #[test]
    fn cap_bytes_strategy() {
        let pool = OpenClDevicePool::empty();
        // 1 MB per layer, cap 10 MB → k_chunk = min(10, 32) = 9 (10MB / 1MB - 2 base slots = 8 extra, +1 = 9)
        let plan = compute_window_plan(&pool, 1_000_000, 32, MemoryStrategy::CapBytes(10_000_000));
        assert!(plan.k_chunk >= 8);
    }

    #[test]
    fn parse_strategy() {
        assert_eq!(MemoryStrategy::parse("auto"), MemoryStrategy::AutoFit);
        assert_eq!(MemoryStrategy::parse("minimal"), MemoryStrategy::Minimal);
        assert_eq!(MemoryStrategy::parse("2048"), MemoryStrategy::CapBytes(2048 * 1024 * 1024));
    }
}
