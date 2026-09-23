//! Process memory metrics (RSS) and streaming memory budget (Phase 5 / PRD §3.1–3.4).

use hayai_model::ModelConfig;

/// Resident set size of the current process in bytes, if available.
pub fn process_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(windows)]
    {
        windows_rss_bytes()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(windows)]
fn windows_rss_bytes() -> Option<u64> {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            ppsmemcounters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }

    unsafe {
        let mut pmc = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        let ok = GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb);
        if ok == 0 {
            None
        } else {
            Some(pmc.working_set_size as u64)
        }
    }
}

pub fn format_bytes(n: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    format!("{:.1} MiB", n as f64 / MIB)
}

/// Enforceable steady-state streaming footprint (Hayai-owned bytes).
///
/// Excludes OS/runtime/OpenCL driver RSS — those are reported separately.
#[derive(Debug, Clone, Copy)]
pub struct StreamingMemoryBudget {
    pub layer_window_bytes: u64,
    pub kv_bytes: u64,
    pub activation_bytes: u64,
    /// Catalog header / tokenizer / misc metadata allowance (not weight windows).
    pub metadata_slack_bytes: u64,
    pub total_budget_bytes: u64,
}

impl StreamingMemoryBudget {
    /// KV cache + activation working set in bytes, independent of the weight
    /// window. Used both for the budget and to reserve room in the planner.
    pub fn kv_activation_bytes(
        config: &ModelConfig,
        num_sink_tokens: usize,
        window_size: usize,
    ) -> u64 {
        let head_dim = config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .unwrap_or(0);
        let kv_dim = config.num_key_value_heads.saturating_mul(head_dim);
        let slots = num_sink_tokens.saturating_add(window_size) as u64;
        let per_layer = slots
            .saturating_mul(kv_dim as u64)
            .saturating_mul(2)
            .saturating_add(slots.saturating_mul(8));
        let kv_bytes = per_layer.saturating_mul(config.num_layers as u64);
        let h = config.hidden_size as u64;
        let ff = config.intermediate_size as u64;
        let activation_bytes = (2 * h + 2 * ff + h) * 4;
        kv_bytes.saturating_add(activation_bytes)
    }

    /// Estimate from model dims + layer-window size + KV sinks/window.
    ///
    /// `k_chunk` = layers per ping-pong slot. The scratch really allocates **two**
    /// slots of `k_chunk × layer` each (`hetero_scratch::allocate_for_pool`), so the
    /// weight window is `2 · k_chunk · layer_pack_bytes` — a full-resident model is
    /// `k_chunk == n_layers`.
    pub fn estimate(
        config: &ModelConfig,
        layer_pack_bytes: usize,
        k_chunk: usize,
        num_sink_tokens: usize,
        window_size: usize,
    ) -> Self {
        let layer_window_bytes = 2u64
            .saturating_mul(k_chunk.max(1) as u64)
            .saturating_mul(layer_pack_bytes as u64);

        let kv_act = Self::kv_activation_bytes(config, num_sink_tokens, window_size);
        let head_dim = config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .unwrap_or(0);
        let kv_dim = config.num_key_value_heads.saturating_mul(head_dim);
        let slots = num_sink_tokens.saturating_add(window_size) as u64;
        let per_layer = slots
            .saturating_mul(kv_dim as u64)
            .saturating_mul(2)
            .saturating_add(slots.saturating_mul(8));
        let kv_bytes = per_layer.saturating_mul(config.num_layers as u64);
        let h = config.hidden_size as u64;
        let ff = config.intermediate_size as u64;
        let activation_bytes = (2 * h + 2 * ff + h) * 4;

        // Tokenizer vocab strings + GGUF header/metadata living in RAM.
        let metadata_slack_bytes = 64 * 1024 * 1024; // 64 MiB slack

        let total_budget_bytes = layer_window_bytes
            .saturating_add(kv_act)
            .saturating_add(metadata_slack_bytes);

        Self {
            layer_window_bytes,
            kv_bytes,
            activation_bytes,
            metadata_slack_bytes,
            total_budget_bytes,
        }
    }
}

/// Runtime accounting of bytes Hayai itself allocates for streaming inference.
#[derive(Debug, Clone, Copy, Default)]
pub struct HayaiOwnedMemory {
    pub scratch_bytes: u64,
    pub kv_bytes: u64,
    pub activation_bytes: u64,
    pub prefetch_staging_bytes: u64,
    pub metadata_bytes: u64,
    pub peak_bytes: u64,
}

impl HayaiOwnedMemory {
    pub fn total(&self) -> u64 {
        self.scratch_bytes
            .saturating_add(self.kv_bytes)
            .saturating_add(self.activation_bytes)
            .saturating_add(self.prefetch_staging_bytes)
            .saturating_add(self.metadata_bytes)
    }

    fn touch_peak(&mut self) {
        self.peak_bytes = self.peak_bytes.max(self.total());
    }

    pub fn note_scratch(&mut self, bytes: u64) {
        self.scratch_bytes = bytes;
        self.touch_peak();
    }

    pub fn note_kv(&mut self, bytes: u64) {
        self.kv_bytes = bytes;
        self.touch_peak();
    }

    pub fn note_activations(&mut self, bytes: u64) {
        self.activation_bytes = bytes;
        self.touch_peak();
    }

    pub fn note_prefetch_staging(&mut self, bytes: u64) {
        self.prefetch_staging_bytes = bytes;
        self.touch_peak();
    }

    pub fn note_metadata(&mut self, bytes: u64) {
        self.metadata_bytes = bytes;
        self.touch_peak();
    }

    /// Fail if peak exceeds the enforceable budget.
    pub fn check_budget(&self, budget: &StreamingMemoryBudget) -> Result<(), String> {
        if self.peak_bytes > budget.total_budget_bytes {
            Err(format!(
                "Hayai-owned peak {} exceeds budget {} (scratch={} kv={} acts={} staging={} meta={})",
                format_bytes(self.peak_bytes),
                format_bytes(budget.total_budget_bytes),
                format_bytes(self.scratch_bytes),
                format_bytes(self.kv_bytes),
                format_bytes(self.activation_bytes),
                format_bytes(self.prefetch_staging_bytes),
                format_bytes(self.metadata_bytes),
            ))
        } else {
            Ok(())
        }
    }
}
