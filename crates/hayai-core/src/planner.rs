//! Hardware-aware placement planner.
//!
//! The engine must not hardcode "attention → CPU, FFN → GPU". Instead it reads the
//! model's op graph (bytes per op, dependency order), reads the measured capabilities
//! of every compute target (host RAM bandwidth, per-device effective GEMV bandwidth,
//! launch overhead, host→device link bandwidth, resident capacity) and *plans* where
//! each op runs so the makespan is minimized while the resident set stays bounded.
//!
//! The plan is cost-driven, not name-driven: a novel architecture with the same op
//! kinds is planned by the same code, and a faster device automatically attracts more
//! work. There are no per-model or per-vendor constants to go stale.

use crate::exec_plan::{LayerOpKind, OpDevice, OpKernel};
use std::collections::BTreeMap;

/// A concrete place an op can run: the host CPU or one OpenCL device (index into the
/// pool). Unlike [`OpDevice`] (a *capability class*), this identifies *which* device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ComputeTarget {
    Cpu,
    Device(usize),
}

impl std::fmt::Display for ComputeTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeTarget::Cpu => write!(f, "CPU"),
            ComputeTarget::Device(i) => write!(f, "dev{i}"),
        }
    }
}

/// Measured capability of one compute target.
#[derive(Debug, Clone)]
pub struct TargetCaps {
    pub target: ComputeTarget,
    pub name: String,
    /// Effective GEMV bandwidth (bytes/s) when the weights stream from host memory
    /// (includes the host→device upload). Host: pure RAM→compute bandwidth.
    pub gemv_bw: f64,
    /// Effective GEMV bandwidth (bytes/s) when the weights are already resident on the
    /// target (VRAM mirror / SVM owned). Host: equal to `gemv_bw`.
    pub resident_gemv_bw: f64,
    /// Fixed per-op launch cost in seconds (kernel enqueue + sync overhead).
    pub launch_s: f64,
    /// Host→target link bandwidth (bytes/s). CPU (host) uses the RAM bandwidth; a
    /// device uses its measured DMA bandwidth.
    pub link_bw: f64,
    /// Bytes usable to keep weights resident on this target.
    pub resident_bytes: u64,
    pub is_gpu: bool,
}

impl TargetCaps {
    /// Seconds to execute a GEMV with `bytes` of weights.
    ///
    /// `resident` = the weights already live on the target (no link transfer); else the
    /// bytes stream from host memory over `link_bw` (folded into `gemv_bw` for the
    /// non-resident device path, which is what the calibration measures).
    pub fn gemv_seconds(&self, bytes: u64, resident: bool) -> f64 {
        let b = bytes as f64;
        if self.is_gpu {
            let bw = if resident {
                self.resident_gemv_bw
            } else {
                self.gemv_bw
            };
            if bw <= 0.0 {
                return f64::INFINITY;
            }
            self.launch_s + b / bw
        } else {
            let bw = self.gemv_bw;
            if bw <= 0.0 {
                return f64::INFINITY;
            }
            self.launch_s + b / bw
        }
    }
}

/// One schedulable op instance distilled from the plan: its kind, weight bytes, the
/// capability class it needs, and its prerequisite task indices (a DAG).
#[derive(Debug, Clone)]
pub struct OpTask {
    pub kind: LayerOpKind,
    pub bytes: u64,
    pub class: OpDevice,
    pub deps: Vec<usize>,
}

/// True when an op can be *placed* (its math is expressible on any GEMV engine).
fn placeable(device: OpDevice, kernel: OpKernel) -> bool {
    match device {
        OpDevice::GpuAsync => true,
        OpDevice::Cpu => matches!(kernel, OpKernel::Gemv),
        OpDevice::HostRowRead | OpDevice::Discard => false,
    }
}

/// The result of planning: where each op kind runs.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Placement {
    pub assignments: BTreeMap<LayerOpKind, ComputeTarget>,
    /// Estimated per-target busy time (seconds) — the planning objective (minimize the
    /// maximum). Exposed for the utilisation gate and diagnostics.
    pub target_load_s: BTreeMap<ComputeTarget, f64>,
    /// Estimated makespan (seconds) of one full layer's op graph.
    pub makespan_s: f64,
}

impl Placement {
    pub fn target_of(&self, kind: LayerOpKind) -> Option<ComputeTarget> {
        self.assignments.get(&kind).copied()
    }
}

/// Total bytes of one op kind across the DAG.
fn bytes_by_kind(tasks: &[OpTask]) -> BTreeMap<LayerOpKind, u64> {
    let mut m: BTreeMap<LayerOpKind, u64> = BTreeMap::new();
    for t in tasks {
        *m.entry(t.kind).or_insert(0) += t.bytes;
    }
    m
}

/// Plan where each op kind runs.
///
/// Strategy: list scheduling over kinds ordered by **descending total bytes** (heavy
/// first). Each placeable kind is assigned to the target that finishes it earliest
/// given that target's accumulated load — the classic heterogeneous earliest-finish-time
/// heuristic. Fixed classes (norms, host row reads, discards) stay on the CPU. This is
/// robust for any model: it only needs the ops' byte sizes and the measured caps.
///
/// Weights are assumed streamed (non-resident) unless the target's `resident_bytes`
/// can hold the whole layer's placeable bytes, in which case the resident bandwidth is
/// used — i.e. the same budget logic the memory window uses, applied to compute.
pub fn plan_placement(tasks: &[OpTask], caps: &[TargetCaps], concurrent: bool) -> Placement {
    let mut out = Placement::default();
    if tasks.is_empty() || caps.is_empty() {
        return out;
    }
    let by_kind = bytes_by_kind(tasks);

    // Which targets can hold this layer's placeable weight set resident.
    let placeable_layer_bytes: u64 = tasks
        .iter()
        .filter(|t| placeable(t.class, kernel_of(t.kind)))
        .map(|t| t.bytes)
        .sum();

    // Heavy first: the kinds that dominate the layer's bytes are the ones worth moving.
    let mut order: Vec<(LayerOpKind, u64)> = by_kind.iter().map(|(k, b)| (*k, *b)).collect();
    order.sort_by(|a, b| b.1.cmp(&a.1));

    let mut load: BTreeMap<ComputeTarget, f64> = caps.iter().map(|c| (c.target, 0.0)).collect();

    for (kind, bytes) in order {
        let class = class_of(kind);
        let kernel = kernel_of(kind);
        if !placeable(class, kernel) {
            // Fixed: CPU or host. Norms / row reads do not move.
            out.assignments.insert(kind, ComputeTarget::Cpu);
            continue;
        }
        // Choose the target for this op.
        let mut best: Option<(ComputeTarget, f64, f64)> = None; // (target, finish, op_time)
        for c in caps {
            let resident =
                placeable_layer_bytes > 0 && placeable_layer_bytes <= c.resident_bytes && c.is_gpu;
            let t = c.gemv_seconds(bytes, resident);
            if !t.is_finite() {
                continue;
            }
            // Concurrent plan: the executor runs devices in parallel, so price the
            // target's accumulated load. Non-concurrent: pure per-op cost, because the
            // executor serializes — load-balancing onto a slower device only adds latency.
            let finish = if concurrent { load[&c.target] + t } else { t };
            if best.map(|(_, f, _)| finish < f).unwrap_or(true) {
                best = Some((c.target, finish, t));
            }
        }
        let (chosen, _finish, op_time) = match best {
            Some(x) => x,
            None => {
                out.assignments.insert(kind, ComputeTarget::Cpu);
                continue;
            }
        };
        *load.entry(chosen).or_insert(0.0) += op_time;
        out.assignments.insert(kind, chosen);
    }

    // Fallback for GEMV kinds with no target chosen (should not happen): CPU.
    for t in tasks {
        out.assignments.entry(t.kind).or_insert(ComputeTarget::Cpu);
    }

    out.target_load_s = load;
    out.makespan_s = if concurrent {
        out.target_load_s.values().fold(0.0f64, |m, &v| m.max(v))
    } else {
        out.target_load_s.values().sum()
    };
    out
}

use crate::exec_plan::op_binding;
fn class_of(kind: LayerOpKind) -> OpDevice {
    op_binding(kind).device
}
fn kernel_of(kind: LayerOpKind) -> OpKernel {
    op_binding(kind).kernel
}

/// The joint per-session plan: what stays resident, what streams and how fast, where
/// the compute goes and the resulting steady-state throughput. Everything is derived
/// from measured hardware capabilities and the model's weight sizes — there are no
/// per-model or per-vendor constants, so any hardware is planned on its measured merits.
#[derive(Debug, Clone)]
pub struct SessionPlan {
    /// Target that runs the weights.
    pub compute: ComputeTarget,
    /// Weight bytes kept resident on `compute` (device VRAM/SVM or host RAM).
    pub resident_bytes: u64,
    /// Weight bytes that must be (re)streamed from disk each token.
    pub stream_bytes: u64,
    /// Layers held in one window; `>= n_layers` means fully resident.
    pub window_layers: usize,
    /// True when the whole weight set is resident (I/O-free decode).
    pub resident: bool,
    /// In-flight layer reads the stream loop keeps to overlap disk with compute.
    pub prefetch_depth: usize,
    pub placement: Placement,
    /// Estimated steady-state tokens/s (1 / max(compute_time, io_time)).
    pub est_tokens_per_s: f64,
    /// The limiting resource: `"disk"` | `"compute"`.
    pub bottleneck: &'static str,
}

impl SessionPlan {
    pub fn summary(&self) -> String {
        format!(
            "compute={} resident={:.0} MiB stream={:.0} MiB window_layers={} prefetch={} est={:.2} tok/s (limited by {})",
            self.compute,
            self.resident_bytes as f64 / 1_048_576.0,
            self.stream_bytes as f64 / 1_048_576.0,
            self.window_layers,
            self.prefetch_depth,
            self.est_tokens_per_s,
            self.bottleneck,
        )
    }
}

/// Jointly decide residency, streaming cadence and compute placement.
///
/// For every measured target it prices the *whole* weight set when the largest
/// residency that target can hold is used: the resident part runs at the target's
/// resident bandwidth, the remainder streams from disk and must be moved over the
/// target's link (folded into `gemv_bw` for the non-resident device path, which is what
/// the calibration measures). Compute and disk I/O overlap, so the steady-state token
/// time is `max(compute, io)`; the target with the smallest makespan wins. The window
/// (cadence) and prefetch depth follow from how much fits resident and the per-layer
/// I/O vs compute ratio — again, only measured numbers.
#[allow(clippy::too_many_arguments)]
pub fn plan_session(
    caps: &[TargetCaps],
    tasks: &[OpTask],
    total_weight_bytes: u64,
    layer_bytes: u64,
    n_layers: usize,
    disk_bw: f64,
    reserve_bytes: u64,
) -> SessionPlan {
    let w = total_weight_bytes.max(1);
    let layered = layer_bytes > 0 && n_layers > 0;
    let mut best: Option<(ComputeTarget, u64, f64, f64, f64)> = None;
    for c in caps {
        let cap = c.resident_bytes.saturating_sub(reserve_bytes);
        let r = w.min(cap);
        let streamed = w - r;
        let t_res = if c.resident_gemv_bw > 0.0 {
            r as f64 / c.resident_gemv_bw
        } else {
            f64::INFINITY
        };
        let t_str = if c.gemv_bw > 0.0 {
            streamed as f64 / c.gemv_bw
        } else {
            f64::INFINITY
        };
        let t_compute = t_res + t_str;
        let t_io = if disk_bw > 0.0 {
            streamed as f64 / disk_bw
        } else {
            0.0
        };
        let makespan = t_compute.max(t_io);
        if !makespan.is_finite() {
            continue;
        }
        if best.map(|(_, _, _, _, m)| makespan < m).unwrap_or(true) {
            best = Some((c.target, r, t_compute, t_io, makespan));
        }
    }
    let (compute, resident_bytes, t_compute, t_io, makespan) = best.unwrap_or((
        ComputeTarget::Cpu,
        0,
        f64::INFINITY,
        0.0,
        f64::INFINITY,
    ));
    let stream_bytes = w - resident_bytes;
    let resident = stream_bytes == 0;
    let window_layers = if resident {
        n_layers
    } else if layered {
        ((resident_bytes / layer_bytes).max(1) as usize).min(n_layers.saturating_sub(1).max(1))
    } else {
        1
    };
    // Cadence: enough in-flight reads to keep the disk busy through one compute slice.
    let per_layer_compute = if layered && makespan.is_finite() {
        makespan / n_layers as f64
    } else {
        0.0
    };
    let per_layer_io = if layered && disk_bw > 0.0 {
        layer_bytes as f64 / disk_bw
    } else {
        0.0
    };
    let depth = if per_layer_compute > 0.0 {
        (per_layer_io / per_layer_compute).ceil() as usize
    } else {
        2
    };
    let prefetch_depth = depth.clamp(2, 8);
    let bottleneck = if t_io >= t_compute { "disk" } else { "compute" };

    // Price each target with the residency actually assigned to it, then place the ops.
    let mut priced = caps.to_vec();
    for c in &mut priced {
        c.resident_bytes = if c.target == compute { resident_bytes } else { 0 };
    }
    let placement = plan_placement(tasks, &priced, true);

    SessionPlan {
        compute,
        resident_bytes,
        stream_bytes,
        window_layers,
        resident,
        prefetch_depth,
        placement,
        est_tokens_per_s: if makespan > 0.0 { 1.0 / makespan } else { 0.0 },
        bottleneck,
    }
}

/// Build the planner's target list from a measured [`crate::calibration::HwProfile`].
///
/// Every number comes from the calibration (host RAM bandwidth, per-device effective
/// and resident GEMV bandwidth, launch overhead, DMA bandwidth, VRAM). Nothing here is
/// a vendor/model constant, so a novel device is planned on its measured merits.
pub fn caps_from_profile(
    profile: &crate::calibration::HwProfile,
    cpu_resident_bytes: u64,
) -> Vec<TargetCaps> {
    let host = profile.host_bw_gbytes_s.max(0.0) * 1e9;
    // The CPU runs a dequantising GEMV, whose real rate is far below the RAM memcpy
    // peak. Price the host with the measured compute rate so a GPU is not passed over.
    let host_gemv = if profile.host_gemv_gbytes_s > 0.0 {
        profile.host_gemv_gbytes_s * 1e9
    } else {
        host
    };
    let mut caps = vec![TargetCaps {
        target: ComputeTarget::Cpu,
        name: "CPU".into(),
        gemv_bw: host_gemv,
        resident_gemv_bw: host_gemv,
        launch_s: 0.0,
        link_bw: host,
        resident_bytes: cpu_resident_bytes,
        is_gpu: false,
    }];
    for d in &profile.devices {
        let stream = d.effective_gemv_gbytes_s.max(0.0) * 1e9;
        // Resident rate: prefer the real Q4_K figure (the quant real FFNs use — F32
        // overstates an integer-dequant kernel), else the F32 resident figure.
        let resident = if d.resident_gemv_q4k_gbytes_s > 0.0 {
            d.resident_gemv_q4k_gbytes_s * 1e9
        } else if d.resident_gemv_gbytes_s > 0.0 {
            d.resident_gemv_gbytes_s * 1e9
        } else {
            stream
        };
        let link = if d.dma_gbytes_s > 0.0 {
            d.dma_gbytes_s * 1e9
        } else {
            stream
        };
        // `gemv_bw` is the *streamed* rate (weights re-moved per op — what the
        // calibration measures end to end); `resident_gemv_bw` is the rate once the
        // weights live on the device. The planner decides which applies per layer from
        // the residency it assigns, so both must be the measured truth — pricing the
        // streamed path at the resident rate is what let an idle GPU look attractive.
        let mut vram_budget =
            (d.global_mem_bytes as f64 * crate::adaptive_window::DEFAULT_UTILIZATION) as u64;
        // A single resident allocation cannot exceed CL_DEVICE_MAX_MEM_ALLOC_SIZE; without
        // per-layer buffers, pricing a bigger model as resident makes the mirror fail and
        // silently re-upload the whole model every token (audit A.3.3).
        if d.max_alloc_bytes > 0 {
            vram_budget = vram_budget.min(d.max_alloc_bytes);
        }
        caps.push(TargetCaps {
            target: ComputeTarget::Device(d.device_index),
            name: d.name.clone(),
            gemv_bw: stream,
            resident_gemv_bw: resident,
            launch_s: d.launch_us.max(0.0) * 1e-6,
            link_bw: link,
            resident_bytes: vram_budget,
            is_gpu: true,
        });
    }
    caps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps2() -> Vec<TargetCaps> {
        vec![
            TargetCaps {
                target: ComputeTarget::Cpu,
                name: "cpu".into(),
                gemv_bw: 20e9,
                resident_gemv_bw: 20e9,
                launch_s: 1e-6,
                link_bw: 20e9,
                resident_bytes: u64::MAX,
                is_gpu: false,
            },
            TargetCaps {
                target: ComputeTarget::Device(0),
                name: "t4".into(),
                gemv_bw: 30e9, // streamed (upload-bound)
                resident_gemv_bw: 250e9,
                launch_s: 5e-5,
                link_bw: 12e9,
                resident_bytes: 0,
                is_gpu: true,
            },
        ]
    }

    fn task(kind: LayerOpKind, bytes: u64) -> OpTask {
        OpTask {
            kind,
            bytes,
            class: op_binding(kind).device,
            deps: vec![],
        }
    }

    #[test]
    fn large_ffn_goes_to_the_faster_device() {
        // dGPU streams FFN faster than the CPU (30 vs 20 GB/s) → FFN picked for it.
        let tasks = vec![
            task(LayerOpKind::AttnQ, 2_000_000),
            task(LayerOpKind::FfnGate, 40_000_000),
        ];
        let p = plan_placement(&tasks, &caps2(), false);
        assert_eq!(
            p.target_of(LayerOpKind::FfnGate),
            Some(ComputeTarget::Device(0))
        );
    }

    #[test]
    fn attention_is_placeable_and_can_leave_the_cpu() {
        // Attention is class Cpu + Gemv (placeable): a fast device should attract it.
        let tasks = vec![
            task(LayerOpKind::AttnQ, 50_000_000),
            task(LayerOpKind::FfnGate, 1_000_000),
        ];
        let p = plan_placement(&tasks, &caps2(), false);
        assert_eq!(
            p.target_of(LayerOpKind::AttnQ),
            Some(ComputeTarget::Device(0))
        );
    }

    #[test]
    fn norms_stay_on_cpu() {
        let tasks = vec![
            task(LayerOpKind::AttnNorm, 1_000_000),
            task(LayerOpKind::FfnNorm, 1_000_000),
        ];
        let p = plan_placement(&tasks, &caps2(), false);
        assert_eq!(p.target_of(LayerOpKind::AttnNorm), Some(ComputeTarget::Cpu));
        assert_eq!(p.target_of(LayerOpKind::FfnNorm), Some(ComputeTarget::Cpu));
    }

    #[test]
    fn load_balances_across_two_fast_devices() {
        // Two identical fast GPUs: two equal heavy kinds should split, not pile on one.
        let mut c = caps2();
        c.push(TargetCaps {
            target: ComputeTarget::Device(1),
            name: "t4b".into(),
            gemv_bw: 30e9,
            resident_gemv_bw: 250e9,
            launch_s: 5e-5,
            link_bw: 12e9,
            resident_bytes: 0,
            is_gpu: true,
        });
        let tasks = vec![
            task(LayerOpKind::FfnGate, 40_000_000),
            task(LayerOpKind::FfnUp, 40_000_000),
        ];
        let p = plan_placement(&tasks, &c, true);
        let d0 = p.target_of(LayerOpKind::FfnGate);
        let d1 = p.target_of(LayerOpKind::FfnUp);
        assert_ne!(d0, d1, "equal heavy kinds must not both land on one device");
    }

    #[test]
    fn no_gpu_puts_everything_on_cpu() {
        let cpu_only = vec![TargetCaps {
            target: ComputeTarget::Cpu,
            name: "cpu".into(),
            gemv_bw: 20e9,
            resident_gemv_bw: 20e9,
            launch_s: 1e-6,
            link_bw: 20e9,
            resident_bytes: u64::MAX,
            is_gpu: false,
        }];
        let tasks = vec![
            task(LayerOpKind::FfnGate, 40_000_000),
            task(LayerOpKind::AttnQ, 2_000_000),
        ];
        let p = plan_placement(&tasks, &cpu_only, false);
        assert_eq!(p.target_of(LayerOpKind::FfnGate), Some(ComputeTarget::Cpu));
        assert_eq!(p.target_of(LayerOpKind::AttnQ), Some(ComputeTarget::Cpu));
    }

    fn cpu_cap(gemv: f64) -> TargetCaps {
        TargetCaps {
            target: ComputeTarget::Cpu,
            name: "cpu".into(),
            gemv_bw: gemv,
            resident_gemv_bw: gemv,
            launch_s: 0.0,
            link_bw: 10e9,
            resident_bytes: u64::MAX,
            is_gpu: false,
        }
    }

    fn gpu_cap(resident_bw: f64, vram: u64) -> TargetCaps {
        TargetCaps {
            target: ComputeTarget::Device(0),
            name: "gpu".into(),
            gemv_bw: 3e9,
            resident_gemv_bw: resident_bw,
            launch_s: 5e-5,
            link_bw: 12e9,
            resident_bytes: vram,
            is_gpu: true,
        }
    }

    #[test]
    fn session_resident_when_device_holds_the_model() {
        // 32 layers × 10 MB = 320 MB; the device holds it all → resident, compute-bound.
        let caps = vec![cpu_cap(10e9), gpu_cap(200e9, 2_000_000_000)];
        let tasks = vec![task(LayerOpKind::FfnGate, 10_000_000)];
        let p = plan_session(&caps, &tasks, 320_000_000, 10_000_000, 32, 1e9, 0);
        assert!(p.resident);
        assert_eq!(p.compute, ComputeTarget::Device(0));
        assert_eq!(p.window_layers, 32);
        assert_eq!(p.stream_bytes, 0);
        assert_eq!(p.bottleneck, "compute");
    }

    #[test]
    fn session_streams_when_the_device_is_too_small() {
        // 100 MB model, both targets hold 40 MB → the rest streams; disk is not the
        // limit here (fast disk) but the slow streaming GEMV is → device still wins.
        let cpu = TargetCaps {
            target: ComputeTarget::Cpu,
            name: "cpu".into(),
            gemv_bw: 3e9,
            resident_gemv_bw: 3e9,
            launch_s: 0.0,
            link_bw: 10e9,
            resident_bytes: 40_000_000,
            is_gpu: false,
        };
        let gpu = TargetCaps {
            target: ComputeTarget::Device(0),
            name: "gpu".into(),
            gemv_bw: 5e9,
            resident_gemv_bw: 200e9,
            launch_s: 5e-5,
            link_bw: 12e9,
            resident_bytes: 40_000_000,
            is_gpu: true,
        };
        let tasks = vec![task(LayerOpKind::FfnGate, 10_000_000)];
        let p = plan_session(&[cpu, gpu], &tasks, 100_000_000, 10_000_000, 10, 10e9, 0);
        assert!(!p.resident);
        assert_eq!(p.compute, ComputeTarget::Device(0));
        assert_eq!(p.resident_bytes, 40_000_000);
        assert_eq!(p.stream_bytes, 60_000_000);
        assert!(p.window_layers >= 1 && p.window_layers < 10);
    }

    #[test]
    fn session_prefers_the_faster_target() {
        // CPU far faster than the tiny/spinning device → CPU wins, stays resident.
        let caps = vec![cpu_cap(50e9), gpu_cap(200e9, 0)];
        let tasks = vec![task(LayerOpKind::FfnGate, 10_000_000)];
        let p = plan_session(&caps, &tasks, 100_000_000, 10_000_000, 10, 1e9, 0);
        assert_eq!(p.compute, ComputeTarget::Cpu);
        assert!(p.resident);
    }
}
