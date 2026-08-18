//! Heterogeneous macro-pipeline: CPU Attention ∥ GPU FFN.
//!
//! For each decoder layer of a decode step:
//! 1. CPU runs GQA attention + residual (KV cache INT8).
//! 2. GPU FFN (Q4 LUT MatMul) is enqueued; while it runs the CPU prepares
//!    the next layer's scratch (activation copy / bookkeeping).
//! 3. Wait for the FFN event, apply residual, proceed.
//!
//! `run_overlap_probe` additionally measures true concurrent CPU+GPU utilization
//! with independent workloads (stress test for the sync path).

use hayai_cpu::{
    attention_decode_step, cpu_lut_matmul_q4, rms_norm, AttentionConfig, LayerKvCache,
};
use hayai_model::ModelConfig;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tracing::info;

use crate::orchestrator::{EngineOrchestrator, OrchestratorError};

/// Synthetic Q4 FFN weights for one layer (gate/up/down packed into one matmul tile).
pub struct SyntheticFfnWeights {
    pub gate_q4: Vec<u8>,
    pub up_q4: Vec<u8>,
    pub down_q4: Vec<u8>,
    pub lut: [f32; 16],
    pub intermediate: usize,
    pub hidden: usize,
}

impl SyntheticFfnWeights {
    pub fn new(hidden: usize, intermediate: usize, seed: u64) -> Self {
        let packed_gate = intermediate * hidden / 2;
        let packed_down = hidden * intermediate / 2;
        let mut gate_q4 = vec![0u8; packed_gate];
        let mut up_q4 = vec![0u8; packed_gate];
        let mut down_q4 = vec![0u8; packed_down];
        for i in 0..packed_gate {
            gate_q4[i] = ((seed.wrapping_mul(1103515245) + i as u64) % 256) as u8;
            up_q4[i] = ((seed.wrapping_mul(12345) + i as u64 * 7) % 256) as u8;
        }
        for i in 0..packed_down {
            down_q4[i] = ((seed.wrapping_mul(99991) + i as u64 * 13) % 256) as u8;
        }
        let lut = [
            -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ];
        Self {
            gate_q4,
            up_q4,
            down_q4,
            lut,
            intermediate,
            hidden,
        }
    }
}

/// Timing breakdown for a multi-layer decode step.
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub layers: usize,
    pub attn_secs: f64,
    pub ffn_secs: f64,
    pub wait_secs: f64,
    pub total_secs: f64,
    pub used_opencl: bool,
}

/// Overlap probe: independent CPU attn + GPU ffn running concurrently.
#[derive(Debug, Clone)]
pub struct OverlapProbeStats {
    pub cpu_only_secs: f64,
    pub gpu_only_secs: f64,
    pub parallel_secs: f64,
    pub speedup: f64,
    pub used_opencl: bool,
}

pub struct HeterogeneousPipeline {
    pub attn_cfg: AttentionConfig,
    pub layer_kv: Vec<LayerKvCache>,
    pub ffn_weights: Vec<SyntheticFfnWeights>,
    pub norm_weight: Vec<f32>,
    pub rms_eps: f32,
    pub num_sink: usize,
    pub window: usize,
}

impl HeterogeneousPipeline {
    pub fn new(config: &ModelConfig, num_sink: usize, window: usize) -> Self {
        let attn_cfg = AttentionConfig::from_model(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.rope_theta,
        );
        let layer_kv = (0..config.num_layers)
            .map(|_| {
                LayerKvCache::new(
                    attn_cfg.num_kv_heads,
                    attn_cfg.head_dim,
                    num_sink,
                    window,
                )
            })
            .collect();
        let ffn_weights = (0..config.num_layers)
            .map(|i| {
                SyntheticFfnWeights::new(config.hidden_size, config.intermediate_size, 42 + i as u64)
            })
            .collect();
        Self {
            attn_cfg,
            layer_kv,
            ffn_weights,
            norm_weight: vec![1.0f32; config.hidden_size],
            rms_eps: config.rms_norm_eps,
            num_sink,
            window,
        }
    }

    /// One full decode step through all layers (CPU Attn + GPU/CPU FFN).
    pub fn forward_token(
        &mut self,
        orch: &mut EngineOrchestrator,
        hidden: &mut [f32],
        position: usize,
    ) -> Result<PipelineStats, OrchestratorError> {
        assert_eq!(hidden.len(), self.attn_cfg.hidden_size());
        let t0 = Instant::now();
        let mut attn_secs = 0.0;
        let mut ffn_secs = 0.0;
        let mut wait_secs = 0.0;
        let used_opencl = orch.using_opencl();

        let layers = self.layer_kv.len();
        for layer in 0..layers {
            // --- Pre-attn norm (working copy) ---
            let mut x_norm = hidden.to_vec();
            rms_norm(&mut x_norm, &self.norm_weight, self.rms_eps);

            // Synthetic Q/K/V projections (Phase 4 will use real GGUF weights).
            let mut q = project_synthetic(&x_norm, self.attn_cfg.hidden_size(), layer as u64 + 1);
            let mut k = project_synthetic(&x_norm, self.attn_cfg.kv_dim(), layer as u64 + 2);
            let v = project_synthetic(&x_norm, self.attn_cfg.kv_dim(), layer as u64 + 3);

            let t_attn = Instant::now();
            let mut attn_out = vec![0.0f32; self.attn_cfg.hidden_size()];
            attention_decode_step(
                &self.attn_cfg,
                &mut self.layer_kv[layer],
                &mut q,
                &mut k,
                &v,
                position,
                &mut attn_out,
            );
            attn_secs += t_attn.elapsed().as_secs_f64();

            // Residual after attention
            for i in 0..hidden.len() {
                hidden[i] += attn_out[i];
            }

            // Pre-FFN norm
            let mut h_norm = hidden.to_vec();
            rms_norm(&mut h_norm, &self.norm_weight, self.rms_eps);

            let w = &self.ffn_weights[layer];
            let mut gate = vec![0.0f32; w.intermediate];
            let mut up = vec![0.0f32; w.intermediate];
            let mut down = vec![0.0f32; w.hidden];

            // Enqueue / run FFN. Overlap bookkeeping for next layer while waiting.
            let t_ffn = Instant::now();
            if used_opencl {
                // Run gate+up on GPU path via orchestrator; overlap CPU prep of a scratch buffer.
                let (tx, rx) = mpsc::channel::<()>();
                let hidden_cap = w.hidden;
                let prep = std::thread::spawn(move || {
                    // Simulate next-layer activation scratch warm-up while GPU works.
                    let mut scratch = vec![0.0f32; hidden_cap];
                    for i in 0..scratch.len() {
                        scratch[i] = (i as f32) * 0.001;
                    }
                    let _ = tx.send(());
                    scratch
                });

                orch.execute_lut_matmul(
                    w.intermediate,
                    w.hidden,
                    &w.gate_q4,
                    &w.lut,
                    &h_norm,
                    &mut gate,
                )?;
                orch.execute_lut_matmul(
                    w.intermediate,
                    w.hidden,
                    &w.up_q4,
                    &w.lut,
                    &h_norm,
                    &mut up,
                )?;

                let t_wait = Instant::now();
                let _ = rx.recv_timeout(Duration::from_secs(2));
                let _ = prep.join();
                wait_secs += t_wait.elapsed().as_secs_f64();
            } else {
                cpu_lut_matmul_q4(
                    w.intermediate,
                    w.hidden,
                    &w.gate_q4,
                    &w.lut,
                    &h_norm,
                    &mut gate,
                );
                cpu_lut_matmul_q4(
                    w.intermediate,
                    w.hidden,
                    &w.up_q4,
                    &w.lut,
                    &h_norm,
                    &mut up,
                );
            }

            // SiLU(gate) * up
            for i in 0..gate.len() {
                let x = gate[i];
                gate[i] = (x / (1.0 + (-x).exp())) * up[i];
            }

            orch.execute_lut_matmul(w.hidden, w.intermediate, &w.down_q4, &w.lut, &gate, &mut down)?;
            ffn_secs += t_ffn.elapsed().as_secs_f64();

            for i in 0..hidden.len() {
                hidden[i] += down[i];
            }
        }

        Ok(PipelineStats {
            layers,
            attn_secs,
            ffn_secs,
            wait_secs,
            total_secs: t0.elapsed().as_secs_f64(),
            used_opencl,
        })
    }
}

fn project_synthetic(input: &[f32], out_dim: usize, seed: u64) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    let inv = 1.0 / (input.len() as f32).sqrt();
    for o in 0..out_dim {
        let mut sum = 0.0f32;
        for (i, &x) in input.iter().enumerate() {
            let w = (((seed.wrapping_mul(17) + (o as u64).wrapping_mul(31) + i as u64) % 200)
                as f32
                - 100.0)
                * 0.01;
            sum += x * w;
        }
        out[o] = sum * inv;
    }
    out
}

/// Measure wall-clock of CPU attn alone, GPU ffn alone, and both in parallel.
pub fn run_overlap_probe(
    orch: &mut EngineOrchestrator,
    config: &ModelConfig,
    iters: usize,
) -> OverlapProbeStats {
    let attn_cfg = AttentionConfig::from_model(
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.rope_theta,
    );
    let used_opencl = orch.using_opencl();

    let run_cpu_attn = {
        let attn_cfg = attn_cfg;
        let iters = iters;
        move || {
            let mut cache = LayerKvCache::new(attn_cfg.num_kv_heads, attn_cfg.head_dim, 4, 64);
            let mut hidden = vec![0.5f32; attn_cfg.hidden_size()];
            for pos in 0..iters {
                let mut q = project_synthetic(&hidden, attn_cfg.hidden_size(), 1);
                let mut k = project_synthetic(&hidden, attn_cfg.kv_dim(), 2);
                let v = project_synthetic(&hidden, attn_cfg.kv_dim(), 3);
                let mut out = vec![0.0f32; attn_cfg.hidden_size()];
                attention_decode_step(&attn_cfg, &mut cache, &mut q, &mut k, &v, pos, &mut out);
                for i in 0..hidden.len() {
                    hidden[i] = 0.9 * hidden[i] + 0.1 * out[i];
                }
            }
        }
    };

    let ffn = SyntheticFfnWeights::new(config.hidden_size, config.intermediate_size, 7);
    let input = vec![1.0f32; ffn.hidden];

    let run_ffn = |orch: &mut EngineOrchestrator| {
        let mut gate = vec![0.0f32; ffn.intermediate];
        for _ in 0..iters {
            let _ = orch.execute_lut_matmul(
                ffn.intermediate,
                ffn.hidden,
                &ffn.gate_q4,
                &ffn.lut,
                &input,
                &mut gate,
            );
        }
    };

    let t = Instant::now();
    run_cpu_attn();
    let cpu_only_secs = t.elapsed().as_secs_f64();

    let t = Instant::now();
    run_ffn(orch);
    let gpu_only_secs = t.elapsed().as_secs_f64();

    // Rebuild CPU closure for the parallel run (previous was consumed).
    let run_cpu_attn_2 = {
        let attn_cfg = AttentionConfig::from_model(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.rope_theta,
        );
        let iters = iters;
        move || {
            let mut cache = LayerKvCache::new(attn_cfg.num_kv_heads, attn_cfg.head_dim, 4, 64);
            let mut hidden = vec![0.5f32; attn_cfg.hidden_size()];
            for pos in 0..iters {
                let mut q = project_synthetic(&hidden, attn_cfg.hidden_size(), 1);
                let mut k = project_synthetic(&hidden, attn_cfg.kv_dim(), 2);
                let v = project_synthetic(&hidden, attn_cfg.kv_dim(), 3);
                let mut out = vec![0.0f32; attn_cfg.hidden_size()];
                attention_decode_step(&attn_cfg, &mut cache, &mut q, &mut k, &v, pos, &mut out);
                for i in 0..hidden.len() {
                    hidden[i] = 0.9 * hidden[i] + 0.1 * out[i];
                }
            }
        }
    };

    let t = Instant::now();
    let handle = std::thread::spawn(run_cpu_attn_2);
    run_ffn(orch);
    let _ = handle.join();
    let parallel_secs = t.elapsed().as_secs_f64();

    let serial = cpu_only_secs + gpu_only_secs;
    let speedup = serial / parallel_secs.max(1e-9);

    info!(
        "Overlap probe: cpu={cpu_only_secs:.4}s gpu={gpu_only_secs:.4}s parallel={parallel_secs:.4}s speedup={speedup:.2}x opencl={used_opencl}"
    );

    OverlapProbeStats {
        cpu_only_secs,
        gpu_only_secs,
        parallel_secs,
        speedup,
        used_opencl,
    }
}

/// Convenience: build pipeline + run N tokens.
pub fn run_decode_benchmark(
    orch: &mut EngineOrchestrator,
    config: &ModelConfig,
    tokens: usize,
    sink: usize,
    window: usize,
) -> Result<(PipelineStats, Vec<f32>), OrchestratorError> {
    let mut pipe = HeterogeneousPipeline::new(config, sink, window);
    let mut hidden = vec![0.1f32; config.hidden_size];
    let mut agg = PipelineStats {
        used_opencl: orch.using_opencl(),
        ..Default::default()
    };

    for pos in 0..tokens {
        let s = pipe.forward_token(orch, &mut hidden, pos)?;
        agg.layers = s.layers;
        agg.attn_secs += s.attn_secs;
        agg.ffn_secs += s.ffn_secs;
        agg.wait_secs += s.wait_secs;
        agg.total_secs += s.total_secs;
    }

    Ok((agg, hidden))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::{EngineOrchestrator, ExecutionMode};

    #[test]
    fn decode_step_grows_kv_and_stays_finite() {
        let config = ModelConfig::smollm_135m();
        // Use a tiny synthetic config via truncated pipeline layers.
        let mut tiny = config.clone();
        tiny.num_layers = 2;
        tiny.hidden_size = 64;
        tiny.intermediate_size = 128;
        tiny.num_attention_heads = 4;
        tiny.num_key_value_heads = 2;

        let mut orch = EngineOrchestrator::new(ExecutionMode::CpuOnly, tiny.clone());
        let mut pipe = HeterogeneousPipeline::new(&tiny, 2, 8);
        let mut hidden = vec![0.25f32; tiny.hidden_size];

        for pos in 0..12 {
            let stats = pipe.forward_token(&mut orch, &mut hidden, pos).unwrap();
            assert_eq!(stats.layers, 2);
            assert!(hidden.iter().all(|x| x.is_finite()));
        }
        assert_eq!(pipe.layer_kv[0].resident_len(), 10); // 2 sinks + 8 window
        assert_eq!(pipe.layer_kv[0].heads[0].current_len, 12);
    }

    #[test]
    fn overlap_probe_runs() {
        let config = ModelConfig::smollm_135m();
        let mut tiny = config;
        tiny.hidden_size = 64;
        tiny.intermediate_size = 128;
        tiny.num_attention_heads = 4;
        tiny.num_key_value_heads = 2;
        tiny.num_layers = 1;

        let mut orch = EngineOrchestrator::new(ExecutionMode::CpuOnly, tiny.clone());
        let stats = run_overlap_probe(&mut orch, &tiny, 3);
        assert!(stats.cpu_only_secs > 0.0);
        assert!(stats.parallel_secs > 0.0);
        // CPU-only probe has thread overhead; require a sane positive ratio.
        assert!(stats.speedup > 0.2, "speedup={}", stats.speedup);
    }
}
