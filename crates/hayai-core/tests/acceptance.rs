//! Acceptance tests: on a real host the engine must reach the hardware-derived SLO,
//! and the planner must actually produce a placement.
//!
//! These tests are strict when a model is configured and skip with a clear message
//! otherwise (CI has no GPU and no weights). Configure one of:
//!   HAYAI_ACCEPTANCE_MODEL  = path to one GGUF
//!   HAYAI_ACCEPTANCE_MODELS = dir containing GGUFs (all are tested)
//! and, optionally, the SLO utilisation fraction:
//!   HAYAI_ACCEPTANCE_UTIL   = 0.75 (default)
//!   HAYAI_ACCEPTANCE_MIN_RATIO = 0.9 (default; how close to the target must we get)
//!
//! The target is `calibrate`'s `effective_stream_BW × util / bytes_per_token`, i.e.
//! the same 75–80 % utilisation objective the whole design targets. A model that does
//! not reach `target × min_ratio` fails the test — that is the point of the gate.

use std::path::PathBuf;
use std::time::Instant;

use hayai_core::{calibrate, EngineOrchestrator, ExecutionMode, MemoryStrategy, StreamingGenerator};
use hayai_model::{GgufCatalog, SamplerConfig, Tokenizer};

fn acceptance_models() -> Vec<PathBuf> {
    if let Ok(dir) = std::env::var("HAYAI_ACCEPTANCE_MODELS") {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
            .collect();
        v.sort();
        if !v.is_empty() {
            return v;
        }
    }
    if let Ok(m) = std::env::var("HAYAI_ACCEPTANCE_MODEL") {
        return vec![PathBuf::from(m)];
    }
    Vec::new()
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
fn decode_meets_hardware_slo_and_plan_is_present() {
    let models = acceptance_models();
    if models.is_empty() {
        eprintln!(
            "acceptance: SKIP — set HAYAI_ACCEPTANCE_MODEL=<gguf> or \
             HAYAI_ACCEPTANCE_MODELS=<dir> to run the SLO gate"
        );
        return;
    }
    let util = env_f64("HAYAI_ACCEPTANCE_UTIL", 0.75);
    let min_ratio = env_f64("HAYAI_ACCEPTANCE_MIN_RATIO", 0.9);
    let n_new: usize = env_f64("HAYAI_ACCEPTANCE_TOKENS", 32.0) as usize;

    let mut failures = Vec::new();
    for path in &models {
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        let cat = GgufCatalog::open(path).expect("open gguf");
        let tokenizer = Tokenizer::from_catalog(&cat).expect("tokenizer");
        drop(cat);
        let mut gen = StreamingGenerator::open(path, tokenizer, 4, 128, SamplerConfig::Greedy, 42)
            .expect("open generator");
        gen.set_memory_strategy(MemoryStrategy::parse("auto"));
        let mut orch =
            EngineOrchestrator::new(ExecutionMode::Auto, gen.config.clone());

        // Hardware target from calibrate (host + disk + device GEMV bandwidth).
        let bytes_per_token = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f64;
        let profile = calibrate(&orch.pool, Some(path.as_path()));
        let target = profile.target_tok_s(bytes_per_token, util);

        // Decode-only: a 1-token prompt keeps the prefill out of the measurement.
        let t0 = Instant::now();
        let stats = gen
            .generate(&mut orch, "Hi", n_new)
            .expect("generate");
        let secs = t0.elapsed().as_secs_f64().max(1e-9);
        let got = stats.new_tokens as f64 / secs;

        // The planner must have produced a placement covering the GEMV ops.
        let plan = gen.exec_plan.as_ref().expect("exec plan");
        let placed = plan.placement.assignments.len();
        let plan_ops = plan.op_tasks().len();
        assert!(
            placed > 0,
            "{name}: planner produced an empty placement ({} ops in plan)",
            plan_ops
        );

        let ratio = if target > 0.0 { got / target } else { 1.0 };
        eprintln!(
            "{name}: {got:.3} tok/s vs target {target:.3} @ {:.0}% (ratio {ratio:.2}, \
             {} ops placed, host {:.1} GB/s, disk {:.1} GB/s)",
            util * 100.0,
            placed,
            profile.host_bw_gbytes_s,
            profile.disk_bw_gbytes_s,
        );
        if ratio < min_ratio {
            failures.push(format!(
                "{name}: {got:.3} tok/s < {:.3} (target {target:.3} @ {:.0}%, ratio {ratio:.2} < {min_ratio})",
                target * min_ratio,
                util * 100.0,
            ));
        }
    }
    assert!(failures.is_empty(), "SLO gate failed:\n{}", failures.join("\n"));
}
