//! Prueba del override de FFN en runtime (Fase 2): `decode_step` (path normal)
//! vs `decode_step_with_override` con override VACÍO deben producir los mismos
//! logits en cualquier arquitectura (Dense, Hybrid). Valida que el override no
//! altera el forward base y que la inyección funciona sin re-embeder.
//!
//!   hayai ov_probe --model <gguf> --prompts <txt> [--n-positions N]

use std::path::PathBuf;

use hayai_core::{EngineOrchestrator, ExecutionMode, FfnOverride, MemoryStrategy, StreamingGenerator};
use hayai_model::{GgufCatalog, SamplerConfig, Tokenizer};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut n_positions = 4usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).map(PathBuf::from);
            }
            "--prompts" => {
                i += 1;
                prompts = args.get(i).map(PathBuf::from);
            }
            "--n-positions" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_positions = v;
                }
            }
            other => {
                eprintln!("ov_probe: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let prompts = prompts.ok_or("falta --prompts <txt>")?;

    let texts: Vec<String> = std::fs::read_to_string(&prompts)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let corpus = texts.join("\n");

    let cat = GgufCatalog::open(&model)?;
    let tokenizer = Tokenizer::from_catalog(&cat)?;
    let mut gen_a = StreamingGenerator::open(&model, tokenizer.clone(), 4, 128, SamplerConfig::default(), 42)?;
    let mut gen_b = StreamingGenerator::open(&model, tokenizer, 4, 128, SamplerConfig::default(), 42)?;
    gen_a.set_memory_strategy(MemoryStrategy::Minimal);
    gen_b.set_memory_strategy(MemoryStrategy::Minimal);
    let mut orch = EngineOrchestrator::new(ExecutionMode::parse("auto"), gen_a.config.clone());
    let mut scratch_a = gen_a.prepare_session(&mut orch)?;
    let mut scratch_b = gen_b.prepare_session(&mut orch)?;

    let tokens = gen_a.tokenizer.encode(&corpus, false);
    let n_pos = tokens.len().min(n_positions);
    let n_layers = gen_a.config.num_layers;
    let empty: Vec<FfnOverride> = vec![FfnOverride::default(); n_layers];

    // Mismo token y misma posición en generadores independientes (KV separadas).
    let mut worst = 0.0f32;
    for (i, &tok) in tokens.iter().take(n_pos).enumerate() {
        let a = gen_a.decode_step(&mut orch, tok, &mut scratch_a)?;
        let b = gen_b.decode_step_with_override(&mut orch, tok, &mut scratch_b, &empty)?;
        let d = max_abs_diff(&a, &b);
        worst = worst.max(d);
        eprintln!("[ov_probe] pos {i}: max|Δ| = {d:.6e}");
    }
    println!(
        "{{\"ok\":{},\"model\":\"{}\",\"n_pos\":{},\"n_layers\":{},\"worst_max_abs_diff\":{:.3e}}}",
        worst < 1e-4,
        model.display(),
        n_pos,
        n_layers,
        worst
    );
    Ok(())
}
