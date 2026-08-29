//! Benchmark de velocidad de decodificación del `StreamingGenerator` (D35): mide
//! tokens/segundo del modelo original frente al esparso embebido de `saor`, en
//! los dos modos de memoria (`AutoFit` = ventana automática, `Minimal` = 2 slots
//! ping-pong). Teacher-forced sobre el corpus de calibración.
//!
//!   hayai bench_speed --model <gguf> --prompts <txt> [--n-tokens 32] [--device cpu]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use hayai_core::{EngineOrchestrator, ExecutionMode, MemoryStrategy, StreamingGenerator};
use hayai_model::{GgufCatalog, SamplerConfig, Tokenizer};

fn bench_strategy(
    gen: &mut StreamingGenerator,
    tokens: &[u32],
    strategy: MemoryStrategy,
    device: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    gen.set_memory_strategy(strategy);
    let mode = ExecutionMode::parse(device);
    let mut orch = EngineOrchestrator::new(mode, gen.config.clone());
    let mut so = gen.prepare_session(&mut orch)?;
    let t0 = Instant::now();
    for &tok in tokens {
        gen.decode_step(&mut orch, tok, &mut so)?;
    }
    let secs = t0.elapsed().as_secs_f64();
    Ok(tokens.len() as f64 / secs)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut n_tokens = 32usize;
    let mut device = "cpu".to_string();
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
            "--n-tokens" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_tokens = v;
                }
            }
            "--device" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    device = d.clone();
                }
            }
            other => {
                eprintln!("bench_speed: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let prompts = prompts.ok_or("falta --prompts <txt>")?;

    let cat = Arc::new(GgufCatalog::open(&model)?);
    let tokenizer = Tokenizer::from_catalog(&cat)?;
    let corpus = std::fs::read_to_string(&prompts)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let tokens = tokenizer.encode(&corpus, tokenizer.add_bos);
    let tokens: Vec<u32> = tokens.iter().take(n_tokens).copied().collect();
    println!(
        "bench_speed: {} | {} tokens ({} prefijo) | device {device}",
        model.display(),
        tokens.len(),
        n_tokens
    );

    for strategy in [MemoryStrategy::Minimal, MemoryStrategy::AutoFit] {
        let mut gen =
            StreamingGenerator::open(&model, tokenizer.clone(), 4, 128, SamplerConfig::default(), 42)?;
        let tps = bench_strategy(&mut gen, &tokens, strategy, &device)?;
        println!(
            "  {strategy:?}: {tps:.2} tokens/s ({:.2} ms/token)",
            1000.0 / tps
        );
    }
    Ok(())
}
