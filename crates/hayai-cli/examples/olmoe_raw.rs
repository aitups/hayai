//! Temp validation: raw continuation (no chat template) through the MoE forward.
use hayai_core::{EngineOrchestrator, ExecutionMode, StreamingGenerator};
use hayai_model::{sample, GgufCatalog, SamplerConfig, Tokenizer};

fn main() {
    let path = "models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf";
    let cat = GgufCatalog::open(path).unwrap();
    let tokenizer = Tokenizer::from_catalog(&cat).unwrap();
    drop(cat);
    let mut gen = StreamingGenerator::open(
        path,
        tokenizer,
        4,
        256,
        SamplerConfig::Greedy,
        42,
    )
    .unwrap();
    let mut orch = EngineOrchestrator::new(ExecutionMode::CpuOnly, gen.config.clone());
    let mut scratch = gen.prepare_session(&mut orch).unwrap();

    // Raw ids: bos + "The capital of France is" (no chat template markers).
    let mut ids = vec![gen.tokenizer.bos_id];
    ids.extend(gen.tokenizer.encode("The capital of France is", false));
    let mut last = gen.prefill(&mut orch, &ids, &mut scratch).unwrap();

    let mut out: Vec<u32> = Vec::new();
    let smoke_n: usize = std::env::var("HAYAI_SMOKE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    for _ in 0..smoke_n {
        let next = sample(&last, gen.sampler, &mut gen.rng);
        if next == gen.tokenizer.eos_id {
            break;
        }
        out.push(next);
        last = gen.decode_step(&mut orch, next, &mut scratch).unwrap();
    }
    println!("RAWTEXT: {:?}", gen.tokenizer.decode(&out));
    let (hits, misses) = gen.moe_cache_stats();
    println!(
        "MOE_CACHE: hits={} misses={} io_bytes={} ({:.1} MiB for {} forwards)",
        hits,
        misses,
        gen.io_bytes,
        gen.io_bytes as f64 / (1024.0 * 1024.0),
        gen.position
    );
}
