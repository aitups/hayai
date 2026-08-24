//! Vuelca los logits teacher-forced de un modelo GGUF para un prompt.
//!
//! Permite medir la divergencia KL original vs. candidato (contrato de Fase 2):
//!   hayai dump_logits --model <gguf> --prompt-file <txt> --out <logits.bin>
//! Escribe los logits `[n_tokens, vocab]` en f32 LE + un JSON con metadatos.
//!
//! Soporta el formato embebido de saor (FFN disperso): los bloques sustituidos
//! se ejecutan vía CSR (Generator::forward).

use std::path::PathBuf;
use std::sync::Arc;

use hayai_core::{EngineOrchestrator, ExecutionMode, Generator};
use hayai_model::{GgufFile, LlamaWeights, SamplerConfig, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompt_file: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).map(PathBuf::from);
            }
            "--prompt-file" => {
                i += 1;
                prompt_file = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            "--device" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    device = d.clone();
                }
            }
            other => {
                eprintln!("dump_logits: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let prompt_file = prompt_file.ok_or("falta --prompt-file <txt>")?;
    let out = out.ok_or("falta --out <logits.bin>")?;

    let prompt = std::fs::read_to_string(&prompt_file)?;
    let gguf = Arc::new(GgufFile::open(&model)?);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = LlamaWeights::load_arc(gguf)?;
    let sampler = SamplerConfig::default();
    let mut orch = EngineOrchestrator::new(ExecutionMode::parse(&device), weights.config.clone());
    let mut gen = Generator::new(weights, tokenizer, 4, 32, sampler, 42);

    let tokens = gen.tokenizer.encode(&prompt, false);
    if tokens.is_empty() {
        return Err("prompt vacío tras tokenizar".into());
    }
    let mut logits_all: Vec<f32> = Vec::with_capacity(tokens.len() * gen.weights.config.vocab_size);
    for &tok in &tokens {
        let logits = gen.forward(&mut orch, tok)?;
        logits_all.extend_from_slice(&logits);
    }

    let mut bytes = Vec::with_capacity(logits_all.len() * 4);
    for x in &logits_all {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(&out, bytes)?;
    let tokens_json: Vec<String> = tokens.iter().map(|t| t.to_string()).collect();
    let meta = format!(
        "{{\"n_tokens\":{},\"vocab\":{},\"tokens\":[{}],\"logits_f32_le\":\"{}\"}}",
        tokens.len(),
        gen.weights.config.vocab_size,
        tokens_json.join(","),
        out.to_string_lossy().replace('\\', "\\\\")
    );
    let mut meta_path = out.clone();
    meta_path.set_extension("json");
    std::fs::write(meta_path, meta)?;
    eprintln!(
        "dump_logits: {} tokens x {} vocab -> {}",
        tokens.len(),
        gen.weights.config.vocab_size,
        out.display()
    );
    Ok(())
}
