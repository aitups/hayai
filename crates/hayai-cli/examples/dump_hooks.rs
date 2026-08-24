//! Captura los **hooks de activación reales** (Fase 2): la entrada FFN de cada
//! capa (salida del `ffn_norm`) para un lote de calibración de alta entropía.
//!
//!   hayai dump-hooks --model <gguf> --prompts <txt con B textos> --out <dir>
//!
//! Escribe `X_<layer>.bin` (f32 LE, `[posiciones, d_in]`) por capa + un JSON con
//! los metadatos (B, d_in, posiciones). Reemplaza el `X ~ N(0,1)` del evolutivo
//! para esparsificar sobre la distribución real de activaciones.

use std::path::PathBuf;
use std::sync::Arc;

use hayai_core::{EngineOrchestrator, ExecutionMode, Generator};
use hayai_model::{GgufFile, LlamaWeights, SamplerConfig, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut max_positions = 4096usize;
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
            "--out" => {
                i += 1;
                out_dir = args.get(i).map(PathBuf::from);
            }
            "--device" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    device = d.clone();
                }
            }
            "--max-positions" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    max_positions = v;
                }
            }
            other => {
                eprintln!("dump-hooks: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let prompts = prompts.ok_or("falta --prompts <txt>")?;
    let out_dir = out_dir.ok_or("falta --out <dir>")?;
    std::fs::create_dir_all(&out_dir)?;

    let texts: Vec<String> = std::fs::read_to_string(&prompts)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if texts.is_empty() {
        return Err("archivo de prompts vacío".into());
    }
    let corpus = texts.join("\n");

    let gguf = Arc::new(GgufFile::open(&model)?);
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let weights = LlamaWeights::load_arc(gguf)?;
    let sampler = SamplerConfig::default();
    let mut orch = EngineOrchestrator::new(ExecutionMode::parse(&device), weights.config.clone());
    let mut gen = Generator::new(weights, tokenizer, 4, 128, sampler, 42);

    let n_layers = gen.weights.config.num_layers;
    let d_in = gen.weights.config.hidden_size;
    let tokens = gen.tokenizer.encode(&corpus, false);
    let n_pos = tokens.len().min(max_positions);

    // X[layer] se acumula en filas de d_in.
    let mut x_per_layer: Vec<Vec<f32>> = vec![Vec::with_capacity(n_pos * d_in); n_layers];
    let mut hooks: Vec<Vec<f32>> = vec![vec![0.0f32; d_in]; n_layers];
    for (t, &tok) in tokens.iter().enumerate().take(n_pos) {
        gen.forward_with_hooks(&mut orch, tok, &mut hooks)?;
        for (layer, hx) in hooks.iter().enumerate() {
            x_per_layer[layer].extend_from_slice(hx);
        }
        if (t + 1) % 64 == 0 {
            eprintln!("dump-hooks: token {}/{}", t + 1, n_pos);
        }
    }

    for (layer, x) in x_per_layer.iter().enumerate() {
        let fname = out_dir.join(format!("X_{layer:04}.bin"));
        let mut bytes = Vec::with_capacity(x.len() * 4);
        for v in x {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&fname, bytes)?;
    }
    let meta = format!(
        "{{\"n_layers\":{},\"d_in\":{},\"n_positions\":{},\"n_texts\":{},\"texts_file\":\"{}\"}}",
        n_layers,
        d_in,
        n_pos,
        texts.len(),
        prompts.to_string_lossy().replace('\\', "\\\\")
    );
    std::fs::write(out_dir.join("hooks.json"), meta)?;
    eprintln!(
        "dump-hooks: {n_layers} capas x {n_pos} posiciones x {d_in} dims -> {out_dir:?}"
    );
    Ok(())
}
