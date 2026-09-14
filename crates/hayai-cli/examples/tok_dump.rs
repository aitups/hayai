//! Vuelca los **ids de token** del corpus de calibración con EXACTAMENTE el
//! protocolo del evaluador del motor (`kl_eval.rs`): líneas del fichero unidas
//! con `\n`, `tokenizer.encode(corpus, add_bos)` y recorte a `n_pos`.
//!
//! Lo usa la referencia del modelo nuevo (grafo de nodos) para construir
//! `h = Embed(ids)` con los mismos tokens con los que se cachearon los logits
//! del profesor.
//!
//!   hayai tok_dump --model <gguf> --prompts <txt> --n-pos N --out <ids.bin>
//!
//! Escribe `[u32 n_tokens]` + `n_tokens` u32 LE.

use std::path::PathBuf;
use std::sync::Arc;

use hayai_model::{GgufCatalog, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut n_pos = 128usize;
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
                out = args.get(i).map(PathBuf::from);
            }
            "--n-pos" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_pos = v;
                }
            }
            other => {
                eprintln!("tok_dump: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let prompts = prompts.ok_or("falta --prompts <txt>")?;
    let out = out.ok_or("falta --out <ids.bin>")?;

    let texts: Vec<String> = std::fs::read_to_string(&prompts)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let corpus = texts.join("\n");
    let cat = Arc::new(GgufCatalog::open(&model)?);
    let tokenizer = Tokenizer::from_catalog(&cat)?;
    let tokens: Vec<u32> = tokenizer
        .encode(&corpus, tokenizer.add_bos)
        .into_iter()
        .take(n_pos)
        .collect();

    let mut buf = Vec::with_capacity(4 + tokens.len() * 4);
    buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for t in &tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    std::fs::write(&out, &buf)?;
    println!(
        "{{\"ok\":true,\"n_tokens\":{},\"out\":\"{}\"}}",
        tokens.len(),
        out.display()
    );
    Ok(())
}
