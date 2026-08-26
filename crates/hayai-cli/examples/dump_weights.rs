//! Vuelca los pesos F32 (dequant) de los bloques FFN del profesor para el
//! evaluador de frontera (`embed_sparse` de saor-streamer).
//!
//!   hayai dump_weights --model <gguf> --out <dir>
//!
//! Escribe `w.{layer}.{gate|up|down}.bin` con `d_out*d_in` f32 en orden i-mayor
//! (conn = i*d_out+j), y un `meta.json` con `d_in`/`d_out` por bloque.

use std::fs;
use std::path::PathBuf;

use hayai_model::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut blocks: Option<String> = None; // "gate" | "all" (default)
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).map(PathBuf::from);
            }
            "--out" => {
                i += 1;
                out = args.get(i).map(PathBuf::from);
            }
            "--blocks" => {
                i += 1;
                blocks = args.get(i).cloned();
            }
            other => {
                eprintln!("dump_weights: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let out = out.ok_or("falta --out <dir>")?;
    let only_gate = blocks.as_deref() == Some("gate");
    fs::create_dir_all(&out)?;

    let gguf = GgufFile::open(&model)?;
    let n_layers: usize = gguf
        .tensors
        .iter()
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);

    let mut meta = String::from("{");
    for layer in 0..n_layers {
        for block in ["ffn_gate", "ffn_up", "ffn_down"] {
            if only_gate && block != "ffn_gate" {
                continue;
            }
            let name = format!("blk.{layer}.{block}.weight");
            let info = match gguf.tensor(&name) {
                Ok(t) => t.clone(),
                Err(_) => continue,
            };
            let bytes = gguf.tensor_bytes(&info)?;
            let w = hayai_model::gguf::dequantize(&info, bytes)?;
            let fname = out.join(format!("w.{layer}.{block}.bin"));
            fs::write(&fname, {
                let mut v = Vec::with_capacity(w.len() * 4);
                for x in &w {
                    v.extend_from_slice(&x.to_le_bytes());
                }
                v
            })?;
            meta.push_str(&format!(
                "\"blk.{layer}.{block}\":{{\"d_in\":{},\"d_out\":{}}},\n",
                info.ncols(),
                info.nrows()
            ));
        }
    }
    meta.push_str("\"n_layers\":");
    meta.push_str(&n_layers.to_string());
    meta.push_str("}");
    fs::write(out.join("meta.json"), meta)?;
    println!("dump_weights: {n_layers} capas -> {}", out.display());
    Ok(())
}
