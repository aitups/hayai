//! Diagnóstico: lista tensores de atención de un modelo GGUF (fused qkv vs separado).
use std::path::PathBuf;
use hayai_model::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model = PathBuf::from(args.get(1).expect("uso: list_attn <model.gguf>"));
    let gguf = GgufFile::open(&model)?;
    let names: Vec<&str> = gguf.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut qkv = 0usize;
    let mut q_sep = 0usize;
    let mut ssm = 0usize;
    for n in &names {
        if n.contains("attn_qkv") { qkv += 1; }
        if n.ends_with("attn_q.weight") { q_sep += 1; }
        if n.contains("ssm_") { ssm += 1; }
    }
    println!("attn_qkv={qkv} attn_q_separado={q_sep} tensores_ssm={ssm}");
    // Mapa por capa: qué atención tiene cada bloque físico.
    let mut max_layer = 0usize;
    for n in &names {
        if let Some(rest) = n.strip_prefix("blk.") {
            if let Some(l) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                max_layer = max_layer.max(l);
            }
        }
    }
    for l in 0..=max_layer {
        let has_qkv = names.iter().any(|n| n == &format!("blk.{l}.attn_qkv.weight"));
        let has_q = names.iter().any(|n| n == &format!("blk.{l}.attn_q.weight"));
        let has_ssm = names.iter().any(|n| n == &format!("blk.{l}.ssm_a"));
        if has_qkv || has_q || has_ssm {
            println!("  blk.{l}: qkv={has_qkv} q_sep={has_q} ssm={has_ssm}");
        }
    }
    Ok(())
}
