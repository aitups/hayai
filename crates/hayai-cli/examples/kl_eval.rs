//! Evaluador KL del streaming: compara los logits teacher-forced de DOS modelos
//! (original vs. embebido disperso D16) con el `StreamingGenerator`.
//!
//!   hayai kl_eval --orig <original.gguf> --sparse <embebido.gguf> --prompts <txt>
//!               [--n-positions N] [--device cpu]
//!
//! Imprime JSON: `{kl_global, d_arch_global, n_positions}` donde `d_arch_global`
//! se estima desde los bloques dispersos del modelo embebido (popcount de la
//! adyacencia). Es el evaluador de la frontera de Pareto para modelos grandes
//! (ALIA/Qwen) sobre el path de producción (StreamingGenerator), que soporta el
//! FFN disperso embebido (D16) y los híbridos Qwen3.5 (DeltaNet/SSM).

use std::path::PathBuf;
use std::sync::Arc;

use hayai_core::{EngineOrchestrator, ExecutionMode, StreamingGenerator};
use hayai_model::{GgufCatalog, SamplerConfig, Tokenizer};

fn softmax_kl(lo: &[f32], lc: &[f32]) -> f32 {
    let max0 = lo.iter().fold(f32::MIN, |a, b| a.max(*b));
    let max1 = lc.iter().fold(f32::MIN, |a, b| a.max(*b));
    let mut p0 = 0.0f32;
    let mut p1 = 0.0f32;
    let mut e0 = Vec::with_capacity(lo.len());
    let mut e1 = Vec::with_capacity(lo.len());
    for i in 0..lo.len() {
        let a = (lo[i] - max0).exp();
        let b = (lc[i] - max1).exp();
        e0.push(a);
        e1.push(b);
        p0 += a;
        p1 += b;
    }
    let eps = 1e-9f32;
    let mut kl0 = 0.0f32;
    let mut kl1 = 0.0f32;
    for i in 0..lo.len() {
        let q0 = e0[i] / (p0 + eps);
        let q1 = e1[i] / (p1 + eps);
        kl0 += q0 * ((q0 + eps) / (q1 + eps)).ln();
        kl1 += q1 * ((q1 + eps) / (q0 + eps)).ln();
    }
    0.5 * (kl0 + kl1)
}

fn popcount(b: u8) -> u32 {
    b.count_ones()
}

/// D_arch global estimado desde los bloques dispersos del modelo embebido:
/// params FFN totales (denominador completo) vs. activos por bloque disperso.
///
/// En el modelo embebido los bloques esparsos ya no tienen `.weight` denso
/// (solo `ffn_dag_adjacency` + metadatos `saor.blk.N.<rol>.*`).
fn d_arch_from_embedded(cat: &mut GgufCatalog) -> Result<f32, Box<dyn std::error::Error>> {
    let mut den = 0.0f32;
    let mut num = 0.0f32;
    // Recolecta (adyacencia, params) primero; las lecturas piden `&mut cat`.
    let mut reads: Vec<(String, f32)> = Vec::new();
    for t in &cat.tensors {
        let name = &t.name;
        // Bloques FFN que siguen densos (sp=0): sus `.weight` permanecen.
        if name.ends_with(".weight") {
            let Some(base) = name.strip_prefix("blk.") else { continue };
            let Some((_layer_s, role)) = base.split_once('.') else { continue };
            let role = role.strip_suffix(".weight").unwrap_or(role);
            if ["ffn_gate", "ffn_up", "ffn_down"].contains(&role) {
                den += (t.ncols() * t.nrows()) as f32;
            }
            continue;
        }
        // Bloques esparsos: `blk.N.<rol>.ffn_dag_adjacency` + metadatos.
        if let Some(rest) = name.strip_suffix(".ffn_dag_adjacency") {
            let Some(base) = rest.strip_prefix("blk.") else { continue };
            let Some((layer_s, role)) = base.split_once('.') else { continue };
            if !["ffn_gate", "ffn_up", "ffn_down"].contains(&role) {
                continue;
            }
            let d_in = cat
                .meta_u32(&format!("saor.blk.{layer_s}.{role}.d_in"))
                .unwrap_or(0) as f32;
            let d_out = cat
                .meta_u32(&format!("saor.blk.{layer_s}.{role}.d_out"))
                .unwrap_or(0) as f32;
            let params = d_in * d_out;
            if params > 0.0 {
                den += params;
                reads.push((name.clone(), params));
            }
        }
    }
    for (adj_name, params) in reads {
        let adj_info = cat.tensor(&adj_name)?.clone();
        // La adyacencia es un bit-tensor: su longitud real es dims[0] (el tipo
        // GGML puede estar mal reportado por el escritor).
        let nbytes = adj_info.dims.first().copied().unwrap_or(0) as usize;
        let mut buf = vec![0u8; nbytes];
        cat.read_raw_at(cat.tensor_abs_offset(&adj_info), &mut buf)?;
        let active: u32 = buf.iter().map(|b| b.count_ones()).sum();
        num += (1.0 - active as f32 / params) * params;
    }
    Ok(if den > 0.0 { num / den } else { 0.0 })
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut orig: Option<PathBuf> = None;
    let mut sparse: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut device = "auto".to_string();
    let mut n_positions = 128usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--orig" => {
                i += 1;
                orig = args.get(i).map(PathBuf::from);
            }
            "--sparse" => {
                i += 1;
                sparse = args.get(i).map(PathBuf::from);
            }
            "--prompts" => {
                i += 1;
                prompts = args.get(i).map(PathBuf::from);
            }
            "--device" => {
                i += 1;
                if let Some(d) = args.get(i) {
                    device = d.clone();
                }
            }
            "--n-positions" => {
                i += 1;
                if let Some(v) = args.get(i).and_then(|s| s.parse().ok()) {
                    n_positions = v;
                }
            }
            other => {
                eprintln!("kl_eval: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let orig = orig.ok_or("falta --orig <gguf>")?;
    let sparse = sparse.ok_or("falta --sparse <gguf>")?;
    let prompts = prompts.ok_or("falta --prompts <txt>")?;

    let texts: Vec<String> = std::fs::read_to_string(&prompts)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let corpus = texts.join("\n");

    let cat_orig = Arc::new(GgufCatalog::open(&orig)?);
    let tokenizer = Tokenizer::from_catalog(&cat_orig)?;
    let tokens = tokenizer.encode(&corpus, tokenizer.add_bos);
    let n_pos = tokens.len().min(n_positions);
    if n_pos == 0 {
        return Err("corpus sin tokens".into());
    }

    let mut gen_o =
        StreamingGenerator::open(&orig, tokenizer.clone(), 4, 128, SamplerConfig::default(), 42)?;
    let mut gen_s =
        StreamingGenerator::open(&sparse, tokenizer, 4, 128, SamplerConfig::default(), 42)?;
    // VRAM limitada (RTX 4050: 6 GB): modo minimal (2 slots ping-pong), sin
    // ventana residente que supere la VRAM en modelos de 40B.
    gen_o.set_memory_strategy(hayai_core::MemoryStrategy::Minimal);
    gen_s.set_memory_strategy(hayai_core::MemoryStrategy::Minimal);
    let mode = ExecutionMode::parse(&device);
    let mut orch = EngineOrchestrator::new(mode, gen_o.config.clone());

    // Secuencial (un scratch SVM a la vez): el forward de 40B en GPU no deja
    // espacio en VRAM para dos generadores simultáneos (RTX 4050: 6 GB).
    let mut so = gen_o.prepare_session(&mut orch)?;
    let mut orig_logits: Vec<Vec<f32>> = Vec::with_capacity(n_pos);
    for &tok in tokens.iter().take(n_pos) {
        orig_logits.push(gen_o.decode_step(&mut orch, tok, &mut so)?);
    }
    drop(so);

    let mut ss = gen_s.prepare_session(&mut orch)?;
    let mut kl_sum = 0.0f32;
    for (i, &tok) in tokens.iter().take(n_pos).enumerate() {
        let ls = gen_s.decode_step(&mut orch, tok, &mut ss)?;
        kl_sum += softmax_kl(&orig_logits[i], &ls);
    }
    let kl_global = kl_sum / n_pos as f32;

    let mut cat_s = GgufCatalog::open(&sparse)?;
    let d_arch_global = d_arch_from_embedded(&mut cat_s)?;

    println!(
        "{{\"kl_global\":{:.6},\"d_arch_global\":{:.4},\"n_positions\":{}}}",
        kl_global, d_arch_global, n_pos
    );
    Ok(())
}
