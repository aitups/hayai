//! Evaluador global Pareto (D20): mide `KL_global` y `D_arch_global` de un
//! candidato de esparsidad por capa (poda por magnitud del gate del profesor).
//!
//!   hayai eval_sparse --model <gguf> --prompts <txt> --sparsities <file>
//!                     [--n-positions N] [--device cpu]
//!
//! `--sparsities`: un float por línea (por capa; 0 = densa). Aplica la poda por
//! magnitud al gate de cada capa y compara los logits del modelo original vs el
//! candidato en lockstep (dos `Generator` con KV independientes). Imprime JSON:
//! `{kl_global, d_arch_global, n_positions}`.

use std::path::PathBuf;
use std::sync::Arc;

use hayai_core::{EngineOrchestrator, ExecutionMode, FfnOverride, Generator};
use hayai_model::{
    sparse_dag_to_csr, CsrSparse, GgufFile, LlamaWeights, SamplerConfig, Tokenizer,
};

fn softmax_kl(lo: &[f32], lc: &[f32]) -> f32 {
    // KL simétrica de las distribuciones softmax de un vector de logits.
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

fn magnitude_prune_csr(w0: &[f32], d_in: usize, d_out: usize, sparsity: f32) -> CsrSparse {
    // Máscara por magnitud: conserva la fracción (1-sparsity) de mayor |w|.
    let total = d_in * d_out;
    let keep = ((1.0 - sparsity) * total as f32) as usize;
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| (w0[b].abs()).total_cmp(&w0[a].abs()));
    let mut active = vec![false; total];
    for &idx in order.iter().take(keep) {
        active[idx] = true;
    }
    // Adyacencia (conn = i*d_out+j) + pesos en orden i-mayor.
    let mut bits = vec![0u8; total.div_ceil(8)];
    let mut weights = Vec::with_capacity(keep);
    for i in 0..d_in {
        for j in 0..d_out {
            let conn = i * d_out + j;
            if active[conn] {
                bits[conn / 8] |= 1 << (conn % 8);
                weights.push(w0[j * d_in + i]);
            }
        }
    }
    let (row_ptr, col_idx, vals) = sparse_dag_to_csr(&bits, &weights, d_in, d_out);
    CsrSparse {
        row_ptr,
        col_idx,
        vals,
        d_in,
        d_out,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut sparsities: Option<PathBuf> = None;
    let mut device = "cpu".to_string();
    let mut n_positions = 128usize;
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
            "--sparsities" => {
                i += 1;
                sparsities = args.get(i).map(PathBuf::from);
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
                eprintln!("eval_sparse: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let prompts = prompts.ok_or("falta --prompts <txt>")?;
    let sparsities = sparsities.ok_or("falta --sparsities <file>")?;

    let sp_raw: Vec<String> = std::fs::read_to_string(&sparsities)?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    // Cada línea: "gate up down" (3 floats) o "gate" (solo gate, retrocompatible).
    let sp_gate: Vec<f32> = sp_raw
        .iter()
        .map(|l| l.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0.0))
        .collect();
    let sp_up: Vec<f32> = sp_raw
        .iter()
        .map(|l| {
            l.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0.0)
        })
        .collect();
    let sp_down: Vec<f32> = sp_raw
        .iter()
        .map(|l| {
            l.split_whitespace().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0.0)
        })
        .collect();

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

    let n_layers = weights.config.num_layers;
    let tokens = tokenizer.encode(&corpus, false);
    let n_pos = tokens.len().min(n_positions);

    // Overrides: poda por magnitud del profesor por bloque y capa (sp>0).
    let mut overrides: Vec<FfnOverride> = vec![FfnOverride::default(); n_layers];
    // D_arch global: esparsidad ponderada por TODOS los parámetros FFN del modelo
    // (denominador completo — no solo las capas esparsas).
    let mut d_arch_num = 0.0f32;
    let mut d_arch_den = 0.0f32;
    for layer_idx in 0..n_layers {
        let trio: [(&str, f32, &hayai_model::QuantMatrix); 3] = [
            ("ffn_gate", sp_gate.get(layer_idx).copied().unwrap_or(0.0), &weights.layers[layer_idx].gate),
            ("ffn_up", sp_up.get(layer_idx).copied().unwrap_or(0.0), &weights.layers[layer_idx].up),
            ("ffn_down", sp_down.get(layer_idx).copied().unwrap_or(0.0), &weights.layers[layer_idx].down),
        ];
        for (block, sp, m) in trio {
            let params = (m.ncols * m.nrows) as f32;
            d_arch_den += params;
            if sp <= 0.0 {
                continue;
            }
            let name = format!("blk.{layer_idx}.{block}.weight");
            let w0 = weights.gguf.dequant_f32(&name)?;
            let csr = magnitude_prune_csr(&w0, m.ncols, m.nrows, sp.min(0.999));
            match block {
                "ffn_gate" => overrides[layer_idx].gate = Some(csr),
                "ffn_up" => overrides[layer_idx].up = Some(csr),
                _ => overrides[layer_idx].down = Some(csr),
            }
            d_arch_num += sp * params;
        }
    }
    let d_arch_global = if d_arch_den > 0.0 {
        d_arch_num / d_arch_den
    } else {
        0.0
    };

    // Dos Generators en lockstep: original vs candidato (KV independientes).
    let mut gen_orig =
        Generator::new(weights.clone(), tokenizer.clone(), 4, 128, sampler.clone(), 42);
    let mut gen_cand = Generator::new(weights, tokenizer, 4, 128, sampler, 42);

    let mut kl_sum = 0.0f32;
    for &tok in tokens.iter().take(n_pos) {
        let lo = gen_orig.forward(&mut orch, tok)?;
        let lc = gen_cand.forward_with_override(&mut orch, tok, &overrides)?;
        kl_sum += softmax_kl(&lo, &lc);
    }
    let kl_global = kl_sum / n_pos as f32;

    println!(
        "{{\"kl_global\":{:.6},\"d_arch_global\":{:.4},\"n_positions\":{},\"n_layers_sparse\":{}}}",
        kl_global,
        d_arch_global,
        n_pos,
        overrides.iter().filter(|o| o.gate.is_some()).count()
    );
    Ok(())
}
