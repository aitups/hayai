//! Evaluador KL **batcheado** de la población (Fase 2): procesa N candidatos
//! por token en un único `StreamingGenerator::forward_batched` (path Dense —
//! llama/ALIA), con overrides de FFN por (candidato, capa) construidos desde las
//! adyacencias de `saor-engine decode-pop` y los logits del profesor cacheados.
//!
//!   hayai kl_eval_batch --model <gguf> --prompts <txt> --adj-dir <dir>
//!                       [--n-positions N] [--device auto] [--teacher-cache <f>]
//!
//! NOTA: path Dense por ahora; los híbridos despachan por `ModelKind` y su
//! forward batcheado vive en `hybrid_infer` (extensión en curso).

use std::path::PathBuf;

use hayai_core::{EngineOrchestrator, ExecutionMode, FfnOverride, MemoryStrategy, StreamingGenerator};
use hayai_cpu::LayerKvCache;
use hayai_model::{CsrSparse, GgufCatalog, SamplerConfig, Tokenizer};

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

fn csr_from_adjacency(w0: &[f32], d_in: usize, d_out: usize, adj: &[u8]) -> CsrSparse {
    let mut row_ptr = vec![0i32; d_out + 1];
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    for j in 0..d_out {
        for i in 0..d_in {
            let conn = i * d_out + j;
            if adj[conn >> 3] & (1 << (conn & 7)) != 0 {
                col_idx.push(i as i32);
                vals.push(w0[j * d_in + i]);
            }
        }
        row_ptr[j + 1] = col_idx.len() as i32;
    }
    CsrSparse {
        row_ptr,
        col_idx,
        vals,
        d_in,
        d_out,
    }
}

fn save_teacher_cache(path: &std::path::Path, logits: &[Vec<f32>]) -> std::io::Result<()> {
    let n_pos = logits.len();
    let vocab = if n_pos > 0 { logits[0].len() } else { 0 };
    let mut buf = Vec::with_capacity(8 + n_pos * vocab * 4);
    buf.extend_from_slice(&(n_pos as u32).to_le_bytes());
    buf.extend_from_slice(&(vocab as u32).to_le_bytes());
    for v in logits {
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    std::fs::write(path, buf)
}

fn load_teacher_cache(path: &std::path::Path) -> std::io::Result<Vec<Vec<f32>>> {
    let raw = std::fs::read(path)?;
    if raw.len() < 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "teacher-cache: fichero demasiado corto",
        ));
    }
    let n_pos = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
    let vocab = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
    if raw.len() != 8 + n_pos * vocab * 4 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "teacher-cache: tamaño incoherente",
        ));
    }
    let mut out = Vec::with_capacity(n_pos);
    let mut off = 8usize;
    for _ in 0..n_pos {
        let mut v = Vec::with_capacity(vocab);
        for _ in 0..vocab {
            v.push(f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]));
            off += 4;
        }
        out.push(v);
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<PathBuf> = None;
    let mut prompts: Option<PathBuf> = None;
    let mut adj_dir: Option<PathBuf> = None;
    let mut teacher_cache: Option<PathBuf> = None;
    let mut device = "auto".to_string();
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
            "--adj-dir" => {
                i += 1;
                adj_dir = args.get(i).map(PathBuf::from);
            }
            "--teacher-cache" => {
                i += 1;
                teacher_cache = args.get(i).map(PathBuf::from);
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
                eprintln!("kl_eval_batch: argumento desconocido '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let model = model.ok_or("falta --model <gguf>")?;
    let prompts = prompts.ok_or("falta --prompts <txt>")?;
    let adj_dir = adj_dir.ok_or("falta --adj-dir <dir>")?;

    let texts: Vec<String> = std::fs::read_to_string(&prompts)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let corpus = texts.join("\n");

    let mut cat = GgufCatalog::open(&model)?;
    let tokenizer = Tokenizer::from_catalog(&cat)?;
    let mut gen = StreamingGenerator::open(&model, tokenizer, 4, 128, SamplerConfig::default(), 42)?;
    gen.set_memory_strategy(MemoryStrategy::Minimal);
    let mut orch = EngineOrchestrator::new(ExecutionMode::parse(&device), gen.config.clone());
    let mut scratch = gen.prepare_session(&mut orch)?;

    let tokens = gen.tokenizer.encode(&corpus, false);
    let n_pos = tokens.len().min(n_positions);

    let meta_raw = std::fs::read_to_string(adj_dir.join("meta.json"))?;
    let meta: serde_json::Value = serde_json::from_str(&meta_raw)?;
    let n_layers = meta["n_layers"].as_u64().unwrap_or(0) as usize;
    let n_cand = meta["n_candidates"].as_u64().unwrap_or(0) as usize;
    let blocks: Vec<String> = meta["blocks"]
        .as_array()
        .map(|a| a.iter().filter_map(|b| b.as_str().map(String::from)).collect())
        .unwrap_or_default();
    if n_layers == 0 || n_cand == 0 || blocks.is_empty() {
        return Err("kl_eval_batch: meta.json incompleto (decode-pop de saor)".into());
    }

    // Dims por bloque (gate/up: hidden→intermediate; down: al revés).
    let dims: Vec<(String, usize, usize)> = blocks
        .iter()
        .map(|b| match b.as_str() {
            "ffn_down" => (
                "ffn_down".into(),
                gen.config.intermediate_size,
                gen.config.hidden_size,
            ),
            _ => (
                b.clone(),
                gen.config.hidden_size,
                gen.config.intermediate_size,
            ),
        })
        .collect();
    let d_arch_den: f64 = dims.iter().map(|(_, a, b)| (a * b) as f64).sum::<f64>() * n_layers as f64;

    // Logits del profesor: cache en disco o una pasada batcheada (n=1, sin override).
    // KV persistente por candidato para teacher-forcing (el token t atiende a su historia).
    let kv_cfg = || {
        vec![(0..gen.config.num_layers)
            .map(|_| LayerKvCache::new(gen.attn_cfg.num_kv_heads, gen.attn_cfg.head_dim, 4, 128))
            .collect()]
    };
    let teacher: Vec<Vec<f32>> = if let Some(p) = &teacher_cache {
        if p.exists() {
            eprintln!("[kl_eval_batch] teacher-cache: {p:?} (cargado)");
            load_teacher_cache(p)?
        } else {
            eprintln!("[kl_eval_batch] computando logits del profesor…");
            let mut tkv = kv_cfg();
            let mut t = Vec::with_capacity(n_pos);
            for (pos, &tok) in tokens.iter().take(n_pos).enumerate() {
                let lg = gen.forward_batched_any(&mut orch, tok, pos, &mut tkv, 1, |_, _| FfnOverride::default(), &mut scratch)?;
                t.push(lg[0].clone());
            }
            save_teacher_cache(p, &t)?;
            t
        }
    } else {
        let mut tkv = kv_cfg();
        let mut t = Vec::with_capacity(n_pos);
        for (pos, &tok) in tokens.iter().take(n_pos).enumerate() {
            let lg = gen.forward_batched_any(&mut orch, tok, pos, &mut tkv, 1, |_, _| FfnOverride::default(), &mut scratch)?;
            t.push(lg[0].clone());
        }
        t
    };

    // Pre-paso: lee cada adyacencia UNA vez por (cand, capa) → d_arch_num + flag
    // de esparsidad. Las capas densas (>95%) usan el GEMM denso batcheado (sin CSR).
    let mut sparse: Vec<Vec<bool>> = vec![vec![false; gen.config.num_layers]; n_cand];
    let mut d_arch_num = vec![0.0f64; n_cand];
    for c in 0..n_cand {
        for l in 0..gen.config.num_layers {
            for (block, din, dout) in &dims {
                let adj_path = adj_dir.join(format!("c{c:03}.l{l:02}.{block}.bin"));
                let Ok(adj) = std::fs::read(&adj_path) else { continue };
                let total = din * dout;
                let active: usize = adj.iter().map(|b| b.count_ones() as usize).sum();
                d_arch_num[c] += (1.0 - active as f64 / total as f64) * total as f64;
                // Skip SOLO si la capa está exactamente densa (CSR == GEMM denso,
                // resultado idéntico). Cualquier poda (>0%) requiere el CSR.
                if active != total {
                    sparse[c][l] = true;
                }
            }
        }
    }
    let mut cand_kv: Vec<Vec<LayerKvCache>> = (0..n_cand)
        .map(|_| {
            (0..gen.config.num_layers)
                .map(|_| LayerKvCache::new(gen.attn_cfg.num_kv_heads, gen.attn_cfg.head_dim, 4, 128))
                .collect()
        })
        .collect();
    // Atajo exacto: si NINGÚN (cand, capa) tiene poda, todos los candidatos son
    // idénticos al profesor → KL = 0 por definición (sin forward).
    if sparse.iter().flatten().all(|&s| !s) {
        let parts: Vec<String> = (0..n_cand)
            .map(|c| {
                let da = d_arch_num[c] / d_arch_den;
                format!("{{\"kl_global\":0.000000,\"d_arch_global\":{da:.4},\"n_positions\":{n_pos}}}")
            })
            .collect();
        println!("[{}]", parts.join(","));
        return Ok(());
    }
    let mut kl_sum = vec![0.0f32; n_cand];
    for (pos, &tok) in tokens.iter().take(n_pos).enumerate() {
        let lg = gen.forward_batched_any(
            &mut orch,
            tok,
            pos,
            &mut cand_kv,
            n_cand,
            |c, l| {
                let mut ov = FfnOverride::default();
                if !sparse[c][l] {
                    return ov;
                }
                for (block, din, dout) in &dims {
                    let adj_path = adj_dir.join(format!("c{c:03}.l{l:02}.{block}.bin"));
                    let Ok(adj) = std::fs::read(&adj_path) else { continue };
                    let name = format!("blk.{l}.{block}.weight");
                    let Ok(w0) = cat.dequant_f32(&name) else { continue };
                    let csr = csr_from_adjacency(&w0, *din, *dout, &adj);
                    match block.as_str() {
                        "ffn_gate" => ov.gate = Some(csr),
                        "ffn_up" => ov.up = Some(csr),
                        _ => ov.down = Some(csr),
                    }
                }
                ov
            },
            &mut scratch,
        )?;
        for c in 0..n_cand {
            kl_sum[c] += softmax_kl(&teacher[pos], &lg[c]);
        }
        eprintln!("[kl_eval_batch] token {pos}/{n_pos} ok");
    }

    let parts: Vec<String> = (0..n_cand)
        .map(|c| {
            let kl = kl_sum[c] / n_pos as f32;
            let da = d_arch_num[c] / d_arch_den;
            format!("{{\"kl_global\":{kl:.6},\"d_arch_global\":{da:.4},\"n_positions\":{n_pos}}}")
        })
        .collect();
    println!("[{}]", parts.join(","));
    Ok(())
}

