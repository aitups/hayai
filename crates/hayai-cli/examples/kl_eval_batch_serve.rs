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

use hayai_core::{
    EngineOrchestrator, ExecutionMode, FfnOverride, MemoryStrategy, SparseAdj,
    StreamingGenerator,
};
use hayai_cpu::LayerKvCache;
use hayai_model::{CsrSparse, GgufCatalog, SamplerConfig, Tokenizer};

/// Build del CSR desde la adyacencia + pesos F32 del profesor (path per-token —
/// verificación de paridad contra el forward seq; en producción se usa SparseAdj).
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

/// Logits del profesor. Para el hybrid (qwen27) el per-token recrea el estado
/// DeltaNet en cada llamada → logits incorrectos; la referencia es el **seq**
/// (una llamada con toda la secuencia, estado persistente). El per-token solo
/// se usa en el modo `--per-token` (paridad Dense).
fn compute_teacher(
    gen: &mut StreamingGenerator,
    orch: &mut EngineOrchestrator,
    scratch: &mut hayai_opencl::StreamingScratch,
    tokens: &[u32],
    n_pos: usize,
    vocab: usize,
    per_token: bool,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut tkv: Vec<Vec<LayerKvCache>> = vec![(0..gen.config.num_layers)
        .map(|_| LayerKvCache::new(gen.attn_cfg.num_kv_heads, gen.attn_cfg.head_dim, 4, 128))
        .collect::<Vec<_>>()];
    if per_token {
        let mut t = Vec::with_capacity(n_pos);
        for (pos, &tok) in tokens.iter().take(n_pos).enumerate() {
            let lg = gen.forward_batched_any(orch, tok, pos, &mut tkv, 1, |_, _| FfnOverride::default(), scratch)?;
            t.push(lg[0].clone());
        }
        Ok(t)
    } else {
        let lg = gen.forward_batched_any_seq(orch, &tokens[..n_pos], &mut tkv, 1, |_, _, _, _, _| FfnOverride::default())?;
        Ok(lg[0]
            .chunks_exact(vocab)
            .take(n_pos)
            .map(|c| c.to_vec())
            .collect())
    }
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
    let mut per_token = false;
    let mut serve: Option<PathBuf> = None;
    let mut memory = "auto".to_string();
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
            "--per-token" => {
                per_token = true;
            }
            "--serve" => {
                i += 1;
                serve = args.get(i).map(PathBuf::from);
            }
            "--memory-strategy" => {
                i += 1;
                if let Some(m) = args.get(i) {
                    memory = m.clone();
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
    // En modo `--serve <workdir>` el adj-dir de cada generación es
    // `<workdir>/adj` (decode-pop lo reescribe por gen) y n_cand se relee del
    // meta.json de cada iteración; el modelo se carga UNA vez.
    let serve_workdir: Option<PathBuf> = serve.clone();
    let adj_dir = match serve_workdir.clone() {
        Some(w) => w.join("adj"),
        None => adj_dir.ok_or("falta --adj-dir <dir> o --serve <workdir>")?,
    };

    let texts: Vec<String> = std::fs::read_to_string(&prompts)?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let corpus = texts.join("\n");

    let mut cat = GgufCatalog::open(&model)?;
    let tokenizer = Tokenizer::from_catalog(&cat)?;
    let mut gen = StreamingGenerator::open(&model, tokenizer, 4, 128, SamplerConfig::default(), 42)?;
    let strat = match memory.as_str() {
        "minimal" => MemoryStrategy::Minimal,
        m if m.chars().all(|c| c.is_ascii_digit()) && !m.is_empty() => {
            MemoryStrategy::CapBytes(m.parse::<u64>().unwrap_or(0) * 1024 * 1024)
        }
        _ => MemoryStrategy::AutoFit,
    };
    gen.set_memory_strategy(strat);
    let mut orch = EngineOrchestrator::new(ExecutionMode::parse(&device), gen.config.clone());
    let mut scratch = gen.prepare_session(&mut orch)?;

    let tokens = gen.tokenizer.encode(&corpus, false);
    let n_pos = tokens.len().min(n_positions);
    let vocab = gen.config.vocab_size;

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
            .collect::<Vec<_>>()]
    };
    let teacher: Vec<Vec<f32>> = if let Some(p) = &teacher_cache {
        if p.exists() {
            eprintln!("[kl_eval_batch] teacher-cache: {p:?} (cargado)");
            load_teacher_cache(p)?
        } else {
            eprintln!("[kl_eval_batch] computando logits del profesor…");
            let t = compute_teacher(&mut gen, &mut orch, &mut scratch, &tokens, n_pos, vocab, per_token)?;
            save_teacher_cache(p, &t)?;
            t
        }
    } else {
        compute_teacher(&mut gen, &mut orch, &mut scratch, &tokens, n_pos, vocab, per_token)?
    };

    // ── Modo servidor (--serve <workdir>): el modelo queda cargado y cada
    // iteración evalúa UNA generación (decode-pop escribe <workdir>/adj + el
    // marcador ready_<gen>). La paridad KL es idéntica al modo de un solo uso.
    let mut gen_idx = 0usize;
    loop {
        if let Some(w) = &serve_workdir {
            let stop = w.join("stop");
            let ready = w.join(format!("ready_{gen_idx}"));
            while !stop.exists() && !ready.exists() {
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
            if stop.exists() {
                return Ok(());
            }
        }
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
        let out = format!("[{}]", parts.join(","));
        match &serve_workdir {
            Some(w) => {
                std::fs::write(w.join(format!("result_{gen_idx}.json")), &out)?;
                let _ = std::fs::remove_file(w.join(format!("ready_{gen_idx}")));
                gen_idx += 1;
                continue;
            }
            None => {
                println!("{out}");
                break;
            }
        }
    }
    let mut kl_sum = vec![0.0f32; n_cand];
    let all_logits = if per_token {
        // Verificación de paridad: forward por token (path original Fase 2) con CSR.
        let mut out: Vec<Vec<f32>> = (0..n_cand).map(|_| Vec::new()).collect();
        for pos in 0..n_pos {
            let logits = gen.forward_batched_any(
                &mut orch,
                tokens[pos],
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
                out[c].extend_from_slice(&logits[c]);
            }
        }
        out
    } else {
    // Forward por capas: UNA llamada con toda la secuencia. El override se construye
    // UNA vez por (candidato, capa); las proyecciones + FFN se batchean sobre N×n_pos.
    // El SparseAdj (bit-tensor + Arc<F32> compartido por capa) evita el gather del
    // CSR por (candidato, capa, token) — Fase 2, C4.
    gen.forward_batched_any_seq(
        &mut orch,
        &tokens[..n_pos],
        &mut cand_kv,
        n_cand,
        |c, l, gate_w, up_w, down_w| {
            let mut ov = FfnOverride::default();
            if !sparse[c][l] {
                return ov;
            }
            for (block, din, dout) in &dims {
                let adj_path = adj_dir.join(format!("c{c:03}.l{l:02}.{block}.bin"));
                let Ok(adj) = std::fs::read(&adj_path) else { continue };
                let w = match block.as_str() {
                    "ffn_gate" => gate_w,
                    "ffn_up" => up_w,
                    _ => down_w,
                };
                // F32 presente en el path CPU/dequant; vacío en el path GPU-Q4
                // (el kernel dequantiza en GPU y `apply_sparse_adj_block` usa w_q4).
                let weights = w.clone().unwrap_or_default();
                let sa = SparseAdj {
                    adjacency: adj,
                    weights,
                    d_in: *din,
                    d_out: *dout,
                };
                match block.as_str() {
                    "ffn_gate" => ov.gate_adj = Some(sa),
                    "ffn_up" => ov.up_adj = Some(sa),
                    _ => ov.down_adj = Some(sa),
                }
            }
            ov
        },
    )?
    };
    for pos in 0..n_pos {
        for c in 0..n_cand {
            kl_sum[c] += softmax_kl(
                &teacher[pos],
                &all_logits[c][pos * vocab..(pos + 1) * vocab],
            );
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
    let out = format!("[{}]", parts.join(","));
    match &serve_workdir {
        Some(w) => {
            std::fs::write(w.join(format!("result_{gen_idx}.json")), &out)?;
            let _ = std::fs::remove_file(w.join(format!("ready_{gen_idx}")));
            gen_idx += 1;
            continue;
        }
        None => {
            println!("{out}");
            break;
        }
    }
    }  // cierre del bucle de generaciones (modo servidor)
    Ok(())
}

